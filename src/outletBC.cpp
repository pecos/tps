// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-
#include "outletBC.hpp"

#include "dgNonlinearForm.hpp"
#include "riemann_solver.hpp"

// TODO(kevin): non-reflecting BC for plasma.
OutletBC::OutletBC(MPI_Groups *_groupsMPI, Equations _eqSystem, RiemannSolver *_rsolver, GasMixture *_mixture,
                   GasMixture *d_mixture, ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt,
                   const int _dim, const int _num_equation, int _patchNumber, double _refLength, OutletType _bcType,
                   const Array<double> &_inputData, const int &_maxIntPoints, const int &_maxDofs, bool axisym)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber,
                        _refLength, axisym),
      groupsMPI(_groupsMPI),
      d_mixture_(d_mixture),
      outletType_(_bcType),
      inputState(_inputData),
      maxIntPoints_(_maxIntPoints),
      maxDofs_(_maxDofs) {
  if ((mixture->GetWorkingFluid() != DRY_AIR) && (outletType_ != SUB_P)) {
    grvy_printf(GRVY_ERROR, "Plasma only supports subsonic reflecting pressure outlet!\n");
    exit(-1);
  }
  groupsMPI->setPatch(_patchNumber);

  meanUp.UseDevice(true);
  meanUp.SetSize(num_equation_);
  meanUp = 0.;

  localMeanUp.UseDevice(true);
  localMeanUp.SetSize(num_equation_ + 1);
  localMeanUp = 0.;

  glob_sum.UseDevice(true);
  glob_sum.SetSize(num_equation_ + 1);
  glob_sum = 0.;

  auto hmeanUp = meanUp.HostWrite();

  hmeanUp[0] = 1.2;
  hmeanUp[1] = 60;
  hmeanUp[2] = 0;
  if (nvel_ == 3) hmeanUp[3] = 0.;
  hmeanUp[1 + nvel_] = 300.0;  // 101300;
  if (eqSystem == NS_PASSIVE) hmeanUp[num_equation_ - 1] = 0.;

  area_ = 0.;
  parallelAreaComputed = false;

  Array<double> coords;

  // assuming planar outlets (for GPU only)
  Vector normal;
  normal.UseDevice(false);
  normal.SetSize(dim_);
  bool normalFull = false;

  // init boundary U
  bdrN = 0;
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == patchNumber) {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
      Array<int> dofs;

      vfes->GetElementVDofs(Tr->Elem1No, dofs);

      int intorder = Tr->Elem1->OrderW() + 2 * vfes->GetFE(Tr->Elem1No)->GetOrder();
      if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);
      for (int i = 0; i < ir.GetNPoints(); i++) {
        IntegrationPoint ip = ir.IntPoint(i);
        Tr->SetAllIntPoints(&ip);
        if (!normalFull) CalcOrtho(Tr->Jacobian(), normal);
        double x[3];
        Vector transip;
        transip.UseDevice(false);
        transip.SetDataAndSize(&x[0], 3);
        Tr->Transform(ip, transip);
        for (int d = 0; d < 3; d++) coords.Append(transip[d]);
        bdrN++;
      }
    }
  }

  boundaryU.UseDevice(true);
  boundaryU.SetSize(bdrN * num_equation_);
  boundaryU = 0.;

  Vector iState, iUp;
  iState.UseDevice(false);
  iUp.UseDevice(false);
  iState.SetSize(num_equation_);
  iUp.SetSize(num_equation_);
  auto hboundaryU = boundaryU.HostWrite();
  for (int i = 0; i < bdrN; i++) {
    for (int eq = 0; eq < num_equation_; eq++) iUp(eq) = hmeanUp[eq];
    mixture->GetConservativesFromPrimitives(iUp, iState);

    //     double gamma = mixture->GetSpecificHeatRatio();
    //     double k = 0.;
    //     for (int d = 0; d < dim; d++) k += iState[1 + d] * iState[1 + d];
    //     double rE = iState[1 + dim] / (gamma - 1.) + 0.5 * iState[0] * k;
    //
    //     for (int d = 0; d < dim; d++) iState[1 + d] *= iState[0];
    //     iState[1 + dim] = rE;
    //     if (eqSystem == NS_PASSIVE) iState(num_equation - 1) *= iState(0);

    for (int eq = 0; eq < num_equation_; eq++) hboundaryU[eq + i * num_equation_] = iState[eq];
  }
  bdrUInit = false;
  bdrN = 0;

  // init. unit tangent vector tangent1
  tangent1.UseDevice(true);
  tangent2.UseDevice(true);
  tangent1.SetSize(dim_);
  tangent2.SetSize(dim_);
  tangent1 = 0.;
  tangent2 = 0.;
  auto htan1 = tangent1.HostWrite();
  auto htan2 = tangent2.HostWrite();
  Array<double> coords1, coords2;
  for (int d = 0; d < dim_; d++) {
    coords1.Append(coords[d]);
    coords2.Append(coords[d + 3]);
  }

  double modVec = 0.;
  for (int d = 0; d < dim_; d++) {
    modVec += (coords2[d] - coords1[d]) * (coords2[d] - coords1[d]);
    htan1[d] = coords2[d] - coords1[d];
  }
  for (int d = 0; d < dim_; d++) htan1[d] *= 1. / sqrt(modVec);

  Vector unitNorm;
  unitNorm.UseDevice(false);
  unitNorm.SetSize(dim_);
  unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim_; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }
  if (dim_ == 3) {
    htan2[0] = unitNorm[1] * htan1[2] - unitNorm[2] * htan1[1];
    htan2[1] = unitNorm[2] * htan1[0] - unitNorm[0] * htan1[2];
    htan2[2] = unitNorm[0] * htan1[1] - unitNorm[1] * htan1[0];
  }

  DenseMatrix M(dim_, dim_);
  for (int d = 0; d < dim_; d++) {
    M(0, d) = unitNorm[d];
    M(1, d) = htan1[d];
    if (dim_ == 3) M(2, d) = htan2[d];
  }

  DenseMatrix invM(dim_, dim_);
  mfem::CalcInverse(M, invM);

  inverseNorm2cartesian.UseDevice(true);
  inverseNorm2cartesian.SetSize(dim_ * dim_);
  auto hinv = inverseNorm2cartesian.HostWrite();
  for (int i = 0; i < dim_; i++)
    for (int j = 0; j < dim_; j++) hinv[i + j * dim_] = invM(i, j);

  initBdrElemsShape();

  meanUp.Read();
  boundaryU.Read();
  tangent1.Read();
  tangent2.Read();
}

void OutletBC::initBdrElemsShape() {
  bdrShape.UseDevice(true);
  int Nbdr = boundaryU.Size() / num_equation_;
  bdrShape.SetSize(maxIntPoints_ * maxDofs_ * Nbdr);
  bdrShape = 0.;
  auto hbdrShape = bdrShape.HostWrite();
  int elCount = 0;
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == patchNumber) {
      elCount++;
    }
  }

  bdrElemsQ.SetSize(2 * elCount);
  bdrDofs.SetSize(maxDofs_ * elCount);
  bdrElemsQ = 0;
  bdrDofs = 0;
  auto hbdrElemsQ = bdrElemsQ.HostWrite();
  auto hbdrDofs = bdrDofs.HostWrite();

  offsetsBoundaryU.SetSize(elCount);
  offsetsBoundaryU = -1;
  auto hoffsetsBoundaryU = offsetsBoundaryU.HostWrite();
  int offsetCount = 0;

  elCount = 0;

  Vector shape;
  shape.UseDevice(false);

  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == patchNumber) {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
      int elDofs = vfes->GetFE(Tr->Elem1No)->GetDof();
      Array<int> dofs;
      vfes->GetElementVDofs(Tr->Elem1No, dofs);

      for (int i = 0; i < elDofs; i++) hbdrDofs[i + elCount * maxDofs_] = dofs[i];

      int intorder = Tr->Elem1->OrderW() + 2 * vfes->GetFE(Tr->Elem1No)->GetOrder();
      if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);

      hbdrElemsQ[2 * elCount] = elDofs;
      hbdrElemsQ[2 * elCount + 1] = ir.GetNPoints();
      hoffsetsBoundaryU[elCount] = offsetCount;

      for (int i = 0; i < ir.GetNPoints(); i++) {
        IntegrationPoint ip = ir.IntPoint(i);
        Tr->SetAllIntPoints(&ip);
        shape.SetSize(elDofs);
        vfes->GetFE(Tr->Elem1No)->CalcShape(Tr->GetElement1IntPoint(), shape);

        for (int n = 0; n < elDofs; n++) hbdrShape[n + i * maxDofs_ + elCount * maxIntPoints_ * maxDofs_] = shape(n);
        offsetCount++;
      }
      elCount++;
    }
  }
#ifdef _GPU_
  bdrElemsQ.ReadWrite();
  bdrDofs.ReadWrite();
  bdrShape.ReadWrite();

  bdrUp.UseDevice(true);
  int size_bdrUp = bdrElemsQ.Size() / 2;
  bdrUp.SetSize(size_bdrUp * num_equation_ * maxIntPoints_);
  bdrUp = 0.;
  bdrUp.Read();

  offsetsBoundaryU.ReadWrite();
#endif
}

OutletBC::~OutletBC() {}

void OutletBC::initBCs() {
  if (!BCinit) {
    computeParallelArea();

#ifdef _GPU_
    face_flux_.UseDevice(true);
    face_flux_.SetSize(num_equation_ * maxIntPoints_ * listElems.Size());
    face_flux_ = 0.;
#endif

    BCinit = true;
  }
}

void OutletBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip, double delta,
                              double distance, Vector &bdrFlux) {
  switch (outletType_) {
    case SUB_P:
      subsonicReflectingPressure(normal, stateIn, bdrFlux);
      break;
    case SUB_P_NR:
      subsonicNonReflectingPressure(normal, stateIn, gradState, bdrFlux);
      break;
    case SUB_MF_NR:
      subsonicNonRefMassFlow(normal, stateIn, gradState, bdrFlux);
      break;
    case SUB_MF_NR_PW:
      subsonicNonRefPWMassFlow(normal, stateIn, gradState, bdrFlux);
      break;
    case RESIST_IN:
      std::cout << " OUTLET BC RESIST_IN NOT FULLY IMPLEMENTED IN computeBdrFlux" << endl;
      exit(1);
      break;
  }
}

void OutletBC::computeParallelArea() {
  if (parallelAreaComputed) return;

  MPI_Comm bcomm = groupsMPI->getComm(patchNumber);
  area_ = aggregateArea(patchNumber, bcomm);
  int nfaces = aggregateBndryFaces(patchNumber, bcomm);
  parallelAreaComputed = true;

  if (groupsMPI->isGroupRoot(bcomm)) {
    grvy_printf(ginfo, "\n[OUTLET]: Patch number                      = %i\n", patchNumber);
    grvy_printf(ginfo, "[OUTLET]: Total Surface Area                = %.5f\n", area_);
    grvy_printf(ginfo, "[OUTLET]: # of boundary faces               = %i\n", nfaces);
    grvy_printf(ginfo, "[OUTLET]: # of participating MPI partitions = %i\n", groupsMPI->groupSize(bcomm));
  }
}

void OutletBC::updateMean_gpu(ParGridFunction *Up, Vector &localMeanUp, const int num_equation, const int numBdrElems,
                              const int totDofs, Vector &bdrUp, Array<int> &bdrElemsQ, Array<int> &bdrDofs,
                              Vector &bdrShape, const int &maxIntPoints, const int &maxDofs) {
#ifdef _GPU_
  const double *d_Up = Up->Read();
  double *d_localMeanUp = localMeanUp.Write();
  double *d_bdrUp = bdrUp.Write();
  auto d_bdrElemQ = bdrElemsQ.Read();
  auto d_bdrDofs = bdrDofs.Read();
  const double *d_bdrShape = bdrShape.Read();

  int groupAveraging = numBdrElems;
  if (groupAveraging % 2 != 0) groupAveraging++;
  groupAveraging /= 2;

  MFEM_FORALL(el, numBdrElems, {
    const int elDof = d_bdrElemQ[2 * el];
    const int Q = d_bdrElemQ[2 * el + 1];
    double elUp[gpudata::MAXEQUATIONS * gpudata::MAXDOFS];  // double elUp[20 * 216];
    double shape[gpudata::MAXDOFS];                         // double shape[216];
    double sum;

    // retreive data
    for (int i = 0; i < elDof; i++) {
      int index = d_bdrDofs[i + el * maxDofs];
      for (int eq = 0; eq < num_equation; eq++) elUp[i + eq * elDof] = d_Up[index + eq * totDofs];
    }

    // compute Up at the integration points
    for (int k = 0; k < Q; k++) {
      for (int i = 0; i < elDof; i++) shape[i] = d_bdrShape[i + k * maxDofs + el * maxIntPoints * maxDofs];
      for (int eq = 0; eq < num_equation; eq++) {
        sum = 0.;
        for (int i = 0; i < elDof; i++) sum += shape[i] * elUp[i + eq * elDof];
        d_bdrUp[eq + k * num_equation + el * num_equation * maxIntPoints] = sum;
      }
    }

    // element sum (for averaging)
    for (int eq = 0; eq < num_equation; eq++) {
      sum = 0.;
      for (int k = 0; k < Q; k++) {
        sum += d_bdrUp[eq + k * num_equation + el * num_equation * maxIntPoints];
      }
      d_bdrUp[eq + el * num_equation * maxIntPoints] = sum;
    }

    // parallel sum
    // // overall sum (for averaging)
    // int interval = 1;
    // // for(int interval=2;interval<=groupAveraging;interval*=2)
    // while (interval < numBdrElems) {
    //   interval *= 2;
    //   if (el % interval == 0) {
    //     int el2 = el + interval / 2;
    //     for (int eq = 0; eq < num_equation; eq++) {
    //       if (el2 < numBdrElems)
    //         d_bdrUp[eq + el * num_equation * maxIntPoints] += d_bdrUp[eq + el2 * num_equation * maxIntPoints];
    //     }
    //   }
    //   MFEM_SYNC_THREAD;
    // }
  });

  MFEM_FORALL(eq, num_equation, {
    // serial sum (for each eq)
    for (int el = 1; el < numBdrElems; el++) {
      d_bdrUp[eq] += d_bdrUp[eq + el * num_equation * maxIntPoints];
    }
    d_localMeanUp[eq] = d_bdrUp[eq];
  });

#endif
}

void OutletBC::initBoundaryU(ParGridFunction *Up) {
  Vector elUp;
  elUp.UseDevice(false);
  Vector shape;
  shape.UseDevice(false);
  auto hboundaryU = boundaryU.HostWrite();
  int Nbdr = 0;

  // double *data = Up->GetData();
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == patchNumber) {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);

      int elDofs = vfes->GetFE(Tr->Elem1No)->GetDof();

      Array<int> dofs;
      vfes->GetElementVDofs(Tr->Elem1No, dofs);

      // retreive data
      elUp.SetSize(elDofs * num_equation_);
      Up->GetSubVector(dofs, elUp);

      int intorder = Tr->Elem1->OrderW() + 2 * vfes->GetFE(Tr->Elem1No)->GetOrder();
      if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);
      for (int i = 0; i < ir.GetNPoints(); i++) {
        IntegrationPoint ip = ir.IntPoint(i);
        Tr->SetAllIntPoints(&ip);
        shape.SetSize(elDofs);
        vfes->GetFE(Tr->Elem1No)->CalcShape(Tr->GetElement1IntPoint(), shape);
        Vector iUp(num_equation_);

        for (int eq = 0; eq < num_equation_; eq++) {
          double sum = 0.;
          for (int d = 0; d < elDofs; d++) {
            sum += shape[d] * elUp(d + eq * elDofs);
          }
          hboundaryU[eq + Nbdr * num_equation_] = sum;
        }
        Nbdr++;
      }
    }
  }

  boundaryU.ReadWrite();
}

void OutletBC::updateMean(IntegrationRules *intRules, ParGridFunction *Up) {
  if (outletType_ == SUB_P) return;

  bdrN = 0;

#ifdef _GPU_
  DGNonLinearForm::setToZero_gpu(bdrUp, bdrUp.Size());

  const int dofs = vfes->GetNDofs();

  updateMean_gpu(Up, localMeanUp, num_equation_, bdrElemsQ.Size() / 2, dofs, bdrUp, bdrElemsQ, bdrDofs, bdrShape,
                 maxIntPoints_, maxDofs_);

  if (!bdrUInit) initBoundaryU(Up);

#else
  int Nbdr = 0;

  Vector elUp;
  Vector shape;
  localMeanUp = 0.;

  // double *data = Up->GetData();
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == patchNumber) {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);

      int elDofs = vfes->GetFE(Tr->Elem1No)->GetDof();

      Array<int> dofs;
      vfes->GetElementVDofs(Tr->Elem1No, dofs);

      // retreive data
      elUp.SetSize(elDofs * num_equation_);
      Up->GetSubVector(dofs, elUp);

      int intorder = Tr->Elem1->OrderW() + 2 * vfes->GetFE(Tr->Elem1No)->GetOrder();
      if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);
      for (int i = 0; i < ir.GetNPoints(); i++) {
        IntegrationPoint ip = ir.IntPoint(i);
        Tr->SetAllIntPoints(&ip);
        shape.SetSize(elDofs);
        vfes->GetFE(Tr->Elem1No)->CalcShape(Tr->GetElement1IntPoint(), shape);
        Vector iUp(num_equation_);

        for (int eq = 0; eq < num_equation_; eq++) {
          double sum = 0.;
          for (int d = 0; d < elDofs; d++) {
            sum += shape[d] * elUp(d + eq * elDofs);
          }
          localMeanUp[eq] += sum;
          if (!bdrUInit) boundaryU[eq + Nbdr * num_equation_] = sum;
        }
        Nbdr++;
      }
    }
  }
#endif

  double *h_localMeanUp = localMeanUp.HostReadWrite();
  int totNbdr = boundaryU.Size() / num_equation_;
  h_localMeanUp[num_equation_] = static_cast<double>(totNbdr);

  double *h_sum = glob_sum.HostWrite();
  MPI_Allreduce(h_localMeanUp, h_sum, num_equation_ + 1, MPI_DOUBLE, MPI_SUM, groupsMPI->getComm(patchNumber));

  BoundaryCondition::copyValues(glob_sum, meanUp, 1. / h_sum[num_equation_]);

  if (!bdrUInit) {
    auto hboundaryU = boundaryU.HostReadWrite();
    Vector iState(num_equation_), iUp(num_equation_);
    for (int i = 0; i < totNbdr; i++) {
      for (int eq = 0; eq < num_equation_; eq++) iUp[eq] = hboundaryU[eq + i * num_equation_];
      mixture->GetConservativesFromPrimitives(iUp, iState);
      //       double gamma = mixture->GetSpecificHeatRatio();
      //       double k = 0.;
      //       for (int d = 0; d < dim; d++) k += iState[1 + d] * iState[1 + d];
      //       double rE = iState[1 + dim] / (gamma - 1.) + 0.5 * iState[0] * k;
      //
      //       for (int d = 0; d < dim; d++) iState[1 + d] *= iState[0];
      //       iState[1 + dim] = rE;
      //       if (eqSystem == NS_PASSIVE) iState[num_equation - 1] *= iState[0];

      for (int eq = 0; eq < num_equation_; eq++) hboundaryU[eq + i * num_equation_] = iState[eq];
    }
    bdrUInit = true;
  }
}

void OutletBC::integrationBC(Vector &y, const Vector &x, const elementIndexingData &elem_index_data,
                             ParGridFunction *Up, ParGridFunction *gradUp,
                             const boundaryFaceIntegrationData &boundary_face_data, const int &maxIntPoints,
                             const int &maxDofs) {
  interpOutlet_gpu(x, elem_index_data, Up, gradUp, boundary_face_data, listElems, offsetsBoundaryU);

  integrateOutlets_gpu(y,  // output
                       x, elem_index_data, boundary_face_data, listElems, offsetsBoundaryU);
}

void OutletBC::subsonicNonReflectingPressure(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = mixture->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim_; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim_);
  meanVel = 0.;
  for (int d = 0; d < dim_; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim_ == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim_; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  double meanP = mixture->ComputePressureFromPrimitives(meanUp);

  // normal gradients
  Vector normGrad(num_equation_);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation_; eq++) {
    // NOTE(kevin): for axisymmetric case, azimuthal normal component will be always zero.
    for (int d = 0; d < dim_; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }
  // gradient of pressure in normal direction
  double dpdn = mixture->ComputePressureDerivative(normGrad, stateIn, false);

  const double speedSound = mixture->ComputeSpeedOfSound(meanUp);

  double meanK = 0.;
  for (int d = 0; d < dim_; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  double L2 = speedSound * speedSound * normGrad[0] - dpdn;
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim_; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim_ == 3) {
    for (int d = 0; d < dim_; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim_; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 = dpdn + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  double L6 = 0.;
  if (eqSystem == NS_PASSIVE) L6 = meanVel[0] * normGrad[num_equation_ - 1];

  // const double p = eqState->ComputePressure(stateIn, dim_);

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  //   double L1 = sigma * (meanUp[1 + dim_] - inputState[0]);
  double L1 = sigma * (meanP - inputState[0]);

  // calc vector d
  const double d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
  //   const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
  const double d3 = L3;
  const double d4 = L4;
  const double d5 = 0.5 * (L5 + L1);
  double d6 = 0.;
  if (eqSystem == NS_PASSIVE) d6 = L6;

  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
  bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
  if (dim_ == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[1 + dim_] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[1 + dim_] += meanUp[0] * meanVel[1] * d3;
  if (dim_ == 3) bdrFlux[1 + dim_] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[1 + dim_] += meanK * d1 + d5 / (gamma - 1.);

  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = d1 * meanUp[num_equation_ - 1] + meanUp[0] * d6;

  // flux gradients in other directions
  //   Vector fluxY(num_equation);
  //   {
  //     // gradients in tangent1 direction
  //     Vector tg1Grads(num_equation);
  //     tg1Grads = 0.;
  //     for(int eq=0;eq<num_equation;eq++)
  //     {
  //       for(int d=0;d<dim;d++) tg1Grads[eq] += tangent1[d]*gradState(eq,d);
  //     }
  //
  //
  //     const double dy_rv = tg1Grads(0)*meanVt1 + rho*tg1Grads(2);
  //     const double dy_ruv = meanVn*dy_rv + rho*meanVt1*tg1Grads(1);
  //     const double dy_rv2 = meanVt1*dy_rv + rho*meanVt1*tg1Grads(2);
  //     const double dy_p = tg1Grads(3);
  //     const double dy_E = dy_p/(gamma-1.) + k*tg1Grads(0) +
  //                         rho*(meanVn*tg1Grads(1) + meanVt1*tg1Grads(2) );
  //     const double dy_vp = meanVt1*dy_p + p*tg1Grads(2);
  //
  //     fluxY[0] = dy_rv;
  //     fluxY[1] = dy_ruv;
  //     fluxY[2] = dy_rv2 + dy_p;
  //     fluxY[3] = stateIn(3)/rho*dy_rv + rho*meanVt1*dy_E + dy_vp;
  //   }

  Vector state2(num_equation_);
  for (int eq = 0; eq < num_equation_; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation_];

  Vector stateN = state2;
  for (int d = 0; d < dim_; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim_; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim_ == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation_);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation_; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim_, dim_);
    for (int d = 0; d < dim_; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim_ == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim_, dim_);
    mfem::CalcInverse(M, invM);
    Vector momN(dim_), momX(dim_);
    for (int d = 0; d < dim_; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim_; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation_; eq++) boundaryU[eq + bdrN * num_equation_] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

// This is more or less right formulation even for two-temperature case.
void OutletBC::subsonicReflectingPressure(Vector &normal, Vector &stateIn, Vector &bdrFlux) {
  Vector state2(num_equation_);

  mixture->modifyEnergyForPressure(stateIn, state2, inputState[0]);

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::subsonicNonRefMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = mixture->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim_; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim_);
  meanVel = 0.;
  for (int d = 0; d < dim_; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim_ == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim_; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  // normal gradients
  Vector normGrad(num_equation_);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation_; eq++) {
    // NOTE(kevin): for axisymmetric case, azimuthal normal component will be always zero.
    for (int d = 0; d < dim_; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }
  // gradient of pressure in normal direction
  double dpdn = mixture->ComputePressureDerivative(normGrad, stateIn, false);

  const double speedSound = mixture->ComputeSpeedOfSound(meanUp);

  double meanK = 0.;
  for (int d = 0; d < dim_; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  //   double L2 = speedSound * speedSound * normGrad[0] - normGrad[1 + dim_];
  double L2 = speedSound * speedSound * normGrad[0] - dpdn;
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim_; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim_ == 3) {
    for (int d = 0; d < dim_; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim_; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[1 + dim_] + meanUp[0] * speedSound * L5;
  L5 = dpdn + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  double L6 = 0.;
  if (eqSystem == NS_PASSIVE) L6 = meanVel[0] * normGrad[num_equation_ - 1];

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L1 = -sigma * (meanVel[0] - inputState[0] / meanUp[0] / area_);
  // cout<<inputState[0]<<" "<<meanUp[0]<<" "<<area<<endl;
  L1 *= meanUp[0] * speedSound;

  // calc vector d
  const double d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
  //   const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
  const double d3 = L3;
  const double d4 = L4;
  const double d5 = 0.5 * (L5 + L1);
  double d6 = 0.;
  if (eqSystem == NS_PASSIVE) d6 = L6;

  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
  bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
  if (dim_ == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[1 + dim_] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[1 + dim_] += meanUp[0] * meanVel[1] * d3;
  if (dim_ == 3) bdrFlux[num_equation_ - 1] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[1 + dim_] += meanK * d1 + d5 / (gamma - 1.);

  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = d1 * meanUp[num_equation_ - 1] + meanUp[0] * d6;

  // flux gradients in other directions
  //   Vector fluxY(num_equation);
  //   {
  //     // gradients in tangent1 direction
  //     Vector tg1Grads(num_equation);
  //     tg1Grads = 0.;
  //     for(int eq=0;eq<num_equation;eq++)
  //     {
  //       for(int d=0;d<dim;d++) tg1Grads[eq] += tangent1[d]*gradState(eq,d);
  //     }
  //
  //
  //     const double dy_rv = tg1Grads(0)*meanVt1 + rho*tg1Grads(2);
  //     const double dy_ruv = meanVn*dy_rv + rho*meanVt1*tg1Grads(1);
  //     const double dy_rv2 = meanVt1*dy_rv + rho*meanVt1*tg1Grads(2);
  //     const double dy_p = tg1Grads(3);
  //     const double dy_E = dy_p/(gamma-1.) + k*tg1Grads(0) +
  //                         rho*(meanVn*tg1Grads(1) + meanVt1*tg1Grads(2) );
  //     const double dy_vp = meanVt1*dy_p + p*tg1Grads(2);
  //
  //     fluxY[0] = dy_rv;
  //     fluxY[1] = dy_ruv;
  //     fluxY[2] = dy_rv2 + dy_p;
  //     fluxY[3] = stateIn(3)/rho*dy_rv + rho*meanVt1*dy_E + dy_vp;
  //   }

  Vector state2(num_equation_);
  for (int eq = 0; eq < num_equation_; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation_];

  Vector stateN = state2;
  for (int d = 0; d < dim_; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim_; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim_ == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation_);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation_; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim_, dim_);
    for (int d = 0; d < dim_; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim_ == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim_, dim_);
    mfem::CalcInverse(M, invM);
    Vector momN(dim_), momX(dim_);
    for (int d = 0; d < dim_; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim_; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation_; eq++) boundaryU[eq + bdrN * num_equation_] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::subsonicNonRefPWMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = mixture->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim_; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim_);
  meanVel = 0.;
  for (int d = 0; d < dim_; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim_ == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim_; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  // normal gradients
  Vector normGrad(num_equation_);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation_; eq++) {
    // NOTE(kevin): for axisymmetric case, azimuthal normal component will be always zero.
    for (int d = 0; d < dim_; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }
  // gradient of pressure in normal direction
  double dpdn = mixture->ComputePressureDerivative(normGrad, stateIn, false);

  const double speedSound = mixture->ComputeSpeedOfSound(meanUp);

  double normVel = 0.;
  for (int d = 0; d < dim_; d++) normVel += stateIn[1 + d] * unitNorm[d];
  normVel /= stateIn[0];

  double meanK = 0.;
  for (int d = 0; d < dim_; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  //   double L2 = speedSound * speedSound * normGrad[0] - normGrad[1 + dim];
  double L2 = speedSound * speedSound * normGrad[0] - dpdn;
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim_; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim_ == 3) {
    for (int d = 0; d < dim_; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim_; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 = dpdn + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  double L6 = 0.;
  if (eqSystem == NS_PASSIVE) L6 = meanVel[0] * normGrad[num_equation_ - 1];

  // const double p = eqState->ComputePressure(stateIn, dim_);

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L1 = -sigma * (normVel - inputState[0] / meanUp[0] / area_);
  // cout<<inputState[0]<<" "<<meanUp[0]<<" "<<area<<endl;
  L1 *= meanUp[0] * speedSound;

  // calc vector d
  const double d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
  //   const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
  const double d3 = L3;
  const double d4 = L4;
  const double d5 = 0.5 * (L5 + L1);
  double d6 = 0.;
  if (eqSystem == NS_PASSIVE) d6 = L6;

  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
  bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
  if (dim_ == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[1 + dim_] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[1 + dim_] += meanUp[0] * meanVel[1] * d3;
  if (dim_ == 3) bdrFlux[num_equation_ - 1] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[1 + dim_] += meanK * d1 + d5 / (gamma - 1.);

  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = d1 * meanUp[num_equation_ - 1] + meanUp[0] * d6;

  Vector state2(num_equation_);
  for (int eq = 0; eq < num_equation_; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation_];

  Vector stateN = state2;
  for (int d = 0; d < dim_; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim_; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim_ == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation_);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation_; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim_, dim_);
    for (int d = 0; d < dim_; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim_ == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim_, dim_);
    mfem::CalcInverse(M, invM);
    Vector momN(dim_), momX(dim_);
    for (int d = 0; d < dim_; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim_; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation_; eq++) boundaryU[eq + bdrN * num_equation_] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::integrateOutlets_gpu(Vector &y, const Vector &x, const elementIndexingData &elem_index_data,
                                    const boundaryFaceIntegrationData &boundary_face_data, Array<int> &listElems,
                                    Array<int> &offsetsBoundaryU) {
#ifdef _GPU_
  double *d_y = y.Write();
  const int *d_elem_dofs_list = elem_index_data.dofs_list.Read();
  const int *d_elem_dof_off = elem_index_data.dof_offset.Read();
  const int *d_elem_dof_num = elem_index_data.dof_number.Read();

  const double *d_face_shape = boundary_face_data.shape.Read();
  const double *d_weight = boundary_face_data.quad_weight.Read();
  const int *d_face_el = boundary_face_data.el.Read();
  const int *d_face_num_quad = boundary_face_data.num_quad.Read();
  const int *d_listElems = listElems.Read();
  // const int *d_offsetBoundaryU = offsetsBoundaryU.Read();

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  const double *d_flux = face_flux_.Read();

  // clang-format off
  MFEM_FORALL_2D(n, numBdrElem, maxDofs, 1, 1,
  {
    MFEM_SHARED double Fcontrib[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // MFEM_SHARED double Fcontrib[216 * 20];
    double Rflux[gpudata::MAXEQUATIONS];  // double Rflux[20];

    const int el = d_listElems[n];

    const int Q    = d_face_num_quad[el];
    const int elID = d_face_el[el];

    const int elOffset = d_elem_dof_off[elID];
    const int elDof = d_elem_dof_num[elID];

    MFEM_FOREACH_THREAD(i, x, elDof) {
      for ( int eq = 0; eq < num_equation; eq++ ) Fcontrib[i+eq*elDof] = 0.;
    }
    MFEM_SYNC_THREAD;

    for (int q = 0; q < Q; q++) {  // loop over int. points
      const double weight = d_weight[el * maxIntPoints + q];

      // get interpolated data
      for (int eq = 0; eq < num_equation; eq++) {
        Rflux[eq] = weight * d_flux[eq + q*num_equation +n*maxIntPoints*num_equation];
      }

      MFEM_FOREACH_THREAD(i, x, elDof) {
        const double shape = d_face_shape[i + q * maxDofs + el * maxIntPoints * maxDofs];
        for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] -= Rflux[eq] * shape;
      }
      MFEM_SYNC_THREAD;
    }

    // add to global data
    MFEM_FOREACH_THREAD(i, x, elDof) {
      const int indexi = d_elem_dofs_list[elOffset + i];
      for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
    }
    MFEM_SYNC_THREAD;
  });  // end MFEM_FORALL_2D
#endif
  // clang-format on
}

void OutletBC::interpOutlet_gpu(const mfem::Vector &x, const elementIndexingData &elem_index_data,
                                mfem::ParGridFunction *Up, mfem::ParGridFunction *gradUp,
                                const boundaryFaceIntegrationData &boundary_face_data, Array<int> &listElems,
                                Array<int> &offsetsBoundaryU) {
#ifdef _GPU_
  const double *d_inputState = inputState.Read();
  const double *d_U = x.Read();
  const double *d_gradUp = gradUp->Read();
  const int *d_elem_dofs_list = elem_index_data.dofs_list.Read();
  const int *d_elem_dof_off = elem_index_data.dof_offset.Read();
  const int *d_elem_dof_num = elem_index_data.dof_number.Read();
  const double *d_face_shape = boundary_face_data.shape.Read();
  const double *d_normal = boundary_face_data.normal.Read();
  const double *d_xyz = boundary_face_data.xyz.Read();
  const int *d_face_el = boundary_face_data.el.Read();
  const int *d_face_num_quad = boundary_face_data.num_quad.Read();
  const int *d_listElems = listElems.Read();
  const int *d_offsetBoundaryU = offsetsBoundaryU.Read();
  const double *d_meanUp = meanUp.Read();
  double *d_boundaryU = boundaryU.ReadWrite();
  const double *d_tang1 = tangent1.Read();
  const double *d_tang2 = tangent2.Read();
  const double *d_inv = inverseNorm2cartesian.Read();

  double *d_flux = face_flux_.Write();

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const double Rg = mixture->GetGasConstant();
  const double gamma = mixture->GetSpecificHeatRatio();

  const WorkingFluid fluid = mixture->GetWorkingFluid();

  const double rL = this->refLength;
  const double area = this->area_;
  const double dtloc = this->dt;
  const Equations es = this->eqSystem;

  const OutletType type = outletType_;

  if ((fluid != DRY_AIR) && (type != SUB_P)) {
    mfem_error("Plasma only supports subsonic reflecting pressure outlet!\n");
    exit(-1);
  }

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;
  const int nvel = nvel_;

  const RiemannSolver *d_rsolver = rsolver;
  GasMixture *d_mix = d_mixture_;

  // MFEM_FORALL(n, numBdrElem, {
  MFEM_FORALL_2D(n, numBdrElem, maxIntPoints, 1, 1, {
    //
    double shape[gpudata::MAXDOFS];  // double shape[216];
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS], gradUp1[gpudata::MAXEQUATIONS * gpudata::MAXDIM],
        Rflux[gpudata::MAXEQUATIONS],
        nor[gpudata::MAXDIM];  // double u1[5], u2[5], gradUp1[5 * 3], Rflux[5], nor[3];
    double xyz[gpudata::MAXDIM];
    int index_i[gpudata::MAXDOFS];  // int index_i[216];

    const int el = d_listElems[n];
    const int offsetBdrU = d_offsetBoundaryU[n];
    const int Q = d_face_num_quad[el];
    const int elID = d_face_el[el];
    const int elOffset = d_elem_dof_off[elID];
    const int elDof = d_elem_dof_num[elID];

    // get data
    for (int i = 0; i < elDof; i++) {
      index_i[i] = d_elem_dofs_list[elOffset + i];
    }

    const int numActiveSpecies = d_mix->GetNumActiveSpecies();

    // for (int q = 0; q < Q; q++) {
    MFEM_FOREACH_THREAD(q, x, Q) {
      // zero state and gradient at this quad point
      for (int eq = 0; eq < num_equation; eq++) {
        u1[eq] = 0.;
        for (int d = 0; d < dim; d++) {
          gradUp1[eq + d * num_equation] = 0.;
        }
      }

      // extract shape functions at this quad point
      for (int j = 0; j < elDof; j++) {
        shape[j] = d_face_shape[j + q * maxDofs + el * maxIntPoints * maxDofs];
      }

      // extract normal vector at this quad point
      for (int d = 0; d < dim; d++) {
        nor[d] = d_normal[el * maxIntPoints * dim + q * dim + d];
        xyz[d] = d_xyz[el * maxIntPoints * dim + q * dim + d];
      }

      // interpolate to this quad point
      for (int eq = 0; eq < num_equation; eq++) {
        for (int i = 0; i < elDof; i++) {
          const int indexi = index_i[i];
          u1[eq] += d_U[indexi + eq * totDofs] * shape[i];
          for (int d = 0; d < dim; d++)
            gradUp1[eq + d * num_equation] += d_gradUp[indexi + eq * totDofs + d * num_equation * totDofs] * shape[i];
        }
      }

      // ensure non-negative densities
      for (int sp = 0; sp < numActiveSpecies; sp++) {
        const int sp_eq = nvel + 2 + sp;
        u1[sp_eq] = max(u1[sp_eq], 0.0);
      }

      // compute mirror state
      switch (type) {
        case OutletType::SUB_P:
          d_mix->modifyEnergyForPressure(u1, u2, d_inputState[0]);
          break;
        case OutletType::SUB_P_NR:
          computeNRSubPress_serial(offsetBdrU + q, &u1[0], &gradUp1[0], d_meanUp, dtloc, &u2[0], d_boundaryU,
                                   &d_inputState[0], &nor[0], d_tang1, d_tang2, d_inv, rL, gamma, Rg, elDof, dim,
                                   num_equation, es);
          break;
        case OutletType::SUB_MF_NR:
          computeNRSubMassFlow_serial(offsetBdrU + q, &u1[0], &gradUp1[0], d_meanUp, dtloc, &u2[0], d_boundaryU,
                                      d_inputState, &nor[0], d_tang1, d_tang2, d_inv, rL, area, gamma, Rg, elDof, dim,
                                      num_equation, es);
          break;
        case OutletType::SUB_MF_NR_PW:
          printf("OUTLET SUB_MF_NR_PW BC NOT IMPLEMENTED");
	  //exit(1);
          break;
        case OutletType::RESIST_IN:
          printf("OUTLET RESIST_IN BC NOT IMPLEMENTED");
	  //exit(1);	  
          break;	  
      }

      // compute flux
      d_rsolver->Eval_LF(u1, u2, nor, Rflux);

      if (d_rsolver->isAxisymmetric()) {
        const double radius = xyz[0];
        for (int eq = 0; eq < num_equation; eq++) {
          Rflux[eq] *= radius;
        }
      }

      for (int eq = 0; eq < num_equation; eq++) {
        d_flux[eq + q * num_equation + n * maxIntPoints * num_equation] = Rflux[eq];
      }
    }
  });
#endif
}
