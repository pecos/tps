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
#include "inletBC.hpp"

#include "dgNonlinearForm.hpp"
#include "riemann_solver.hpp"

// TODO(kevin): non-reflecting bc for plasam.
InletBC::InletBC(MPI_Groups *_groupsMPI, Equations _eqSystem, RiemannSolver *_rsolver, GasMixture *_mixture,
                 GasMixture *d_mixture, ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt,
                 const int _dim, const int _num_equation, int _patchNumber, double _refLength, InletType _bcType,
                 const Array<double> &_inputData, const int &_maxIntPoints, const int &_maxDofs, bool axisym)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber,
                        _refLength, axisym),
      groupsMPI(_groupsMPI),
      d_mixture_(d_mixture),
      inletType_(_bcType),
      maxIntPoints_(_maxIntPoints),
      maxDofs_(_maxDofs) {
  if ((mixture->GetWorkingFluid() != DRY_AIR) && (inletType_ != SUB_DENS_VEL)) {
    grvy_printf(GRVY_ERROR, "Plasma only supports subsonic reflecting density velocity inlet!\n");
    exit(-1);
  }
  inputState.UseDevice(true);
  inputState.SetSize(_inputData.Size());
  auto hinputState = inputState.HostWrite();
  // NOTE: regardless of dimension, inletBC saves first 4 elements for density and velocity.
  for (int i = 0; i < 4; i++) hinputState[i] = _inputData[i];

  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  if (numActiveSpecies_ > 0) {  // read species input state for multi-component flow.
    // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
    // NOTE: Inlet BC do not specify total energy, therefore skips one index.
    for (int sp = 0; sp < numActiveSpecies_; sp++) {
      // int inputIndex = (*mixtureToInputMap)[sp];
      // store species density into inputState in the order of mixture-sorted index.
      hinputState[4 + sp] = _inputData[0] * _inputData[4 + sp];
    }
  }

  // auto dinputState = inputState.ReadWrite();
  inputState.ReadWrite();

  groupsMPI->setPatch(_patchNumber);

  localMeanUp.UseDevice(true);
  localMeanUp.SetSize(num_equation_ + 1);
  localMeanUp = 0.;

  glob_sum.UseDevice(true);
  glob_sum.SetSize(num_equation_ + 1);
  glob_sum = 0.;

  meanUp.UseDevice(true);
  meanUp.SetSize(num_equation_);
  meanUp = 0.;
  auto hmeanUp = meanUp.HostWrite();

  hmeanUp[0] = 1.2;
  hmeanUp[1] = 60;
  hmeanUp[2] = 0;
  if (nvel_ == 3) hmeanUp[3] = 0.;
  hmeanUp[1 + nvel_] = 300.0;  // 101300;
  if (eqSystem == NS_PASSIVE) {
    hmeanUp[num_equation_ - 1] = 0.;
  } else if (numActiveSpecies_ > 0) {
    for (int sp = 0; sp < numActiveSpecies_; sp++) {
      hmeanUp[nvel_ + 2 + sp] = 0.0;
    }
  }

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

void InletBC::initBdrElemsShape() {
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
#endif
}

InletBC::~InletBC() {}

void InletBC::initBCs() {
  if (!BCinit) {
    // placeholder to compute aggregate mass flow rate for inlets

    MPI_Comm bcomm = groupsMPI->getComm(patchNumber);
    double area = aggregateArea(patchNumber, bcomm);
    int nfaces = aggregateBndryFaces(patchNumber, bcomm);

    if (groupsMPI->isGroupRoot(bcomm)) {
      grvy_printf(ginfo, "\n[INLET]: Patch number                      = %i\n", patchNumber);
      grvy_printf(ginfo, "[INLET]: Total Surface Area                = %.5e\n", area);
      grvy_printf(ginfo, "[INLET]: # of boundary faces               = %i\n", nfaces);
      grvy_printf(ginfo, "[INLET]: # of participating MPI partitions = %i\n", groupsMPI->groupSize(bcomm));
    }

#ifdef _GPU_
    face_flux_.UseDevice(true);
    face_flux_.SetSize(num_equation_ * maxIntPoints_ * listElems.Size());
    face_flux_ = 0.;
#endif

    BCinit = true;
  }
}

void InletBC::updateMean_gpu(ParGridFunction *Up, Vector &localMeanUp, const int num_equation, const int numBdrElems,
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

    // overall sum (for averaging)
    int interval = 1;
    // for(int interval=2;interval<=groupAveraging;interval*=2)
    while (interval < numBdrElems) {
      interval *= 2;
      if (el % interval == 0) {
        int el2 = el + interval / 2;
        for (int eq = 0; eq < num_equation; eq++) {
          if (el2 < numBdrElems)
            d_bdrUp[eq + el * num_equation * maxIntPoints] += d_bdrUp[eq + el2 * num_equation * maxIntPoints];
        }
      }
      MFEM_SYNC_THREAD;
    }
  });

  MFEM_FORALL(eq, num_equation, { d_localMeanUp[eq] = d_bdrUp[eq]; });

#endif
}

void InletBC::initBoundaryU(ParGridFunction *Up) {
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

  boundaryU.Read();
}

void InletBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux) {
  switch (inletType_) {
    case SUB_DENS_VEL:
      subsonicReflectingDensityVelocity(normal, stateIn, bdrFlux);
      break;
    case SUB_DENS_VEL_NR:
      subsonicNonReflectingDensityVelocity(normal, stateIn, gradState, bdrFlux);
      break;
    case SUB_VEL_CONST_ENT:
      subsonicNonReflectingDensityVelocity(normal, stateIn, gradState, bdrFlux);
      break;
  }
}

void InletBC::updateMean(IntegrationRules *intRules, ParGridFunction *Up) {
  if (inletType_ == SUB_DENS_VEL) return;
  bdrN = 0;

  int Nbdr = 0;

#ifdef _GPU_
  DGNonLinearForm::setToZero_gpu(bdrUp, bdrUp.Size());

  const int dofs = vfes->GetNDofs();

  updateMean_gpu(Up, localMeanUp, num_equation_, bdrElemsQ.Size() / 2, dofs, bdrUp, bdrElemsQ, bdrDofs, bdrShape,
                 maxIntPoints_, maxDofs_);

  if (!bdrUInit) initBoundaryU(Up);
#else

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
    // for(int i=0;i<Nbdr;i++)
    Vector iState(num_equation_), iUp(num_equation_);
    for (int i = 0; i < totNbdr; i++) {
      for (int eq = 0; eq < num_equation_; eq++) iUp[eq] = boundaryU[eq + i * num_equation_];
      mixture->GetConservativesFromPrimitives(iUp, iState);

      for (int eq = 0; eq < num_equation_; eq++) boundaryU[eq + i * num_equation_] = iState[eq];
    }
    bdrUInit = true;
  }
}

void InletBC::integrationBC(Vector &y,  // output
                            const Vector &x, const elementIndexingData &elem_index_data,
                            ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                            Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs) {
  interpInlet_gpu(x, elem_index_data, shapesBC, normalsWBC, intPointsElIDBC, listElems, offsetsBoundaryU);

  integrateInlets_gpu(y,  // output
                      x, elem_index_data, shapesBC, normalsWBC, intPointsElIDBC, listElems, offsetsBoundaryU);
}

void InletBC::subsonicNonReflectingDensityVelocity(Vector &normal, Vector &stateIn, DenseMatrix &gradState,
                                                   Vector &bdrFlux) {
  const double gamma = mixture->GetSpecificHeatRatio();
  // const double p = eqState->ComputePressure(stateIn, dim);

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim_; d++) mod += normal[d] * normal[d];
    unitNorm *= -1. / sqrt(mod);  // point into domain!!
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(nvel_);
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

  // velocity difference between actual and desired values
  Vector meanDV(nvel_);
  for (int d = 0; d < nvel_; d++) meanDV[d] = meanUp[1 + d] - inputState[1 + d];

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
  for (int d = 0; d < nvel_; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristic
  double L1 = 0.;
  for (int d = 0; d < dim_; d++) L1 += unitNorm[d] * normGrad[1 + d];  // dVn/dn
  L1 = dpdn - meanUp[0] * speedSound * L1;
  L1 *= meanVel[0] - speedSound;

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L5 = 0.;
  for (int d = 0; d < dim_; d++) L5 += meanDV[d] * unitNorm[d];
  L5 *= sigma * 2. * meanUp[0] * speedSound;

  double L3 = 0.;
  for (int d = 0; d < dim_; d++) L3 += meanDV[d] * tangent1[d];
  L3 *= sigma;

  double L4 = 0.;
  if (dim_ == 3) {
    for (int d = 0; d < dim_; d++) L4 += meanDV[d] * tangent2[d];
    L4 *= sigma;
  }

  double L2 = sigma * speedSound * speedSound * (meanUp[0] - inputState[0]) - 0.5 * L5;
  if (inletType_ == SUB_VEL_CONST_ENT) L2 = 0.;

  // calc vector d
  const double d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
  //   const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
  const double d3 = L3;
  const double d4 = L4;
  const double d5 = 0.5 * (L5 + L1);

  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
  bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
  if (nvel_ == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[1 + nvel_] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[1 + nvel_] += meanUp[0] * meanVel[1] * d3;
  if (nvel_ == 3) bdrFlux[1 + nvel_] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[1 + nvel_] += meanK * d1 + d5 / (gamma - 1.);

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
  for (int d = 0; d < nvel_; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim_; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim_ == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation_);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation_; i++) newU[i] = stateN[i] - dt * bdrFlux[i];
  if (eqSystem == NS_PASSIVE) newU[num_equation_ - 1] = 0.;

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

void InletBC::subsonicReflectingDensityVelocity(Vector &normal, Vector &stateIn, Vector &bdrFlux) {
  // NOTE: it is likely that for two-temperature case inlet will also specify electron temperature,
  // whether it is equal to the gas temperature or not.
  const double p = mixture->ComputePressure(stateIn);

  Vector state2(num_equation_);
  state2 = stateIn;

  state2[0] = inputState[0];
  state2[1] = inputState[0] * inputState[1];
  state2[2] = inputState[0] * inputState[2];
  if (nvel_ == 3) state2[3] = inputState[0] * inputState[3];

  if (eqSystem == NS_PASSIVE) {
    state2[num_equation_ - 1] = 0.;
  } else if (numActiveSpecies_ > 0) {
    for (int sp = 0; sp < numActiveSpecies_; sp++) {
      // NOTE: inlet BC does not specify total energy. therefore skips one index.
      // NOTE: regardless of dim_ension, inletBC save the first 4 elements for density and velocity.
      state2[nvel_ + 2 + sp] = inputState[4 + sp];
    }
  }

  // NOTE: If two-temperature, BC for electron temperature is T_e = T_h, where the total pressure is p.
  mixture->modifyEnergyForPressure(state2, state2, p, true);

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void InletBC::integrateInlets_gpu(Vector &y, const Vector &x, const elementIndexingData &elem_index_data,
                                  Vector &shapesBC, Vector &normalsWBC,
                                  Array<int> &intPointsElIDBC, Array<int> &listElems, Array<int> &offsetsBoundaryU) {
#ifdef _GPU_
  double *d_y = y.Write();
  const int *d_elem_dofs_list = elem_index_data.element_dofs_list.Read();
  const int *d_posDofIds = elem_index_data.posDofIds.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_listElems = listElems.Read();
  const int *d_offsetBoundaryU = offsetsBoundaryU.Read();

  const double *d_flux = face_flux_.Read();

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  MFEM_FORALL_2D(n, numBdrElem, maxDofs, 1, 1, {
    MFEM_SHARED double Fcontrib[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // MFEM_SHARED double Fcontrib[216 * 5];
    double Rflux[gpudata::MAXEQUATIONS];                                    // double Rflux[5];

    const int el = d_listElems[n];
    const int Q = d_intPointsElIDBC[2 * el];
    const int elID = d_intPointsElIDBC[2 * el + 1];
    const int elOffset = d_posDofIds[2 * elID];
    const int elDof = d_posDofIds[2 * elID + 1];

    MFEM_FOREACH_THREAD(i, x, elDof) {
      for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] = 0.;
    }
    MFEM_SYNC_THREAD;

    for (int q = 0; q < Q; q++) {  // loop over int. points
      const double weight = d_normW[dim + q * (dim + 1) + el * maxIntPoints * (dim + 1)];

      // get interpolated data
      for (int eq = 0; eq < num_equation; eq++) {
        Rflux[eq] = weight * d_flux[eq + q * num_equation + n * maxIntPoints * num_equation];
      }

      // sum contributions to integral
      MFEM_FOREACH_THREAD(i, x, elDof) {
        const double shape = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
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
  });
#endif
}

void InletBC::interpInlet_gpu(const mfem::Vector &x,  const elementIndexingData &elem_index_data,
                              mfem::Vector &shapesBC, mfem::Vector &normalsWBC, Array<int> &intPointsElIDBC,
                              Array<int> &listElems, Array<int> &offsetsBoundaryU) {
#ifdef _GPU_
  const double *d_inputState = inputState.Read();
  const double *d_U = x.Read();
  const int *d_elem_dofs_list = elem_index_data.element_dofs_list.Read();
  const int *d_posDofIds = elem_index_data.posDofIds.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_listElems = listElems.Read();
  const int *d_offsetBoundaryU = offsetsBoundaryU.Read();

  double *d_flux = face_flux_.Write();

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const double Rg = mixture->GetGasConstant();
  const double gamma = mixture->GetSpecificHeatRatio();

  const WorkingFluid fluid = mixture->GetWorkingFluid();

  const InletType type = inletType_;

  if ((fluid != DRY_AIR) && (type != SUB_DENS_VEL)) {
    mfem_error("Plasma only supports subsonic reflecting density velocity inlet!\n");
    exit(-1);
  }

  const int dim = dim_;
  const int nvel = nvel_;
  const int num_equation = num_equation_;
  const int numActiveSpecies = numActiveSpecies_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  const RiemannSolver *d_rsolver = rsolver;
  GasMixture *d_mix = d_mixture_;

  // MFEM_FORALL(n, numBdrElem, {
  MFEM_FORALL_2D(n, numBdrElem, maxIntPoints, 1, 1, {
    double shape[gpudata::MAXDOFS];  // double shape[216];
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS], Rflux[gpudata::MAXEQUATIONS],
        nor[gpudata::MAXDIM];  // double u1[5], u2[5], Rflux[5], nor[3];

    double p;

    const int el = d_listElems[n];
    const int Q = d_intPointsElIDBC[2 * el];
    const int elID = d_intPointsElIDBC[2 * el + 1];
    const int elOffset = d_posDofIds[2 * elID];
    const int elDof = d_posDofIds[2 * elID + 1];

    // for (int q = 0; q < Q; q++) {
    MFEM_FOREACH_THREAD(q, x, Q) {
      for (int d = 0; d < dim; d++) nor[d] = d_normW[d + q * (dim + 1) + el * maxIntPoints * (dim + 1)];

      for (int i = 0; i < elDof; i++) {
        shape[i] = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
      }

      for (int eq = 0; eq < num_equation; eq++) {
        u1[eq] = 0.;
        for (int i = 0; i < elDof; i++) {
          const int indexi = d_elem_dofs_list[elOffset + i];
          u1[eq] += d_U[indexi + eq * totDofs] * shape[i];
        }
      }

      // compute mirror state
      switch (type) {
        case InletType::SUB_DENS_VEL:
          p = d_mix->ComputePressure(u1);
          pluginInputState(d_inputState, u2, nvel, numActiveSpecies);
          d_mix->modifyEnergyForPressure(u2, u2, p, true);
          break;
        case InletType::SUB_DENS_VEL_NR:
          printf("INLET BC NOT IMPLEMENTED");
          break;
        case InletType::SUB_VEL_CONST_ENT:
          printf("INLET BC NOT IMPLEMENTED");
          break;
      }

      // compute flux
      d_rsolver->Eval_LF(u1, u2, nor, Rflux);

      // save to global memory
      for (int eq = 0; eq < num_equation; eq++) {
        d_flux[eq + q * num_equation + n * maxIntPoints * num_equation] = Rflux[eq];
      }
    }  // end loop over intergration points
  });
#endif
}
