// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

OutletBC::OutletBC(MPI_Groups *_groupsMPI, RiemannSolver *_rsolver, EquationOfState *_eqState,
                   ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt, const int _dim,
                   const int _num_equation, int _patchNumber, double _refLength, OutletType _bcType,
                   const Array<double> &_inputData, const int &_maxIntPoints, const int &_maxDofs)
    : BoundaryCondition(_rsolver, _eqState, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber, _refLength),
      groupsMPI(_groupsMPI),
      outletType(_bcType),
      inputState(_inputData),
      maxIntPoints(_maxIntPoints),
      maxDofs(_maxDofs) {
  groupsMPI->setAsOutlet(_patchNumber);

  meanUp.UseDevice(true);
  meanUp.SetSize(num_equation);
  meanUp = 0.;

  localMeanUp.UseDevice(true);
  localMeanUp.SetSize(num_equation + 1);
  localMeanUp = 0.;

  glob_sum.UseDevice(true);
  glob_sum.SetSize(num_equation + 1);
  glob_sum = 0.;

  auto hmeanUp = meanUp.HostWrite();

  hmeanUp[0] = 1.2;
  hmeanUp[1] = 60;
  hmeanUp[2] = 0;
  if (dim == 3) hmeanUp[3] = 0.;
  hmeanUp[num_equation - 1] = 101300;

  area = 0.;
  parallelAreaComputed = false;

  Array<double> coords;

  // assuming planar outlets (for GPU only)
  Vector normal;
  normal.UseDevice(false);
  normal.SetSize(dim);
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
  boundaryU.SetSize(bdrN * num_equation);
  boundaryU = 0.;
  auto hboundaryU = boundaryU.HostWrite();
  for (int i = 0; i < bdrN; i++) {
    Vector iState;
    iState.UseDevice(false);
    iState.SetSize(num_equation);
    for (int eq = 0; eq < num_equation; eq++) iState(eq) = hmeanUp[eq];

    double gamma = eqState->GetSpecificHeatRatio();
    double k = 0.;
    for (int d = 0; d < dim; d++) k += iState[1] * iState[1];
    double rE = iState[3] / (gamma - 1.) + 0.5 * iState[0] * k;

    for (int d = 0; d < dim; d++) iState[1 + d] *= iState[0];
    iState[num_equation - 1] = rE;
    for (int eq = 0; eq < num_equation; eq++) hboundaryU[eq + i * num_equation] = iState[eq];
  }
  bdrUInit = false;
  bdrN = 0;

  // init. unit tangent vector tangent1
  tangent1.UseDevice(true);
  tangent2.UseDevice(true);
  tangent1.SetSize(dim);
  tangent2.SetSize(dim);
  tangent1 = 0.;
  tangent2 = 0.;
  auto htan1 = tangent1.HostWrite();
  auto htan2 = tangent2.HostWrite();
  Array<double> coords1, coords2;
  for (int d = 0; d < dim; d++) {
    coords1.Append(coords[d]);
    coords2.Append(coords[d + 3]);
  }

  double modVec = 0.;
  for (int d = 0; d < dim; d++) {
    modVec += (coords2[d] - coords1[d]) * (coords2[d] - coords1[d]);
    htan1[d] = coords2[d] - coords1[d];
  }
  for (int d = 0; d < dim; d++) htan1[d] *= 1. / sqrt(modVec);

  Vector unitNorm;
  unitNorm.UseDevice(false);
  unitNorm.SetSize(dim);
  unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }
  if (dim == 3) {
    htan2[0] = unitNorm[1] * htan1[2] - unitNorm[2] * htan1[1];
    htan2[1] = unitNorm[2] * htan1[0] - unitNorm[0] * htan1[2];
    htan2[2] = unitNorm[0] * htan1[1] - unitNorm[1] * htan1[0];
  }

  DenseMatrix M(dim, dim);
  for (int d = 0; d < dim; d++) {
    M(0, d) = unitNorm[d];
    M(1, d) = htan1[d];
    if (dim == 3) M(2, d) = htan2[d];
  }

  DenseMatrix invM(dim, dim);
  mfem::CalcInverse(M, invM);

  inverseNorm2cartesian.UseDevice(true);
  inverseNorm2cartesian.SetSize(dim * dim);
  auto hinv = inverseNorm2cartesian.HostWrite();
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) hinv[i + j * dim] = invM(i, j);

  initBdrElemsShape();

  meanUp.Read();
  boundaryU.Read();
  tangent1.Read();
  tangent2.Read();
}

void OutletBC::initBdrElemsShape() {
  bdrShape.UseDevice(true);
  int Nbdr = boundaryU.Size() / num_equation;
  bdrShape.SetSize(maxIntPoints * maxDofs * Nbdr);
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
  bdrDofs.SetSize(maxDofs * elCount);
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

      for (int i = 0; i < elDofs; i++) hbdrDofs[i + elCount * maxDofs] = dofs[i];

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

        for (int n = 0; n < elDofs; n++) hbdrShape[n + i * maxDofs + elCount * maxIntPoints * maxDofs] = shape(n);
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
  bdrUp.SetSize(size_bdrUp * num_equation * maxIntPoints);
  bdrUp = 0.;
  bdrUp.Read();

  offsetsBoundaryU.ReadWrite();
#endif
}

OutletBC::~OutletBC() {}

void OutletBC::initBCs() {
  if (!BCinit) {
    computeParallelArea();
  }

  BCinit = true;
}

void OutletBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  switch (outletType) {
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
  }
}

void OutletBC::computeParallelArea() {
  if (parallelAreaComputed) return;

  MPI_Comm bcomm = groupsMPI->getOutletComm();
  area = aggregateArea(patchNumber, bcomm);
  int nfaces = aggregateBndryFaces(patchNumber, bcomm);
  parallelAreaComputed = true;

  if (groupsMPI->isGroupRoot(bcomm)) {
    grvy_printf(ginfo, "\n[OUTLET]: Patch number                      = %i\n", patchNumber);
    grvy_printf(ginfo, "[OUTLET]: Total Surface Area                = %.5f\n", area);
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
    //     double elUp[5*maxDofs];
    //     double shape[maxDofs];
    // compiler does not allow to have maxDofs.
    // currently maxDofs = 64;
    double elUp[5 * 216];
    double shape[216];
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
      elUp.SetSize(elDofs * num_equation);
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
        Vector iUp(num_equation);

        for (int eq = 0; eq < num_equation; eq++) {
          double sum = 0.;
          for (int d = 0; d < elDofs; d++) {
            sum += shape[d] * elUp(d + eq * elDofs);
          }
          hboundaryU[eq + Nbdr * num_equation] = sum;
        }
        Nbdr++;
      }
    }
  }

  boundaryU.ReadWrite();
}

void OutletBC::updateMean(IntegrationRules *intRules, ParGridFunction *Up) {
  if (outletType == SUB_P) return;

  bdrN = 0;

  int Nbdr = 0;

#ifdef _GPU_
  DGNonLinearForm::setToZero_gpu(bdrUp, bdrUp.Size());

  const int dofs = vfes->GetNDofs();

  updateMean_gpu(Up, localMeanUp, num_equation, bdrElemsQ.Size() / 2, dofs, bdrUp, bdrElemsQ, bdrDofs, bdrShape,
                 maxIntPoints, maxDofs);

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
      elUp.SetSize(elDofs * num_equation);
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
        Vector iUp(num_equation);

        for (int eq = 0; eq < num_equation; eq++) {
          double sum = 0.;
          for (int d = 0; d < elDofs; d++) {
            sum += shape[d] * elUp(d + eq * elDofs);
          }
          localMeanUp[eq] += sum;
          if (!bdrUInit) boundaryU[eq + Nbdr * num_equation] = sum;
        }
        Nbdr++;
      }
    }
  }
#endif

  double *h_localMeanUp = localMeanUp.HostReadWrite();
  int totNbdr = boundaryU.Size() / num_equation;
  h_localMeanUp[num_equation] = static_cast<double>(totNbdr);

  double *h_sum = glob_sum.HostWrite();
  MPI_Allreduce(h_localMeanUp, h_sum, num_equation + 1, MPI_DOUBLE, MPI_SUM, groupsMPI->getOutletComm());

  BoundaryCondition::copyValues(glob_sum, meanUp, 1. / h_sum[num_equation]);

  if (!bdrUInit) {
    auto hboundaryU = boundaryU.HostReadWrite();
    for (int i = 0; i < totNbdr; i++) {
      Vector iState(num_equation);
      for (int eq = 0; eq < num_equation; eq++) iState[eq] = hboundaryU[eq + i * num_equation];
      double gamma = eqState->GetSpecificHeatRatio();
      double k = 0.;
      for (int d = 0; d < dim; d++) k += iState[1 + d] * iState[1 + d];
      double rE = iState[num_equation - 1] / (gamma - 1.) + 0.5 * iState[0] * k;

      for (int d = 0; d < dim; d++) iState[1 + d] *= iState[0];
      iState[num_equation - 1] = rE;
      for (int eq = 0; eq < num_equation; eq++) hboundaryU[eq + i * num_equation] = iState[eq];
    }
    bdrUInit = true;
  }
}

void OutletBC::integrationBC(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                             ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                             Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs) {
  integrateOutlets_gpu(outletType, inputState, dt,
                       y,  // output
                       x, nodesIDs, posDofIds, Up, gradUp, meanUp, boundaryU, tangent1, tangent2, inverseNorm2cartesian,
                       shapesBC, normalsWBC, intPointsElIDBC, listElems, offsetsBoundaryU, maxIntPoints, maxDofs, dim,
                       num_equation, eqState->GetSpecificHeatRatio(), eqState->GetGasConstant(), refLength, area);
}

void OutletBC::subsonicNonReflectingPressure(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = eqState->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim);
  meanVel = 0.;
  for (int d = 0; d < dim; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  // normal gradients
  Vector normGrad(num_equation);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }

  const double speedSound = sqrt(gamma * meanUp[num_equation - 1] / meanUp[0]);
  double meanK = 0.;
  for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  double L2 = speedSound * speedSound * normGrad[0] - normGrad[num_equation - 1];
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim == 3) {
    for (int d = 0; d < dim; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 = normGrad[num_equation - 1] + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  // const double p = eqState->ComputePressure(stateIn, dim);

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L1 = sigma * (meanUp[num_equation - 1] - inputState[0]);

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
  if (dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[num_equation - 1] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[num_equation - 1] += meanUp[0] * meanVel[1] * d3;
  if (dim == 3) bdrFlux[num_equation - 1] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[num_equation - 1] += meanK * d1 + d5 / (gamma - 1.);

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

  Vector state2(num_equation);
  for (int eq = 0; eq < num_equation; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation];

  Vector stateN = state2;
  for (int d = 0; d < dim; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim, dim);
    for (int d = 0; d < dim; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim, dim);
    mfem::CalcInverse(M, invM);
    Vector momN(dim), momX(dim);
    for (int d = 0; d < dim; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation; eq++) boundaryU[eq + bdrN * num_equation] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::subsonicReflectingPressure(Vector &normal, Vector &stateIn, Vector &bdrFlux) {
  const double gamma = eqState->GetSpecificHeatRatio();
  Vector state2(num_equation);
  state2 = stateIn;
  double k = 0.;
  for (int d = 0; d < dim; d++) k += stateIn[1 + d] * stateIn[1 + d];
  state2[num_equation - 1] = inputState[0] / (gamma - 1.) + 0.5 * k / stateIn[0];

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::subsonicNonRefMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = eqState->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim);
  meanVel = 0.;
  for (int d = 0; d < dim; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  // normal gradients
  Vector normGrad(num_equation);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }

  const double speedSound = sqrt(gamma * meanUp[num_equation - 1] / meanUp[0]);
  double meanK = 0.;
  for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  double L2 = speedSound * speedSound * normGrad[0] - normGrad[num_equation - 1];
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim == 3) {
    for (int d = 0; d < dim; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 = normGrad[num_equation - 1] + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  // const double p = eqState->ComputePressure(stateIn, dim);

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L1 = -sigma * (meanVel[0] - inputState[0] / meanUp[0] / area);
  // cout<<inputState[0]<<" "<<meanUp[0]<<" "<<area<<endl;
  L1 *= meanUp[0] * speedSound;

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
  if (dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[num_equation - 1] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[num_equation - 1] += meanUp[0] * meanVel[1] * d3;
  if (dim == 3) bdrFlux[num_equation - 1] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[num_equation - 1] += meanK * d1 + d5 / (gamma - 1.);

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

  Vector state2(num_equation);
  for (int eq = 0; eq < num_equation; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation];

  Vector stateN = state2;
  for (int d = 0; d < dim; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim, dim);
    for (int d = 0; d < dim; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim, dim);
    mfem::CalcInverse(M, invM);
    Vector momN(dim), momX(dim);
    for (int d = 0; d < dim; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation; eq++) boundaryU[eq + bdrN * num_equation] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::subsonicNonRefPWMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = eqState->GetSpecificHeatRatio();

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for (int d = 0; d < dim; d++) mod += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(mod);
  }

  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim);
  meanVel = 0.;
  for (int d = 0; d < dim; d++) {
    meanVel[0] += unitNorm[d] * meanUp[d + 1];
    meanVel[1] += tangent1[d] * meanUp[d + 1];
  }
  if (dim == 3) {
    tangent2[0] = unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1];
    tangent2[1] = unitNorm[2] * tangent1[0] - unitNorm[0] * tangent1[2];
    tangent2[2] = unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0];
    for (int d = 0; d < dim; d++) meanVel[2] += tangent2[d] * meanUp[d + 1];
  }

  // normal gradients
  Vector normGrad(num_equation);
  normGrad = 0.;
  for (int eq = 0; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) normGrad[eq] += unitNorm[d] * gradState(eq, d);
  }
  double normVel = 0.;
  for (int d = 0; d < dim; d++) normVel += stateIn[1 + d] * unitNorm[d];
  normVel /= stateIn[0];

  const double speedSound = sqrt(gamma * meanUp[num_equation - 1] / meanUp[0]);
  double meanK = 0.;
  for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
  meanK *= 0.5;

  // compute outgoing characteristics
  double L2 = speedSound * speedSound * normGrad[0] - normGrad[num_equation - 1];
  L2 *= meanVel[0];

  double L3 = 0;
  for (int d = 0; d < dim; d++) L3 += tangent1[d] * normGrad[1 + d];
  L3 *= meanVel[0];

  double L4 = 0.;
  if (dim == 3) {
    for (int d = 0; d < dim; d++) L4 += tangent2[d] * normGrad[1 + d];
    L4 *= meanVel[0];
  }

  double L5 = 0.;
  for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
  //   L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 = normGrad[num_equation - 1] + meanUp[0] * speedSound * L5;
  L5 *= meanVel[0] + speedSound;

  // const double p = eqState->ComputePressure(stateIn, dim);

  // estimate ingoing characteristic
  const double sigma = speedSound / refLength;
  double L1 = -sigma * (normVel - inputState[0] / meanUp[0] / area);
  // cout<<inputState[0]<<" "<<meanUp[0]<<" "<<area<<endl;
  L1 *= meanUp[0] * speedSound;

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
  if (dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
  bdrFlux[num_equation - 1] = meanUp[0] * meanVel[0] * d2;
  bdrFlux[num_equation - 1] += meanUp[0] * meanVel[1] * d3;
  if (dim == 3) bdrFlux[num_equation - 1] += meanUp[0] * meanVel[2] * d4;
  bdrFlux[num_equation - 1] += meanK * d1 + d5 / (gamma - 1.);

  Vector state2(num_equation);
  for (int eq = 0; eq < num_equation; eq++) state2[eq] = boundaryU[eq + bdrN * num_equation];

  Vector stateN = state2;
  for (int d = 0; d < dim; d++) stateN[1 + d] = 0.;
  for (int d = 0; d < dim; d++) {
    stateN[1] += state2[1 + d] * unitNorm[d];
    stateN[2] += state2[1 + d] * tangent1[d];
    if (dim == 3) stateN[3] += state2[1 + d] * tangent2[d];
  }

  Vector newU(num_equation);
  // for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for (int i = 0; i < num_equation; i++) newU[i] = stateN[i] - dt * bdrFlux[i];

  // transform back into x-y coords
  {
    DenseMatrix M(dim, dim);
    for (int d = 0; d < dim; d++) {
      M(0, d) = unitNorm[d];
      M(1, d) = tangent1[d];
      if (dim == 3) M(2, d) = tangent2[d];
    }

    DenseMatrix invM(dim, dim);
    mfem::CalcInverse(M, invM);
    Vector momN(dim), momX(dim);
    for (int d = 0; d < dim; d++) momN[d] = newU[1 + d];
    invM.Mult(momN, momX);
    for (int d = 0; d < dim; d++) newU[1 + d] = momX[d];
  }
  for (int eq = 0; eq < num_equation; eq++) boundaryU[eq + bdrN * num_equation] = newU[eq];
  bdrN++;

  rsolver->Eval(stateIn, state2, normal, bdrFlux, true);
}

void OutletBC::integrateOutlets_gpu(const OutletType type, const Array<double> &inputState, const double &dt, Vector &y,
                                    const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                                    ParGridFunction *Up, ParGridFunction *gradUp, Vector &meanUp, Vector &boundaryU,
                                    Vector &tangent1, Vector &tangent2, Vector &inverseNorm2cartesian, Vector &shapesBC,
                                    Vector &normalsWBC, Array<int> &intPointsElIDBC, Array<int> &listElems,
                                    Array<int> &offsetsBoundaryU, const int &maxIntPoints, const int &maxDofs,
                                    const int &dim, const int &num_equation, const double &gamma, const double &Rg,
                                    const double &refLength, const double &area) {
#ifdef _GPU_
  const double *d_inputState = inputState.Read();
  double *d_y = y.Write();
  const double *d_U = x.Read();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  //   const double *d_Up = Up->Read();
  const double *d_gradUp = gradUp->Read();
  const double *d_meanUp = meanUp.Read();
  double *d_boundaryU = boundaryU.ReadWrite();
  const double *d_tang1 = tangent1.Read();
  const double *d_tang2 = tangent2.Read();
  const double *d_inv = inverseNorm2cartesian.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_listElems = listElems.Read();
  const int *d_offsetBoundaryU = offsetsBoundaryU.Read();

  const int totDofs = x.Size() / num_equation;
  const int numBdrElem = listElems.Size();

  MFEM_FORALL_2D(n,numBdrElem,maxDofs,1,1,      // NOLINT
  {
    MFEM_FOREACH_THREAD(i,x,maxDofs) {          // NOLINT
      MFEM_SHARED double Ui[216*5], Fcontrib[216*5], gradUpi[216*3*5];
      MFEM_SHARED double shape[216];
      MFEM_SHARED double Rflux[5], u1[5], u2[5], gradUp[5*3], nor[3];
      MFEM_SHARED double weight;

      const int el = d_listElems[n];
      const int offsetBdrU = d_offsetBoundaryU[n];

      const int Q    = d_intPointsElIDBC[2*el  ];
      const int elID = d_intPointsElIDBC[2*el+1];

      const int elOffset = d_posDofIds[2*elID  ];
      const int elDof    = d_posDofIds[2*elID+1];
      int indexi;
      if (i < elDof)
        indexi = d_nodesIDs[elOffset+i];

      // retreive data
      for (int eq=0; eq < num_equation; eq++) {
        if (i < elDof) {
          Ui[i + eq * elDof] = d_U[indexi + eq * totDofs];
          for (int d = 0; d < dim; d++)
            gradUpi[i + eq * elDof + d * num_equation * elDof] =
                d_gradUp[indexi + eq * totDofs + d * num_equation * totDofs];

          Fcontrib[i + eq * elDof] = 0.;
        }
      }

      for (int q=0; q < Q; q++) {  // loop over int. points
        if (i < elDof) shape[i] = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
        if (i < dim) nor[i] = d_normW[i + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
        if (dim == 2 && i == maxDofs - 2) nor[2] = 0.;
        if (i == maxDofs - 1) weight = d_normW[dim + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
        MFEM_SYNC_THREAD;

        // interpolate to int. point
        if (i < num_equation) {
          u1[i] = 0.;
          for (int d = 0; d < dim; d++) gradUp[i + d * num_equation] = 0.;

          for (int k = 0; k < elDof; k++) {
            u1[i] += Ui[k + i * elDof] * shape[k];
            for (int d = 0; d < dim; d++)
              gradUp[i + d * num_equation] += gradUpi[k + i * elDof + d * num_equation * elDof] * shape[k];
          }
        }
        MFEM_SYNC_THREAD;

        // compute mirror state
        switch (type) {
          case OutletType::SUB_P:
            if (i < num_equation)
              computeSubPressure(i, &u1[0], &u2[0], &nor[0], d_inputState[0], gamma, dim, num_equation);
            break;
          case OutletType::SUB_P_NR:
            computeNRSubPress(i, offsetBdrU + q, &u1[0], &gradUp[0], d_meanUp, dt, &u2[0], d_boundaryU,
                              &d_inputState[0], &nor[0], d_tang1, d_tang2, d_inv, refLength, gamma, elDof,
                              dim, num_equation);
            break;
          case OutletType::SUB_MF_NR:
            computeNRSubMassFlow(i, offsetBdrU + q, &u1[0], &gradUp[0], d_meanUp, dt, &u2[0], d_boundaryU,
                                 d_inputState, &nor[0], d_tang1, d_tang2, d_inv, refLength, area, gamma, elDof,
                                 dim, num_equation);
            break;
          case OutletType::SUB_MF_NR_PW:
            computeNR_PW_SubMF(i, offsetBdrU + q, &u1[0], &gradUp[0], d_meanUp, dt, &u2[0], d_boundaryU, d_inputState,
                               &nor[0], d_tang1, d_tang2, d_inv, refLength, area, gamma, elDof, dim, num_equation);
            break;
        }
        MFEM_SYNC_THREAD;

        // compute flux
        if (i == 0) RiemannSolver::riemannLF_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
        MFEM_SYNC_THREAD;

        // sum contributions to integral
        if (i < elDof) {
          for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] -= Rflux[eq] * shape[i] * weight;
        }
        MFEM_SYNC_THREAD;
      }

      // add to global data
      if (i < elDof) {
        for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
      }
    }
  });
#endif
}
