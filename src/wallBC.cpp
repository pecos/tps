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
#include "wallBC.hpp"

#include "riemann_solver.hpp"

WallBC::WallBC(RiemannSolver *_rsolver, GasMixture *_mixture, Equations _eqSystem, Fluxes *_fluxClass,
               ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt, const int _dim,
               const int _num_equation, int _patchNumber, WallType _bcType, const Array<double> _inputData,
               const Array<int> &_intPointsElIDBC, const int &_maxIntPoints)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber,
                        1),  // so far walls do not require ref. length. Left at 1
      wallType(_bcType),
      fluxClass(_fluxClass),
      intPointsElIDBC(_intPointsElIDBC),
      maxIntPoints(_maxIntPoints) {
  if (wallType == VISC_ISOTH) {
    wallTemp = _inputData[0];
  }
}

WallBC::~WallBC() {}

void WallBC::initBCs() {
  if (!BCinit) {
    buildWallElemsArray(intPointsElIDBC);

#ifdef _GPU_
    interpolated_Ubdr_.UseDevice(true);
    interpolatedGradUpbdr_.UseDevice(true);

    interpolated_Ubdr_.SetSize(num_equation * maxIntPoints * listElems.Size());
    interpolatedGradUpbdr_.SetSize(dim * num_equation * maxIntPoints * listElems.Size());

    interpolated_Ubdr_ = 0.;
    interpolatedGradUpbdr_ = 0.;
#endif

    BCinit = true;
  }
}

void WallBC::buildWallElemsArray(const Array<int> &intPointsElIDBC) {
  auto hlistElems = listElems.HostRead();  // this is actually a list of faces
  auto hintPointsElIDBC = intPointsElIDBC.HostRead();

  std::vector<int> unicElems;
  unicElems.clear();

  for (int f = 0; f < listElems.Size(); f++) {
    int bcFace = hlistElems[f];
    int elID = hintPointsElIDBC[2 * bcFace + 1];

    bool elInList = false;
    for (int i = 0; i < unicElems.size(); i++) {
      if (unicElems[i] == elID) elInList = true;
    }
    if (!elInList) unicElems.push_back(elID);
  }

  wallElems.SetSize(7 * unicElems.size());
  wallElems = -1;
  auto hwallElems = wallElems.HostWrite();
  for (int el = 0; el < wallElems.Size() / 7; el++) {
    int elID = unicElems[el];
    for (int f = 0; f < listElems.Size(); f++) {
      int bcFace = hlistElems[f];
      int elID2 = hintPointsElIDBC[2 * bcFace + 1];
      if (elID2 == elID) {
        int nf = hwallElems[0 + 7 * el];
        if (nf == -1) nf = 0;
        nf++;
        hwallElems[nf + 7 * el] = f;
        hwallElems[0 + 7 * el] = nf;
      }
    }
  }
  auto dwallElems = wallElems.ReadWrite();
}

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  switch (wallType) {
    case INV:
      computeINVwallFlux(normal, stateIn, gradState, bdrFlux);
      break;
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal, stateIn, gradState, bdrFlux);
      break;
    case VISC_ISOTH:
      computeIsothermalWallFlux(normal, stateIn, gradState, bdrFlux);
      break;
  }
}

void WallBC::integrationBC(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                           ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                           Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs) {
  interpWalls_gpu(wallType, wallTemp, interpolated_Ubdr_, interpolatedGradUpbdr_, x, nodesIDs, posDofIds, Up, gradUp,
                  shapesBC, normalsWBC, intPointsElIDBC, wallElems, listElems, maxIntPoints, maxDofs, dim,
                  num_equation);

  integrateWalls_gpu(wallType, wallTemp,
                     y,  // output
                     x, interpolated_Ubdr_, interpolatedGradUpbdr_, nodesIDs, posDofIds, Up, gradUp, shapesBC,
                     normalsWBC, intPointsElIDBC, wallElems, listElems, eqSystem, maxIntPoints, maxDofs, dim,
                     num_equation, mixture);
}

void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  Vector vel(dim);
  for (int d = 0; d < dim; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  double norm = 0.;
  for (int d = 0; d < dim; d++) norm += normal[d] * normal[d];
  norm = sqrt(norm);

  Vector unitN(dim);
  for (int d = 0; d < dim; d++) unitN[d] = normal[d] / norm;

  double vn = 0;
  for (int d = 0; d < dim; d++) vn += vel[d] * unitN[d];

  Vector stateMirror(num_equation);

  stateMirror[0] = stateIn[0];
  stateMirror[1] = stateIn[0] * (vel[0] - 2. * vn * unitN[0]);
  stateMirror[2] = stateIn[0] * (vel[1] - 2. * vn * unitN[1]);
  if (dim == 3) stateMirror[3] = stateIn[0] * (vel[2] - 2. * vn * unitN[2]);
  stateMirror[1 + dim] = stateIn[1 + dim];
  if (eqSystem == NS_PASSIVE) stateMirror[num_equation - 1] = stateIn[num_equation - 1];

  rsolver->Eval(stateIn, stateMirror, normal, bdrFlux);

  // evaluate viscous fluxes at the wall
  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateMirror, gradState, viscFw);

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  double p = mixture->ComputePressure(stateIn);
  double T = mixture->ComputeTemperature(stateIn);

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg = mixture->GetGasConstant();

  Vector wallState = stateIn;
  for (int d = 0; d < dim; d++) wallState[1 + d] = 0.;
  wallState[1 + dim] = p / (gamma - 1.);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation - 1] = 0.;

  // incoming visc flux
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, viscF);

  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  {
    double normN = 0.;
    for (int d = 0; d < dim; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);
  }

  // modify gradient temperature so dT/dn=0 at the wall
  double DrhoDn = 0.;
  for (int d = 0; d < dim; d++) DrhoDn += gradState(0, d) * unitNorm[d];
  double dpdn = Rg * T * DrhoDn;  // this is the BC -> grad(T)*normal=0

  Vector gradP(dim);
  double old_dpdn = 0.;
  for (int d = 0; d < dim; d++) {
    Vector grads(gradState.GetColumn(d), num_equation);
    double dpdx = mixture->ComputePressureDerivative(grads, stateIn, false);
    gradP(d) = dpdx;
    old_dpdn += dpdx * unitNorm(d);
  }

  // force the new dp/dn
  for (int d = 0; d < dim; d++) gradP(d) += (-old_dpdn + dpdn) * unitNorm(d);

  // modify grad(T) with the corrected grad(p)
  for (int d = 0; d < dim; d++) {
    // temperature gradient for ideal Dry Air
    gradState(1 + dim, d) = T * (gradP(d) / p - gradState(0, d) / stateIn[0]);
  }

  if (eqSystem == NS_PASSIVE) {
    for (int d = 0; d < dim; d++) gradState(num_equation - 1, d) = 0.;
  }

  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(wallState, gradState, viscFw);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = mixture->GetSpecificHeatRatio();

  Vector wallState(num_equation);
  wallState[0] = stateIn[0];
  for (int d = 0; d < dim; d++) wallState[1 + d] = 0.;

  wallState[1 + dim] = mixture->GetGasConstant() / (gamma - 1.);  // Cv
  wallState[1 + dim] *= stateIn[0] * wallTemp;

  if (eqSystem == NS_PASSIVE) wallState[num_equation - 1] = stateIn[num_equation - 1];

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // evaluate viscous fluxes at the wall
  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(wallState, gradState, viscFw);

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::integrateWalls_gpu(const WallType type, const double &wallTemp, Vector &y, const Vector &x,
                                Vector &interpolated_Ubdr_, Vector &interpolatedGradUpbdr_, const Array<int> &nodesIDs,
                                const Array<int> &posDofIds, ParGridFunction *Up, ParGridFunction *gradUp,
                                Vector &shapesBC, Vector &normalsWBC, Array<int> &intPointsElIDBC,
                                Array<int> &wallElems, Array<int> &listElems, const Equations &eqSystem,
                                const int &maxIntPoints, const int &maxDofs, const int &dim, const int &num_equation,
                                GasMixture *mixture) {
#ifdef _GPU_
  double *d_y = y.Write();
  //   const double *d_U = x.Read();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();

  const int totDofs = x.Size() / num_equation;
  const int numBdrElem = listElems.Size();

  const double Rg = mixture->GetGasConstant();
  const double gamma = mixture->GetSpecificHeatRatio();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();

  const double *d_interpolU = interpolated_Ubdr_.Read();
  const double *d_interpGrads = interpolatedGradUpbdr_.Read();

  // clang-format on
  MFEM_FORALL_2D(el, wallElems.Size() / 7, maxDofs, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, maxDofs) {
      MFEM_SHARED double Fcontrib[216 * 20];
      MFEM_SHARED double shape[216];
      MFEM_SHARED double Rflux[20], u1[20], u2[20], nor[3], gradUpi[20 * 3];
      MFEM_SHARED double vF1[20 * 3], vF2[20 * 3];
      MFEM_SHARED double weight;

      const int numFaces = d_wallElems[0 + el * 7];
      bool elemDataRecovered = false;
      int elOffset;
      int elDof;
      int elID;
      int indexi;

      for (int f = 0; f < numFaces; f++) {
    const int n = d_wallElems[1 + f + el * 7];

    const int el = d_listElems[n];
    const int Q = d_intPointsElIDBC[2 * el];

    if (!elemDataRecovered) {
      elID = d_intPointsElIDBC[2 * el + 1];

      elOffset = d_posDofIds[2 * elID];
      elDof = d_posDofIds[2 * elID + 1];

      if (i < elDof) indexi = d_nodesIDs[elOffset + i];
    }

    // set contriution to 0
    if (i < elDof && !elemDataRecovered) {
      for (int eq = 0; eq < num_equation; eq++) {
        Fcontrib[i + eq * elDof] = 0.;
      }
      elemDataRecovered = true;
    }

    for (int q = 0; q < Q; q++) {  // loop over int. points
      if (i < elDof) shape[i] = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
      if (i < dim) nor[i] = d_normW[i + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
      if (dim == 2 && i == maxDofs - 2) nor[2] = 0.;
      if (i == maxDofs - 1) weight = d_normW[dim + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
      MFEM_SYNC_THREAD;

      // recover interpolated data
      if (i < num_equation) {
        u1[i] = d_interpolU[i + q * num_equation + n * maxIntPoints * num_equation];
        for (int d = 0; d < dim; d++)
          gradUpi[i + d * num_equation] =
              d_interpGrads[i + d * num_equation + q * dim * num_equation + n * maxIntPoints * dim * num_equation];
      }
      MFEM_SYNC_THREAD;

      // compute mirror state
      switch (type) {
        case WallType::INV:
          if (i < num_equation) computeInvWallState(i, &u1[0], &u2[0], &nor[0], dim, num_equation);
          break;
        case WallType::VISC_ISOTH:
          if (i < num_equation)
            computeIsothermalState(i, &u1[0], &u2[0], &nor[0], wallTemp, gamma, Rg, dim, num_equation);
          break;
        case WallType::VISC_ADIAB:
          break;
      }
      MFEM_SYNC_THREAD;

      // compute flux
      RiemannSolver::riemannLF_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, eqSystem, num_equation, i,
                                   maxDofs);

      // compute viscous flux
      Fluxes::viscousFlux_gpu(&vF1[0], &u1[0], &gradUpi[0], eqSystem, gamma, Rg, viscMult, bulkViscMult, Pr, Sc, i,
                              maxDofs, dim, num_equation);
      Fluxes::viscousFlux_gpu(&vF2[0], &u2[0], &gradUpi[0], eqSystem, gamma, Rg, viscMult, bulkViscMult, Pr, Sc, i,
                              maxDofs, dim, num_equation);
      MFEM_SYNC_THREAD;

      // add visc flux contribution
      for (int eq = i; eq < num_equation; eq += maxDofs)
        for (int d = 0; d < dim; d++)
          Rflux[eq] -= 0.5 * (vF2[eq + d * num_equation] + vF1[eq + d * num_equation]) * nor[d];
      MFEM_SYNC_THREAD;

      // sum contributions to integral
      if (i < elDof) {
        for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] -= Rflux[eq] * shape[i] * weight;
      }
      MFEM_SYNC_THREAD;
    }
      }

      // add to global data
      if (i < elDof) {
    for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
      }
}
});
#endif
}

void WallBC::interpWalls_gpu(const WallType type, const double &wallTemp, mfem::Vector &interpolated_Ubdr_,
                             Vector &interpolatedGradUpbdr_, const mfem::Vector &x, const Array<int> &nodesIDs,
                             const Array<int> &posDofIds, mfem::ParGridFunction *Up, mfem::ParGridFunction *gradUp,
                             mfem::Vector &shapesBC, mfem::Vector &normalsWBC, Array<int> &intPointsElIDBC,
                             Array<int> &wallElems, Array<int> &listElems, const int &maxIntPoints, const int &maxDofs,
                             const int &dim, const int &num_equation) {
#ifdef _GPU_
  double *d_interpolU = interpolated_Ubdr_.Write();
  double *d_interpGrads = interpolatedGradUpbdr_.Write();

  const double *d_U = x.Read();
  const double *d_gradUp = gradUp->Read();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();

  const int totDofs = x.Size() / num_equation;
  const int numBdrElem = listElems.Size();

  // clang-format on
  MFEM_FORALL_2D(el, wallElems.Size() / 7, maxDofs, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, maxDofs) {
      MFEM_SHARED double Ui[216], gradUpi[216*3];;
      MFEM_SHARED double shape[216];
      MFEM_SHARED double u1, gUp[3];
      MFEM_SHARED double weight;

      const int numFaces = d_wallElems[0 + el * 7];
      bool changeElem = true;
      int elOffset;
      int elDof;
      int elID = -666;
      int indexi;

      for (int f = 0; f < numFaces; f++) {
    const int n = d_wallElems[1 + f + el * 7];
    const int el = d_listElems[n];
    const int Q = d_intPointsElIDBC[2 * el];
    int newID = d_intPointsElIDBC[2 * el + 1];
    if (newID != elID) {
      changeElem = true;
      elID = newID;
    }

    if (changeElem) {  // NOTE: in new implementation this doesn't save much
      elOffset = d_posDofIds[2 * elID];
      elDof = d_posDofIds[2 * elID + 1];
      changeElem = false;
    }

    for (int eq = 0; eq < num_equation; eq++) {
      // load data
      if (i < elDof) {
        indexi = d_nodesIDs[elOffset + i];
        Ui[i] = d_U[indexi + eq * totDofs];
        for (int d = 0; d < dim; d++)
          gradUpi[i + d * elDof] = d_gradUp[indexi + eq * totDofs + d * num_equation * totDofs];
      }

      for (int q = 0; q < Q; q++) {
        if (i < elDof) shape[i] = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
        MFEM_SYNC_THREAD;

        // interpolation
        // NOTE: make parallel
        if (i == 0) {
          u1 = 0.;
          for (int j = 0; j < elDof; j++) u1 += shape[j] * Ui[j];
        }
        if (i < dim) {
          gUp[i] = 0.;
          for (int j = 0; j < elDof; j++) gUp[i] += gradUpi[j + i * elDof] * shape[j];
        }
        MFEM_SYNC_THREAD;

        // save to global
        if (i == 0) d_interpolU[eq + q * num_equation + n * maxIntPoints * num_equation] = u1;
        if (i < dim)
          d_interpGrads[eq + i * num_equation + q * dim * num_equation + n * maxIntPoints * dim * num_equation] =
              gUp[i];
        MFEM_SYNC_THREAD;
      }  // end loop integration points
    }    // end loop equations
      }  // end loop faces
}
});
#endif
}

// clang-format on
