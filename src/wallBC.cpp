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
               const int _num_equation, int _patchNumber, WallType _bcType, const wallData _inputData,
               const Array<int> &_intPointsElIDBC, const int &_maxIntPoints, bool axisym)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber, 1,
                        axisym),  // so far walls do not require ref. length. Left at 1
      wallType_(_bcType),
      wallData_(_inputData),
      fluxClass(_fluxClass),
      intPointsElIDBC(_intPointsElIDBC),
      maxIntPoints_(_maxIntPoints) {
  const int numSpecies = mixture->GetNumSpecies();
  const int primFluxSize = (mixture->IsTwoTemperature()) ? numSpecies + nvel_ + 2 : numSpecies + nvel_ + 1;
  bcFlux_.primFlux.SetSize(primFluxSize);
  bcFlux_.primFluxIdxs.SetSize(primFluxSize);
  bcFlux_.primFlux = 0.0;
  for (int i = 0; i < primFluxSize; i++) bcFlux_.primFluxIdxs[i] = false;
  for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;
  if ((wallType_ == VISC_ADIAB) || ((wallType_ == INV) && axisymmetric_)) {
    bcFlux_.primFlux(numSpecies + 1) = 0.0;
    bcFlux_.primFluxIdxs[numSpecies + nvel_] = true;
    if (mixture->IsTwoTemperature()) {
      bcFlux_.primFlux(numSpecies + 2) = 0.0;
      bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;
    }
  }
  if (wallType_ == VISC_ISOTH) {
    wallTemp_ = _inputData.Th;
  }
}

WallBC::~WallBC() {}

void WallBC::initBCs() {
  if (!BCinit) {
    buildWallElemsArray(intPointsElIDBC);

#ifdef _GPU_
    face_flux_.UseDevice(true);
    face_flux_.SetSize(num_equation_ * maxIntPoints_ * listElems.Size());
    face_flux_ = 0.;
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

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux) {
  switch (wallType_) {
    case INV:
      computeINVwallFlux(normal, stateIn, gradState, radius, bdrFlux);
      break;
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal, stateIn, gradState, radius, bdrFlux);
      break;
    case VISC_ISOTH:
      computeIsothermalWallFlux(normal, stateIn, gradState, radius, bdrFlux);
      break;
  }
}

void WallBC::integrationBC(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                           ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                           Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs) {
  interpWalls_gpu(x, nodesIDs, posDofIds, Up, gradUp, shapesBC, normalsWBC, intPointsElIDBC, maxDofs);

  integrateWalls_gpu(y,  // output
                     x, nodesIDs, posDofIds, shapesBC, normalsWBC, intPointsElIDBC, maxDofs);
}

void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                Vector &bdrFlux) {
  Vector vel(nvel_);
  for (int d = 0; d < nvel_; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  double norm = 0.;
  for (int d = 0; d < dim_; d++) norm += normal[d] * normal[d];
  norm = sqrt(norm);

  Vector unitN(dim_);
  for (int d = 0; d < dim_; d++) unitN[d] = normal[d] / norm;

  double vn = 0;
  for (int d = 0; d < dim_; d++) vn += vel[d] * unitN[d];

  Vector stateMirror(num_equation_);
  stateMirror = stateIn;

  // only momentum needs to be changed
  stateMirror[1] = stateIn[0] * (vel[0] - 2. * vn * unitN[0]);
  stateMirror[2] = stateIn[0] * (vel[1] - 2. * vn * unitN[1]);
  if (dim_ == 3) stateMirror[3] = stateIn[0] * (vel[2] - 2. * vn * unitN[2]);
  if ((nvel_ == 3) && (dim_ == 2)) stateMirror[3] = stateIn[0] * vel[2];

  rsolver->Eval(stateIn, stateMirror, normal, bdrFlux);

  Vector wallViscF(num_equation_);
  DenseMatrix viscF(num_equation_, dim_);

  if (axisymmetric_) {
    // here we have hijacked the inviscid wall condition to implement
    // the axis... but... should implement this separately

    // incoming visc flux
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

    // modify gradients so that wall is adibatic
    Vector unitNorm = normal;
    double normN = 0.;
    for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);

    bcFlux_.normal = unitNorm;
    fluxClass->ComputeBdrViscousFluxes(stateMirror, gradState, radius, bcFlux_, wallViscF);
    wallViscF *= sqrt(normN);  // in case normal is not a unit vector..
  } else {
    DenseMatrix viscFw(num_equation_, dim_);

    // evaluate viscous fluxes at the wall
    fluxClass->ComputeViscousFluxes(stateMirror, gradState, radius, viscFw);
    viscFw.Mult(normal, wallViscF);

    // evaluate internal viscous fluxes
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);
  }

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                      Vector &bdrFlux) {
  Vector wallState(num_equation_);
  mixture->computeStagnationState(stateIn, wallState);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = 0.;

  // incoming visc flux
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  bcFlux_.normal = unitNorm;

  if (eqSystem == NS_PASSIVE) {
    for (int d = 0; d < dim_; d++) gradState(num_equation_ - 1, d) = 0.;
  }

  // zero out species normal diffusion fluxes.
  const int numSpecies = mixture->GetNumSpecies();
  Vector normDiffVel(numSpecies);
  normDiffVel = 0.0;

  // evaluate viscous fluxes at the wall
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, radius, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                       Vector &bdrFlux) {
  Vector wallState(num_equation_);
  mixture->computeStagnantStateWithTemp(stateIn, wallTemp_, wallState);
  // TODO(kevin): set stangant state with two separate temperature.

  if (eqSystem == NS_PASSIVE) wallState[num_equation_ - 1] = stateIn[num_equation_ - 1];

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // unit normal vector
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  bcFlux_.normal = unitNorm;

  // evaluate viscous fluxes at the wall
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, radius, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::integrateWalls_gpu(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                                Vector &shapesBC, Vector &normalsWBC, Array<int> &intPointsElIDBC, const int &maxDofs) {
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

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const double Rg = mixture->GetGasConstant();
  const double gamma = mixture->GetSpecificHeatRatio();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;

  const double *d_flux = face_flux_.Read();

  // clang-format on
  // MFEM_FORALL(el_wall, wallElems.Size() / 7, {
  MFEM_FORALL_2D(el_wall, wallElems.Size() / 7, maxDofs, 1, 1, {
    // double Fcontrib[216 * 20];
    // double shape[216];
    // double Rflux[20], u1[20], u2[20], nor[3], gradUpi[20 * 3];
    // double vF1[20 * 3], vF2[20 * 3];
    MFEM_SHARED double Fcontrib[216 * 5];
    double Rflux[5];

    const int numFaces = d_wallElems[0 + el_wall * 7];
    bool elemDataRecovered = false;
    int elOffset = 0;
    int elDof = 0;
    int el = 0;

    for (int f = 0; f < numFaces; f++) {
      const int n = d_wallElems[1 + f + el_wall * 7];

      const int el_bdry = d_listElems[n];
      const int Q = d_intPointsElIDBC[2 * el_bdry];

      el = d_intPointsElIDBC[2 * el_bdry + 1];

      elOffset = d_posDofIds[2 * el];
      elDof = d_posDofIds[2 * el + 1];

      if (!elemDataRecovered) {
        MFEM_FOREACH_THREAD(i, x, elDof) {
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * elDof] = 0.;
          }
        }
        MFEM_SYNC_THREAD;
        elemDataRecovered = true;
      }

      for (int q = 0; q < Q; q++) {  // loop over int. points
        const double weight = d_normW[dim + q * (dim + 1) + el_bdry * maxIntPoints * (dim + 1)];

        for (int eq = 0; eq < num_equation; eq++)
          Rflux[eq] = weight * d_flux[eq + q * num_equation + n * maxIntPoints * num_equation];

        // sum contributions to integral
        // for (int i = 0; i < elDof; i++) {
        MFEM_FOREACH_THREAD(i, x, elDof) {
          const double shape = d_shapesBC[i + q * maxDofs + el_bdry * maxIntPoints * maxDofs];
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * elDof] -= Rflux[eq] * shape;
          }
        }
        MFEM_SYNC_THREAD;
      }
    }

    // add to global data
    // for (int i = 0; i < elDof; i++) {
    MFEM_FOREACH_THREAD(i, x, elDof) {
      const int indexi = d_nodesIDs[elOffset + i];
      for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
    }
    MFEM_SYNC_THREAD;
  });
#endif
}

void WallBC::interpWalls_gpu(const mfem::Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                             mfem::ParGridFunction *Up, mfem::ParGridFunction *gradUp, mfem::Vector &shapesBC,
                             mfem::Vector &normalsWBC, Array<int> &intPointsElIDBC, const int &maxDofs) {
#ifdef _GPU_
  double *d_flux = face_flux_.Write();

  const double *d_U = x.Read();
  const double *d_gradUp = gradUp->Read();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();

  const int totDofs = x.Size() / num_equation_;
  const int numBdrElem = listElems.Size();

  const double Rg = mixture->GetGasConstant();
  const double gamma = mixture->GetSpecificHeatRatio();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();

  const WorkingFluid fluid = mixture->GetWorkingFluid();
  const WallType type = wallType_;
  const double wallTemp = wallTemp_;

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;

  // clang-format on
  // el_wall is index within wall boundary elements?
  MFEM_FORALL_2D(el_wall, wallElems.Size() / 7, maxIntPoints, 1, 1, {
    double u1[5], u2[5], nor[3], Rflux[5], vF1[5 * 3], vF2[5 * 3];
    double gradUp1[5 * 3];
    double shape[216];
    int index_i[216];

    const int numFaces = d_wallElems[0 + el_wall * 7];

    for (int f = 0; f < numFaces; f++) {
      const int n = d_wallElems[1 + f + el_wall * 7];
      const int el_bdry = d_listElems[n];  // element number within all boundary elements?
      const int Q = d_intPointsElIDBC[2 * el_bdry];
      const int el = d_intPointsElIDBC[2 * el_bdry + 1];  // global element number (on this mpi rank) ?

      const int elOffset = d_posDofIds[2 * el];
      const int elDof = d_posDofIds[2 * el + 1];

      for (int i = 0; i < elDof; i++) {
        index_i[i] = d_nodesIDs[elOffset + i];
      }

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
          shape[j] = d_shapesBC[j + q * maxDofs + el_bdry * maxIntPoints * maxDofs];
        }

        // extract normal vector at this quad point
        for (int d = 0; d < dim; d++) {
          nor[d] = d_normW[d + q * (dim + 1) + el_bdry * maxIntPoints * (dim + 1)];
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

        // compute mirror state
        switch (type) {
          case WallType::INV:
            computeInvWallState_gpu_serial(&u1[0], &u2[0], &nor[0], dim, num_equation);
            break;
          case WallType::VISC_ISOTH:
            computeIsothermalState_gpu_serial(&u1[0], &u2[0], &nor[0], wallTemp, gamma, Rg, dim, num_equation, fluid);
            break;
          case WallType::VISC_ADIAB:
            break;
        }

        // evaluate flux
        RiemannSolver::riemannLF_serial_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
        Fluxes::viscousFlux_serial_gpu(&vF1[0], &u1[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);
        Fluxes::viscousFlux_serial_gpu(&vF2[0], &u2[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);

        // add visc flux contribution
        for (int eq = 0; eq < num_equation; eq++)
          for (int d = 0; d < dim; d++)
            Rflux[eq] -= 0.5 * (vF2[eq + d * num_equation] + vF1[eq + d * num_equation]) * nor[d];

        // store flux (TODO: change variable name)
        for (int eq = 0; eq < num_equation; eq++) {
          d_flux[eq + q * num_equation + n * maxIntPoints * num_equation] = Rflux[eq];
        }
      }  // end quadrature point loop
    }    // end face loop
  });    // end element loop
#endif
}

// clang-format on
