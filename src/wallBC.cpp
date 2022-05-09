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
               const Array<int> &_intPointsElIDBC, const int &_maxIntPoints, bool axisym)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber, 1,
                        axisym),  // so far walls do not require ref. length. Left at 1
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

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux) {
  switch (wallType) {
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
  interpWalls_gpu(wallType, wallTemp, interpolated_Ubdr_, interpolatedGradUpbdr_, x, nodesIDs, posDofIds, Up, gradUp,
                  shapesBC, normalsWBC, intPointsElIDBC, wallElems, listElems, maxIntPoints, maxDofs, dim,
                  num_equation);

  integrateWalls_gpu(wallType, wallTemp,
                     y,  // output
                     x, interpolated_Ubdr_, interpolatedGradUpbdr_, nodesIDs, posDofIds, Up, gradUp, shapesBC,
                     normalsWBC, intPointsElIDBC, wallElems, listElems, eqSystem, maxIntPoints, maxDofs, dim,
                     num_equation, mixture);
}

void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                Vector &bdrFlux) {
  Vector vel(nvel);
  for (int d = 0; d < nvel; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  double norm = 0.;
  for (int d = 0; d < dim; d++) norm += normal[d] * normal[d];
  norm = sqrt(norm);

  Vector unitN(dim);
  for (int d = 0; d < dim; d++) unitN[d] = normal[d] / norm;

  double vn = 0;
  for (int d = 0; d < dim; d++) vn += vel[d] * unitN[d];

  Vector stateMirror(num_equation);
  stateMirror = stateIn;

  // only momentum needs to be changed
  stateMirror[1] = stateIn[0] * (vel[0] - 2. * vn * unitN[0]);
  stateMirror[2] = stateIn[0] * (vel[1] - 2. * vn * unitN[1]);
  if (dim == 3) stateMirror[3] = stateIn[0] * (vel[2] - 2. * vn * unitN[2]);
  if ((nvel == 3) && (dim == 2)) stateMirror[3] = stateIn[0] * vel[2];

  rsolver->Eval(stateIn, stateMirror, normal, bdrFlux);

  DenseMatrix viscFw(num_equation, dim);
  DenseMatrix viscF(num_equation, dim);

  if (axisymmetric_) {
    // here we have hijacked the inviscid wall condition to implement
    // the axis... but... should implement this separately

    // incoming visc flux
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

    // modify gradients so that wall is adibatic
    Vector unitNorm = normal;
    {
      double normN = 0.;
      for (int d = 0; d < dim; d++) normN += normal[d] * normal[d];
      unitNorm *= 1. / sqrt(normN);
    }

    fluxClass->ComputeViscousFluxes(stateMirror, gradState, radius, viscFw);

    // zero out normal heat flux.
    // NOTE: instead of zeroing the gradient, zeroed the resultant flux.
    double normalHeatFlux = 0.0;
    for (int d = 0; d < dim; d++) normalHeatFlux += unitNorm(d) * viscFw(1 + dim, d);
    for (int d = 0; d < dim; d++) viscFw(1 + dim, d) -= unitNorm(d) * normalHeatFlux;

    // zero out species normal diffusion fluxes.
    const int numActiveSpecies = mixture->GetNumActiveSpecies();
    for (int eq = nvel + 2; eq < nvel + 2 + numActiveSpecies; eq++) {
      double normalDiffusionFlux = 0.0;
      for (int d = 0; d < dim; d++) normalDiffusionFlux += unitNorm(d) * viscFw(eq, d);
      for (int d = 0; d < dim; d++) viscFw(eq, d) -= unitNorm(d) * normalDiffusionFlux;
    }

    // adiabatic wall must also have zero heat flux from electron.
    if (mixture->IsTwoTemperature()) {
      double normalElectronHeatFlux = 0.0;
      for (int d = 0; d < dim; d++) normalElectronHeatFlux += unitNorm(d) * viscFw(num_equation - 1, d);
      for (int d = 0; d < dim; d++) viscFw(num_equation - 1, d) -= unitNorm(d) * normalElectronHeatFlux;
    }
  } else {
    // evaluate viscous fluxes at the wall
    fluxClass->ComputeViscousFluxes(stateMirror, gradState, radius, viscFw);

    // evaluate internal viscous fluxes
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);
  }

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                      Vector &bdrFlux) {
  Vector wallState(num_equation);
  mixture->computeStagnationState(stateIn, wallState);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation - 1] = 0.;

  // incoming visc flux
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  {
    double normN = 0.;
    for (int d = 0; d < dim; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);
  }

  // modify gradient temperature so dT/dn=0 at the wall
  // double normGradT = 0.;
  // for (int d = 0; d < dim; d++) normGradT += unitNorm(d) * gradState(1 + dim, d);
  // for (int d = 0; d < dim; d++) gradState(1 + dim, d) -= normGradT * unitNorm(d);

  if (eqSystem == NS_PASSIVE) {
    for (int d = 0; d < dim; d++) gradState(num_equation - 1, d) = 0.;
  }

  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(wallState, gradState, radius, viscFw);

  // zero out normal heat flux.
  // NOTE: instead of zeroing the gradient, zeroed the resultant flux. Must be equivalent.
  double normalHeatFlux = 0.0;
  for (int d = 0; d < dim; d++) normalHeatFlux += unitNorm(d) * viscFw(1 + dim, d);
  for (int d = 0; d < dim; d++) viscFw(1 + dim, d) -= unitNorm(d) * normalHeatFlux;

  // zero out species normal diffusion fluxes.
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  for (int eq = nvel + 2; eq < nvel + 2 + numActiveSpecies; eq++) {
    double normalDiffusionFlux = 0.0;
    for (int d = 0; d < dim; d++) normalDiffusionFlux += unitNorm(d) * viscFw(eq, d);
    for (int d = 0; d < dim; d++) viscFw(eq, d) -= unitNorm(d) * normalDiffusionFlux;
  }

  // adiabatic wall must also have zero heat flux from electron.
  if (mixture->IsTwoTemperature()) {
    double normalElectronHeatFlux = 0.0;
    for (int d = 0; d < dim; d++) normalElectronHeatFlux += unitNorm(d) * viscFw(num_equation - 1, d);
    for (int d = 0; d < dim; d++) viscFw(num_equation - 1, d) -= unitNorm(d) * normalElectronHeatFlux;
  }

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                       Vector &bdrFlux) {
  Vector wallState(num_equation);
  mixture->computeStagnantStateWithTemp(stateIn, wallTemp, wallState);
  // TODO(kevin): set stangant state with two separate temperature.

  if (eqSystem == NS_PASSIVE) wallState[num_equation - 1] = stateIn[num_equation - 1];

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // evaluate viscous fluxes at the wall
  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(wallState, gradState, radius, viscFw);

  // unit normal vector
  Vector unitNorm = normal;
  {
    double normN = 0.;
    for (int d = 0; d < dim; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);
  }

  // zero out species normal diffusion fluxes.
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  for (int eq = nvel + 2; eq < nvel + 2 + numActiveSpecies; eq++) {
    double normalDiffusionFlux = 0.0;
    for (int d = 0; d < dim; d++) normalDiffusionFlux += unitNorm(d) * viscFw(eq, d);
    for (int d = 0; d < dim; d++) viscFw(eq, d) -= unitNorm(d) * normalDiffusionFlux;
  }

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, viscF);

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

  const WorkingFluid fluid = mixture->GetWorkingFluid();

  // clang-format on
  MFEM_FORALL(el_wall, wallElems.Size() / 7,
  {
    double Fcontrib[216 * 20];
    double shape[216];
    double Rflux[20], u1[20], u2[20], nor[3], gradUpi[20 * 3];
    double vF1[20 * 3], vF2[20 * 3];
    double weight;

    const int numFaces = d_wallElems[0 + el_wall * 7];
    bool elemDataRecovered = false;
    int elOffset = 0;
    int elDof = 0;
    int el = 0;

    for (int f = 0; f < numFaces; f++) {
      const int n = d_wallElems[1 + f + el_wall * 7];

      const int el_bdry = d_listElems[n];
      const int Q = d_intPointsElIDBC[2 * el_bdry];

      if (!elemDataRecovered) {
        el = d_intPointsElIDBC[2 * el_bdry + 1];

        elOffset = d_posDofIds[2 * el];
        elDof = d_posDofIds[2 * el + 1];

        for (int i = 0; i < elDof; i++) {
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * elDof] = 0.;
          }
        }
        elemDataRecovered = true;
      }

      for (int q = 0; q < Q; q++) {  // loop over int. points
        for (int i = 0; i < elDof; i++) shape[i] = d_shapesBC[i + q * maxDofs + el_bdry * maxIntPoints * maxDofs];
        for (int d = 0; d < dim; d++) nor[d] = d_normW[d + q * (dim + 1) + el_bdry * maxIntPoints * (dim + 1)];
        weight = d_normW[dim + q * (dim + 1) + el_bdry * maxIntPoints * (dim + 1)];

        for (int eq = 0; eq < num_equation; eq++) {// recover interpolated data
          u1[eq] = d_interpolU[eq + q * num_equation + n * maxIntPoints * num_equation];
          for (int d = 0; d < dim; d++)
            gradUpi[eq + d * num_equation] =
              d_interpGrads[eq + d * num_equation + q * dim * num_equation + n * maxIntPoints * dim * num_equation];
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
        Fluxes::viscousFlux_serial_gpu(&vF1[0], &u1[0], &gradUpi[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);
        Fluxes::viscousFlux_serial_gpu(&vF2[0], &u2[0], &gradUpi[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);

        // add visc flux contribution
        for (int eq = 0; eq < num_equation; eq++ )
          for (int d = 0; d < dim; d++)
            Rflux[eq] -= 0.5 * (vF2[eq + d * num_equation] + vF1[eq + d * num_equation]) * nor[d];

        // sum contributions to integral
        for (int i = 0; i < elDof; i++) {
          for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] -= Rflux[eq] * shape[i] * weight;
        }
      }
    }

    // add to global data
    for (int i = 0; i < elDof; i++) {
      const int indexi = d_nodesIDs[elOffset + i];
      for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
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
  MFEM_FORALL(el_wall, wallElems.Size() / 7, // el_wall is index within wall boundary elements?
  {
    double Ui[216], gradUpi[216*3];
    double shape[216];

    const int numFaces = d_wallElems[0 + el_wall * 7];

    for (int f = 0; f < numFaces; f++) {

      const int n = d_wallElems[1 + f + el_wall * 7];
      const int el_bdry = d_listElems[n];  // element number within all boundary elements?
      const int Q = d_intPointsElIDBC[2 * el_bdry];
      const int el = d_intPointsElIDBC[2 * el_bdry + 1];  // global element number (on this mpi rank) ?

      const int elOffset = d_posDofIds[2 * el];
      const int elDof = d_posDofIds[2 * el + 1];

      for (int eq = 0; eq < num_equation; eq++) {
        for (int i = 0; i < elDof; i++) {
          // load data
          const int indexi = d_nodesIDs[elOffset + i];
          Ui[i] = d_U[indexi + eq * totDofs];
          for (int d = 0; d < dim; d++)
            gradUpi[i + d * elDof] = d_gradUp[indexi + eq * totDofs + d * num_equation * totDofs];

        }

        for (int q = 0; q < Q; q++) {
          for (int j = 0; j < elDof; j++) shape[j] = d_shapesBC[j + q * maxDofs + el_bdry * maxIntPoints * maxDofs];

          double u1 = 0.;
          for (int j = 0; j < elDof; j++) u1 += shape[j] * Ui[j];

          double gUp[3];
          for (int d = 0; d < dim; d++) {
            gUp[d] = 0.;
            for (int j = 0; j < elDof; j++) gUp[d] += gradUpi[j + d * elDof] * shape[j];
          }

          // save to global
          d_interpolU[eq + q * num_equation + n * maxIntPoints * num_equation] = u1;

          for (int d = 0; d < dim; d++) {
            d_interpGrads[eq + d * num_equation + q * dim * num_equation + n * maxIntPoints * dim * num_equation] =
              gUp[d];
          }
        }  // end quadrature point loop
      }  // end equation loop
    }  // end face loop
  });  // end element loop
#endif
}

// clang-format on
