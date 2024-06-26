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

#include "wallBC.hpp"

#include "riemann_solver.hpp"

WallBC::WallBC(RiemannSolver *_rsolver, GasMixture *_mixture, GasMixture *d_mixture, Equations _eqSystem,
               Fluxes *_fluxClass, ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt,
               const int _dim, const int _num_equation, int _patchNumber, WallType _bcType, const WallData _inputData,
               const boundaryFaceIntegrationData &boundary_face_data, const int &_maxIntPoints, bool axisym,
               bool useBCinGrad)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber, 1,
                        axisym),  // so far walls do not require ref. length. Left at 1
      wallType_(_bcType),
      wallData_(_inputData),
      useBCinGrad_(useBCinGrad),
      d_mixture_(d_mixture),
      fluxClass(_fluxClass),
      boundary_face_data_(boundary_face_data),
      maxIntPoints_(_maxIntPoints) {
  // Initialize bc state.
  // bcState_.prim.SetSize(num_equation_);
  // bcState_.primIdxs.SetSize(num_equation_);
  for (int eq = 0; eq < num_equation_; eq++) bcState_.prim[eq] = 0.0;
  for (int eq = 0; eq < num_equation_; eq++) bcState_.primIdxs[eq] = false;

  // Initialize bc flux.
  const int numSpecies = mixture->GetNumSpecies();
  const int primFluxSize = (mixture->IsTwoTemperature()) ? numSpecies + nvel_ + 2 : numSpecies + nvel_ + 1;
  // bcFlux_.primFlux.SetSize(primFluxSize);
  // bcFlux_.primFluxIdxs.SetSize(primFluxSize);
  for (int i = 0; i < primFluxSize; i++) bcFlux_.primFlux[i] = 0.0;
  for (int i = 0; i < primFluxSize; i++) bcFlux_.primFluxIdxs[i] = false;

  switch (wallType_) {
    case INV: {
      // NOTE(kevin): for axisymmetric case, no prescription on viscous stress. Is this handled by mirror state?
      for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;
      if (axisymmetric_) {
        bcFlux_.primFluxIdxs[numSpecies + nvel_] = true;
        if (mixture->IsTwoTemperature()) {
          bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;
        }
      }
      // NOTE(kevin): for INV, we do not use bcState_ at all.
    } break;
    case SLIP: {
      for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;
      if (axisymmetric_) {
        bcFlux_.primFluxIdxs[numSpecies + nvel_] = true;
        if (mixture->IsTwoTemperature()) {
          bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;
        }
      }
    } break;
    case VISC_ADIAB: {
      // no diffusion and heat flux.
      for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;
      bcFlux_.primFluxIdxs[numSpecies + nvel_] = true;
      if (mixture->IsTwoTemperature()) bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;

      // NOTE(kevin): for VISC_ADIAB, we do not use bcState_ at this point. (but could use)
      // no slip condition.
      for (int d = 0; d < nvel_; d++) bcState_.primIdxs[d + 1] = true;
    } break;
    case VISC_ISOTH: {
      // no diffusion flux.
      for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;

      // NOTE(kevin): for VISC_ISOTH, we do not use bcState_ at this point. (but could use)
      wallTemp_ = _inputData.Th;
      // no slip condition.
      for (int d = 0; d < nvel_; d++) bcState_.primIdxs[d + 1] = true;
      bcState_.prim[nvel_ + 1] = _inputData.Th;
      bcState_.primIdxs[nvel_ + 1] = true;
      if (mixture->IsTwoTemperature()) {
        // NOTE(kevin): for VISC_ISOTH, _inputData.Th == _inputData.Te
        bcState_.prim[num_equation_ - 1] = _inputData.Te;
        bcState_.primIdxs[num_equation_ - 1] = true;
      }
    } break;
    case VISC_GNRL: {
      // No slip boundary condition.
      for (int d = 0; d < nvel_; d++) bcState_.primIdxs[d + 1] = true;
      // Zero or prescribed diffusion flux.
      for (int i = 0; i < numSpecies; i++) bcFlux_.primFluxIdxs[i] = true;
      // Heavy-species isothermal condition.
      switch (wallData_.hvyThermalCond) {
        case ISOTH: {
          bcState_.prim[nvel_ + 1] = wallData_.Th;
          bcState_.primIdxs[nvel_ + 1] = true;
        } break;
        case ADIAB: {  // Heat flux is already set to zero.
          bcFlux_.primFluxIdxs[numSpecies + nvel_] = true;
        } break;
        default:
          mfem_error("Thermal condition not understood.");
          break;
      }
      // Electron isothermal condition.
      switch (wallData_.elecThermalCond) {
        case ISOTH: {
          bcState_.prim[num_equation_ - 1] = wallData_.Te;
          bcState_.primIdxs[num_equation_ - 1] = true;
        } break;
        case ADIAB: {  // Heat flux is already set to zero.
          bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;
        } break;
        case SHTH: {  // If single temperature, only determines diffusion flux (computed on the fly).
          if (mixture->IsTwoTemperature())  // If two-temperature, heat flux will be computed on the fly.
            bcFlux_.primFluxIdxs[numSpecies + nvel_ + 1] = true;
        } break;
        default:
          mfem_error("Electron thermal condition not understood.");
          break;
      }
    } break;
  }
}

WallBC::~WallBC() {}

void WallBC::initBCs() {
  if (!BCinit) {
#ifdef _GPU_
    buildWallElemsArray();
    face_flux_.UseDevice(true);
    face_flux_.SetSize(num_equation_ * maxIntPoints_ * listElems.Size());
    face_flux_ = 0.;
#endif

    BCinit = true;
  }
}

void WallBC::buildWallElemsArray() {
  auto hlistElems = listElems.HostRead();  // this is actually a list of faces
  auto h_face_el = boundary_face_data_.el.HostRead();

  std::vector<int> unicElems;
  unicElems.clear();

  for (int f = 0; f < listElems.Size(); f++) {
    int bcFace = hlistElems[f];
    int elID = h_face_el[bcFace];

    bool elInList = false;
    for (size_t i = 0; i < unicElems.size(); i++) {
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
      int elID2 = h_face_el[bcFace];
      if (elID2 == elID) {
        int nf = hwallElems[0 + 7 * el];
        if (nf == -1) nf = 0;
        nf++;
        hwallElems[nf + 7 * el] = f;
        hwallElems[0 + 7 * el] = nf;
      }
    }
  }
  // auto dwallElems = wallElems.ReadWrite();
  wallElems.ReadWrite();
}

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip, double delta,
                            double time, double distance, Vector &bdrFlux) {
  switch (wallType_) {
      /*
      case INV:
        computeINVwallFlux(normal, stateIn, gradState, radius, bdrFlux);
        break;
      case VISC_ADIAB:
        computeAdiabaticWallFlux(normal, stateIn, gradState, radius, delta, bdrFlux);
        break;
      case VISC_ISOTH:
        computeIsothermalWallFlux(normal, stateIn, gradState, radius, delta, bdrFlux);
        break;
      case VISC_GNRL:
        computeGeneralWallFlux(normal, stateIn, gradState, radius, delta, bdrFlux);
        break;
      */

    case INV:
      computeINVwallFlux(normal, stateIn, gradState, transip, delta, distance, bdrFlux);
      break;
    case SLIP:
      computeSlipWallFlux(normal, stateIn, gradState, transip, delta, bdrFlux);
      break;
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal, stateIn, gradState, transip, delta, bdrFlux);
      break;
    case VISC_ISOTH:
      computeIsothermalWallFlux(normal, stateIn, gradState, transip, delta, bdrFlux);
      break;
    case VISC_GNRL:
      computeGeneralWallFlux(normal, stateIn, gradState, transip, delta, bdrFlux);
      break;
  }
}

void WallBC::computeBdrPrimitiveStateForGradient(const Vector &primIn, Vector &primBC) const {
  primBC = primIn;

  switch (wallType_) {
    case INV:
      // TODO(trevilo): fix
      break;
    case VISC_ADIAB:
      // TODO(trevilo): fix
      break;
    case VISC_ISOTH:
      // no-slip
      for (int i = 0; i < nvel_; i++) {
        primBC[1 + i] = 0.0;
      }
      // isothermal
      primBC[nvel_ + 1] = wallTemp_;
      break;
    case VISC_GNRL:
      // TODO(trevilo): fix
      break;
    default:
      // If BC is unknown to this function, don't do anything
      break;
  }
}

void WallBC::integrationBC(Vector &y, const Vector &x, const elementIndexingData &elem_index_data, ParGridFunction *Up,
                           ParGridFunction *gradUp, const boundaryFaceIntegrationData &boundary_face_data,
                           const int &maxIntPoints, const int &maxDofs) {
  interpWalls_gpu(x, elem_index_data, Up, gradUp, boundary_face_data, maxDofs);

  integrateWalls_gpu(y,  // output
                     x, elem_index_data, boundary_face_data, maxDofs);
}

void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip, double delta,
                                double distance, Vector &bdrFlux) {
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

  DenseMatrix viscFw(num_equation_, dim_);

  // evaluate viscous fluxes at the wall
  fluxClass->ComputeViscousFluxes(stateMirror, gradState, transip, delta, distance, viscFw);
  viscFw.Mult(normal, wallViscF);

  // evaluate internal viscous fluxes
  fluxClass->ComputeViscousFluxes(stateIn, gradState, transip, delta, distance, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

/**
Inviscid slip boundary condition.  Finds interior velocity in wall-coordinates, flips normal
component (mirror state), transforms back to global, send to riemann
*/
void WallBC::computeSlipWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip, double delta,
                                 Vector &bdrFlux) {
  Vector prim(num_equation_);
  mixture->GetPrimitivesFromConservatives(stateIn, prim);

  // stateIn is conserved
  Vector vel(nvel_);
  for (int d = 0; d < nvel_; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  // outward-facing normal
  double sml = 1.0e-15;
  Vector unitNorm = normal;
  double mod = 0.;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  normN = sqrt(max(normN, sml));
  unitNorm *= 1. / normN;

  // build arbitrary tangents
  Vector tangent1 = unitNorm;
  Vector tangent2 = unitNorm;
  tangent1 = 0.;
  tangent2 = 0.;

  // dominant direction
  int dir, next_dir, previous_dir;
  if (dim_ == 3) {
    if (abs(unitNorm[0]) >= abs(unitNorm[1]) && abs(unitNorm[0]) >= abs(unitNorm[2])) dir = 0;
    if (abs(unitNorm[1]) >= abs(unitNorm[0]) && abs(unitNorm[1]) >= abs(unitNorm[2])) dir = 1;
    if (abs(unitNorm[2]) >= abs(unitNorm[0]) && abs(unitNorm[2]) >= abs(unitNorm[1])) dir = 2;
  } else if (dim_ == 2) {
    if (abs(unitNorm[0]) >= abs(unitNorm[1])) dir = 0;
    if (abs(unitNorm[1]) >= abs(unitNorm[0])) dir = 1;
  } else {
    assert(false);
  }
  next_dir = (dir + 1) % (dim_);
  previous_dir = (dir + 2) % (dim_);

  // tangent 1
  tangent1[next_dir] = +1.;
  tangent1[previous_dir] = -1.;
  tangent1[dir] = unitNorm[previous_dir] * tangent1[previous_dir] + unitNorm[next_dir] * tangent1[next_dir];
  tangent1[dir] *= -1. / unitNorm[dir];
  mod = 0.;
  for (int d = 0; d < dim_; d++) mod += tangent1[d] * tangent1[d];
  tangent1 *= 1. / max(sqrt(mod), sml);

  // tangent 2
  if (dim_ == 3) {
    tangent2[0] = +(unitNorm[1] * tangent1[2] - unitNorm[2] * tangent1[1]);
    tangent2[1] = -(unitNorm[0] * tangent1[2] - unitNorm[2] * tangent1[0]);
    tangent2[2] = +(unitNorm[0] * tangent1[1] - unitNorm[1] * tangent1[0]);
    mod = 0.;
    for (int d = 0; d < dim_; d++) mod += tangent2[d] * tangent2[d];
    tangent2 *= 1. / max(sqrt(mod), sml);
  } else if (dim_ == 2) {
    // tangent2 is not used in 2d
    tangent2[0] = 0.0;
    tangent2[1] = 0.0;
  } else {
    assert(false);
  }

  // velocity in wall coordinate system
  Vector nVel(dim_);
  nVel = 0.;

  // just re-use M
  DenseMatrix M(dim_, dim_);
  Vector momN(dim_), momX(dim_);
  for (int d = 0; d < dim_; d++) {
    M(0, d) = unitNorm[d];
    M(1, d) = tangent1[d];
    if (dim_ == 3) M(2, d) = tangent2[d];
  }
  for (int d = 0; d < dim_; d++) momN[d] = vel[d];
  M.Mult(momN, momX);
  for (int d = 0; d < dim_; d++) nVel[d] = momX[d];

  // energy of normal component: not necessary
  // double ke_n = 0.5*nVel[0]*nVel[0];

  // mirror normal component
  nVel[0] = -nVel[0];

  // transform back to global coords
  DenseMatrix invM(dim_, dim_);
  mfem::CalcInverse(M, invM);
  for (int d = 0; d < dim_; d++) momN[d] = nVel[d];
  invM.Mult(momN, momX);
  for (int d = 0; d < dim_; d++) nVel[d] = momX[d];

  // zero-penetration conserved state
  Vector state2(num_equation_);
  state2 = stateIn;
  state2[1] = stateIn[0] * nVel[0];
  state2[2] = stateIn[0] * nVel[1];
  if (dim_ == 3) state2[3] = stateIn[0] * nVel[2];

  // now send to Reimann solver
  rsolver->Eval(stateIn, state2, normal, bdrFlux);
}

void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip,
                                      double delta, Vector &bdrFlux) {
  Vector wallState(num_equation_);
  mixture->computeStagnationState(stateIn, wallState);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = 0.;

  // incoming visc flux
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, transip, delta, 0.0, viscF);

  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  for (int d = 0; d < dim_; d++) bcFlux_.normal[d] = unitNorm[d];

  if (eqSystem == NS_PASSIVE) {
    for (int d = 0; d < dim_; d++) gradState(num_equation_ - 1, d) = 0.;
  }

  // zero out species normal diffusion fluxes.
  const int numSpecies = mixture->GetNumSpecies();
  Vector normDiffVel(numSpecies);
  normDiffVel = 0.0;

  // evaluate viscous fluxes at the wall
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, transip, delta, 0.0, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip,
                                       double delta, Vector &bdrFlux) {
  Vector wallState(num_equation_);
  wallState = stateIn;

  if (useBCinGrad_) {
    for (int i = 0; i < nvel_; i++) {
      wallState[i + 1] *= -1.0;
    }
  } else {
    mixture->computeStagnantStateWithTemp(stateIn, wallTemp_, wallState);
  }

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // unit normal vector
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  for (int d = 0; d < dim_; d++) bcFlux_.normal[d] = unitNorm[d];

  // evaluate viscous fluxes at the wall
  wallState = stateIn;
  mixture->computeStagnantStateWithTemp(stateIn, wallTemp_, wallState);
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, transip, delta, 0.0, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, transip, delta, 0.0, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::computeGeneralWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip,
                                    double delta, Vector &bdrFlux) {
  Vector wallState(num_equation_);
  mixture->modifyStateFromPrimitive(stateIn, bcState_, wallState);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // unit normal vector
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  for (int d = 0; d < dim_; d++) bcFlux_.normal[d] = unitNorm[d];

  if (wallData_.elecThermalCond == SHTH) mixture->computeSheathBdrFlux(wallState, bcFlux_);

  // evaluate viscous fluxes at the wall
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, transip, delta, 0.0, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, transip, delta, 0.0, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}

void WallBC::integrateWalls_gpu(Vector &y, const Vector &x, const elementIndexingData &elem_index_data,
                                const boundaryFaceIntegrationData &boundary_face_data, const int &maxDofs) {
#ifdef _GPU_
  double *d_y = y.Write();
  //   const double *d_U = x.Read();
  const int *d_elem_dofs_list = elem_index_data.dofs_list.Read();
  const int *d_elem_dof_off = elem_index_data.dof_offset.Read();
  const int *d_elem_dof_num = elem_index_data.dof_number.Read();
  const double *d_face_shape = boundary_face_data.shape.Read();
  const double *d_weight = boundary_face_data.quad_weight.Read();
  const int *d_face_el = boundary_face_data.el.Read();
  const int *d_face_num_quad = boundary_face_data.num_quad.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();

  const int totDofs = x.Size() / num_equation_;
  // const int numBdrElem = listElems.Size();

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
    MFEM_SHARED double Fcontrib[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // MFEM_SHARED double Fcontrib[216 * 5];
    double Rflux[gpudata::MAXEQUATIONS];                                    // double Rflux[5];

    const int numFaces = d_wallElems[0 + el_wall * 7];
    bool elemDataRecovered = false;
    int elOffset = 0;
    int elDof = 0;
    int el = 0;

    for (int f = 0; f < numFaces; f++) {
      const int n = d_wallElems[1 + f + el_wall * 7];

      const int el_bdry = d_listElems[n];
      const int Q = d_face_num_quad[el_bdry];

      el = d_face_el[el_bdry];

      elOffset = d_elem_dof_off[el];
      elDof = d_elem_dof_num[el];

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
        const double weight = d_weight[el_bdry * maxIntPoints + q];

        for (int eq = 0; eq < num_equation; eq++)
          Rflux[eq] = weight * d_flux[eq + q * num_equation + n * maxIntPoints * num_equation];

        // sum contributions to integral
        // for (int i = 0; i < elDof; i++) {
        MFEM_FOREACH_THREAD(i, x, elDof) {
          const double shape = d_face_shape[i + q * maxDofs + el_bdry * maxIntPoints * maxDofs];
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
      const int indexi = d_elem_dofs_list[elOffset + i];
      for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
    }
    MFEM_SYNC_THREAD;
  });
#endif
}

void WallBC::interpWalls_gpu(const mfem::Vector &x, const elementIndexingData &elem_index_data,
                             mfem::ParGridFunction *Up, mfem::ParGridFunction *gradUp,
                             const boundaryFaceIntegrationData &boundary_face_data, const int &maxDofs) {
#ifdef _GPU_
  double *d_flux = face_flux_.Write();

  const double *d_U = x.Read();
  const double *d_gradUp = gradUp->Read();
  const int *d_elem_dofs_list = elem_index_data.dofs_list.Read();
  const int *d_elem_dof_off = elem_index_data.dof_offset.Read();
  const int *d_elem_dof_num = elem_index_data.dof_number.Read();
  const double *d_face_shape = boundary_face_data.shape.Read();
  const double *d_normal = boundary_face_data.normal.Read();
  const double *d_xyz = boundary_face_data.xyz.Read();
  const int *d_face_num_quad = boundary_face_data.num_quad.Read();
  const int *d_face_el = boundary_face_data.el.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();
  const double *d_dist = boundary_face_data.dist.Read();

  auto d_delta = boundary_face_data.delta_el1.Read();

  const int totDofs = x.Size() / num_equation_;
  // const int numBdrElem = listElems.Size();

  const WallType type = wallType_;

  const int dim = dim_;
  const int nvel = nvel_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const BoundaryPrimitiveData d_bcState = bcState_;
  const BoundaryViscousFluxData d_bcFlux = bcFlux_;

  const bool computeSheath = (wallData_.elecThermalCond == SHTH);

  const RiemannSolver *d_rsolver = rsolver;
  GasMixture *d_mix = d_mixture_;
  Fluxes *d_fluxclass = fluxClass;

  const bool useBCinGrad = useBCinGrad_;

  // clang-format on
  // el_wall is index within wall boundary elements?
  MFEM_FORALL_2D(el_wall, wallElems.Size() / 7, maxIntPoints, 1, 1, {
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS], nor[gpudata::MAXDIM], Rflux[gpudata::MAXEQUATIONS];
    double xyz[gpudata::MAXDIM];
    double vF1[gpudata::MAXEQUATIONS * gpudata::MAXDIM], vF2[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double gradUp1[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double shape[gpudata::MAXDOFS];
    int index_i[gpudata::MAXDOFS];

    BoundaryPrimitiveData bcState;
    BoundaryViscousFluxData bcFlux;
    // double unitNorm[gpudata::MAXDIM], normN;
    double normN;

    const int numFaces = d_wallElems[0 + el_wall * 7];

    for (int f = 0; f < numFaces; f++) {
      const int n = d_wallElems[1 + f + el_wall * 7];
      const int el_bdry = d_listElems[n];  // element number within all boundary elements?
      const int Q = d_face_num_quad[el_bdry];
      const int el = d_face_el[el_bdry];  // global element number (on this mpi rank) ?

      const int elOffset = d_elem_dof_off[el];
      const int elDof = d_elem_dof_num[el];

      for (int i = 0; i < elDof; i++) {
        index_i[i] = d_elem_dofs_list[elOffset + i];
      }

      const int numActiveSpecies = d_fluxclass->GetNumActiveSpecies();

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
          shape[j] = d_face_shape[j + q * maxDofs + el_bdry * maxIntPoints * maxDofs];
        }

        // extract normal vector at this quad point
        normN = 0.0;
        for (int d = 0; d < dim; d++) {
          nor[d] = d_normal[el_bdry * maxIntPoints * dim + q * dim + d];
          normN += nor[d] * nor[d];

          xyz[d] = d_xyz[el_bdry * maxIntPoints * dim + q * dim + d];
        }
        const double distance = d_dist[el_bdry * maxIntPoints + q];

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

        // fill out the boundary data.
        bcState = d_bcState;
        bcFlux = d_bcFlux;
        for (int d = 0; d < dim; d++) {
          bcFlux.normal[d] = nor[d] / sqrt(normN);
        }

        if (computeSheath) d_mix->computeSheathBdrFlux(u2, bcFlux);

        if (type == WallType::INV || type == WallType::SLIP) {
          // compute mirror state
          computeInvWallState_gpu_serial(&u1[0], &u2[0], &nor[0], dim, num_equation);
        } else {
          if (useBCinGrad) {
            for (int eq = 0; eq < num_equation; eq++) {
              u2[eq] = u1[eq];
            }
            for (int i = 0; i < nvel; i++) {
              u2[1 + i] *= -1.0;
            }
          } else {
            d_mix->modifyStateFromPrimitive(u1, bcState, u2);
          }
        }
        d_rsolver->Eval_LF(u1, u2, nor, Rflux);

        if (type != WallType::SLIP) {
          if (type == WallType::INV) {
            // evaluate viscous fluxes at the wall
            d_fluxclass->ComputeViscousFluxes(u2, gradUp1, xyz, d_delta[el_bdry], distance, vF2);
            d_fluxclass->ComputeViscousFluxes(u1, gradUp1, xyz, d_delta[el_bdry], distance, vF1);

            // add visc flux contribution
            for (int eq = 0; eq < num_equation; eq++) {
              for (int d = 0; d < dim; d++) {
                Rflux[eq] -= 0.5 * (vF1[eq + d * num_equation] + vF2[eq + d * num_equation]) * nor[d];
              }
            }
          } else {
            d_fluxclass->ComputeViscousFluxes(u1, gradUp1, xyz, d_delta[el_bdry], 0.0, vF1);

            for (int eq = 0; eq < num_equation; eq++) {
              u2[eq] = u1[eq];
            }
            d_mix->modifyStateFromPrimitive(u1, bcState, u2);

            d_fluxclass->ComputeBdrViscousFluxes(u2, gradUp1, xyz, d_delta[el_bdry], 0.0, bcFlux, vF2);
            for (int eq = 0; eq < num_equation; eq++) vF2[eq] *= sqrt(normN);

            // add visc flux contribution
            for (int eq = 0; eq < num_equation; eq++) {
              Rflux[eq] -= 0.5 * vF2[eq];
              for (int d = 0; d < dim; d++) Rflux[eq] -= 0.5 * vF1[eq + d * num_equation] * nor[d];
            }
          }
        }

        if (d_fluxclass->isAxisymmetric()) {
          const double radius = xyz[0];
          for (int eq = 0; eq < num_equation; eq++) {
            Rflux[eq] *= radius;
          }
        }

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
