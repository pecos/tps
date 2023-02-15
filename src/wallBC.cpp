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
               const Array<int> &_intPointsElIDBC, const int &_maxIntPoints, bool axisym)
    : BoundaryCondition(_rsolver, _mixture, _eqSystem, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber, 1,
                        axisym),  // so far walls do not require ref. length. Left at 1
      wallType_(_bcType),
      wallData_(_inputData),
      d_mixture_(d_mixture),
      fluxClass(_fluxClass),
      intPointsElIDBC(_intPointsElIDBC),
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
  // auto dwallElems = wallElems.ReadWrite();
  wallElems.ReadWrite();
}

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &delState, 
			    double radius, Vector transip, double delta, double time, TransportProperties *_transport,
			    //			    Vector &bdrFlux) {
			    int ip, Vector &bdrFlux) {
  
  switch (wallType_) {
    case INV:
      computeINVwallFlux(normal, stateIn, gradState, radius, transip, delta, bdrFlux);
      break;
    case SLIP:
      computeSlipWallFlux(normal, stateIn, gradState, radius, transip, delta, bdrFlux);
      break;      
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal, stateIn, gradState, radius, transip, delta, bdrFlux);
      break;
    case VISC_ISOTH:
      computeIsothermalWallFlux(normal, stateIn, gradState, radius, transip, delta, bdrFlux);
      break;
    case VISC_GNRL:
      computeGeneralWallFlux(normal, stateIn, gradState, radius, transip, delta, bdrFlux);
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


/**
Old inviscid boundary condition.  There is a bug in mirror-state calculation (will break for some orientations 
of flow and wall).  Also, viscous terms are included.
*/
void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector transip, double delta, 
                                Vector &bdrFlux) {

  // stateIn is conserved
  Vector vel(nvel_);
  for (int d = 0; d < nvel_; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  // this is outward facing
  double norm = 0.;
  Vector unitN(dim_);  
  for (int d = 0; d < dim_; d++) norm += normal[d] * normal[d];
  norm = sqrt(norm);
  for (int d = 0; d < dim_; d++) unitN[d] = normal[d] / norm;

  /// velocity in normal direction
  double vn = 0;
  for (int d = 0; d < dim_; d++) vn += vel[d] * unitN[d];

  Vector stateMirror(num_equation_);
  stateMirror = stateIn;

  // THIS IS NOT CORRECT FOR ALL SURFACES, will erroneously ACCELERATE fluids in some arrangements
  // only momentum needs to be changed (how does energy not need to change???): rho*(u_i - 2*u_n*e_i) reflection about u_n
  stateMirror[1] = stateIn[0] * (vel[0] - 2.0*vn*unitN[0]);
  stateMirror[2] = stateIn[0] * (vel[1] - 2.0*vn*unitN[1]);
  if (dim_ == 3) stateMirror[3] = stateIn[0] * (vel[2] - 2.0*vn*unitN[2]);
  if ((nvel_ == 3) && (dim_ == 2)) stateMirror[3] = stateIn[0] * vel[2];

  rsolver->Eval(stateIn, stateMirror, normal, bdrFlux);
  
  // this case is supposed to be inviscid...
  Vector wallViscF(num_equation_);
  DenseMatrix viscF(num_equation_, dim_);

  if (axisymmetric_) {
    
    // here we have hijacked the inviscid wall condition to implement
    // the axis... but... should implement this separately

    // incoming visc flux
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, transip, delta, viscF);

    // modify gradients so that wall is adibatic
    Vector unitNorm = normal;
    double normN = 0.;
    for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);

    for (int d = 0; d < dim_; d++) bcFlux_.normal[d] = unitNorm[d];
    fluxClass->ComputeBdrViscousFluxes(stateMirror, gradState, radius, transip, delta, bcFlux_, wallViscF);
    wallViscF *= sqrt(normN);  // in case normal is not a unit vector..
    
  } else {
    
    DenseMatrix viscFw(num_equation_, dim_);

    // evaluate viscous fluxes at the wall
    fluxClass->ComputeViscousFluxes(stateMirror, gradState, radius, transip, delta, viscFw);
    viscFw.Mult(normal, wallViscF);

    // evaluate internal viscous fluxes
    fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, transip, delta, viscF);
    
  }

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
void WallBC::computeSlipWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
				 Vector transip, double delta, Vector &bdrFlux) {

  Vector prim(num_equation_);
  mixture->GetPrimitivesFromConservatives(stateIn, prim);
  double wallT = prim[dim_ + 1];
  
  // stateIn is conserved
  Vector vel(nvel_);
  for (int d = 0; d < nvel_; d++) vel[d] = stateIn[1 + d] / stateIn[0];
  
  // outward-facing normal
  double sml = 1.0e-15;
  Vector unitNorm = normal;
  double mod = 0.;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  normN = sqrt(max(normN,sml));
  unitNorm *= 1. / normN;

  // build arbitrary tangents
  Vector tangent1 = unitNorm;
  Vector tangent2 = unitNorm;
  tangent1 = 0.;
  tangent2 = 0.;  

  // dominant direction
  int dir, next_dir, previous_dir;
  if(abs(unitNorm[0]) >= abs(unitNorm[1]) && abs(unitNorm[0]) >= abs(unitNorm[2])) dir = 0;
  if(abs(unitNorm[1]) >= abs(unitNorm[0]) && abs(unitNorm[1]) >= abs(unitNorm[2])) dir = 1;
  if(abs(unitNorm[2]) >= abs(unitNorm[0]) && abs(unitNorm[2]) >= abs(unitNorm[1])) dir = 2;
  next_dir = (dir+1) % (dim_);
  previous_dir = (dir+2) % (dim_);
  
  // tangent 1
  tangent1[next_dir] = +1.;
  tangent1[previous_dir] = -1.;
  tangent1[dir] = unitNorm[previous_dir]*tangent1[previous_dir] + unitNorm[next_dir]*tangent1[next_dir];
  tangent1[dir] *= -1. / unitNorm[dir] ;  
  mod = 0.;
  for (int d = 0; d < dim_; d++) mod += tangent1[d] * tangent1[d];
  tangent1 *= 1. / max(sqrt(mod),sml); 
    
  // tangent 2
  tangent2[0] = +(unitNorm[1]*tangent1[2] - unitNorm[2]*tangent1[1]);
  tangent2[1] = -(unitNorm[0]*tangent1[2] - unitNorm[2]*tangent1[0]);
  tangent2[2] = +(unitNorm[0]*tangent1[1] - unitNorm[1]*tangent1[0]);
  mod = 0.;
  for (int d = 0; d < dim_; d++) mod += tangent2[d] * tangent2[d];
  tangent2 *= 1. / max(sqrt(mod),sml); 

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
  //double ke_n = 0.5*nVel[0]*nVel[0];

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


void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector transip, double delta, 
                                      Vector &bdrFlux) {
  Vector wallState(num_equation_);
  mixture->computeStagnationState(stateIn, wallState);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
  if (eqSystem == NS_PASSIVE) bdrFlux[num_equation_ - 1] = 0.;

  // incoming visc flux
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, transip, delta, viscF);

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
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, radius, transip, delta, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
}


void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
				       Vector transip, double delta, Vector &bdrFlux) {

  Vector wallState(num_equation_);
  //mixture->computeStagnantStateWithTemp(stateIn, wallTemp_, wallState);
  // TODO(kevin): set stangant state with two separate temperature.

  // reflected state (rho*u only) to prevent penetration
  for (int eq = 0; eq <= num_equation_; eq++) wallState[eq] = stateIn[eq];  
  for (int eq = 1; eq <= dim_; eq++) wallState[eq] = -stateIn[eq];
  
  if (eqSystem == NS_PASSIVE) wallState[num_equation_ - 1] = stateIn[num_equation_ - 1];

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // unit normal vector
  Vector unitNorm = normal;
  double normN = 0.;
  for (int d = 0; d < dim_; d++) normN += normal[d] * normal[d];
  unitNorm *= 1. / sqrt(normN);
  for (int d = 0; d < dim_; d++) bcFlux_.normal[d] = unitNorm[d];

  // evaluate viscous fluxes at the wall
  Vector wallViscF(num_equation_);
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, radius, transip, delta, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, transip, delta, viscF);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation_; eq++) {
    bdrFlux[eq] -= 0.5 * wallViscF(eq);
    for (int d = 0; d < dim_; d++) bdrFlux[eq] -= 0.5 * viscF(eq, d) * normal[d];
  }
  
}


void WallBC::computeGeneralWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector transip, double delta, 
                                    Vector &bdrFlux) {
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
  fluxClass->ComputeBdrViscousFluxes(wallState, gradState, radius, transip, delta, bcFlux_, wallViscF);
  wallViscF *= sqrt(normN);  // in case normal is not a unit vector..

  // evaluate internal viscous fluxes
  DenseMatrix viscF(num_equation_, dim_);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, radius, transip, delta, viscF);

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
  const BoundaryPrimitiveData d_bcState = bcState_;
  const BoundaryViscousFluxData d_bcFlux = bcFlux_;

  const bool computeSheath = (wallData_.elecThermalCond == SHTH);

  const RiemannSolver *d_rsolver = rsolver;
  GasMixture *d_mix = d_mixture_;
  Fluxes *d_fluxclass = fluxClass;

  // clang-format on
  // el_wall is index within wall boundary elements?
  MFEM_FORALL_2D(el_wall, wallElems.Size() / 7, maxIntPoints, 1, 1, {
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS], nor[gpudata::MAXDIM], Rflux[gpudata::MAXEQUATIONS];
    double vF1[gpudata::MAXEQUATIONS * gpudata::MAXDIM],
#if defined(_CUDA_)
        vF2[gpudata::MAXEQUATIONS];
#elif defined(_HIP_)
           vF2[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
#endif
    double gradUp1[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double shape[gpudata::MAXDOFS];
    int index_i[gpudata::MAXDOFS];

    BoundaryPrimitiveData bcState;
    BoundaryViscousFluxData bcFlux;
    double unitNorm[gpudata::MAXDIM], normN;

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
        normN = 0.0;
        for (int d = 0; d < dim; d++) {
          nor[d] = d_normW[d + q * (dim + 1) + el_bdry * maxIntPoints * (dim + 1)];
          normN += nor[d] * nor[d];
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

        // fill out the boundary data.
        bcState = d_bcState;
        bcFlux = d_bcFlux;
        for (int d = 0; d < dim; d++) {
          bcFlux.normal[d] = nor[d] / sqrt(normN);
        }

// only implemented general viscous wall flux, as it can supersede all the other viscous wall types.
// TODO(kevin): implement radius.
#if defined(_CUDA_)
        if (computeSheath) d_mix->computeSheathBdrFlux(u2, bcFlux);

        if (type == WallType::INV) {
          // compute mirror state
          computeInvWallState_gpu_serial(&u1[0], &u2[0], &nor[0], dim, num_equation);
        } else {
          d_mix->modifyStateFromPrimitive(u1, bcState, u2);
        }
        d_rsolver->Eval_LF(u1, u2, nor, Rflux);
        d_fluxclass->ComputeViscousFluxes(u1, gradUp1, 0.0, vF1);
        d_fluxclass->ComputeBdrViscousFluxes(u2, gradUp1, 0.0, bcFlux, vF2);
        for (int eq = 0; eq < num_equation; eq++) vF2[eq] *= sqrt(normN);

        // add visc flux contribution
        for (int eq = 0; eq < num_equation; eq++) {
          Rflux[eq] -= 0.5 * vF2[eq];
          for (int d = 0; d < dim; d++) Rflux[eq] -= 0.5 * vF1[eq + d * num_equation] * nor[d];
        }
#elif defined(_HIP_)
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
#endif

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
