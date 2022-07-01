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

#include "equation_of_state.hpp"

// EquationOfState::EquationOfState() {}

GasMixture::GasMixture(RunConfiguration &_runfile, int _dim, int nvel) : GasMixture(_runfile.workFluid, _dim, nvel) {}

MFEM_HOST_DEVICE GasMixture::GasMixture(WorkingFluid f, int _dim, int nvel) {
  fluid = f;
  dim = _dim;
  nvel_ = nvel;
}

void GasMixture::UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up) {
  double *pGridFunc = press->HostWrite();
  const double *UpData = Up->HostRead();

  const int nnode = press->FESpace()->GetNDofs();

  for (int n = 0; n < nnode; n++) {
    Vector UpAtNode(num_equation);
    for (int eq = 0; eq < num_equation; eq++) {
      UpAtNode[eq] = UpData[n + eq * nnode];
    }
    pGridFunc[n] = ComputePressureFromPrimitives(UpAtNode);
  }
}

// total energy of stagnation state is essentially the same as
// the total energy of input state subtracted by its bulk kinetic energy.
// Do not necessarily need mixture-specific routines.
void GasMixture::computeStagnationState(const mfem::Vector &stateIn, mfem::Vector &stagnationState) {
  stagnationState.SetSize(num_equation);
  stagnationState = stateIn;

  // momentum = 0.;
  for (int d = 0; d < nvel_; d++) stagnationState(1 + d) = 0.;

  // compute total energy
  double kineticEnergy = 0.0;
  for (int d = 0; d < nvel_; d++) {
    kineticEnergy += 0.5 * stateIn(1 + d) * stateIn(1 + d) / stateIn(0);
  }
  stagnationState(1 + nvel_) = stateIn(1 + nvel_) - kineticEnergy;

  // NOTE: electron energy is purely internal energy, so no change.
}

void GasMixture::modifyStateFromPrimitive(const Vector &state, const BoundaryPrimitiveData &bcState,
                                          Vector &outputState) {
  outputState.SetSize(num_equation);
  // assert(bcState.prim.Size() == num_equation);

  Vector prim(num_equation);
  GetPrimitivesFromConservatives(state, prim);
  for (int i = 0; i < num_equation; i++) {
    if (bcState.primIdxs[i]) prim(i) = bcState.prim[i];
  }

  GetConservativesFromPrimitives(prim, outputState);
}

MFEM_HOST_DEVICE void GasMixture::modifyStateFromPrimitive(const double *state, const BoundaryPrimitiveData &bcState,
                                                           double *outputState) {
  double prim[gpudata::MAXEQUATIONS];
  GetPrimitivesFromConservatives(state, prim);
  for (int i = 0; i < num_equation; i++) {
    if (bcState.primIdxs[i]) prim[i] = bcState.prim[i];
  }

  GetConservativesFromPrimitives(prim, outputState);
}

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

DryAir::DryAir(RunConfiguration &_runfile, int _dim, int nvel)
    // : DryAir(_runfile.workFluid, _runfile.GetEquationSystem(), _runfile.visc_mult, _runfile.bulk_visc, _dim, nvel) {}
    : DryAir(_runfile.dryAirInput, _dim, nvel) {}

MFEM_HOST_DEVICE DryAir::DryAir(const DryAirInput inputs, int _dim, int nvel)
    : GasMixture(inputs.f, _dim, nvel) {
  numSpecies = (inputs.eq_sys == NS_PASSIVE) ? 2 : 1;
  ambipolar = false;
  twoTemperature_ = false;

  SetNumActiveSpecies();
  SetNumEquations();
#ifdef _GPU_
  assert(nvel_ <= gpudata::MAXDIM);
  assert(numSpecies <= gpudata::MAXSPECIES);
#endif

  gas_constant = 287.058;

  specific_heat_ratio = 1.4;
// TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
#if defined(_HIP_)
  visc_mult = inputs.visc_mult;
  bulk_visc_mult = inputs.bulk_visc_mult;
  Pr = 0.71;
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
  Sc = 0.71;
#endif
  // TODO(kevin): replace Nconservative/Nprimitive.
  // add extra equation for passive scalar
  if (inputs.eq_sys == Equations::NS_PASSIVE) {
    Nconservative++;
    Nprimitive++;
    num_equation++;
  }
}

DryAir::DryAir() {
  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;
// TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
#ifdef _GPU_
  visc_mult = 1.;
  Pr = 0.71;
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
  Sc = 0.71;
#endif
}

// DryAir::DryAir(int _dim, int _num_equation) {
//   gas_constant = 287.058;
//   // gas_constant = 1.; // for comparison against ex18
//   specific_heat_ratio = 1.4;
// // TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
// #ifdef _GPU_
//   visc_mult = 1.;
//   Pr = 0.71;
//   cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
//   Sc = 0.71;
// #endif
//
//   dim = _dim;
//   num_equation = _num_equation;
// }

void DryAir::setNumEquations() {
  Nconservative = nvel_ + 2;
  Nprimitive = Nconservative;
  num_equation = Nconservative;
}

// void EquationOfState::SetFluid(WorkingFluid _fluid) {
//   fluid = _fluid;
//   switch (fluid) {
//     case DRY_AIR:
//       gas_constant = 287.058;
//       // gas_constant = 1.; // for comparison against ex18
//       specific_heat_ratio = 1.4;
//       visc_mult = 1.;
//       Pr = 0.71;
//       cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
//       Sc = 0.71;
//       break;
//     default:
//       break;
//   }
// }

void DryAir::computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies) {
  speciesEnthalpies.SetSize(numSpecies);
  speciesEnthalpies = 0.0;

  return;
}

MFEM_HOST_DEVICE void DryAir::computeSpeciesEnthalpies(const double *state, double *speciesEnthalpies) {
  for (int sp = 0; sp < numSpecies; sp++) speciesEnthalpies[sp] = 0.0;
  return;
}

bool DryAir::StateIsPhysical(const mfem::Vector &state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, nvel_);
  const double den_energy = state(1 + nvel_);

  if (den < 0) {
    cout << "Negative density: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  if (den_energy <= 0) {
    cout << "Negative energy: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }

  double den_vel2 = 0;
  for (int i = 0; i < nvel_; i++) {
    den_vel2 += den_vel(i) * den_vel(i);
  }
  den_vel2 /= den;

  const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

  if (pres <= 0) {
    cout << "Negative pressure: " << pres << ", state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  return true;
}

// TODO(kevin): We need to move this routine to upper level, i.e. M2ulPhys.
// Diffusion velocity contributes to the characteristic speed, which mixture cannot handle or know.
// Compute the maximum characteristic speed.
double DryAir::ComputeMaxCharSpeed(const Vector &state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, nvel_);

  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state);
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

MFEM_HOST_DEVICE double DryAir::ComputeMaxCharSpeed(const double *state) const {
  const double den = state[0];

  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) {
    den_vel2 += state[d + 1] * state[d + 1];
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state);
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

void DryAir::GetConservativesFromPrimitives(const Vector &primit, Vector &conserv) {
  conserv = primit;

  double v2 = 0.;
  for (int d = 0; d < nvel_; d++) {
    v2 += primit[1 + d] * primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  // total energy
  conserv[1 + nvel_] = gas_constant * primit[0] * primit[1 + nvel_] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;

  // case of passive scalar
  if (num_equation > nvel_ + 2) {
    for (int n = 0; n < num_equation - nvel_ - 2; n++) {
      conserv[nvel_ + 2 + n] *= primit[0];
    }
  }
}

MFEM_HOST_DEVICE void DryAir::GetConservativesFromPrimitives(const double *primit, double *conserv) {
  for (int eq = 0; eq < num_equation; eq++) conserv[eq] = primit[eq];

  double v2 = 0.;
  for (int d = 0; d < nvel_; d++) {
    v2 += primit[1 + d] * primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  // total energy
  conserv[1 + nvel_] = gas_constant * primit[0] * primit[1 + nvel_] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;

  // case of passive scalar
  if (num_equation > nvel_ + 2) {
    for (int n = 0; n < num_equation - nvel_ - 2; n++) {
      conserv[nvel_ + 2 + n] *= primit[0];
    }
  }
}

void DryAir::GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit) {
  double T = ComputeTemperature(conserv);
  primit = conserv;

  for (int d = 0; d < nvel_; d++) primit[1 + d] /= conserv[0];

  primit[nvel_ + 1] = T;

  // case of passive scalar
  if (num_equation > nvel_ + 2) {
    for (int n = 0; n < num_equation - nvel_ - 2; n++) {
      primit[nvel_ + 2 + n] /= primit[0];
    }
  }
}

MFEM_HOST_DEVICE void DryAir::GetPrimitivesFromConservatives(const double *conserv, double *primit) {
  double T = ComputeTemperature(conserv);
  for (int eq = 0; eq < num_equation; eq++) primit[eq] = conserv[eq];

  for (int d = 0; d < nvel_; d++) primit[1 + d] /= conserv[0];

  primit[nvel_ + 1] = T;

  // case of passive scalar
  if (num_equation > nvel_ + 2) {
    for (int n = 0; n < num_equation - nvel_ - 2; n++) {
      primit[nvel_ + 2 + n] /= primit[0];
    }
  }
}

double DryAir::ComputeSpeedOfSound(const mfem::Vector &Uin, bool primitive) {
  double T;

  if (primitive) {
    T = Uin[1 + nvel_];
  } else {
    // conservatives passed in
    T = ComputeTemperature(Uin);
  }

  return sqrt(specific_heat_ratio * gas_constant * T);
}

double DryAir::ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive) {
  double T;
  if (primitive) {
    T = Uin[1 + nvel_];
  } else {
    T = ComputeTemperature(Uin);
  }

  return gas_constant * (T * dUp_dx[0] + Uin[0] * dUp_dx[1 + nvel_]);
}

double DryAir::ComputePressureFromPrimitives(const mfem::Vector &Up) { return gas_constant * Up[0] * Up[1 + nvel_]; }

MFEM_HOST_DEVICE double DryAir::ComputePressureFromPrimitives(const double *Up) {
  return gas_constant * Up[0] * Up[1 + nvel_];
}

void DryAir::computeStagnationState(const mfem::Vector &stateIn, mfem::Vector &stagnationState) {
  const double p = ComputePressure(stateIn);

  stagnationState.SetSize(num_equation);
  stagnationState = stateIn;

  // zero momentum
  for (int d = 0; d < nvel_; d++) stagnationState(1 + d) = 0.;

  // total energy
  stagnationState(1 + nvel_) = p / (specific_heat_ratio - 1.);
}

void DryAir::computeStagnantStateWithTemp(const mfem::Vector &stateIn, const double Temp, mfem::Vector &stateOut) {
  stateOut.SetSize(num_equation);
  stateOut = stateIn;

  for (int d = 0; d < nvel_; d++) stateOut(1 + d) = 0.;

  stateOut(1 + nvel_) = gas_constant / (specific_heat_ratio - 1.) * stateIn(0) * Temp;
}

// NOTE: modifyElectronEnergy will not be used for DryAir.
void DryAir::modifyEnergyForPressure(const mfem::Vector &stateIn, mfem::Vector &stateOut, const double &p,
                                     bool modifyElectronEnergy) {
  stateOut.SetSize(num_equation);
  stateOut = stateIn;

  double ke = 0.;
  for (int d = 0; d < nvel_; d++) ke += stateIn(1 + d) * stateIn(1 + d);
  ke *= 0.5 / stateIn(0);

  stateOut(1 + nvel_) = p / (specific_heat_ratio - 1.) + ke;
}

MFEM_HOST_DEVICE void DryAir::modifyEnergyForPressure(const double *stateIn, double *stateOut, const double &p,
                                                      bool modifyElectronEnergy) {
  for (int eq = 0; eq < num_equation; eq++) stateOut[eq] = stateIn[eq];

  double ke = 0.;
  for (int d = 0; d < nvel_; d++) ke += stateIn[1 + d] * stateIn[1 + d];
  ke *= 0.5 / stateIn[0];

  stateOut[1 + nvel_] = p / (specific_heat_ratio - 1.) + ke;
}

// TODO(kevin): check if this works for axisymmetric case.
void DryAir::computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal,
                                                     Vector &conservedState) {
  const double gamma = specific_heat_ratio;

  double temp = 0.;
  // For axisymmetric case, normal[2] = 0.0. For-loop only includes up to d < dim = 2.
  for (int d = 0; d < dim; d++) temp += meanNormalFluxes[1 + d] * normal[d];
  double A = 1. - 2. * gamma / (gamma - 1.);
  double B = 2 * temp / (gamma - 1.);
  double C = -2. * meanNormalFluxes[0] * meanNormalFluxes[1 + nvel_];
  for (int d = 0; d < nvel_; d++) C += meanNormalFluxes[1 + d] * meanNormalFluxes[1 + d];
  //   double p = (-B+sqrt(B*B-4.*A*C))/(2.*A);
  double p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);  // real solution

  Vector Up(num_equation);
  Up[0] = meanNormalFluxes[0] * meanNormalFluxes[0] / (temp - p);
  Up[1 + nvel_] = Temperature(&Up[0], &p, 1);
  //   Up[1+dim] = p;

  for (int d = 0; d < nvel_; d++) {
    if (d < dim) {
      Up[1 + d] = (meanNormalFluxes[1 + d] - p * normal[d]) / meanNormalFluxes[0];
    } else {  // azimuthal direction for axisymmetric case.
      Up[1 + d] = meanNormalFluxes[1 + d] / meanNormalFluxes[0];
    }
  }

  GetConservativesFromPrimitives(Up, conservedState);
}

// void DryAir::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up) {
//   double* pGridFunc = press->HostWrite();
//   const double* UpData = Up->HostRead();
//
//   const int nnode = press->FESpace()->GetNDofs();
//
//   for(int n=0;n<nnode;n++){
//     double tmp = UpData[n + (1+dim)*nnode];
//     double rho = UpData[n];
//     pGridFunc[n] = rho*gas_constant*tmp;
//   }
// }

// GPU FUNCTIONS
/*#ifdef _GPU_
double EquationOfState::pressure( double *state,
                                  double *KE,
                                  const double gamma,
                                  const int dim,
                                  const int num_equation)
{
  double p = 0.;
  for(int k=0;k<dim;k++) p += KE[k];
  return (gamma-1.)*(state[num_equation-1] - p);
}
#endif*/

//////////////////////////////////////////////////////////////////////////
////// Perfect Mixture GasMixture                     ////////////////////
//////////////////////////////////////////////////////////////////////////

PerfectMixture::PerfectMixture(RunConfiguration &_runfile, int _dim, int nvel)
    : PerfectMixture(_runfile.perfectMixtureInput, _dim, nvel) {}

MFEM_HOST_DEVICE PerfectMixture::PerfectMixture(PerfectMixtureInput inputs, int _dim, int nvel)
    : GasMixture(inputs.f, _dim, nvel) {
  numSpecies = inputs.numSpecies;
  assert(nvel_ <= gpudata::MAXDIM);
  assert(numSpecies <= gpudata::MAXSPECIES);

  // backgroundInputIndex_ = _runfile.backgroundIndex;
  // assert((backgroundInputIndex_ > 0) && (backgroundInputIndex_ <= numSpecies));
  // If electron is not included, then ambipolar and two-temperature are false.
  ambipolar = false;
  twoTemperature_ = false;

  // NOTE(kevin): electron species is enforced to be included in the input file.
  // bool isElectronIncluded = false;
  if (inputs.isElectronIncluded) {
    ambipolar = inputs.ambipolar;
    twoTemperature_ = inputs.twoTemperature;
  }

  gasParams.SetSize(numSpecies, GasParams::NUM_GASPARAMS);
  // composition_.SetSize(numSpecies, _runfile.numAtoms);

  // gasParams = 0.0;
  // int paramIdx = 0;  // new species index.
  // int targetIdx;
  // for (int sp = 0; sp < numSpecies; sp++) {  // input file species index.
  //   if (sp == backgroundInputIndex_ - 1) {
  //     targetIdx = numSpecies - 1;
  //   } else if (_runfile.speciesNames[sp] == "E") {
  //     targetIdx = numSpecies - 2;
  //     isElectronIncluded = true;
  //     // Set ambipolar and twoTemperature if electron is included.
  //     ambipolar = _runfile.IsAmbipolar();
  //     twoTemperature_ = _runfile.IsTwoTemperature();
  //   } else {
  //     targetIdx = paramIdx;
  //     paramIdx++;
  //   }
  //   speciesMapping_[_runfile.speciesNames[sp]] = targetIdx;
  //   mixtureToInputMap_[targetIdx] = sp;
  //   std::cout << "name, input index, mixture index: " << _runfile.speciesNames[sp] << ", " << sp << ", " << targetIdx
  //             << std::endl;
  //
  //   for (int param = 0; param < GasParams::NUM_GASPARAMS; param++)
  //     gasParams(targetIdx, param) = _runfile.GetGasParams(sp, (GasParams)param);
  //
  //   /* TODO(kevin): not sure we need atomMW and composition in GasMixture.
  //      not initialized at thit point. */
  //   // for (int a = 0; a < _runfile.numAtoms; a++)
  //   //   composition_(targetIdx, a) = _runfile.speciesComposition(sp, a);
  // }
  for (int sp = 0; sp < numSpecies; sp++) {
    // if (_runfile.speciesNames[sp] == "E") {
    //   isElectronIncluded = true;
    //   ambipolar = _runfile.IsAmbipolar();
    //   twoTemperature_ = _runfile.IsTwoTemperature();
    // }

    // speciesMapping_[_runfile.speciesNames[sp]] = sp;

    for (int param = 0; param < GasParams::NUM_GASPARAMS; param++)
      // gasParams(sp, param) = _runfile.gasParams(sp, param);
      gasParams[sp + param * numSpecies] = inputs.gasParams[sp + param * numSpecies];
  }

  SetNumActiveSpecies();
  SetNumEquations();

  // We assume the background species is neutral.
  assert(gasParams[(numSpecies - 1) + GasParams::SPECIES_CHARGES * numSpecies] == 0.0);
  // TODO(kevin): release electron species enforcing.
  assert(inputs.isElectronIncluded);
  // We assume the background species and electron have zero formation energy.
  assert(gasParams[(numSpecies - 2) + GasParams::FORMATION_ENERGY * numSpecies] == 0.0);
  assert(gasParams[(numSpecies - 1) + GasParams::FORMATION_ENERGY * numSpecies] == 0.0);

  for (int sp = 0; sp < gpudata::MAXSPECIES; sp++) {
    molarCV_[sp] = -1.0;
    molarCP_[sp] = -1.0;
    specificGasConstants_[sp] = -1.0;
    specificHeatRatios_[sp] = -1.0;
  }

  for (int sp = 0; sp < numSpecies; sp++) {
    // double inputSp = mixtureToInputMap_[sp];
    specificGasConstants_[sp] = UNIVERSALGASCONSTANT / GetGasParams(sp, GasParams::SPECIES_MW);

    // TODO(kevin): read these from input parser.
    // molarCV_(sp) = _runfile.getConstantMolarCV(inputSp) * UNIVERSALGASCONSTANT;
    molarCV_[sp] = inputs.molarCV[sp] * UNIVERSALGASCONSTANT;
    // NOTE: for perfect gas, CP = CV + R
    molarCP_[sp] = molarCV_[sp] + UNIVERSALGASCONSTANT;
    specificHeatRatios_[sp] = molarCP_[sp] / molarCV_[sp];
  }
}

// compute heavy-species heat capacity from number densities.
double PerfectMixture::computeHeaviesHeatCapacity(const double *n_sp, const double &nB) {
  double heatCapacity = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;  // neglect electron.
    heatCapacity += n_sp[sp] * molarCV_[sp];
  }
  heatCapacity += nB * molarCV_[numSpecies - 1];

  return heatCapacity;
}

double PerfectMixture::computeAmbipolarElectronNumberDensity(const double *n_sp) {
  double n_e = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_e += GetGasParams(sp, GasParams::SPECIES_CHARGES) * n_sp[sp];
  }  // Background species doesn't have to be included due to its neutral charge.
  assert(n_e >= 0.0);

  return n_e;
}

double PerfectMixture::computeBackgroundMassDensity(const double &rho, const double *n_sp, double &n_e,
                                                    bool isElectronComputed) {
  if ((!isElectronComputed) && (ambipolar)) {
    n_e = computeAmbipolarElectronNumberDensity(n_sp);
  }

  // computing background number density.
  double rhoB = rho;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // If not ambipolar, electron number density is included in this loop.
    rhoB -= GetGasParams(sp, GasParams::SPECIES_MW) * n_sp[sp];
  }

  if (ambipolar) {  // Electron species is assumed be to the second to last species.
    rhoB -= n_e * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
  }

  // assert(rhoB >= 0.0);
  if (rhoB < 0.) {
    grvy_printf(GRVY_ERROR, "\nNegative background density -> %f\n", rhoB);
    exit(-1);
  }

  return rhoB;
}

// TODO(kevin): Need to expand the number of primitive variables and store all of them.
// Right now, it's alwasy the same as conserved variables, to minimize gradient computation.
// Additional primitive variables (such as temperature currently) needs to be evaluated every time it is needed.
// Gradient computation can still keep the same number, while expanding the primitive variables.
void PerfectMixture::GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit) {
  Vector n_sp;
  computeNumberDensities(conserv, n_sp);

  // NOTE: we fix the number density as "necessary" species primitive and
  // compute its gradient only.
  // Conversion from grad n to grad X or grad Y will be provided.
  // Conversion routines will benefit from expanding Up with all X, Y and n.
  for (int sp = 0; sp < numActiveSpecies; sp++) primit[nvel_ + 2 + sp] = n_sp[sp];

  primit[0] = conserv[0];
  for (int d = 0; d < nvel_; d++) primit[d + 1] = conserv[d + 1] / conserv[0];

  double T_h, T_e;
  computeTemperaturesBase(conserv, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  primit[nvel_ + 1] = T_h;

  if (twoTemperature_)  // electron temperature as primitive variable.
    primit[num_equation - 1] = T_e;
}

void PerfectMixture::GetConservativesFromPrimitives(const Vector &primit, Vector &conserv) {
  conserv[0] = primit[0];
  for (int d = 0; d < nvel_; d++) conserv[d + 1] = primit[d + 1] * primit[0];

  // Convert species rhoY first.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    conserv[nvel_ + 2 + sp] = primit[nvel_ + 2 + sp] * GetGasParams(sp, GasParams::SPECIES_MW);
  }

  // NOTE: For now, we do not include all species number densities into Up.
  // This requires us to re-evaluate electron/background-species number density.
  double n_e = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&primit[nvel_ + 2]);
  } else {
    n_e = primit[nvel_ + 2 + numSpecies - 2];
  }
  double rhoB = computeBackgroundMassDensity(primit[0], &primit[nvel_ + 2], n_e, true);
  double nB = rhoB / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);

  if (twoTemperature_) conserv[num_equation - 1] = n_e * molarCV_[numSpecies - 2] * primit[num_equation - 1];

  // compute mixture heat capacity.
  double totalHeatCapacity = computeHeaviesHeatCapacity(&primit[nvel_ + 2], nB);
  if (!twoTemperature_) totalHeatCapacity += n_e * molarCV_[numSpecies - 2];

  double totalEnergy = 0.0;
  for (int d = 0; d < nvel_; d++) totalEnergy += primit[d + 1] * primit[d + 1];
  totalEnergy *= 0.5 * primit[0];
  totalEnergy += totalHeatCapacity * primit[nvel_ + 1];
  if (twoTemperature_) {
    totalEnergy += conserv[num_equation - 1];
  }

  for (int sp = 0; sp < numSpecies - 2; sp++) {
    totalEnergy += primit[nvel_ + 2 + sp] * GetGasParams(sp, GasParams::FORMATION_ENERGY);
  }

  conserv[nvel_ + 1] = totalEnergy;
}

// NOTE: Almost for sure ambipolar will remain true, then we have to always compute at least both Y and n.
// Mole fraction X will be needed almost everywhere, though it requires Y and n to be evaluated first.
// TODO(kevin): It is better to include all X, Y, n into primitive variable Up,
// in order to reduce repeated evaluation.
void PerfectMixture::computeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp) {
  X_sp.SetSize(numSpecies);
  Y_sp.SetSize(numSpecies);
  n_sp.SetSize(numSpecies);
  X_sp = 0.0;
  Y_sp = 0.0;
  n_sp = 0.0;

  double n_e = 0.0;
  double n = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_sp[sp] = conservedState[nvel_ + 2 + sp] / GetGasParams(sp, GasParams::SPECIES_MW);
    n += n_sp[sp];
    if (ambipolar) n_e += GetGasParams(sp, GasParams::SPECIES_CHARGES) * n_sp[sp];
  }  // Background species doesn't have to be included due to its neutral charge.
  if (ambipolar) {
    n_sp[numSpecies - 2] = n_e;  // Electron species is assumed be to the second to last species.
    n += n_e;
  }

  double Yb = 1.;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    Y_sp[sp] = conservedState[nvel_ + 2 + sp] / conservedState[0];
    Yb -= Y_sp[sp];
  }

  if (ambipolar) {  // Electron species is assumed be to the second to last species.
    Y_sp[numSpecies - 2] = n_e * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW) / conservedState[0];
    Yb -= Y_sp[numSpecies - 2];
  }
  if (Yb < 0.0) {
    std::cout << "rho: " << conservedState(0) << std::endl;
    std::cout << "n_sp: ";
    for (int sp = 0; sp < numActiveSpecies; sp++) std::cout << n_sp[sp] << ",\t";
    std::cout << std::endl;
    std::cout << "n_e: " << n_e << std::endl;
    std::cout << "Yb: " << Yb << std::endl;
    assert(Yb >= 0.0);  // In case of negative mass fraction.
  }
  Y_sp[numSpecies - 1] = Yb;

  n_sp[numSpecies - 1] = Y_sp[numSpecies - 1] * conservedState[0] / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  n += n_sp[numSpecies - 1];

  for (int sp = 0; sp < numSpecies; sp++) X_sp[sp] = n_sp[sp] / n;
}

void PerfectMixture::computeNumberDensities(const Vector &conservedState, Vector &n_sp) {
  n_sp.SetSize(numSpecies);
  n_sp = 0.0;

  double n_e = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_sp[sp] = conservedState[nvel_ + 2 + sp] / GetGasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&n_sp[0]);
    n_sp[numSpecies - 2] = n_e;  // Electron species is assumed be to the second to last species.
  }
  double rhoB = computeBackgroundMassDensity(conservedState[0], &n_sp[0], n_e, true);

  n_sp[numSpecies - 1] = rhoB / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
}

double PerfectMixture::ComputePressureFromPrimitives(const mfem::Vector &Up) {
  // NOTE: For now, we do not include all species number densities into Up.
  // This requires us to re-evaluate electron/background-species number density.
  double n_e = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&Up[nvel_ + 2]);
  } else {
    n_e = Up[nvel_ + 2 + numSpecies - 2];
  }
  double rhoB = computeBackgroundMassDensity(Up[0], &Up[nvel_ + 2], n_e, true);
  double nB = rhoB / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);

  double T_h = Up[nvel_ + 1];
  double T_e;
  if (twoTemperature_) {
    T_e = Up[num_equation - 1];
  } else {
    T_e = Up[nvel_ + 1];
  }
  double p = computePressureBase(&Up[nvel_ + 2], n_e, nB, T_h, T_e);

  return p;
}

// NOTE: This is almost the same as GetPrimitivesFromConservatives except storing other primitive variables.
double PerfectMixture::ComputePressure(const Vector &state, double *electronPressure) {
  Vector n_sp;
  computeNumberDensities(state, n_sp);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  if (electronPressure != NULL) *electronPressure = n_sp(numSpecies - 2) * UNIVERSALGASCONSTANT * T_e;

  // NOTE: compute pressure.
  double p = computePressureBase(&n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  return p;
}

double PerfectMixture::computePressureBase(const double *n_sp, const double n_e, const double n_B, const double T_h,
                                           const double T_e) {
  // NOTE: compute pressure.
  double n_h = 0.0;  // total number density of all heavy species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;  // treat electron separately.
    n_h += n_sp[sp];
  }
  n_h += n_B;

  double p = n_h * T_h;
  if (twoTemperature_) {
    p += n_e * T_e;
  } else {
    p += n_e * T_h;
  }
  p *= UNIVERSALGASCONSTANT;

  return p;
}

bool PerfectMixture::StateIsPhysical(const Vector &state) {
  bool physical = true;

  if (state(0) < 0) {
    cout << "Negative density! " << endl;
    physical = false;
  }
  if (state(nvel_ + 1) <= 0) {
    cout << "Negative energy! " << endl;
    physical = false;
  }

  /* TODO(kevin): take primitive variables as input for physicality check. */
  // Vector primitiveState(num_equation);
  // GetPrimitivesFromConservatives(state, primitiveState);
  // if (primitiveState(nvel_+1) < 0) {
  //   cout << "Negative pressure: " << primitiveState(nvel_+1) << "! " << endl;
  //   physical = false;
  // }
  // if (primitiveState(num_equation-1) <= 0) {
  //   cout << "Negative electron temperature: " << primitiveState(num_equation-1) << "! " << endl;
  //   physical = false;
  // }
  //
  // Vector n_sp;
  // Vector Y_sp;
  // Vector X_sp;
  // ComputeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  // for (int sp = 0; sp < numSpecies; sp++){
  //   if (n_sp[sp] < 0.0) {
  //     cout << "Negative " << sp << "-th species number density: " << n_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  //   if ((X_sp[sp] < 0.0) || (X_sp[sp] > 1.0)) {
  //     cout << "Unphysical " << sp << "-th species mole fraction: " << X_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  //   if ((Y_sp[sp] < 0.0) || (Y_sp[sp] > 1.0)) {
  //     cout << "Unphysical " << sp << "-th species mass fraction: " << Y_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  // }

  if (!physical) {
    cout << "Unphysical state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
  }
  return physical;
}

// NOTE: this routine only return heavy-species temperature. Need name-change.
double PerfectMixture::ComputeTemperature(const Vector &state) {
  Vector n_sp;
  computeNumberDensities(state, n_sp);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  return T_h;
}

void PerfectMixture::computeTemperaturesBase(const Vector &conservedState, const double *n_sp, const double n_e,
                                             const double n_B, double &T_h, double &T_e) {
  // compute mixture heat capacity.
  double totalHeatCapacity = computeHeaviesHeatCapacity(&n_sp[0], n_B);
  if (!twoTemperature_) totalHeatCapacity += n_e * molarCV_[numSpecies - 2];

  double totalEnergy = conservedState[nvel_ + 1];
  for (int sp = 0; sp < numSpecies - 2; sp++) {
    totalEnergy -= n_sp[sp] * GetGasParams(sp, GasParams::FORMATION_ENERGY);
  }

  // Comptue heavy-species temperature. If not two temperature, then this works as the unique temperature.
  T_h = 0.0;
  for (int d = 0; d < nvel_; d++) T_h -= conservedState[d + 1] * conservedState[d + 1];
  T_h *= 0.5 / conservedState[0];
  T_h += totalEnergy;
  if (twoTemperature_) T_h -= conservedState[num_equation - 1];
  T_h /= totalHeatCapacity;

  // electron temperature as primitive variable.
  // TODO(kevin): this will have a singularity when there is no electron.
  // It is highly unlikely to have zero electron in the simulation,
  // but it's better to have a capability to avoid this.
  if (twoTemperature_) {
    T_e = conservedState[num_equation - 1] / n_e / molarCV_[numSpecies - 2];
  } else {
    T_e = T_h;
  }

  return;
}

void PerfectMixture::computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies) {
  speciesEnthalpies.SetSize(numSpecies);

  Vector n_sp;
  computeNumberDensities(state, n_sp);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == numSpecies - 2) ? T_e : T_h;
    speciesEnthalpies(sp) = n_sp[sp] * (molarCP_[sp] * temp + GetGasParams(sp, GasParams::FORMATION_ENERGY));
  }

  return;
}

MFEM_HOST_DEVICE void PerfectMixture::computeSpeciesEnthalpies(const double *state, double *speciesEnthalpies) {
  // TODO(kevin): develop gpu version.
  printf("ERROR: PerfextMixture::computeSpeciesEnthalpies is not yet developed for gpu!");

  return;
}

double PerfectMixture::ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive) {
  if (primitive) {
    return computePressureDerivativeFromPrimitives(dUp_dx, Uin);
  } else {
    return computePressureDerivativeFromConservatives(dUp_dx, Uin);
  }
}

// NOTE(kevin): normal-vector-related parts are already handled.
double PerfectMixture::computePressureDerivativeFromPrimitives(const Vector &dUp_dx, const Vector &Uin) {
  double pressureGradient = 0.0;

  double n_e = 0.0;
  double dne_dx = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&Uin[nvel_ + 2]);
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      dne_dx += dUp_dx[nvel_ + 2 + sp] * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
  } else {
    n_e = Uin[nvel_ + 2 + numSpecies - 2];
    dne_dx = dUp_dx[nvel_ + 2 + numSpecies - 2];
  }

  double rhoB = computeBackgroundMassDensity(Uin[0], &Uin[nvel_ + 2], n_e, true);
  double nB = rhoB / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);

  double n_h = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    n_h += Uin[nvel_ + 2 + sp];
  }
  n_h += nB;
  pressureGradient += n_h * dUp_dx[nvel_ + 1];

  double numDenGrad = dUp_dx[0] / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    numDenGrad += dUp_dx[nvel_ + 2 + sp] *
                  (1.0 - GetGasParams(sp, GasParams::SPECIES_MW) / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW));
  }
  // Kevin: this electron-related term comes from background species.
  numDenGrad -=
      dne_dx * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW) / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  pressureGradient += numDenGrad * Uin[nvel_ + 1];

  if (twoTemperature_) {
    pressureGradient += n_e * dUp_dx[num_equation - 1] + dne_dx * Uin[num_equation - 1];
  } else {
    pressureGradient += n_e * dUp_dx[nvel_ + 1] + dne_dx * Uin[nvel_ + 1];
  }

  pressureGradient *= UNIVERSALGASCONSTANT;
  return pressureGradient;
}

// NOTE(kevin): normal-vector-related parts are already handled.
double PerfectMixture::computePressureDerivativeFromConservatives(const Vector &dUp_dx, const Vector &Uin) {
  double pressureGradient = 0.0;

  Vector n_sp;
  computeNumberDensities(Uin, n_sp);
  double dne_dx = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      dne_dx += dUp_dx[nvel_ + 2 + sp] * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
  } else {
    dne_dx = dUp_dx[nvel_ + 2 + numSpecies - 2];
  }

  double T_h, T_e;
  computeTemperaturesBase(Uin, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  double n_h = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    n_h += n_sp[sp];
  }
  pressureGradient += n_h * dUp_dx[nvel_ + 1];

  double numDenGrad = dUp_dx[0] / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    numDenGrad += dUp_dx[nvel_ + 2 + sp] *
                  (1.0 - GetGasParams(sp, GasParams::SPECIES_MW) / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW));
  }
  // Kevin: this electron-related term comes from background species.
  numDenGrad -=
      dne_dx * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW) / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  pressureGradient += numDenGrad * T_h;

  if (twoTemperature_) {
    pressureGradient += n_sp[numSpecies - 2] * dUp_dx[num_equation - 1] + dne_dx * T_e;
  } else {
    pressureGradient += n_sp[numSpecies - 2] * dUp_dx[nvel_ + 1] + dne_dx * T_h;
  }

  pressureGradient *= UNIVERSALGASCONSTANT;
  return pressureGradient;
}

double PerfectMixture::computeHeaviesMixtureCV(const double *n_sp, const double n_B) {
  double mixtureCV = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    mixtureCV += n_sp[sp] * molarCV_[sp];
  }
  mixtureCV += n_B * molarCV_[numSpecies - 1];

  return mixtureCV;
}

double PerfectMixture::computeHeaviesMixtureHeatRatio(const double *n_sp, const double n_B) {
  double mixtureCV = computeHeaviesMixtureCV(n_sp, n_B);
  double n_h = n_B;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    n_h += n_sp[sp];
  }

  return 1.0 + n_h * UNIVERSALGASCONSTANT / mixtureCV;
}

double PerfectMixture::computeSpeedOfSoundBase(const double *n_sp, const double n_B, const double rho, const double p) {
  double gamma = computeHeaviesMixtureHeatRatio(n_sp, n_B);

  return sqrt(gamma * p / rho);
}

// Compute the maximum characteristic speed.
double PerfectMixture::ComputeMaxCharSpeed(const Vector &state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, nvel_);

  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double sound = ComputeSpeedOfSound(state, false);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

double PerfectMixture::ComputeSpeedOfSound(const mfem::Vector &Uin, bool primitive) {
  if (primitive) {
    double n_e = 0.0;
    if (ambipolar) {
      n_e = computeAmbipolarElectronNumberDensity(&Uin[nvel_ + 2]);
    } else {
      n_e = Uin[nvel_ + 2 + numSpecies - 2];
    }
    double rhoB = computeBackgroundMassDensity(Uin[0], &Uin[nvel_ + 2], n_e, true);
    double nB = rhoB / GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);

    double T_e = (twoTemperature_) ? Uin[num_equation - 1] : Uin[nvel_ + 1];
    double p = computePressureBase(&Uin[nvel_ + 2], n_e, nB, Uin[nvel_ + 1], T_e);

    return computeSpeedOfSoundBase(&Uin[nvel_ + 2], nB, Uin[0], p);

  } else {
    Vector n_sp;
    computeNumberDensities(Uin, n_sp);

    double T_h, T_e;
    computeTemperaturesBase(Uin, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

    double p = computePressureBase(&n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

    return computeSpeedOfSoundBase(&n_sp[0], n_sp[numSpecies - 1], Uin[0], p);
  }
}

// NOTE: numberDensities have all species number density.
// NOTE(kevin): for axisymmetric case, this handles only r- and z-direction.
void PerfectMixture::ComputeMassFractionGradient(const double rho, const Vector &numberDensities,
                                                 const DenseMatrix &gradUp, DenseMatrix &massFractionGrad) {
  massFractionGrad.SetSize(numSpecies, dim);
  massFractionGrad = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {  // if not ambipolar, electron is included.
    for (int d = 0; d < dim; d++) {
      massFractionGrad(sp, d) = gradUp(nvel_ + 2 + sp, d) / rho - numberDensities(sp) / rho / rho * gradUp(0, d);
      massFractionGrad(sp, d) *= GetGasParams(sp, GasParams::SPECIES_MW);
    }
  }

  Vector neGrad(dim);
  neGrad = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      for (int d = 0; d < dim; d++) neGrad(d) += gradUp(nvel_ + 2 + sp, d) * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
    for (int d = 0; d < dim; d++) {
      massFractionGrad(numSpecies - 2, d) =
          neGrad(d) / rho - numberDensities(numSpecies - 2) / rho / rho * gradUp(0, d);
      massFractionGrad(numSpecies - 2, d) *= GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
    }
  }

  Vector mGradN(dim);
  mGradN = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) mGradN(d) += gradUp(nvel_ + 2 + sp, d) * GetGasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    for (int d = 0; d < dim; d++) mGradN(d) += neGrad(d) * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
  }
  double Yb = numberDensities(numSpecies - 1) * GetGasParams(numSpecies - 1, GasParams::SPECIES_MW) / rho;
  for (int d = 0; d < dim; d++) {
    massFractionGrad(numSpecies - 1, d) = ((1.0 - Yb) * gradUp(0, d) - mGradN(d)) / rho;
  }
}

// NOTE(kevin): for axisymmetric case, this handles only r- and z-direction.
void PerfectMixture::ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                                 DenseMatrix &moleFractionGrad) {
  moleFractionGrad.SetSize(numSpecies, dim);
  double totalN = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) totalN += numberDensities(sp);

  Vector neGrad;
  neGrad.SetSize(dim);
  neGrad = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      for (int d = 0; d < dim; d++) neGrad(d) += gradUp(nvel_ + 2 + sp, d) * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
  }

  Vector nBGrad(dim);
  for (int d = 0; d < dim; d++) nBGrad(d) = gradUp(0, d);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) nBGrad(d) -= gradUp(nvel_ + 2 + sp, d) * GetGasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    // nBGrad -= gasParams(numSpecies - 2, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW) *
    // neGrad;
    nBGrad.Add(-GetGasParams(numSpecies - 2, GasParams::SPECIES_MW), neGrad);
  }
  nBGrad /= GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);

  Vector totalNGrad;
  totalNGrad.SetSize(dim);
  totalNGrad = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {  // if not ambipolar, electron is included.
    for (int d = 0; d < dim; d++) totalNGrad(d) += gradUp(nvel_ + 2 + sp, d);
  }
  if (ambipolar) totalNGrad += neGrad;
  totalNGrad += nBGrad;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) {
      moleFractionGrad(sp, d) =
          gradUp(nvel_ + 2 + sp, d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
    }
  }
  if (ambipolar) {
    int sp = numSpecies - 2;
    for (int d = 0; d < dim; d++) {
      moleFractionGrad(sp, d) = neGrad(d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
    }
  }
  int sp = numSpecies - 1;
  for (int d = 0; d < dim; d++) {
    moleFractionGrad(sp, d) = nBGrad(d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
  }
}

// NOTE: this is used in isothermal wall, where gas temperature and electron temperature is assumed equal.
// i.e. it assumes single temperature at the wall regardless of two temperature condition inside the domain.
void PerfectMixture::computeStagnantStateWithTemp(const mfem::Vector &stateIn, const double Temp,
                                                  mfem::Vector &stateOut) {
  stateOut.SetSize(num_equation);
  stateOut = stateIn;

  // momentum = 0.;
  for (int d = 0; d < nvel_; d++) stateOut(1 + d) = 0.;

  // compute total energy
  Vector n_sp(numSpecies);
  computeNumberDensities(stateIn, n_sp);

  double Ch = computeHeaviesHeatCapacity(&n_sp[0], n_sp[numSpecies - 1]);
  // assuming electrons also have wall temperature
  double Ue = n_sp[numSpecies - 2] * molarCV_[numSpecies - 2] * Temp;

  stateOut(1 + nvel_) = Ch * Temp + Ue;

  if (twoTemperature_) {
    stateOut(num_equation - 1) = Ue;
  }

  // Kevin: added formation energies.
  for (int sp = 0; sp < numSpecies - 2; sp++)
    stateOut(1 + nvel_) += n_sp(sp) * GetGasParams(sp, GasParams::FORMATION_ENERGY);
}

// At Inlet BC, for two-temperature, electron temperature is set to be equal to gas temperature, where the total
// pressure is p.
void PerfectMixture::modifyEnergyForPressure(const mfem::Vector &stateIn, mfem::Vector &stateOut, const double &p,
                                             bool modifyElectronEnergy) {
  // will change the total energy to adjust to p
  stateOut.SetSize(num_equation);
  stateOut = stateIn;

  // number densities
  Vector n_sp(numSpecies);
  computeNumberDensities(stateIn, n_sp);

  double Th = 0., pe = 0.0;
  if (twoTemperature_ && (!modifyElectronEnergy)) {
    double Xeps = 1.0e-30;  // To avoid dividing by zero.
    double Te = stateIn(num_equation - 1) / (n_sp(numSpecies - 2) + Xeps) / molarCV_[numSpecies - 2];
    pe = n_sp(numSpecies - 2) * UNIVERSALGASCONSTANT * Te;
  }

  for (int sp = 0; sp < numSpecies; sp++) {
    if (twoTemperature_ && (!modifyElectronEnergy) && (sp == numSpecies - 2)) continue;
    Th += n_sp(sp);
  }
  Th = (p - pe) / (Th * UNIVERSALGASCONSTANT);

  // compute total energy with the modified temperature of heavies
  double totalHeatCapacity = computeHeaviesHeatCapacity(&n_sp[0], n_sp[numSpecies - 1]);
  // if (!twoTemperature_) totalHeatCapacity += n_sp[numSpecies - 2] * molarCV_(numSpecies - 2);
  double rE = totalHeatCapacity * Th;
  // if (twoTemperature_) rE += stateIn(num_equation - 1);

  double electronEnergy = 0.0;
  if (twoTemperature_) {
    electronEnergy =
        (modifyElectronEnergy) ? n_sp[numSpecies - 2] * molarCV_[numSpecies - 2] * Th : stateIn(num_equation - 1);
    stateOut(num_equation - 1) = electronEnergy;
  } else {
    electronEnergy = n_sp[numSpecies - 2] * molarCV_[numSpecies - 2] * Th;
  }
  rE += electronEnergy;

  for (int d = 0; d < nvel_; d++) rE += 0.5 * stateIn[d + 1] * stateIn[d + 1] / stateIn[0];

  // double nB = stateIn(0);  // background species
  // for (int sp = 0; sp < numActiveSpecies; sp++) nB -= stateIn(2 + nvel_ + sp);
  //
  // if (ambipolar) nB -= ne * gasParams(numSpecies - 2, GasParams::SPECIES_MW);
  // nB /= gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  //
  // double Th = 0., Te = 0.;
  // if (twoTemperature) {
  //   Te = stateIn(num_equation - 1) / ne / molarCV_(numSpecies - 2);
  //   double pe = stateIn(2 + nvel_ + numSpecies - 2) / GetGasParams(numSpecies - 2, GasParams::SPECIES_MW) *
  //               UNIVERSALGASCONSTANT * Te;
  //
  //   for (int sp = 0; sp < numActiveSpecies; sp++) Th += n_s(sp);
  //   Th += nB;
  //   Th = (p - pe) / (Th * UNIVERSALGASCONSTANT);
  // } else {
  //   for (int sp = 0; sp < numActiveSpecies; sp++) Th += stateIn(2 + nvel_ + sp) / gasParams(sp,
  //   GasParams::SPECIES_MW); if (ambipolar) Th += ne; Th += nB; Th = p / (Th * UNIVERSALGASCONSTANT);
  // }
  //
  // // compute total energy with the modified temperature of heavies
  // double rE = 0.;
  // for (int sp = 0; sp < numActiveSpecies; sp++) rE += n_s(sp) * molarCV_(sp) * Th;
  // if (twoTemperature) rE += ne * molarCV_(numSpecies - 2) * Te;
  // rE += nB * molarCV_(numSpecies - 1) * Th;

  // Kevin: added formation energies.
  for (int sp = 0; sp < numSpecies - 2; sp++) rE += n_sp(sp) * GetGasParams(sp, GasParams::FORMATION_ENERGY);

  stateOut(1 + nvel_) = rE;
}

// TODO(kevin): check if this works for axisymmetric case.
void PerfectMixture::computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal,
                                                             Vector &conservedState) {
  Vector Up(num_equation);

  Vector numberDensityFluxes(numActiveSpecies);
  double formEnergyFlux = 0.0;
  double nBFlux = meanNormalFluxes(0);  // background species number density flux.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    numberDensityFluxes(sp) = meanNormalFluxes(nvel_ + 2 + sp) / GetGasParams(sp, GasParams::SPECIES_MW);
    formEnergyFlux += numberDensityFluxes(sp) * GetGasParams(sp, GasParams::FORMATION_ENERGY);
    nBFlux -= meanNormalFluxes(nvel_ + 2 + sp);
  }
  double neFlux = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      neFlux += numberDensityFluxes(sp) * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
    formEnergyFlux += neFlux * GetGasParams(numSpecies - 2, GasParams::FORMATION_ENERGY);
    nBFlux -= neFlux * GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
  } else {  // if not ambipolar, mass flux is already subtracted from nBFlux.
    neFlux = numberDensityFluxes(numSpecies - 2);
  }
  nBFlux /= GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  formEnergyFlux += nBFlux * GetGasParams(numSpecies - 1, GasParams::FORMATION_ENERGY);

  double Te = 0.0;
  if (twoTemperature_) {
    Te = meanNormalFluxes(num_equation - 1) / molarCP_[numSpecies - 2] / neFlux;
    // Te = meanNormalFluxes(num_equation - 1) / molarCV_(numSpecies - 2) / neFlux;
    Up(num_equation - 1) = Te;
  }

  double nMix = 0.0, cpMix = 0.0;
  for (int sp = 0; sp < numSpecies - 2; sp++) {
    nMix += numberDensityFluxes(sp);
    cpMix += numberDensityFluxes(sp) * molarCP_[sp];
  }
  nMix += nBFlux;
  cpMix += nBFlux * molarCP_[numSpecies - 1];
  if (!twoTemperature_) {
    nMix += neFlux;
    cpMix += neFlux * molarCP_[numSpecies - 2];
  }
  cpMix /= nMix;

  double temp = 0.;
  // NOTE(kevin): for axisymmetric case, normal[2] = 0.0. for-loop only includes up to d < dim = 2.
  for (int d = 0; d < dim; d++) temp += meanNormalFluxes[1 + d] * normal[d];
  double A = 1. - 2. * cpMix / UNIVERSALGASCONSTANT;
  double B = 2. * temp * (cpMix / UNIVERSALGASCONSTANT - 1.);
  double C = -2. * meanNormalFluxes[0] * meanNormalFluxes[1 + nvel_];
  for (int d = 0; d < nvel_; d++) C += meanNormalFluxes[1 + d] * meanNormalFluxes[1 + d];
  if (twoTemperature_) C += 2.0 * meanNormalFluxes[0] * neFlux * (molarCP_[numSpecies - 2] - cpMix) * Te;
  C += 2.0 * meanNormalFluxes[0] * formEnergyFlux;
  //   double p = (-B+sqrt(B*B-4.*A*C))/(2.*A);
  double p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);  // real solution

  double Th = (temp - p) / meanNormalFluxes[0] * p / UNIVERSALGASCONSTANT;
  if (twoTemperature_) Th -= neFlux * Te;
  Th /= nMix;

  Up[0] = meanNormalFluxes[0] * meanNormalFluxes[0] / (temp - p);
  for (int d = 0; d < nvel_; d++) {
    if (d < dim) {
      Up[1 + d] = (meanNormalFluxes[1 + d] - p * normal[d]) / meanNormalFluxes[0];
    } else {  // For axisymmetric case.
      Up[1 + d] = meanNormalFluxes[1 + d] / meanNormalFluxes[0];
    }
  }
  Up[1 + nvel_] = Th;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    Up[2 + nvel_ + sp] = numberDensityFluxes(sp) * meanNormalFluxes[0] / (temp - p);
  }

  GetConservativesFromPrimitives(Up, conservedState);
}

// NOTE(kevin): for axisymmetric case, this handles only r- and z-direction.
void PerfectMixture::computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                                 Vector &gradPe) {
  gradPe.SetSize(dim);

  Vector neGrad;
  neGrad.SetSize(dim);
  neGrad = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      for (int d = 0; d < dim; d++) neGrad(d) += gradUp(nvel_ + 2 + sp, d) * GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
  } else {
    for (int d = 0; d < dim; d++) neGrad(d) = gradUp(nvel_ + numSpecies, d);  // nvel_ + 2 + (numSpecies - 2)
  }

  for (int d = 0; d < dim; d++)
    gradPe(d) = (neGrad(d) * T_e + n_e * gradUp(num_equation - 1, d)) * UNIVERSALGASCONSTANT;

  return;
}

// TODO(kevin): this assumes all positive ions, whatever they are,
// are recombined with electron into the background species.
// For air plasma, we will need specific reaction inputs.
// We may need to move this to rather chemistry class.
void PerfectMixture::computeSheathBdrFlux(const Vector &state, BoundaryViscousFluxData &bcFlux) {
  Vector n_sp(numSpecies);
  computeNumberDensities(state, n_sp);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  for (int sp = 0; sp < numSpecies; sp++) bcFlux.primFlux[sp] = 0.0;

  // Compute Bohm velocity for positive ions.
  for (int sp = 0; sp < numSpecies; sp++) {
    double Zsp = GetGasParams(sp, GasParams::SPECIES_CHARGES);
    if (Zsp > 0.0) {
      double msp = GetGasParams(sp, GasParams::SPECIES_MW);
      double VB = sqrt((T_h + Zsp * T_e) * UNIVERSALGASCONSTANT / msp);
      // TODO(kevin): need to check the sign (out of wall or into the wall)
      bcFlux.primFlux[sp] = VB;
      bcFlux.primFlux[numSpecies - 2] += Zsp * n_sp(sp) * VB;
      bcFlux.primFlux[numSpecies - 1] -= msp * n_sp(sp) * VB;  // fully catalytic wall.
    }
  }
  bcFlux.primFlux[numSpecies - 2] /= n_sp(numSpecies - 2);
  bcFlux.primFlux[numSpecies - 1] -=
      GetGasParams(numSpecies - 2, GasParams::SPECIES_MW) * n_sp(numSpecies - 2) * bcFlux.primFlux[numSpecies - 2];
  bcFlux.primFlux[numSpecies - 1] /= GetGasParams(numSpecies - 1, GasParams::SPECIES_MW) * n_sp(numSpecies - 1);

  if (twoTemperature_) {
    double vTe = sqrt(8.0 * UNIVERSALGASCONSTANT * T_e / PI / GetGasParams(numSpecies - 2, GasParams::SPECIES_MW));
    double gamma = -log(4.0 / vTe * bcFlux.primFlux[numSpecies - 2]);

    bcFlux.primFlux[numSpecies + nvel_ + 1] =
        bcFlux.primFlux[numSpecies - 2] * (gamma + 2.0) * n_sp(numSpecies - 2) * UNIVERSALGASCONSTANT * T_e;
  }
}
