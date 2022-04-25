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
#ifndef EQUATION_OF_STATE_HPP_
#define EQUATION_OF_STATE_HPP_

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <grvy.h>
#include <tps_config.h>

#include <mfem.hpp>
#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

// TODO(kevin): move these constants to dataStructures.hpp
static const int MAXSPECIES = 200;
static const double UNIVERSALGASCONSTANT = 8.3144598;  // J * mol^(-1) * K^(-1)
static const double AVOGADRONUMBER = 6.0221409e+23;    // mol^(-1)
static const double BOLTZMANNCONSTANT = UNIVERSALGASCONSTANT / AVOGADRONUMBER;
static const double VACUUMPERMITTIVITY = 8.8541878128e-12;
static const double ELECTRONCHARGE = 1.60218e-19;
static const double MOLARELECTRONCHARGE = ELECTRONCHARGE * AVOGADRONUMBER;

class GasMixture {
 protected:
  WorkingFluid fluid;
  int num_equation;
  int dim;
  int nvel_; // number of velocity, which differs from dim for axisymmetric case.

  int numSpecies;
  int numActiveSpecies;

  int backgroundInputIndex_ = MAXSPECIES;

  bool ambipolar;
  bool twoTemperature_;

  DenseMatrix gasParams;

  std::map<int, int> mixtureToInputMap_;

  // NOTE: the indexes here starts from 0, 1, ...
  // Mapping between name and species index in gas parameter arrays.
  std::map<std::string, int> speciesMapping_;

  // We fix species primitive to be number density and compute its gradient only.
  // Conversion to X- and Y-gradient are easy, compared to the other way around.
  // Regardless of this, we may need to consider expanding Up with all X, Y, and n.
  // SpeciesPrimitiveType speciesPrimitiveType = NUM_SPECIES_PRIMITIVES;

  // number of conservative and primitive/visualization variables
  int Nconservative, Nprimitive;

// TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
#ifdef _GPU_
  double specific_heat_ratio;
  double gas_constant;
  double visc_mult;
  double bulk_visc_mult;
  // Prandtl number
  double Pr;         // Prandtl number
  double cp_div_pr;  // cp divided by Pr (used in conductivity calculation)
  // Fick's law
  double Sc;  // Schmidt number
#endif

  // If not ambipolar, one species continuity equation is replaced by global continuity equation.
  // If ambipolar, electron continuity equation is also replaced by an algebraic equation (determined by GasMixture).
  void SetNumActiveSpecies() { numActiveSpecies = ambipolar ? (numSpecies - 2) : (numSpecies - 1); }
  // Add electron energy equation if two temperature.
  void SetNumEquations() {
    num_equation = twoTemperature_ ? (nvel_ + 3 + numActiveSpecies) : (nvel_ + 2 + numActiveSpecies);
  }

 public:
  GasMixture(WorkingFluid _fluid, int _dim, int nvel);
  GasMixture() {}

  ~GasMixture() {}

  void SetFluid(WorkingFluid _fluid);

  WorkingFluid GetWorkingFluid() { return fluid; }

  int GetNumSpecies() { return numSpecies; }
  int GetNumActiveSpecies() { return numActiveSpecies; }
  int GetNumEquations() { return num_equation; }
  int GetDimension() { return dim; }
  int GetNumVels() { return nvel_; }
  bool IsAmbipolar() { return ambipolar; }
  bool IsTwoTemperature() { return twoTemperature_; }
  int getInputIndexOf(int mixtureIndex) { return mixtureToInputMap_[mixtureIndex]; }
  // int getElectronMixtureIndex() { return (speciesMapping_.count("E")) ? speciesMapping_["E"] : -1; }
  std::map<int, int> *getMixtureToInputMap() { return &mixtureToInputMap_; }
  std::map<std::string, int> *getSpeciesMapping() { return &speciesMapping_; }

  double GetGasParams(int species, GasParams param) { return gasParams(species, param); }

  int GetNumConservativeVariables() { return Nconservative; }
  int GetNumPrimitiveVariables() { return Nprimitive; }

  virtual double ComputePressure(const Vector &state,
                                 double *electronPressure = NULL) = 0;  // pressure from conservatives
  virtual double ComputePressureFromPrimitives(const Vector &Up) = 0;   // pressure from primitive variables
  virtual double ComputeTemperature(const Vector &state) = 0;
  virtual double Temperature(double *rho, double *p,
                             int nsp) = 0;  // temperature given densities and pressures of all species

  virtual void computeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp) {}
  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies) = 0;

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit) = 0;
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv) = 0;

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true) = 0;

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state) = 0;

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true) = 0;

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state) = 0;

  // Compute X, Y gradients from number density gradient.
  // NOTE(kevin): for axisymmetric case, these handle only r- and z-direction.
  virtual void ComputeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad) = 0;
  virtual void ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad) = 0;
  // TODO(kevin): Compute pressure gradient from temperature gradient.
  virtual void ComputePressureGradient(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &PressureGrad) {}

// TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
#ifdef _GPU_
  // TODO(kevin): Need to remove these and fix wherever they are used.
  // We cannot use these for multi species (heat ratio of which species?)
  // These are used in forcingTerm, Fluxes ASSUMING that the fluid is single species.
  virtual double GetSpecificHeatRatio() { return specific_heat_ratio; }
  virtual double GetGasConstant() { return gas_constant; }
#else
  virtual double GetSpecificHeatRatio() = 0;
  virtual double GetGasConstant() = 0;
#endif

  virtual void computeNumberDensities(const Vector &conservedState, Vector &n_sp) {}

  virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

// TODO(kevin): GPU routines are not yet fully gas-agnostic. Need to be removed.
#ifdef _GPU_
  double GetPrandtlNum() { return Pr; }
  double GetSchmidtNum() { return Sc; }
  double GetViscMultiplyer() { return visc_mult; }
  double GetBulkViscMultiplyer() { return bulk_visc_mult; }
#endif

  virtual double getMolarCV(int species) {}
  virtual double getMolarCP(int species) {}

  // BC related functions
  virtual void computeStagnationState(const Vector &stateIn, Vector &stagnationState);
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut) {}
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false) {}

  virtual double computeAmbipolarElectronNumberDensity(const double *n_sp) {}
  virtual double computeBackgroundMassDensity(const double &rho, const double *n_sp, double &n_e,
                                              bool isElectronComputed = false) {}

  // TODO(kevin): check if this works for axisymmetric case.
  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal,
                                                       Vector &conservedState) {}

  virtual double computeElectronEnergy(const double n_e, const double T_e) = 0;
  virtual double computeElectronPressure(const double n_e, const double T_e) = 0;
  // NOTE(kevin): for axisymmetric case, this handles only r- and z-direction.
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                           Vector &gradPe) = 0;
};

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

class DryAir : public GasMixture {
 private:
  double specific_heat_ratio;
  double gas_constant;

  virtual void setNumEquations();

 public:
  DryAir(RunConfiguration &_runfile, int _dim, int nvel);
  DryAir();  // this will only be usefull to get air constants
  // DryAir(int dim, int num_equation);

  ~DryAir() {}

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state, double *electronPressure = NULL);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double ComputeTemperature(const Vector &state);
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / gas_constant / rho[0]; }

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state);

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual double GetSpecificHeatRatio() { return specific_heat_ratio; }
  virtual double GetGasConstant() { return gas_constant; }

  virtual void ComputeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad) {}
  virtual void ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad) {}

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  // BC related functions
  virtual void computeStagnationState(const Vector &stateIn, Vector &stagnationState);
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);

  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal,
                                                       Vector &conservedState);

  virtual double computeElectronEnergy(const double n_e, const double T_e) {}
  virtual double computeElectronPressure(const double n_e, const double T_e) {}
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                           Vector &gradPe) {}
  // GPU functions
  // TODO(kevin): GPU part is not refactored for axisymmetric case.
#ifdef _GPU_
  static MFEM_HOST_DEVICE double pressure(const double *state, double *KE, const double &gamma, const int &dim,
                                          const int &num_equation) {
    double p = 0.;
    for (int k = 0; k < dim; k++) p += KE[k];
    return (gamma - 1.) * (state[1 + dim] - p);
  }

  static MFEM_HOST_DEVICE double ComputePressureFromPrimitives_gpu(const double *Up, const double &Rg, const int &dim) {
    return Up[0] * Rg * Up[1 + dim];
  }

  static MFEM_HOST_DEVICE double temperature(const double *state, double *KE, const double &gamma, const double &Rgas,
                                             const int &dim, const int &num_equation) {
    double temp = 0.;
    for (int k = 0; k < dim; k++) temp += KE[k];
    temp /= state[0];
    return (gamma - 1.0) / Rgas * (state[1 + dim] / state[0] - temp);
  }

  static MFEM_HOST_DEVICE double temperatureFromConservative(const double *u, const double &gamma, const double &Rg,
                                                             const int &dim, const int &num_equation) {
    double k = 0.;
    for (int d = 0; d < dim; d++) k += u[1 + d] * u[1 + d];
    k /= u[0] * u[0];
    return (gamma - 1.) / Rg * (u[1 + dim] / u[0] - 0.5 * k);
  }

  // Sutherland's law
  static MFEM_HOST_DEVICE double GetViscosity_gpu(const double &temp) {
    return 1.458e-6 * pow(temp, 1.5) / (temp + 110.4);
  }

  static MFEM_HOST_DEVICE double GetThermalConductivity_gpu(const double &visc, const double &gamma, const double &Rg,
                                                            const double &Pr) {
    const double cp = gamma * Rg / (gamma - 1.);
    return visc * cp / Pr;
  }

  static MFEM_HOST_DEVICE void computeStagnantStateWithTemp_gpu(const double *stateIn, double *stagState,
                                                                const double &Temp, const double &gamma,
                                                                const double &Rg, const int &num_equation,
                                                                const int &dim, const int &thrd,
                                                                const int &maxThreads) {
    if (thrd > 0 && thrd <= dim) stagState[thrd] = 0.;

    if (thrd == 1 + dim) stagState[thrd] = Rg / (gamma - 1.) * stateIn[0] * Temp;
  }

  static MFEM_HOST_DEVICE void modifyEnergyForPressure_gpu(const double *stateIn, double *stateOut, const double &p,
                                                           const double &gamma, const double &Rg,
                                                           const int &num_equation, const int &dim, const int &thrd,
                                                           const int &maxThreads) {
    MFEM_SHARED double ke;
    if (thrd == maxThreads - 1) {
      ke = 0.;
      for (int d = 0; d < dim; d++) ke += stateIn[1 + d] * stateIn[1 + d];
      ke *= 0.5 / stateIn[0];
    }
    for (int eq = thrd; eq < num_equation; eq += maxThreads) stateOut[eq] = stateIn[eq];
    MFEM_SYNC_THREAD;

    if (thrd == 0) stateOut[1 + dim] = p / (gamma - 1.) + ke;
  }
#endif
};

// class EquationOfState {
//  private:
//   WorkingFluid fluid;
//   double specific_heat_ratio;
//   double gas_constant;
//   double visc_mult;
//   double thermalConductivity;
//
//   double bulk_visc_mult;
//
//   // Prandtl number
//   double Pr;         // Prandtl number
//   double cp_div_pr;  // cp divided by Pr (used in conductivity calculation)
//
//   // Fick's law
//   double Sc;  // Schmidt number
//
//  public:
//   EquationOfState();
//
//   void SetFluid(WorkingFluid _fluid);
//
//   void SetViscMult(double _visc_mult) { visc_mult = _visc_mult; }
//   void SetBulkViscMult(double _bulk_mult) { bulk_visc_mult = _bulk_mult; }
//
//   double ComputePressure(const Vector &state, int dim);
//
//   void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit, const int &dim, const int
//   &num_equations); void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv, const int &dim, const
//   int &num_equations);
//
//   // Compute the maximum characteristic speed.
//   double ComputeMaxCharSpeed(const Vector &state, const int dim);
//
//   // Physicality check (at end)
//   bool StateIsPhysical(const Vector &state, const int dim);
//
//   double GetSpecificHeatRatio() { return specific_heat_ratio; }
//   double GetGasConstant() { return gas_constant; }
//   double GetViscosity(const double &temp);
//   double GetPrandtlNum() { return Pr; }
//   double GetSchmidtNum() { return Sc; }
//   double GetViscMultiplyer() { return visc_mult; }
//   double GetBulkViscMultiplyer() { return bulk_visc_mult; }
//   double GetThermalConductivity(const double &visc);
//
//
// };

// additional functions inlined for speed...
inline double DryAir::ComputePressure(const Vector &state, double *electronPressure) {
  if (electronPressure != NULL) *electronPressure = 0.0;
  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) * (state[1 + nvel_] - 0.5 * den_vel2);
}

inline double DryAir::ComputeTemperature(const Vector &state) {
  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) / gas_constant * (state[1 + nvel_] - 0.5 * den_vel2) / state[0];
}

//////////////////////////////////////////////////////
//////// Test Binary Air mixture
//////////////////////////////////////////////////////
// NOTE: this is mixture of two idential air species.
// mixture variables (density, pressure, temperature) are treated as single species.
// Only mass fractions are treated as binary mixture, essentially the same as PASSIVE_SCALAR.

class TestBinaryAir : public GasMixture {
 private:
  double specific_heat_ratio;
  double gas_constant;

  // virtual void SetNumEquations();
 public:
  TestBinaryAir(RunConfiguration &_runfile, int _dim, int nvel);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~TestBinaryAir() {}

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state, double *electronPressure = NULL);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double ComputeTemperature(const Vector &state);
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / rho[0] / gas_constant; }

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);
  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual void ComputeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp);

  virtual void ComputeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);

  virtual double GetSpecificHeatRatio() { return specific_heat_ratio; }
  virtual double GetGasConstant() { return gas_constant; }

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  virtual double computeElectronEnergy(const double n_e, const double T_e) {}
  virtual double computeElectronPressure(const double n_e, const double T_e) {}
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                           Vector &gradPe) {}

  // GPU functions
#ifdef _GPU_

#endif
};

// additional functions inlined for speed...
inline double TestBinaryAir::ComputePressure(const Vector &state, double *electronPressure) {
  if (electronPressure != NULL) *electronPressure = 0.0;
  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) * (state[1 + nvel_] - 0.5 * den_vel2);
}

// additional functions inlined for speed...
inline double TestBinaryAir::ComputeTemperature(const Vector &state) {
  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  double rhoU = state[1 + nvel_] - 0.5 * den_vel2;
  double cv = gas_constant / (specific_heat_ratio - 1.0);

  return rhoU / state[0] / cv;
}

//////////////////////////////////////////////////////////////////////////
////// Perfect Mixture GasMixture                     ////////////////////
//////////////////////////////////////////////////////////////////////////

class PerfectMixture : public GasMixture {
 private:
  Vector specificHeatRatios_;
  Vector specificGasConstants_;
  Vector molarCV_;
  Vector molarCP_;

  // virtual void SetNumEquations();
 public:
  PerfectMixture(RunConfiguration &_runfile, int _dim, int nvel);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~PerfectMixture() {}

  virtual double getMolarCV(int species) { return molarCV_(species); }
  virtual double getMolarCP(int species) { return molarCP_(species); }
  virtual double getSpecificHeatRatio(int species) { return specificHeatRatios_(species); }
  virtual double getSpecificGasConstant(int species) { return specificGasConstants_(species); }

  // Kevin: these are mixture heat ratio and gas constant. need to change argument.
  virtual double GetSpecificHeatRatio() { return molarCP_(numSpecies - 1) / molarCV_(numSpecies - 1); }
  virtual double GetGasConstant() { return specificGasConstants_(numSpecies - 1); }

  virtual double computeHeaviesHeatCapacity(const double *n_sp, const double &nB);
  virtual double computeAmbipolarElectronNumberDensity(const double *n_sp);
  virtual double computeBackgroundMassDensity(const double &rho, const double *n_sp, double &n_e,
                                              bool isElectronComputed = false);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);

  virtual void computeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp);
  virtual void computeNumberDensities(const Vector &conservedState, Vector &n_sp);

  virtual double ComputePressure(const Vector &state, double *electronPressure = NULL);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double computePressureBase(const double *n_sp, const double n_e, const double n_B, const double T_h,
                                     const double T_e);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual double ComputeTemperature(const Vector &state);
  virtual void computeTemperaturesBase(const Vector &conservedState, const double *n_sp, const double n_e,
                                       const double n_B, double &T_h, double &T_e);

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);

  // TODO(kevin): Kevin - I don't think we should use this for boundary condition.
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / rho[0] / GetGasConstant(); }

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);
  virtual double computePressureDerivativeFromPrimitives(const Vector &dUp_dx, const Vector &Uin);
  virtual double computePressureDerivativeFromConservatives(const Vector &dUp_dx, const Vector &Uin);

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state);

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);
  virtual double computeSpeedOfSoundBase(const double *n_sp, const double n_B, const double rho, const double p);

  virtual double computeHeaviesMixtureCV(const double *n_sp, const double n_B);
  virtual double computeHeaviesMixtureHeatRatio(const double *n_sp, const double n_B);

  virtual void ComputeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);

  // functions needed for BCs
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);

  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal,
                                                       Vector &conservedState);

  virtual double computeElectronEnergy(const double n_e, const double T_e) {
    return n_e * molarCV_(numSpecies - 2) * T_e;
  }
  virtual double computeElectronPressure(const double n_e, const double T_e) {
    return n_e * UNIVERSALGASCONSTANT * T_e;
  }
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                           Vector &gradPe);
  // GPU functions
#ifdef _GPU_

#endif
};

#endif  // EQUATION_OF_STATE_HPP_
