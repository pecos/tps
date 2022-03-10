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

#include "dataStructures.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

// Kevin: maybe use namespace?
static const int MAXSPECIES = 200;
static const double UNIVERSALGASCONSTANT = 8.3144598;  // J * mol^(-1) * K^(-1)
static const double AVOGADRONUMBER = 6.0221409e+23;    // mol^(-1)
static const double BOLTZMANNCONSTANT = UNIVERSALGASCONSTANT / AVOGADRONUMBER;
static const double VACUUMPERMITTIVITY = 8.8541878128e-12;
static const double ELECTRONCHARGE = 1.60218e-19;

class GasMixture {
 protected:
  WorkingFluid fluid;
  int num_equation;
  int dim;

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

  // If not ambipolar, one species continuity equation is replaced by global continuity equation.
  // If ambipolar, electron continuity equation is also replaced by an algebraic equation (determined by GasMixture).
  void SetNumActiveSpecies() { numActiveSpecies = ambipolar ? (numSpecies - 2) : (numSpecies - 1); }
  // Add electron energy equation if two temperature.
  void SetNumEquations() {
    num_equation = twoTemperature_ ? (dim + 3 + numActiveSpecies) : (dim + 2 + numActiveSpecies);
  }

 public:
  GasMixture(WorkingFluid _fluid, int _dim);
  GasMixture(){}

  ~GasMixture(){}

  void SetFluid(WorkingFluid _fluid);

  WorkingFluid GetWorkingFluid() { return fluid; }

  int GetNumSpecies() { return numSpecies; }
  int GetNumActiveSpecies() { return numActiveSpecies; }
  int GetNumEquations() { return num_equation; }
  int GetDimension() { return dim; }
  bool IsAmbipolar() { return ambipolar; }
  bool IsTwoTemperature() { return twoTemperature_; }
  int getInputIndexOf(int mixtureIndex) { return mixtureToInputMap_[mixtureIndex]; }
  // int getElectronMixtureIndex() { return (speciesMapping_.count("E")) ? speciesMapping_["E"] : -1; }
  std::map<int, int> *getMixtureToInputMap() { return &mixtureToInputMap_; }
  std::map<std::string, int> *getSpeciesMapping() { return &speciesMapping_; }

  double GetGasParams(int species, GasParams param) { return gasParams(species, param); }
  DenseMatrix GetGasParam() {return gasParams;}
  

  int GetNumConservativeVariables() { return Nconservative; }
  int GetNumPrimitiveVariables() { return Nprimitive; }
  
  virtual const Vector getMolarCVs(){}

  virtual double ComputePressure(const Vector &state, double &electronPressure) = 0;             // pressure from conservatives
  virtual double ComputePressureFromPrimitives(const Vector &Up) = 0;  // pressure from primitive variables
  virtual double ComputeTemperature(const Vector &state) = 0;
  virtual double Temperature(double *rho, double *p,
                             int nsp) = 0;  // temperature given densities and pressures of all species

  virtual void computeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp) {};
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
  // TODO: Fluxes class should take Up as input variable.
  // Currently cannot receive Up inside Fluxes class.
  virtual void computeMassFractionGradient(const Vector &state, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad){};
  virtual void computeMoleFractionGradient(const Vector &state, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad){};
  // // TODO: Compute pressure gradient from temperature gradient.
  // virtual void ComputePressureGradient(const Vector &state,
  //                                      const DenseMatrix &gradUp,
  //                                      DenseMatrix &PressureGrad) {};

  // Kevin: These are defined by derived classes.
  // TODO: need to discuss how they are used throughout the code.
  // For now, these are used to get mixture speicific heat ratio and mixture gas constant.
  virtual double GetSpecificHeatRatio() = 0;
  virtual double GetGasConstant() = 0;

  virtual void computeNumberDensities(const Vector &conservedState, Vector &n_sp) {};

  // virtual double GetViscosity(const Vector &state) = 0;
  // virtual double GetThermalConductivity(const Vector &state) = 0;
  //
  // virtual double GetSchmidtNum() = 0;
  // virtual double GetPrandtlNum() = 0;

  virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  //   double GetPrandtlNum() { return Pr; }
  //   double GetSchmidtNum() { return Sc; }

  // double GetViscMultiplyer() { return visc_mult; }
  // double GetBulkViscMultiplyer() { return bulk_visc_mult; }

  virtual double getMolarCV(int species) {};
  virtual double getMolarCP(int species) {};

  // BC related functions
  virtual void computeStagnationState(const Vector &stateIn, Vector &stagnationState);
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut){};
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false){};

  virtual double computeAmbipolarElectronNumberDensity(const double *n_sp) {};
  virtual double computeBackgroundMassDensity(const double &rho, const double *n_sp, double &n_e,
                                              bool isElectronComputed = false) {};

  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal, Vector &conservedState) {};
};

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

class DryAir : public GasMixture {
 private:
  double specific_heat_ratio;
  double gas_constant;

  // virtual void SetNumEquations();
 public:
  DryAir(RunConfiguration &_runfile, int _dim);
  DryAir();  // this will only be usefull to get air constants
  DryAir(int dim, int num_equation);

  ~DryAir() {}

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state, double &electronPressure);
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

  // virtual double GetViscosity(const Vector &state);
  // virtual double GetThermalConductivity(const Vector &state);
  //
  // virtual double GetSchmidtNum(){return Sc;};
  // virtual double GetPrandtlNum() { return Pr; }

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  // BC related functions
  virtual void computeStagnationState(const Vector &stateIn, Vector &stagnationState);
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);

  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal, Vector &conservedState);

  // GPU functions
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
  
  static MFEM_HOST_DEVICE void computeSpeciesEnthalpies_gpu(double *speciesEnthalpies, const int &numSpecies,
                                                            const int &thrd, const int &maxThreads ) {
    for (int sp = thrd; sp < numSpecies; sp += maxThreads) {
      speciesEnthalpies[sp] = 0.;
    }
  }
  
  static MFEM_HOST_DEVICE void GetPrimitivesFromConservatives_gpu(const double *state,
                                                                  double *primit,
                                                                  const Equations &eqSystem,
                                                                  const double &gamma,
                                                                  const double &Rgas,
                                                                  const int &num_equation,
                                                                  const int &dim,
                                                                  const int &thrd,
                                                                  const int &maxThreads ) {
    MFEM_SHARED double KE[3];
    // compute temperature
    if (thrd < dim) KE[thrd] = 0.5 * state[1 + thrd] * state[1 + thrd] / state[0];
    if (thrd == maxThreads - 1 && dim == 2) KE[2] = 0;
    MFEM_SYNC_THREAD;

    if (thrd == 0) primit[0] = state[0];
    if (thrd == 1) primit[1] = state[1] / state[0];
    if (thrd == 2) primit[2] = state[2] / state[0];
    if (thrd == 3 && dim == 3) primit[3] = state[3] / state[0];
    if (thrd == 1 + dim)
      primit[1+dim] = DryAir::temperature(&state[0], &KE[0], gamma, Rgas, dim, num_equation);
    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE)
      primit[num_equation - 1] = state[num_equation - 1] / state[0];
  }
  
  static MFEM_HOST_DEVICE void modifyEnergyForPressure_gpu(const double *stateIn,
                                                           double *stateOut,
                                                           const double &p,
                                                           const double &gamma,
                                                           const int &num_equation,
                                                           const int &dim,
                                                           const int &thrd,
                                                           const int &maxThreads
  ) {
    if (thrd < num_equation) stateOut[thrd] = stateIn[thrd];
    MFEM_SYNC_THREAD;
    
    double ke = 0.;
    for (int d = 0; d < dim; d++) ke += stateIn[1 + d] * stateIn[1 + d];
    ke *= 0.5 / stateIn[0];

    stateOut[1 + dim] = p / (gamma - 1.) + ke;
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
//   &num_equation); void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv, const int &dim, const
//   int &num_equation);
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
inline double DryAir::ComputePressure(const Vector &state, double &electronPressure) {
  electronPressure = 0.0;
  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) * (state[1 + dim] - 0.5 * den_vel2);
}

inline double DryAir::ComputeTemperature(const Vector &state) {
  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) / gas_constant * (state[1 + dim] - 0.5 * den_vel2) / state[0];
}

// // Sutherland's law
// inline double DryAir::GetViscosity(const Vector &state) {
//   double p = ComputePressure(state);
//   double temp = p/gas_constant/state[0];
//   return (1.458e-6 * visc_mult * pow(temp, 1.5) / (temp + 110.4));
// }

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
  TestBinaryAir(RunConfiguration &_runfile, int _dim);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~TestBinaryAir(){}

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state, double &electronPressure);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double ComputeTemperature(const Vector &state);
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / rho[0] / gas_constant; }

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state);

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual void ComputeSpeciesPrimitives(const Vector &conservedState, Vector &X_sp, Vector &Y_sp, Vector &n_sp);

  virtual void ComputeMassFractionGradient(const Vector &state, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void ComputeMoleFractionGradient(const Vector &state, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);

  virtual double GetSpecificHeatRatio() { return specific_heat_ratio; }
  virtual double GetGasConstant() { return gas_constant; }

    // GPU functions
#ifdef _GPU_

#endif
};

// additional functions inlined for speed...
inline double TestBinaryAir::ComputePressure(const Vector &state, double &electronPressure) {
  electronPressure = 0.0;
  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  return (specific_heat_ratio - 1.0) * (state[1 + dim] - 0.5 * den_vel2);
}

// additional functions inlined for speed...
inline double TestBinaryAir::ComputeTemperature(const Vector &state) {
  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) den_vel2 += state(d + 1) * state(d + 1);
  den_vel2 /= state[0];

  double rhoU = state[1 + dim] - 0.5 * den_vel2;
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
  PerfectMixture(RunConfiguration &_runfile, int _dim);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~PerfectMixture(){}

  virtual double getMolarCV(int species) { return molarCV_(species); }
  virtual const Vector getMolarCVs(){return molarCV_;}
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

  virtual double ComputePressure(const Vector &state, double &electronPressure);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double computePressureBase(const double *n_sp, const double n_e, const double n_B, const double T_h,
                                     const double T_e);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual double ComputeTemperature(const Vector &state);
  virtual void computeTemperaturesBase(const Vector &conservedState, const double *n_sp, const double n_e,
                                       const double n_B, double &T_h, double &T_e);

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);

  // TODO: Kevin - I don't think we should use this for boundary condition.
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / rho[0] / GetGasConstant(); };

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

  virtual void computeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void computeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);

  // functions needed for BCs
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);

  virtual void computeConservedStateFromConvectiveFlux(const Vector &meanNormalFluxes, const Vector &normal, Vector &conservedState);
  // GPU functions
#ifdef _GPU_
  static MFEM_HOST_DEVICE double computePressure_gpu(const double *state, 
                                                     double *electronPressure,
                                                     const double *gasParams,
                                                     const double *molarCV,
                                                     const int &num_equation,
                                                     const int &dim,
                                                     const int &numSpecies,
                                                     const int &numActiveSpecies,
                                                     const bool &ambipolar,
                                                     const bool &twoTemperature,
                                                     const int &thrd,
                                                     const int &maxTheads ){
    MFEM_SHARED double n_sp[15]; // WARNING: assuming a maximum of 15 species
    PerfectMixture::computeNumberDensities_gpu(state, &n_sp[0], gasParams, dim, numSpecies, numActiveSpecies, 
                                               ambipolar, thrd, maxTheads);
    double T_h, T_e;
    PerfectMixture::computeTemperatureBase_gpu(state,
                                               &n_sp[0],
                                               molarCV,
                                               gasParams,
                                               num_equation,
                                               dim,
                                               numSpecies,
                                               numActiveSpecies,
                                               n_sp[numSpecies-1],
                                               n_sp[numSpecies-2],
                                               twoTemperature,
                                               T_e,
                                               T_h );
    electronPressure[0] = n_sp[numSpecies - 2] * UNIVERSALGASCONSTANT * T_e;
    
    return PerfectMixture::computePressureBase_gpu(&n_sp[0],
                                                       n_sp[numSpecies-2],
                                                       n_sp[numSpecies-1],
                                                       T_h,
                                                       T_e,
						       twoTemperature,
                                                       numSpecies,
                                                       numActiveSpecies );
  }
  
  static MFEM_HOST_DEVICE double computePressureBase_gpu(const double *n_sp,
                                                         const double &n_e,
                                                         const double &n_B,
                                                         const double &T_h,
                                                         const double &T_e,
							 const bool &twoTemperature,
                                                         const int &numSpecies,
                                                         const int &numActiveSpecies ) {
    double n_h = 0.0;  // total number density of all heavy species.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (sp == numSpecies - 2) continue;  // treat electron separately.
      n_h += n_sp[sp];
    }
    n_h += n_B;

    double p = n_h * T_h;
    if (twoTemperature) {
      p += n_e * T_e;
    } else {
      p += n_e * T_h;
    }
    p *= UNIVERSALGASCONSTANT;

    return p;
  }
  
  static MFEM_HOST_DEVICE void computeTemperatureBase_gpu(const double *state,
                                                          const double *n_sp,
                                                          const double *molarCV,
                                                          const double *gasParams,
                                                          const int &num_equation,
                                                          const int &dim,
                                                          const int &numSpecies,
                                                          const int &numActiveSpecies,
                                                          const double &nb,
                                                          const double &ne,
                                                          const bool &twoTemperature,
                                                          double &T_e,
                                                          double &T_h ) {
    double totalHeatCapacity = 0.;
    // heavies
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (sp == numSpecies - 2) continue;  // neglect electron.
      totalHeatCapacity += n_sp[sp] * molarCV[sp];
    }
    totalHeatCapacity += nb * molarCV[numSpecies - 1];
    
    double totalEnergy = state[dim + 1];
    for (int sp = 0; sp < numSpecies - 2; sp++) {
      totalEnergy -= n_sp[sp] * gasParams[sp + numSpecies* int(GasParams::FORMATION_ENERGY)];
    }
    
    // Comptue heavy-species temperature. If not two temperature, then this works as the unique temperature.
    T_h = 0.0;
    for (int d = 0; d < dim; d++) T_h -= state[d + 1] * state[d + 1];
    T_h *= 0.5 / state[0];
    T_h += totalEnergy;
    if (twoTemperature) T_h -= state[num_equation - 1];
    T_h /= totalHeatCapacity;

    // electron temperature as primitive variable.
    if (twoTemperature) {
      T_e = state[num_equation - 1] / ne / molarCV[numSpecies - 2];
    } else {
      T_e = T_h;
    }
  }
  
  static MFEM_HOST_DEVICE void computeNumberDensities_gpu(const double *state, 
                                                          double *n_sp,
                                                          const double *gasParams,
                                                          const int &dim,
                                                          const int &numSpecies,
                                                          const int &numActiveSpecies,
                                                          const bool ambipolar,
                                                          const int &thrd,
                                                          const int &maxTheads ) {
    for (int sp = thrd; sp < numActiveSpecies; sp += maxTheads) {
      n_sp[sp] = state[dim+2+sp] / gasParams[sp + numSpecies * int(GasParams::SPECIES_MW)];
    }
    MFEM_SYNC_THREAD;
    
    // NOTE: all threads are computing n_e
    if (ambipolar) {
      double n_e = 0.;
      for (int sp = 0; sp < numActiveSpecies; sp++) 
        n_e += n_sp[sp] * gasParams[sp + numSpecies * int(GasParams::SPECIES_CHARGES)];
      n_sp[numSpecies - 2] = n_e;
    }
    
    double rhoB = PerfectMixture::computeBackgroundMassDensity_gpu(state,
                                                                  n_sp,
                                                                  gasParams,
                                                                  dim,
                                                                  numSpecies,
                                                                  numActiveSpecies,
                                                                  ambipolar,
                                                                  true );
    n_sp[numSpecies - 1] = rhoB / gasParams[numSpecies - 1 + numSpecies*int(GasParams::SPECIES_MW)];
  }
  
  static MFEM_HOST_DEVICE double computeBackgroundMassDensity_gpu(const double *state,
                                                                  const double *n_sp,
                                                                  const double *gasParams,
                                                                  const int &dim,
                                                                  const int &numSpecies,
                                                                  const int &numActiveSpecies,
                                                                  const bool &ambipolar,
                                                                  const bool isElectronComputed ){
    double ne = 0.;
    if (!isElectronComputed) {
      if (ambipolar) {
        for (int sp = 0; sp < numActiveSpecies; sp++) 
          ne += n_sp[sp] * gasParams[sp + numSpecies * int(GasParams::SPECIES_CHARGES)];
      }
    }else {
      ne = n_sp[numSpecies -1 ];
    }
    
    double rhoB = state[0];
    for (int sp = 0; sp < numActiveSpecies; sp++) 
      rhoB -= gasParams[sp + numSpecies * int(GasParams::SPECIES_MW)] * n_sp[sp];
    
    return rhoB;
  }
  
  static MFEM_HOST_DEVICE void GetPrimitivesFromConservatives_gpu(const double *state,
                                                                  double *primitives,
                                                                  const double *gasParams,
                                                                  const double *molarCV,
                                                                  const bool &ambipolar,
                                                                  const bool &twoTemperature,
                                                                  const int &num_equation,
                                                                  const int &dim,
                                                                  const int &numSpecies,
                                                                  const int &numActiveSpecies,
                                                                  const int &thrd,
                                                                  const int &maxThreads ) {
    MFEM_SHARED double n_sp[15]; // WARNING: assuming a maximum of 15 species
    
    PerfectMixture::computeNumberDensities_gpu(state, 
                                               n_sp,
                                               gasParams,
                                               dim,
                                               numSpecies,
                                               numActiveSpecies,
                                               ambipolar,
                                               thrd,
                                               maxThreads);
    MFEM_SYNC_THREAD;
    for (int sp = thrd; sp < numActiveSpecies; sp += maxThreads) 
      primitives[dim + 2 + sp] = n_sp[sp];
    
    if (thrd == maxThreads-1) primitives[0] = state[0];
    
    for (int d = thrd; d < dim; d += maxThreads) primitives[d + 1] = state[d + 1] / state[0];
    MFEM_SYNC_THREAD;
    
    double T_h, T_e;
    PerfectMixture::computeTemperatureBase_gpu(state,
                                               &n_sp[0],
                                               molarCV,
                                               gasParams,
                                               num_equation,
                                               dim,
                                               numSpecies,
                                               numActiveSpecies,
                                               n_sp[numSpecies-1],
                                               n_sp[numSpecies-2],
                                               twoTemperature,
                                               T_e,
                                               T_h );
    primitives[dim + 1] = T_h;

    if (twoTemperature)  // electron temperature as primitive variable.
      primitives[num_equation - 1] = T_e;
  }
  
  static MFEM_HOST_DEVICE void GetConservativesFromPrimitives_gpu(const double *primit,
                                                              double *conserv,
                                                              const double *gasParams,
                                                              const double *molarCV,
                                                              const double &ambipolar,
                                                              const double &twoTemperature,
                                                              const int &num_equation,
                                                              const int &dim,
                                                              const int &numSpecies,
                                                              const int &numActiveSpecies,
                                                              const int &thrd,
                                                              const int &maxThreads ) {
    if (thrd == maxThreads) conserv[0] = primit[0];
    for (int d = thrd; d < dim; d += maxThreads) conserv[d + 1] = primit[d + 1] * primit[0];

    // Convert species rhoY first.
    for (int sp = thrd; sp < numActiveSpecies; sp += maxThreads) {
      conserv[dim + 2 + sp] = primit[dim + 2 + sp] * gasParams[sp + numSpecies *(int)GasParams::SPECIES_MW];
    }
    MFEM_SYNC_THREAD;

    double n_e = 0.0;
    if (ambipolar) {
      for (int sp = 0; sp < numActiveSpecies; sp++)
        n_e += primit[2+dim+sp] * gasParams[sp + numSpecies * (int)GasParams::SPECIES_CHARGES];
//       n_e = computeAmbipolarElectronNumberDensity(&primit[dim + 2]);
    } else {
      n_e = primit[dim + 2 + numSpecies - 2];
    }
    double rhoB =  PerfectMixture::computeBackgroundMassDensity_gpu(conserv,
                                                                  &primit[2+dim],
                                                                  gasParams,
                                                                  dim,
                                                                  numSpecies,
                                                                  numActiveSpecies,
                                                                  ambipolar,
                                                                  true);
    double nB = rhoB / gasParams[numSpecies - 1 + numSpecies * (int)GasParams::SPECIES_MW];

    if (twoTemperature) conserv[num_equation - 1] = n_e * molarCV[numSpecies - 2] * primit[num_equation - 1];

    // compute mixture heat capacity.
    double totalHeatCapacity = 0.;
    // heavies
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (sp == numSpecies - 2) continue;  // neglect electron.
      totalHeatCapacity += primit[2+dim+sp] * molarCV[sp];
    }
    totalHeatCapacity += nB * molarCV[numSpecies - 1];
    if (!twoTemperature) totalHeatCapacity += n_e * molarCV[numSpecies - 2];

    double totalEnergy = 0.0;
    for (int d = 0; d < dim; d++) totalEnergy += primit[d + 1] * primit[d + 1];
    totalEnergy *= 0.5 * primit[0];
    totalEnergy += totalHeatCapacity * primit[dim + 1];
    if (twoTemperature) {
      totalEnergy += conserv[num_equation - 1];
    }

    for (int sp = 0; sp < numSpecies - 2; sp++) {
      totalEnergy += primit[dim + 2 + sp] * gasParams[sp + numSpecies * (int)GasParams::FORMATION_ENERGY];
    }

    conserv[dim + 1] = totalEnergy;
  }
  
  static MFEM_HOST_DEVICE void modifyEnergyForPressure_gpu(const double *stateIn,
                                                           double *stateOut,
                                                           const double &p,
                                                           const bool &modifyElectronEnergy,
                                                           const double *molarCV,
                                                           const double *gasParams,
                                                           const bool &ambipolar,
                                                           const bool &twoTemperature,
                                                           const int &num_equation,
                                                           const int &dim,
                                                           const int &numSpecies,
                                                           const int &numActiveSpecies,
                                                           const int &thrd,
                                                           const int &maxThreads
  ) {
    MFEM_SHARED double n_sp[15];
    PerfectMixture::computeNumberDensities_gpu(stateIn, 
                                               n_sp,
                                               gasParams,
                                               dim,
                                               numSpecies,
                                               numActiveSpecies,
                                               ambipolar,
                                               thrd,
                                               maxThreads);
    MFEM_SYNC_THREAD;
    
    double Th = 0., pe = 0.0;
    if (twoTemperature && (!modifyElectronEnergy)) {
      double Xeps = 1.0e-30; // To avoid dividing by zero.
      double Te = stateIn[num_equation - 1] / (n_sp[numSpecies - 2] + Xeps) / molarCV[numSpecies - 2];
      pe = n_sp[numSpecies - 2] * UNIVERSALGASCONSTANT * Te;
    }

    for (int sp = 0; sp < numSpecies; sp++) {
      if (twoTemperature && (!modifyElectronEnergy) && (sp == numSpecies - 2)) continue;
      Th += n_sp[sp];
    }
    Th = (p - pe) / (Th * UNIVERSALGASCONSTANT);
    
    // compute total energy with the modified temperature of heavies
    double totalHeatCapacity = 0.;
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (sp == numSpecies - 2) continue;  // neglect electron.
      totalHeatCapacity += n_sp[sp] * molarCV[sp];
    }
    totalHeatCapacity += n_sp[numSpecies-1] * molarCV[numSpecies - 1];
    if (!twoTemperature) totalHeatCapacity += n_sp[numSpecies-2] * molarCV[numSpecies - 2];
    
    // if (!twoTemperature_) totalHeatCapacity += n_sp[numSpecies - 2] * molarCV_(numSpecies - 2);
    double rE = totalHeatCapacity * Th;
    // if (twoTemperature_) rE += stateIn(num_equation - 1);

    double electronEnergy = 0.0;
    if (twoTemperature) {
      electronEnergy = (modifyElectronEnergy) ? n_sp[numSpecies - 2] * molarCV[numSpecies - 2] * Th : stateIn[num_equation - 1];
      stateOut[num_equation - 1] = electronEnergy;
    } else {
      electronEnergy = n_sp[numSpecies - 2] * molarCV[numSpecies - 2] * Th;
    }
    rE += electronEnergy;
    
    for (int d = 0; d < dim; d++) 
      rE += 0.5 * stateIn[d + 1] * stateIn[d + 1] / stateIn[0];
    
    for (int sp = 0; sp < numSpecies - 2; sp++) 
      rE += n_sp[sp] * gasParams[sp + numSpecies * (int)GasParams::FORMATION_ENERGY];

    stateOut[1 + dim] = rE;
  }
#endif
};

#endif  // EQUATION_OF_STATE_HPP_
