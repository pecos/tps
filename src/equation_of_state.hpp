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

#include <tps_config.h>

#include <mfem.hpp>

#include "dataStructures.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

// Kevin: maybe use namespace?
static const int MAXSPECIES = 200;
static const double UNIVERSALGASCONSTANT        = 8.3144598; // J * mol^(-1) * K^(-1)
static const double AVOGADRONUMBER              = 6.0221409e+23; // mol^(-1)

class GasMixture{
protected:
  WorkingFluid fluid;
  int num_equation;
  int dim;

  int numSpecies;
  int numActiveSpecies;
  bool ambipolar;
  bool twoTemperature;

  DenseMatrix gasParams;

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
  void SetNumEquations() { num_equation = twoTemperature ? (dim + 3 + numActiveSpecies) : (dim + 2 + numActiveSpecies); }
public:
  GasMixture(WorkingFluid _fluid, int _dim);
  GasMixture(){};

  ~GasMixture(){};

  void SetFluid(WorkingFluid _fluid);

  int GetNumSpecies() { return numSpecies; }
  int GetNumActiveSpecies() { return numActiveSpecies; }
  int GetNumEquations() { return num_equation; }
  int GetDimension() { return dim; }
  bool IsAmbipolar() { return ambipolar; }
  bool IsTwoTemperature() { return twoTemperature; }

  double GetGasParams(int species, GasParams param) { return gasParams(species,param); }

  int GetNumConservativeVariables() { return Nconservative; }
  int GetNumPrimitiveVariables() { return Nprimitive; }

  virtual double ComputePressure(const Vector &state) = 0;             // pressure from conservatives
  virtual double ComputePressureFromPrimitives(const Vector &Up) = 0;  // pressure from primitive variables
  virtual double ComputeTemperature(const Vector &state) = 0;
  virtual double Temperature(double *rho, double *p,
                             int nsp) = 0;  // temperature given densities and pressures of all species

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
  virtual void ComputeMassFractionGradient(const Vector &state,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad) {};
  virtual void ComputeMoleFractionGradient(const Vector &state,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad) {};
  // TODO: Compute pressure gradient from temperature gradient.
  virtual void ComputePressureGradient(const Vector &state,
                                       const DenseMatrix &gradUp,
                                       DenseMatrix &PressureGrad) {};

  // TODO: Need to remove these and fix wherever they are used.
  // We cannot use these for multi species (heat ratio of which species?)
  // These are used in forcingTerm, Fluxes ASSUMING that the fluid is single species.
  virtual double GetSpecificHeatRatio() = 0;
  virtual double GetGasConstant() { return UNIVERSALGASCONSTANT / gasParams(0,GasParams::SPECIES_MW); }

  virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

//   double GetPrandtlNum() { return Pr; }
//   double GetSchmidtNum() { return Sc; }

  // double GetViscMultiplyer() { return visc_mult; }
  // double GetBulkViscMultiplyer() { return bulk_visc_mult; }
};

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

class DryAir : public GasMixture{
private:
  double specific_heat_ratio;
  double gas_constant;

  virtual void setNumEquations();

 public:
  DryAir(RunConfiguration &_runfile, int _dim);
  DryAir();  // this will only be usefull to get air constants
  DryAir(int dim, int num_equation);

  ~DryAir() {}

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double ComputeTemperature(const Vector &state);
  virtual double Temperature(double *rho, double *p, int nsp = 1) { return p[0] / gas_constant / rho[0]; }

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


  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

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
inline double DryAir::ComputePressure(const Vector &state) {
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

//////////////////////////////////////////////////////
//////// Test Binary Air mixture
//////////////////////////////////////////////////////
// NOTE: this is mixture of two idential air species.
// mixture variables (density, pressure, temperature) are treated as single species.
// Only mass fractions are treated as binary mixture, essentially the same as PASSIVE_SCALAR.

class TestBinaryAir : public GasMixture{
private:
  double specific_heat_ratio;
  double gas_constant;

  // virtual void SetNumEquations();
public:
  TestBinaryAir(RunConfiguration &_runfile, int _dim);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~TestBinaryAir(){};

  // implementation virtual methods
  virtual double ComputePressure(const Vector &state);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double ComputeTemperature(const Vector &state);
  virtual double Temperature(double *rho, double *p, int nsp = 1){return p[0] / rho[0] / gas_constant;};

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);

  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv,
                                              Vector &primit );
  virtual void GetConservativesFromPrimitives(const Vector &primit,
                                              Vector &conserv);

  // Compute the maximum characteristic speed.
  virtual double ComputeMaxCharSpeed(const Vector &state);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual void ComputeSpeciesPrimitives(const Vector &conservedState,
                                        Vector &X_sp, Vector &Y_sp, Vector &n_sp);

  virtual void ComputeMassFractionGradient(const Vector &state,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void ComputeMoleFractionGradient(const Vector &state,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);

  virtual double GetSpecificHeatRatio(){return specific_heat_ratio;};
  virtual double GetGasConstant(){return gas_constant;};

  // virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

    // GPU functions
#ifdef _GPU_

#endif
};

// additional functions inlined for speed...
inline double TestBinaryAir::ComputePressure(const Vector &state) {
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

class PerfectMixture : public GasMixture{
private:

  Vector specificHeatRatios_;
  Vector specificGasConstants_;
  Vector molarCV_;
  Vector molarCP_;

  // virtual void SetNumEquations();
public:
  PerfectMixture(RunConfiguration &_runfile, int _dim);
  // TestBinaryAir(); //this will only be usefull to get air constants

  ~PerfectMixture(){};

  virtual double getMolarCV(int species) { return molarCV_(species); };
  virtual double getMolarCP(int species) { return molarCP_(species); };
  virtual double getSpecificHeatRatio(int species) { return specificHeatRatios_(species); };
  virtual double getSpecificGasConstant(int species) { return specificGasConstants_(species); };

  // Kevin: these are mixture heat ratio and gas constant. need to change argument.
  virtual double GetSpecificHeatRatio() { return molarCP_(numSpecies-1) / molarCV_(numSpecies-1); };
  virtual double GetGasConstant() { return specificGasConstants_(numSpecies-1); };

  virtual double computeHeaviesHeatCapacity(const double *n_sp, const double &nB);
  virtual double computeAmbipolarElectronNumberDensity(const double *n_sp);
  virtual double computeBackgroundMassDensity(const double &rho, const double *n_sp,
                                              double &n_e, bool isElectronComputed = false);

  virtual void GetPrimitivesFromConservatives(const Vector &conserv,
                                              Vector &primit );
  virtual void GetConservativesFromPrimitives(const Vector &primit,
                                              Vector &conserv);

  virtual void computeSpeciesPrimitives(const Vector &conservedState,
                                        Vector &X_sp, Vector &Y_sp, Vector &n_sp);
  virtual void computeNumberDensities(const Vector &conservedState, Vector &n_sp);

  virtual double ComputePressure(const Vector &state);
  virtual double ComputePressureFromPrimitives(const Vector &Up);
  virtual double computePressureBase(const double *n_sp, const double n_e, const double n_B,
                                     const double T_h, const double T_e);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual double ComputeTemperature(const Vector &state);
  virtual void computeTemperaturesBase(const Vector &conservedState,
                                       const double *n_sp, const double n_e, const double n_B,
                                       double &T_h, double &T_e);

  // TODO: Kevin - I don't think we should use this for boundary condition.
  virtual double Temperature(double *rho, double *p, int nsp = 1){return p[0] / rho[0] / GetGasConstant();};

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

  virtual void computeMassFractionGradient(const double rho, const Vector &numberDensities,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad);
  virtual void computeMoleFractionGradient(const Vector &numberDensities,
                                           const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad);


    // GPU functions
#ifdef _GPU_

#endif
};

#endif  // EQUATION_OF_STATE_HPP_
