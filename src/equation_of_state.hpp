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
static const double UNIVERSALGASCONSTANT        = 8.3144598;
static const double AVOGADRONUMBER              = 6.0221409e+23;

class GasMixture{
protected:
  WorkingFluid fluid;
  int num_equation;
  int dim;

  int numSpecies;
  int numActiveSpecies;
  bool ambipolar;
  bool twoTemperature;

  Vector gasParams;

  SpeciesPrimitiveType speciesPrimitiveType;

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

  double GetGasParams(int species, GasParams param) { return gasParams[param*numSpecies + species]; }

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

  virtual double GetSpecificHeatRatio() = 0;
  virtual double GetGasConstant() = 0;

  virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up) = 0;
};

class DryAir : public GasMixture {
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


  virtual void UpdatePressureGridFunction(ParGridFunction *press, const ParGridFunction *Up);

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

#endif  // EQUATION_OF_STATE_HPP_
