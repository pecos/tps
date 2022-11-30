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
#ifndef LTE_MIXTURE_HPP_
#define LTE_MIXTURE_HPP_

/** @file
 * Provides themodynamic properties are related functions for a gas or
 * plasma in local thermodynamic equilibrium.
 */

#include <tps_config.h>

#include "equation_of_state.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

#ifndef _GPU_  // this class only available for CPU currently

/** \brief Mixture class assuming local thermodynamic equilibrium
 *
 * If the gas or plasma is in local thermodynamic equilibrium, then
 * the composition and all resulting thermodynamic properties of the
 * mixture may be computed offline and stored as look-up
 * (interpolation) tables as a function of two state parameters.
 * Here, tables are read and
 * stored for all required properties in terms of the mixture density
 * (rho) and temperature (T).
 */
class LteMixture : public GasMixture {
 private:
  TableInterpolator2D *energy_table_;
  TableInterpolator2D *R_table_;
  TableInterpolator2D *c_table_;

  // if generate_e_to_T is true at construction, generate this table,
  // which maps (rho, energy) to temperature.  This table is used to
  // get a good IC for the Newton solve when evaluating the
  // temperature from the converved variables.
  TableInterpolator2D *T_table_;

 public:
  LteMixture(RunConfiguration &_runfile, int _dim, int nvel, bool generate_e_to_T = false);
  virtual ~LteMixture();

  double evaluateInternalEnergy(const double &T, const double &rho) { return energy_table_->eval(T, rho); }
  double evaluateGasConstant(const double &T, const double &rho) { return R_table_->eval(T, rho); }

  virtual double ComputePressure(const Vector &state, double *electronPressure = NULL);

  virtual double ComputePressureFromPrimitives(const Vector &Up);

  virtual double ComputeTemperature(const Vector &state);

  double ComputeTemperatureFromDensityPressure(const double rho, const double p);

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);
  virtual void computeSpeciesEnthalpies(const double *state, double *speciesEnthalpies) {
    for (int sp = 0; sp < numSpecies; sp++) speciesEnthalpies[sp] = 0.0;
    return;
  }

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);

  virtual double ComputeMaxCharSpeed(const Vector &state);

  /// only used in non-reflecting BCs... don't implement for now
  virtual double ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive = true);

  // Physicality check (at end)
  virtual bool StateIsPhysical(const Vector &state);

  virtual void ComputeMassFractionGradient(const double rho, const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &massFractionGrad) {
    mfem_error("computeMassFractionGradient not implemented");
  }
  virtual void ComputeMoleFractionGradient(const Vector &numberDensities, const DenseMatrix &gradUp,
                                           DenseMatrix &moleFractionGrad) {
    mfem_error("computeMoleFractionGradient not implemented");
  }
  virtual void ComputeMoleFractionGradient(const double *numberDensities, const double *gradUp,
                                           double *moleFractionGrad) {
    printf("computeMoleFractionGradient not implemented");
    assert(false);
  }

  virtual double GetSpecificHeatRatio() { return 0; }
  virtual double GetGasConstant() { return 0; }

  // BC related functions
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);
  virtual void computeSheathBdrFlux(const Vector &state, BoundaryViscousFluxData &bcFlux) {
    mfem_error("computeSheathBdrFlux not implemented");
  }

  virtual double computeElectronEnergy(const double n_e, const double T_e) {
    mfem_error("computeElectronEnergy not implemented");
    return 0;
  }
  virtual double computeElectronPressure(const double n_e, const double T_e) {
    mfem_error("computeElectronPressure not implemented");
    return 0;
  }
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const DenseMatrix &gradUp,
                                           Vector &gradPe) {
    mfem_error("computeElectronPressureGrad not implemented");
  }
  virtual void computeElectronPressureGrad(const double n_e, const double T_e, const double *gradUp, double *gradPe) {
    printf("computeElectronPressureGrad not implemented");
    assert(false);
  }
};

#endif  // _GPU_
#endif  // LTE_MIXTURE_HPP_
