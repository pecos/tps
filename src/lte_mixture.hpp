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
 * Provides themodynamic properties and related functions for a gas or
 * plasma in local thermodynamic equilibrium.
 */

#include <tps_config.h>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

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
#ifdef _GPU_
  // On the device path, we only support LinearTable
  LinearTable energy_table_;
  LinearTable R_table_;
  LinearTable c_table_;
  LinearTable T_table_;
#else
  // On the cpu path, simultaneously support LinearTable and GslTableInterpolator2D
  TableInterface *energy_table_;
  TableInterface *R_table_;
  TableInterface *c_table_;
  TableInterface *T_table_;
#endif

  MFEM_HOST_DEVICE bool ComputeTemperatureInternal(const double *state, double &T);

 public:
  MFEM_HOST_DEVICE LteMixture(WorkingFluid f, int _dim, int nvel, double pc, TableInput energy_table_input,
                              TableInput R_table_input, TableInput c_table_input, TableInput T_table_input);
#ifndef _GPU_
  LteMixture(RunConfiguration &_runfile, int _dim, int nvel);
#endif

  MFEM_HOST_DEVICE virtual ~LteMixture();

  MFEM_HOST_DEVICE double evaluateInternalEnergy(const double &T, const double &rho) {
#ifdef _GPU_
    return energy_table_.eval(T, rho);
#else
    return energy_table_->eval(T, rho);
#endif
  }
  MFEM_HOST_DEVICE double evaluateGasConstant(const double &T, const double &rho) {
#ifdef _GPU_
    return R_table_.eval(T, rho);
#else
    return R_table_->eval(T, rho);
#endif
  }

  virtual double ComputePressure(const Vector &state, double *electronPressure = NULL);
  MFEM_HOST_DEVICE virtual double ComputePressure(const double *state, double *electronPressure = NULL);

  virtual double ComputePressureFromPrimitives(const Vector &Up);
  MFEM_HOST_DEVICE virtual double ComputePressureFromPrimitives(const double *Up);

  virtual double ComputeTemperature(const Vector &state);
  MFEM_HOST_DEVICE virtual double ComputeTemperature(const double *state);

  MFEM_HOST_DEVICE double ComputeTemperatureFromDensityPressure(const double rho, const double p);

  virtual void computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies);
  MFEM_HOST_DEVICE virtual void computeSpeciesEnthalpies(const double *state, double *speciesEnthalpies) {
    for (int sp = 0; sp < numSpecies; sp++) speciesEnthalpies[sp] = 0.0;
    return;
  }

  virtual void GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit);
  MFEM_HOST_DEVICE virtual void GetPrimitivesFromConservatives(const double *conserv, double *primit);

  virtual void GetConservativesFromPrimitives(const Vector &primit, Vector &conserv);
  MFEM_HOST_DEVICE virtual void GetConservativesFromPrimitives(const double *primit, double *conserv);

  virtual double ComputeSpeedOfSound(const Vector &Uin, bool primitive = true);
  MFEM_HOST_DEVICE virtual double ComputeSpeedOfSound(const double *Uin, bool primitive = true);

  virtual double ComputeMaxCharSpeed(const Vector &state);
  MFEM_HOST_DEVICE virtual double ComputeMaxCharSpeed(const double *state);

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
  MFEM_HOST_DEVICE virtual void ComputeMoleFractionGradient(const double *numberDensities, const double *gradUp,
                                                            double *moleFractionGrad) {
    printf("computeMoleFractionGradient not implemented");
    assert(false);
  }

  MFEM_HOST_DEVICE virtual double GetSpecificHeatRatio() { return 0; }
  MFEM_HOST_DEVICE virtual double GetGasConstant() { return 0; }

  // BC related functions
  virtual void computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut);
  MFEM_HOST_DEVICE virtual void computeStagnantStateWithTemp(const double *stateIn, const double Temp,
                                                             double *stateOut);

  virtual void modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                       bool modifyElectronEnergy = false);

  MFEM_HOST_DEVICE virtual void modifyEnergyForPressure(const double *stateIn, double *stateOut, const double &p,
                                                        bool modifyElectronEnergy = false);

  virtual void computeSheathBdrFlux(const Vector &state, BoundaryViscousFluxData &bcFlux) {
    mfem_error("computeSheathBdrFlux not implemented");
  }

  MFEM_HOST_DEVICE virtual void computeSheathBdrFlux(const double *state, BoundaryViscousFluxData &bcFlux) {
    printf("ERROR: computeSheathBdrFlux is not supposed to be executed for LteMixture!");
    return;
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
  MFEM_HOST_DEVICE virtual void computeElectronPressureGrad(const double n_e, const double T_e, const double *gradUp,
                                                            double *gradPe) {
    printf("computeElectronPressureGrad not implemented");
    assert(false);
  }
};

#endif  // LTE_MIXTURE_HPP_
