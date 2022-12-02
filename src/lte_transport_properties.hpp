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
#ifndef LTE_TRANSPORT_PROPERTIES_HPP_
#define LTE_TRANSPORT_PROPERTIES_HPP_

/** @file
 * Transport property calculations for an equilibrium mixture,
 * implemented as table look-ups as function of mixture density and
 * temperature.
 */

#include <tps_config.h>

#include "table.hpp"
#include "transport_properties.hpp"

#ifndef _GPU_  // this class only available for CPU currently

/** \brief Transport class assuming local thermodynamic equilibrium
 *
 * If the gas or plasma is in local thermodynamic equilibrium, then
 * the composition and all resulting transport properties of the
 * mixture may be computed offline and stored as look-up
 * (interpolation) tables as a function of two state parameters.
 * Here, tables are read and stored for all required properties in
 * terms of the mixture density (rho) and temperature (T).
 */
class LteTransport : public TransportProperties {
 protected:
  TableInterpolator2D *mu_table_;     // dynamic viscosity
  TableInterpolator2D *kappa_table_;  // thermal conductivity
  TableInterpolator2D *sigma_table_;  // electrical conductivity

 public:
  LteTransport(GasMixture *_mixture, RunConfiguration &_runfile);

  virtual ~LteTransport();

  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp, const double *Efield,
                                              double *transportBuffer, double *diffusionVelocity) {
    mfem_error("This variant of LteTransport::ComputeFluxTransportProperties is not implemented\n");
  }
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);
  virtual void ComputeSourceTransportProperties(const double *state, const double *Up, const double *gradUp,
                                                const double *Efield, double *globalTransport, double *speciesTransport,
                                                double *diffusionVelocity, double *n_sp) {
    mfem_error("This variant of LteTransport::ComputeSourceTransportProperties is not implemented\n");
  }

  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc);
};

#endif  // _GPU_
#endif  // LTE_TRANSPORT_PROPERTIES_HPP_
