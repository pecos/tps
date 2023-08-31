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

#include "lte_transport_properties.hpp"

#ifndef _GPU_

LteTransport::LteTransport(GasMixture *_mixture, RunConfiguration &_runfile) : MolecularTransport(_mixture) {
#ifdef HAVE_GSL
  mu_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.trans_file_name, 0, /* temperature column */
                                         1,                                           /* density column */
                                         2,                                           /* viscosity column */
                                         5 /* total number of columns */);

  kappa_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.trans_file_name, 0, /* temperature column */
                                            1,                                           /* density column */
                                            3, /* thermal conductivity column */
                                            5 /* total number of columns */);

  sigma_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.trans_file_name, 0, /* temperature column */
                                            1,                                           /* density column */
                                            4, /* electrical conductivity column */
                                            5 /* total number of columns */);
#else
  mu_table_ = NULL;
  kappa_table_ = NULL;
  sigma_table_ = NULL;
  mfem_error("2D LTE transport tables require GSL support.");
#endif
}

LteTransport::LteTransport(GasMixture *_mixture, TableInput mu_table_input, TableInput kappa_table_input,
                           TableInput sigma_table_input)
    : MolecularTransport(_mixture) {
  mu_table_ = new LinearTable(mu_table_input);
  kappa_table_ = new LinearTable(kappa_table_input);
  sigma_table_ = new LinearTable(sigma_table_input);
}

LteTransport::~LteTransport() {
  delete sigma_table_;
  delete kappa_table_;
  delete mu_table_;
}

void LteTransport::ComputeFluxMolecularTransport(const double *state, const double *gradUp, const double *Efield,
                                                 double *transportBuffer, double *diffusionVelocity) {
  const double rho = state[0];
  const double T = mixture->ComputeTemperature(state);

  transportBuffer[FluxTrns::VISCOSITY] = mu_table_->eval(T, rho);
  transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;  // bulk_visc_mult * viscosity;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = kappa_table_->eval(T, rho);
  transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = 0.0;  // electron conductivity already accounted for
}

void LteTransport::ComputeSourceMolecularTransport(const double *state, const double *Up, const double *gradUp,
                                                   const double *Efield, double *globalTransport,
                                                   double *speciesTransport, double *diffusionVelocity, double *n_sp) {
  const double rho = Up[0];
  const double T = Up[1 + nvel_];
  double sigma = sigma_table_->eval(T, rho);

  if (sigma < 1.0) {
    sigma = 1.0;
  }
  globalTransport[SrcTrns::ELECTRIC_CONDUCTIVITY] = sigma;
}

void LteTransport::GetViscosities(const double *conserved, const double *primitive, double *visc) {
  const double rho = primitive[0];
  const double T = primitive[1 + nvel_];

  visc[0] = mu_table_->eval(T, rho);
  visc[1] = 0.;
}

#endif  // _GPU_
