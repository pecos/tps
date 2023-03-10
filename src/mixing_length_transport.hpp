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
#ifndef MIXING_LENGTH_TRANSPORT_HPP_
#define MIXING_LENGTH_TRANSPORT_HPP_

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace std;
using namespace mfem;

class MixingLengthTransport : public TransportProperties {
 protected:
  // for turbulent transport calculations
  const double max_mixing_length_;  // user-specifed maximum mixing length
  const double Prt_;                // eddy Prandtl number
  const double Let_;                // eddy Lewis number

  // for molecular transport (owned)
  TransportProperties *molecular_transport_;

 public:
  MixingLengthTransport(GasMixture *mix, RunConfiguration &runfile, TransportProperties *molecular_transport);
  MFEM_HOST_DEVICE MixingLengthTransport(GasMixture *mix, const mixingLengthTransportData &inputs,
                                         TransportProperties *molecular_transport);

  MFEM_HOST_DEVICE virtual ~MixingLengthTransport() { delete molecular_transport_; }

  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              double radius, double distance, Vector &transportBuffer,
                                              DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double radius, double distance,
                                                               double *transportBuffer, double *diffusionVelocity);
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, double distance, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);
  MFEM_HOST_DEVICE virtual void ComputeSourceTransportProperties(const double *state, const double *Up,
                                                                 const double *gradUp, const double *Efield,
                                                                 double distance, double *globalTransport,
                                                                 double *speciesTransport, double *diffusionVelocity,
                                                                 double *n_sp);

  MFEM_HOST_DEVICE void GetViscosities(const double *conserved, const double *primitive, double *visc) override;
  MFEM_HOST_DEVICE void GetViscosities(const double *conserved, const double *primitive, const double *gradUp,
                                       double radius, double distance, double *visc) override;
};

MFEM_HOST_DEVICE inline void MixingLengthTransport::GetViscosities(const double *conserved, const double *primitive,
                                                                   double *visc) {
  assert(false);
}

MFEM_HOST_DEVICE inline void MixingLengthTransport::GetViscosities(const double *conserved, const double *primitive,
                                                                   const double *gradUp, double radius, double distance,
                                                                   double *visc) {
  molecular_transport_->GetViscosities(conserved, primitive, visc);

  const double rho = conserved[0];

  // Compute divergence
  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    divV += gradUp[(1 + i) + i * num_equation];
  }

  // If axisymmetric
  double ur = 0;
  if (nvel_ != dim) {
    ur = primitive[1];
    if (radius > 0) divV += ur / radius;
  }

  // eddy viscosity
  double S = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double ui_xj = gradUp[(1 + i) + j * num_equation];
      const double uj_xi = gradUp[(1 + j) + i * num_equation];
      double Sij = 0.5 * (ui_xj + uj_xi);
      if (i == j) Sij -= divV / 3.;
      S += 2 * Sij * Sij;
    }
  }

  // If axisymmetric, update S to account for utheta contributions
  if (nvel_ != dim) {
    const double ut = primitive[3];
    const double ut_r = gradUp[3 + 0 * num_equation];
    const double ut_z = gradUp[3 + 1 * num_equation];

    double Szx = 0.5 * ut_r;
    if (radius > 0) Szx -= 0.5 * ut / radius;
    const double Szy = 0.5 * ut_z;
    double Szz = - divV / 3.;
    if (radius > 0) Szz += ur / radius;

    S += 2 * (2 * Szx * Szx + 2 * Szy * Szy + Szz * Szz);
  }

  S = sqrt(S);

  const double mixing_length = std::min(0.41 * distance, max_mixing_length_);
  const double mut = rho * mixing_length * mixing_length * S;

  visc[0] += mut;
}


#endif  // MIXING_LENGTH_TRANSPORT_HPP_
