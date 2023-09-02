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

#include "mixing_length_transport.hpp"

#include <math.h>

using namespace std;
using namespace mfem;

MixingLengthTransport::MixingLengthTransport(GasMixture *mix, RunConfiguration &runfile,
                                             MolecularTransport *molecular_transport)
    : MixingLengthTransport(mix, runfile.mix_length_trans_input_, molecular_transport) {}

MFEM_HOST_DEVICE MixingLengthTransport::MixingLengthTransport(GasMixture *mix, const mixingLengthTransportData &inputs,
                                                              MolecularTransport *molecular_transport)
    : TransportProperties(mix),
      max_mixing_length_(inputs.max_mixing_length_),
      Prt_(inputs.Prt_),
      Let_(inputs.Let_),
      bulk_mult_(inputs.bulk_multiplier_),
      molecular_transport_(molecular_transport) {}

void MixingLengthTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                           const Vector &Efield, double radius, double distance,
                                                           Vector &transportBuffer, DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  diffusionVelocity.SetSize(numSpecies, nvel_);
  ComputeFluxTransportProperties(&state[0], gradUp.Read(), &Efield[0], radius, distance, &transportBuffer[0],
                                 diffusionVelocity.Write());
}

MFEM_HOST_DEVICE void MixingLengthTransport::ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                                            const double *Efield, double radius,
                                                                            double distance, double *transportBuffer,
                                                                            double *diffusionVelocity) {
  molecular_transport_->ComputeFluxMolecularTransport(state, gradUp, Efield, transportBuffer, diffusionVelocity);

  const double cp_over_Pr =
      transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] / transportBuffer[FluxTrns::VISCOSITY];

  const double kappa = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  const double mu = transportBuffer[FluxTrns::VISCOSITY];

  // Add mixing length model results to computed molecular transport
  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  const double rho = state[0];
  // const double Th = primitiveState[nvel_ + 1];

  // Compute divergence
  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    divV += gradUp[(1 + i) + i * num_equation];
  }

  // If axisymmetric
  double ur = 0;
  if (nvel_ != dim) {
    ur = primitiveState[1];
    if (radius > 0) divV += ur / radius;
  }

  // eddy viscosity
  double S = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double ui_xj = gradUp[(1 + i) + j * num_equation];
      const double uj_xi = gradUp[(1 + j) + i * num_equation];
      double Sij = 0.5 * (ui_xj + uj_xi);
      S += 2 * Sij * Sij;
    }
  }

  // If axisymmetric, update S to account for utheta contributions
  if (nvel_ != dim) {
    const double ut = primitiveState[3];
    const double ut_r = gradUp[3 + 0 * num_equation];
    const double ut_z = gradUp[3 + 1 * num_equation];

    double Szx = 0.5 * ut_r;
    if (radius > 0) Szx -= 0.5 * ut / radius;
    const double Szy = 0.5 * ut_z;
    double Szz = 0.0;
    if (radius > 0) Szz += ur / radius;

    S += 2 * (2 * Szx * Szx + 2 * Szy * Szy + Szz * Szz);
  }

  S = sqrt(S);

  double mixing_length = 0.41 * distance;
  if (mixing_length > max_mixing_length_) {
    mixing_length = max_mixing_length_;
  }
  double mut = rho * mixing_length * mixing_length * S;

  transportBuffer[FluxTrns::VISCOSITY] += mut;
  transportBuffer[FluxTrns::BULK_VISCOSITY] += bulk_mult_ * mut;

  // eddy thermal conductivity
  const double Pr_over_Prt = Prt_;  // FIXME: change varaible name
  const double kappat = mut * cp_over_Pr * Pr_over_Prt;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] += kappat;

  // TODO(trevilo): Deal with species diffusivities
}

void MixingLengthTransport::ComputeSourceTransportProperties(const Vector &state, const Vector &Up,
                                                             const DenseMatrix &gradUp, const Vector &Efield,
                                                             double distance, Vector &globalTransport,
                                                             DenseMatrix &speciesTransport,
                                                             DenseMatrix &diffusionVelocity, Vector &n_sp) {
  globalTransport.SetSize(SrcTrns::NUM_SRC_TRANS);
  speciesTransport.SetSize(numSpecies, SpeciesTrns::NUM_SPECIES_COEFFS);
  n_sp.SetSize(numSpecies);
  diffusionVelocity.SetSize(numSpecies, nvel_);
  ComputeSourceTransportProperties(&state[0], &Up[0], gradUp.Read(), &Efield[0], distance, &globalTransport[0],
                                   speciesTransport.Write(), diffusionVelocity.Write(), &n_sp[0]);

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      if (std::isnan(diffusionVelocity(sp, d))) {
        grvy_printf(GRVY_ERROR, "\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity(sp, d));
        exit(-1);
      }
    }
  }
}

MFEM_HOST_DEVICE void MixingLengthTransport::ComputeSourceTransportProperties(
    const double *state, const double *Up, const double *gradUp, const double *Efield, double distance,
    double *globalTransport, double *speciesTransport, double *diffusionVelocity, double *n_sp) {
  molecular_transport_->ComputeSourceMolecularTransport(state, Up, gradUp, Efield, globalTransport, speciesTransport,
                                                        diffusionVelocity, n_sp);
}
