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
                                             TransportProperties *molecular_transport)
    : MixingLengthTransport(mix, runfile.mix_length_trans_input_, molecular_transport) {}

MFEM_HOST_DEVICE MixingLengthTransport::MixingLengthTransport(GasMixture *mix, const mixingLengthTransportData &inputs,
                                                              TransportProperties *molecular_transport)
    : TransportProperties(mix),
      max_mixing_length_(inputs.max_mixing_length_),
      Prt_(inputs.Prt_),
      Let_(inputs.Let_),
      molecular_transport_(molecular_transport) {}

void MixingLengthTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                           const Vector &Efield, double distance,
                                                           Vector &transportBuffer, DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  diffusionVelocity.SetSize(numSpecies, nvel_);
  ComputeFluxTransportProperties(&state[0], gradUp.Read(), &Efield[0], distance, &transportBuffer[0],
                                 diffusionVelocity.Write());
}

MFEM_HOST_DEVICE void MixingLengthTransport::ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                                            const double *Efield, double distance,
                                                                            double *transportBuffer,
                                                                            double *diffusionVelocity) {
  molecular_transport_->ComputeFluxTransportProperties(state, gradUp, Efield, distance, transportBuffer,
                                                       diffusionVelocity);

  const double cp_over_Pr =
      transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] / transportBuffer[FluxTrns::VISCOSITY];

  // Add mixing length model results to computed molecular transport
  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  const double rho = state[0];
  const double Th = primitiveState[nvel_ + 1];

  // eddy viscosity
  double S = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double u_x = gradUp[(1 + i) + j * num_equation];
      S += 2 * u_x * u_x;  // todo: subtract divergence part
    }
  }
  S = sqrt(S);

  const double mixing_length = std::max(0.41 * distance, max_mixing_length_);
  const double mut = rho * mixing_length * mixing_length * S;

  transportBuffer[FluxTrns::VISCOSITY] += mut;

  // eddy thermal conductivity
  const double Pr_over_Prt = Prt_;  // FIXME: change varaible name
  const double kappat = mut * cp_over_Pr * Pr_over_Prt;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] += kappat;
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
  molecular_transport_->ComputeSourceTransportProperties(state, Up, gradUp, Efield, distance, globalTransport,
                                                         speciesTransport, diffusionVelocity, n_sp);

  // Add mixing length model results to computed molecular transport
  //
  // for (int i = 0; i < SrcTrns::NUM_SRC_TRANS; i++) globalTransport[i] = 0.0;
  // for (int c = 0; c < SpeciesTrns::NUM_SPECIES_COEFFS; c++)
  //   for (int sp = 0; sp < numSpecies; sp++) speciesTransport[sp + c * numSpecies] = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) n_sp[sp] = 0.0;

  // // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // // diffusionVelocity.SetSize(numSpecies, nvel_);
  // for (int v = 0; v < nvel_; v++)
  //   for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;

  // double primitiveState[gpudata::MAXEQUATIONS];
  // mixture->GetPrimitivesFromConservatives(state, primitiveState);

  // const double rho = state[0];
  // const double Th = primitiveState[nvel_ + 1];

  // // eddy viscosity
  // double S = 0;
  // for (int i = 0; i < dim; i++) {
  //   for (int j = 0; j < dim; j++) {
  //     const double u_x = gradUp[(1 + i) + j * num_equation];
  //     S += 2 * u_x * u_x; // todo: subtract divergence part
  //   }
  // }
  // S = sqrt(S);

  // //const double mut = rho * mixing_length_ * mixing_length_ * S;

  // // eddy thermal conductivity
  // //const double kappat = mut * cp / Prt_;

  // // TODO(trevilo): Call molecular transport
}
