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

#include <math.h>
#include "mixing_length_transport.hpp"

using namespace std;
using namespace mfem;

MixingLengthTransport::MixingLengthTransport(GasMixture *mix, RunConfiguration &runfile)
    : MixingLengthTransport(mix, runfile.mix_length_trans_input_) {}

MFEM_HOST_DEVICE MixingLengthTransport::MixingLengthTransport(GasMixture *mix, const mixingLengthTransportData &inputs)
    : TransportProperties(mix), mu0_(inputs.mu0_), T0_(inputs.T0_), visc_power_(inputs.visc_power_), Pr_(inputs.Pr_),
      Le_(inputs.Le_), nDe_(inputs.nDe_), mixing_length_(inputs.mixing_length_), Prt_(inputs.Prt_), Let_(inputs.Let_) {
  electronIndex_ = inputs.electronIndex_;
  if (mixture->IsTwoTemperature()) {
    if (electronIndex_ < 0) {
      printf("\nMixing length transport: two-temperature plasma requires the species 'E' !\n");
      assert(false);
    }
  }
}

void MixingLengthTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                       const Vector &Efield, Vector &transportBuffer,
                                                       DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  diffusionVelocity.SetSize(numSpecies, nvel_);
  ComputeFluxTransportProperties(&state[0], gradUp.Read(), &Efield[0], &transportBuffer[0], diffusionVelocity.Write());
}

MFEM_HOST_DEVICE void MixingLengthTransport::ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                                        const double *Efield, double *transportBuffer,
                                                                        double *diffusionVelocity) {
  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  const double rho = state[0];
  const double Th = primitiveState[nvel_ + 1];
  const double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : Th;

  // viscosity
  const double mu = mu0_ * pow(Th / T0_, visc_power_);

  const double gas_constant = mixture->GetGasConstant();
  const double specific_heat_ratio = mixture->GetSpecificHeatRatio();
  const double cp = specific_heat_ratio * gas_constant / (specific_heat_ratio - 1.);

  // heavy thermal conductivity
  const double kappa = mu * cp / Pr_;

  // eddy viscosity
  double S = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double u_x = gradUp[(1 + i) + j * num_equation];
      S += 2 * u_x * u_x; // todo: subtract divergence part
    }
  }
  S = sqrt(S);

  const double mut = rho * mixing_length_ * mixing_length_ * S;

  // eddy thermal conductivity
  const double kappat = mut * cp / Prt_;

  const double diff = (mu / rho) / (Le_ * Pr_);
  const double difft = (mut / rho) / (Let_ * Prt_);

  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);

  double diffusivity[gpudata::MAXSPECIES];
  for (int i = 0; i < numSpecies; i++) {
    diffusivity[i] = diff + difft;
  }
  if (electronIndex_ >= 0) {
    diffusivity[electronIndex_] = nDe_ / n_sp[electronIndex_];
  }

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // diffusionVelocity.SetSize(numSpecies, nvel_);
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;

  // Compute mole fraction gradient from number density gradient.
  // concentration-driven diffusion only determines the first dim-components.
  // DenseMatrix gradX(numSpecies, dim);
  double gradX[gpudata::MAXSPECIES * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      diffusionVelocity[sp + d * numSpecies] = -diffusivity[sp] * gradX[sp + d * numSpecies] / (X_sp[sp] + Xeps_);
  }

  double mobility[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      if (isnan(diffusionVelocity[sp + d * numSpecies])) {
        // grvy_printf(GRVY_ERROR, "\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity(sp, d));
        printf("\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity[sp + d * numSpecies]);
        // exit(-1);
        assert(false);
      }
    }
  }

  // Set output params
  transportBuffer[FluxTrns::VISCOSITY] = mu + mut;
  transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = kappa + kappat;
  transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = 2.5 * UNIVERSALGASCONSTANT * nDe_;



  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // diffusionVelocity.SetSize(numSpecies, nvel_);
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;

}

void MixingLengthTransport::ComputeSourceTransportProperties(const Vector &state, const Vector &Up,
                                                         const DenseMatrix &gradUp, const Vector &Efield,
                                                         Vector &globalTransport, DenseMatrix &speciesTransport,
                                                         DenseMatrix &diffusionVelocity, Vector &n_sp) {
  globalTransport.SetSize(SrcTrns::NUM_SRC_TRANS);
  speciesTransport.SetSize(numSpecies, SpeciesTrns::NUM_SPECIES_COEFFS);
  n_sp.SetSize(numSpecies);
  diffusionVelocity.SetSize(numSpecies, nvel_);
  ComputeSourceTransportProperties(&state[0], &Up[0], gradUp.Read(), &Efield[0], &globalTransport[0],
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

MFEM_HOST_DEVICE void MixingLengthTransport::ComputeSourceTransportProperties(const double *state, const double *Up,
                                                                          const double *gradUp, const double *Efield,
                                                                          double *globalTransport,
                                                                          double *speciesTransport,
                                                                          double *diffusionVelocity, double *n_sp) {
  for (int i = 0; i < SrcTrns::NUM_SRC_TRANS; i++) globalTransport[i] = 0.0;
  for (int c = 0; c < SpeciesTrns::NUM_SPECIES_COEFFS; c++)
    for (int sp = 0; sp < numSpecies; sp++) speciesTransport[sp + c * numSpecies] = 0.0;
  // n_sp.SetSize(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++) n_sp[sp] = 0.0;
  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // diffusionVelocity.SetSize(numSpecies, nvel_);
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;

  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  const double rho = state[0];
  const double Th = primitiveState[nvel_ + 1];
  const double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : Th;

  // viscosity
  const double mu = mu0_ * pow(Th / T0_, visc_power_);

  const double gas_constant = mixture->GetGasConstant();
  const double specific_heat_ratio = mixture->GetSpecificHeatRatio();
  const double cp = specific_heat_ratio * gas_constant / (specific_heat_ratio - 1.);

  // heavy thermal conductivity
  const double kappa = mu * cp / Pr_;

  // eddy viscosity
  double S = 0;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      const double u_x = gradUp[(1 + i) + j * num_equation];
      S += 2 * u_x * u_x; // todo: subtract divergence part
    }
  }
  S = sqrt(S);

  const double mut = rho * mixing_length_ * mixing_length_ * S;

  // eddy thermal conductivity
  const double kappat = mut * cp / Prt_;

  const double diff = (mu / rho) / (Le_ * Pr_);
  const double difft = (mut / rho) / (Let_ * Prt_);

  double X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);

  double diffusivity[gpudata::MAXSPECIES];
  for (int i = 0; i < numSpecies; i++) {
    diffusivity[i] = diff + difft;
  }
  if (electronIndex_ >= 0) {
    diffusivity[electronIndex_] = nDe_ / n_sp[electronIndex_];
  }

  // Compute mole fraction gradient from number density gradient.
  // concentration-driven diffusion only determines the first dim-components.
  // DenseMatrix gradX(numSpecies, dim);
  double gradX[gpudata::MAXSPECIES * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      diffusionVelocity[sp + d * numSpecies] = -diffusivity[sp] * gradX[sp + d * numSpecies] / (X_sp[sp] + Xeps_);
  }

  double mobility[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }
  globalTransport[SrcTrns::ELECTRIC_CONDUCTIVITY] =
      computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  //for (int sp = 0; sp < numSpecies; sp++) speciesTransport[sp + SpeciesTrns::MF_FREQUENCY * numSpecies] = 0.0;

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      if (isnan(diffusionVelocity[sp + d * numSpecies])) {
        // grvy_printf(GRVY_ERROR, "\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity(sp, d));
        printf("\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity[sp + d * numSpecies]);
        // exit(-1);
        assert(false);
      }
    }
  }
}
