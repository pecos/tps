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

#include "transport_properties.hpp"

TransportProperties::TransportProperties(GasMixture *_mixture) : mixture(_mixture) {
  numSpecies = mixture->GetNumSpecies();
  dim = mixture->GetDimension();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
}

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

DryAirTransport::DryAirTransport(GasMixture *_mixture, RunConfiguration &_runfile) : TransportProperties(_mixture) {
  visc_mult = _runfile.GetViscMult();
  bulk_visc_mult = _runfile.GetBulkViscMult();

  Pr = 0.71;
  Sc = 0.71;
  gas_constant = UNIVERSALGASCONSTANT / mixture->GetGasParams(0, GasParams::SPECIES_MW);
  const double specific_heat_ratio = mixture->GetSpecificHeatRatio();
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
}

void DryAirTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                     Vector &transportBuffer, DenseMatrix &diffusionVelocity) {
  double dummy; // electron pressure. won't compute anything.
  double p = mixture->ComputePressure(state, dummy);
  double temp = p / gas_constant / state[0];

  transportBuffer.SetSize(GlobalTrnsCoeffs::NUM_GLOBAL_COEFFS);
  transportBuffer = 0.0;
  transportBuffer[GlobalTrnsCoeffs::VISCOSITY] = (1.458e-6 * visc_mult * pow(temp, 1.5) / (temp + 110.4));
  transportBuffer[GlobalTrnsCoeffs::BULK_VISCOSITY] = bulk_visc_mult;
  transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY] =
      cp_div_pr * transportBuffer[GlobalTrnsCoeffs::VISCOSITY];

  diffusionVelocity.SetSize(numSpecies, 3);
  diffusionVelocity = 0.0;
  if (numActiveSpecies > 0) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      double diffusivity = transportBuffer[GlobalTrnsCoeffs::VISCOSITY] / Sc;

      for (int d = 0; d < dim; d++) {
        if (fabs(state[2+dim+sp]) > 1e-10) {
          diffusionVelocity(sp, d) = 0.0;
        } else {
          // compute mass fraction gradient
          double dY = mixture->GetGasParams(sp, GasParams::SPECIES_MW) * gradUp(2+dim+sp, d);
          dY -= state(2+dim+sp)/state(0) * gradUp(0, d);
          dY /= state(0);

          diffusionVelocity(sp, d) = diffusivity * dY / state[2+dim+sp];
        }
      }

      for (int d = 0; d < dim; d++) {
        assert(!std::isnan(diffusionVelocity(0, d)));
      }
    }
  }
}

//////////////////////////////////////////////////////
//////// Test Binary Air mixture
//////////////////////////////////////////////////////

TestBinaryAirTransport::TestBinaryAirTransport(GasMixture *_mixture, RunConfiguration &_runfile)
    : DryAirTransport(_mixture, _runfile) {}

void TestBinaryAirTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                            Vector &transportBuffer, DenseMatrix &diffusionVelocity) {
  double dummy; // electron pressure. won't compute anything.
  double p = mixture->ComputePressure(state, dummy);
  double temp = p / gas_constant / state[0];

  transportBuffer.SetSize(GlobalTrnsCoeffs::NUM_GLOBAL_COEFFS);
  transportBuffer[GlobalTrnsCoeffs::VISCOSITY] = (1.458e-6 * visc_mult * pow(temp, 1.5) / (temp + 110.4));
  transportBuffer[GlobalTrnsCoeffs::BULK_VISCOSITY] = bulk_visc_mult;
  transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY] =
      cp_div_pr * transportBuffer[GlobalTrnsCoeffs::VISCOSITY];

  diffusionVelocity.SetSize(numSpecies, 3);
  diffusionVelocity = 0.0;
  if (numActiveSpecies > 0) {

    double diffusivity = transportBuffer[GlobalTrnsCoeffs::VISCOSITY] / Sc;

    // Compute mass fraction gradient from number density gradient.
    DenseMatrix massFractionGrad;
    mixture->ComputeMassFractionGradient(state, gradUp, massFractionGrad);
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (state[dim + 2 + sp] == 0.0) continue;

      for (int d = 0; d < dim; d++)
        diffusionVelocity(sp, d) = diffusivity * massFractionGrad(sp, d) / state[dim + 2 + sp] * state[0];
    }

    for (int d = 0; d < dim; d++) {
      assert(!std::isnan(diffusionVelocity(0, d)));
    }
  }
}

//////////////////////////////////////////////////////
//////// Constant transport
//////////////////////////////////////////////////////

ConstantTransport::ConstantTransport(GasMixture *_mixture, RunConfiguration &_runfile) : TransportProperties(_mixture) {
  viscosity_ = _runfile.constantTransport.viscosity;
  bulkViscosity_ = _runfile.constantTransport.bulkViscosity;
  diffusivity_ = _runfile.constantTransport.diffusivity;
  thermalConductivity_ = _runfile.constantTransport.thermalConductivity;
}

void ConstantTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                       Vector &transportBuffer, DenseMatrix &diffusionVelocity) {

  transportBuffer.SetSize(GlobalTrnsCoeffs::NUM_GLOBAL_COEFFS);
  transportBuffer[GlobalTrnsCoeffs::VISCOSITY] = viscosity_;
  transportBuffer[GlobalTrnsCoeffs::BULK_VISCOSITY] = bulkViscosity_;
  transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY] = thermalConductivity_;
  transportBuffer[GlobalTrnsCoeffs::ELECTRON_THERMAL_CONDUCTIVITY] = thermalConductivity_;

  diffusionVelocity.SetSize(numSpecies, dim);
  diffusionVelocity = 0.0;

  // Compute mass fraction gradient from number density gradient.
  DenseMatrix massFractionGrad(numSpecies, dim);
  mixture->ComputeMassFractionGradient(state, gradUp, massFractionGrad);
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      diffusionVelocity(sp, d) = diffusivity_ * massFractionGrad(sp, d) / (state[dim + 2 + sp] + Xeps_) * state[0];
  }

  for (int d = 0; d < dim; d++) {
    assert(!std::isnan(diffusionVelocity(0, d)));
  }
}
