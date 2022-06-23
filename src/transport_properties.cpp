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

#include "transport_properties.hpp"

TransportProperties::TransportProperties(GasMixture *_mixture) : mixture(_mixture) {
  numSpecies = mixture->GetNumSpecies();
  dim = mixture->GetDimension();
  nvel_ = mixture->GetNumVels();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
}

MFEM_HOST_DEVICE TransportProperties::TransportProperties(GasMixture *_mixture) {
  mixture = _mixture
  numSpecies = mixture->GetNumSpecies();
  dim = mixture->GetDimension();
  nvel_ = mixture->GetNumVels();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
}

void TransportProperties::correctMassDiffusionFlux(const Vector &Y_sp, DenseMatrix &diffusionVelocity) {
  // Correction Velocity
  Vector Vc(nvel_);
  Vc = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // NOTE: we'll have to handle small Y case.
    for (int d = 0; d < nvel_; d++) Vc(d) += Y_sp(sp) * diffusionVelocity(sp, d);
  }
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) diffusionVelocity(sp, d) -= Vc(d);
  }
}

double TransportProperties::computeMixtureElectricConductivity(const Vector &mobility, const Vector &n_sp) {
  double mho = 0.0;  // electric conductivity.

  for (int sp = 0; sp < numSpecies; sp++) {
    mho += mobility(sp) * n_sp(sp) * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);
  }

  return mho;
}

void TransportProperties::addAmbipolarEfield(const Vector &mobility, const Vector &n_sp,
                                             DenseMatrix &diffusionVelocity) {
  double mho = computeMixtureElectricConductivity(mobility, n_sp);

  Vector ambE(nvel_);
  ambE = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      ambE(d) -= diffusionVelocity(sp, d) * n_sp(sp) * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);
    }
  }

  for (int d = 0; d < nvel_; d++)
    ambE(d) /= (mho + Xeps_);  // NOTE: add Xeps for the case of no charged-species at the point.

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) diffusionVelocity(sp, d) += mobility(sp) * ambE(d);
  }
}

void TransportProperties::addMixtureDrift(const Vector &mobility, const Vector &n_sp, const Vector &Efield,
                                          DenseMatrix &diffusionVelocity) {
  for (int sp = 0; sp < numSpecies; sp++) {
    if (mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) == 0.0) continue;

    for (int d = 0; d < nvel_; d++) diffusionVelocity(sp, d) += mobility(sp) * Efield(d);
  }
}

double TransportProperties::linearAverage(const Vector &X_sp, const Vector &speciesTransport) {
  double average = 0.0;

  for (int sp = 0; sp < numSpecies; sp++) average += X_sp(sp) * speciesTransport(sp);

  return average;
}

void TransportProperties::CurtissHirschfelder(const Vector &X_sp, const Vector &Y_sp,
                                              const DenseSymmetricMatrix &binaryDiff, Vector &avgDiff) {
  avgDiff.SetSize(numSpecies);
  avgDiff = 0.0;

  for (int spI = 0; spI < numSpecies; spI++) {
    for (int spJ = 0; spJ < numSpecies; spJ++) {
      if (spI == spJ) continue;
      avgDiff(spI) += (X_sp(spJ) + Xeps_) / binaryDiff(spI, spJ);
    }
    avgDiff(spI) = (1.0 - Y_sp(spI)) / avgDiff(spI);
  }
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
                                                     const Vector &Efield, Vector &transportBuffer,
                                                     DenseMatrix &diffusionVelocity) {
  double p = mixture->ComputePressure(state);
  double temp = p / gas_constant / state[0];

  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  transportBuffer = 0.0;
  double viscosity = (1.458e-6 * visc_mult * pow(temp, 1.5) / (temp + 110.4));
  transportBuffer[FluxTrns::VISCOSITY] = viscosity;
  transportBuffer[FluxTrns::BULK_VISCOSITY] = bulk_visc_mult * viscosity;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = cp_div_pr * transportBuffer[FluxTrns::VISCOSITY];

  diffusionVelocity.SetSize(numSpecies, nvel_);
  diffusionVelocity = 0.0;
  if (numActiveSpecies > 0) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      double diffusivity = transportBuffer[FluxTrns::VISCOSITY] / Sc;

      // NOTE(kevin): only compute the first dim-components.
      for (int d = 0; d < dim; d++) {
        if (fabs(state[2 + nvel_ + sp]) > 1e-10) {
          // compute mass fraction gradient
          double dY = mixture->GetGasParams(sp, GasParams::SPECIES_MW) * gradUp(2 + nvel_ + sp, d);
          dY -= state(2 + nvel_ + sp) / state(0) * gradUp(0, d);
          dY /= state(0);

          diffusionVelocity(sp, d) = diffusivity * dY / state[2 + nvel_ + sp];
        }
      }

      for (int d = 0; d < nvel_; d++) {
        assert(!std::isnan(diffusionVelocity(0, d)));
      }
    }
  }
}

//////////////////////////////////////////////////////
//////// Constant transport
//////////////////////////////////////////////////////

ConstantTransport::ConstantTransport(GasMixture *_mixture, RunConfiguration &_runfile) : TransportProperties(_mixture) {
  viscosity_ = _runfile.constantTransport.viscosity;
  bulkViscosity_ = _runfile.constantTransport.bulkViscosity;
  thermalConductivity_ = _runfile.constantTransport.thermalConductivity;
  electronThermalConductivity_ = _runfile.constantTransport.electronThermalConductivity;

  diffusivity_.SetSize(numSpecies);
  mtFreq_.SetSize(numSpecies);
  std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  for (int mixSp = 0; mixSp < numSpecies; mixSp++) {
    int inputSp = (*mixtureToInputMap)[mixSp];
    diffusivity_(mixSp) = _runfile.constantTransport.diffusivity(inputSp);
    mtFreq_(mixSp) = _runfile.constantTransport.mtFreq(inputSp);
  }

  if (mixture->IsTwoTemperature()) {
    std::map<std::string, int> *speciesMapping = mixture->getSpeciesMapping();
    if (speciesMapping->count("E")) {
      electronIndex_ = (*speciesMapping)["E"];
    } else {
      grvy_printf(GRVY_ERROR, "\nConstant transport: two-temperature plasma requires the species 'E' !\n");
      exit(ERROR);
    }
  }
}

void ConstantTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                       const Vector &Efield, Vector &transportBuffer,
                                                       DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  transportBuffer[FluxTrns::VISCOSITY] = viscosity_;
  transportBuffer[FluxTrns::BULK_VISCOSITY] = bulkViscosity_;
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = thermalConductivity_;
  transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = electronThermalConductivity_;

  Vector primitiveState(num_equation);
  mixture->GetPrimitivesFromConservatives(state, primitiveState);
  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];

  Vector n_sp(numSpecies), X_sp(numSpecies), Y_sp(numSpecies);
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  diffusionVelocity.SetSize(numSpecies, nvel_);
  diffusionVelocity = 0.0;

  // Compute mole fraction gradient from number density gradient.
  // concentration-driven diffusion only determines the first dim-components.
  DenseMatrix gradX(numSpecies, dim);
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < dim; d++) diffusionVelocity(sp, d) = -diffusivity_(sp) * gradX(sp, d) / (X_sp(sp) + Xeps_);
  }

  Vector mobility(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility(sp) = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity_(sp);
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      if (std::isnan(diffusionVelocity(sp, d))) {
        grvy_printf(GRVY_ERROR, "\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity(sp, d));
        exit(-1);
      }
    }
  }
}

void ConstantTransport::ComputeSourceTransportProperties(const Vector &state, const Vector &Up,
                                                         const DenseMatrix &gradUp, const Vector &Efield,
                                                         Vector &globalTransport, DenseMatrix &speciesTransport,
                                                         DenseMatrix &diffusionVelocity, Vector &n_sp) {
  globalTransport.SetSize(SrcTrns::NUM_SRC_TRANS);
  globalTransport = 0.0;
  speciesTransport.SetSize(numSpecies, SpeciesTrns::NUM_SPECIES_COEFFS);
  speciesTransport = 0.0;
  n_sp.SetSize(numSpecies);
  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  diffusionVelocity.SetSize(numSpecies, nvel_);
  diffusionVelocity = 0.0;

  Vector primitiveState(num_equation);
  mixture->GetPrimitivesFromConservatives(state, primitiveState);
  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];

  Vector X_sp(numSpecies), Y_sp(numSpecies);
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);

  // Compute mole fraction gradient from number density gradient.
  // concentration-driven diffusion only determines the first dim-components.
  DenseMatrix gradX(numSpecies, dim);
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < dim; d++) diffusionVelocity(sp, d) = -diffusivity_(sp) * gradX(sp, d) / (X_sp(sp) + Xeps_);
  }

  Vector mobility(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility(sp) = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity_(sp);
  }
  globalTransport(SrcTrns::ELECTRIC_CONDUCTIVITY) =
      computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  for (int sp = 0; sp < numSpecies; sp++) speciesTransport(sp, SpeciesTrns::MF_FREQUENCY) = mtFreq_(sp);

  for (int sp = 0; sp < numSpecies; sp++) {
    for (int d = 0; d < nvel_; d++) {
      if (std::isnan(diffusionVelocity(sp, d))) {
        grvy_printf(GRVY_ERROR, "\nDiffusion velocity of species %d is NaN! -> %f\n", sp, diffusionVelocity(sp, d));
        exit(-1);
      }
    }
  }
}
