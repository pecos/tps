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

#include "argon_transport.hpp"

ArgonMinimalTransport::ArgonMinimalTransport(GasMixture *_mixture, RunConfiguration &_runfile)
    : TransportProperties(_mixture) {
  // if (!ambipolar) {
  //   grvy_printf(GRVY_ERROR, "\nArgon ternary transport currently supports ambipolar condition only. Set
  //   plasma_models/ambipolar = true.\n"); exit(ERROR);
  // }
  if (numSpecies != 3) {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport only supports ternary mixture of Ar, Ar.+1, and E !\n");
    exit(ERROR);
  }

  std::map<std::string, int> *speciesMapping = mixture->getSpeciesMapping();
  if (speciesMapping->count("Ar")) {
    neutralIndex_ = (*speciesMapping)["Ar"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar' !\n");
    exit(ERROR);
  }
  if (speciesMapping->count("Ar.+1")) {
    ionIndex_ = (*speciesMapping)["Ar.+1"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar.+1' !\n");
    exit(ERROR);
  }
  if (speciesMapping->count("E")) {
    electronIndex_ = (*speciesMapping)["E"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
    exit(ERROR);
  }

  // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
  // multiplying/dividing big numbers are risky of losing precision.
  mw_.SetSize(3);
  mw_(electronIndex_) = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
  mw_(neutralIndex_) = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
  mw_(ionIndex_) = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);
  // assumes input mass is consistent with this.
  assert(abs(mw_(neutralIndex_) - mw_(electronIndex_) - mw_(ionIndex_)) < 1.0e-15);
  mw_ /= AVOGADRONUMBER;
  // mA_ /= AVOGADRONUMBER;
  // mI_ /= AVOGADRONUMBER;

  muw_.SetSize(3);
  computeEffectiveMass(mw_, muw_);

  thirdOrderkElectron_ = _runfile.thirdOrderkElectron;
}

void ArgonMinimalTransport::computeEffectiveMass(const Vector &mw, DenseSymmetricMatrix &muw) {
  muw.SetSize(numSpecies);
  muw = 0.0;

  for (int spI = 0; spI < numSpecies; spI++)
    for (int spJ = spI + 1; spJ < numSpecies; spJ++)
      muw(spI, spJ) = mw(spI) * mw(spJ) / (mw(spI) + mw(spJ));
}

void ArgonMinimalTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                           const Vector &Efield, Vector &transportBuffer,
                                                           DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  transportBuffer = 0.0;

  Vector primitiveState(num_equation);
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  // Vector n_sp(3); //number densities
  // n_sp(ionIndex_) = primitiveState[dim + 2 + ionIndex_];
  // n_sp(electronIndex_) = mixture->computeAmbipolarElectronNumberDensity(&primitiveState[dim + 2]);
  // n_sp(neutralIndex_) = mixture->computeBackgroundMassDensity(state[0], &primitiveState[dim + 2],
  // n_sp(electronIndex_), true); n_sp(neutralIndex_) /= mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
  // double nTotal = n_sp(0) + n_sp(1) + n_sp(2);
  // Vector X_sp(3), Y_sp(3);
  // for (int sp = 0; sp < numSpecies; sp++) {
  //   X_sp(sp) = n_sp(sp) / nTotal;
  //   Y_sp(sp) = n_sp(sp) / state(0) * mixture->GetGasParams(sp, GasParams::SPECIES_MW);
  // }
  Vector n_sp(3), X_sp(3), Y_sp(3);
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp(sp);

  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];
  // std::cout << "temp: " << Th << ",\t" << Te << std::endl;

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp(electronIndex_) + Xeps_) / Te + (n_sp(ionIndex_) + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  Vector speciesViscosity(3), speciesHvyThrmCnd(3);
  speciesViscosity(ionIndex_) =
      viscosityFactor_ * sqrt(mw_(ionIndex_) * Th) / (collision::charged::rep22(nondimTh) * debyeCircle);
  speciesViscosity(neutralIndex_) = viscosityFactor_ * sqrt(mw_(neutralIndex_) * Th) / collision::argon::ArAr22(Th);
  speciesViscosity(electronIndex_) = 0.0;
  // speciesViscosity(0) = 5. / 16. * sqrt(PI_ * mI_ * kB_ * Th) / (collision::charged::rep22(nondimTe) * PI_ *
  // debyeLength * debyeLength);
  for (int sp = 0; sp < numSpecies; sp++) {
    speciesHvyThrmCnd(sp) = speciesViscosity(sp) * kOverEtaFactor_ / mw_(sp);
  }
  transportBuffer[FluxTrns::VISCOSITY] = linearAverage(X_sp, speciesViscosity);
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = linearAverage(X_sp, speciesHvyThrmCnd);
  transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;

  if (thirdOrderkElectron_) {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
        computeThirdOrderElectronThermalConductivity(X_sp, debyeLength, Te, nondimTe);
  } else {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = viscosityFactor_ * kOverEtaFactor_ *
                                                               sqrt(Te / mw_(electronIndex_)) * X_sp(electronIndex_) /
                                                               (collision::charged::rep22(nondimTe) * debyeCircle);
  }

  DenseSymmetricMatrix binaryDiff(3);
  binaryDiff = 0.0;
  binaryDiff(electronIndex_, neutralIndex_) = diffusivityFactor_ * sqrt(Te / muw_(electronIndex_, neutralIndex_)) /
                                              nTotal / collision::argon::eAr11(Te);
  binaryDiff(neutralIndex_, ionIndex_) = diffusivityFactor_ * sqrt(Th / muw_(neutralIndex_, ionIndex_)) / nTotal /
                                         collision::argon::ArAr1P11(Th);
  binaryDiff(ionIndex_, electronIndex_) =
      diffusivityFactor_ * sqrt(Te / muw_(ionIndex_, electronIndex_)) / nTotal /
      (collision::charged::att11(nondimTe) * debyeCircle);

  Vector diffusivity(3), mobility(3);
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility(sp) = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity(sp);
  }

  DenseMatrix gradX(numSpecies, dim);
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  diffusionVelocity.SetSize(numSpecies, nvel_);
  diffusionVelocity = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity(sp) * gradX(sp, d);
      // NOTE: we'll have to handle small X case.
      diffusionVelocity(sp, d) = -DgradX / (X_sp(sp) + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++) speciesSpeed += diffusionVelocity(sp, d) * diffusionVelocity(sp, d);
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

double ArgonMinimalTransport::computeThirdOrderElectronThermalConductivity(const Vector &X_sp, const double debyeLength,
                                                                           const double Te, const double nondimTe) {
  double debyeCircle = PI_ * debyeLength * debyeLength;
  // std::cout << "LD: " << debyeLength << std::endl;
  Vector Q2(3);
  Q2(0) = debyeCircle * collision::charged::rep22(nondimTe);
  Q2(1) = debyeCircle * collision::charged::rep23(nondimTe);
  Q2(2) = debyeCircle * collision::charged::rep24(nondimTe);
  // std::cout << "Q2r: " << Q2(0) << ",\t" << Q2(1) << ",\t" << Q2(2) << std::endl;

  Vector Q1Ion(5);
  Q1Ion(0) = debyeCircle * collision::charged::att11(nondimTe);
  Q1Ion(1) = debyeCircle * collision::charged::att12(nondimTe);
  Q1Ion(2) = debyeCircle * collision::charged::att13(nondimTe);
  Q1Ion(3) = debyeCircle * collision::charged::att14(nondimTe);
  Q1Ion(4) = debyeCircle * collision::charged::att15(nondimTe);
  //   std::cout << "Q1i: ";
  // for (int i = 0; i < 5; i++) std::cout << Q1Ion(i) << ",\t";
  // std::cout << std::endl;

  Vector Q1Neutral(5);
  Q1Neutral(0) = collision::argon::eAr11(Te);
  Q1Neutral(1) = collision::argon::eAr12(Te);
  Q1Neutral(2) = collision::argon::eAr13(Te);
  Q1Neutral(3) = collision::argon::eAr14(Te);
  Q1Neutral(4) = collision::argon::eAr15(Te);
  //   std::cout << "Q1A: ";
  // for (int i = 0; i < 5; i++) std::cout << Q1Neutral(i) << ",\t";
  // std::cout << std::endl;

  double L11 = sqrt(2.0) * X_sp(electronIndex_) * L11ee(Q2);
  L11 += X_sp(ionIndex_) * L11ea(Q1Ion);
  L11 += X_sp(neutralIndex_) * L11ea(Q1Neutral);
  double L12 = sqrt(2.0) * X_sp(electronIndex_) * L12ee(Q2);
  L12 += X_sp(ionIndex_) * L12ea(Q1Ion);
  L12 += X_sp(neutralIndex_) * L12ea(Q1Neutral);
  double L22 = sqrt(2.0) * X_sp(electronIndex_) * L22ee(Q2);
  L22 += X_sp(ionIndex_) * L22ea(Q1Ion);
  L22 += X_sp(neutralIndex_) * L22ea(Q1Neutral);
  // std::cout << "L11: " << L11 << std::endl;
  // std::cout << "L12: " << L12 << std::endl;
  // std::cout << "L22: " << L22 << std::endl;

  return viscosityFactor_ * kOverEtaFactor_ * sqrt(2.0 * Te / mw_(electronIndex_)) * X_sp(electronIndex_) /
         (L11 - L12 * L12 / L22);
}

void ArgonMinimalTransport::computeMixtureAverageDiffusivity(const Vector &state, Vector &diffusivity) {
  Vector primitiveState(num_equation);
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  Vector n_sp(3), X_sp(3), Y_sp(3);
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp(sp);

  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp(electronIndex_) + Xeps_) / Te + (n_sp(ionIndex_) + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  DenseSymmetricMatrix binaryDiff(3);
  binaryDiff = 0.0;
  binaryDiff(electronIndex_, neutralIndex_) = diffusivityFactor_ * sqrt(Te / muw_(electronIndex_, neutralIndex_)) /
                                              nTotal / collision::argon::eAr11(Te);
  binaryDiff(neutralIndex_, ionIndex_) = diffusivityFactor_ * sqrt(Th / muw_(neutralIndex_, ionIndex_)) / nTotal /
                                         collision::argon::ArAr1P11(Th);
  binaryDiff(ionIndex_, electronIndex_) =
      diffusivityFactor_ * sqrt(Te / muw_(ionIndex_, electronIndex_)) / nTotal /
      (collision::charged::att11(nondimTe) * debyeCircle);

  diffusivity.SetSize(3);
  diffusivity = 0.0;
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);
}

void ArgonMinimalTransport::ComputeSourceTransportProperties(const Vector &state, const Vector &Up,
                                                             const DenseMatrix &gradUp, const Vector &Efield,
                                                             Vector &globalTransport, DenseMatrix &speciesTransport,
                                                             DenseMatrix &diffusionVelocity, Vector &n_sp) {
  globalTransport.SetSize(SrcTrns::NUM_SRC_TRANS);
  globalTransport = 0.0;
  speciesTransport.SetSize(numSpecies, SpeciesTrns::NUM_SPECIES_COEFFS);
  speciesTransport = 0.0;

  n_sp.SetSize(3);
  Vector X_sp(3), Y_sp(3);
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp(sp);

  double Te = (twoTemperature_) ? Up[num_equation - 1] : Up[nvel_ + 1];
  double Th = Up[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp(electronIndex_) + Xeps_) / Te + (n_sp(ionIndex_) + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double Qea = collision::argon::eAr11(Te);
  double Qai = collision::argon::ArAr1P11(Th);
  double Qie = collision::charged::att11(nondimTe) * debyeCircle;

  DenseSymmetricMatrix binaryDiff(3);
  binaryDiff = 0.0;
  binaryDiff(electronIndex_, neutralIndex_) = diffusivityFactor_ * sqrt(Te / muw_(electronIndex_, neutralIndex_)) / nTotal / Qea;
  binaryDiff(neutralIndex_, ionIndex_) = diffusivityFactor_ * sqrt(Th / muw_(neutralIndex_, ionIndex_)) / nTotal / Qai;
  binaryDiff(ionIndex_, electronIndex_) =
      diffusivityFactor_ * sqrt(Te / muw_(ionIndex_, electronIndex_)) / nTotal / Qie;

  Vector diffusivity(3), mobility(3);
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility(sp) = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity(sp);
  }
  globalTransport(SrcTrns::ELECTRIC_CONDUCTIVITY) =
      computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;

  DenseMatrix gradX(numSpecies, dim);
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  diffusionVelocity.SetSize(numSpecies, nvel_);
  diffusionVelocity = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity(sp) * gradX(sp, d);
      // NOTE: we'll have to handle small X case.
      diffusionVelocity(sp, d) = -DgradX / (X_sp(sp) + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  speciesTransport(ionIndex_, SpeciesTrns::MF_FREQUENCY) =
      mfFreqFactor_ * sqrt(Te / mw_(electronIndex_)) * n_sp(ionIndex_) * Qie;
  speciesTransport(neutralIndex_, SpeciesTrns::MF_FREQUENCY) =
      mfFreqFactor_ * sqrt(Te / mw_(electronIndex_)) * n_sp(neutralIndex_) * Qea;
  // // relative electron collision speed
  // double ge = sqrt(8.0 * kB_ * Te / PI_ / mw_(electronIndex_));
  // speciesTransport(ionIndex_, SpeciesTrns::MF_FREQUENCY) = 4.0 / 3.0 * AVOGADRONUMBER * n_sp(ionIndex_) * ge * Qie;

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++) speciesSpeed += diffusionVelocity(sp, d) * diffusionVelocity(sp, d);
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

void ArgonMinimalTransport::GetViscosities(const Vector &conserved, const Vector &primitive, double &visc,
                                           double &bulkVisc) {
  Vector n_sp(3), X_sp(3), Y_sp(3);
  mixture->computeSpeciesPrimitives(conserved, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp(sp);

  double Te = (twoTemperature_) ? primitive[num_equation - 1] : primitive[nvel_ + 1];
  double Th = primitive[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp(electronIndex_) + Xeps_) / Te + (n_sp(ionIndex_) + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  Vector speciesViscosity(3);
  speciesViscosity(ionIndex_) =
      viscosityFactor_ * sqrt(mw_(ionIndex_) * Th) / (collision::charged::rep22(nondimTh) * debyeCircle);
  speciesViscosity(neutralIndex_) = viscosityFactor_ * sqrt(mw_(neutralIndex_) * Th) / collision::argon::ArAr22(Th);
  speciesViscosity(electronIndex_) = 0.0;

  visc = linearAverage(X_sp, speciesViscosity);
  bulkVisc = 0.0;

  return;
}
