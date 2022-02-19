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

ArgonTernaryTransport::ArgonTernaryTransport(GasMixture *_mixture, RunConfiguration &_runfile) : TransportProperties(_mixture) {
  if (!ambipolar) {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport currently supports ambipolar condition only. Set plasma_models/ambipolar = true.\n");
    exit(ERROR);
  }
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

  mw_.SetSize(3);
  mw_(electronIndex_) = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
  mw_(neutralIndex_) = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
  mw_(ionIndex_) = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);
  // assumes input mass is consistent with this.
  assert( abs(mw_(neutralIndex_) - mw_(electronIndex_) - mw_(ionIndex_)) < 1.0e-15 );
  mw_ /= AVOGADRONUMBER;
  // mA_ /= AVOGADRONUMBER;
  // mI_ /= AVOGADRONUMBER;

  muAE_ = mw_(electronIndex_) * mw_(neutralIndex_) / (mw_(electronIndex_) + mw_(neutralIndex_));
  muAI_ = mw_(ionIndex_) * mw_(neutralIndex_) / (mw_(ionIndex_) + mw_(neutralIndex_));

  thirdOrderkElectron_ = _runfile.thirdOrderkElectron;

}

void ArgonTernaryTransport::ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp,
                                                            Vector &transportBuffer, DenseMatrix &diffusionVelocity) {
  transportBuffer.SetSize(GlobalTrnsCoeffs::NUM_GLOBAL_COEFFS);
  transportBuffer = 0.0;

  Vector primitiveState(num_equation);
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  Vector n_sp(3); //number densities
  n_sp(ionIndex_) = primitiveState[dim + 2 + ionIndex_];
  n_sp(electronIndex_) = mixture->computeAmbipolarElectronNumberDensity(&primitiveState[dim + 2]);
  n_sp(neutralIndex_) = mixture->computeBackgroundMassDensity(state[0], &primitiveState[dim + 2], n_sp(electronIndex_), true);
  n_sp(neutralIndex_) /= mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
  double nTotal = n_sp(0) + n_sp(1) + n_sp(2);
  Vector X_sp(3);
  for (int sp = 0; sp < numSpecies; sp++) X_sp(sp) = n_sp(sp) / nTotal;
  // std::cout << "N inside: ";
  // for (int sp = 0; sp < 3; sp++) {
  //   std::cout << n_sp(sp) << ",\t";
  // }
  // std::cout << std::endl;

  double Te = (twoTemperature) ? primitiveState[num_equation - 1] : primitiveState[dim + 1];
  double Th = primitiveState[dim + 1];
  // std::cout << "temp: " << Th << ",\t" << Te << std::endl;

  double nOverT = n_sp(electronIndex_) / Te + n_sp(ionIndex_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  Vector speciesViscosity(3);
  speciesViscosity(ionIndex_) = viscosityFactor_ * sqrt(mw_(ionIndex_) * Th) / (collision::charged::rep22(nondimTh) * debyeCircle);
  speciesViscosity(neutralIndex_) = viscosityFactor_ * sqrt(mw_(neutralIndex_) * Th) / collision::argon::ArAr22(Th);
  // std::cout << "viscosity: " << speciesViscosity(neutralIndex_) << ",\t" << speciesViscosity(ionIndex_) << std::endl;
  speciesViscosity(electronIndex_) = 0.0;
  // speciesViscosity(0) = 5. / 16. * sqrt(PI_ * mI_ * kB_ * Th) / (collision::charged::rep22(nondimTe) * PI_ * debyeLength * debyeLength);
  for (int sp = 0; sp < numSpecies; sp++) {
    transportBuffer[GlobalTrnsCoeffs::VISCOSITY] +=  X_sp(sp) * speciesViscosity(sp);
    transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY] += X_sp(sp) * speciesViscosity(sp) * kOverEtaFactor_ / mw_(sp);
  }
  transportBuffer[GlobalTrnsCoeffs::BULK_VISCOSITY] = 0.0;

  if (thirdOrderkElectron_) {
    transportBuffer[GlobalTrnsCoeffs::ELECTRON_THERMAL_CONDUCTIVITY] = computeThirdOrderElectronThermalConductivity(X_sp, debyeLength, Te, nondimTe);
  } else {
    transportBuffer[GlobalTrnsCoeffs::ELECTRON_THERMAL_CONDUCTIVITY] = viscosityFactor_ * kOverEtaFactor_ * sqrt(Te / mw_(electronIndex_)) * X_sp(electronIndex_)
                                                                                        / (collision::charged::rep22(nondimTe) * debyeCircle);
  }


}

double ArgonTernaryTransport::computeThirdOrderElectronThermalConductivity(const Vector &X_sp, const double debyeLength, const double Te, const double nondimTe) {
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

  return viscosityFactor_ * kOverEtaFactor_ * sqrt(2.0 * Te / mw_(electronIndex_)) * X_sp(electronIndex_) / (L11 - L12 * L12 / L22);
}
