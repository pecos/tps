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

#include "gas_transport.hpp"

//////////////////////////////////////////////////////
//////// Gas Minimal Transport (ternary mixture)
//////////////////////////////////////////////////////

GasMinimalTransport::GasMinimalTransport(GasMixture *_mixture, RunConfiguration &_runfile)
    : GasMinimalTransport(_mixture, _runfile.gasTransportInput) {}

MFEM_HOST_DEVICE GasMinimalTransport::GasMinimalTransport(GasMixture *_mixture, const GasTransportInput &inputs)
    : MolecularTransport(_mixture) {
  viscosityFactor_ = 5. / 16. * sqrt(PI_ * kB_);
  kOverEtaFactor_ = 15. / 4. * kB_;
  diffusivityFactor_ = 3. / 16. * sqrt(2.0 * PI_ * kB_) / AVOGADRONUMBER;
  mfFreqFactor_ = 4. / 3. * AVOGADRONUMBER * sqrt(8. * kB_ / PI_);

  // Argon
  if (inputs.gas == "Ar" || inputs.gas == "argon") {
    gasType_ = Ar;

    if (numSpecies != 3) {
      printf("\nGas:Ar ternary transport only supports ternary mixture of Ar, Ar.+1, and E !\n");
      assert(false);
    }

    // TODO(swh):
    // if this works, replace all "collision::argon::" and "collision::nitrogen" with "gas::"
    // and then create an array of pointers to the collision functions with indexing
    // matching the binary diffusivity indexing, for each namespace in collision.
    // Would allow for generic index-based computation of all binary-diffs
    // namespace gas = collision::argon;

    neutralIndex_ = inputs.neutralIndex;
    if (neutralIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'Ar' !\n");
      assert(false);
    }
    ionIndex_ = inputs.ionIndex;
    if (ionIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'Ar.+1' !\n");
      assert(false);
    }
    electronIndex_ = inputs.electronIndex;
    if (electronIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'E' !\n");
      assert(false);
    }

    // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
    // multiplying/dividing big numbers are risky of losing precision.
    // mw_.SetSize(3);
    mw_[electronIndex_] = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
    mw_[neutralIndex_] = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
    mw_[ionIndex_] = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);

    // assumes input mass is consistent with this.
    assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-12);
    for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;

    // Nitrogen
  } else if (inputs.gas == "Ni" || inputs.gas == "nitrogen") {
    gasType_ = Ni;

    if (numSpecies != 8) {
      printf("\nGas: Ni transport only supports mixture of Ni, Ni.+1, N2, N2.+1, and E !\n");
      assert(false);
    }

    // namespace gas = collision::nitrogen;

    neutralIndex_ = inputs.neutralIndex;
    if (neutralIndex_ < 0) {
      printf("\nNitrogen transport requires the species 'N2' !\n");
      assert(false);
    }
    ionIndex_ = inputs.ionIndex;
    if (ionIndex_ < 0) {
      printf("\nNitrogen transport requires the species 'N2.+1' !\n");
      assert(false);
    }
    electronIndex_ = inputs.electronIndex;
    if (electronIndex_ < 0) {
      printf("\nNitrogen transport requires the species 'E' !\n");
      assert(false);
    }
    neutralIndex2_ = inputs.neutralIndex2;
    if (neutralIndex2_ < 0) {
      printf("\nNitrogen transport requires the species 'Ni' !\n");
      assert(false);
    }
    ionIndex2_ = inputs.ionIndex2;
    if (ionIndex2_ < 0) {
      printf("\nNitrogen transport requires the species 'Ni.+1' !\n");
      assert(false);
    }

    mw_[electronIndex_] = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
    mw_[neutralIndex_] = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
    mw_[ionIndex_] = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);
    mw_[neutralIndex2_] = mixture->GetGasParams(neutralIndex2_, GasParams::SPECIES_MW);
    mw_[ionIndex2_] = mixture->GetGasParams(ionIndex2_, GasParams::SPECIES_MW);

    // assumes input mass is consistent with this.
    assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-12);
    assert(abs(mw_[neutralIndex2_] - mw_[electronIndex_] - mw_[ionIndex2_]) < 1.0e-12);
    for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;

  } else {
    printf("Unknown gasType.");
    assert(false);
  }

  if (!ambipolar) {
    grvy_printf(
        GRVY_WARN,
        "\nGas ternary transport currently supports ambipolar condition only. Set plasma_models/ambipolar = true.\n");
  }

  computeEffectiveMass(mw_, muw_);
  thirdOrderkElectron_ = inputs.thirdOrderkElectron;
  setArtificialMultipliers(inputs);
}

MFEM_HOST_DEVICE GasMinimalTransport::GasMinimalTransport(GasMixture *_mixture) : MolecularTransport(_mixture) {
  viscosityFactor_ = 5. / 16. * sqrt(PI_ * kB_);
  kOverEtaFactor_ = 15. / 4. * kB_;
  diffusivityFactor_ = 3. / 16. * sqrt(2.0 * PI_ * kB_) / AVOGADRONUMBER;
  mfFreqFactor_ = 4. / 3. * AVOGADRONUMBER * sqrt(8. * kB_ / PI_);
}

// void GasMinimalTransport::computeEffectiveMass(const Vector &mw, DenseSymmetricMatrix &muw) {
MFEM_HOST_DEVICE void GasMinimalTransport::computeEffectiveMass(const double *mw, double *muw) {
  // muw.SetSize(numSpecies);
  // muw = 0.0;

  for (int spI = 0; spI < numSpecies; spI++) {
    for (int spJ = spI; spJ < numSpecies; spJ++) {
      muw[spI + spJ * numSpecies] = mw[spI] * mw[spJ] / (mw[spI] + mw[spJ]);
      if (spI != spJ) muw[spJ + spI * numSpecies] = muw[spI + spJ * numSpecies];
    }
  }
}

MFEM_HOST_DEVICE void GasMinimalTransport::setArtificialMultipliers(const GasTransportInput &inputs) {
  multiply_ = inputs.multiply;
  if (multiply_) {
    for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) fluxTrnsMultiplier_[t] = inputs.fluxTrnsMultiplier[t];
    for (int t = 0; t < SpeciesTrns::NUM_SPECIES_COEFFS; t++) spcsTrnsMultiplier_[t] = inputs.spcsTrnsMultiplier[t];
    diffMult_ = inputs.diffMult;
    mobilMult_ = inputs.mobilMult;
  }
}

collisionInputs GasMinimalTransport::computeCollisionInputs(const Vector &primitive, const Vector &n_sp) {
  return computeCollisionInputs(&primitive[0], &n_sp[0]);
}

MFEM_HOST_DEVICE collisionInputs GasMinimalTransport::computeCollisionInputs(const double *primitive,
                                                                             const double *n_sp) {
  collisionInputs collInputs;
  collInputs.Te = (twoTemperature_) ? primitive[num_equation - 1] : primitive[nvel_ + 1];
  collInputs.Th = primitive[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    nOverT += (n_sp[sp] + Xeps_) / collInputs.Te * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) *
              mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);
  }
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  collInputs.debyeCircle = PI_ * debyeLength * debyeLength;

  collInputs.ndimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * collInputs.Te;
  collInputs.ndimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * collInputs.Th;

  return collInputs;
}

MFEM_HOST_DEVICE void GasMinimalTransport::ComputeFluxMolecularTransport(const double *state, const double *gradUp,
                                                                         const double *Efield, double *transportBuffer,
                                                                         double *diffusionVelocity) {
  // transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
  for (int p = 0; p < FluxTrns::NUM_FLUX_TRANS; p++) transportBuffer[p] = 0.0;

  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];
  // std::cout << "temp: " << Th << ",\t" << Te << std::endl;

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp[electronIndex_] + Xeps_) / Te + (n_sp[ionIndex_] + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double speciesViscosity[gpudata::MAXSPECIES], speciesHvyThrmCnd[gpudata::MAXSPECIES];
  speciesViscosity[ionIndex_] =
      viscosityFactor_ * sqrt(mw_[ionIndex_] * Th) / (collision::charged::rep22(nondimTh) * debyeCircle);

  switch (gasType_) {
    case Ar:
      speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::argon::ArAr22(Th);
      break;
    case Ni:
      speciesViscosity[neutralIndex_] =
          viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::NiNi22(Th);
      speciesViscosity[neutralIndex2_] =
          viscosityFactor_ * sqrt(mw_[neutralIndex2_] * Th) / collision::nitrogen::N2N222(Th);
      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  speciesViscosity[electronIndex_] = 0.0;
  // speciesViscosity(0) = 5. / 16. * sqrt(PI_ * mI_ * kB_ * Th) / (collision::charged::rep22(nondimTe) * PI_ *
  // debyeLength * debyeLength);
  for (int sp = 0; sp < numSpecies; sp++) {
    speciesHvyThrmCnd[sp] = speciesViscosity[sp] * kOverEtaFactor_ / mw_[sp];
  }
  transportBuffer[FluxTrns::VISCOSITY] = linearAverage(X_sp, speciesViscosity);
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = linearAverage(X_sp, speciesHvyThrmCnd);
  transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;

  if (thirdOrderkElectron_) {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
        computeThirdOrderElectronThermalConductivity(X_sp, debyeLength, Te, nondimTe);
  } else {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = viscosityFactor_ * kOverEtaFactor_ *
                                                               sqrt(Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
                                                               (collision::charged::rep22(nondimTe) * debyeCircle);
  }

  double binaryDiff[3 * 3];
  for (int i = 0; i < 3 * 3; i++) binaryDiff[i] = 0.0;

  // from: transport_properties.cpp
  //  for (int spI = 0; spI < numSpecies; spI++) {
  //    for (int spJ = 0; spJ < numSpecies; spJ++) {
  //      if (spI == spJ) continue;
  //      avgDiff[spI] += (X_sp[spJ] + Xeps_) / binaryDiff[spI + spJ * numSpecies];
  //    }
  //    avgDiff[spI] = (1.0 - Y_sp[spI]) / avgDiff[spI];
  //  }
  // diffusion of i into j (or j into i): D_{ij} = binaryDiff[i + j * numSpecies]

  switch (gasType_) {
    case Ar:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::argon::eAr11(Te);
      break;
    case Ni:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] = diffusivityFactor_ *
                                                                sqrt(Te / getMuw(electronIndex_, neutralIndex_)) /
                                                                nTotal / collision::nitrogen::eNi11(Te);
      binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies] = diffusivityFactor_ *
                                                                 sqrt(Te / getMuw(electronIndex_, neutralIndex2_)) /
                                                                 nTotal / collision::nitrogen::eN211(Te);
      binaryDiff[neutralIndex2_ + electronIndex_ * numSpecies] =
          binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies];
      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[neutralIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + neutralIndex_ * numSpecies];

  switch (gasType_) {
    case Ar:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::argon::ArAr1P11(Th);
      break;
    case Ni:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Th / getMuw(neutralIndex2_, ionIndex2_)) / nTotal /
                                                             collision::nitrogen::N2N21P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex2_, ionIndex_)) / nTotal /
                                                            collision::nitrogen::N2Ni1P11(Th);
      binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex_, ionIndex2_)) / nTotal /
                                                            collision::nitrogen::NiN21P11(Th);

      binaryDiff[ionIndex2_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies];
      binaryDiff[ionIndex_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies];
      binaryDiff[ionIndex2_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies];
      binaryDiff[electronIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Te / getMuw(ionIndex2_, electronIndex_)) / nTotal /
                                                             (collision::charged::att11(nondimTe) * debyeCircle);
      binaryDiff[ionIndex2_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex2_ * numSpecies];

      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[ionIndex_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex_ * numSpecies];
  binaryDiff[electronIndex_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                        sqrt(Te / getMuw(ionIndex_, electronIndex_)) / nTotal /
                                                        (collision::charged::att11(nondimTe) * debyeCircle);
  binaryDiff[ionIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex_ * numSpecies];

  // TODO(swh): add more for Ni system => compiler my throw a warning...
  double diffusivity[numSpecies], mobility[numSpecies];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  // Apply artificial multipliers.
  if (multiply_) {
    for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) transportBuffer[t] *= fluxTrnsMultiplier_[t];
    for (int sp = 0; sp < numSpecies; sp++) {
      diffusivity[sp] *= diffMult_;
      mobility[sp] *= mobilMult_;
    }
  }

  // DenseMatrix gradX(numSpecies, dim);
  double gradX[gpudata::MAXSPECIES * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // diffusionVelocity.SetSize(numSpecies, nvel_);
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity[sp] * gradX[sp + d * numSpecies];
      // NOTE: we'll have to handle small X case.
      diffusionVelocity[sp + d * numSpecies] = -DgradX / (X_sp[sp] + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++)
      speciesSpeed += diffusionVelocity[sp + d * numSpecies] * diffusionVelocity[sp + d * numSpecies];
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

MFEM_HOST_DEVICE double GasMinimalTransport::computeThirdOrderElectronThermalConductivity(const double *X_sp,
                                                                                          const double debyeLength,
                                                                                          const double Te,
                                                                                          const double nondimTe) {
  // Q1: momentum-transfer cross-section
  // Q2: thermal-conductivity / viscosity cross-section

  double debyeCircle = PI_ * debyeLength * debyeLength;

  double Q2[3];
  Q2[0] = debyeCircle * collision::charged::rep22(nondimTe);
  Q2[1] = debyeCircle * collision::charged::rep23(nondimTe);
  Q2[2] = debyeCircle * collision::charged::rep24(nondimTe);

  // Ions are treated as point-charges (might nor be valid for molecular Ni)
  double Q1Ion[5];
  Q1Ion[0] = debyeCircle * collision::charged::att11(nondimTe);
  Q1Ion[1] = debyeCircle * collision::charged::att12(nondimTe);
  Q1Ion[2] = debyeCircle * collision::charged::att13(nondimTe);
  Q1Ion[3] = debyeCircle * collision::charged::att14(nondimTe);
  Q1Ion[4] = debyeCircle * collision::charged::att15(nondimTe);

  double Q1Neutral[5];
  double Q2Neutral[5];
  double L11, L12, L22;

  switch (gasType_) {
    case Ar:

      Q1Neutral[0] = collision::argon::eAr11(Te);
      Q1Neutral[1] = collision::argon::eAr12(Te);
      Q1Neutral[2] = collision::argon::eAr13(Te);
      Q1Neutral[3] = collision::argon::eAr14(Te);
      Q1Neutral[4] = collision::argon::eAr15(Te);

      L11 = sqrt(2.0) * X_sp[electronIndex_] * L11ee(Q2);
      L11 += X_sp[ionIndex_] * L11ea(Q1Ion);
      L11 += X_sp[neutralIndex_] * L11ea(Q1Neutral);

      L12 = sqrt(2.0) * X_sp[electronIndex_] * L12ee(Q2);
      L12 += X_sp[ionIndex_] * L12ea(Q1Ion);
      L12 += X_sp[neutralIndex_] * L12ea(Q1Neutral);

      L22 = sqrt(2.0) * X_sp[electronIndex_] * L22ee(Q2);
      L22 += X_sp[ionIndex_] * L22ea(Q1Ion);
      L22 += X_sp[neutralIndex_] * L22ea(Q1Neutral);

      break;
    case Ni:

      Q1Neutral[0] = collision::nitrogen::eNi11(Te);
      Q1Neutral[1] = collision::nitrogen::eNi12(Te);
      Q1Neutral[2] = collision::nitrogen::eNi13(Te);
      Q1Neutral[3] = collision::nitrogen::eNi14(Te);
      Q1Neutral[4] = collision::nitrogen::eNi15(Te);

      Q2Neutral[0] = collision::nitrogen::eN211(Te);
      Q2Neutral[1] = collision::nitrogen::eN212(Te);
      Q2Neutral[2] = collision::nitrogen::eN213(Te);
      Q2Neutral[3] = collision::nitrogen::eN214(Te);
      Q2Neutral[4] = collision::nitrogen::eN215(Te);

      L11 = sqrt(2.0) * X_sp[electronIndex_] * L11ee(Q2);
      L11 += X_sp[ionIndex_] * L11ea(Q1Ion);
      L11 += X_sp[neutralIndex_] * L11ea(Q1Neutral);
      L11 += X_sp[neutralIndex2_] * L11ea(Q2Neutral);

      L12 = sqrt(2.0) * X_sp[electronIndex_] * L12ee(Q2);
      L12 += X_sp[ionIndex_] * L12ea(Q1Ion);
      L12 += X_sp[neutralIndex_] * L12ea(Q1Neutral);
      L12 += X_sp[neutralIndex2_] * L12ea(Q2Neutral);

      L22 = sqrt(2.0) * X_sp[electronIndex_] * L22ee(Q2);
      L22 += X_sp[ionIndex_] * L22ea(Q1Ion);
      L22 += X_sp[neutralIndex_] * L22ea(Q1Neutral);
      L22 += X_sp[neutralIndex2_] * L22ea(Q2Neutral);

      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  double TOETC = viscosityFactor_ * kOverEtaFactor_ * sqrt(2.0 * Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
                 (L11 - L12 * L12 / L22);

  return TOETC;
}

/**/
void GasMinimalTransport::computeMixtureAverageDiffusivity(const Vector &state, const Vector &Efield,
                                                           Vector &diffusivity, bool unused) {
  diffusivity.SetSize(3);
  diffusivity = 0.0;
  computeMixtureAverageDiffusivity(&state[0], &Efield[0], &diffusivity[0], unused);
}

MFEM_HOST_DEVICE void GasMinimalTransport::computeMixtureAverageDiffusivity(const double *state, const double *Efield,
                                                                            double *diffusivity, bool unused) {
  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  double n_sp[3], X_sp[3], Y_sp[3];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp[electronIndex_] + Xeps_) / Te + (n_sp[ionIndex_] + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  // double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double binaryDiff[3 * 3];
  for (int i = 0; i < 3 * 3; i++) binaryDiff[i] = 0.0;

  switch (gasType_) {
    case Ar:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::argon::eAr11(Te);
      break;
    case Ni:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] = diffusivityFactor_ *
                                                                sqrt(Te / getMuw(electronIndex_, neutralIndex_)) /
                                                                nTotal / collision::nitrogen::eNi11(Te);
      binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies] = diffusivityFactor_ *
                                                                 sqrt(Te / getMuw(electronIndex_, neutralIndex2_)) /
                                                                 nTotal / collision::nitrogen::eN211(Te);

      binaryDiff[neutralIndex2_ + electronIndex_ * numSpecies] =
          binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies];

      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[neutralIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + neutralIndex_ * numSpecies];

  switch (gasType_) {
    case Ar:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::argon::ArAr1P11(Th);
      break;
    case Ni:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Th / getMuw(neutralIndex2_, ionIndex2_)) / nTotal /
                                                             collision::nitrogen::N2N21P11(Th);
      binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex_, ionIndex2_)) / nTotal /
                                                            collision::nitrogen::NiN21P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex_, ionIndex2_)) / nTotal /
                                                            collision::nitrogen::N2Ni1P11(Th);

      binaryDiff[ionIndex2_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies];
      binaryDiff[ionIndex_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies];
      binaryDiff[ionIndex2_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies];
      binaryDiff[electronIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Te / getMuw(ionIndex2_, electronIndex_)) / nTotal /
                                                             (collision::charged::att11(nondimTe) * debyeCircle);
      binaryDiff[ionIndex2_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex2_ * numSpecies];

      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[ionIndex_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex_ * numSpecies];
  binaryDiff[electronIndex_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                        sqrt(Te / getMuw(ionIndex_, electronIndex_)) / nTotal /
                                                        (collision::charged::att11(nondimTe) * debyeCircle);
  binaryDiff[ionIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex_ * numSpecies];

  // diffusivity.SetSize(3);
  for (int sp = 0; sp < 3; sp++) diffusivity[sp] = 0.0;
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);
}
/**/

MFEM_HOST_DEVICE void GasMinimalTransport::ComputeSourceMolecularTransport(const double *state, const double *Up,
                                                                           const double *gradUp, const double *Efield,
                                                                           double *globalTransport,
                                                                           double *speciesTransport,
                                                                           double *diffusionVelocity, double *n_sp) {
  for (int p = 0; p < SrcTrns::NUM_SRC_TRANS; p++) globalTransport[p] = 0.0;
  for (int p = 0; p < SpeciesTrns::NUM_SPECIES_COEFFS; p++)
    for (int sp = 0; sp < numSpecies; sp++) speciesTransport[sp + p * numSpecies] = 0.0;

  for (int sp = 0; sp < 3; sp++) n_sp[sp] = 0.0;
  double X_sp[3], Y_sp[3];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double Te = (twoTemperature_) ? Up[num_equation - 1] : Up[nvel_ + 1];
  double Th = Up[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp[electronIndex_] + Xeps_) / Te + (n_sp[ionIndex_] + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  // double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double Qea, Qea2;

  switch (gasType_) {
    case Ar:
      Qea = collision::argon::eAr11(Te);
      Qea2 = collision::argon::eAr11(Te);  // wont actually be used but gti checks complain if this is not set
      break;
    case Ni:
      Qea = collision::nitrogen::eNi11(Te);
      Qea2 = collision::nitrogen::eN211(Te);
      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  // double Qai = collision::gas::ArAr1P11(Th);
  double Qie = collision::charged::att11(nondimTe) * debyeCircle;

  double binaryDiff[3 * 3];
  for (int i = 0; i < 3 * 3; i++) binaryDiff[i] = 0.0;

  switch (gasType_) {
    case Ar:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::argon::eAr11(Te);
      break;
    case Ni:
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] = diffusivityFactor_ *
                                                                sqrt(Te / getMuw(electronIndex_, neutralIndex_)) /
                                                                nTotal / collision::nitrogen::eNi11(Te);
      binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies] = diffusivityFactor_ *
                                                                 sqrt(Te / getMuw(electronIndex_, neutralIndex2_)) /
                                                                 nTotal / collision::nitrogen::eN211(Te);

      binaryDiff[neutralIndex2_ + electronIndex_ * numSpecies] =
          binaryDiff[electronIndex_ + neutralIndex2_ * numSpecies];
      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[neutralIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + neutralIndex_ * numSpecies];

  switch (gasType_) {
    case Ar:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::argon::ArAr1P11(Th);
      break;
    case Ni:
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
          diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex2_, ionIndex_)) / nTotal /
                                                            collision::nitrogen::N2Ni1P11(Th);
      binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                            sqrt(Th / getMuw(neutralIndex_, ionIndex2_)) / nTotal /
                                                            collision::nitrogen::NiN21P11(Th);
      binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Th / getMuw(neutralIndex2_, ionIndex2_)) / nTotal /
                                                             collision::nitrogen::N2N21P11(Th);

      binaryDiff[ionIndex2_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex2_ * numSpecies];
      binaryDiff[ionIndex_ + neutralIndex2_ * numSpecies] = binaryDiff[neutralIndex2_ + ionIndex_ * numSpecies];
      binaryDiff[ionIndex2_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex2_ * numSpecies];
      binaryDiff[electronIndex_ + ionIndex2_ * numSpecies] = diffusivityFactor_ *
                                                             sqrt(Te / getMuw(ionIndex2_, electronIndex_)) / nTotal /
                                                             (collision::charged::att11(nondimTe) * debyeCircle);
      binaryDiff[ionIndex2_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex2_ * numSpecies];

      speciesTransport[ionIndex2_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
          mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[ionIndex2_] * Qie;

      speciesTransport[neutralIndex_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
          mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[neutralIndex_] * Qea;

      speciesTransport[neutralIndex2_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
          mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[neutralIndex2_] * Qea2;

      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  binaryDiff[ionIndex_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex_ * numSpecies];
  binaryDiff[electronIndex_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
                                                        sqrt(Te / getMuw(ionIndex_, electronIndex_)) / nTotal /
                                                        (collision::charged::att11(nondimTe) * debyeCircle);
  binaryDiff[ionIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex_ * numSpecies];

  speciesTransport[ionIndex_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
      mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[ionIndex_] * Qie;
  speciesTransport[neutralIndex_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
      mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[neutralIndex_] * Qea;  // WARNING "Qea" probably a hard-code
  speciesTransport[neutralIndex2_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
      mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[neutralIndex_] * Qea2;

  // TODO(swh): add mroe for Ni => compiler might throw a warning
  double diffusivity[numSpecies], mobility[numSpecies];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  // Apply artificial multipliers.
  if (multiply_) {
    for (int sp = 0; sp < numSpecies; sp++) {
      diffusivity[sp] *= diffMult_;
      mobility[sp] *= mobilMult_;
      speciesTransport[sp + SpeciesTrns::MF_FREQUENCY * numSpecies] *= spcsTrnsMultiplier_[SpeciesTrns::MF_FREQUENCY];
    }
  }

  globalTransport[SrcTrns::ELECTRIC_CONDUCTIVITY] =
      computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;

  double gradX[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity[sp] * gradX[sp + d * numSpecies];
      // NOTE: we'll have to handle small X case.
      diffusionVelocity[sp + d * numSpecies] = -DgradX / (X_sp[sp] + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++)
      speciesSpeed += diffusionVelocity[sp + d * numSpecies] * diffusionVelocity[sp + d * numSpecies];
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

MFEM_HOST_DEVICE void GasMinimalTransport::GetViscosities(const double *conserved, const double *primitive,
                                                          double *visc) {
  double n_sp[3], X_sp[3], Y_sp[3];
  mixture->computeSpeciesPrimitives(conserved, X_sp, Y_sp, n_sp);
  // double nTotal = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double Te = (twoTemperature_) ? primitive[num_equation - 1] : primitive[nvel_ + 1];
  double Th = primitive[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp[electronIndex_] + Xeps_) / Te + (n_sp[ionIndex_] + Xeps_) / Th;
  double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  // double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double speciesViscosity[3];
  speciesViscosity[ionIndex_] =
      viscosityFactor_ * sqrt(mw_[ionIndex_] * Th) / (collision::charged::rep22(nondimTh) * debyeCircle);

  switch (gasType_) {
    case Ar:
      speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::argon::ArAr22(Th);
      break;
    case Ni:
      speciesViscosity[neutralIndex_] =
          viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::NiNi22(Th);
      speciesViscosity[neutralIndex2_] =
          viscosityFactor_ * sqrt(mw_[neutralIndex2_] * Th) / collision::nitrogen::N2N222(Th);
      break;
    default:
      printf("Unknown gasType.");
      assert(false);
      break;
  }

  speciesViscosity[electronIndex_] = 0.0;

  visc[0] = linearAverage(X_sp, speciesViscosity);
  visc[1] = 0.0;

  // Apply artificial multipliers.
  if (multiply_) {
    visc[0] *= fluxTrnsMultiplier_[FluxTrns::VISCOSITY];
    visc[1] *= fluxTrnsMultiplier_[FluxTrns::BULK_VISCOSITY];
  }
}

/*
MFEM_HOST_DEVICE void GasMinimalTransport::GetThermalConductivities(const double *conserved, const double *primitive,
double *kappa) { double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(conserved, X_sp, Y_sp, n_sp);
  // double nTotal = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];
  collisionInputs collInputs = computeCollisionInputs(primitive, n_sp);

  double speciesViscosity[gpudata::MAXSPECIES], speciesHvyThrmCnd[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) {
      speciesViscosity[sp] = 0.0;
      speciesHvyThrmCnd[sp] = 0.0;
      continue;
    }
    speciesViscosity[sp] =
        viscosityFactor_ * sqrt(mw_[sp] * collInputs.Th) / collisionIntegral(sp, sp, 2, 2, collInputs);
    speciesHvyThrmCnd[sp] = speciesViscosity[sp] * kOverEtaFactor_ / mw_[sp];
  }
  // transportBuffer[FluxTrns::VISCOSITY] = linearAverage(X_sp, speciesViscosity);
  // transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = linearAverage(X_sp, speciesHvyThrmCnd);
  //transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;
  kappa[0]  = linearAverage(X_sp, speciesHvyThrmCnd);

  if (thirdOrderkElectron_) {
    // transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] = computeThirdOrderElectronThermalConductivity(X_sp,
collInputs); kappa[1] = computeThirdOrderElectronThermalConductivity(X_sp, collInputs);

  } else {
    // transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
    //    viscosityFactor_ * kOverEtaFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
    //    collisionIntegral(electronIndex_, electronIndex_, 2, 2, collInputs);
   kappa[1] = viscosityFactor_ * kOverEtaFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
              collisionIntegral(electronIndex_, electronIndex_, 2, 2, collInputs);

  }

}
*/

// MFEM_HOST_DEVICE void GasMinimalTransport::GetThermalConductivities(const double *conserved, const double *primitive,
//                                                                       double *kappa) {
//   std::cout<< "GasMinimalTransport::GetThermalConductivities not implemented!" << endl;
// }

//////////////////////////////////////////////////////
//////// Gas Mixture Transport
//////////////////////////////////////////////////////

GasMixtureTransport::GasMixtureTransport(GasMixture *_mixture, RunConfiguration &_runfile)
    : GasMixtureTransport(_mixture, _runfile.gasTransportInput) {}

MFEM_HOST_DEVICE GasMixtureTransport::GasMixtureTransport(GasMixture *_mixture, const GasTransportInput &inputs)
    : GasMinimalTransport(_mixture) {
  electronIndex_ = inputs.electronIndex;
  neutralIndex_ = inputs.neutralIndex;
  ionIndex_ = inputs.ionIndex;

  constantTransport_ = inputs.constantActive;
  if (constantTransport_) {
    viscosity_ = inputs.constantTransport.viscosity;
    bulkViscosity_ = inputs.constantTransport.bulkViscosity;
    thermalConductivity_ = inputs.constantTransport.thermalConductivity;
    electronThermalConductivity_ = inputs.constantTransport.electronThermalConductivity;
    for (int mixSp = 0; mixSp < numSpecies; mixSp++) {
      diffusivity_[mixSp] = inputs.constantTransport.diffusivity[mixSp];
      mtFreq_[mixSp] = inputs.constantTransport.mtFreq[mixSp];
    }
  }

  multiply_ = inputs.multiply;

  // Argon
  if (inputs.gas == "Ar" || inputs.gas == "argon") {
    gasType_ = Ar;

    if (numSpecies != 3) {
      printf("\nGas:Ar ternary transport only supports ternary mixture of Ar, Ar.+1, and E !\n");
      assert(false);
    }

    // namespace gas = collision::argon;

    neutralIndex_ = inputs.neutralIndex;
    if (neutralIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'Ar' !\n");
      assert(false);
    }
    ionIndex_ = inputs.ionIndex;
    if (ionIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'Ar.+1' !\n");
      assert(false);
    }
    electronIndex_ = inputs.electronIndex;
    if (electronIndex_ < 0) {
      printf("\nArgon ternary transport requires the species 'E' !\n");
      assert(false);
    }

    // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
    // multiplying/dividing big numbers are risky of losing precision.
    // mw_.SetSize(3);
    mw_[electronIndex_] = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
    mw_[neutralIndex_] = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
    mw_[ionIndex_] = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);

    // assumes input mass is consistent with this.
    assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-12);
    for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;

    // assumes input mass is consistent with this.
    if (abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) > 1.0e-12) {
      std::cout << "bad input mass:" << mw_[neutralIndex_] << " " << mw_[electronIndex_] << " " << mw_[ionIndex_]
                << " ni: " << neutralIndex_ << endl;
    }
    assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-12);

    // Nitrogen
  } else if (inputs.gas == "Ni" || inputs.gas == "nitrogen") {
    gasType_ = Ni;
    printf("gas mixture type nitrogen...\n");
    neutralIndex2_ = inputs.neutralIndex2;
    ionIndex2_ = inputs.ionIndex2;
  } else {
    printf("Unknown gasType.");
    assert(false);
  }

  // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
  // multiplying/dividing big numbers are risky of losing precision.
  // mw_.SetSize(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++) mw_[sp] = mixture->GetGasParams(sp, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;

  // check mass consistency
  assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-12);
  assert(abs(mw_[neutralIndex2_] - mw_[electronIndex_] - mw_[ionIndex2_]) < 1.0e-12);

  // muw_.SetSize(numSpecies);
  computeEffectiveMass(mw_, muw_);

  thirdOrderkElectron_ = inputs.thirdOrderkElectron;

  // // Identify species types
  // identifySpeciesType();
  //
  // // Determines collision type for species pair i and j.
  // identifyCollisionType();
  for (int spI = 0; spI < numSpecies; spI++)
    for (int spJ = spI; spJ < numSpecies; spJ++)
      collisionIndex_[spI + spJ * numSpecies] = inputs.collisionIndex[spI + spJ * numSpecies];

  setArtificialMultipliers(inputs);
}

// Current valid collision index tags (see reacting_flow.cpp):
// ALL:  CLMB_REP, CLMB_ATT
// Argon: AR_AR, AR_AR1P, AR_E
// Nitrogen: N2_N2, N2_NI, N2_NI1P, N2_E, NI_NI, NI_NI1P, NI_E
//
//  Note: this is very messy and should be targetted for streamlining if
//        we want to add support for more reacting gases
//
MFEM_HOST_DEVICE double GasMixtureTransport::collisionIntegral(const int _spI, const int _spJ, const int l, const int r,
                                                               const collisionInputs collInputs) {
  const int spI = (_spI > _spJ) ? _spJ : _spI;
  const int spJ = (_spI > _spJ) ? _spI : _spJ;

  double temp;
  GasColl collIdx = collisionIndex_[spI + spJ * numSpecies];
  if ((collIdx == CLMB_ATT) || (collIdx == CLMB_REP)) {
    temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.ndimTe : collInputs.ndimTh;
  } else {
    temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.Te : collInputs.Th;
  }

  switch (collIdx) {
    // Generic charged-particle section
    case CLMB_ATT: {
      if (l == 1) {
        switch (r) {
          case 1:
            return collInputs.debyeCircle * collision::charged::att11(temp);
            break;
          case 2:
            return collInputs.debyeCircle * collision::charged::att12(temp);
            break;
          case 3:
            return collInputs.debyeCircle * collision::charged::att13(temp);
            break;
          case 4:
            return collInputs.debyeCircle * collision::charged::att14(temp);
            break;
          case 5:
            return collInputs.debyeCircle * collision::charged::att15(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for attractive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      } else if (l == 2) {
        switch (r) {
          case 2:
            return collInputs.debyeCircle * collision::charged::att22(temp);
            break;
          case 3:
            return collInputs.debyeCircle * collision::charged::att23(temp);
            break;
          case 4:
            return collInputs.debyeCircle * collision::charged::att24(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for attractive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      }
    } break;

    case CLMB_REP: {
      if (l == 1) {
        switch (r) {
          case 1:
            return collInputs.debyeCircle * collision::charged::rep11(temp);
            break;
          case 2:
            return collInputs.debyeCircle * collision::charged::rep12(temp);
            break;
          case 3:
            return collInputs.debyeCircle * collision::charged::rep13(temp);
            break;
          case 4:
            return collInputs.debyeCircle * collision::charged::rep14(temp);
            break;
          case 5:
            return collInputs.debyeCircle * collision::charged::rep15(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for repulsive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      } else if (l == 2) {
        switch (r) {
          case 2:
            return collInputs.debyeCircle * collision::charged::rep22(temp);
            break;
          case 3:
            return collInputs.debyeCircle * collision::charged::rep23(temp);
            break;
          case 4:
            return collInputs.debyeCircle * collision::charged::rep24(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for repulsive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      }
    } break;

      // Argon specific section
      // AR_AR, AR_AR1P, AR_E

    case AR_AR1P: {
      if ((l == 1) && (r == 1)) {
        return collision::argon::ArAr1P11(temp);
      } else {
        printf("(%d, %d)-collision integral for Ar-Ar.1+ pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case AR_E: {
      if (l == 1) {
        switch (r) {
          case 1:
            return collision::argon::eAr11(temp);
            break;
          case 2:
            return collision::argon::eAr12(temp);
            break;
          case 3:
            return collision::argon::eAr13(temp);
            break;
          case 4:
            return collision::argon::eAr14(temp);
            break;
          case 5:
            return collision::argon::eAr15(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for repulsive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      } else {
        printf("(%d, %d)-collision integral for Ar-E pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case AR_AR: {
      if ((l == 1) && (r == 1)) {
        return collision::argon::ArAr11(temp);
      } else if ((l == 2) && (r == 2)) {
        return collision::argon::ArAr22(temp);
      } else {
        printf("(%d, %d)-collision integral for Ar-Ar pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

      // Nitrogen-specific section
      // N2_N2, N2_NI, N2_NI1P, N2_E, NI_NI, NI_NI1P, NI_E

    case NI_NI1P: {
      if ((l == 1) && (r == 1)) {
        return collision::nitrogen::NiNi1P11(temp);
      } else {
        printf("(%d, %d)-collision integral for N-N.1+ pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case N2_NI1P: {
      if ((l == 1) && (r == 1)) {
        return collision::nitrogen::N2Ni1P11(temp);
      } else {
        printf("(%d, %d)-collision integral for N2-N.1+ pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case N2_E: {
      if (l == 1) {
        switch (r) {
          case 1:
            return collision::nitrogen::eN211(temp);
            break;
          case 2:
            return collision::nitrogen::eN212(temp);
            break;
          case 3:
            return collision::nitrogen::eN213(temp);
            break;
          case 4:
            return collision::nitrogen::eN214(temp);
            break;
          case 5:
            return collision::nitrogen::eN215(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for repulsive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      } else {
        printf("(%d, %d)-collision integral for N2-E pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case NI_E: {
      if (l == 1) {
        switch (r) {
          case 1:
            return collision::nitrogen::eNi11(temp);
            break;
          case 2:
            return collision::nitrogen::eNi12(temp);
            break;
          case 3:
            return collision::nitrogen::eNi13(temp);
            break;
          case 4:
            return collision::nitrogen::eNi14(temp);
            break;
          case 5:
            return collision::nitrogen::eNi15(temp);
            break;
          default:
            printf(
                "(%d, %d)-collision integral for repulsive Coulomb potential is not supported in "
                "GasMixtureTransport! \n",
                l, r);
            assert(false);
            break;
        }
      } else {
        printf("(%d, %d)-collision integral for N-E pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case NI_NI: {
      if ((l == 1) && (r == 1)) {
        return collision::nitrogen::NiNi11(temp);
      } else if ((l == 2) && (r == 2)) {
        return collision::nitrogen::NiNi22(temp);
      } else {
        printf("(%d, %d)-collision integral for Ni-Ni pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case N2_N2: {
      if ((l == 1) && (r == 1)) {
        return collision::nitrogen::N2N211(temp);
      } else if ((l == 2) && (r == 2)) {
        return collision::nitrogen::N2N222(temp);
      } else {
        printf("(%d, %d)-collision integral for N2-N2 pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    case N2_NI: {
      if ((l == 1) && (r == 1)) {
        return collision::nitrogen::N2Ni11(temp);
      } else if ((l == 2) && (r == 2)) {
        return collision::nitrogen::N2Ni22(temp);
      } else {
        printf("(%d, %d)-collision integral for N2-N pair is not supported in GasMixtureTransport! \n", l, r);
        assert(false);
      }
    } break;

    default:
      break;
  }
  return -1;
}

MFEM_HOST_DEVICE void GasMixtureTransport::ComputeFluxMolecularTransport(const double *state, const double *gradUp,
                                                                         const double *Efield, double *transportBuffer,
                                                                         double *diffusionVelocity) {
  for (int p = 0; p < FluxTrns::NUM_FLUX_TRANS; p++) transportBuffer[p] = 0.0;

  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  collisionInputs collInputs = computeCollisionInputs(primitiveState, n_sp);

  double speciesViscosity[gpudata::MAXSPECIES], speciesHvyThrmCnd[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) {
      speciesViscosity[sp] = 0.0;
      speciesHvyThrmCnd[sp] = 0.0;
      continue;
    }
    speciesViscosity[sp] =
        viscosityFactor_ * sqrt(mw_[sp] * collInputs.Th) / collisionIntegral(sp, sp, 2, 2, collInputs);
    speciesHvyThrmCnd[sp] = speciesViscosity[sp] * kOverEtaFactor_ / mw_[sp];
  }
  transportBuffer[FluxTrns::VISCOSITY] = linearAverage(X_sp, speciesViscosity);
  transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = linearAverage(X_sp, speciesHvyThrmCnd);
  transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;

  if (thirdOrderkElectron_) {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
        computeThirdOrderElectronThermalConductivity(X_sp, collInputs);
  } else {
    transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
        viscosityFactor_ * kOverEtaFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
        collisionIntegral(electronIndex_, electronIndex_, 2, 2, collInputs);
  }

  double binaryDiff[gpudata::MAXSPECIES * gpudata::MAXSPECIES];
  // binaryDiff = 0.0;
  for (int spI = 0; spI < numSpecies - 1; spI++) {
    for (int spJ = spI + 1; spJ < numSpecies; spJ++) {
      double temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.Te : collInputs.Th;
      binaryDiff[spI + spJ * numSpecies] =
          diffusivityFactor_ * sqrt(temp / getMuw(spI, spJ)) / nTotal / collisionIntegral(spI, spJ, 1, 1, collInputs);
      binaryDiff[spJ + spI * numSpecies] = binaryDiff[spI + spJ * numSpecies];
    }
  }

  double diffusivity[gpudata::MAXSPECIES], mobility[gpudata::MAXSPECIES];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? collInputs.Te : collInputs.Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  // Apply artificial multipliers.
  if (multiply_) {
    for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) transportBuffer[t] *= fluxTrnsMultiplier_[t];
    for (int sp = 0; sp < numSpecies; sp++) {
      diffusivity[sp] *= diffMult_;
      mobility[sp] *= mobilMult_;
    }
  }

  // DenseMatrix gradX(numSpecies, dim);
  double gradX[gpudata::MAXSPECIES * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  // diffusionVelocity.SetSize(numSpecies, nvel_);
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity[sp] * gradX[sp + d * numSpecies];
      // NOTE: we'll have to handle small X case.
      diffusionVelocity[sp + d * numSpecies] = -DgradX / (X_sp[sp] + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++)
      speciesSpeed += diffusionVelocity[sp + d * numSpecies] * diffusionVelocity[sp + d * numSpecies];
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

MFEM_HOST_DEVICE double GasMixtureTransport::computeThirdOrderElectronThermalConductivity(
    const double *X_sp, const collisionInputs &collInputs) {
  double Q2[3];
  for (int r = 0; r < 3; r++) Q2[r] = collisionIntegral(electronIndex_, electronIndex_, 2, r + 2, collInputs);

  double L11 = sqrt(2.0) * X_sp[electronIndex_] * L11ee(Q2);
  double L12 = sqrt(2.0) * X_sp[electronIndex_] * L12ee(Q2);
  double L22 = sqrt(2.0) * X_sp[electronIndex_] * L22ee(Q2);
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) continue;
    double Q1[5];
    for (int r = 0; r < 5; r++) Q1[r] = collisionIntegral(sp, electronIndex_, 1, r + 1, collInputs);
    L11 += X_sp[sp] * L11ea(Q1);
    L12 += X_sp[sp] * L12ea(Q1);
    L22 += X_sp[sp] * L22ea(Q1);
  }

  return viscosityFactor_ * kOverEtaFactor_ * sqrt(2.0 * collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
         (L11 - L12 * L12 / L22);
}

MFEM_HOST_DEVICE void GasMixtureTransport::ComputeSourceMolecularTransport(const double *state, const double *Up,
                                                                           const double *gradUp, const double *Efield,
                                                                           double *globalTransport,
                                                                           double *speciesTransport,
                                                                           double *diffusionVelocity, double *n_sp) {
  for (int p = 0; p < SrcTrns::NUM_SRC_TRANS; p++) globalTransport[p] = 0.0;
  for (int p = 0; p < SpeciesTrns::NUM_SPECIES_COEFFS; p++)
    for (int sp = 0; sp < numSpecies; sp++) speciesTransport[sp + p * numSpecies] = 0.0;

  for (int sp = 0; sp < numSpecies; sp++) n_sp[sp] = 0.0;
  double X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  collisionInputs collInputs = computeCollisionInputs(Up, n_sp);

  double binaryDiff[gpudata::MAXSPECIES * gpudata::MAXSPECIES];
  // binaryDiff = 0.0;
  for (int spI = 0; spI < numSpecies - 1; spI++) {
    for (int spJ = spI + 1; spJ < numSpecies; spJ++) {
      double temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.Te : collInputs.Th;
      binaryDiff[spI + spJ * numSpecies] =
          diffusivityFactor_ * sqrt(temp / getMuw(spI, spJ)) / nTotal / collisionIntegral(spI, spJ, 1, 1, collInputs);
      binaryDiff[spJ + spI * numSpecies] = binaryDiff[spI + spJ * numSpecies];
    }
  }

  double diffusivity[gpudata::MAXSPECIES], mobility[gpudata::MAXSPECIES];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? collInputs.Te : collInputs.Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  // NOTE(kevin): collision integrals could be reused from diffusivities.. but not done that way at this point.
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) continue;
    speciesTransport[sp + SpeciesTrns::MF_FREQUENCY * numSpecies] =
        mfFreqFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * n_sp[sp] *
        collisionIntegral(sp, electronIndex_, 1, 1, collInputs);
  }

  // Apply artificial multipliers.
  if (multiply_) {
    for (int sp = 0; sp < numSpecies; sp++) {
      diffusivity[sp] *= diffMult_;
      mobility[sp] *= mobilMult_;
      speciesTransport[sp + SpeciesTrns::MF_FREQUENCY * numSpecies] *= spcsTrnsMultiplier_[SpeciesTrns::MF_FREQUENCY];
    }
  }

  globalTransport[SrcTrns::ELECTRIC_CONDUCTIVITY] =
      computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;

  double gradX[gpudata::MAXSPECIES * gpudata::MAXDIM];
  mixture->ComputeMoleFractionGradient(n_sp, gradUp, gradX);

  // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
  for (int v = 0; v < nvel_; v++)
    for (int sp = 0; sp < numSpecies; sp++) diffusionVelocity[sp + v * numSpecies] = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    // concentration-driven diffusion only determines the first dim-components.
    for (int d = 0; d < dim; d++) {
      double DgradX = diffusivity[sp] * gradX[sp + d * numSpecies];
      // NOTE: we'll have to handle small X case.
      diffusionVelocity[sp + d * numSpecies] = -DgradX / (X_sp[sp] + Xeps_);
    }
  }

  if (ambipolar) addAmbipolarEfield(mobility, n_sp, diffusionVelocity);

  addMixtureDrift(mobility, n_sp, Efield, diffusionVelocity);

  correctMassDiffusionFlux(Y_sp, diffusionVelocity);

  double charSpeed = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    double speciesSpeed = 0.0;
    // azimuthal component does not participate in flux.
    for (int d = 0; d < dim; d++)
      speciesSpeed += diffusionVelocity[sp + d * numSpecies] * diffusionVelocity[sp + d * numSpecies];
    speciesSpeed = sqrt(speciesSpeed);
    if (speciesSpeed > charSpeed) charSpeed = speciesSpeed;
    // charSpeed = max(charSpeed, speciesSpeed);
  }
  // std::cout << "max diff. vel: " << charSpeed << std::endl;
}

MFEM_HOST_DEVICE void GasMixtureTransport::GetViscosities(const double *conserved, const double *primitive,
                                                          double *visc) {
  if (constantTransport_) {
    visc[0] = viscosity_;
    visc[1] = bulkViscosity_;
    return;
  }

  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(conserved, X_sp, Y_sp, n_sp);
  // double nTotal = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  collisionInputs collInputs = computeCollisionInputs(primitive, n_sp);

  double speciesViscosity[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) {
      speciesViscosity[sp] = 0.0;
      continue;
    }
    speciesViscosity[sp] =
        viscosityFactor_ * sqrt(mw_[sp] * collInputs.Th) / collisionIntegral(sp, sp, 2, 2, collInputs);
  }  // 1.016 * 5/16 * 1/sigma^2 * (kb*m*T/pi)^(1/2)
  // sigma^2 = Q^(22) / (pi * Q^(22)*)
  // mu = 1.016 * 5/16 * 1/Q^(22) * (pi * Q^(22)*) * (kb*m*T/pi)^(1/2)
  // mu = 1.016 * 5/16 * 1/Q^(22) * pi^(1/2) * Q^(22)* * (kb*m*T)^(1/2)
  // mu = 1.016 * 5/16 * (pi*kb)^(1/2) * (m*T)^(1/2) * Q^(22)*/Q^(22)

  // 2.84a: Q^(22)* = Q^(22) / (pi * lambda_D^2)

  // collisionalIntegral returns pi*lambda_D^2 * Q (from collisionalIntegral code)
  // so, this is saying Q^(22)* = 1/(pi*lambda_D^2) which contradicts 2.84a

  // viscosityFactor_ = 5. / 16. * sqrt(PI_ * kB_);
  // 2.21 in torch doc: 5/16 * (pi *kb)^(1/2) * (m*T)^(1/2) / \bar{Q}_{aa}^(2,2)
  // but \bar{Q}_{aa}^(2,2) = pi*sigma^2 * Q_{aa}^(2,2)*
  visc[0] = linearAverage(X_sp, speciesViscosity);
  visc[1] = 0.0;

  // Apply artificial multipliers.
  if (multiply_) {
    visc[0] *= fluxTrnsMultiplier_[FluxTrns::VISCOSITY];
    visc[1] *= fluxTrnsMultiplier_[FluxTrns::BULK_VISCOSITY];
  }

  return;
}

MFEM_HOST_DEVICE void GasMixtureTransport::GetThermalConductivities(const double *conserved, const double *primitive,
                                                                    double *kappa) {
  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];

  if (constantTransport_) {
    kappa = &thermalConductivity_;
    return;
  }

  mixture->computeSpeciesPrimitives(conserved, X_sp, Y_sp, n_sp);
  // double nTotal = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];
  collisionInputs collInputs = computeCollisionInputs(primitive, n_sp);

  double speciesViscosity[gpudata::MAXSPECIES], speciesHvyThrmCnd[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp == electronIndex_) {
      speciesViscosity[sp] = 0.0;
      speciesHvyThrmCnd[sp] = 0.0;
      continue;
    }
    speciesViscosity[sp] =
        viscosityFactor_ * sqrt(mw_[sp] * collInputs.Th) / collisionIntegral(sp, sp, 2, 2, collInputs);
    speciesHvyThrmCnd[sp] = speciesViscosity[sp] * kOverEtaFactor_ / mw_[sp];
  }
  // transportBuffer[FluxTrns::VISCOSITY] = linearAverage(X_sp, speciesViscosity);
  // transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY] = linearAverage(X_sp, speciesHvyThrmCnd);
  // transportBuffer[FluxTrns::BULK_VISCOSITY] = 0.0;
  kappa[0] = linearAverage(X_sp, speciesHvyThrmCnd);

  if (thirdOrderkElectron_) {
    // transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
    //   computeThirdOrderElectronThermalConductivity(X_sp, collInputs);
    kappa[1] = computeThirdOrderElectronThermalConductivity(X_sp, collInputs);

  } else {
    // transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY] =
    //    viscosityFactor_ * kOverEtaFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
    //    collisionIntegral(electronIndex_, electronIndex_, 2, 2, collInputs);
    kappa[1] = viscosityFactor_ * kOverEtaFactor_ * sqrt(collInputs.Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
               collisionIntegral(electronIndex_, electronIndex_, 2, 2, collInputs);
  }
}

void GasMixtureTransport::computeMixtureAverageDiffusivity(const Vector &state, const Vector &Efield,
                                                           Vector &diffusivity, bool unused) {
  // diffusivity.SetSize(3);
  // diffusivity.SetSize(numSpecies);
  diffusivity.SetSize(gpudata::MAXSPECIES);  // should already be set coming in
  diffusivity = 0.0;
  computeMixtureAverageDiffusivity(&state[0], &Efield[0], &diffusivity[0], unused);
}

MFEM_HOST_DEVICE void GasMixtureTransport::computeMixtureAverageDiffusivity(const double *state, const double *Efield,
                                                                            double *diffusivity, bool unused) {
  if (constantTransport_) {
    for (int sp = 0; sp < numSpecies; sp++) {
      diffusivity[sp] = diffusivity_[sp];
    }
    return;
  }

  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  double n_sp[gpudata::MAXSPECIES], X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  collisionInputs collInputs = computeCollisionInputs(primitiveState, n_sp);

  double binaryDiff[gpudata::MAXSPECIES * gpudata::MAXSPECIES];
  for (int spI = 0; spI < numSpecies - 1; spI++) {
    for (int spJ = spI + 1; spJ < numSpecies; spJ++) {
      double temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.Te : collInputs.Th;
      binaryDiff[spI + spJ * numSpecies] =
          diffusivityFactor_ * sqrt(temp / getMuw(spI, spJ)) / nTotal / collisionIntegral(spI, spJ, 1, 1, collInputs);
      binaryDiff[spJ + spI * numSpecies] = binaryDiff[spI + spJ * numSpecies];
    }
  }

  for (int sp = 0; sp < numSpecies; sp++) diffusivity[sp] = 0.0;
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  // ambipolar diffusivity
  // NB: This form is only valid if there is ONE ion!
  if (ambipolar) {
    double mobility[gpudata::MAXSPECIES];
    for (int sp = 0; sp < numSpecies; sp++) {
      double temp = (sp == electronIndex_) ? collInputs.Te : collInputs.Th;
      mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
    }

    const double Di = diffusivity[ionIndex_];
    const double De = diffusivity[electronIndex_];

    const double mui = mobility[ionIndex_];
    const double mue = mobility[electronIndex_];
    diffusivity[ionIndex_] = (-Di * mue + De * mui) / (mui - mue);
    diffusivity[electronIndex_] = diffusivity[ionIndex_];
  }
}

MFEM_HOST_DEVICE void GasMixtureTransport::ComputeElectricalConductivity(const double *state, double &sigma) {
  if (constantTransport_) {
    sigma = electronThermalConductivity_;
    return;
  }

  double X_sp[gpudata::MAXSPECIES], Y_sp[gpudata::MAXSPECIES], n_sp[gpudata::MAXSPECIES];
  for (int sp = 0; sp < numSpecies; sp++) n_sp[sp] = X_sp[sp] = Y_sp[sp] = 0.0;
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double primitiveState[gpudata::MAXEQUATIONS];
  mixture->GetPrimitivesFromConservatives(state, primitiveState);

  collisionInputs collInputs = computeCollisionInputs(primitiveState, n_sp);

  double binaryDiff[gpudata::MAXSPECIES * gpudata::MAXSPECIES];
  // binaryDiff = 0.0;
  for (int spI = 0; spI < numSpecies - 1; spI++) {
    for (int spJ = spI + 1; spJ < numSpecies; spJ++) {
      double temp = ((spI == electronIndex_) || (spJ == electronIndex_)) ? collInputs.Te : collInputs.Th;
      binaryDiff[spI + spJ * numSpecies] =
          diffusivityFactor_ * sqrt(temp / getMuw(spI, spJ)) / nTotal / collisionIntegral(spI, spJ, 1, 1, collInputs);
      binaryDiff[spJ + spI * numSpecies] = binaryDiff[spI + spJ * numSpecies];
    }
  }

  double diffusivity[gpudata::MAXSPECIES], mobility[gpudata::MAXSPECIES];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? collInputs.Te : collInputs.Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  // Apply artificial multipliers.
  if (multiply_) {
    for (int sp = 0; sp < numSpecies; sp++) {
      mobility[sp] *= mobilMult_;
    }
  }

  sigma = computeMixtureElectricConductivity(mobility, n_sp) * MOLARELECTRONCHARGE;
}
