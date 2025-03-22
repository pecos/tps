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

  // TODO: add gas type in inputs and reinstate checks here
  //gasType_ = inputs.gas;

  if (inputs.gas=="Ar" || inputs.gas=="argon") {
    gasType_ = Ar;
  } else if (inputs.gas=="Ni" || inputs.gas=="nitrogen") {
    gasType_ = Ni;
  } else {
   printf("Unknown gasType.");
   assert(false);
  }  
  
  // if (!ambipolar) {
  //   grvy_printf(GRVY_ERROR, "\nGas ternary transport currently supports ambipolar condition only. Set
  //   plasma_models/ambipolar = true.\n"); exit(ERROR);
  // }
  
  //if (numSpecies != 3) {
  //  printf("\nGas ternary transport only supports ternary mixture of Ar, Ar.+1, and E !\n");
  //  assert(false);
  //}

  // std::map<std::string, int> *speciesMapping = mixture->getSpeciesMapping();
  // if (speciesMapping->count("Ar")) {
  //   neutralIndex_ = (*speciesMapping)["Ar"];
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nGas ternary transport requires the species 'Ar' !\n");
  //   exit(ERROR);
  // }
  // if (speciesMapping->count("Ar.+1")) {
  //   ionIndex_ = (*speciesMapping)["Ar.+1"];
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nGas ternary transport requires the species 'Ar.+1' !\n");
  //   exit(ERROR);
  // }
  // if (speciesMapping->count("E")) {
  //   electronIndex_ = (*speciesMapping)["E"];
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nGas ternary transport requires the species 'E' !\n");
  //   exit(ERROR);
  // }

  /*
  neutralIndex_ = inputs.neutralIndex;
  if (neutralIndex_ < 0) {
    printf("\nGas ternary transport requires the species 'Ar' !\n");
    assert(false);
  }
  ionIndex_ = inputs.ionIndex;
  if (ionIndex_ < 0) {
    printf("\nGas ternary transport requires the species 'Ar.+1' !\n");
    assert(false);
  }
  electronIndex_ = inputs.electronIndex;
  if (electronIndex_ < 0) {
    printf("\nGas ternary transport requires the species 'E' !\n");
    assert(false);
  }
  */

  // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
  // multiplying/dividing big numbers are risky of losing precision.
  // mw_.SetSize(3);
  mw_[electronIndex_] = mixture->GetGasParams(electronIndex_, GasParams::SPECIES_MW);
  mw_[neutralIndex_] = mixture->GetGasParams(neutralIndex_, GasParams::SPECIES_MW);
  mw_[ionIndex_] = mixture->GetGasParams(ionIndex_, GasParams::SPECIES_MW);
  // assumes input mass is consistent with this.
  assert(abs(mw_[neutralIndex_] - mw_[electronIndex_] - mw_[ionIndex_]) < 1.0e-15);
  for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;
  // mA_ /= AVOGADRONUMBER;
  // mI_ /= AVOGADRONUMBER;

  // muw_.SetSize(3);
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
                                                                           const double *Efield,
                                                                           double *transportBuffer,
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
      //speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::NiNi22(Th);   
      speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::N2N222(Th);    
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


  switch (gasType_) {
    case Ar: 
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::argon::eAr11(Te);
      break;
    case Ni:
      //binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eNi11(Te);     
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eN211(Te);       
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
      //binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::N2N21P11(Th); 
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

  double diffusivity[3], mobility[3];
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

// HERE HERE HERE
MFEM_HOST_DEVICE double GasMinimalTransport::computeThirdOrderElectronThermalConductivity(const double *X_sp,
                                                                                            const double debyeLength,
                                                                                            const double Te,
                                                                                            const double nondimTe) {
  // Q1: momentum-transfer cross-section
  // Q2: thermal-conductivity / viscosity cross-section
  
  double debyeCircle = PI_ * debyeLength * debyeLength;
  // std::cout << "LD: " << debyeLength << std::endl;
  double Q2[3];
  Q2[0] = debyeCircle * collision::charged::rep22(nondimTe);
  Q2[1] = debyeCircle * collision::charged::rep23(nondimTe);
  Q2[2] = debyeCircle * collision::charged::rep24(nondimTe);
  // std::cout << "Q2r: " << Q2(0) << ",\t" << Q2(1) << ",\t" << Q2(2) << std::endl;

  double Q1Ion[5];
  Q1Ion[0] = debyeCircle * collision::charged::att11(nondimTe);
  Q1Ion[1] = debyeCircle * collision::charged::att12(nondimTe);
  Q1Ion[2] = debyeCircle * collision::charged::att13(nondimTe);
  Q1Ion[3] = debyeCircle * collision::charged::att14(nondimTe);
  Q1Ion[4] = debyeCircle * collision::charged::att15(nondimTe);
  //   std::cout << "Q1i: ";
  // for (int i = 0; i < 5; i++) std::cout << Q1Ion(i) << ",\t";
  // std::cout << std::endl;

  double Q1Neutral[5];
  Q1Neutral[0] = collision::argon::eAr11(Te);
  Q1Neutral[1] = collision::argon::eAr12(Te);
  Q1Neutral[2] = collision::argon::eAr13(Te);
  Q1Neutral[3] = collision::argon::eAr14(Te);
  Q1Neutral[4] = collision::argon::eAr15(Te);
  
  //   std::cout << "Q1A: ";
  // for (int i = 0; i < 5; i++) std::cout << Q1Neutral(i) << ",\t";
  // std::cout << std::endl;

  double L11 = sqrt(2.0) * X_sp[electronIndex_] * L11ee(Q2);
  L11 += X_sp[ionIndex_] * L11ea(Q1Ion);
  L11 += X_sp[neutralIndex_] * L11ea(Q1Neutral);
  
  double L12 = sqrt(2.0) * X_sp[electronIndex_] * L12ee(Q2);
  L12 += X_sp[ionIndex_] * L12ea(Q1Ion);
  L12 += X_sp[neutralIndex_] * L12ea(Q1Neutral);
  
  double L22 = sqrt(2.0) * X_sp[electronIndex_] * L22ee(Q2);
  L22 += X_sp[ionIndex_] * L22ea(Q1Ion);
  L22 += X_sp[neutralIndex_] * L22ea(Q1Neutral);
  
  // std::cout << "L11: " << L11 << std::endl;
  // std::cout << "L12: " << L12 << std::endl;
  // std::cout << "L22: " << L22 << std::endl;

  return viscosityFactor_ * kOverEtaFactor_ * sqrt(2.0 * Te / mw_[electronIndex_]) * X_sp[electronIndex_] /
         (L11 - L12 * L12 / L22);
}

void GasMinimalTransport::computeMixtureAverageDiffusivity(const Vector &state, const Vector &Efield,
                                                             Vector &diffusivity, bool unused) {
  diffusivity.SetSize(3);
  diffusivity = 0.0;
  computeMixtureAverageDiffusivity(&state[0], &Efield[0], &diffusivity[0], unused);

  // Vector primitiveState(num_equation);
  // mixture->GetPrimitivesFromConservatives(state, primitiveState);
  //
  // Vector n_sp(3), X_sp(3), Y_sp(3);
  // mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  // double nTotal = 0.0;
  // for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp(sp);
  //
  // double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  // double Th = primitiveState[nvel_ + 1];
  //
  // // Add Xeps to avoid zero number density case.
  // double nOverT = (n_sp(electronIndex_) + Xeps_) / Te + (n_sp(ionIndex_) + Xeps_) / Th;
  // double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  // double debyeCircle = PI_ * debyeLength * debyeLength;
  //
  // double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  // // double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;
  //
  // DenseMatrix binaryDiff(3);
  // binaryDiff = 0.0;
  // binaryDiff(electronIndex_, neutralIndex_) =
  //     diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::gas::eAr11(Te);
  // binaryDiff(neutralIndex_, electronIndex_) = binaryDiff(electronIndex_, neutralIndex_);
  // binaryDiff(ionIndex_, neutralIndex_) =
  //     diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::gas::ArAr1P11(Th);
  // binaryDiff(neutralIndex_, ionIndex_) = binaryDiff(ionIndex_, neutralIndex_);
  // binaryDiff(electronIndex_, ionIndex_) = diffusivityFactor_ * sqrt(Te / getMuw(ionIndex_, electronIndex_)) / nTotal
  // /
  //                                         (collision::charged::att11(nondimTe) * debyeCircle);
  // binaryDiff(ionIndex_, electronIndex_) = binaryDiff(electronIndex_, ionIndex_);
  //
  // diffusivity.SetSize(3);
  // diffusivity = 0.0;
  // CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);
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
      //binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eNi11(Te);
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eN211(Te);     
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
      //binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::N2N21P11(Th);      
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
      //binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eNi11(Te);
      binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::nitrogen::eN211(Te);        
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
      //binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
      //  diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::NiNi1P11(Th);
      binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
        diffusivityFactor_ * sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::nitrogen::N2N21P11(Th);  
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

  double diffusivity[3], mobility[3];
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);

  for (int sp = 0; sp < numSpecies; sp++) {
    double temp = (sp == electronIndex_) ? Te : Th;
    mobility[sp] = qeOverkB_ * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) / temp * diffusivity[sp];
  }

  speciesTransport[ionIndex_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
      mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[ionIndex_] * Qie;
  speciesTransport[neutralIndex_ + SpeciesTrns::MF_FREQUENCY * numSpecies] =
      mfFreqFactor_ * sqrt(Te / mw_[electronIndex_]) * n_sp[neutralIndex_] * Qea;

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
      //speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::NiNi22(Th);
      speciesViscosity[neutralIndex_] = viscosityFactor_ * sqrt(mw_[neutralIndex_] * Th) / collision::nitrogen::N2N222(Th);  
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

//////////////////////////////////////////////////////
//////// Gas Mixture Transport
//////////////////////////////////////////////////////

GasMixtureTransport::GasMixtureTransport(GasMixture *_mixture, RunConfiguration &_runfile)
    : GasMixtureTransport(_mixture, _runfile.gasTransportInput) {}

MFEM_HOST_DEVICE GasMixtureTransport::GasMixtureTransport(GasMixture *_mixture, const GasTransportInput &inputs)
    : GasMinimalTransport(_mixture) {
  ionIndex_ = inputs.ionIndex;

  //gasType_ = inputs.gas;
  if (inputs.gas=="Ar" || inputs.gas=="argon") {
    gasType_ = Ar;
  } else if (inputs.gas=="Ni" || inputs.gas=="nitrogen") {
    gasType_ = Ni;
  } else {
   printf("Unknown gasType.");
   assert(false);
  }  
  
  // numAtoms_(_runfile.numAtoms),
  // atomMap_(_runfile.atomMap),
  // speciesNames_(_runfile.speciesNames) {
  // std::map<std::string, int> *speciesMapping = mixture->getSpeciesMapping();
  // if (speciesMapping->count("E")) {
  //   electronIndex_ = (*speciesMapping)["E"];
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nGas ternary transport requires the species 'E' !\n");
  //   exit(ERROR);
  // }

  /*
  electronIndex_ = inputs.electronIndex;
  if (electronIndex_ < 0) {
    printf("\nGas ternary transport requires the species 'E' !\n");
    assert(false);
  }
  */

  // composition_.SetSize(numSpecies, numAtoms_);
  // // mixtureToInputMap_ = mixture->getMixtureToInputMap();
  // for (int sp = 0; sp < numSpecies; sp++) {
  //   // int inputSp = (*mixtureToInputMap_)[sp];
  //   for (int a = 0; a < numAtoms_; a++) {
  //     composition_(sp, a) = _runfile.speciesComposition(sp, a);
  //   }
  // }

  // TODO(kevin): need to factor out avogadro numbers throughout all transport property.
  // multiplying/dividing big numbers are risky of losing precision.
  // mw_.SetSize(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++) mw_[sp] = mixture->GetGasParams(sp, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numSpecies; sp++) mw_[sp] /= AVOGADRONUMBER;

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

// HERE HERE HERE check if att stuff is argon hard-coded, Ar eAr def is...
MFEM_HOST_DEVICE double GasMixtureTransport::collisionIntegral(const int _spI, const int _spJ, const int l,
                                                                 const int r, const collisionInputs collInputs) {
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
    default:
      break;
  }
  return -1;
}

MFEM_HOST_DEVICE void GasMixtureTransport::ComputeFluxMolecularTransport(const double *state, const double *gradUp,
                                                                           const double *Efield,
                                                                           double *transportBuffer,
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
  }
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
  diffusivity.SetSize(3);
  diffusivity = 0.0;
  computeMixtureAverageDiffusivity(&state[0], &Efield[0], &diffusivity[0], unused);
}

/*
MFEM_HOST_DEVICE void GasMixtureTransport::computeMixtureAverageDiffusivity(const double *state, double *diffusivity)
{ double primitiveState[gpudata::MAXEQUATIONS]; mixture->GetPrimitivesFromConservatives(state, primitiveState);

  double n_sp[3], X_sp[3], Y_sp[3];
  mixture->computeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);

  double nTotal = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) nTotal += n_sp[sp];

  double Te = (twoTemperature_) ? primitiveState[num_equation - 1] : primitiveState[nvel_ + 1];
  double Th = primitiveState[nvel_ + 1];

  // Add Xeps to avoid zero number density case.
  double nOverT = (n_sp[electronIndex_] + Xeps_) / Te + (n_sp[ionIndex_] + Xeps_) / Th;
  // double debyeLength = sqrt(debyeFactor_ / AVOGADRONUMBER / nOverT);
  double debyeLength;
  debyeLength = debyeFactor_;
  debyeLength /= AVOGADRONUMBER;
  debyeLength /= nOverT;
  debyeLength = std::sqrt(debyeLength);
  double debyeCircle = PI_ * debyeLength * debyeLength;

  double nondimTe = debyeLength * 4.0 * PI_ * debyeFactor_ * Te;
  // double nondimTh = debyeLength * 4.0 * PI_ * debyeFactor_ * Th;

  double binaryDiff[3 *3];
  for (int i = 0; i < 3 * 3; i++) binaryDiff[i] = 0.0;
  binaryDiff[electronIndex_ + neutralIndex_ * numSpecies] =
    diffusivityFactor_ * std::sqrt(Te / getMuw(electronIndex_, neutralIndex_)) / nTotal / collision::gas::eAr11(Te);
  binaryDiff[neutralIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + neutralIndex_ * numSpecies];
  binaryDiff[neutralIndex_ + ionIndex_ * numSpecies] =
    diffusivityFactor_ * std::sqrt(Th / getMuw(neutralIndex_, ionIndex_)) / nTotal / collision::gas::ArAr1P11(Th);
  binaryDiff[ionIndex_ + neutralIndex_ * numSpecies] = binaryDiff[neutralIndex_ + ionIndex_ * numSpecies];
  binaryDiff[electronIndex_ + ionIndex_ * numSpecies] = diffusivityFactor_ *
    std::sqrt(Te / getMuw(ionIndex_, electronIndex_)) / nTotal /
                                                        (collision::charged::att11(nondimTe) * debyeCircle);
  binaryDiff[ionIndex_ + electronIndex_ * numSpecies] = binaryDiff[electronIndex_ + ionIndex_ * numSpecies];

  // diffusivity.SetSize(3);
  for (int sp = 0; sp < 3; sp++) diffusivity[sp] = 0.0;
  CurtissHirschfelder(X_sp, Y_sp, binaryDiff, diffusivity);
  for (int sp = 0; sp < 3; sp++) {
  }
}
*/

MFEM_HOST_DEVICE void GasMixtureTransport::computeMixtureAverageDiffusivity(const double *state, const double *Efield,
                                                                              double *diffusivity, bool unused) {
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
