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

#include "chemistry.hpp"

using namespace mfem;
using namespace std;

Chemistry::Chemistry(GasMixture *mixture, RunConfiguration &config) : Chemistry(mixture, config.chemistryInput) {}

MFEM_HOST_DEVICE Chemistry::Chemistry(GasMixture *mixture, const ChemistryInput &inputs) : mixture_(mixture) {
  numEquations_ = mixture->GetNumEquations();
  numSpecies_ = mixture->GetNumSpecies();
  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  ambipolar_ = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  dim_ = mixture->GetDimension();

  model_ = inputs.model;

  electronIndex_ = inputs.electronIndex;

  numReactions_ = inputs.numReactions;
  for (int r = 0; r < numReactions_; r++) {
    reactionEnergies_[r] = inputs.reactionEnergies[r];
    detailedBalance_[r] = inputs.detailedBalance[r];
  }

  for (int r = 0; r < numReactions_; r++)
    for (int p = 0; p < 3; p++) equilibriumConstantParams_[p + r * gpudata::MAXCHEMPARAMS] = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    for (int mixSp = 0; mixSp < numSpecies_; mixSp++) {
      // int inputSp = (*mixtureToInputMap_)[mixSp];
      reactantStoich_[mixSp + r * numSpecies_] = inputs.reactantStoich[mixSp + r * numSpecies_];
      productStoich_[mixSp + r * numSpecies_] = inputs.productStoich[mixSp + r * numSpecies_];
    }

    switch (inputs.reactionModels[r]) {
      case ARRHENIUS: {
        assert(inputs.reactionInputs[r].modelParams != NULL);
        double A = inputs.reactionInputs[r].modelParams[0];
        double b = inputs.reactionInputs[r].modelParams[1];
        double E = inputs.reactionInputs[r].modelParams[2];

        reactions_[r] = new Arrhenius(A, b, E);
      } break;
      case HOFFERTLIEN: {
        assert(inputs.reactionInputs[r].modelParams != NULL);
        double A = inputs.reactionInputs[r].modelParams[0];
        double b = inputs.reactionInputs[r].modelParams[1];
        double E = inputs.reactionInputs[r].modelParams[2];

        reactions_[r] = new HoffertLien(A, b, E);
      } break;
      case TABULATED_RXN: {
        reactions_[r] = new Tabulated(inputs.reactionInputs[r].tableInput);
      } break;
      default:
        printf("Unknown reactionModel.");
        assert(false);
        break;
    }
    if (detailedBalance_[r]) {
      for (int d = 0; d < 3; d++) {
        equilibriumConstantParams_[d + r * gpudata::MAXCHEMPARAMS] =
            inputs.equilibriumConstantParams[d + r * gpudata::MAXCHEMPARAMS];
      }
    }
  }
}

MFEM_HOST_DEVICE Chemistry::~Chemistry() {
  for (int r = 0; r < numReactions_; r++) {
    if (reactions_[r] != NULL) delete reactions_[r];
  }
}

void Chemistry::computeForwardRateCoeffs(const double &T_h, const double &T_e, Vector &kfwd) {
  kfwd.SetSize(numReactions_);
  computeForwardRateCoeffs(T_h, T_e, &kfwd[0]);
  // kfwd = 0.0;
  //
  // for (int r = 0; r < numReactions_; r++) {
  //   bool isElectronInvolved = isElectronInvolvedAt(r);
  //   kfwd(r) = reactions_[r]->computeRateCoefficient(T_h, T_e, isElectronInvolved);
  // }

  return;
}

MFEM_HOST_DEVICE void Chemistry::computeForwardRateCoeffs(const double &T_h, const double &T_e, double *kfwd) {
  // kfwd.SetSize(numReactions_);
  for (int r = 0; r < numReactions_; r++) kfwd[r] = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    bool isElectronInvolved = isElectronInvolvedAt(r);
    kfwd[r] = reactions_[r]->computeRateCoefficient(T_h, T_e, isElectronInvolved);
  }

  return;
}

// NOTE: if not detailedBalance, equilibrium constant is returned as zero, though it cannot be used.
void Chemistry::computeEquilibriumConstants(const double &T_h, const double &T_e, Vector &kC) {
  kC.SetSize(numReactions_);
  computeEquilibriumConstants(T_h, T_e, &kC[0]);
  // kC = 0.0;
  //
  // for (int r = 0; r < numReactions_; r++) {
  //   double temp = (isElectronInvolvedAt(r)) ? T_e : T_h;
  //   if (detailedBalance_[r]) {
  //     kC(r) = equilibriumConstantParams_[0 + r * gpudata::MAXCHEMPARAMS] * pow(temp, equilibriumConstantParams_[1 + r
  //     * gpudata::MAXCHEMPARAMS]) *
  //             exp(-equilibriumConstantParams_[2 + r * gpudata::MAXCHEMPARAMS] / temp);
  //   }
  // }

  return;
}

MFEM_HOST_DEVICE void Chemistry::computeEquilibriumConstants(const double &T_h, const double &T_e, double *kC) {
  for (int r = 0; r < numReactions_; r++) kC[r] = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    double temp = (isElectronInvolvedAt(r)) ? T_e : T_h;
    if (detailedBalance_[r]) {
      kC[r] = equilibriumConstantParams_[0 + r * gpudata::MAXCHEMPARAMS] *
              pow(temp, equilibriumConstantParams_[1 + r * gpudata::MAXCHEMPARAMS]) *
              exp(-equilibriumConstantParams_[2 + r * gpudata::MAXCHEMPARAMS] / temp);
    }
  }

  return;
}

// compute progress rate based on mass-action law.
void Chemistry::computeProgressRate(const mfem::Vector &ns, const mfem::Vector &kfwd, const mfem::Vector &keq,
                                    mfem::Vector &progressRate) {
  progressRate.SetSize(numReactions_);
  computeProgressRate(&ns[0], &kfwd[0], &keq[0], &progressRate[0]);
  // for (int r = 0; r < numReactions_; r++) {
  //   // forward reaction rate
  //   double rate = 1.;
  //   for (int sp = 0; sp < numSpecies_; sp++) rate *= pow(ns(sp), reactantStoich_[sp + r * numSpecies_]);
  //   if (detailedBalance_[r]) {
  //     double rateBWD = 1.;
  //     for (int sp = 0; sp < numSpecies_; sp++) rateBWD *= pow(ns(sp), productStoich_[sp + r * numSpecies_]);
  //     rate -= rateBWD / keq(r);
  //   }
  //
  //   progressRate(r) = kfwd(r) * rate;
  // }
}

MFEM_HOST_DEVICE void Chemistry::computeProgressRate(const double *ns, const double *kfwd, const double *keq,
                                                     double *progressRate) {
  // progressRate.SetSize(numReactions_);
  for (int r = 0; r < numReactions_; r++) {
    // forward reaction rate
    double rate = 1.;
    for (int sp = 0; sp < numSpecies_; sp++) rate *= pow(ns[sp], reactantStoich_[sp + r * numSpecies_]);
    if (detailedBalance_[r]) {
      double rateBWD = 1.;
      for (int sp = 0; sp < numSpecies_; sp++) rateBWD *= pow(ns[sp], productStoich_[sp + r * numSpecies_]);
      rate -= rateBWD / keq[r];
    }

    progressRate[r] = kfwd[r] * rate;
  }
}

// compute creation rate based on progress rates.
void Chemistry::computeCreationRate(const mfem::Vector &progressRate, mfem::Vector &creationRate) {
  creationRate.SetSize(numSpecies_);
  computeCreationRate(&progressRate[0], &creationRate[0]);
  // creationRate = 0.;
  // for (int sp = 0; sp < numSpecies_; sp++) {
  //   for (int r = 0; r < numReactions_; r++) {
  //     creationRate(sp) += progressRate(r) * (productStoich_[sp + r * numSpecies_] - reactantStoich_[sp + r *
  //     numSpecies_]);
  //   }
  //   creationRate(sp) *= mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
  // }
  //
  // // check total created mass is 0
  // double totMass = 0.;
  // for (int sp = 0; sp < numSpecies_; sp++) totMass += creationRate(sp);
  // // NOTE: this assertion below should be made non-dimensional with dt and density
  // //   assert(fabs(totMass) < 1e-7);
}

MFEM_HOST_DEVICE void Chemistry::computeCreationRate(const double *progressRate, double *creationRate) {
  // creationRate.SetSize(numSpecies_);
  for (int sp = 0; sp < numSpecies_; sp++) creationRate[sp] = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) {
    for (int r = 0; r < numReactions_; r++) {
      creationRate[sp] +=
          progressRate[r] * (productStoich_[sp + r * numSpecies_] - reactantStoich_[sp + r * numSpecies_]);
    }
    creationRate[sp] *= mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
  }

  // check total created mass is 0
  double totMass = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) totMass += creationRate[sp];
  // NOTE: this assertion below should be made non-dimensional with dt and density
  //   assert(fabs(totMass) < 1e-7);
}
