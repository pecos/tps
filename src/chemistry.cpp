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

#include "chemistry.hpp"

using namespace mfem;
using namespace std;

Chemistry::Chemistry(GasMixture *mixture, RunConfiguration &config) : mixture_(mixture) {
  numEquations_ = mixture->GetNumEquations();
  numSpecies_ = mixture->GetNumSpecies();
  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  ambipolar_ = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  dim_ = mixture->GetDimension();

  model = config.GetChemistryModel();

  mixtureToInputMap_ = mixture->getMixtureToInputMap();
  speciesMapping_ = mixture->getSpeciesMapping();
  electronIndex_ = (speciesMapping_->count("E")) ? (*speciesMapping_)["E"] : -1;

  // TODO: make tps input parser accessible to all classes.
  // TODO: reaction classes read input options directly in their initialization.
  numReactions_ = config.numReactions;
  // std::cout << "number of reactions: " << numReactions_ << std::endl;
  reactionEnergies_.SetSize(numReactions_);
  reactionEnergies_ = config.reactionEnergies;

  detailedBalance_.SetSize(numReactions_);
  detailedBalance_ = config.detailedBalance;

  reactions_.resize(numReactions_);
  reactantStoich_.SetSize(numSpecies_, numReactions_);
  productStoich_.SetSize(numSpecies_, numReactions_);
  equilibriumConstantParams_.SetSize(numReactions_, 3);
  equilibriumConstantParams_ = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    // std::cout << "reactions-" << r  << std::endl;
    for (int mixSp = 0; mixSp < numSpecies_; mixSp++) {
      int inputSp = (*mixtureToInputMap_)[mixSp];
      // std::cout << mixSp << " : " << inputSp << std::endl;
      reactantStoich_(mixSp, r) = config.reactantStoich(inputSp, r);
      productStoich_(mixSp, r) = config.productStoich(inputSp, r);
      // std::cout << reactantStoich_(mixSp, r) << " - " << productStoich_(mixSp, r) << std::endl;
    }
    switch (config.reactionModels[r]) {
      case ARRHENIUS: {
        // std::cout << "reactions-" << r << ": Arrhenius" << std::endl;
        double A = (config.reactionModelParams[r])[0];
        double b = (config.reactionModelParams[r])[1];
        double E = (config.reactionModelParams[r])[2];
        reactions_[r] = new Arrhenius(A, b, E);
        // std::cout << A << ", " << b << ", " << E << std::endl;
      } break;
      case HOFFERTLIEN: {
        // std::cout << "reactions-" << r << ": HoffertLien" << std::endl;
        double A = (config.reactionModelParams[r])[0];
        double b = (config.reactionModelParams[r])[1];
        double E = (config.reactionModelParams[r])[2];
        reactions_[r] = new HoffertLien(A, b, E);
        // std::cout << A << ", " << b << ", " << E << std::endl;
      } break;
    }
    if (detailedBalance_[r]) {
      // std::cout << "detail balanced: ";
      for (int d = 0; d < 3; d++) {
        equilibriumConstantParams_(r, d) = (config.equilibriumConstantParams[r])[d];
        // std::cout << equilibriumConstantParams_(r, d);
      }
      std::cout << std::endl;
    }
  }
}

Chemistry::~Chemistry() {
//   if (mixture_ != NULL) delete mixture_;
//   if (mixtureToInputMap_ != NULL) delete mixtureToInputMap_;
//   if (speciesMapping_ != NULL) delete speciesMapping_;
  for (int r = 0; r < numReactions_; r++) {
    if (reactions_[r] != NULL) delete reactions_[r];
  }
};

void Chemistry::computeForwardRateCoeffs(const double T_h, const double T_e, Vector &kfwd) {
  kfwd.SetSize(numReactions_);
  kfwd = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    bool isElectronInvolved = isElectronInvolvedAt(r);
    kfwd = reactions_[r]->computeRateCoefficient(T_h, T_e, isElectronInvolved);
  }

  return;
}

// NOTE: if not detailedBalance, equilibrium constant is returned as zero, though it cannot be used.
void Chemistry::computeEquilibriumConstants(const double T_h, const double T_e, Vector &kC) {
  kC.SetSize(numReactions_);
  kC = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    double temp = (isElectronInvolvedAt(r)) ? T_e : T_h;
    if (detailedBalance_[r]) {
      kC(r) = equilibriumConstantParams_(r, 0) * pow(temp, equilibriumConstantParams_(r, 1)) *
              exp(-equilibriumConstantParams_(r, 2) / temp);
    }
  }

  return;
}

MassActionLaw::MassActionLaw(GasMixture* mixture, RunConfiguration &config):
  Chemistry(mixture, config)
{
}


void MassActionLaw::computeCreationRate(const mfem::Vector& ns, const mfem::Vector& kfwd,
                                       const mfem::Vector& keq, mfem::Vector& creationRate)
{
  Vector progressRate(numReactions_);
  for (int r = 0; r < numReactions_; r++) {
    // forward reaction rate
    double rateFWD = 1., rateBWD = 1.;
    for (int sp = 0; sp < numSpecies_; sp++) rateFWD *= pow(ns(sp), reactantStoich_(sp,r));
    for (int sp = 0; sp < numSpecies_; sp++) rateBWD *= pow(ns(sp), productStoich_(sp,r));
    progressRate(r) = kfwd(r) * (rateFWD - rateBWD / keq(r) );
  }

  creationRate.SetSize(numSpecies_);
  creationRate = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) {
    for (int r = 0; r < numReactions_; r++) {
      creationRate(sp) += progressRate(r) * (productStoich_(sp,r) - reactantStoich_(sp,r));
    }
    creationRate(sp) *= mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
  }

  // check total created mass is 0
  double totMass = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) totMass += creationRate(sp);
  // NOTE: this assertion below should be made non-dimensional with dt and density
//   assert(fabs(totMass) < 1e-7);
}
