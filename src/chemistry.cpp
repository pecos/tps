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

Chemistry::Chemistry(GasMixture* mixture, RunConfiguration& config) : mixture_(mixture) {
  numEquations_ = mixture->GetNumEquations();
  numSpecies_ = mixture->GetNumSpecies();
  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  ambipolar_ = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  dim_ = mixture->GetDimension();

  model_ = config.GetChemistryModel();

  mixtureToInputMap_ = mixture->getMixtureToInputMap();
  speciesMapping_ = mixture->getSpeciesMapping();
  electronIndex_ = (speciesMapping_->count("E")) ? (*speciesMapping_)["E"] : -1;

  numReactions_ = config.numReactions;
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
    for (int mixSp = 0; mixSp < numSpecies_; mixSp++) {
      int inputSp = (*mixtureToInputMap_)[mixSp];
      reactantStoich_(mixSp, r) = config.reactantStoich(inputSp, r);
      productStoich_(mixSp, r) = config.productStoich(inputSp, r);
    }

    // check conservations.
    {
      Vector react(numSpecies_), product(numSpecies_);
      config.reactantStoich.GetColumn(r, react);
      config.productStoich.GetColumn(r, product);

      // atom conservation.
      DenseMatrix composition;
      composition.Transpose(config.speciesComposition);
      Vector reactAtom(config.numAtoms), prodAtom(config.numAtoms);
      composition.Mult(react, reactAtom);
      composition.Mult(product, prodAtom);
      for (int a = 0; a < config.numAtoms; a++) {
        if (reactAtom(a) != prodAtom(a)) {
          grvy_printf(GRVY_ERROR, "Reaction %d does not conserve atom %d.\n", r, a);
          exit(-1);
        }
      }

      // mass conservation. (already ensured with atom but checking again.)
      double reactMass = 0.0, prodMass = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        int inputSp = (*mixtureToInputMap_)[sp];
        reactMass += react(inputSp) * mixture->GetGasParams(sp, SPECIES_MW);
        prodMass += product(inputSp) * mixture->GetGasParams(sp, SPECIES_MW);
      }
      // This may be too strict..
      if (reactMass != prodMass) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve mass.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E =/= %.8E\n", reactMass, prodMass);
        exit(-1);
      }

      // energy conservation.
      // TODO(kevin): this will need an adjustion when radiation comes into play.
      double reactEnergy = 0.0, prodEnergy = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        int inputSp = (*mixtureToInputMap_)[sp];
        reactEnergy += react(inputSp) * mixture->GetGasParams(sp, FORMATION_ENERGY);
        prodEnergy += product(inputSp) * mixture->GetGasParams(sp, FORMATION_ENERGY);
      }
      // This may be too strict..
      if (reactEnergy + reactionEnergies_(r) != prodEnergy) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve energy.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E + %.8E = %.8E =/= %.8E\n",
                    reactEnergy, reactionEnergies_(r), reactEnergy + reactionEnergies_(r), prodEnergy);
        exit(-1);
      }
    }

    switch (config.reactionModels[r]) {
      case ARRHENIUS: {
        double A = (config.reactionModelParams[r])[0];
        double b = (config.reactionModelParams[r])[1];
        double E = (config.reactionModelParams[r])[2];
        reactions_[r] = new Arrhenius(A, b, E);
      } break;
      case HOFFERTLIEN: {
        double A = (config.reactionModelParams[r])[0];
        double b = (config.reactionModelParams[r])[1];
        double E = (config.reactionModelParams[r])[2];
        reactions_[r] = new HoffertLien(A, b, E);
      } break;
    }
    if (detailedBalance_[r]) {
      for (int d = 0; d < 3; d++) {
        equilibriumConstantParams_(r, d) = (config.equilibriumConstantParams[r])[d];
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
}

void Chemistry::computeForwardRateCoeffs(const double T_h, const double T_e, Vector& kfwd) {
  kfwd.SetSize(numReactions_);
  kfwd = 0.0;

  for (int r = 0; r < numReactions_; r++) {
    bool isElectronInvolved = isElectronInvolvedAt(r);
    kfwd(r) = reactions_[r]->computeRateCoefficient(T_h, T_e, isElectronInvolved);
  }

  return;
}

// NOTE: if not detailedBalance, equilibrium constant is returned as zero, though it cannot be used.
void Chemistry::computeEquilibriumConstants(const double T_h, const double T_e, Vector& kC) {
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

// compute progress rate based on mass-action law.
void Chemistry::computeProgressRate(const mfem::Vector& ns, const mfem::Vector& kfwd, const mfem::Vector& keq,
                                    mfem::Vector& progressRate) {
  progressRate.SetSize(numReactions_);
  for (int r = 0; r < numReactions_; r++) {
    // forward reaction rate
    double rate = 1.;
    for (int sp = 0; sp < numSpecies_; sp++) rate *= pow(ns(sp), reactantStoich_(sp, r));
    if (detailedBalance_[r]) {
      double rateBWD = 1.;
      for (int sp = 0; sp < numSpecies_; sp++) rateBWD *= pow(ns(sp), productStoich_(sp, r));
      rate -= rateBWD / keq(r);
    }

    progressRate(r) = kfwd(r) * rate;
  }
}

// compute creation rate based on progress rates.
void Chemistry::computeCreationRate(const mfem::Vector& progressRate, mfem::Vector& creationRate) {
  creationRate.SetSize(numSpecies_);
  creationRate = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) {
    for (int r = 0; r < numReactions_; r++) {
      creationRate(sp) += progressRate(r) * (productStoich_(sp, r) - reactantStoich_(sp, r));
    }
    creationRate(sp) *= mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
  }

  // check total created mass is 0
  double totMass = 0.;
  for (int sp = 0; sp < numSpecies_; sp++) totMass += creationRate(sp);
  // NOTE: this assertion below should be made non-dimensional with dt and density
  //   assert(fabs(totMass) < 1e-7);
}
