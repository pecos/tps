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
#ifndef CHEMISTRY_HPP_
#define CHEMISTRY_HPP_

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <tps_config.h>

#include <mfem.hpp>
#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "reaction.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

class Chemistry {
 protected:
  int numEquations_;
  int dim_;
  int numSpecies_;
  int numActiveSpecies_;
  bool ambipolar_;
  bool twoTemperature_;

  int numReactions_ = 0;
  DenseMatrix reactantStoich_, productStoich_;  // size of (numSpecies, numReactions)

  std::vector<Reaction *> reactions_;
  Vector reactionEnergies_;

  std::map<int, int> *mixtureToInputMap_;
  std::map<std::string, int> *speciesMapping_;
  int electronIndex_ = -1;

  Array<bool> detailedBalance_;
  DenseMatrix equilibriumConstantParams_;

  ChemistryModel model;

  // /*
  //   From input file, reaction/reaction%d/equation will specify this mapping.
  // */
  // std::vector<std::pair<int, int>> reactionMapping; // size of numReactions_
  // // Kevin: should we use a vector of function pointers?

  // Kevin: currently, I doubt we need mixture class here. but left it just in case.
  GasMixture *mixture_ = NULL;

 public:
  Chemistry(GasMixture *mixture, RunConfiguration &config);

  ~Chemistry();

  // return Vector of reaction rate coefficients, with the size of numReaction_.
  // WARNING(marc) I have removed "virtual" qualifier here assuming these functions will not
  // change for child classes. Correct if wrong
  void computeForwardRateCoeffs(const double T_h, const double T_e, Vector &kfwd);

  void computeEquilibriumConstants(const double T_h, const double T_e, Vector &kC);

  // return rate coefficients of (reactionIndex)-th reaction. (start from 0)
  // reactionIndex is taken from reactionMapping.right.
  // virtual Vector computeRateCoeffOf(const int reactionIndex, const double T_h, const double T_e) {};

  const double *getReactantStoichiometry(const int reactionIndex) { return reactantStoich_.GetColumn(reactionIndex); }
  const double *getProductStoichiometry(const int reactionIndex) { return productStoich_.GetColumn(reactionIndex); }

  virtual void computeCreationRate(const Vector &ns, const Vector &kfwd, const Vector &keq, Vector &creationRate){}

  bool isElectronInvolvedAt(const int reactionIndex) {
    return (electronIndex_ < 0) ? false : (reactantStoich_(electronIndex_, reactionIndex) != 0);
  }
  
  int GetElectronIndex(){return electronIndex_;}
  
  int GetNumReactions(){return numReactions_;}

  ChemistryModel getChemistryModel() { return model; }
  
  const DenseMatrix &GetReactantStoichiometryVector()const{return reactantStoich_;}
  const DenseMatrix &getProductStoichiometryVector()const{return productStoich_;}
  Reaction *GetReaction(int r){return reactions_[r];}
  
  Array<bool> &GetDetailBalanceArray(){return detailedBalance_;}
  DenseMatrix &GetEquilibriumConstantsMatrix(){return equilibriumConstantParams_;}
  
#ifdef _GPU_
  static MFEM_HOST_DEVICE void computeForwardRateCoeffs_gpu(const double &T_h, 
                                                            const double &T_e, 
                                                            double *kfwd,
                                                            const int &electronIndex,
                                                            const int &numSpecies,
                                                            const int &numReactions,
                                                            const double *reactantStoich,
                                                            const ReactionModel *reactionsModel,
                                                            const reactionConstants *constants
                                                           ) {
    for (int r = 0; r < numReactions; r++) {
      bool isElectronInvolved;
      if (electronIndex < 0) {
        isElectronInvolved = false;
      }else {
        isElectronInvolved = reactantStoich[electronIndex + numSpecies* r] != 0;
      }

      switch (reactionsModel[r]) {
        case ReactionModel::ARRHENIUS:
          kfwd[r] = Arrhenius::computeRateCoefficient_gpu(T_h, 
                                                        T_e, 
                                                        isElectronInvolved,
                                                        constants[r].A_,
                                                        constants[r].b_,
                                                        constants[r].E_);

          break;
        case ReactionModel::HOFFERTLIEN:
          kfwd[r] = HoffertLien::computeRateCoefficient_gpu(T_h, 
                                                          T_e, 
                                                          isElectronInvolved,
                                                          constants[r].A_,
                                                          constants[r].b_,
                                                          constants[r].E_);
          break;
        default:
          printf("[ERROR] Chemistry::computeForwardRateCoeffs_gpu(): bad ReactionModel");
          break;
      }
    }
  }
  
  static MFEM_HOST_DEVICE void computeEquilibriumConstants_gpu(const double &T_h, 
                                                            const double &T_e, 
                                                            double *kC,
                                                            const bool *detailedBalance,
                                                            const int &electronIndex,
                                                            const int &numSpecies,
                                                            const int &numReactions,
                                                            const double *reactantStoich,
                                                            const double *equilibriumConstantParams ) {
    for (int r = 0; r < numReactions; r++) {
      bool isElectronInvolved;
      if (electronIndex < 0) {
        isElectronInvolved = false;
      }else {
        isElectronInvolved = reactantStoich[electronIndex + numSpecies* r] != 0;
      }
      double temp = isElectronInvolved ? T_e : T_h;
      if (detailedBalance[r]) {
        kC[r] = equilibriumConstantParams[r] * pow(temp, equilibriumConstantParams[r +numReactions]) *
                exp(-equilibriumConstantParams[r + numReactions * 2] / temp);
      }else {
        kC[r] = 0.;
      }
    }
  }
#endif // _GPU_
};

// Implementation of the Mass-action-law class
class MassActionLaw : public Chemistry {
 public:
  MassActionLaw(GasMixture *mixture, RunConfiguration &config);
  ~MassActionLaw(){}

  virtual void computeCreationRate(const Vector &ns, const Vector &kfwd, const Vector &keq, Vector &creationRate);
  
#ifdef _GPU_
  static MFEM_HOST_DEVICE void computeCreationRate_gpu(const double *ns,
                                                       const double *kfwd,
                                                       const double *keq,
                                                       double *creationRate,
                                                       const double *gasParams,
                                                       const double *reactantStoich,
                                                       const double *productsStoich,
                                                       const int &numReactions,
                                                       const int &numSpecies,
                                                       const int &thrd,
                                                       const int &maxThreads ) {
    MFEM_SHARED double progressRate[20];
    for (int r = thrd; r < numReactions; r += maxThreads) {
      // forward reaction rate
      double rateFWD = 1., rateBWD = 1.;
      for (int sp = 0; sp < numSpecies; sp++) rateFWD *= pow(ns[sp], reactantStoich[sp + r * numSpecies]);
      for (int sp = 0; sp < numSpecies; sp++) rateBWD *= pow(ns[sp], productsStoich[sp + r * numSpecies]);
      progressRate[r] = kfwd[r] * (rateFWD - rateBWD / keq[r]);
    }
    
    for (int sp = thrd; sp < numSpecies; sp += maxThreads ) creationRate[sp] = 0.;
    MFEM_SYNC_THREAD;

    for (int sp = thrd; sp < numSpecies; sp += maxThreads ) {
      for (int r = 0; r < numReactions; r++) {
        creationRate[sp] += progressRate[r] * (productsStoich[sp + r * numSpecies] - 
                                               reactantStoich[sp + r * numSpecies]);
      }
      creationRate[sp] *= gasParams[sp + numSpecies * (int)GasParams::SPECIES_MW];
    }
    MFEM_SYNC_THREAD;

    // check total created mass is 0
//     double totMass = 0.;
//     for (int sp = 0; sp < numSpecies_; sp++) totMass += creationRate(sp);
  }
#endif // _GPU_
};

#endif  // CHEMISTRY_HPP_
