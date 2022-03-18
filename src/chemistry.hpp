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
  // int dim_;
  int numSpecies_;
  int numActiveSpecies_;
  bool ambipolar_;
  bool twoTemperature_;

  int numReactions_;
  DenseMatrix reactantStoich_, productStoich_;  // size of (numSpecies, numReactions)

  std::vector<Reaction *> reactions_;

  std::map<int, int> *mixtureToInputMap_;
  std::map<std::string, int> *speciesMapping_;
  int electronIndex_ = -1;

  Array<bool> detailedBalance_;
  DenseMatrix equilibriumConstantParams_;

  // /*
  //   From input file, reaction/reaction%d/equation will specify this mapping.
  // */
  // std::vector<std::pair<int, int>> reactionMapping; // size of numReactions_
  // // Kevin: should we use a vector of function pointers?

  // Kevin: currently, I doubt we need mixture class here. but left it just in case.
  GasMixture *mixture_ = NULL;

 public:
  Chemistry(GasMixture *mixture, RunConfiguration config);

  ~Chemistry();

  // return Vector of reaction rate coefficients, with the size of numReaction_.
  virtual void computeForwardRateCoeffs(const double T_h, const double T_e, Vector &kfwd);

  virtual void computeEquilibriumConstants(const double T_h, const double T_e, Vector &kC);

  // return rate coefficients of (reactionIndex)-th reaction. (start from 0)
  // reactionIndex is taken from reactionMapping.right.
  // virtual Vector computeRateCoeffOf(const int reactionIndex, const double T_h, const double T_e) {};

  const double *getReactantStoichiometry(const int reactionIndex) { return reactantStoich_.GetColumn(reactionIndex); }
  const double *getProductStoichiometry(const int reactionIndex) { return productStoich_.GetColumn(reactionIndex); }

  bool isElectronInvolvedAt(const int reactionIndex) {
    return (electronIndex_ < 0) ? false : (reactantStoich_(electronIndex_, reactionIndex) != 0);
  }
};

#endif  // CHEMISTRY_HPP_
