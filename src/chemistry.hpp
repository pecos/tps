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

#include <mfem/general/forall.hpp>
#include "dataStructures.hpp"
#include "run_configuration.hpp"
#include "equation_of_state.hpp"
#include <mfem.hpp>

using namespace mfem;
using namespace std;

class Chemistry{
  protected:
    int numEquations_;
    int dim_;
    int numSpecies_;
    int numActiveSpecies_;
    bool ambipolar_;
    bool twoTemperature_;

    int numReactions_;
    DenseMatrix stoichLeft, stoichRight; // size of (numSpecies, numReactions)

    /*
      From input file, reaction/reaction%d/equation will specify this mapping.
    */
    std::vector<std::pair<int, int>> reactionMapping; // size of numReactions_
    // Kevin: should we use a vector of function pointers?

    // Kevin: currently, I doubt we need mixture class here. but left it just in case.
    GasMixture *mixture_;

  public:
    Chemistry(GasMixture *mixture, RunConfiguration config);

    ~Chemistry(){};

    // return Vector of reaction rate coefficients, with the size of numReaction_.
    virtual Vector computeReactRateCoeffs(const double T_h, const double T_e) {};

    // return rate coefficients of (reactionIndex)-th reaction. (start from 0)
    // reactionIndex is taken from reactionMapping.right.
    virtual Vector computeRateCoeffOf(const int reactionIndex, const double T_h, const double T_e) {};

};

#endif  // CHEMISTRY_HPP_
