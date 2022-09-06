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
#ifndef REACTION_HPP_
#define REACTION_HPP_

/*
  Reaction class determines reaction rate coefficient of a single reaction.
  The set of reaction classes is handled by Chemistry class.
*/

#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

class Reaction {
 protected:
 public:
  MFEM_HOST_DEVICE Reaction() {}

  MFEM_HOST_DEVICE virtual ~Reaction() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         const bool isElectronInvolved = false) {
    printf("computeRateCoefficient not implemented");
    return 0;
  }
};

class Arrhenius : public Reaction {
 private:
  // coefficients for modified Arrhenius model: k = A * T^b * exp( - E / R / T )
  double A_;
  double b_;
  double E_;

 public:
  MFEM_HOST_DEVICE Arrhenius(const double &A, const double &b, const double &E);

  MFEM_HOST_DEVICE virtual ~Arrhenius() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         const bool isElectronInvolved = false);
};

class HoffertLien : public Reaction {
  /*
    referece: Hoffert, M. I., & Lien, H. (1967). Quasi-one-dimensional, nonequilibrium gas dynamics of partially ionized
    two-temperature argon. Physics of Fluids, 10(8), 1769–1777. https://doi.org/10.1063/1.1762356
  */
 private:
  // coefficients for modified Arrhenius model: k = A * T^b * exp( - E / R / T )
  double A_;
  double b_;
  double E_;

 public:
  MFEM_HOST_DEVICE HoffertLien(const double &A, const double &b, const double &E);

  MFEM_HOST_DEVICE virtual ~HoffertLien() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         const bool isElectronInvolved = false);
};

class Tabulated : public Reaction {
 private:
  TableInterpolator *table_ = NULL;

 public:
  MFEM_HOST_DEVICE Tabulated(const TableInput &input);

  MFEM_HOST_DEVICE virtual ~Tabulated();

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         const bool isElectronInvolved = false);
};

#endif  // REACTION_HPP_
