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
#ifndef SBP_INTEGRATORS_HPP_
#define SBP_INTEGRATORS_HPP_

#include <tps_config.h>

#include <mfem.hpp>

#include "equation_of_state.hpp"
#include "fluxes.hpp"

using namespace mfem;

// Volume integrals...DESCRIBE THE OPERATIONS
class SBPintegrator : public NonlinearFormIntegrator {
 private:
  const int dim;
  const int num_equation;

  const double &alpha;

  GasMixture *mixture;
  Fluxes *fluxClass;

  IntegrationRules *intRules;

 public:
  SBPintegrator(GasMixture *mixture, Fluxes *_fluxClass, IntegrationRules *_intRules, const int _dim,
                const int _num_equation, double &_alpha);

  virtual void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
                                     Vector &elvect);
};

#endif  // SBP_INTEGRATORS_HPP_
