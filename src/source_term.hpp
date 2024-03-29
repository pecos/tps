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
#ifndef SOURCE_TERM_HPP_
#define SOURCE_TERM_HPP_

#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "forcing_terms.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace mfem;
using namespace std;

class SourceTerm : public ForcingTerms {
 private:
  // These are protected variables in ForcingTerms base class.
  // double time;
  //
  // const int &dim;
  // const int &num_equation;
  // const int &order;
  // const int &intRuleType;
  // IntegrationRules *intRules;
  // ParFiniteElementSpace *vfes;
  // ParGridFunction *Up;
  // ParGridFunction *gradUp;
  //
  // const precomputedIntegrationData &gpu_precomputed_data_
  // const int *h_num_elems_of_type;

  int numSpecies_;
  int numActiveSpecies_;
  int numReactions_;

  bool ambipolar_;
  bool twoTemperature_;
  bool enableRadiation_;

  GasMixture *mixture_ = NULL;
  GasMixture *d_mixture_ = NULL;
  TransportProperties *transport_ = NULL;
  Chemistry *chemistry_ = NULL;
  Radiation *radiation_ = NULL;
  ParGridFunction *plasma_conductivity_;
  ParGridFunction *distance_;

 public:
  SourceTerm(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
             IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up,
             ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays, RunConfiguration &_config,
             GasMixture *mixture, GasMixture *d_mixture, TransportProperties *transport, Chemistry *chemistry,
             Radiation *radiation, ParGridFunction *pc, ParGridFunction *distance);
  ~SourceTerm();

  // Terms do not need updating
  virtual void updateTerms(Vector &in);
};

#endif  // SOURCE_TERM_HPP_
