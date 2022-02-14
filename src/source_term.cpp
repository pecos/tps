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
#include "source_term.hpp"

#include <vector>

SourceTerm::SourceTerm(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                       ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays,
                       RunConfiguration &_config, GasMixture *mixture, TransportProperties *transport, Chemistry *chemistry)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType,
                    _intRules, _vfes, _Up,
                    _gradUp, gpuArrays),
      mixture_(mixture),
      transport_(transport),
      chemistry_(chemistry) {
  h_numElems = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();

  //   b = new ParGridFunction(vfes);
  //
  //   // Initialize to zero
  //   int dof = vfes->GetNDofs();
  //   double *data = b->HostWrite();
  //   for(int ii=0;ii<dof*num_equation;ii++) data[ii] = 0.;
  //   auto d_b = b->ReadWrite();
}

SourceTerm::~SourceTerm() {
  if (mixture_ != NULL) delete mixture_;
  if (transport_ != NULL) delete transport_;
  if (chemistry_ != NULL) delete chemistry_;
}

void SourceTerm::updateTerms(mfem::Vector& in)
{
}

