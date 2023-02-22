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

#include "radiation.hpp"

MFEM_HOST_DEVICE NetEmission::NetEmission(const RadiationInput &inputs)
    : Radiation(), necTable_(LinearTable(inputs.necTableInput)) {
  assert(inputs.model == NET_EMISSION);
  assert(inputs.necModel == TABULATED_NEC);
  assert(inputs.necTableInput.order == 1);
  // To generalize beyond LinearTable, need to have necTable_ be a
  // pointer to TableInterpolator and then instantiate the appropriate
  // class here (e.g., necTable_ = new LinearTable(inputs.necTableInput);).
  // However, this design causes unexplained problems for the CUDA
  // version.  See Issue #184.  To avoid this issue, we use the design
  // here (i.e., necTable_ has type LinearTable).  Since this is the
  // only option currently supported, this approach is ok.  It will
  // have to be revisited if other options are added in the future.
}

MFEM_HOST_DEVICE NetEmission::~NetEmission() {}
