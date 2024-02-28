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

#include "split_flow_base.hpp"

using namespace mfem;

ZeroFlow::ZeroFlow(mfem::ParMesh *pmesh, int vorder) : pmesh_(pmesh), vorder_(vorder), dim_(pmesh->Dimension()) {}

ZeroFlow::~ZeroFlow() {
  delete velocity_;
  delete fes_;
  delete fec_;
}

void ZeroFlow::initializeSelf() {
  fec_ = new H1_FECollection(vorder_, dim_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_, dim_);
  velocity_ = new ParGridFunction(fes_);
  *velocity_ = 0.0;

  toThermoChem_interface_.velocity = velocity_;
  toThermoChem_interface_.swirl_supported = false;

  toTurbModel_interface_.gradU = nullptr;
  toTurbModel_interface_.gradV = nullptr;
  toTurbModel_interface_.gradW = nullptr;
}
