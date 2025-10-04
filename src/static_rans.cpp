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

#include "static_rans.hpp"

#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"

using namespace mfem;

StaticRans::StaticRans(ParMesh *pmesh, const Array<int> &partitioning, int order, TPS::Tps *tps) : pmesh_(pmesh), sorder_(sorder) {}
: pmesh_(pmesh), order_(order) {


  // tps->getInput("loMach/static-rans/visc-file", visc_file_);

  // Scalar FEM space
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

}

StaticRans::~StaticRans() {
  delete eddy_viscosity_;
  delete sfes_;
  delete sfec_;
  delete mut_;
  delete extData_;
}

void StaticRans::initializeSelf() {
  // Eddy viscosity
  mut_ = new ParGridFunction(sfes_);
  *mut_ = 1.0;

  toFlow_interface_.eddy_viscosity = mut_;
  toThermoChem_interface_.eddy_viscosity = mut_;

  // get interpolated nu_t field
  nut_field_ = new GridFunctionCoefficient(extData_interface_->NuTdata);
}

void StaticRans::initializeViz(mfem::ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("muT", mut_);
}

void StaticRans::step() {

  *mut_ *= nut_field_;
  *mut_ *= *thermoChem_interface_->density;
}