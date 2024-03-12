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
/** @file
 * @copydoc algebraic_rans.hpp
 */

#include "algebraic_rans.hpp"

using namespace mfem;

AlgebraicRans::AlgebraicRans(Mesh *smesh, ParMesh *pmesh, const Array<int> &partitioning, int order, TPS::Tps *tps)
    : pmesh_(pmesh), order_(order) {
  // Serial FE space for scalar variable
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  FiniteElementSpace serial_fes(smesh, sfec_);
  GridFunction serial_distance(&serial_fes);

  // Get coordinates from serial mesh
  if (smesh->GetNodes() == NULL) {
    smesh->SetCurvature(1);
  }

  int dim = smesh->Dimension();
  FiniteElementSpace tmp_dfes(smesh, sfec_, dim, Ordering::byNODES);
  GridFunction coordinates(&tmp_dfes);
  smesh->GetNodes(coordinates);

  // Build a list of wall patches based on BCs
  Array<int> wall_patch_list;

  // Evaluate the (serial) distance function
  evaluateDistanceSerial(*smesh, wall_patch_list, coordinates, serial_distance);

  // If necessary, parallelize distance function
  distance_ = new ParGridFunction(pmesh, &serial_distance, partitioning.HostRead());
  if (partitioning.HostRead() == nullptr) {
    *distance_ = serial_distance;
  }

  // Necessary?
  // distance_->ParFESpace()->ExchangeFaceNbrData();
  // distance_->ExchangeFaceNbrData();
}

AlgebraicRans::~AlgebraicRans() {
  // Alloced in initializeSelf
  delete mut_;

  // Alloced in ctor
  delete distance_;
  delete sfes_;
  delete sfec_;
}

void AlgebraicRans::initializeSelf() {
  mut_ = new ParGridFunction(sfes_);
  *mut_ = 0.0;

  toFlow_interface_.eddy_viscosity = mut_;
  toThermoChem_interface_.eddy_viscosity = mut_;
}

void AlgebraicRans::initializeViz(mfem::ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("muT", mut_);
  pvdc.RegisterField("distance", distance_);
}

void AlgebraicRans::step() {}
