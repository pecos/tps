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

#include "turb_model_base.hpp"

using namespace mfem;

ZeroTurbModel::ZeroTurbModel(ParMesh *pmesh, int sorder) : pmesh_(pmesh), sorder_(sorder) {}

ZeroTurbModel::~ZeroTurbModel() {
  delete eddy_viscosity_;
  delete bufferGridScale_;
  delete fes_;
  delete fec_;
}

void ZeroTurbModel::initializeSelf() {
  fec_ = new H1_FECollection(sorder_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_);

  eddy_viscosity_ = new ParGridFunction(fes_);

  *eddy_viscosity_ = 0.0;

  toFlow_interface_.eddy_viscosity = eddy_viscosity_;
  toThermoChem_interface_.eddy_viscosity = eddy_viscosity_;
}

void ZeroTurbModel::setup() {
  /// Grid-related ///

  // Build grid size vector and grid function
  bufferGridScale_ = new ParGridFunction(fes_);
  ParGridFunction dofCount(fes_);
  {
    int elndofs;
    Array<int> vdofs;
    Vector vals;
    Vector loc_data;
    int nSize = bufferGridScale_->Size();
    Array<int> zones_per_vdof;
    zones_per_vdof.SetSize(fes_->GetVSize());
    zones_per_vdof = 0;
    Array<int> zones_per_vdofALL;
    zones_per_vdofALL.SetSize(fes_->GetVSize());
    zones_per_vdofALL = 0;

    double *data = bufferGridScale_->HostReadWrite();
    double *count = dofCount.HostReadWrite();

    for (int i = 0; i < fes_->GetNDofs(); i++) {
      data[i] = 0.0;
      count[i] = 0.0;
    }

    // element loop
    for (int e = 0; e < fes_->GetNE(); ++e) {
      fes_->GetElementVDofs(e, vdofs);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes_->GetElementTransformation(e);
      const FiniteElement *el = fes_->GetFE(e);
      elndofs = el->GetDof();
      double delta;

      // element dof
      for (int dof = 0; dof < elndofs; ++dof) {
        const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
        tr->SetIntPoint(&ip);
        delta = pmesh_->GetElementSize(tr->ElementNo, 1);
        delta = delta / ((double)sorder_);
        vals(dof) = delta;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        data[ldof + 0 * nSize] += vals[j];
      }

      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        zones_per_vdof[ldof]++;
      }
    }

    // Count the zones globally.
    GroupCommunicator &gcomm = bufferGridScale_->ParFESpace()->GroupComm();
    gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
    gcomm.Bcast(zones_per_vdof);

    // Accumulate for all vdofs.
    gcomm.Reduce<double>(bufferGridScale_->GetData(), GroupCommunicator::Sum);
    gcomm.Bcast<double>(bufferGridScale_->GetData());

    // Compute means.
    for (int i = 0; i < nSize; i++) {
      const int nz = zones_per_vdof[i];
      if (nz) {
        data[i] /= nz;
      }
    }
    
  }

}
