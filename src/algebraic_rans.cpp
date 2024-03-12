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

#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"

using namespace mfem;

AlgebraicRans::AlgebraicRans(Mesh *smesh, ParMesh *pmesh, const Array<int> &partitioning, int order, TPS::Tps *tps)
    : pmesh_(pmesh), order_(order) {
  dim_ = pmesh_->Dimension();

  // Check for axisymmetric flag
  tps->getInput("loMach/axisymmetric", axisym_, false);

  tps->getInput("loMach/algebraic-rans/max-mixing-length", max_mixing_length_, 1e10);
  tps->getInput("loMach/algebraic-rans/von-karman", kappa_von_karman_, 0.41);

  // Scalar FEM space
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  // Evaluate distance function
  FiniteElementSpace serial_fes(smesh, sfec_);
  GridFunction serial_distance(&serial_fes);

  // Get coordinates from serial mesh
  if (smesh->GetNodes() == NULL) {
    smesh->SetCurvature(1);
  }

  FiniteElementSpace tmp_dfes(smesh, sfec_, dim_, Ordering::byNODES);
  GridFunction coordinates(&tmp_dfes);
  smesh->GetNodes(coordinates);

  // Build a list of wall patches based on BCs
  Array<int> wall_patch_list;

  int num_walls;
  tps->getInput("boundaryConditions/numWalls", num_walls, 0);

  // Wall Bcs
  for (int i = 1; i <= num_walls; i++) {
    int patch;
    std::string type;
    std::string basepath("boundaryConditions/wall" + std::to_string(i));

    tps->getRequiredInput((basepath + "/patch").c_str(), patch);
    tps->getRequiredInput((basepath + "/type").c_str(), type);

    // NB: Only catch "no-slip" type walls here
    if (type == "viscous_isothermal" || type == "viscous_adiabatic" || type == "viscous" || type == "no-slip") {
      wall_patch_list.Append(patch);
    }
  }

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
  delete ell_mix_gf_;
  delete swirl_vorticity_gf_;
  delete vorticity_gf_;
  delete vfes_;
  delete vfec_;
  delete mut_;

  // Alloced in ctor
  delete distance_;
  delete sfes_;
  delete sfec_;
}

void AlgebraicRans::initializeSelf() {
  // Eddy viscosity
  mut_ = new ParGridFunction(sfes_);
  *mut_ = 0.0;

  // Space for vorticity
  vfec_ = new H1_FECollection(order_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  // Vorticity
  vorticity_gf_ = new ParGridFunction(vfes_);
  *vorticity_gf_ = 0.0;

  // If necessary, the swirl contribution to the vorticity
  if (axisym_) {
    swirl_vorticity_gf_ = new ParGridFunction(vfes_);
    *swirl_vorticity_gf_ = 0.0;
  }

  ell_mix_gf_ = new ParGridFunction(sfes_);
  *ell_mix_gf_ = 0.0;

  // Initialize the interfaces
  toFlow_interface_.eddy_viscosity = mut_;
  toThermoChem_interface_.eddy_viscosity = mut_;
}

void AlgebraicRans::initializeViz(mfem::ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("muT", mut_);
  pvdc.RegisterField("distance", distance_);
}

void AlgebraicRans::step() {
  // Evaluate the vorticity
  if (axisym_) {
    ComputeCurlAxi(*flow_interface_->velocity, *vorticity_gf_, false);    // only first component valid
    ComputeCurlAxi(*flow_interface_->swirl, *swirl_vorticity_gf_, true);  // both components valid
  } else {
    if (dim_ == 2) {
      ComputeCurl2D(*flow_interface_->velocity, *vorticity_gf_, false);  // only first component valid
    } else {
      ComputeCurl3D(*flow_interface_->velocity, *vorticity_gf_);  // all components valid
    }
  }

  // Compute the magnitude of the vorticity (dof-wise) and fill into
  // mut_, since rest of operations are multiplicative
  if (axisym_) {
    int ndof = sfes_->GetNDofs();
    double *d_mut = mut_->Write();
    const double *d_omega = vorticity_gf_->Read();
    const double *d_swirl_omega = swirl_vorticity_gf_->Read();
    MFEM_FORALL(i, ndof, {
      double omega_r = d_swirl_omega[i];
      double omega_th = d_omega[i];
      double omega_z = d_swirl_omega[ndof + i];
      double magn_omega_2 = omega_r * omega_r + omega_th * omega_th + omega_z * omega_z;
      d_mut[i] = sqrt(magn_omega_2);
    });
  } else {
    if (dim_ == 2) {
      int ndof = sfes_->GetNDofs();
      double *d_mut = mut_->Write();
      const double *d_omega = vorticity_gf_->Read();
      MFEM_FORALL(i, ndof, {
        double omega_z = d_omega[i];
        double magn_omega_2 = omega_z * omega_z;
        d_mut[i] = sqrt(magn_omega_2);
      });
    } else {  // dim_ == 3
      int ndof = sfes_->GetNDofs();
      double *d_mut = mut_->Write();
      const double *d_omega = vorticity_gf_->Read();
      MFEM_FORALL(i, ndof, {
        double omega_x = d_omega[i];
        double omega_y = d_omega[ndof + i];
        double omega_z = d_omega[2 * ndof + i];
        double magn_omega_2 = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z;
        d_mut[i] = sqrt(magn_omega_2);
      });
    }
  }

  // Evaluate the mixing length
  {
    int ndof = sfes_->GetNDofs();
    double *d_ellmix = ell_mix_gf_->Write();
    const double *d_dist = distance_->Read();
    const double kap = kappa_von_karman_;
    const double max_ell = max_mixing_length_;
    MFEM_FORALL(i, ndof, { d_ellmix[i] = min(kap * d_dist[i], max_ell); });
  }

  // Fill the eddy viscosity (dof-wise)
  *mut_ *= (*ell_mix_gf_);
  *mut_ *= (*ell_mix_gf_);
  *mut_ *= *thermoChem_interface_->density;
}
