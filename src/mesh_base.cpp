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

#include "mesh_base.hpp"

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "io.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;

MeshBase::MeshBase(TPS::Tps *tps, LoMachOptions *loMach_opts, int order)
    : tpsP_(tps),
      groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      rank0_(groupsMPI->isWorldRoot()),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      loMach_opts_(loMach_opts),
      order_(order) {}

MeshBase::~MeshBase() {
  delete gridScale_;
  delete distance_;
  delete fes_;
  delete fec_;
  delete[] local_to_global_element_;
  delete pmesh_;
  delete serial_mesh_;
  delete groupsMPI;
}

/// Prepare the mesh and compute secondary mesh-related data
void MeshBase::initializeMesh() {
  // Determine domain bounding box size
  {
    Mesh temp_mesh = Mesh(loMach_opts_->mesh_file.c_str());
    dim_ = temp_mesh.Dimension();
    H1_FECollection fec(order_, dim_);
    FiniteElementSpace fes(&temp_mesh, &fec);
    GridFunction coordsVert(&fes);
    temp_mesh.GetVertices(coordsVert);
    int nVert = coordsVert.Size() / dim_;
    {
      double local_xmin = 1.0e18;
      double local_ymin = 1.0e18;
      double local_zmin = 1.0e18;
      double local_xmax = -1.0e18;
      double local_ymax = -1.0e18;
      double local_zmax = -1.0e18;

      double coords[3];
      for (int d = 0; d < 3; d++) {
        coords[d] = 0.0;
      }

      for (int n = 0; n < nVert; n++) {
        auto hcoords = coordsVert.HostRead();
        for (int d = 0; d < dim_; d++) {
          coords[d] = hcoords[n + d * nVert];
        }
        local_xmin = min(coords[0], local_xmin);
        local_ymin = min(coords[1], local_ymin);
        if (dim_ == 3) {
          local_zmin = min(coords[2], local_zmin);
        }
        local_xmax = max(coords[0], local_xmax);
        local_ymax = max(coords[1], local_ymax);
        if (dim_ == 3) {
          local_zmax = max(coords[2], local_zmax);
        }
      }
      xmin_ = local_xmin;
      xmax_ = local_xmax;
      ymin_ = local_ymin;
      ymax_ = local_ymax;
      zmin_ = local_zmin;
      zmax_ = local_zmax;
    }

    if (loMach_opts_->periodicX) {
      loMach_opts_->x_trans = xmax_ - xmin_;
    }
    if (loMach_opts_->periodicY) {
      loMach_opts_->y_trans = ymax_ - ymin_;
    }
    if (loMach_opts_->periodicZ) {
      loMach_opts_->z_trans = zmax_ - zmin_;
    }
    if (loMach_opts_->periodicX || loMach_opts_->periodicY || loMach_opts_->periodicZ) {
      loMach_opts_->periodic = true;
    }
  }

  // Generate serial mesh, making periodic if requested
  if (loMach_opts_->periodic) {
    Mesh temp_mesh = Mesh(loMach_opts_->mesh_file.c_str());
    Vector x_translation({loMach_opts_->x_trans, 0.0, 0.0});
    Vector y_translation({0.0, loMach_opts_->y_trans, 0.0});
    Vector z_translation({0.0, 0.0, loMach_opts_->z_trans});
    std::vector<Vector> translations = {x_translation, y_translation, z_translation};

    if (rank0_) {
      std::cout << " Making the mesh periodic using the following offsets:" << std::endl;
      std::cout << "   xTrans: " << loMach_opts_->x_trans << std::endl;
      std::cout << "   yTrans: " << loMach_opts_->y_trans << std::endl;
      std::cout << "   zTrans: " << loMach_opts_->z_trans << std::endl;
    }

    serial_mesh_ =
        new Mesh(std::move(Mesh::MakePeriodic(temp_mesh, temp_mesh.CreatePeriodicVertexMapping(translations))));
  } else {
    serial_mesh_ = new Mesh(loMach_opts_->mesh_file.c_str());
  }
  if (rank0_) grvy_printf(ginfo, "Mesh read...\n");

  // Scale domain (NB: after periodicity applied)
  if (loMach_opts_->scale_mesh != 1) {
    if (rank0_) grvy_printf(ginfo, "Scaling mesh factor of scale_mesh = %.6e\n", loMach_opts_->scale_mesh);
    serial_mesh_->EnsureNodes();
    GridFunction *nodes = serial_mesh_->GetNodes();
    *nodes *= loMach_opts_->scale_mesh;
  }

  // check if a simulation is being restarted
  if (loMach_opts_->io_opts_.enable_restart_) {
    // uniform refinement, user-specified number of times
    for (int l = 0; l < loMach_opts_->ref_levels; l++) {
      if (rank0_) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh_->UniformRefinement();
    }

    // read partitioning info from original decomposition (unless restarting from serial soln)
    nelemGlobal_ = serial_mesh_->GetNE();
    // nelemGlobal_ = mesh->GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      assert(!loMach_opts_->io_opts_.restart_serial_read_);
      // TODO(trevilo): Add support for serial read/write
      partitioning_file_hdf5("read", groupsMPI, nelemGlobal_, partitioning_);
    }

  } else {
    // remove previous solution
    if (rank0_) {
      string command = "rm -r ";
      command.append(loMach_opts_->io_opts_.output_dir_);
      int err = system(command.c_str());
      if (err != 0) {
        cout << "Error deleting previous data in " << loMach_opts_->io_opts_.output_dir_ << endl;
      }
    }

    // uniform refinement, user-specified number of times
    for (int l = 0; l < loMach_opts_->ref_levels; l++) {
      if (rank0_) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh_->UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh_->GetNE();
    if (nprocs_ > 1) {
      assert(serial_mesh_->Conforming());
      partitioning_ = Array<int>(serial_mesh_->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write", groupsMPI, nelemGlobal_, partitioning_);
    }
  }

  // Partition the mesh
  pmesh_ = new ParMesh(groupsMPI->getTPSCommWorld(), *serial_mesh_, partitioning_);
  if (rank0_) grvy_printf(ginfo, "Mesh partitioned...\n");

  // Build local->global element numbering map
  if (nprocs_ > 1) {
    assert(partitioning_.Size() == nelemGlobal_);
    local_to_global_element_ = new int[pmesh_->GetNE()];
    int lelem = 0;
    for (int gelem = 0; gelem < nelemGlobal_; gelem++) {
      if (rank_ == partitioning_[gelem]) {
        local_to_global_element_[lelem] = gelem;
        lelem += 1;
      }
    }
  }

  // Determine the minimum element size.
  double hmin, hmax;
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmin = min(pmesh_->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
  }
  if (rank0_) cout << "Minimum element size: " << hmin << "m" << endl;

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmax = max(pmesh_->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  }
  if (rank0_) cout << "Maximum element size: " << hmax << "m" << endl;

  fec_ = new H1_FECollection(order_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_);

  // used in loMach to remove need to create local sfes
  sDof_ = fes_->GetNDofs();

  // scalar measure of mesh size
  computeGridScale();

  // wall distance calc
  if (loMach_opts_->compute_wallDistance) {
    computeWallDistance();
  }
}

void MeshBase::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("resolution", gridScale_);
  if (loMach_opts_->compute_wallDistance) {
    pvdc.RegisterField("wall_dist", distance_);
  }
}

/// Build grid scale grid function
void MeshBase::computeGridScale() {
  gridScale_ = new ParGridFunction(fes_);
  ParGridFunction dofCount(fes_);
  {
    int elndofs;
    Array<int> vdofs;
    Vector vals;
    Vector loc_data;
    int nSize = gridScale_->Size();
    Array<int> zones_per_vdof;
    zones_per_vdof.SetSize(fes_->GetVSize());
    zones_per_vdof = 0;
    Array<int> zones_per_vdofALL;
    zones_per_vdofALL.SetSize(fes_->GetVSize());
    zones_per_vdofALL = 0;

    double *data = gridScale_->HostReadWrite();
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
        delta = delta / ((double)order_);
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
    GroupCommunicator &gcomm = gridScale_->ParFESpace()->GroupComm();
    gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
    gcomm.Bcast(zones_per_vdof);

    // Accumulate for all vdofs.
    gcomm.Reduce<double>(gridScale_->GetData(), GroupCommunicator::Sum);
    gcomm.Bcast<double>(gridScale_->GetData());

    // Compute means.
    for (int i = 0; i < nSize; i++) {
      const int nz = zones_per_vdof[i];
      if (nz) {
        data[i] /= nz;
      }
    }
  }
}

/// Evaluate distance to wall function
void MeshBase::computeWallDistance() {
  FiniteElementSpace serial_fes(serial_mesh_, fec_);
  GridFunction serial_distance(&serial_fes);

  // Get coordinates from serial mesh
  if (serial_mesh_->GetNodes() == NULL) {
    serial_mesh_->SetCurvature(1);
  }

  FiniteElementSpace tmp_dfes(serial_mesh_, fec_, dim_, Ordering::byNODES);
  GridFunction coordinates(&tmp_dfes);
  serial_mesh_->GetNodes(coordinates);

  // Build a list of wall patches based on BCs
  Array<int> wall_patch_list;

  int num_walls;
  tpsP_->getInput("boundaryConditions/numWalls", num_walls, 0);

  // Wall Bcs
  for (int i = 1; i <= num_walls; i++) {
    int patch;
    std::string type;
    std::string basepath("boundaryConditions/wall" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    // NB: Only catch "no-slip" type walls here
    if (type == "viscous_isothermal" || type == "viscous_adiabatic" || type == "viscous" || type == "no-slip") {
      wall_patch_list.Append(patch);
    }
  }

  // Evaluate the (serial) distance function
  evaluateDistanceSerial(*serial_mesh_, wall_patch_list, coordinates, serial_distance);

  // If necessary, parallelize distance function
  distance_ = new ParGridFunction(pmesh_, &serial_distance, partitioning_.HostRead());
  if (partitioning_.HostRead() == nullptr) {
    *distance_ = serial_distance;
  }
}
