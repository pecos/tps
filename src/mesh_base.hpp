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
#ifndef MESH_BASE_HPP_
#define MESH_BASE_HPP_
/** @file
 * @brief Contains base class and simplest possible variant of mesh
 * construction and information.
 */

#include <tps_config.h>

#include "loMach_options.hpp"
#include "tps_mfem_wrap.hpp"

class LoMachOptions;

class MeshBase {
 private:
  TPS::Tps *tpsP_ = nullptr;

  MPI_Groups *groupsMPI = nullptr;
  bool rank0_;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank

  LoMachOptions *loMach_opts_ = nullptr;
  const int order_;
  int dim_;

  ParMesh *pmesh_ = nullptr;
  Mesh *serial_mesh_ = nullptr;

  // mapping from local to global element index
  // int *locToGlobElem = nullptr;

  // total number of mesh elements (serial)
  int nelemGlobal_;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;

  // mapping from local to global element index
  int *local_to_global_element_ = nullptr;

  // min/max element size
  double hmin_, hmax_;

  // domain extent
  double xmin_, ymin_, zmin_;
  double xmax_, ymax_, zmax_;

  mfem::FiniteElementCollection *fec_ = nullptr;
  mfem::ParFiniteElementSpace *fes_ = nullptr;
  mfem::ParGridFunction *gridScale_ = nullptr;
  mfem::ParGridFunction *distance_ = nullptr;

  // used in loMach
  int sDof_;

 public:
  MeshBase(TPS::Tps *tps, LoMachOptions *loMach_opts, int order);
  virtual ~MeshBase();

  virtual void initializeMesh();
  virtual void computeGridScale();
  virtual void computeWallDistance();
  virtual void initializeViz(mfem::ParaViewDataCollection &pvdc);

  virtual ParMesh *getMesh() { return pmesh_; }
  virtual Mesh *getSerialMesh() { return serial_mesh_; }
  virtual ParGridFunction *getGridScale() { return gridScale_; }
  virtual ParGridFunction *getWallDistance() { return distance_; }
  virtual int getDofSize() { return sDof_; }
  virtual double getMinGridScale() { return hmin_; }
  virtual Array<int> getPartition() { return partitioning_; }
  // virtual int getDim() final { return dim_; }

  int *getLocalToGlobalElementMap() const { return local_to_global_element_; }
};
#endif  // MESH_BASE_HPP_
