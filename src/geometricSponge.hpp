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

#ifndef GEOMETRICSPONGE_HPP_
#define GEOMETRICSPONGE_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <tps_config.h>

#include <iostream>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "io.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "run_configuration.hpp"
#include "split_flow_base.hpp"
#include "sponge_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"

struct geomUniform {
  bool isEnabled;
  double mult;
};

struct geomPlane {
  bool isEnabled;
  double point[3];
  double normal[3];
  double width;
  double mult;
};

struct geomCylinder {
  bool isEnabled;
  double point[3];
  double normal[3];
  double radiusX;
  double radiusY;
  double radiusZ;
  double width;
  double mult;
};

struct geomAnnulus {
  bool isEnabled;
  double point[3];
  double normal[3];
  double radiusX;
  double radiusY;
  double radiusZ;
  double width;
  double mult;
};

class LoMachOptions;

/// Add some description here...
class GeometricSponge : public SpongeBase {
 private:
  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;

  // Run options
  LoMachOptions *loMach_opts_;

  // MPI helpers
  MPI_Groups *groupsMPI_ = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // space sizes
  int Sdof_, SdofInt_;

  /// Enable/disable verbose output.
  bool verbose = true;

  ParMesh *pmesh_;

  // The order of the scalar spaces
  int order_;

  int dim_;

  // for future use of sponge to stabilize and IC
  // double dt;
  // double time;
  // int iter;

  geomUniform uniform;
  geomPlane plane;
  geomCylinder cylinder;
  geomAnnulus annulus;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Vector \f$H^1\f$ finite element collection & space
  FiniteElementCollection *vfec_ = nullptr;
  ParFiniteElementSpace *vfes_ = nullptr;

  ParGridFunction mult_gf_;
  // Vector mult_;

 public:
  GeometricSponge(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps);
  virtual ~GeometricSponge();

  void initializeSelf();
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void setup();
  void step();

  void spongeUniform(double &wgt);
  void spongePlane(double *x, double &wgt);
  void spongeCylinder(double *x, double &wgt);
  void spongeAnnulus(double *x, double &wgt);

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentMultiplier() { return &mult_gf_; }
};
#endif  // GEOMETRICSPONGE_HPP_
