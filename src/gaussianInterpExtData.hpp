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

#ifndef GAUSSIANINTERPEXTDATA_HPP_
#define GAUSSIANINTERPEXTDATA_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

// forward-declaration of struct hold temporal coefficients (loMach.hpp)
struct temporalSchemeCoefficients;

#include <tps_config.h>

#include <iostream>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "externalData_base.hpp"
#include "io.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "run_configuration.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"

class LoMachSolver;
class LoMachOptions;

/// Add some description here...
class GaussianInterpExtData : public ExternalDataBase {
  friend class LoMachSolver;

 private:
  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;

  // Run options
  LoMachOptions *loMach_opts_;  //  = nullptr;

  // MPI helpers
  MPI_Groups *groupsMPI_ = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // for reading in species
  int nSpec_;  // number of species to read-in

  // space sizes
  int Sdof_, SdofInt_;

  /// Enable/disable verbose output.
  bool verbose = true;

  ParMesh *pmesh_;  //  = nullptr;

  // The order of the scalar spaces
  int order_;

  // dimension of mesh
  int dim_;

  // Coefficients necessary to take a time step (including dt).
  // Assumed to be externally managed and determined, so just get a
  // reference here.
  const temporalSchemeCoefficients &coeff_;

  // to-be used fro time or timestep dep bc
  // double dt;
  // double time;
  // int iter;

  bool isInterpInlet_;

  bool axisym_;

  std::string fname_;
  std::string fname_turb_;
  double v_fac_;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Vector \f$H^1\f$ finite element collection & space
  FiniteElementCollection *vfec_ = nullptr;
  ParFiniteElementSpace *vfes_ = nullptr;

  // for species dirichlet bc
  FiniteElementCollection *yfec_ = nullptr;
  ParFiniteElementSpace *yfes_ = nullptr;

  ParGridFunction temperature_gf_;
  ParGridFunction velocity_gf_;
  ParGridFunction vel0_gf_;
<<<<<<< HEAD
  ParGridFunction swirl_gf_;
  ParGridFunction swirl0_gf_;
  ParGridFunction nut_gf_;
=======
  ParGridFunction Yn_gf_;
>>>>>>> dev-semiImpConv

  // gradual increase of external bc over multiple steps
  int rampSteps_;

  // messy but required for compilation
  const int maxSpec_ = 10;

 public:
  GaussianInterpExtData(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &coeff,
                        TPS::Tps *tps);
  virtual ~GaussianInterpExtData();

  void initializeSelf();
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void setup();
  void step();

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetExternalInterpolatedTemperature() { return &temperature_gf_; }

  /// Return a pointer to the current velocity ParGridFunction.
  ParGridFunction *GetExternalInterpolatedVelocity() { return &velocity_gf_; }
};
#endif  // GAUSSIANINTERPEXTDATA_HPP_
