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

#ifndef ALGEBRAICSUBGRIDMODELS_HPP_
#define ALGEBRAICSUBGRIDMODELS_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
// #include <hdf5.h>
#include <tps_config.h>

// #include "loMach_options.hpp"
#include "../utils/mfem_extras/pfem_extras.hpp"
#include "io.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "run_configuration.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "turb_model_base.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachSolver;
class LoMachOptions;
struct temporalSchemeCoefficients;

/// Add some description here...
class AlgebraicSubgridModels : public TurbModelBase {
  friend class LoMachSolver;

 private:
  TPS::Tps *tpsP_;
  // LoMachSolver *loMach_;
  LoMachOptions *loMach_opts_ = nullptr;

  // MPI_Groups *groupsMPI;
  // int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // space sizes
  int Sdof_, SdofInt_;

  /// Enable/disable verbose output.
  bool verbose = true;

  ParMesh *pmesh_ = nullptr;

  // The order of the scalar spaces
  int order_;

  int nvel_, dim_;

  // just keep these saved for ease
  int numWalls_, numInlets_, numOutlets_;

  // model selection, passed in
  int sModel_;

  // Coefficients necessary to take a time step (including dt).
  // Assumed to be externally managed and determined, so just get a
  // reference here.
  // const temporalSchemeCoefficients &timeCoeff_;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec_ = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes_ = nullptr;

  /// spaces for filtered eddy viscosity
  FiniteElementCollection *sfec_filter_ = nullptr;
  ParFiniteElementSpace *sfes_filter_ = nullptr;

  ParGridFunction *gradU_gf_ = nullptr;
  ParGridFunction *gradV_gf_ = nullptr;
  ParGridFunction *gradW_gf_ = nullptr;
  Vector gradU_;
  Vector gradV_;
  Vector gradW_;

  ParGridFunction *rn_gf_ = nullptr;
  Vector rn_;

  ParGridFunction *delta_gf_ = nullptr;
  Vector delta_;

  ParGridFunction subgridVisc_gf_;
  Vector subgridVisc_;
  Vector muT_NM0_;
  Vector muT_NM1_;
  Vector muT_NM2_;
  Vector muT_NM3_;

  // filter related
  ParGridFunction muT_NM1_gf_;
  ParGridFunction muT_filtered_gf_;

  // grid information
  ParGridFunction *gridScale_ = nullptr;

  double sgs_model_const_;
  int sgs_model_nFilter_;
  bool sgs_model_smooth_;
  int aveSteps_;

 public:
  AlgebraicSubgridModels(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps, ParGridFunction *gridScale,
                         int sModel);
  virtual ~AlgebraicSubgridModels();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void setup() final;
  void initializeViz(ParaViewDataCollection &pvdc) final;

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *getCurrentEddyViscosity() { return &subgridVisc_gf_; }

  // subgrid scale models => move to turb model class
  void sgsSmag(const DenseMatrix &gradUp, double delta, double &nu_sgs);
  void sgsSigma(const DenseMatrix &gradUp, double delta, double &nu_sgs);
};
#endif  // ALGEBRAICSUBGRIDMODELS_HPP_
