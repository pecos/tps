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

#ifndef ZETAMODEL_HPP_
#define ZETAMODEL_HPP_

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
class ZetaModel : public TurbModelBase {
  friend class LoMachSolver;

 private:
  TPS::Tps *tpsP_;
  LoMachOptions *loMach_opts_ = nullptr;

  // MPI_Groups *groupsMPI;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // space sizes
  int Sdof_, SdofInt_;

  /// Enable/disable verbose output.
  bool verbose = true;

  /// pointer to mesh 
  ParMesh *pmesh_ = nullptr;

  // The order of the scalar spaces
  int order_;

  int nvel_, dim_;

  // just keep these saved for ease
  int numWalls_, numInlets_, numOutlets_;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec_ = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes_ = nullptr;

  /// velocity
  ParGridFunction *vel_gf_ = nullptr;
  Vector vel_;
  
  /// velocity gradients
  ParGridFunction *gradU_gf_ = nullptr;
  ParGridFunction *gradV_gf_ = nullptr;
  ParGridFunction *gradW_gf_ = nullptr;
  Vector gradU_;
  Vector gradV_;
  Vector gradW_;

  /// fluid density
  ParGridFunction *rn_gf_ = nullptr;
  Vector rn_;
  
  /// grid information
  ParGridFunction *gridScale_ = nullptr;
  ParGridFunction *delta_gf_ = nullptr;
  Vector delta_;  
  
  /// eddy viscosity
  ParGridFunction eddyVisc_gf_;
  Vector eddyVisc_;

  /// turbulent kinetic energy
  ParGridFunction tke_gf_;
  Vector tke_;

  /// turbulent dissipation rate  
  ParGridFunction tdr_gf_;
  Vector tdr_;

  /// ratio of wall normal stress component to k  
  ParGridFunction zeta_gf_;
  Vector zeta_;

  /// elliptic relaxation rate
  ParGridFunction fRate_gf_;
  Vector fRate_;

  /// turbulent length scale
  ParGridFunction tls_gf_;
  Vector tls_;

  /// turbulent time scale
  ParGridFunction tts_gf_;
  Vector tts_;    

  /// turbulent production
  ParGridFunction prod_gf_;
  Vector prod_;      
  
  /// model coefficients
  double Cmu_ = 0.22;
  double sigmaK_ = 1.0;
  double sigmaO_ = 1.3;
  double sigmaZ_ = 1.2;
  double Ce2_ = 1.9;
  double C1_ = 1.4;
  double C2_ = 0.65;
  double Ct_ = 6.0;
  double Cl_ = 0.36;
  double Cn_ = 85.0;
  double Ce1_; // function of local zeta

  /// coefficient fields for operators
  VectorGridFunctionCoefficient *un_next_coeff_ = nullptr;
  GridFunctionCoefficient *rhon_next_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rhou_coeff_ = nullptr;
  GridFunctionCoefficient *mut_coeff_ = nullptr;
  GridFunctionCoefficient *mult_coeff_ = nullptr;
  GridFunctionCoefficient *rho_over_dt_coeff_ = nullptr;
  GridFunctionCoefficient *rho_coeff_ = nullptr;
  
  /// operators and solvers
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *Hs_form_ = nullptr;  
  OperatorHandle Ms_; 
  OperatorHandle Hs_;   
  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *HsInvPC_ = nullptr;
  mfem::CGSolver *HsInv_ = nullptr;
  
 public:
  ZetaModel(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps, ParGridFunction *gridScale);
  virtual ~ZetaModel();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void setup() final;
  void initializeViz(ParaViewDataCollection &pvdc) final;

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *getCurrentEddyViscosity() { return &eddyVisc_gf_; }

};
#endif  // ZETAMODEL_HPP_
