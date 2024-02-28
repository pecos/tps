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

// TODO(trevilo): List of capabilities removed during refactoring (to be added back asap)
//
// * Statstics calculations.  The Averaging class is somewhat brittle
//   and only plays nice with state stored as done in the compressible
//   flow code path, which is not convenient here.  This should be
//   refactored into a more general capability so it can be used here.
//   For the moment however, averaging is disabled.
//
// * Variable time stepping.  Need to get dt info from models.  Not
//   provided yet.  Put variable dt back once that part of the
//   interface is written. This will include adding the control logic
//   to LoMachSolver and likely some new parameters in
//   LoMachTemporalScheme.
//
// * Some options parsing.  Some options previously parsed by
//   LoMachSolver have been eliminated.  Generally speaking these were
//   either options that were not being used (as far as I could tell)
//   or that seemed model-specific, and thus could be better handled
//   by the model classes.  So, some will be brought back, but clearly
//   until that is done, the options don't exist.

#ifndef LOMACH_HPP_
#define LOMACH_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <tps_config.h>

#include "io.hpp"
#include "loMach_options.hpp"
#include "solver.hpp"
#include "split_flow_base.hpp"
#include "sponge_base.hpp"
#include "thermo_chem_base.hpp"
#include "externalData_base.hpp"
#include "tps_mfem_wrap.hpp"

struct temporalSchemeCoefficients {
  // Current time
  double time;

  // Time step
  double dt;

  // Previous time steps
  double dt1;
  double dt2;
  double dt3;

  // "Adams-Bashforth" coefficients (explicit)
  double ab1;
  double ab2;
  double ab3;

  // "Backward difference" coefficients (implicit)
  double bd0;
  double bd1;
  double bd2;
  double bd3;
};

/**
 * @brief Driver class for models based on low Mach, variable density formulation
 */
class LoMachSolver : public TPS::Solver {
 protected:
  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;

  // Run options
  LoMachOptions loMach_opts_;

  // MPI helpers
  MPI_Groups *groupsMPI = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // Various codes and flags
  int exit_status_;     // exit status code;
  bool loadFromAuxSol;  // load restart of different polynomial order

  // Model classes
  // TurbModel *turbClass = nullptr;
  ThermoChemModelBase *thermo_ = nullptr;
  FlowBase *flow_ = nullptr;
  SpongeBase *sponge_ = nullptr;
  ExternalDataBase *extData_ = nullptr;  

  // Mesh and geometry related
  ParMesh *pmesh_ = nullptr;
  Mesh *serial_mesh_ = nullptr;

  int dim_;
  int nvel_;

  // mapping from local to global element index
  int *locToGlobElem = nullptr;

  // total number of mesh elements (serial)
  int nelemGlobal_;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;

  // min/max element size
  double hmin, hmax;

  // domain extent
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  /// Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec = nullptr;

  /// Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes = nullptr;

  Vector gridScaleSml;
  Vector gridScaleXSml;
  Vector gridScaleYSml;
  Vector gridScaleZSml;

  // for plotting
  ParGridFunction resolution_gf;

  // grid information
  ParGridFunction *bufferGridScale = nullptr;
  ParGridFunction *bufferGridScaleX = nullptr;
  ParGridFunction *bufferGridScaleY = nullptr;
  ParGridFunction *bufferGridScaleZ = nullptr;

  // Time marching related parameters
  int iter;
  int iter_start_;
  double CFL;

  int max_bdf_order;  // input option now
  temporalSchemeCoefficients temporal_coeff_;

  /// The order of the velocity and pressure space.
  int order;

  // Timers.
  StopWatch sw_setup, sw_step;
  double tlast_;

  // I/O helpers
  ParaViewDataCollection *pvdc_;  // visualization
  IODataOrganizer ioData;         // restart

  /// Update the EXTk/BDF time integration coefficient.
  void SetTimeIntegrationCoefficients(int step);

 public:
  /// Ctor
  LoMachSolver(TPS::Tps *tps);

  /// Dtor
  virtual ~LoMachSolver();

  // overrides of Solver functions
  void initialize() override;
  void parseSolverOptions() override;
  void solve() override;
  void solveStep() override;
  void solveBegin() override;
  void solveEnd() override;

  // time-step related functions
  void initialTimeStep();
  void updateTimestep();
  void setTimestep();
  double computeCFL();

  // i/o routines
  void restart_files_hdf5(string mode, string inputFileName = std::string());
  void write_restart_files_hdf5(hid_t file, bool serialized_write);
  void read_restart_files_hdf5(hid_t file, bool serialized_read);

  /// Print timing summary of the solving routine.
  void PrintTimingData();

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);
};

#endif  // LOMACH_HPP_
