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

// TODO(trevilo): List of capabilities removed during refactoring
//
// * Statstics calculations.  The Averaging class is somewhat brittle
//   and only plays nice with state stored as done in the compressible
//   flow code path, which is not convenient here.  This should be
//   refactored into a more general capability so it can be used here.
//   For the moment however, averaging is disabled.

#ifndef LOMACH_HPP_
#define LOMACH_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <tps_config.h>

#include "io.hpp"
#include "loMach_options.hpp"
#include "run_configuration.hpp"
#include "solver.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps_mfem_wrap.hpp"

/**
 * @brief Driver class for models based on low Mach, variable density formulation
 */
class LoMachSolver : public TPS::Solver {
 protected:
  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;

  // Run options
  LoMachOptions loMach_opts_;
  RunConfiguration config;

  // MPI helpers
  MPI_Groups *groupsMPI = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // Various codes and flags
  int exit_status_;     // exit status code;
  int earlyExit = 0;    // Early terminatation flag
  bool loadFromAuxSol;  // load restart of different polynomial order

  // Model classes
  // TurbModel *turbClass = nullptr;
  ThermoChemModelBase *thermo_ = nullptr;
  FlowBase *flow_ = nullptr;

  // Mesh and geometry related
  ParMesh *pmesh_ = nullptr;

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

  /// Distance to nearest no-slip wall
  ParGridFunction *distance_ = nullptr;

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

  // just keep these saved for ease
  int numWalls, numInlets, numOutlets;

  // Time marching related parameters
  double dt;
  double time;
  int iter;

  int MaxIters;
  double CFL;

  int max_bdf_order;  // input option now
  std::vector<double> dthist = {0.0, 0.0, 0.0};

  // BDFk/EXTk coefficients.
  double bd0 = 0.0;
  double bd1 = 0.0;
  double bd2 = 0.0;
  double bd3 = 0.0;
  double ab1 = 0.0;
  double ab2 = 0.0;
  double ab3 = 0.0;
  std::vector<double> abCoef = {ab1, ab2, ab3};
  std::vector<double> bdfCoef = {bd0, bd1, bd2, bd3};

  /// The order of the velocity and pressure space.
  int order;
  int porder;
  int norder;

  // Timers.
  StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

  // I/O helpers
  ParaViewDataCollection *paraviewColl = nullptr;  // visualization
  IODataOrganizer ioData;                          // restart

  /// Update the EXTk/BDF time integration coefficient.
  void SetTimeIntegrationCoefficients(int step);

 public:
  /// Ctor
  LoMachSolver(LoMachOptions loMach_opts, TPS::Tps *tps);

  /// Dtor TODO(trevilo): this needs to free memory
  virtual ~LoMachSolver();

  void initialize();
  void parseSolverOptions() override;
  void parseSolverOptions2();
  void parsePeriodicInputs();
  void parseFlowOptions();
  void parseTimeIntegrationOptions();
  void parseStatOptions();
  void parseIOSettings();
  void parseRMSJobOptions();
  void parseBCInputs();
  void parseICOptions();
  void parsePostProcessVisualizationInputs();
  void parseFluidPreset();
  void parseViscosityOptions();
  void initialTimeStep();
  void solve();
  void updateTimestep();
  void setTimestep();

  // i/o routines
  void restart_files_hdf5(string mode, string inputFileName = std::string());
  void write_restart_files_hdf5(hid_t file, bool serialized_write);
  void read_restart_files_hdf5(hid_t file, bool serialized_read);

  void projectInitialSolution();
  void writeHDF5(string inputFileName = std::string()) { restart_files_hdf5("write", inputFileName); }
  void readHDF5(string inputFileName = std::string()) { restart_files_hdf5("read", inputFileName); }
  void writeParaview(int iter, double time) {
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
  }

  /// Return a pointer to the current total resolution ParGridFunction.
  ParGridFunction *GetCurrentResolution() { return &resolution_gf; }

  LoMachOptions GetOptions() { return loMach_opts_; }

  /// Print timing summary of the solving routine.
  void PrintTimingData();

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Set the maximum order to use for the BDF method.
  void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; }

  /// Compute CFL
  // double ComputeCFL(ParGridFunction &u, double dt);
  double computeCFL(double dt);
};

#endif  // LOMACH_HPP_
