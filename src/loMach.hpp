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

#ifndef LOMACH_HPP_
#define LOMACH_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "argon_transport.hpp"
#include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "flow.hpp"
#include "io.hpp"
#include "loMach_options.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "thermoChem.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"
#include "turbModel.hpp"

/**
 * @brief Driver class for models based on low Mach, variable density formulation
 */
class LoMachSolver : public TPS::Solver {
 protected:
  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;
  LoMachOptions loMach_opts_;

  TurbModel *turbClass = nullptr;
  ThermoChem *tcClass = nullptr;
  Flow *flowClass = nullptr;

  MPI_Groups *groupsMPI = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // Run options
  RunConfiguration config;

  // History file
  std::ofstream histFile;

  // Early terminatation flag
  int earlyExit = 0;

  // Number of dimensions
  int dim;
  int nvel;

  // Number of equations
  int num_equation;

  // Is it ambipolar?
  bool ambipolar = false;

  // Is it two-temperature?
  bool twoTemperature_ = false;

  // loMach options to run as incompressible
  bool constantViscosity = false;
  bool constantDensity = false;
  bool incompressibleSolve = false;
  bool pFilter = false;

  // Average handler
  Averaging *average;

  double dt;
  double time;
  int iter;

  mfem::ParMesh *pmesh_ = nullptr;
  int dim_;
  int true_size_;
  Array<int> offsets_;

  // min/max element size
  double hmin, hmax;

  // domain extent
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  // space sizes;
  int Sdof, Pdof, Vdof, Ndof, NdofR0;
  int SdofInt, PdofInt, VdofInt, NdofInt, NdofR0Int;

  int MaxIters;
  double max_speed;
  double CFL;
  int numActiveSpecies, numSpecies;

  // exit status code;
  int exit_status_;

  // mapping from local to global element index
  int *locToGlobElem = nullptr;

  // just keep these saved for ease
  int numWalls, numInlets, numOutlets;

  /// Update the EXTk/BDF time integration coefficient.
  void SetTimeIntegrationCoefficients(int step);

  /// Enable/disable debug output.
  bool debug = false;

  /// Enable/disable verbose output.
  bool verbose = true;

  /// Enable/disable partial assembly of forms.
  bool partial_assembly = false;
  bool partial_assembly_pressure = true;

  /// Enable/disable numerical integration rules of forms.
  // bool numerical_integ = false;
  bool numerical_integ = true;

  /// The parallel mesh.
  ParMesh *pmesh = nullptr;

  /// The order of the velocity and pressure space.
  int order;
  int porder;
  int norder;

  // total number of mesh elements (serial)
  int nelemGlobal_;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes = nullptr;

  //// total space for compatibility
  ParFiniteElementSpace *fvfes = nullptr;
  ParFiniteElementSpace *fvfes2 = nullptr;

  /// Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec = nullptr;

  /// Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes = nullptr;

  Vector gridScaleSml;
  Vector gridScaleXSml;
  Vector gridScaleYSml;
  Vector gridScaleZSml;

  int max_bdf_order;  // input option now
  int cur_step = 0;
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

  // Timers.
  StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

  // Equations solved
  Equations eqSystem;

  // order of polynomials for auxiliary solution
  bool loadFromAuxSol;

  // Pointers to the different classes
  GasMixture *mixture;    // valid on host
  GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture

  Chemistry *chemistry_ = NULL;
  Radiation *radiation_ = NULL;

  /// Distance to nearest no-slip wall
  ParGridFunction *distance_ = nullptr;

  // to interface with existing code
  ParGridFunction *U = nullptr;
  ParGridFunction *Up = nullptr;

  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets = nullptr;
  BlockVector *u_block = nullptr;
  BlockVector *up_block = nullptr;

  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;

  std::vector<ParGridFunction *> visualizationVariables_;
  std::vector<std::string> visualizationNames_;
  AuxiliaryVisualizationIndexes visualizationIndexes_;
  ParGridFunction *plasma_conductivity_ = nullptr;
  ParGridFunction *joule_heating_ = nullptr;

  // I/O organizer
  IODataOrganizer ioData;

  // for plotting
  ParGridFunction resolution_gf;

  // grid information
  ParGridFunction *bufferGridScale = nullptr;
  ParGridFunction *bufferGridScaleX = nullptr;
  ParGridFunction *bufferGridScaleY = nullptr;
  ParGridFunction *bufferGridScaleZ = nullptr;

  ParGridFunction *bufferMeanUp = nullptr;

  bool channelTest;

 public:
  /// Ctor
  LoMachSolver(LoMachOptions loMach_opts, TPS::Tps *tps);

  /// Dtor TODO(trevilo): this needs to free memory
  virtual ~LoMachSolver() {}

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
  void initSolutionAndVisualizationVectors();
  void initialTimeStep();
  void solve();
  void updateU();
  void copyU();
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

  /// Initialize forms, solvers and preconditioners.
  void Setup(double dt);

  /// Return a pointer to the current total resolution ParGridFunction.
  ParGridFunction *GetCurrentResolution() { return &resolution_gf; }

  LoMachOptions GetOptions() { return loMach_opts_; }

  /// Enable partial assembly for every operator.
  void EnablePA(bool pa) { partial_assembly = pa; }

  /// Enable numerical integration rules. This means collocated quadrature at
  /// the nodal points.
  void EnableNI(bool ni) { numerical_integ = ni; }

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
