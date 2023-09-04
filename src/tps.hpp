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
#ifndef TPS_HPP_
#define TPS_HPP_

/** @file
 * Top-level TPS class
 */

#include <grvy.h>
#include <tps_config.h>

#undef PACKAGE_NAME
#undef PACKAGE_VERSION
#undef PACKAGE_TARNAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING

#include "tps_mfem_wrap.hpp"
#undef PACKAGE_NAME
#undef PACKAGE_VERSION
#undef PACKAGE_TARNAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING

#include <string>

#include "M2ulPhyS.hpp"
#include "logger.hpp"
#include "quasimagnetostatic.hpp"
#include "solver.hpp"
#include "tps2Boltzmann.hpp"
#include "utils.hpp"

namespace TPS {

/** \brief Top-level TPS class.
 *
 * Top-level class that provides input parsing, initialization of
 * devices (e.g., gpus), initialization of solver classes requested
 * through input file, and execution of solve.
 */
class Tps {
 private:
  MPI_Comm TPSCommWorld_;   // top-level mpi context
  std::string tpsVersion_;  // code version
  int rank_;                // local MPI rank
  int nprocs_;              // total number of MPI procs
  bool isRank0_;            // flag to indicate rank0

  // runtime controls
  std::string iFile_;              // name of runtime input file (new ini format)
  std::string input_solver_type_;  // choice of desired solver
  int numGpusPerRank_;             // number of GPUs to use per MPI rank

  // execution device controls
  std::string deviceConfig_;
  mfem::Device device_;

  // supported high-level physics configurations
  bool isFlowOnlyMode_;
  bool isEMOnlyMode_;
  bool isFlowEMCoupledMode_;

  // mesh
  std::string meshFile_;

  // pointer to solver implementation chosen at runtime
  TPS::Solver *solver_ = NULL;

  // post-process visualization mode
  bool isVisualizationMode_;

 public:
  Tps() : Tps(MPI_COMM_WORLD) {}
  Tps(MPI_Comm world);
  ~Tps();
  GRVY::GRVY_Input_Class iparse_;  ///< runtime input parser (from libgrvy)

  void chooseSolver();
  void chooseDevices();
  std::string getDeviceConfig() { return deviceConfig_; }

  // TPS::Solver getSolver() { return solver_; }

  /// Input parsing support (variants with default value supplied)
  template <typename T>
  void getInput(const char *name, T &var, T varDefault);

  /// Read optional mfem::Vector input (set to vdef if absent in input file)
  void getVec(const char *name, Vector &vec, size_t numElems, const Vector &vdef);

  /// Parsing support for required (scalar) inputs
  template <typename T>
  void getRequiredInput(const char *name, T &var);

  template <typename T>
  T getRequiredInput(const std::string &name) {
    T var;
    this->getRequiredInput(name.c_str(), var);
    return var;
  }

  template <typename T>
  T getInput(const std::string &name, T varDefault) {
    T var;
    this->getInput(name.c_str(), var, varDefault);
    return var;
  }

  /// Parsing support for required (vector) inputs
  void getRequiredVec(const char *name, std::vector<double> &var, size_t numElems);
  void getRequiredVec(const char *name, Vector &var, size_t numElems);
  void getRequiredVec(const char *name, Array<double> &var, size_t numElems);

  /// Parsing support for required (single element of vector) inputs
  void getRequiredVecElem(const char *name, double &var, int ithElem);

  /// Read string-string pairs for keyword [name] and store in var.
  void getRequiredPairs(const char *name, std::vector<pair<std::string, std::string>> &var);

  /// Get solver status
  int getStatus() { return solver_->getStatus(); }

  /// Initialize the requested solver
  void initialize() {
    solver_->initialize();
    return;
  }

  /// Execute the requested solve
  void solve() {
    solver_->solve();
    return;
  }

  /// Prepare solver before starting time loop
  void solveBegin() {
    solver_->solveBegin();
    return;
  }

  /// Advance the requested solver of one time step
  void solveStep() {
    solver_->solveStep();
    return;
  }

  /// Operation performed after the time loop
  void solveEnd() {
    solver_->solveEnd();
    return;
  }

  /// Execute the solver 'visualization' mode for post-processing
  void visualization() {
    solver_->visualization();
    return;
  }

  /// Initialize the interface
  void initInterface(Tps2Boltzmann &interface) { solver_->initInterface(interface); }

  /// Push solver variables to interface
  void push(Tps2Boltzmann &interface) { solver_->push(interface); }

  /// Fetch solver variables from interface
  void fetch(Tps2Boltzmann &interface) { solver_->fetch(interface); }

  void printHeader();
  void parseCommandLineArgs(int argc, char *argv[]);  // variant used in C++ interface
  void parseArgs(std::vector<std::string> argv);      // variant used in python interface

  void parseInput();
  void parseInputFile(std::string iFile);
  void closeInputFile() { iparse_.Close(); }

  [[deprecated("Use getTPSCommWorld")]] MPI_Comm getMPISession() { return TPSCommWorld_; }

  MPI_Comm getTPSCommWorld() { return TPSCommWorld_; }

  std::string &getInputFilename() { return iFile_; }
  bool isFlowEMCoupled() const { return isFlowEMCoupledMode_; }
  bool isVisualizationMode() const { return isVisualizationMode_; }
  const std::string &getSolverType() { return input_solver_type_; }
};

std::string ltrim(const std::string &s);
std::string rtrim(const std::string &s);
std::string trim(const std::string &s);

}  // end namespace TPS

#endif  // TPS_HPP_
