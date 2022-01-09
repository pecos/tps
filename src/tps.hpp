// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

#include <string>
#include <grvy.h>
#include <mfem.hpp>

#include "tps_config.h"
#include "logger.hpp"
#include "utils.hpp"
#include "solver.hpp"
#include "M2ulPhyS.hpp"
#include "quasimagnetostatic.hpp"

namespace TPS {

// Top-level TPS class
class Tps {

 private:
  std::string tpsVersion_;        // code version
  mfem::MPI_Session mpi_;         // top-level mpi context
  int rank_;                      // local MPI rank
  int nprocs_;                    // total number of MPI procs
  bool isRank0_;                  // flag to indicate rank0

  // runtime controls
  std::string iFile_;             // name of runtime input file (new ini format)
  std::string iFile_old_;         // name of runtime input file (old format)
  std::string input_solver_type_; // choice of desired solver
  int numGpusPerRank_;            // number of GPUs to use per MPI rank

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
  TPS::Solver *solver_;

 public:
  Tps(int argc, char *argv[]);    // constructor
  GRVY::GRVY_Input_Class iparse_; // runtime input parser

  void chooseSolver();
  void chooseDevices();
  std::string getDeviceConfig(){return deviceConfig_;};

  // input parsing support (variants with default value supplied)
  // supported types are T={int,double,bool,std::string}
  template <typename T> void getInput (const char *name, T &var, T varDefault);

  // input parsing support (variants value is required to be provided)
  // supported types are T={int,double,std::string}
  template <typename T> void getRequiredInput(const char *name, T &var);

  int  getStatus(){return solver_->getStatus();}
  void initialize(){solver_->initialize(); return;}
  void solve(){solver_->solve(); return;}
  void printHeader();
  void parseCommandLineArgs(int argc,char *argv[]);
  void parseInput();
};

} // end namespace TPS

#endif  // TPS_HPP_
