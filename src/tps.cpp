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
#include "tps.hpp"

Tps::Tps(int argc, char *argv[]) {

  nprocs_ = mpi_.WorldSize();
  rank_   = mpi_.WorldRank();
  if (rank_ == 0)
    isRank0_ = true;
  else
    isRank0_ = false;

  // default input file
  iFile_ = "runfile.ini";
  iFile_old_ = "unknown";

  // default physics configuration
  isFlowOnlyMode_      = true;
  isEMOnlyMode_        = false;
  isFlowEMCoupledMode_ = false;

}

void Tps::PrintHeader() {
  if (isRank0_) {
    grvy_printf(ginfo, "\n------------------------------------\n");
    grvy_printf(ginfo, "  _______ _____   _____\n");
    grvy_printf(ginfo, " |__   __|  __ \\ / ____|\n");
    grvy_printf(ginfo, "    | |  | |__) | (___  \n");
    grvy_printf(ginfo, "    | |  |  ___/ \\___ \\ \n");
    grvy_printf(ginfo, "    | |  | |     ____) | \n");
    grvy_printf(ginfo, "    |_|  |_|    |_____/ \n\n");
    grvy_printf(ginfo, "TPS Version:  %s (%s)\n", PACKAGE_VERSION, BUILD_DEVSTATUS);
    grvy_printf(ginfo, "Git Version:  %s\n", BUILD_VERSION);
    grvy_printf(ginfo, "MFEM Version: %s\n", mfem::GetVersionStr());
    grvy_printf(ginfo, "------------------------------------\n\n");
  }
}

/// Register and parse supported command line arguments and runtime inputs
void Tps::ParseCommandLineArgs(int argc, char *argv[]) {

  mfem::OptionsParser args(argc,argv);
  bool showVersion = false;
  const char *astring = iFile_.c_str();

  if(isRank0_)
  {
    grvy_printf(GRVY_INFO,"# of command-line arguments = %i\n",argc);
    for(int i=0;i<argc;i++)
      grvy_printf(GRVY_INFO,"--> %s\n",argv[i]);
  }

  // Register supported command-line arguments
  args.AddOption(&isFlowOnlyMode_, "-flow", "--flow-only", "-nflow", "--not-flow-only", "Perform flow only simulation");
  args.AddOption(&showVersion, "-v", "--version", "", "--no-version", "Print code version and exit");
  args.AddOption(&astring, "-run", "--runFile", "Name of the input file with run options.");
  args.Parse();

  if (!args.Good()) {
    if (isRank0_) args.PrintUsage(std::cout);
    exit(ERROR);
  }
  args.PrintOptions(std::cout);

  // koomie update here when input file conversion complete
  //iFile_ = astring;
  iFile_old_ = astring;

  // Version info
  PrintHeader();
  if (showVersion)
    exit(0);

  return;
}

/// Choose desired solver class
void Tps::ChooseSolver() {
  // koomie update here when input file conversion complete
  solver = new M2ulPhyS(mpi_,iFile_old_,this);

  // load solver specific inputs
  solver->parseSolverOptions();
}

void Tps::Iterate() {
  solver->Iterate();
}

/// Read runtime input file on single MPI process and distribute so that
/// runtime inputs are available for query on on all processors.
void Tps::ParseInput() {

  std::stringstream buffer;
  std::string ss;

  if(isRank0_)
  {
    grvy_printf(GRVY_INFO,"Caching input file -> %s\n",iFile_.c_str());
    std::ifstream file(iFile_);
    buffer << file.rdbuf();
  }

  // distribute buffer to remaining tasks
  int bufferSize = buffer.str().size();
  MPI_Bcast(&bufferSize,1,MPI_INT,0,MPI_COMM_WORLD);

  if(isRank0_)
    ss = buffer.str();
  else
    ss.resize(bufferSize);

  MPI_Bcast(&ss[0],ss.capacity(),MPI_CHAR,0,MPI_COMM_WORLD);
  buffer.str(ss);

  // now, all procs can load the input file contents for subsequent parsing
  if( !iparse_.Load(buffer) )
    {
      grvy_printf(GRVY_ERROR,"Unable to load runtime inputs from file -> %s\n",iFile_.c_str());
      exit(ERROR);
    }

  int flag = 1;

  // load common inputs needed for all solvers

  // [mesh] options
  flag *= iparse_.Read_Var("mesh/file",&meshFile_);

  std::cout << "meshfile [new] = " << meshFile_ << std::endl;

  return;
}
