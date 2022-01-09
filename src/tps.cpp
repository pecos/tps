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

namespace TPS {

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

  // execution device inferred from build setup
#ifdef _HIP_
  deviceConfig_ = "hip";
#elif _CUDA_
  deviceConfig_ = "cuda";
#else
  deviceConfig_ = "cpu";
#endif

}

void Tps::printHeader() {
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
void Tps::parseCommandLineArgs(int argc, char *argv[]) {

  mfem::OptionsParser args(argc,argv);
  bool showVersion = false;
  const char *astring  = iFile_.c_str();
  const char *astring2 = iFile_.c_str();

  if(isRank0_)
  {
    grvy_printf(GRVY_DEBUG,"# of command-line arguments = %i\n",argc);
    for(int i=0;i<argc;i++)
      grvy_printf(GRVY_DEBUG,"--> %s\n",argv[i]);
  }

  // Register supported command-line arguments
  args.AddOption(&showVersion, "-v", "--version", "", "--no-version", "Print code version and exit");
  args.AddOption(&astring, "-run", "--runFile", "Name of the input file with run options.");
  args.AddOption(&astring2, "-input", "--inputFile", "Name of ini-style input file with run options.");
  args.Parse();

  if (!args.Good()) {
    if (isRank0_) args.PrintUsage(std::cout);
    exit(ERROR);
  }

  // koomie TODO: update here when input file conversion complete
  iFile_old_ = astring;
  iFile_ = astring2;

  // Version info
  printHeader();
  if (showVersion)
    exit(0);

  args.PrintOptions(std::cout);

  return;
}

/// Setup desired execution devices for MFEM
void Tps::chooseDevices()
{
#ifdef _GPU_
  device_.Configure(deviceConfig_,nprocs_ % numGpusPerRank_);
#endif

  if(isRank0_)
  {
    printf("\n---------------------------------\n");
    printf("MFEM Device configuration:\n");
    device_.Print();
    printf("---------------------------------\n\n");
  }

  return;
}


/// Choose desired solver class
void Tps::chooseSolver() {

  if(input_solver_type_ == "flow") {
    isFlowOnlyMode_ = true;
    // koomie TODO: update here when input file conversion complete - should not need iFile_old_)
    solver_ = new M2ulPhyS(mpi_,iFile_old_,this);
  }
  else if (input_solver_type_ == "em") {
    isEMOnlyMode_ = true;
    iFile_ = iFile_old_;
    ElectromagneticOptions em_opt;
    solver_ = new QuasiMagnetostaticSolver(mpi_,em_opt,this);
  }
  else if (input_solver_type_ == "coupled") {
   isFlowEMCoupledMode_ = true;
   grvy_printf(GRVY_ERROR,"\nSlow your roll.  Solid high-five for whoever implements this coupled solver mode!\n");
   exit(ERROR);
  }
  else {
    grvy_printf(GRVY_ERROR,"\nUnsupported solver choice specified -> %s\n",input_solver_type_.c_str());
    exit(ERROR);
  }

  // with a solver chosen, we can now parse remaining solver-specific inputs
  solver_->parseSolverOptions();
}

/// Read runtime input file on single MPI process and distribute so that
/// runtime inputs are available for query on on all processors.
void Tps::parseInput() {

  std::stringstream buffer;
  std::string ss;

  if(isRank0_)
  {
    grvy_printf(GRVY_INFO,"\nCaching input file -> %s\n",iFile_.c_str());
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

  // parse common inputs
  getRequiredInput("solver/type",input_solver_type_);
#ifdef _GPU_
  getRequiredInput("gpu/numGpusPerRank",numGpusPerRank_);
#endif

  return;
}

/// read an input value for keyword [name] and store in var - error if value
/// is not supplied
template <typename T> void Tps::getRequiredInput(const char *name, T &var)
{
  if(iparse_.Read_Var(name,&var) == 0)
  {
    std::cout << "ERROR: Unable to read required input variable -> " << name << std::endl;
    exit(ERROR);
  }
  return;
}

/// read an input value for keyword [name] and store in var - use defaultValue if
/// keyword not present
template <typename T> void Tps::getInput(const char *name, T &var, T varDefault)
{
  if(iparse_.Read_Var(name,&var,varDefault) == 0)
  {
    std::cout << "ERROR: Unable to read input variable -> " << name << std::endl;
    exit(ERROR);
  }
  return;
}

// supported templates for getInput()
template void Tps::getInput <int>         (const char *name,    int &var, int    varDefault);
template void Tps::getInput <double>      (const char *name, double &var, double varDefault);
template void Tps::getInput <std::string> (const char *name, std::string &var, std::string varDefault);
template void Tps::getInput <bool>        (const char *name, bool &var, bool varDefault);

// supported templates for getRequiredInput()
template void Tps::getRequiredInput <int>    (const char *name,    int &var);
template void Tps::getRequiredInput <double> (const char *name, double &var);
template void Tps::getRequiredInput <std::string> (const char *name, std::string &var);

} // end namespace TPS
