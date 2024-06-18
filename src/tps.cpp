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

/** @file
 * @copydoc tps.hpp
 */

#include "tps.hpp"

#ifdef HAVE_MPI_EXT
#include <mpi-ext.h>
#endif

#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef HAVE_MPI4PY
#include <mpi4py/mpi4py.h>
#endif

#endif

#include "cycle_avg_joule_coupling.hpp"
#include "independent_coupling.hpp"
#include "loMach.hpp"

namespace TPS {

/** \brief Constructs default Tps object.
 *
 * Only generic initialization is done.  After construction, user must
 * call input parsing functions, choose solver, and initialize to set
 * information for simulation.  For an example, see main.cpp.
 */
Tps::Tps(MPI_Comm world) : TPSCommWorld_(world) {
  MPI_Comm_size(world, &nprocs_);
  MPI_Comm_rank(world, &rank_);
  if (rank_ == 0)
    isRank0_ = true;
  else
    isRank0_ = false;

  // default input file
  iFile_ = "runfile.ini";

  // default physics configuration
  isFlowOnlyMode_ = true;
  isEMOnlyMode_ = false;
  isFlowEMCoupledMode_ = false;

  // default post-process visualization mode
  isVisualizationMode_ = false;

  // execution device inferred from build setup
#ifdef _HIP_
  deviceConfig_ = "hip";
#elif _CUDA_
  deviceConfig_ = "cuda";
#else
  deviceConfig_ = "cpu";
#endif
}

/// Destructor
Tps::~Tps() {
  if (solver_ != NULL) delete solver_;
}

/// Print a pretty ascii header including version information
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

#ifdef _MASA_
    grvy_printf(ginfo, "MASA is enabled. \n\n");
#endif
  }
}

/// Register and parse supported command line arguments and runtime inputs
void Tps::parseCommandLineArgs(int argc, char *argv[]) {
  mfem::OptionsParser args(argc, argv);
  bool showVersion = false;
  bool debugMode = false;
  bool visualMode = false;
  const char *astring = iFile_.c_str();

  if (isRank0_) {
    grvy_printf(GRVY_DEBUG, "# of command-line arguments = %i\n", argc);
    for (int i = 0; i < argc; i++) grvy_printf(GRVY_DEBUG, "--> %s\n", argv[i]);
  }

  // Register supported command-line arguments
  args.AddOption(&showVersion, "-v", "--version", "", "--no-version", "Print code version and exit,");
  args.AddOption(&astring, "-run", "--runFile", "Name of the input file with run options.");
  args.AddOption(&debugMode, "-d", "--debug", "", "--no-debug", "Launch in debug mode for gdb attach.");
  args.AddOption(&visualMode, "-visual", "--visualization", "", "--no-visualization",
                 "Launch post-process visualization.");
  gpu_aware_mpi_ = false;
  args.AddOption(&gpu_aware_mpi_, "-ga", "--gpu-aware-mpi", "", "--no-gpu-aware-mpi", "Set GPU-aware MPI.");
  args.Parse();

  if (!args.Good()) {
    if (isRank0_) args.PrintUsage(std::cout);
    exit(ERROR);
  }

  iFile_ = astring;
  isVisualizationMode_ = visualMode;

  // Version info
  printHeader();
  if (showVersion) exit(0);

  if (isRank0_) args.PrintOptions(std::cout);

  // Debug mode: user's can attach gdb to the rank0 PID and set the gdb
  // variable to be non-zero in order to exit the startup sleep process.
  // gdb syntax is: set var gdb = 1 then you can type "continue" to run under
  // the debugger
  if (debugMode) {
    volatile int gdb = 1;
    MPI_Barrier(TPSCommWorld_);
    if (isRank0_) {
      gdb = 0;
      grvy_printf(GRVY_INFO, "\nDEBUG Mode enabled:\n");
      grvy_printf(GRVY_INFO, "--> Rank 0 PID = %i\n", getpid());
    }

    while (gdb == 0) sleep(5);
    MPI_Barrier(TPSCommWorld_);
  }

  return;
}

/** \brief Wrapper function for parseCommandLineArgs
 *
 * This wrapper takes the command-line args as a vector of strings.
 * It is intended for use with Python interface.
 */
void Tps::parseArgs(std::vector<std::string> argv) {
  std::vector<char *> argv_char;
  for (auto &s : argv) argv_char.push_back(&s.front());
  parseCommandLineArgs(argv_char.size(), argv_char.data());
}

/// Setup desired execution devices for MFEM
void Tps::chooseDevices() {
#ifdef _GPU_
  device_.Configure(deviceConfig_, rank_ % numGpusPerRank_);

#if _CUDA_ && defined(MPIX_CUDA_AWARE_SUPPORT)
  // check for cuda-aware mpi (if possible) and overwrite flag if needed
  // Trust the command line flag if  MPIX_Query_cuda_support is not available
  int mpi_gpu_aware = MPIX_Query_cuda_support();

  if (mpi_gpu_aware == 1 && gpu_aware_mpi_ == false) {
    if (isRank0_) {
      grvy_printf(GRVY_WARN, "Cuda-aware MPI detected, but flag is false");
    }
    gpu_aware_mpi_ = true;
  } else if (mpi_gpu_aware == 0 && gpu_aware_mpi_ == true) {
    if (isRank0_) {
      grvy_printf(GRVY_WARN, "No cuda-aware MPI detected, but flag is true");
    }
    gpu_aware_mpi_ = false;
  }

#endif

  device_.SetGPUAwareMPI(gpu_aware_mpi_);
#endif

  if (isRank0_) {
    printf("\n---------------------------------\n");
    printf("MFEM Device configuration:\n");
    device_.Print();
    printf("---------------------------------\n\n");
  }

#ifdef _GPU_
  if (isRank0_) {
    if (gpu_aware_mpi_) {
      grvy_printf(GRVY_INFO, "\nTPS is using GPU-aware MPI.\n");
    } else {
      grvy_printf(GRVY_INFO, "\nTPS is using non-GPU-aware MPI.\n");
    }
  }
#endif
  return;
}

/** \brief Choose desired solver class
 *
 * Based on the solver requested in the input file, instantiate the
 * appropriate solver class and parse the input file for
 * solver-specific options.
 */
void Tps::chooseSolver() {
  if (input_solver_type_ == "flow") {
    isFlowOnlyMode_ = true;
    solver_ = new M2ulPhyS(iFile_, this);
  } else if (input_solver_type_ == "em") {
    isEMOnlyMode_ = true;
    ElectromagneticOptions em_opt;
    solver_ = new QuasiMagnetostaticSolver3D(em_opt, this);
  } else if (input_solver_type_ == "loMach") {
    solver_ = new LoMachSolver(this);
  } else if (input_solver_type_ == "em-axi") {
    isEMOnlyMode_ = true;
    ElectromagneticOptions em_opt;
    solver_ = new QuasiMagnetostaticSolverAxiSym(em_opt, this);
  } else if (input_solver_type_ == "independent-coupled") {
    isFlowEMCoupledMode_ = true;
    solver_ = new IndependentCoupling(iFile_, this);
  } else if (input_solver_type_ == "cycle-avg-joule-coupled") {
    isFlowEMCoupledMode_ = true;
    solver_ = new CycleAvgJouleCoupling(iFile_, this);
  } else if (input_solver_type_ == "coupled") {
    isFlowEMCoupledMode_ = true;
    grvy_printf(GRVY_ERROR, "\nSlow your roll.  Solid high-five for whoever implements this coupled solver mode!\n");
    exit(ERROR);
  } else {
    grvy_printf(GRVY_ERROR, "\nUnsupported solver choice specified -> %s\n", input_solver_type_.c_str());
    exit(ERROR);
  }

  // with a solver chosen, we can now parse remaining solver-specific inputs
  solver_->parseSolverOptions();
}

/** \brief Read runtime input file
 *
 * The file is read on single MPI process and then distributed so that
 * runtime inputs are available for query on on all processors.
 */
void Tps::parseInputFile(std::string iFile) {
  iFile_ = iFile;
  parseInput();
}

/** @copydoc Tps::parseInputFile
 *
 * This variant reads from file specified by private variable iFile_,
 * which can be set, for example, by Tps::parseCommandLineArgs
 */
void Tps::parseInput() {
  std::stringstream buffer;
  std::string ss;

  if (isRank0_) {
    grvy_printf(GRVY_INFO, "\nCaching input file -> %s\n", iFile_.c_str());
    std::ifstream file(iFile_);
    buffer << file.rdbuf();
  }

  // distribute buffer to remaining tasks
  int bufferSize = buffer.str().size();
  MPI_Bcast(&bufferSize, 1, MPI_INT, 0, TPSCommWorld_);

  if (isRank0_)
    ss = buffer.str();
  else
    ss.resize(bufferSize);

  MPI_Bcast(&ss[0], ss.capacity(), MPI_CHAR, 0, TPSCommWorld_);
  buffer.str(ss);

  // now, all procs can load the input file contents for subsequent parsing
  if (!iparse_.Load(buffer)) {
    grvy_printf(GRVY_ERROR, "Unable to load runtime inputs from file -> %s\n", iFile_.c_str());
    exit(ERROR);
  }

  // if (isRank0_) iparse_.Fdump();
  // parse common inputs
  getInput("solver/type", input_solver_type_, std::string("flow"));
#ifdef _GPU_
  getInput("gpu/numGpusPerRank", numGpusPerRank_, 1);
#endif

  return;
}

/** Read an input value for keyword [name] and store in var - use defaultValue if
 *  keyword not present.  Supported types are T={int,double,bool,std::string}
 */
template <typename T>
void Tps::getInput(const char *name, T &var, T varDefault) {
  if (!iparse_.Read_Var(name, &var, varDefault)) {
    std::cout << "ERROR: Unable to read input variable -> " << name << std::endl;
    exit(ERROR);
  }
  return;
}

/** Read an input vector for keyword [name] and store in var - use vdef if
 *  keyword not present.  Only mfem::Vector supported.
 */
void Tps::getVec(const char *name, Vector &vec, size_t numElems, const Vector &vdef) {
  if ((size_t)vdef.Size() < numElems) exit(ERROR);
  if ((size_t)vec.Size() < numElems) vec.SetSize(numElems);
  if (!iparse_.Read_Var_Vec(name, vec.HostWrite(), numElems)) {
    grvy_printf(GRVY_INFO, "Setting input vector %s to default.", name);
    double *hv = vec.HostWrite();
    const double *hd = vdef.HostRead();
    for (size_t i = 0; i < numElems; i++) {
      hv[i] = hd[i];
    }
  }
}

/** Read an input value for keyword [name] and store in var.  If
 *  keyword not present in input, exits with error.  Supported types
 *  are T={int,double,bool,std::string}
 */
template <typename T>
void Tps::getRequiredInput(const char *name, T &var) {
  if (!iparse_.Read_Var(name, &var)) {
    std::cout << "ERROR: Unable to read required input variable -> " << name << std::endl;
    exit(ERROR);
  }
  return;
}

/** \brief Input parsing for vector quantities that must be provided.
 *
 *  @param[in] name of require input
 *  @param[out] var vector variable to set (resized if necessary)
 *  @param[in] numElems length of vector
 */
void Tps::getRequiredVec(const char *name, std::vector<double> &vec, size_t numElems) {
  if (vec.size() < numElems) vec.reserve(numElems);
  if (!iparse_.Read_Var_Vec(name, vec.data(), numElems)) {
    std::cout << "ERROR: Unable to read input vector -> " << name << std::endl;
    exit(ERROR);
  }
}

/// @copydoc Tps::getRequiredVec(const char *,std::vector<double> &,size_t)
void Tps::getRequiredVec(const char *name, Vector &vec, size_t numElems) {
  if ((size_t)vec.Size() < numElems) vec.SetSize(numElems);
  if (!iparse_.Read_Var_Vec(name, vec.HostWrite(), numElems)) {
    std::cout << "ERROR: Unable to read input vector -> " << name << std::endl;
    exit(ERROR);
  }
}

/// @copydoc Tps::getRequiredVec(const char *,Vector &,size_t)
void Tps::getRequiredVec(const char *name, mfem::Array<double> &vec, size_t numElems) {
  if ((size_t)vec.Size() < numElems) vec.SetSize(numElems);
  if (!iparse_.Read_Var_Vec(name, vec.HostWrite(), numElems)) {
    std::cout << "ERROR: Unable to read input vector -> " << name << std::endl;
    exit(ERROR);
  }
}

/** Input parsing for an individual element of a vector that must be provided.
 *
 * @param[in] name of require input
 * @param[inout] var vector variable to set
 * @param[in] ithElem index of element to set
 */
void Tps::getRequiredVecElem(const char *name, double &value, int ithElem) {
#if 1
  if (!iparse_.Read_Var_iVec(name, &value, ithElem)) {
    grvy_printf(GRVY_ERROR, "Unable to read %ith element from input vector -> %s\n", ithElem, name);
    exit(ERROR);
  }
#endif
}

void Tps::getRequiredPairs(const char *name, std::vector<pair<std::string, std::string>> &var) {
  std::string inputString;
  if (!iparse_.Read_Var(name, &inputString)) {
    std::cout << "ERROR: Unable to read required input variable -> " << name << std::endl;
    exit(ERROR);
  }
  const int len = inputString.length();
  if ((inputString[0] != '{') || (inputString[len - 1] != '}')) {
    std::cout << "ERROR: " << name << " does not have a map format {...}" << std::endl;
    exit(ERROR);
  }
  std::string content = inputString.substr(1, len - 2);

  var.clear();
  std::stringstream s_stream(content);
  while (s_stream.good()) {
    std::string substr;
    std::getline(s_stream, substr, ',');  // get first string delimited by comma

    std::stringstream sub_stream(substr);
    std::string left, right;
    std::pair<std::string, std::string> subvar;
    std::getline(sub_stream, left, ':');
    std::getline(sub_stream, right, ':');
    subvar.first = trim(left);
    subvar.second = trim(right);

    var.push_back(subvar);
  }
  return;
}

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(" \n\r\t\f\v");
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(" \n\r\t\f\v");
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) { return rtrim(ltrim(s)); }

// supported templates for getInput()
template void Tps::getInput<int>(const char *name, int &var, int varDefault);
template void Tps::getInput<double>(const char *name, double &var, double varDefault);
template void Tps::getInput<std::string>(const char *name, std::string &var, std::string varDefault);
template void Tps::getInput<bool>(const char *name, bool &var, bool varDefault);

// supported templates for getRequiredInput()
template void Tps::getRequiredInput<int>(const char *name, int &var);
template void Tps::getRequiredInput<double>(const char *name, double &var);
template void Tps::getRequiredInput<std::string>(const char *name, std::string &var);

}  // end namespace TPS

// --------------------------------
// Python interface using pybind11
// --------------------------------

#ifdef HAVE_PYTHON

namespace py = pybind11;

namespace tps_wrappers {

void tps(py::module &m) {
#ifdef HAVE_MPI4PY
  // initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    // mpi4py calls the Python C API
    // we let pybind11 give us the detailed traceback
    throw py::error_already_set();
  }
#endif

  py::class_<TPS::Tps>(m, "Tps")
      .def(py::init<>())
#ifdef HAVE_MPI4PY
      .def(py::init([](py::object py_comm) {
        auto comm_ptr = PyMPIComm_Get(py_comm.ptr());
        if (!comm_ptr) throw py::error_already_set();
        return new TPS::Tps(*comm_ptr);
      }))
#endif
      .def("chooseDevices", &TPS::Tps::chooseDevices)
      .def("chooseSolver", &TPS::Tps::chooseSolver)
      .def("getStatus", &TPS::Tps::getStatus)
      .def("initialize", &TPS::Tps::initialize)
      .def("parseCommandLineArgs", &TPS::Tps::parseArgs)
      .def("parseInput", &TPS::Tps::parseInput)
      .def("getRequiredInput",
           static_cast<double (TPS::Tps::*)(const std::string &)>(&TPS::Tps::getRequiredInput<double>))
      .def("getRequiredInput", static_cast<int (TPS::Tps::*)(const std::string &)>(&TPS::Tps::getRequiredInput<int>))
      .def("getInput", static_cast<double (TPS::Tps::*)(const std::string &, double)>(&TPS::Tps::getInput<double>))
      .def("getInput", static_cast<int (TPS::Tps::*)(const std::string &, int)>(&TPS::Tps::getInput<int>))
      .def("solve", &TPS::Tps::solve)
      .def("solveBegin", &TPS::Tps::solveBegin)
      .def("solveStep", &TPS::Tps::solveStep)
      .def("solveEnd", &TPS::Tps::solveEnd)
      .def("push", &TPS::Tps::push)
      .def("fetch", &TPS::Tps::fetch)
      .def("initInterface", &TPS::Tps::initInterface);
}

}  // namespace tps_wrappers
#endif
