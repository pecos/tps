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
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <mfem.hpp>
#include <sstream>

#include "tps.hpp"

int main(int argc, char *argv[]) {
  MPI_Session mpi(argc, argv);

  const char *inputFile = "<unknown>";
  bool showVersion = false;
  tps tps(mpi, argc, argv);
  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&inputFile, "-run", "--runFile", "Name of the input file with run options.");
  args.AddOption(&showVersion, "-v", "--version", "", "--no-version", "Print code version and exit");

  // device_config inferred from build setup
  std::string device_config = "cpu";
#ifdef _HIP_
  device_config = "hip";
#elif _CUDA_
  device_config = "cuda";
#endif

#ifdef DEBUG
  int threads = 0.;
  args.AddOption(&threads, "-thr", "--threads",
                 " Set -thr 1 so that the program stops at the beginning in debug mode for gdb attach.");
#endif

  args.Parse();
  if (!args.Good()) {
    if (mpi.Root()) args.PrintUsage(cout);
    return 1;
  }

  // version info
  tps.PrintHeader();
  if (showVersion) exit(0);

  if (mpi.Root()) args.PrintOptions(cout);

#ifdef DEBUG
  if (threads != 0) {
    int gdb = 0;
    cout << "Process " << mpi.WorldRank() + 1 << "/" << mpi.WorldSize() << ", id: " << getpid() << endl;
    while (gdb == 0) sleep(5);
  }
#endif

  const int NUM_GPUS_NODE = 4;
  Device device(device_config, mpi.WorldRank() % NUM_GPUS_NODE);
  if (mpi.Root()) device.Print();

  string inputFileName(inputFile);
  M2ulPhyS solver(mpi, inputFileName);

  // Initiate solver iterations
  solver.Iterate();

  return (solver.GetStatus());
}
