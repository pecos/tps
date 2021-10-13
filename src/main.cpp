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
#include <fstream>
#include <iostream>
#include <sstream>

#include "mfem.hpp"

// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
#include <unistd.h>

#include "M2ulPhyS.hpp"

int main(int argc, char *argv[]) {
  MPI_Session mpi(argc, argv);

  const char *inputFile = "../data/periodic-square.mesh";
  // const char *device_config = "cpu";

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&inputFile, "-run", "--runFile", "Name of the input file with run options.");
  //  args.AddOption(&device_config, "-d", "--device",
  // "Device configuration string, see Device::Configure().");

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

  // Start the timer.
  tic_toc.Clear();
  tic_toc.Start();

  solver.Iterate();

  tic_toc.Stop();
  if (mpi.Root()) cout << " done, " << tic_toc.RealTime() << "s." << endl;

  return (solver.GetStatus());
}
