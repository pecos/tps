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
 * C++ driver application for TPS simulation.
 * The purpose of this driver is to test TPS when running on a subcommunicator
 */

#include "tps.hpp"

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(argc, argv);
  int color(MPI_UNDEFINED);
  color = mfem::Mpi::WorldRank()%2;
  MPI_Comm tpsComm;
  MPI_Comm_split(MPI_COMM_WORLD, color, mfem::Mpi::WorldRank(), &tpsComm);


  int status(0);

  if(color==0)
  {


    TPS::Tps tps(tpsComm);

    tps.parseCommandLineArgs(argc, argv);
    tps.parseInput();
    tps.chooseDevices();
    tps.chooseSolver();
    tps.initialize();

    if (tps.isVisualizationMode()) {  // post-process visualization process.
      tps.visualization();
    } else {  // regular forward time-integration.
      tps.solve();
    }

    status = tps.getStatus();
  }

#if defined(_CUDA_)
  cudaDeviceReset();
#elif defined(_HIP_)
  hipDeviceReset();
#endif

  return status;
}