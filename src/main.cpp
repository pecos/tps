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
 */

#include "tps.hpp"
#include "mfem.hpp"

int main(int argc, char *argv[]) {

  // apparaently mfem does this?
  //int size, rank;
  //MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, &size);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int status;
  
  {

    //std::cout << " AND BEGIN..." << endl;
    TPS::Tps tps;
    //std::cout << " check 1..." << endl;    

    tps.parseCommandLineArgs(argc, argv);
    //std::cout << " check 2..." << endl;        
    tps.parseInput();
    //std::cout << " check 3..." << endl;        
    tps.chooseDevices();
    //std::cout << " check 4..." << endl;        
    tps.chooseSolver();
    //std::cout << " check 5..." << endl;        
    tps.initialize();
    //std::cout << " check 6..." << endl;        

    if (tps.isVisualizationMode()) {  // post-process visualization process.
      tps.visualization();
    } else {  // regular forward time-integration.
      tps.solve();
    }
    //std::cout << " check 7..." << endl;        

    //std::cout << " back in main..." << endl;
    status = tps.getStatus();
    //std::cout << " getStatus..." << status << endl;
    
  }
  //std::cout << " here 1..." << status << endl;  
  
#if defined(_CUDA_)
  cudaDeviceReset();
#elif defined(_HIP_)
  hipDeviceReset();
#endif

  //std::cout << " here 2..." << status << endl;  
  return status;
  //std::cout << " returned..." << status << endl;
  
}
