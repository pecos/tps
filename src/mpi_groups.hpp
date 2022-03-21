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
#ifndef MPI_GROUPS_HPP_
#define MPI_GROUPS_HPP_

#include <mpi.h>
#include <tps_config.h>

#include <mfem.hpp>
#include <string>

using namespace mfem;

class MPI_Groups {
 private:
  MPI_Session *mpi;
<<<<<<< HEAD

  Array<int> patchList_;  // list of patches visible to process

  std::unordered_map<int, MPI_Comm>
      patchGroupMap_;  // maps of all group communicators by patch, i.e. <patch, communicator_patch>
=======
  
  Array<int> patchList_; // list of patches visible to process
  
  std::unordered_map<int, MPI_Comm> patchGroupMap_; // maps of all group communicators by patch, i.e. <patch, communicator_patch>
>>>>>>> 11564b0... Changing logic for inlet/outlet group communicators

 public:
  MPI_Groups(MPI_Session *mpi);
  ~MPI_Groups();

  bool isGroupRoot(MPI_Comm);  // check if rank 0 on local group
  int groupSize(MPI_Comm);     // size of BC MPI group

  // function that creates the groups
  void init();

  void setPatch(int patchNum) { patchList_.Append(patchNum); }

  MPI_Session *getSession() { return mpi; }

  MPI_Comm getComm(int patch) { return patchGroupMap_[patch]; }

  std::string getParallelName(std::string name);
};

#endif  // MPI_GROUPS_HPP_
