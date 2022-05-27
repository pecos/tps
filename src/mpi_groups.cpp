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
#include "mpi_groups.hpp"

#include <vector>

using namespace std;

MPI_Groups::MPI_Groups(MPI_Session* _mpi) : mpi(_mpi) {
  patchList_.SetSize(0);

  patchGroupMap_.clear();
}

MPI_Groups::~MPI_Groups() {
  for (auto pg = patchGroupMap_.begin(); pg != patchGroupMap_.end(); pg++) {
    MPI_Comm_free(&(pg->second));
  }
}

bool MPI_Groups::isGroupRoot(MPI_Comm group_comm) {
  int rank;
  MPI_Comm_rank(group_comm, &rank);
  if (rank == 0)
    return true;
  else
    return false;
}

int MPI_Groups::groupSize(MPI_Comm group_comm) {
  int numRanks;
  MPI_Comm_size(group_comm, &numRanks);
  return (numRanks);
}

void MPI_Groups::init() {
  int localMaxPatch = 0;
  if (patchList_.Size() > 0) localMaxPatch = patchList_.Max();
  int globalMax = 0.;
  MPI_Allreduce(&localMaxPatch, &globalMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  for (int p = 0; p <= globalMax; p++) {
    int processContainsPatch = 0;
    for (int n = 0; n < patchList_.Size(); n++) {
      if (patchList_[n] == p) processContainsPatch = p;
    }

    MPI_Comm_split(MPI_COMM_WORLD, processContainsPatch, mpi->WorldRank(), &patchGroupMap_[p]);
  }
}

string MPI_Groups::getParallelName(string serialName) {
  if (mpi->WorldSize() == 1) {  // serial name
    return serialName;
  } else {
    // find the dots
    vector<size_t> dots;
    size_t lastPoint = serialName.find(".");
    dots.push_back(lastPoint);
    size_t newPoint = 0;
    while (newPoint != string::npos) {
      newPoint = serialName.find(".", lastPoint + 1);
      if (newPoint != string::npos) {
        lastPoint = newPoint;
        dots.push_back(lastPoint);
      }
    }

    string baseName = serialName.substr(0, lastPoint);
    string extension = serialName.substr(lastPoint + 1, serialName.length() - 1);

    string parallelName = baseName + "." + to_string(mpi->WorldRank()) + "." + extension;
    return parallelName;
  }
}
