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
#include "utils.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "M2ulPhyS.hpp"
#include "tps_mfem_wrap.hpp"

#ifdef HAVE_SLURM
#include <slurm/slurm.h>

// check if the amount of time left in SLURM job is less than desired threshold
bool slurm_job_almost_done(int threshold, int rank) {
  char *SLURM_JOB_ID = getenv("SLURM_JOB_ID");
  if (SLURM_JOB_ID == NULL) {
    printf("[ERROR]: SLURM_JOB_ID env variable not set. Unable to query how much time remaining\n");
    return false;
  }

  int64_t secsRemaining;
  if (rank == 0) secsRemaining = slurm_get_rem_time(atoi(SLURM_JOB_ID));

  MPI_Bcast(&secsRemaining, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  if (secsRemaining < 0) {
    printf("[WARN]: Unable to query how much time left for current SLURM job\n");
    return false;
  }

  if (rank == 0) printf("  --> Seconds remaining for current job = %li (threshold = %i)\n", secsRemaining, threshold);

  if (secsRemaining <= threshold) {
    if (rank == 0) {
      string delim(80, '=');
      cout << endl;
      cout << delim << endl;
      cout << "Remaining job time is less than " << threshold << " secs." << endl;
      cout << "--> Dumping final restart and terminating with exit_code = " << JOB_RESTART << endl;
      cout << "--> Use this restart exit code in your job scripts to auto-trigger a resubmit. " << endl;
      cout << delim << endl;
    }
    return true;
  } else {
    return false;
  }
}

// submit a new resource manager job with provided job script
int rm_restart(std::string jobFile, std::string mode) {
  // grvy_printf(GRVY_ginfo,"[RMS] Submitting fresh %s job using script -> %s\n",mode.c_str(),jobFile.c_str());
  std::cout << "[RMS] Submitting fresh" << mode << " job using script -> " << jobFile << std::endl;

  if (mode == "SLURM") {
    std::string command = "sbatch " + jobFile;
    std::string result = systemCmd(command.c_str());
    std::cout << result << std::endl;
  }

  return 0;
}
bool M2ulPhyS::Check_JobResubmit() {
  if (config.isAutoRestart()) {
    if (slurm_job_almost_done(config.rm_threshold(), mpi.WorldRank()))
      return true;
    else
      return false;
  } else {
    return false;
  }
}

#else
bool slurm_job_almost_done(int threshold, int rank) {
  std::cout << "SLURM functionality not enabled in this build" << std::endl;
  exit(1);
}

int rm_restart(std::string jobFile, std::string mode) {
  std::cout << "SLURM functionality not enabled in this build" << std::endl;
  exit(1);
}

#endif

// Check for existence of DIE file on rank 0
bool M2ulPhyS::Check_ExitEarly(int iter) {
  if ((iter % config.exit_checkFreq()) == 0) {
    int status;
    if (rank0_) {
      status = static_cast<int>(file_exists("DIE"));
      if (status == 1) grvy_printf(ginfo, "Detected DIE file, terminating early...\n");
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return (static_cast<bool>(status));
  } else {
    return (false);
  }
}

// check if file exists
bool file_exists(const std::string &name) { return (access(name.c_str(), F_OK) != -1); }

// Look for existence of output.pvd files in vis output directory and
// keep a sequentially numbered copy. Useful when doing restart,
// otherwise, MFEM ParaviewDataCollection output will delete old
// timesteps from the file on restarts.
void M2ulPhyS::Cache_Paraview_Timesteps() {
  const int MAX_FILES = 9999;
  char suffix[5];

  std::string fileName = config.GetOutputName() + "/output.pvd";
  if (file_exists(fileName)) {
    for (int i = 1; i <= MAX_FILES; i++) {
      snprintf(suffix, sizeof(suffix), "%04i", i);
      std::string testFile = fileName + "." + suffix;
      if (file_exists(testFile)) {
        continue;
      } else {
        std::cout << "--> caching output.pvd file -> " << testFile << std::endl;
        std::string command = "cp " + fileName + " " + testFile;
        systemCmd(command.c_str());
        // std::filesystem::copy_file(fileName,testFile);
        return;
      }
    }
  }

  return;
}

// Run system command and capture output
std::string systemCmd(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

void LocalProjectDiscCoefficient(GridFunction &gf, VectorCoefficient &coeff, Array<int> &dof_attr) {
  Array<int> vdofs;
  Vector vals;

  gf.HostWrite();
  FiniteElementSpace *fes = gf.FESpace();

  // maximal element attribute for each dof
  dof_attr.SetSize(fes->GetVSize());
  dof_attr = -1;

  // NB: DofTransformation does not exist until after mfem 4.3, so
  // don't use here yet
  // DofTransformation * doftrans = NULL;

  // local projection
  for (int i = 0; i < fes->GetNE(); i++) {
    // doftrans = fes->GetElementVDofs(i, vdofs);
    fes->GetElementVDofs(i, vdofs);
    vals.SetSize(vdofs.Size());
    fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

    // if (doftrans) {
    //   doftrans->TransformPrimal(vals);
    // }

    // the values in shared dofs are determined from the element with maximal
    // attribute
    int attr = fes->GetAttribute(i);
    for (int j = 0; j < vdofs.Size(); j++) {
      // Mimic how negative indices are handled in Vector::SetSubVector
      int ind = vdofs[j];
      if (ind < 0) {
        ind = -1 - ind;
        vals[j] = -vals[j];
      }

      if (attr > dof_attr[ind]) {
        gf(ind) = vals[j];
        dof_attr[ind] = attr;
      }
    }
  }
}

void GlobalProjectDiscCoefficient(ParGridFunction &gf, VectorCoefficient &coeff) {
  // local maximal element attribute for each dof
  Array<int> ldof_attr;

  // local projection
  LocalProjectDiscCoefficient(gf, coeff, ldof_attr);

  // global maximal element attribute for each dof
  Array<int> gdof_attr;
  ldof_attr.Copy(gdof_attr);

  ParFiniteElementSpace *pfes = gf.ParFESpace();

  GroupCommunicator &gcomm = pfes->GroupComm();
  gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Max);
  gcomm.Bcast(gdof_attr);

  // set local value to zero if global maximal element attribute is larger than
  // the local one, and mark (in gdof_attr) if we have the correct value
  for (int i = 0; i < pfes->GetVSize(); i++) {
    if (gdof_attr[i] > ldof_attr[i]) {
      gf(i) = 0.0;
      gdof_attr[i] = 0;
    } else {
      gdof_attr[i] = 1;
    }
  }

  // parallel averaging plus interpolation to determine final values
  HypreParVector *tv = pfes->NewTrueDofVector();
  gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Sum);
  gcomm.Bcast(gdof_attr);

  FiniteElementSpace *fes = gf.FESpace();
  for (int i = 0; i < fes->GetVSize(); i++) {
    gf(i) /= gdof_attr[i];
  }
  gf.ParallelAssemble(*tv);
  gf.Distribute(tv);
  delete tv;
}
