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
 * @copydoc io.hpp
 */
#include <grvy.h>
#include <hdf5.h>

#include "io.hpp"
#include "loMach.hpp"
#include "logger.hpp"

void LoMachSolver::write_restart_files_hdf5(hid_t file, bool serialized_write) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();

  // -------------------------------------------------------------------
  // Attributes - relevant solution metadata saved as attributes
  // -------------------------------------------------------------------

  // note: all tasks save unless we are writing a serial restart file
  if (rank0_ || !serialized_write) {
    // current iteration count
    h5_save_attribute(file, "iteration", iter);
    // total time
    h5_save_attribute(file, "time", temporal_coeff_.time);
    // timestep
    h5_save_attribute(file, "dt", temporal_coeff_.dt);
    // solution order
    h5_save_attribute(file, "order", order);
    // spatial dimension
    h5_save_attribute(file, "dimension", dim_);
    // thermodynamic pressure
    h5_save_attribute(file, "Po", thermoPressure_);

    if (average_->ComputeMean()) {
      // samples meanUp
      h5_save_attribute(file, "samplesMean", average_->GetSamplesMean());
      h5_save_attribute(file, "samplesRMS", average_->GetSamplesRMS());
      h5_save_attribute(file, "samplesInterval", average_->GetSamplesInterval());
    }

    // code revision
#ifdef BUILD_VERSION
    {
      hid_t ctype = H5Tcopy(H5T_C_S1);
      int shaLength = strlen(BUILD_VERSION);
      hsize_t dims[1] = {1};
      H5Tset_size(ctype, shaLength);

      hid_t dspace1dim = H5Screate_simple(1, dims, NULL);

      hid_t attr = H5Acreate(file, "revision", ctype, dspace1dim, H5P_DEFAULT, H5P_DEFAULT);
      assert(attr >= 0);
      herr_t status = H5Awrite(attr, ctype, BUILD_VERSION);
      assert(status >= 0);
      H5Sclose(dspace1dim);
      H5Aclose(attr);
    }
  }
#endif

  // included total dofs for partitioned files
  if (!serialized_write) {
    // int ldofs = sfes_->GetNDofs();
    int ldofs = meshData_->getDofSize();
    int gdofs;
    MPI_Allreduce(&ldofs, &gdofs, 1, MPI_INT, MPI_SUM, TPSCommWorld);
    h5_save_attribute(file, "dofs_global", gdofs);
  }

  // -------------------------------------------------------------------
  // Data - actual data write handled by IODataOrganizer class
  // -------------------------------------------------------------------
  ioData.write(file, serialized_write);
}

void LoMachSolver::read_restart_files_hdf5(hid_t file, bool serialized_read) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();
  int read_order;

  // normal restarts have each process read their own portion of the
  // solution; a serial restart only reads on rank 0 and distributes to
  // the remaining processes

  // -------------------------------------------------------------------
  // Attributes - read relevant solution metadata for this Solver
  // -------------------------------------------------------------------

  if (rank0_ || !serialized_read) {
    h5_read_attribute(file, "iteration", iter);
    h5_read_attribute(file, "time", temporal_coeff_.time);
    h5_read_attribute(file, "dt", temporal_coeff_.dt);
    h5_read_attribute(file, "order", read_order);
    if (H5Aexists(file, "Po")) {
      h5_read_attribute(file, "Po", thermoPressure_);
    }
    if (average_->ComputeMean() && average_->ContinueMean()) {
      int samplesMean, samplesRMS, intervals;
      h5_read_attribute(file, "samplesMean", samplesMean);
      h5_read_attribute(file, "samplesInterval", intervals);
      average_->SetSamplesMean(samplesMean);
      average_->SetSamplesInterval(intervals);

      int istatus = H5Aexists(file, "samplesRMS");
      if (istatus > 0) {
        h5_read_attribute(file, "samplesRMS", samplesRMS);
      } else {
        samplesRMS = samplesMean;
      }

      if (average_->RestartRMS() == true) {
        samplesRMS = 0;
      }
      average_->SetSamplesRMS(samplesRMS);
    }
  }

  if (serialized_read) {
    MPI_Bcast(&iter, 1, MPI_INT, 0, TPSCommWorld);
    MPI_Bcast(&(temporal_coeff_.time), 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&(temporal_coeff_.dt), 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&read_order, 1, MPI_INT, 0, TPSCommWorld);
    MPI_Bcast(&thermoPressure_, 1, MPI_DOUBLE, 0, TPSCommWorld);
  }

  if (rank0_) {
    grvy_printf(ginfo, "Restarting from iteration = %i\n", iter);
    grvy_printf(ginfo, "--> time = %e\n", temporal_coeff_.time);
    grvy_printf(ginfo, "--> dt   = %e\n", temporal_coeff_.dt);
  }

  // -------------------------------------------------------------------
  // Data - actual data read handled by IODataOrganizer class
  // -------------------------------------------------------------------
  ioData.read(file, serialized_read, read_order);
}

void LoMachSolver::restart_files_hdf5(string mode, string inputFileName) {
  assert((mode == "read") || (mode == "write"));
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();
#ifdef HAVE_GRVY
  grvy_timer_begin(__func__);
#endif

  string serialName;
  if (inputFileName.length() > 0) {
    if (inputFileName.substr(inputFileName.length() - 3) != ".h5") {
      grvy_printf(gerror, "[ERROR]: M2ulPhyS::restart_files_hdf5 - input file name has a wrong format -> %s\n",
                  inputFileName.c_str());
      grvy_printf(GRVY_INFO, "format: %s\n", (inputFileName.substr(inputFileName.length() - 3)).c_str());
      exit(ERROR);
    }
    serialName = inputFileName;
  } else {
    serialName = loMach_opts_.io_opts_.restart_dir_;
    serialName.append("/restart_");
    serialName.append(loMach_opts_.io_opts_.output_dir_);
    serialName.append(".sol.h5");
  }

  // determine restart file name (either serial or partitioned)
  string fileName;
  bool restart_serial = false;
  if (mode == "read") restart_serial = loMach_opts_.io_opts_.restart_serial_read_;
  if (mode == "write") restart_serial = loMach_opts_.io_opts_.restart_serial_write_;
  if (restart_serial) {
    fileName = serialName;
  } else {
    fileName = groupsMPI->getParallelName(serialName);
  }

  if (rank0_) cout << "HDF5 restart files mode: " << mode << endl;

  // open restart files
  hid_t file = -1;
  if (mode == "write") {
    if (rank0_ || !loMach_opts_.io_opts_.restart_serial_write_) {
      file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert(file >= 0);
    }
  } else if (mode == "read") {
    if (loMach_opts_.io_opts_.restart_serial_read_) {
      if (rank0_) {
        file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        assert(file >= 0);
      }
    } else {
      // verify we have all desired files and open on each process
      int gstatus;
      int status = static_cast<int>(file_exists(fileName));
      MPI_Allreduce(&status, &gstatus, 1, MPI_INT, MPI_MIN, TPSCommWorld);

      if (gstatus == 0) {
        grvy_printf(gerror, "[ERROR]: Unable to access desired restart file -> %s\n", fileName.c_str());
        exit(ERROR);
      }

      file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file >= 0);
    }
  }

  // -------------------------------------------------------------------
  // Attributes - relevant solution metadata saved as attributes
  // -------------------------------------------------------------------

  if (mode == "write") {
    write_restart_files_hdf5(file, loMach_opts_.io_opts_.restart_serial_write_);
  } else {  // read
    read_restart_files_hdf5(file, loMach_opts_.io_opts_.restart_serial_read_);
  }

  if (file >= 0) H5Fclose(file);

#ifdef HAVE_GRVY
  grvy_timer_end(__func__);
#endif
  return;
}
