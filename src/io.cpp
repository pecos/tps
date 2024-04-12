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
#include "io.hpp"

#include <hdf5.h>

#include "M2ulPhyS.hpp"
#include "utils.hpp"

void M2ulPhyS::write_restart_files_hdf5(hid_t file, bool serialized_write) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();

  // -------------------------------------------------------------------
  // Attributes - relevant solution metadata saved as attributes
  // -------------------------------------------------------------------

  // note: all tasks save unless we are writing a serial restart file
  if (rank0_ || !serialized_write) {
    // current iteration count
    h5_save_attribute(file, "iteration", iter);
    // total time
    h5_save_attribute(file, "time", time);
    // timestep
    h5_save_attribute(file, "dt", dt);
    // solution order
    h5_save_attribute(file, "order", order);
    // spatial dimension
    h5_save_attribute(file, "dimension", dim);

    if (average->ComputeMean()) {
      // samples meanUp
      h5_save_attribute(file, "samplesMean", average->GetSamplesMean());
      h5_save_attribute(file, "samplesRMS", average->GetSamplesRMS());
      h5_save_attribute(file, "samplesInterval", average->GetSamplesInterval());
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
    int ldofs = vfes->GetNDofs();
    int gdofs;
    MPI_Allreduce(&ldofs, &gdofs, 1, MPI_INT, MPI_SUM, TPSCommWorld);
    h5_save_attribute(file, "dofs_global", gdofs);
  }

  // -------------------------------------------------------------------
  // Data - actual data write handled by IODataOrganizer class
  // -------------------------------------------------------------------
  ioData.write(file, serialized_write);
}

void M2ulPhyS::read_restart_files_hdf5(hid_t file, bool serialized_read) {
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
    h5_read_attribute(file, "time", time);
    h5_read_attribute(file, "dt", dt);
    h5_read_attribute(file, "order", read_order);
    if (average->ComputeMean() && config.GetRestartMean()) {
      int samplesMean, samplesRMS, intervals;
      h5_read_attribute(file, "samplesMean", samplesMean);
      h5_read_attribute(file, "samplesInterval", intervals);
      average->SetSamplesMean(samplesMean);
      average->SetSamplesInterval(intervals);

      int istatus = H5Aexists(file, "samplesRMS");
      if (istatus > 0) {
        h5_read_attribute(file, "samplesRMS", samplesRMS);
      } else {
        samplesRMS = samplesMean;
      }

      if (config.avg_opts_.zero_variances_ == true) {
        samplesRMS = 0;
      }
      average->SetSamplesRMS(samplesRMS);
    }
  }

  if (serialized_read) {
    MPI_Bcast(&iter, 1, MPI_INT, 0, TPSCommWorld);
    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&read_order, 1, MPI_INT, 0, TPSCommWorld);
    if (average->ComputeMean() && config.GetRestartMean()) {
      int sampMean = average->GetSamplesMean();
      int sampRMS = average->GetSamplesRMS();
      int intervals = average->GetSamplesInterval();
      MPI_Bcast(&sampMean, 1, MPI_INT, 0, TPSCommWorld);
      MPI_Bcast(&intervals, 1, MPI_INT, 0, TPSCommWorld);
      MPI_Bcast(&sampRMS, 1, MPI_INT, 0, TPSCommWorld);
      average->SetSamplesMean(sampMean);
      average->SetSamplesInterval(intervals);
      average->SetSamplesRMS(sampRMS);
    }
  }

  if (rank0_) {
    grvy_printf(ginfo, "Restarting from iteration = %i\n", iter);
    grvy_printf(ginfo, "--> time = %e\n", time);
    grvy_printf(ginfo, "--> dt   = %e\n", dt);
    if (average != NULL) {
      grvy_printf(ginfo, "Restarting averages with %i\n samples", average->GetSamplesMean());
    }
  }

  // -------------------------------------------------------------------
  // Data - actual data read handled by IODataOrganizer class
  // -------------------------------------------------------------------
  ioData.read(file, serialized_read, read_order);

  if (loadFromAuxSol) {
    // Update primitive variables.  This will be done automatically
    // before taking a time step, but we may write paraview output
    // prior to that, in which case the primitives are incorrect.
    // We would like to just call rhsOperator::updatePrimitives, but
    // rhsOperator has not been constructed yet.  As a workaround,
    // that code is duplicated here.
    // TODO(kevin): use mixture comptue primitive.
    double *dataUp = Up->HostReadWrite();
    const double *x = U->HostRead();
    for (int i = 0; i < vfes->GetNDofs(); i++) {
      Vector conserved(num_equation);
      Vector primitive(num_equation);
      for (int eq = 0; eq < num_equation; eq++) conserved[eq] = x[i + eq * vfes->GetNDofs()];
      mixture->GetPrimitivesFromConservatives(conserved, primitive);
      for (int eq = 0; eq < num_equation; eq++) dataUp[i + eq * vfes->GetNDofs()] = primitive[eq];
    }
  }
}

void M2ulPhyS::restart_files_hdf5(string mode, string inputFileName) {
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
    serialName = "restart_";
    serialName.append(config.GetOutputName());
    serialName.append(".sol.h5");
  }

  // determine restart file name (either serial or partitioned)
  string fileName;
  if (config.isRestartSerialized(mode)) {
    fileName = serialName;
  } else {
    fileName = groupsMPI->getParallelName(serialName);
  }

  if (rank0_) cout << "HDF5 restart files mode: " << mode << endl;

  // open restart files
  hid_t file = -1;
  if (mode == "write") {
    if (rank0_ || config.isRestartPartitioned(mode)) {
      file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert(file >= 0);
    }
  } else if (mode == "read") {
    if (config.isRestartSerialized(mode)) {
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
    write_restart_files_hdf5(file, config.isRestartSerialized(mode));
  } else {  // read
    read_restart_files_hdf5(file, config.isRestartSerialized(mode));
  }

  if (file >= 0) H5Fclose(file);

#ifdef HAVE_GRVY
  grvy_timer_end(__func__);
#endif
  return;
}

void partitioning_file_hdf5(std::string mode, MPI_Groups *groupsMPI, int nelemGlobal, Array<int> &partitioning) {
  MPI_Comm TPSCommWorld = groupsMPI->getTPSCommWorld();
  const bool rank0 = groupsMPI->isWorldRoot();
  const int nprocs = groupsMPI->getTPSWorldSize();

  grvy_timer_begin(__func__);

  // only rank 0 writes partitioning file
  if (!rank0 && (mode == "write")) return;

  // hid_t file, dataspace, data_soln;
  hid_t file = -1, dataspace;
  herr_t status;
  std::string fileName("partition");
  fileName += "." + std::to_string(nprocs) + "p.h5";

  assert((mode == "read") || (mode == "write"));

  if (mode == "write") {
    assert(partitioning.Size() > 0);

    if (file_exists(fileName)) {
      grvy_printf(gwarn, "Removing existing partition file: %s\n", fileName.c_str());
      string command = "rm " + fileName;
      int err = system(command.c_str());
      if (err != 0) grvy_printf(gwarn, "%s returned error code %d", command.c_str(), err);
    } else {
      grvy_printf(ginfo, "Saving original domain decomposition partition file: %s\n", fileName.c_str());
    }
    file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else if (mode == "read") {
    if (file_exists(fileName)) {
      file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    } else {
      grvy_printf(gerror, "[ERROR]: Unable to access necessary partition file on restart -> %s\n", fileName.c_str());
      exit(ERROR);
    }

    assert(file >= 0);
  }

  if (mode == "write") {
    // Attributes
    h5_save_attribute(file, "numProcs", nprocs);

    // Raw partition info
    hsize_t dims[1];
    hid_t data;

    dims[0] = partitioning.Size();
    dataspace = H5Screate_simple(1, dims, NULL);
    assert(dataspace >= 0);

    data = H5Dcreate2(file, "partitioning", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(data >= 0);
    status = H5Dwrite(data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, partitioning.HostRead());
    assert(status >= 0);
    H5Dclose(data);
    H5Fclose(file);
  } else if (mode == "read") {
    partitioning.SetSize(nelemGlobal);

    if (rank0) {
      grvy_printf(ginfo, "Reading original domain decomposition partition file: %s\n", fileName.c_str());

      // verify partition info matches current proc count
      {
        int np = 0;
        h5_read_attribute(file, "numProcs", np);
        grvy_printf(ginfo, "--> # partitions defined = %i\n", np);
        if (np != nprocs) {
          grvy_printf(gerror, "[ERROR]: Partition info does not match current processor count -> %i\n", nprocs);
          exit(ERROR);
        }
      }

      // read partition vector
      hid_t data;
      hsize_t numInFile;
      data = H5Dopen2(file, "partitioning", H5P_DEFAULT);
      assert(data >= 0);
      dataspace = H5Dget_space(data);
      numInFile = H5Sget_simple_extent_npoints(dataspace);

      assert((int)numInFile == nelemGlobal);

      status = H5Dread(data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, partitioning.HostWrite());
      assert(status >= 0);
      H5Dclose(data);
    }  // <-- end rank0_

#if 0
    // distribute partition vector to all procs
    MPI_Bcast(partitioning_.GetData(), nelemGlobal, MPI_INT, 0, TPSCommWorld);
#endif

    // distribute partition vector to all procs (serialzed per process variant)
    int tag = 21;

    if (rank0) {
      for (int rank = 1; rank < nprocs; rank++) {
        MPI_Send(partitioning.HostRead(), nelemGlobal, MPI_INT, rank, tag, TPSCommWorld);
        grvy_printf(gdebug, "Sent partitioning data to rank %i\n", rank);
      }
    } else {
      MPI_Recv(partitioning.HostWrite(), nelemGlobal, MPI_INT, 0, tag, TPSCommWorld, MPI_STATUS_IGNORE);
    }
    if (rank0) grvy_printf(ginfo, "--> partition file read complete\n");
  }
  grvy_timer_end(__func__);
}

// convenience function to check size of variable in hdf5 file
hsize_t get_variable_size_hdf5(hid_t file, std::string name) {
  hid_t data_soln = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  assert(data_soln >= 0);
  hid_t dataspace = H5Dget_space(data_soln);
  hsize_t numInSoln = H5Sget_simple_extent_npoints(dataspace);
  H5Dclose(data_soln);
  return numInSoln;
}

// convenience function to read solution data for parallel restarts
void read_variable_data_hdf5(hid_t file, string varName, size_t index, double *data) {
  hid_t data_soln;
  herr_t status;

  data_soln = H5Dopen2(file, varName.c_str(), H5P_DEFAULT);
  assert(data_soln >= 0);
  status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[index]);
  assert(status >= 0);
  H5Dclose(data_soln);
}

IOOptions::IOOptions() : output_dir_("output"), restart_mode_("standard") {}

void IOOptions::read(TPS::Tps *tps, std::string prefix) {
  std::string basename;
  if (!prefix.empty()) {
    basename = prefix + "/io";
  } else {
    basename = "io";
  }
  tps->getInput((basename + "/outdirBase").c_str(), output_dir_, std::string("output-default"));
  tps->getInput((basename + "/enableRestart").c_str(), enable_restart_, false);
  tps->getInput((basename + "/restartFromLTE").c_str(), enable_restart_from_lte_, false);
  tps->getInput((basename + "/exitCheckFreq").c_str(), exit_check_frequency_, 500);
  assert(exit_check_frequency_ > 0);

  tps->getInput((basename + "/restartMode").c_str(), restart_mode_, std::string("standard"));
  setRestartFlags();
}

void IOOptions::setRestartFlags() {
  if (restart_mode_ == "variableP") {
    restart_variable_order_ = true;
    restart_serial_write_ = false;
    restart_serial_read_ = false;
  } else if (restart_mode_ == "singleFileWrite") {
    restart_variable_order_ = false;
    restart_serial_write_ = true;
    restart_serial_read_ = false;
  } else if (restart_mode_ == "singleFileRead") {
    restart_variable_order_ = false;
    restart_serial_write_ = false;
    restart_serial_read_ = true;
  } else if (restart_mode_ == "singleFileReadWrite") {
    restart_variable_order_ = false;
    restart_serial_write_ = true;
    restart_serial_read_ = true;
  } else if (restart_mode_ == "standard") {
    restart_variable_order_ = false;
    restart_serial_write_ = false;
    restart_serial_read_ = false;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown restart mode -> %s\n", restart_mode_.c_str());
    exit(1);
  }
}

// convenience function to read and distribute solution data for serialized restarts
void IOFamily::readDistributeSerializedVariable(hid_t file, const IOVar &var, int numDof, double *data) {
  std::string varName = group_ + "/" + var.varName_;
  if (rank0_) grvy_printf(ginfo, "--> Reading h5 path = %s\n", varName.c_str());

  const int varOffset = var.index_;

  MPI_Comm comm = pfunc_->ParFESpace()->GetComm();
  const int myrank = pfunc_->ParFESpace()->GetMyRank();
  const int nprocs = pfunc_->ParFESpace()->GetNRanks();

  Array<int> &partitioning = *partitioning_;
  assert(partitioning.Size() == global_ne_);

  const unsigned int numStateVars = pfunc_->Size() / pfunc_->ParFESpace()->GetNDofs();
  assert(numStateVars == vars_.size());

  const int tag = 20;

  if (rank0_) {
    grvy_printf(ginfo, "[RestartSerial]: Reading %s for distribution\n", varName.c_str());

    Vector data_serial;
    data_serial.SetSize(numDof);

    read_variable_data_hdf5(file, varName, 0, data_serial.HostWrite());

    // assign solution owned by rank 0
    Array<int> lvdofs, gvdofs;
    Vector lnodes;
    int counter = 0;
    // int ndof_per_elem;

    for (int gelem = 0; gelem < global_ne_; gelem++) {
      if (partitioning[gelem] == 0) {
        // cull out subset of local vdof vector to use for this solution var
        pfunc_->ParFESpace()->GetElementVDofs(counter, lvdofs);
        int numDof_per_this_elem = lvdofs.Size() / numStateVars;
        int ldof_start_index = varOffset * numDof_per_this_elem;

        // cull out global vdofs - [subtle note]: we only use the
        // indexing corresponding to the first state vector reference in
        // gvdofs() since data_serial() holds the solution for a single state vector
        serial_fes_->GetElementVDofs(gelem, gvdofs);
        int gdof_start_index = 0;

        for (int i = 0; i < numDof_per_this_elem; i++)
          data[lvdofs[i + ldof_start_index]] = data_serial[gvdofs[i + gdof_start_index]];
        counter++;
      }
    }

    // pack remaining data and send to other processors
    for (int rank = 1; rank < nprocs; rank++) {
      std::vector<double> packedData;
      for (int gelem = 0; gelem < global_ne_; gelem++)
        if (partitioning[gelem] == rank) {
          serial_fes_->GetElementVDofs(gelem, gvdofs);
          int numDof_per_this_elem = gvdofs.Size() / numStateVars;
          for (int i = 0; i < numDof_per_this_elem; i++) packedData.push_back(data_serial[gvdofs[i]]);
        }
      int tag = 20;
      MPI_Send(packedData.data(), packedData.size(), MPI_DOUBLE, rank, tag, comm);
    }
  } else {  // <-- end rank 0
    // Figure out how much data to expect from rank 0
    Array<int> lvdofs;
    int numlDofs = 0;
    for (int lelem = 0; lelem < local_ne_; lelem++) {
      pfunc_->ParFESpace()->GetElementVDofs(lelem, lvdofs);
      int numDof_per_this_elem = lvdofs.Size() / numStateVars;
      numlDofs += numDof_per_this_elem;
    }

    grvy_printf(gdebug, "[%i]: local number of state vars to receive = %i (var=%s)\n", myrank, numlDofs,
                varName.c_str());

    if (pfunc_->ParFESpace()->IsDGSpace()) assert(numlDofs == pfunc_->ParFESpace()->GetNDofs());

    std::vector<double> packedData(numlDofs);

    // receive solution data from rank 0
    MPI_Recv(packedData.data(), numlDofs, MPI_DOUBLE, 0, tag, comm, MPI_STATUS_IGNORE);

    // NB: The unpack code below implicitly assumes that the global
    // element index is a monotonically increasing fcn of the local
    // element index.  This ensures that, in the unpack operation
    // below, elements on this rank are accessed in the same order
    // they are accessed by the pack operation on rank 0 above (where
    // we loop over the global element index and check what rank that
    // global element belongs to).  If this assumption were ever not
    // true, the unpack would jumble the data.

    // Unpack data received from rank 0
    int count = 0;
    for (int lelem = 0; lelem < local_ne_; lelem++) {
      pfunc_->ParFESpace()->GetElementVDofs(lelem, lvdofs);
      int numDof_per_this_elem = lvdofs.Size() / numStateVars;
      int ldof_start_index = varOffset * numDof_per_this_elem;

      for (int i = 0; i < numDof_per_this_elem; i++) data[lvdofs[i + ldof_start_index]] = packedData[i + count];
      count += numDof_per_this_elem;
    }
    assert(count == numlDofs);
  }
}

// convenience function to write HDF5 data
void write_variable_data_hdf5(hid_t group, string varName, hid_t dataspace, const double *data) {
  hid_t data_soln;
  herr_t status;
  assert(group >= 0);

  data_soln = H5Dcreate2(group, varName.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(data_soln >= 0);

  status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  assert(status >= 0);
  H5Dclose(data_soln);
}

void M2ulPhyS::readTable(const std::string &inputPath, TableInput &result) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();
  tpsP->getInput((inputPath + "/x_log").c_str(), result.xLogScale, false);
  tpsP->getInput((inputPath + "/f_log").c_str(), result.fLogScale, false);
  tpsP->getInput((inputPath + "/order").c_str(), result.order, 1);

  config.tableHost.push_back(DenseMatrix());

  int Ndata;
  Array<int> dims(2);
  bool success = false;
  int suc_int = 0;
  if (rank0_) {
    std::string filename;
    tpsP->getRequiredInput((inputPath + "/filename").c_str(), filename);
    success = h5ReadTable(filename, "table", config.tableHost.back(), dims);
    suc_int = (int)success;

    // TODO(kevin): extend for multi-column array?
    Ndata = dims[0];
  }
  // MPI_Bcast(&success, 1, MPI_CXX_BOOL, 0, TPSCommWorld);
  MPI_Bcast(&suc_int, 1, MPI_INT, 0, TPSCommWorld);
  success = (suc_int != 0);
  if (!success) exit(ERROR);

  int *d_dims = dims.GetData();
  MPI_Bcast(&Ndata, 1, MPI_INT, 0, TPSCommWorld);
  MPI_Bcast(d_dims, 2, MPI_INT, 0, TPSCommWorld);
  assert(dims[0] > 0);
  assert(dims[1] == 2);

  if (!rank0_) config.tableHost.back().SetSize(dims[0], dims[1]);
  double *d_table = config.tableHost.back().HostReadWrite();
  MPI_Bcast(d_table, dims[0] * dims[1], MPI_DOUBLE, 0, TPSCommWorld);

  result.Ndata = Ndata;
  result.xdata = config.tableHost.back().Read();
  result.fdata = config.tableHost.back().Read() + Ndata;

  return;
}

// ---------------------------------------------
// Routines for I/O data organizer helper class
// ---------------------------------------------

IOFamily::IOFamily(std::string desc, std::string grp, mfem::ParGridFunction *pf)
    : description_(desc), group_(grp), pfunc_(pf) {
  rank0_ = (pfunc_->ParFESpace()->GetMyRank() == 0);

  MPI_Comm comm = this->pfunc_->ParFESpace()->GetComm();

  // Set local and global number of elements
  local_ne_ = this->pfunc_->ParFESpace()->GetParMesh()->GetNE();
  MPI_Allreduce(&local_ne_, &global_ne_, 1, MPI_INT, MPI_SUM, comm);

  // Set local and global number of dofs (per scalar field)
  local_ndofs_ = pfunc_->ParFESpace()->GetNDofs();
  global_ndofs_ = pfunc_->ParFESpace()->GlobalTrueVSize() / pfunc_->ParFESpace()->GetVDim();
}

void IOFamily::serializeForWrite() {
  MPI_Comm comm = this->pfunc_->ParFESpace()->GetComm();

  const Array<int> &partitioning = *(this->partitioning_);
  assert(partitioning.Size() == global_ne_);

  const int *locToGlobElem = this->local_to_global_elem_;
  assert(locToGlobElem != NULL);

  ParGridFunction *pfunc = this->pfunc_;
  if (rank0_) {
    grvy_printf(ginfo, "Generating serialized restart file (group %s...)\n", this->group_.c_str());
    // copy my own data
    Array<int> lvdofs, gvdofs;
    Vector lsoln;
    for (int elem = 0; elem < local_ne_; elem++) {
      int gelem = locToGlobElem[elem];
      this->pfunc_->ParFESpace()->GetElementVDofs(elem, lvdofs);
      pfunc->GetSubVector(lvdofs, lsoln);
      this->serial_fes_->GetElementVDofs(gelem, gvdofs);
      this->serial_sol_->SetSubVector(gvdofs, lsoln);
    }

    // have rank 0 receive data from other tasks and copy its own
    for (int gelem = 0; gelem < global_ne_; gelem++) {
      int from_rank = partitioning[gelem];
      if (from_rank != 0) {
        this->serial_fes_->GetElementVDofs(gelem, gvdofs);
        lsoln.SetSize(gvdofs.Size());

        MPI_Recv(lsoln.HostReadWrite(), gvdofs.Size(), MPI_DOUBLE, from_rank, gelem, comm, MPI_STATUS_IGNORE);

        this->serial_sol_->SetSubVector(gvdofs, lsoln);
      }
    }

  } else {
    // have non-zero ranks send their data to rank 0
    Array<int> lvdofs;
    Vector lsoln;
    for (int elem = 0; elem < local_ne_; elem++) {
      int gelem = locToGlobElem[elem];
      //       assert(gelem > 0);
      this->pfunc_->ParFESpace()->GetElementVDofs(elem, lvdofs);
      pfunc->GetSubVector(lvdofs, lsoln);  // work for gpu build?

      // send to task 0
      MPI_Send(lsoln.HostReadWrite(), lsoln.Size(), MPI_DOUBLE, 0, gelem, comm);
    }
  }
}

void IOFamily::writePartitioned(hid_t file) {
  hsize_t dims[1];
  hid_t group = -1;
  hid_t dataspace = -1;

  // use all ranks to write data from this->.pfunc_
  dims[0] = pfunc_->ParFESpace()->GetNDofs();
  dataspace = H5Screate_simple(1, dims, NULL);
  assert(dataspace >= 0);
  group = H5Gcreate(file, group_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(group >= 0);

  // get pointer to raw data
  const double *data = pfunc_->HostRead();

  // save raw data
  for (auto var : vars_) {
    if (rank0_) grvy_printf(ginfo, "  --> Saving (%s)\n", var.varName_.c_str());
    write_variable_data_hdf5(group, var.varName_, dataspace, data + var.index_ * dims[0]);
  }

  if (group >= 0) H5Gclose(group);
  if (dataspace >= 0) H5Sclose(dataspace);
}

void IOFamily::writeSerial(hid_t file) {
  // Only get here if need to serialize
  serializeForWrite();

  // Only rank 0 writes data (which has just been collected into this->serial_sol_)
  if (rank0_) {
    hsize_t dims[1];
    hid_t group = -1;
    hid_t dataspace = -1;

    assert(serial_fes_ != NULL);
    dims[0] = serial_fes_->GetNDofs();

    dataspace = H5Screate_simple(1, dims, NULL);
    assert(dataspace >= 0);
    group = H5Gcreate(file, group_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(group >= 0);

    // get pointer to raw data
    assert(serial_sol_ != NULL);
    const double *data = serial_sol_->HostRead();

    // save raw data
    for (auto var : vars_) {
      grvy_printf(ginfo, "  --> Saving (%s)\n", var.varName_.c_str());
      write_variable_data_hdf5(group, var.varName_, dataspace, data + var.index_ * dims[0]);
    }

    if (group >= 0) H5Gclose(group);
    if (dataspace >= 0) H5Sclose(dataspace);
  }
}

void IOFamily::readPartitioned(hid_t file) {
  // Ensure that size of read matches expectations
  std::string varGroupName = group_ + "/" + vars_[0].varName_;
  const hsize_t numInSoln = get_variable_size_hdf5(file, varGroupName);
  assert((int)numInSoln == local_ndofs_);

  // get pointer to raw data
  double *data = pfunc_->HostWrite();

  // read from file into appropriate spot in data
  for (auto var : vars_) {
    if (var.inRestartFile_) {
      std::string h5Path = group_ + "/" + var.varName_;
      if (rank0_) grvy_printf(ginfo, "--> Reading h5 path = %s\n", h5Path.c_str());
      read_variable_data_hdf5(file, h5Path.c_str(), var.index_ * numInSoln, data);
    }
  }
}

void IOFamily::readSerial(hid_t file) {
  // Ensure that size of read matches expectations
  if (rank0_) {
    std::string varGroupName = group_ + "/" + vars_[0].varName_;
    const hsize_t numInSoln = get_variable_size_hdf5(file, varGroupName);
    assert((int)numInSoln == global_ndofs_);
  }

  // get pointer to raw data
  double *data = pfunc_->HostWrite();

  // read on rank 0 and distribute into data
  for (auto var : vars_) {
    if (var.inRestartFile_) {
      readDistributeSerializedVariable(file, var, global_ndofs_, data);
    }
  }
}

void IOFamily::readChangeOrder(hid_t file, int read_order) {
  if (allowsAuxRestart_) {
    // Set up "auxilliary" FiniteElementCollection, which matches original, but with different order
    assert(fec_ != NULL);
    aux_fec_ = fec_->Clone(read_order);

    // Set up "auxilliary" ParGridFunction, to read data into before order change
    const int nvars = vars_.size();
    aux_pfes_ = new ParFiniteElementSpace(pfunc_->ParFESpace()->GetParMesh(), aux_fec_, nvars, Ordering::byNODES);
    double *aux_U_data = new double[nvars * aux_pfes_->GetNDofs()];
    aux_pfunc_ = new ParGridFunction(aux_pfes_, aux_U_data);
    const int aux_dof = aux_pfes_->GetNDofs();

    // Ensure size of read matches expectations
    std::string varGroupName = group_ + "/" + vars_[0].varName_;
    const hsize_t numInSoln = get_variable_size_hdf5(file, varGroupName);
    assert((int)numInSoln == aux_dof);

    // get pointer to raw data
    double *data = aux_pfunc_->HostWrite();

    for (auto var : vars_) {
      if (var.inRestartFile_) {
        std::string h5Path = group_ + "/" + var.varName_;
        if (rank0_) grvy_printf(ginfo, "--> Reading h5 path = %s\n", h5Path.c_str());
        read_variable_data_hdf5(file, h5Path.c_str(), var.index_ * aux_dof, data);
      }
    }

    // Interpolate from "auxilliary" to new order
    if (rank0_) cout << "Interpolating from read_order = " << read_order << " to solution space" << endl;
    pfunc_->ProjectGridFunction(*aux_pfunc_);

#if 0
    // If interpolation was successful, the L2 norm of the
    // difference between the auxOrder and order versions should be
    // zero (well... to within round-off-induced error)
    VectorGridFunctionCoefficient lowOrderCoeff(aux_pfunc_);
    double err = pfunc_->ComputeL2Error(lowOrderCoeff);

    if (rank0_) cout << "|| interpolation error ||_2 = " << err << endl;
#endif

    // clean up aux data
    delete aux_pfunc_;
    delete[] aux_U_data;
    delete aux_pfes_;
    delete aux_fec_;
  } else {
    if (rank0_) cout << "Skipping family " << group_ << " because it doesn't allow changing order." << endl;
  }
}

// register a new IO family which maps to a ParGridFunction
void IODataOrganizer::registerIOFamily(std::string description, std::string group, ParGridFunction *pfunc,
                                       bool auxRestart, bool inRestartFile, FiniteElementCollection *fec) {
  IOFamily family(description, group, pfunc);
  family.allowsAuxRestart_ = auxRestart;
  family.inRestartFile_ = inRestartFile;
  family.fec_ = fec;
  if (family.allowsAuxRestart_) {
    assert(family.fec_ != NULL);
  }

  families_.push_back(family);
}

// register individual variables for IO family
void IODataOrganizer::registerIOVar(std::string group, std::string varName, int index, bool inRestartFile) {
  IOVar newvar{varName, index, inRestartFile};
  const int f = getIOFamilyIndex(group);
  assert(f >= 0);
  families_[f].vars_.push_back(newvar);
}

// return the index to IO family given a group name
int IODataOrganizer::getIOFamilyIndex(std::string group) const {
  for (size_t i = 0; i < families_.size(); i++)
    if (families_[i].group_ == group) return (i);

  return (-1);
}

void IODataOrganizer::initializeSerial(bool root, bool serial, Mesh *serial_mesh, int *locToGlob, Array<int> *part) {
  supports_serial_ = serial;

  // loop through families
  for (size_t n = 0; n < families_.size(); n++) {
    IOFamily &fam = families_[n];
    fam.serial_fes_ = NULL;
    fam.serial_sol_ = NULL;

    // If serial support is requested, need to initialize required data
    if (supports_serial_) {
      fam.local_to_global_elem_ = locToGlob;
      fam.partitioning_ = part;
      if (root) {
        const FiniteElementCollection *fec = fam.pfunc_->ParFESpace()->FEColl();
        int numVars = fam.pfunc_->Size() / fam.pfunc_->ParFESpace()->GetNDofs();

        fam.serial_fes_ = new FiniteElementSpace(serial_mesh, fec, numVars, Ordering::byNODES);
        fam.serial_sol_ = new GridFunction(fam.serial_fes_);
        //       cout<<"I/O organizer for group "<<fam.group_<<" initialized."<<endl;
      }
    }
  }
}

void IODataOrganizer::write(hid_t file, bool serial) {
  if (serial) assert(supports_serial_);

  // NOTE(trevilo): this loop envisions families that have different
  // number of mpi ranks, but rest of the code is not general enough
  // to support this configuration yet (or at a minimum there are no
  // tests of such a set up)
  int nprocs_max = 0;
  for (auto fam : families_) {
    int nprocs_tmp = fam.pfunc_->ParFESpace()->GetNRanks();
    if (nprocs_tmp > nprocs_max) {
      nprocs_max = nprocs_tmp;
    }
  }
  assert(nprocs_max > 0);

  // Do we need to serialize the data prior to the write?
  const bool require_serialization = (serial && nprocs_max > 1);

  // Loop over defined IO families to save desired output
  for (auto fam : families_) {
    const int rank = fam.pfunc_->ParFESpace()->GetMyRank();
    const bool rank0 = (rank == 0);

    if (rank0) {
      grvy_printf(ginfo, "\nCreating HDF5 group for defined IO families\n");
      grvy_printf(ginfo, "--> %s : %s\n", fam.group_.c_str(), fam.description_.c_str());
    }

    // Write handled by appropriate method from IOFamily
    if (require_serialization) {
      fam.writeSerial(file);
    } else {
      fam.writePartitioned(file);
    }
  }
}

void IODataOrganizer::read(hid_t file, bool serial, int read_order) {
  if (serial) assert(supports_serial_);

  // NOTE(trevilo): this loop envisions families that have different
  // number of mpi ranks, but rest of the code is not general enough
  // to support this configuration yet (or at a minimum there are no
  // tests of such a set up)
  int nprocs_max = 0;
  for (auto fam : families_) {
    int nprocs_tmp = fam.pfunc_->ParFESpace()->GetNRanks();
    if (nprocs_tmp > nprocs_max) {
      nprocs_max = nprocs_tmp;
    }
  }
  assert(nprocs_max > 0);

  // Loop over defined IO families to load desired input
  for (auto fam : families_) {
    if (fam.inRestartFile_) {  // read mode
      const int rank = fam.pfunc_->ParFESpace()->GetMyRank();
      const bool rank0 = (rank == 0);

      if (rank0) cout << "Reading in solution data from restart..." << endl;

      int fam_order;
      bool change_order = false;
      if (read_order >= 0) {
        assert(!fam.pfunc_->ParFESpace()->IsVariableOrder());
        fam_order = fam.pfunc_->ParFESpace()->FEColl()->GetOrder();
        change_order = (fam_order != read_order);
      }

      // Read handled by appropriate method from IOFamily
      if (change_order) {
        if (serial) {
          if (rank0) grvy_printf(gerror, "[ERROR]: Serial read does not support order change.\n");
          exit(ERROR);
        }
        fam.readChangeOrder(file, read_order);
      } else {
        if (serial && nprocs_max > 1) {
          fam.readSerial(file);
        } else {
          fam.readPartitioned(file);
        }
      }
    }
  }
}

IODataOrganizer::~IODataOrganizer() {
  for (size_t n = 0; n < families_.size(); n++) {
    IOFamily fam = families_[n];
    if (fam.serial_fes_ != NULL) delete fam.serial_fes_;
    if (fam.serial_sol_ != NULL) delete fam.serial_sol_;
  }
}
