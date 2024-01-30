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
  // Write solution data defined by IO families
  // -------------------------------------------------------------------
  hsize_t dims[1];

  //-------------------------------------------------------
  // Loop over defined IO families to save desired output
  //-------------------------------------------------------
  for (size_t n = 0; n < ioData.families_.size(); n++) {
    IOFamily &fam = ioData.families_[n];

    if (serialized_write && (nprocs_ > 1) && (rank0_)) {
      assert((locToGlobElem != NULL) && (partitioning_ != NULL));
      assert(fam.serial_fes != NULL);
      dims[0] = fam.serial_fes->GetNDofs();
    } else {
      dims[0] = fam.pfunc_->ParFESpace()->GetNDofs();
    }
    // define groups based on defined IO families
    if (rank0_) {
      grvy_printf(ginfo, "\nCreating HDF5 group for defined IO families\n");
      grvy_printf(ginfo, "--> %s : %s\n", fam.group_.c_str(), fam.description_.c_str());
    }

    hid_t group = -1;
    hid_t dataspace = -1;

    if (rank0_ || !serialized_write) {
      dataspace = H5Screate_simple(1, dims, NULL);
      assert(dataspace >= 0);
      group = H5Gcreate(file, fam.group_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(group >= 0);
    }

    // get pointer to raw data
    double *data = fam.pfunc_->HostReadWrite();
    // special case if writing a serial restart
    if (serialized_write && (nprocs_ > 1)) {
      serialize_soln_for_write(fam, groupsMPI, mesh->GetNE(), nelemGlobal_, locToGlobElem, partitioning_);
      if (rank0_) data = fam.serial_sol->HostReadWrite();
    }

    // get defined variables for this IO family
    vector<IOVar> vars = ioData.vars_[fam.group_];

    // save raw data
    if (rank0_ || !serialized_write) {
      for (auto var : vars) write_soln_data(group, var.varName_, dataspace, data + var.index_ * dims[0], rank0_);
    }

    if (group >= 0) H5Gclose(group);
    if (dataspace >= 0) H5Sclose(dataspace);
  }
}

void M2ulPhyS::read_restart_files_hdf5(hid_t file, bool serialized_read) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();

  int read_order;

  // Variables used if (and only if) restarting from different order
  FiniteElementCollection *aux_fec = NULL;
  ParFiniteElementSpace *aux_vfes = NULL;
  ParGridFunction *aux_U = NULL;
  double *aux_U_data = NULL;
  int auxOrder = -1;

  // normal restarts have each process read their own portion of the
  // solution; a serial restart only reads on rank 0 and distributes to
  // the remaining processes

  if (rank0_ || !serialized_read) {
    h5_read_attribute(file, "iteration", iter);
    h5_read_attribute(file, "time", time);
    h5_read_attribute(file, "dt", dt);
    h5_read_attribute(file, "order", read_order);
    if (average->ComputeMean() && config.GetRestartMean()) {
      int samplesMean, intervals;
      h5_read_attribute(file, "samplesMean", samplesMean);
      h5_read_attribute(file, "samplesInterval", intervals);
      average->SetSamplesMean(samplesMean);
      average->SetSamplesInterval(intervals);
    }
  }

  if (serialized_read) {
    MPI_Bcast(&iter, 1, MPI_INT, 0, TPSCommWorld);
    MPI_Bcast(&time, 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, TPSCommWorld);
    MPI_Bcast(&read_order, 1, MPI_INT, 0, TPSCommWorld);
    if (average->ComputeMean() && config.GetRestartMean()) {
      int sampMean = average->GetSamplesMean();
      int intervals = average->GetSamplesInterval();
      MPI_Bcast(&sampMean, 1, MPI_INT, 0, TPSCommWorld);
      MPI_Bcast(&intervals, 1, MPI_INT, 0, TPSCommWorld);
    }
  }

  if (loadFromAuxSol) {
    assert(!serialized_read);  // koomie, check cis
    auxOrder = read_order;

    if (basisType == 0)
      aux_fec = new DG_FECollection(auxOrder, dim, BasisType::GaussLegendre);
    else if (basisType == 1)
      aux_fec = new DG_FECollection(auxOrder, dim, BasisType::GaussLobatto);

    aux_vfes = new ParFiniteElementSpace(mesh, aux_fec, num_equation, Ordering::byNODES);

    aux_U_data = new double[num_equation * aux_vfes->GetNDofs()];
    aux_U = new ParGridFunction(aux_vfes, aux_U_data);
  } else {
    assert(read_order == order);
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
  // Read/write solution data defined by IO families
  // -------------------------------------------------------------------
  hsize_t numInSoln = 0;
  hid_t data_soln;

  //-------------------------------------------------------
  // Loop over defined IO families to save desired output
  //-------------------------------------------------------
  for (size_t n = 0; n < ioData.families_.size(); n++) {
    IOFamily &fam = ioData.families_[n];
    if (fam.inRestartFile) {  // read mode
      if (rank0_) cout << "Reading in solutiond data from restart..." << endl;

      // verify Dofs match expectations with current mesh
      if (rank0_ || !serialized_read) {
        hid_t dataspace;

        vector<IOVar> vars = ioData.vars_[fam.group_];
        string varGroupName = fam.group_;
        varGroupName.append("/");
        varGroupName.append(vars[0].varName_);

        data_soln = H5Dopen2(file, varGroupName.c_str(), H5P_DEFAULT);
        assert(data_soln >= 0);
        dataspace = H5Dget_space(data_soln);
        numInSoln = H5Sget_simple_extent_npoints(dataspace);
        H5Dclose(data_soln);
      }

      int dof = fam.pfunc_->ParFESpace()->GetNDofs();
      if (serialized_read) {
        int dof_global;
        MPI_Reduce(&dof, &dof_global, 1, MPI_INT, MPI_SUM, 0, TPSCommWorld);
        dof = dof_global;
      }

      if (loadFromAuxSol && fam.allowsAuxRestart) dof = aux_vfes->GetNDofs();

      if (rank0_ || !serialized_read) assert((int)numInSoln == dof);

      // get pointer to raw data
      double *data = fam.pfunc_->HostReadWrite();
      // special case if starting from aux soln
      if (loadFromAuxSol && fam.allowsAuxRestart) {
        data = aux_U->HostReadWrite();

        // ks note: would need to add additional logic to handle read of
        // multiple pargridfunctions when changing the solution order
        assert(ioData.families_.size() == 1);
      }

      vector<IOVar> vars = ioData.vars_[fam.group_];
      for (auto var : vars) {
        if (var.inRestartFile_) {
          std::string h5Path = fam.group_ + "/" + var.varName_;
          if (rank0_) grvy_printf(ginfo, "--> Reading h5 path = %s\n", h5Path.c_str());
          if (!serialized_read) {
            read_partitioned_soln_data(file, h5Path.c_str(), var.index_ * numInSoln, data);
          } else {
            assert(partitioning_ != NULL);
            assert(serialized_read);
            read_serialized_soln_data(file, h5Path.c_str(), dof, var.index_, data, fam, groupsMPI, partitioning_,
                                      nelemGlobal_);
          }
        }
      }
    }
  }

  if (loadFromAuxSol) {
    if (rank0_) cout << "Interpolating from auxOrder = " << auxOrder << " to order = " << order << endl;

    // Interpolate from aux to new order
    U->ProjectGridFunction(*aux_U);

    // If interpolation was successful, the L2 norm of the
    // difference between the auxOrder and order versions should be
    // zero (well... to within round-off-induced error)
    VectorGridFunctionCoefficient lowOrderCoeff(aux_U);
    double err = U->ComputeL2Error(lowOrderCoeff);

    if (rank0_) cout << "|| interpolation error ||_2 = " << err << endl;

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

    // clean up aux data
    delete aux_U;
    delete aux_U_data;
    delete aux_vfes;
    delete aux_fec;
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

void partitioning_file_hdf5(std::string mode, const RunConfiguration &config, MPI_Groups *groupsMPI, int nelemGlobal,
                            Array<int> &partitioning) {
  MPI_Comm TPSCommWorld = groupsMPI->getTPSCommWorld();
  const bool rank0 = groupsMPI->isWorldRoot();
  const int nprocs = groupsMPI->getTPSWorldSize();

  grvy_timer_begin(__func__);

  // only rank 0 writes partitioning file
  if (!rank0 && (mode == "write")) return;

  // hid_t file, dataspace, data_soln;
  hid_t file = -1, dataspace;
  herr_t status;
  std::string fileName = config.GetPartitionBaseName();
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

void serialize_soln_for_write(IOFamily &fam, MPI_Groups *groupsMPI, int local_ne, int global_ne,
                              const int *locToGlobElem, const Array<int> &partitioning) {
  MPI_Comm TPSCommWorld = groupsMPI->getTPSCommWorld();
  const bool rank0 = groupsMPI->isWorldRoot();

  // Ensure consistency
  int global_ne_check;
  MPI_Reduce(&local_ne, &global_ne_check, 1, MPI_INT, MPI_SUM, 0, TPSCommWorld);
  if (rank0) {
    assert(global_ne_check == global_ne);
    assert(partitioning.Size() == global_ne);
  }
  assert(locToGlobElem != NULL);

  ParGridFunction *pfunc = fam.pfunc_;
  if (rank0) {
    grvy_printf(ginfo, "Generating serialized restart file (group %s...)\n", fam.group_.c_str());
    // copy my own data
    Array<int> lvdofs, gvdofs;
    Vector lsoln;
    for (int elem = 0; elem < local_ne; elem++) {
      int gelem = locToGlobElem[elem];
      fam.pfunc_->ParFESpace()->GetElementVDofs(elem, lvdofs);
      pfunc->GetSubVector(lvdofs, lsoln);
      fam.serial_fes->GetElementVDofs(gelem, gvdofs);
      fam.serial_sol->SetSubVector(gvdofs, lsoln);
    }

    // have rank 0 receive data from other tasks and copy its own
    for (int gelem = 0; gelem < global_ne; gelem++) {
      int from_rank = partitioning[gelem];
      if (from_rank != 0) {
        fam.serial_fes->GetElementVDofs(gelem, gvdofs);
        lsoln.SetSize(gvdofs.Size());

        MPI_Recv(lsoln.HostReadWrite(), gvdofs.Size(), MPI_DOUBLE, from_rank, gelem, TPSCommWorld, MPI_STATUS_IGNORE);

        fam.serial_sol->SetSubVector(gvdofs, lsoln);
      }
    }

  } else {
    // have non-zero ranks send their data to rank 0
    Array<int> lvdofs;
    Vector lsoln;
    for (int elem = 0; elem < local_ne; elem++) {
      int gelem = locToGlobElem[elem];
      //       assert(gelem > 0);
      fam.pfunc_->ParFESpace()->GetElementVDofs(elem, lvdofs);
      pfunc->GetSubVector(lvdofs, lsoln);  // work for gpu build?

      // send to task 0
      MPI_Send(lsoln.HostReadWrite(), lsoln.Size(), MPI_DOUBLE, 0, gelem, TPSCommWorld);
    }
  }
}  // end function: serialize_soln_for_write()

// convenience function to read solution data for parallel restarts
void read_partitioned_soln_data(hid_t file, string varName, size_t index, double *data) {
  hid_t data_soln;
  herr_t status;

  data_soln = H5Dopen2(file, varName.c_str(), H5P_DEFAULT);
  assert(data_soln >= 0);
  status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[index]);
  assert(status >= 0);
  H5Dclose(data_soln);
}

// convenience function to read and distribute solution data for serialized restarts
void read_serialized_soln_data(hid_t file, string varName, int numDof, int varOffset, double *data, IOFamily &fam,
                               MPI_Groups *groupsMPI, Array<int> partitioning, int nelemGlobal) {
  MPI_Comm TPSCommWorld = groupsMPI->getTPSCommWorld();
  const bool rank0 = groupsMPI->isWorldRoot();
  const int nprocs = groupsMPI->getTPSWorldSize();
  const int rank = groupsMPI->getTPSWorldRank();

  hid_t data_soln;
  herr_t status;
  int numStateVars;

  numStateVars = fam.pfunc_->Size() / fam.pfunc_->ParFESpace()->GetNDofs();

  const int tag = 20;

  if (rank0) {
    grvy_printf(ginfo, "[RestartSerial]: Reading %s for distribution\n", varName.c_str());

    Vector data_serial;
    data_serial.SetSize(numDof);
    data_soln = H5Dopen2(file, varName.c_str(), H5P_DEFAULT);
    assert(data_soln >= 0);
    status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_serial.HostReadWrite());
    assert(status >= 0);
    H5Dclose(data_soln);

    // assign solution owned by rank 0
    assert(partitioning != NULL);

    Array<int> lvdofs, gvdofs;
    Vector lnodes;
    int counter = 0;
    // int ndof_per_elem;

    for (int gelem = 0; gelem < nelemGlobal; gelem++) {
      if (partitioning[gelem] == 0) {
        // cull out subset of local vdof vector to use for this solution var
        fam.pfunc_->ParFESpace()->GetElementVDofs(counter, lvdofs);
        int numDof_per_this_elem = lvdofs.Size() / numStateVars;
        int ldof_start_index = varOffset * numDof_per_this_elem;

        // cull out global vdofs - [subtle note]: we only use the
        // indexing corresponding to the first state vector reference in
        // gvdofs() since data_serial() holds the solution for a single state vector
        fam.serial_fes->GetElementVDofs(gelem, gvdofs);
        int gdof_start_index = 0;

        for (int i = 0; i < numDof_per_this_elem; i++)
          data[lvdofs[i + ldof_start_index]] = data_serial[gvdofs[i + gdof_start_index]];
        counter++;
      }
    }

    // pack remaining data and send to other processors
    for (int rank = 1; rank < nprocs; rank++) {
      std::vector<double> packedData;
      for (int gelem = 0; gelem < nelemGlobal; gelem++)
        if (partitioning[gelem] == rank) {
          fam.serial_fes->GetElementVDofs(gelem, gvdofs);
          int numDof_per_this_elem = gvdofs.Size() / numStateVars;
          for (int i = 0; i < numDof_per_this_elem; i++) packedData.push_back(data_serial[gvdofs[i]]);
        }
      int tag = 20;
      MPI_Send(packedData.data(), packedData.size(), MPI_DOUBLE, rank, tag, TPSCommWorld);
    }
  } else {  // <-- end rank 0
    int numlDofs = fam.pfunc_->ParFESpace()->GetNDofs();
    grvy_printf(gdebug, "[%i]: local number of state vars to receive = %i (var=%s)\n", rank, numlDofs, varName.c_str());

    std::vector<double> packedData(numlDofs);

    // receive solution data from rank 0
    MPI_Recv(packedData.data(), numlDofs, MPI_DOUBLE, 0, tag, TPSCommWorld, MPI_STATUS_IGNORE);

    // update local state vector
    for (int i = 0; i < numlDofs; i++) data[i + varOffset * numlDofs] = packedData[i];
  }
}

// convenience function to write HDF5 data
void write_soln_data(hid_t group, string varName, hid_t dataspace, double *data, bool rank0) {
  hid_t data_soln;
  herr_t status;
  assert(group >= 0);

  if (rank0) grvy_printf(ginfo, "  --> Saving (%s)\n", varName.c_str());

  data_soln = H5Dcreate2(group, varName.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(data_soln >= 0);

  status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  assert(status >= 0);
  H5Dclose(data_soln);

  return;
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

// register a new IO family which maps to a ParGridFunction
void IODataOrganizer::registerIOFamily(std::string description, std::string group, ParGridFunction *pfunc,
                                       bool auxRestart, bool _inRestartFile) {
  IOFamily family{description, group, pfunc};
  std::vector<IOVar> vars;
  family.allowsAuxRestart = auxRestart;
  family.inRestartFile = _inRestartFile;

  families_.push_back(family);
  vars_[group] = vars;

  return;
}

// register individual variables for IO family
void IODataOrganizer::registerIOVar(std::string group, std::string varName, int index, bool inRestartFile) {
  IOVar newvar{varName, index, inRestartFile};
  vars_[group].push_back(newvar);

  return;
}

// return the index to IO family given a group name
int IODataOrganizer::getIOFamilyIndex(std::string group) {
  for (size_t i = 0; i < families_.size(); i++)
    if (families_[i].group_ == group) return (i);

  return (-1);
}

void IODataOrganizer::initializeSerial(bool root, bool serial, Mesh *serial_mesh) {
  // loop through families
  for (size_t n = 0; n < families_.size(); n++) {
    IOFamily &fam = families_[n];
    fam.serial_fes = NULL;
    fam.serial_sol = NULL;
    if (root && serial) {
      const FiniteElementCollection *fec = fam.pfunc_->ParFESpace()->FEColl();
      int numVars = fam.pfunc_->Size() / fam.pfunc_->ParFESpace()->GetNDofs();

      fam.serial_fes = new FiniteElementSpace(serial_mesh, fec, numVars, Ordering::byNODES);
      fam.serial_sol = new GridFunction(fam.serial_fes);
      //       cout<<"I/O organizer for group "<<fam.group_<<" initialized."<<endl;
    }
  }
}

IODataOrganizer::~IODataOrganizer() {
  for (size_t n = 0; n < families_.size(); n++) {
    IOFamily fam = families_[n];
    if (fam.serial_fes != NULL) delete fam.serial_fes;
    if (fam.serial_sol != NULL) delete fam.serial_sol;
  }
}
