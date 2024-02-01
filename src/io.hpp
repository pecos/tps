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
#ifndef IO_HPP_
#define IO_HPP_

#include <hdf5.h>

#include <string>
#include <vector>

#include "mpi_groups.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"

class IOVar {
 public:
  std::string varName_;  // solution variable
  int index_;            // variable index in the pargrid function
  bool inRestartFile_;   // Check if we want to read this variable
};

class IOFamily {
 protected:
  bool rank0_;

  void serializeForWrite();
  void readDistributeSerializedVariable(hid_t file, string varName, int numDof, int varOffset, double *data);

 public:
  std::string description_;       // family description
  std::string group_;             // HDF5 group name
  mfem::ParGridFunction *pfunc_;  // pargrid function owning the data for this IO family
  std::vector<IOVar> vars_;       // solution var info for this family

  bool allowsAuxRestart;

  // true if the family is to be found in restart files
  bool inRestartFile;

  // space and grid function used for serial read/write
  mfem::FiniteElementSpace *serial_fes = nullptr;
  mfem::GridFunction *serial_sol = nullptr;

  // additional data used for serial read/write (needed to properly
  // serialize fields in write or properly distribute data in read)
  int *local_to_global_elem_ = nullptr;       // maps from local elem index to global
  mfem::Array<int> *partitioning_ = nullptr;  // partitioning[i] mpi rank for element i (in global numbering)

  // Variables used for "auxilliary" read (i.e., restart from different order soln)
  mfem::FiniteElementCollection *fec_ = nullptr;  // collection used to instantiate pfunc_
  mfem::FiniteElementCollection *aux_fec_ = nullptr;
  mfem::ParFiniteElementSpace *aux_pfes_ = nullptr;
  mfem::ParGridFunction *aux_pfunc_ = nullptr;

  IOFamily(std::string desc, std::string grp, mfem::ParGridFunction *pf);

  void writePartitioned(hid_t file);
  void writeSerial(hid_t file);
  void readPartitioned(hid_t file);
  void readSerial(hid_t file);
  void readChangeOrder(hid_t file, int read_order);
};

class IODataOrganizer {
 protected:
  bool supports_serial_ = false;

 public:
  std::vector<IOFamily> families_;  // registered IO families

  void registerIOFamily(std::string description, std::string group, mfem::ParGridFunction *pfunc,
                        bool auxRestart = true, bool inRestartFile = true, mfem::FiniteElementCollection *fec = NULL);
  ~IODataOrganizer();

  void registerIOVar(std::string group, std::string varName, int index, bool inRestartFile = true);
  int getIOFamilyIndex(std::string group);

  void initializeSerial(bool root, bool serial, mfem::Mesh *serial_mesh, int *locToGlob, mfem::Array<int> *part);

  void write(hid_t file, bool serial);
  void read(hid_t file, bool serial, int read_order = -1);
};

void read_partitioned_soln_data(hid_t file, std::string varName, size_t index, double *data);
void write_soln_data(hid_t group, std::string varName, hid_t dataspace, const double *data);
void partitioning_file_hdf5(std::string mode, const RunConfiguration &config, MPI_Groups *groupsMPI, int nelemGlobal,
                            mfem::Array<int> &partitioning);
#endif  // IO_HPP_
