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
/** @file
 * @brief Contains utilities for abstracting field IO tasks
 *
 * The primary class is IODataOrganizer.  For example usage, see M2ulPhyS.
 */

#include <hdf5.h>

#include <string>
#include <vector>

#include "mpi_groups.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"

class IODataOrganizer;

/**
 * @brief Stores info about variables to read/write.
 *
 * This info is used in IOFamily and IODataOrganizer and is not
 * intended to be needed separately.
 */
class IOVar {
 public:
  /** Variable name, which combines with family name to set path in hdf5 file */
  std::string varName_;

  /** Variable index within variables stored in corresponding ParGridFunction */
  int index_;  // variable index in the pargrid function

  /** Whether variable is to be read
   *
   * This allows variables in the state to be skipped, e.g., to enable
   * restarts from models that do not have all the same state variables.
   */
  bool inRestartFile_;
};

/**
 * @brief Stores info about a "family" of variables to read/write.
 *
 * Here a "family" refers to data stored in a single
 * mfem::ParGridFunction object.  Thus, the family has a single mesh
 * and parallel decomposition.  For the moment, the underlying mesh
 * and decomposition is assumed fixed after initialization.
 *
 * This class is intended to be used through IODataOrganizer, which
 * can track multiple IOFamily objects.
 */
class IOFamily {
  friend class IODataOrganizer;

 protected:
  /** Indicates whether current mpi rank is rank 0*/
  bool rank0_ = false;

  /** Local number of elements in this mpi rank */
  int local_ne_ = -1;

  /** Global number of elements in whole mesh */
  int global_ne_ = -1;

  /** Local number of degrees of freedom on this mpi rank, as returned by ParGridFunction::GetNDofs */
  int local_ndofs_ = -1;

  /** Global number of degrees of freedom */
  int global_ndofs_ = -1;

  /** Description string */
  std::string description_;

  /** Group name, used to set corresponding group in hdf5 restart files */
  std::string group_;

  /** mfem::ParGridFunction owning the data to be written or read into */
  mfem::ParGridFunction *pfunc_ = nullptr;

  /** Variables that make up the family */
  std::vector<IOVar> vars_;

  /** Whether this family allows initialization (read) from other order elements */
  bool allowsAuxRestart;

  /** true if the family is to be found in restart files */
  bool inRestartFile;

  /** mfem::FiniteElementSpace for serial mesh (used if serial read and/or write requested) */
  mfem::FiniteElementSpace *serial_fes = nullptr;

  /** mfem::GridFunction on serial mesh (used if serial read and/or write requested) */
  mfem::GridFunction *serial_sol = nullptr;

  /** Map from local to global element numbering (used for serial read/write) */
  int *local_to_global_elem_ = nullptr;

  /** Partition array; partitioning[i] = mpi rank that owns the ith element (global numbering) */
  mfem::Array<int> *partitioning_ = nullptr;

  /** mfem::FiniteElementCollection used by pfunc_ (used for variable order read) */
  mfem::FiniteElementCollection *fec_ = nullptr;

  /** mfem::FiniteElementCollection with read order (used for variable order read) */
  mfem::FiniteElementCollection *aux_fec_ = nullptr;

  /** mfem::FiniteElementCollection with read order elements (used for variable order read) */
  mfem::ParFiniteElementSpace *aux_pfes_ = nullptr;

  /** mfem::ParGridFunction with read order elements (used for variable order read) */
  mfem::ParGridFunction *aux_pfunc_ = nullptr;

  /**
   * @brief Prepare for serial write by collecting data onto rank 0
   *
   * Each rank sends its data in pfunc_ to rank 0 and stores the
   * results in serial_sol
   */
  void serializeForWrite();

  /**
   * @brief Read and distribute a single variable in a serialized restart file
   *
   * Rank 0 reads the data and sends to appropriate rank and location in pfunc_
   * @param file Open HDF5 file handle (e.g., as returned by H5Fcreate or H5Fopen)
   * @param var IOVar for the variable being read
   */
  void readDistributeSerializedVariable(hid_t file, const IOVar &var, int numDof, double *data);

 public:
  /**
   * @brief Constructor, must provide description, group name, and ParGridFunction
   *
   * @param desc Description string
   * @param grp Group name
   * @param pf Pointer to mfem::ParGridFunction that owns the data
   */
  IOFamily(std::string desc, std::string grp, mfem::ParGridFunction *pf);

  /** @brief "Partitioned" write (each mpi rank writes its local part of the ParGridFunction to a separate file) */
  void writePartitioned(hid_t file);

  /** @brief "Serial" write (all data communicated to rank 0, which writes everything to a single file) */
  void writeSerial(hid_t file);

  /** @brief "Partitioned" read (inverse of writePartitioned) */
  void readPartitioned(hid_t file);

  /** @brief "Serial" read (inverse of readPartitioned) */
  void readSerial(hid_t file);

  /**
   * @brief "Partitioned" read of a different order solution
   *
   * which is then interpolated onto the desired space, as defined by pfunc_
   * @param file Open HDF5 file handle (e.g., as returned by H5Fcreate or H5Fopen)
   * @param read_order Polynomial order of function in HDF5 file
   */
  void readChangeOrder(hid_t file, int read_order);
};

/**
 * @brief Provides IO interface for multiple "families"
 *
 * This class is used to read & write the solution data contained in
 * the mfem::ParGridFunction object in each IOFamily that is
 * "registered".  Once families and variables are registered, the
 * underlying data can be read or written with a single call.  Since
 * it is expected that the calling class will have some class-specific
 * metadata to read or write in addition to the solution data, this
 * class does not handle opening or creating the hdf5 file or reading
 * or writing solution metadata.  These tasks must be handled outside
 * the IODataOrganizer class.  For example usage, see
 * M2ulPhyS::initSolutionAndVisualizationVectors and
 * M2ulPhyS::write_restart_files_hdf5.
 */
class IODataOrganizer {
 protected:
  /** Whether serial read/write is supported.  To enable serial support, must call initializeSerial */
  bool supports_serial_ = false;

  /** Registered IO families */
  std::vector<IOFamily> families_;

  /** Get the index of the IOFamily with group name group */
  int getIOFamilyIndex(std::string group) const;

 public:
  /**
   * @brief Register an IOFamily
   *
   * Constructs the family using the input arguments.
   *
   * @param description Description string for the family
   * @param group Group name for the family
   * @param pfunc mfem::ParGridFunction for the family
   * @param auxRestart Allow initialization from different order read (optional, defaults to true)
   * @param inRestartFile If false, skip family on read (optional, defaults to true)
   * @param fec FiniteElementCollection for grid function (optional, only used for different order read)
   */
  void registerIOFamily(std::string description, std::string group, mfem::ParGridFunction *pfunc,
                        bool auxRestart = true, bool inRestartFile = true, mfem::FiniteElementCollection *fec = NULL);
  /** Destructor */
  ~IODataOrganizer();

  /**
   * @brief Register an IOVar as part of the corresponding family
   *
   * @param group Group name for the family to add the variable to
   * @param varName Name of the variable
   * @param index Variable index within the ParGridFunction of the family
   * @param inRestartFile If false, skip variable on read (optional, defaults to true)
   */
  void registerIOVar(std::string group, std::string varName, int index, bool inRestartFile = true);

  /**
   * @brief Initialize data supporting serial read/write.
   *
   * @param root True for mpi rank = 0
   * @param serial True if we want seral read and/or write
   * @param serial_mesh Pointer to serial mesh object
   * @param locToGlob Map from local element index to global element index
   * @param part Map from global element index to mpi rank owning that element
   */
  void initializeSerial(bool root, bool serial, mfem::Mesh *serial_mesh, int *locToGlob, mfem::Array<int> *part);

  /**
   * @brief Write data from all families
   *
   * All families written to the same HDF5 file, which must be already
   * open.  If serial write requested, a data is written to a single
   * HDF5 file by rank 0.  Otherwise, all ranks write to separate
   * files.
   *
   * @param file HDF5 file handle (e.g., as returned by H5Fcreate or H5Fopen)
   * @param serial True for serial write
   */
  void write(hid_t file, bool serial);

  /**
   * @brief Read data for all families
   *
   * All families are read from the same HDF5 file, which must be
   * already open.  If serial read is requested, the data is read from
   * a single HDF5 file by rank 0.  Otherwise, all ranks read from
   * separate files.  For variable order (i.e., data in file had
   * different order than desired), the order of the file being read
   * must be supplied.
   *
   * @param file HDF5 file handle (e.g., as returned by H5Fopen)
   * @param serial True for serial read
   * @param read_order Polynomial order for data in file (optional, only required for variable order read)
   */
  void read(hid_t file, bool serial, int read_order = -1);
};

/**
 * @brief Get number of points to be read
 *
 * This convenience function returns the number of points to be read
 * for a given name.  This is used frequently to check for consistency
 * between the size of the read and the size of the container being
 * read into.
 *
 * @param file Open HDF5 file
 * @param name Name to query (corresponds to "family group/variable name")
 * @returns Size of corresponding data in file
 */
hsize_t get_variable_size_hdf5(hid_t file, std::string name);

/**
 * @brief Read data from hdf5 file
 *
 * Reads data from the hdf5 file into the data buffer, starting at
 * location index (i.e., fill starting at data[index])
 *
 * @param file Open HDF5 file
 * @param varName Name of object to read  (corresponds to "family group/variable name")
 * @param index Starting location within data buffer
 * @param data Data buffer
 */
void read_variable_data_hdf5(hid_t file, std::string varName, size_t index, double *data);

/**
 * @brief Write data to an hdf5 file
 *
 * @param group Group within the hdf5 file to write to (e.g., as returned by H5Gcreate)
 * @param varName Name of variable to write
 * @param dataspace HDF5 dataspace (e.g., as returned by H5Screate_simple)
 * @param data Buffer with data to write
 */
void write_variable_data_hdf5(hid_t group, std::string varName, hid_t dataspace, const double *data);

/**
 * @brief Read/write partitioning information
 *
 * For normal restarts, this is used to write/read the partitioning
 * information to ensure that it remains consistent across the
 * restart.  For serial restarts, it is also used to properly
 * distribute the incoming fields.
 *
 * @todo Refactor this function to make it more generic and fit better
 * into the IODataOrganizer paradigm.
 */
void partitioning_file_hdf5(std::string mode, const RunConfiguration &config, MPI_Groups *groupsMPI, int nelemGlobal,
                            mfem::Array<int> &partitioning);
#endif  // IO_HPP_
