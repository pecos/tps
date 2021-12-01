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
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <assert.h>
#include <hdf5.h>

#include <mfem.hpp>
#include <string>

// Misc. utilities
bool file_exists(const std::string &name);
std::string systemCmd(const char *cmd);

// HDF5 convenience utilities
inline hid_t h5_getType(int) { return (H5T_NATIVE_INT); }
inline hid_t h5_getType(double) { return (H5T_NATIVE_DOUBLE); }

template <typename T>
void h5_save_attribute(hid_t dest, std::string attribute, T value) {
  hid_t attr, status;
  hid_t attrType = h5_getType(value);
  hid_t dataspaceId = H5Screate(H5S_SCALAR);

  attr = H5Acreate(dest, attribute.c_str(), attrType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
  assert(attr >= 0);
  status = H5Awrite(attr, attrType, &value);
  assert(status >= 0);
  H5Aclose(attr);
  H5Sclose(dataspaceId);
}

template <typename T>
void h5_read_attribute(hid_t source, std::string attribute, T &value) {
  herr_t status;
  hid_t attr;
  hid_t attrType = h5_getType(value);
  //  T temp;

  attr = H5Aopen_name(source, attribute.c_str());
  status = H5Aread(attr, attrType, &value);
  assert(status >= 0);
  H5Aclose(attr);
}

// MFEM extensions

/** Project discontinous function onto FE space
 *
 * Project a discontinuous vector coefficient into the finite element
 * space provided by gf and returns the maximum attribute of the
 * elements containing each dof in dof_attr on the local mpi rank.
 * This is a helper function for GlobalProjectDiscCoefficient.
 *
 * This function is based on
 * mfem::GridFunction::ProjectDiscCoefficient but has fixes s.t. it
 * will work with Nedelec elements.
 */
void LocalProjectDiscCoefficient(mfem::GridFunction &gf, mfem::VectorCoefficient &coeff, mfem::Array<int> &dof_attr);

/** Project discontinous function onto FE space
 *
 * Project a discontinuous vector coefficient into the finite element
 * space provided by gf.  The values in shared dofs are determined
 * from the element with maximal attribute.
 *
 * This function is based on
 * mfem::ParGridFunction::ProjectDiscCoefficient but calls
 * LocalProjectDiscCoefficient.
 */
void GlobalProjectDiscCoefficient(mfem::ParGridFunction &gf, mfem::VectorCoefficient &coeff);

#endif  // UTILS_HPP_
