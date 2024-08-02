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
#ifndef UTILS_HPP_
#define UTILS_HPP_

/** @file
 * @brief A place for miscellaneous utility functions.
 */

#include <assert.h>
#include <hdf5.h>

#include <string>

#include "dataStructures.hpp"
#include "tps_mfem_wrap.hpp"

// application exit codes
enum ExitCodes { NORMAL = 0, ERROR = 1, JOB_RESTART = 10, EARLY_EXIT = 11 };

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

// A simple HDF5 routine to read two-dimensional array into DenseMatrix.
// Return result as boolean.
bool h5ReadTable(const std::string &fileName, const std::string &datasetName, mfem::DenseMatrix &output,
                 mfem::Array<int> &shape);

/** Read multi-column table
 *
 * Read multi-column table data from an hdf5 file into a set of input
 * TableInput structs.  The read happens only on rank 0 and the data
 * are broadcast to all ranks.  The length of the input std::vector of
 * TableInput must match the number of columsn read.
 */
bool h5ReadBcastMultiColumnTable(const std::string &fileName, const std::string &datasetName, MPI_Comm TPSCommWorld,
                                 mfem::DenseMatrix &output, std::vector<TableInput> &tables);

void readTable(MPI_Comm TPSCommWorld, std::string filename, bool xLogScale, bool fLogScale, int order,
               TableInput &result);

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

/** @brief Evaluate the distance function
 *
 *  Compute minimum distance from each node to a no-slip wall.
 *
 *  @param mesh A serial mesh
 *  @param wall_patches Array of patch indices corresponding to walls
 *  @param coords Grid function containing mesh coordinates
 *  @param distance Grid function to store the distance
 *
 *  No allocation or resizing is done.  The incoming
 *  GridFunction objects must already be set up and sized correctly.
 *
 *  This function is not guarenteed to work for nonlinear elements.
 *  See comments in source about why.
 */
void evaluateDistanceSerial(mfem::Mesh &mesh, const mfem::Array<int> &wall_patches, const mfem::GridFunction &coords,
                            mfem::GridFunction &distance);

void multConstScalar(double A, Vector B, Vector *C);
void multConstScalarInv(double A, Vector B, Vector *C);
void multConstVector(double A, Vector B, Vector *C);
void multConstScalarIP(double A, Vector *C);
void multConstScalarInvIP(double A, Vector *C);
void multConstVectorIP(double A, Vector *C);
void multScalarScalar(Vector A, Vector B, Vector *C);
void multScalarScalarInv(Vector A, Vector B, Vector *C);
void multScalarVector(Vector A, Vector B, Vector *C, int dim = 3);
void multScalarInvVector(Vector A, Vector B, Vector *C, int dim = 3);
void multScalarInvVectorIP(Vector A, Vector *C, int dim = 3);
void multVectorVector(Vector A, Vector B, Vector *C1, Vector *C2, Vector *C3, int dim = 3);
void dotVector(Vector A, Vector B, Vector *C, int dim = 3);
void multScalarScalarIP(Vector A, Vector *C);
void multScalarInvScalarIP(Vector A, Vector *C);
void multScalarVectorIP(Vector A, Vector *C, int dim = 3);
void setScalarFromVector(Vector A, int ind, Vector *C);
void setVectorFromScalar(Vector A, int ind, Vector *C);

/// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^2\f$.
void ComputeCurl2D(const ParGridFunction &u, ParGridFunction &cu, bool assume_scalar = false);

void ComputeCurlAxi(const ParGridFunction &u, ParGridFunction &cu, bool assume_scalar = false);

/// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^3\f$.
void ComputeCurl3D(const ParGridFunction &u, ParGridFunction &cu);

void vectorGrad3D(ParGridFunction &uSub, ParGridFunction &u, ParGridFunction &gu, ParGridFunction &gv,
                  ParGridFunction &gw);
void scalarGrad3D(ParGridFunction &u, ParGridFunction &gu);
// void vectorGrad3DV(FiniteElementSpace *fes, Vector u, Vector *gu, Vector *gv, Vector *gw);
void scalarGrad3DV(FiniteElementSpace *fes, FiniteElementSpace *vfes, Vector u, Vector *gu);
void makeContinuous(ParGridFunction &u);

bool copyFile(const char *SRC, const char *DEST);

/// Eliminate essential BCs in an Operator and apply to RHS.
/// rename this to something sensible "ApplyEssentialBC" or something
void EliminateRHS(Operator &A, ConstrainedOperator &constrainedA, const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                  Vector &X, Vector &B, int copy_interior = 0);

/// Remove mean from a Vector.
/**
 * Modify the Vector @a v by subtracting its mean using
 * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
 */
void Orthogonalize(Vector &v, MPI_Comm comm);

// Adding to the mfem namespace
namespace mfem {

/// Matrix coefficient defined as the Gradient of a Vector GridFunction
class GradientVectorGridFunctionCoefficient : public MatrixCoefficient {
 protected:
  const GridFunction *GridFunc;

 public:
  /** @brief Construct the coefficient with a scalar grid function @a gf. The
      grid function is not owned by the coefficient. */
  GradientVectorGridFunctionCoefficient(const GridFunction *gf);

  /// Set the scalar grid function.
  void SetGridFunction(const GridFunction *gf);

  /// Get the scalar grid function.
  const GridFunction *GetGridFunction() const { return GridFunc; }

  /// Evaluate the gradient vector coefficient at @a ip.
  virtual void Eval(DenseMatrix &G, ElementTransformation &T, const IntegrationPoint &ip);

  virtual ~GradientVectorGridFunctionCoefficient() {}
};
}  // namespace mfem

#endif  // UTILS_HPP_
