// -----------------------------------------------------------------------------------bl-
// -----------------------------------------------------------------------------------el-
#ifndef UTILS_
#define UTILS_

#include <hdf5.h>
#include <assert.h>
#include <string>
#include <mfem.hpp>

// Misc. utilities
bool file_exists (const std::string &name);
std::string systemCmd(const char *cmd);

// HDF5 convenience utilities
inline hid_t h5_getType( int)   { return(H5T_NATIVE_INT);    };
inline hid_t h5_getType(double) { return(H5T_NATIVE_DOUBLE); };

template <typename T> void h5_save_attribute(hid_t dest,std::string attribute, T value)
{
  hid_t attr, status;
  hid_t attrType    = h5_getType(value);
  hid_t dataspaceId = H5Screate(H5S_SCALAR);

  attr = H5Acreate(dest,attribute.c_str(), attrType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
  assert(attr >= 0);
  status = H5Awrite(attr,attrType,&value);
  assert(status >= 0);
  H5Aclose(attr);
  H5Sclose(dataspaceId);
}

template <typename T> void h5_read_attribute(hid_t source,std::string attribute, T &value)
{
  herr_t status;
  hid_t attr;
  hid_t attrType = h5_getType( value );
  //  T temp;

  attr   = H5Aopen_name(source,attribute.c_str());
  status = H5Aread(attr,attrType,&value);
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
void LocalProjectDiscCoefficient(mfem::GridFunction &gf,
                                 mfem::VectorCoefficient &coeff,
                                 mfem::Array<int> &dof_attr);

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
void GlobalProjectDiscCoefficient(mfem::ParGridFunction &gf,
                                  mfem::VectorCoefficient &coeff);

#endif



