// -----------------------------------------------------------------------------------bl-
// -----------------------------------------------------------------------------------el-
#ifndef UTILS_
#define UTILS_

#include <hdf5.h>
#include <assert.h>
#include <string>

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

#endif



