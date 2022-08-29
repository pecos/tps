#include "../src/table.hpp"
#include "../src/tps.hpp"
#include "mfem.hpp"
#include "../src/utils.hpp"
#include <fstream>
#include <hdf5.h>

using namespace mfem;
using namespace std;

const double scalarErrorThreshold = 5e-13;
const int nTrials = 10;

double uniformRandomNumber() {
  return (double) rand() / (RAND_MAX);
}
//
// Array<int> readTable(const string fileName, const string datasetName, DenseMatrix &output) {
//   hid_t file = -1;
//   if (file_exists(fileName)) {
//     file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//   } else {
//     grvy_printf(GRVY_ERROR, "[ERROR]: Unable to open file -> %s\n", fileName.c_str());
//     exit(ERROR);
//   }
//   assert(file >= 0);
//
//   hid_t datasetID, dataspace;
//   datasetID = H5Dopen2(file, datasetName.c_str(), H5P_DEFAULT);
//   assert(datasetID >= 0);
//   dataspace = H5Dget_space(datasetID);
//   const int ndims = H5Sget_simple_extent_ndims(dataspace);
//   hsize_t dims[ndims];
//   // int dummy = H5Sget_simple_extent_dims(dataspace,dims,NULL);
//   H5Sget_simple_extent_dims(dataspace,dims,NULL);
//
//   // DenseMatrix memory is column-major, while HDF5 follows row-major.
//   output.SetSize(dims[1], dims[0]);
//
//   herr_t status;
//   status = H5Dread(datasetID, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.HostReadWrite());
//   assert(status >= 0);
//   H5Dclose(datasetID);
//   H5Fclose(file);
//
//   // DenseMatrix memory is column-major, while HDF5 follows row-major.
//   output.Transpose();
//
//   Array<int> shape(2);
//   shape[0] = dims[0];
//   shape[1] = dims[1];
//
//   return shape;
// }

int main (int argc, char *argv[])
{
  TPS::Tps tps;
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  srand (time(NULL));
  // double dummy = uniformRandomNumber();

  const int Ndata = 57;
  const double L = 5.0 * uniformRandomNumber();
  double dx = L / (Ndata - 1);

  TableInput input;
  input.Ndata = Ndata;
  input.xdata = new double[gpudata::MAXTABLE];
  input.fdata = new double[gpudata::MAXTABLE];
  for (int k = 0; k < Ndata; k++) {
    input.xdata[k] = k * dx + 0.4 * dx * (2.0 * uniformRandomNumber() - 1.0);
  }
  TableInterpolator *table = new LinearTable(input);

  // test findInterval routine.
  const int Ntest = 100;
  for (int k = 0; k < Ntest; k++) {
    double xEval = input.xdata[0] + (input.xdata[Ndata-1] - input.xdata[0]) * uniformRandomNumber();
    int index = table->findInterval(xEval);
    if ((input.xdata[index] > xEval) || (input.xdata[index+1] < xEval)) {
      grvy_printf(GRVY_ERROR, "findIndex failed!: %.5f < xEval: %.5f < %.5f\n", input.xdata[index], xEval, input.xdata[index+1]);
      exit(ERROR);
    }
  }

  delete table;

  std::string fileName = "ref_solns/reaction/excitation.3000K.ion1e-4.h5";
  std::string datasetName = "table";
  DenseMatrix refValues;
  Array<int> dims1 = h5ReadTable(fileName, datasetName, refValues);
  input.Ndata = dims1[0];
  for (int k = 0; k < input.Ndata; k++) {
    input.xdata[k] = refValues(k, 0);
    input.fdata[k] = refValues(k, 1);
  }
  input.xLogScale = true;
  input.fLogScale = true;
  table = new LinearTable(input);

  // TODO(kevin): write a sort of sanity check for this. The values are manually checked at this point.
//  std::cout << "[";
//  for (int k = 0; k < Ntest; k++) {
//    double xtest = 300.0 * pow(1.0e4 / 300.0, 1.0 * k / Ntest);
//    double ftest = table->eval(xtest);
//    std::cout << "[" << xtest << "," << ftest << "]";
//    if (k < Ntest-1) std::cout << ",";
//    std::cout << std::endl;
//  }
//  std::cout << "]" << std::endl;

  delete table;

  return 0;
}
