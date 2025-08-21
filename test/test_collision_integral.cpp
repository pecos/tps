/* A utility to transfer tps solutions from one mesh to another */

#include <hdf5.h>
#include <grvy.h>
#include "../src/utils.hpp"

#include "../src/collision_integrals.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

Array<int> readTable(const string fileName, const string datasetName, DenseMatrix &output) {
  hid_t file = -1;
  if (file_exists(fileName)) {
    file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } else {
    grvy_printf(GRVY_ERROR, "[ERROR]: Unable to open file -> %s\n", fileName.c_str());
    exit(ERROR);
  }
  assert(file >= 0);

  hid_t datasetID, dataspace;
  datasetID = H5Dopen2(file, datasetName.c_str(), H5P_DEFAULT);
  assert(datasetID >= 0);
  dataspace = H5Dget_space(datasetID);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims[ndims];
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  output.SetSize(dims[0], dims[1]);

  herr_t status;
  status = H5Dread(datasetID, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.HostReadWrite());
  assert(status >= 0);
  H5Dclose(datasetID);
  H5Fclose(file);

  Array<int> shape(2);
  shape[0] = dims[0];
  shape[1] = dims[1];

  return shape;
}

int main (int argc, char *argv[])
{
  {
    std::string fileName = "./ref_solns/collisions/Mason.h5";
    std::string datasetName = "Mason1967";

    DenseMatrix Mason, MasonFitted;
    Array<int> dims = readTable(fileName, datasetName, Mason);
    fileName = "./ref_solns/collisions/Mason.fitted.h5";
    Array<int> dummy = readTable(fileName, datasetName, MasonFitted);

    Vector Tm(dims[0]);
    Mason.GetColumn(0, Tm);

    std::vector<std::string> rString(5);
    rString[0] = "11";
    rString[1] = "12";
    rString[2] = "13";
    rString[3] = "22";
    rString[4] = "23";
    std::vector<std::string> typeString(2);
    typeString[0] = "att";
    typeString[1] = "rep";

    double errorThreshold = 1.5e-4;

    for (int r = 0; r < 5; r++) {
      for (int type = 0; type < 2; type++) {
        Vector O(dims[0]);
        double relError = 0.0, relError1 = 0.0;
        for (int m = 0; m < dims[0]; m++) {
          switch (r) {
            case 0:
              if (type == 0) {
                O(m) = collision::charged::att11(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep11(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 1:
              if (type == 0) {
                O(m) = collision::charged::att12(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep12(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 2:
              if (type == 0) {
                O(m) = collision::charged::att13(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep13(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 3:
              if (type == 0) {
                O(m) = collision::charged::att22(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep22(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 4:
              if (type == 0) {
                O(m) = collision::charged::att23(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep23(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
          }
          int index = 1 + type + 2 * r;

          relError += abs((MasonFitted(m,index) - O(m)) / Mason(m,index));
          relError1 += abs((Mason(m,index) - O(m)) / Mason(m,index));
          // std::cout << O(m) << " =?= " << MasonFitted(m,index) << std::endl;
          // std::cout << O(m) * Tm(m) * Tm(m) << " =?= " << Mason(m,1) << std::endl;
        }
        relError /= dims[0];
        relError1 /= dims[0];
        grvy_printf(GRVY_INFO, "\n O%s-%s with respect to reference value: %.8E\n", rString[r].c_str(), typeString[type].c_str(), relError);
        grvy_printf(GRVY_INFO, "\n O%s-%s with respect to Mason (1967): %.8E\n", rString[r].c_str(), typeString[type].c_str(), relError1);
        if (relError > errorThreshold) {
          grvy_printf(GRVY_ERROR, "\n Collision integral error beyond threshold: %.8E\n", errorThreshold);
          exit(ERROR);
        }
        // std::cout << relError << std::endl;
        // std::cout << relError1 << std::endl;
      } // for type
    } // for r

    grvy_printf(GRVY_INFO, "\nPASS: charged species (1,1-3), (2,2-3) collision integrals.\n");
  }

  {
    std::string fileName = "./ref_solns/collisions/Devoto.h5";
    std::string datasetName = "Devoto1973";

    DenseMatrix Devoto, DevotoFitted;
    Array<int> dims = readTable(fileName, datasetName, Devoto);
    fileName = "./ref_solns/collisions/Devoto.fitted.h5";
    Array<int> dummy = readTable(fileName, datasetName, DevotoFitted);

    Vector Tm(dims[0]);
    Devoto.GetColumn(0, Tm);

    std::vector<std::string> rString(5);
    rString[0] = "14";
    rString[1] = "15";
    rString[2] = "24";
    std::vector<std::string> typeString(2);
    typeString[0] = "att";
    typeString[1] = "rep";

    double errorThreshold = 2.0e-3;

    for (int r = 0; r < 3; r++) {
      for (int type = 0; type < 2; type++) {
        Vector O(dims[0]);
        double relError = 0.0, relError1 = 0.0;
        for (int m = 0; m < dims[0]; m++) {
          switch (r) {
            case 0:
              if (type == 0) {
                O(m) = collision::charged::att14(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep14(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 1:
              if (type == 0) {
                O(m) = collision::charged::att15(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep15(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
            case 2:
              if (type == 0) {
                O(m) = collision::charged::att24(Tm(m)) * Tm(m) * Tm(m);
              } else {
                O(m) = collision::charged::rep24(Tm(m)) * Tm(m) * Tm(m);
              }
            break;
          }
          int index = 1 + type + 2 * r;

          relError += abs((DevotoFitted(m,index) - O(m)) / Devoto(m,index));
          relError1 += abs((Devoto(m,index) - O(m)) / Devoto(m,index));
          // std::cout << O(m) << " =?= " << DevotoFitted(m,index) << std::endl;
          // std::cout << O(m) * Tm(m) * Tm(m) << " =?= " << Mason(m,1) << std::endl;
        }
        relError /= dims[0];
        relError1 /= dims[0];
        grvy_printf(GRVY_INFO, "\n O%s-%s with respect to reference value: %.8E\n", rString[r].c_str(), typeString[type].c_str(), relError);
        grvy_printf(GRVY_INFO, "\n O%s-%s with respect to Devoto (1973): %.8E\n", rString[r].c_str(), typeString[type].c_str(), relError1);
        if (relError > errorThreshold) {
          grvy_printf(GRVY_ERROR, "\n Collision integral error beyond threshold: %.8E\n", errorThreshold);
          exit(ERROR);
        }
        // std::cout << relError << std::endl;
        // std::cout << relError1 << std::endl;
      } // for type
    } // for r

    grvy_printf(GRVY_INFO, "\nPASS: charged species (1,4), (1,5), (2,4) collision integrals.\n");
  }

  {
    DenseMatrix fit;
    std::string fileName = "./ref_solns/collisions/Oliver.fit.h5";
    std::string datasetName1 = "Qe_Ar1r";
    Array<int> dims = readTable(fileName, datasetName1, fit);

    Vector Tm(dims[0]);
    fit.GetColumn(0, Tm);

    std::vector<std::string> rString(5);
    rString[0] = "11";
    rString[1] = "12";
    rString[2] = "13";
    rString[3] = "14";
    rString[4] = "15";

    double errorThreshold = 2e-3;

    for (int r = 0; r < 5; r++) {
      Vector O(dims[0]);
      double relError = 0.0;
      for (int m = 0; m < dims[0]; m++) {
        switch (r) {
          case 0:
            O(m) = collision::argon::eAr11(Tm(m));
          break;
          case 1:
            O(m) = collision::argon::eAr12(Tm(m));
          break;
          case 2:
            O(m) = collision::argon::eAr13(Tm(m));
          break;
          case 3:
            O(m) = collision::argon::eAr14(Tm(m));
          break;
          case 4:
            O(m) = collision::argon::eAr15(Tm(m));
          break;
        }
        int index = 1 + r;

        relError += abs((fit(m,index) - O(m)) / fit(m,index));
      }
      relError /= dims[0];
      grvy_printf(GRVY_INFO, "\n O%s with respect to reference value: %.8E\n", rString[r].c_str(), relError);
      if (relError > errorThreshold) {
        grvy_printf(GRVY_ERROR, "\n Collision integral error beyond threshold: %.8E\n", errorThreshold);
        exit(ERROR);
      }
      // std::cout << relError << std::endl;
      // std::cout << relError1 << std::endl;
    } // for r

    grvy_printf(GRVY_INFO, "\nPASS: e-Ar (1,r) collision integrals.\n");
  }

  {
    std::string fileName = "./ref_solns/collisions/Devoto.argon.IA.h5";
    std::string datasetName = "Qia";

    DenseMatrix Devoto;
    Array<int> dims = readTable(fileName, datasetName, Devoto);

    Vector Tm(dims[0]);
    Devoto.GetColumn(0, Tm);

    double errorThreshold = 3.0e-3;

    Vector O(dims[0]);
    double relError = 0.0; // , relError1 = 0.0;
    for (int m = 0; m < dims[0]; m++) {
      O(m) = collision::argon::ArAr1P11(Tm(m));
      int index = 1;

      relError += abs((Devoto(m,index) - O(m)) / Devoto(m,index));
      std::cout << O(m) << " =?= " << Devoto(m,index) << std::endl;
      // std::cout << O(m) * Tm(m) * Tm(m) << " =?= " << Mason(m,1) << std::endl;
    }
    relError /= dims[0];
    grvy_printf(GRVY_INFO, "\n O11 with respect to Devoto (1973): %.8E\n", relError);
    if (relError > errorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Collision integral error beyond threshold: %.8E\n", errorThreshold);
      exit(ERROR);
    }
    // std::cout << relError << std::endl;
    // std::cout << relError1 << std::endl;

    grvy_printf(GRVY_INFO, "\nPASS: Ar-Ar1+ (1,1) collision integral.\n");
  }

  {
    std::string fileName = "./ref_solns/collisions/Amdur_Mason.estimated.h5";
    std::string datasetName = "Amdur_est";

    DenseMatrix Amdur;
    Array<int> dims = readTable(fileName, datasetName, Amdur);

    Vector Tm(dims[0]);
    Amdur.GetColumn(0, Tm);

    double errorThreshold = 1.0e-2;

    Vector O(dims[0]);
    double relError = 0.0; // , relError1 = 0.0;
    for (int m = 0; m < dims[0]; m++) {
      O(m) = collision::argon::ArAr11(Tm(m));
      int index = 3;

      relError += abs((Amdur(m,index) - O(m)) / Amdur(m,index));
      std::cout << O(m) << " =?= " << Amdur(m,index) << std::endl;
      // std::cout << O(m) * Tm(m) * Tm(m) << " =?= " << Mason(m,1) << std::endl;
    }
    relError /= dims[0];
    grvy_printf(GRVY_INFO, "\n Q11 with respect to Amdur & Mason (1958): %.8E\n", relError);
    if (relError > errorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Collision integral error beyond threshold: %.8E\n", errorThreshold);
      exit(ERROR);
    }
    // std::cout << relError << std::endl;
    // std::cout << relError1 << std::endl;

    grvy_printf(GRVY_INFO, "\nPASS: Ar-Ar (1,1) collision integral.\n");
  }

  return 0;
}
