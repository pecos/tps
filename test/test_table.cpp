#include <tps_config.h>
#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>
#include <hdf5.h>

using namespace mfem;
using namespace std;

const double scalarErrorThreshold = 1e-13;

double uniformRandomNumber() {
  return (double) rand() / (RAND_MAX);
}

void testTableInterpolator1D(TPS::Tps &tps, int rank) {
  srand (time(NULL));

  const int Ndata = 57;
  const double L = 5.0 * uniformRandomNumber();
  double dx = L / (Ndata - 1);

  TableInput input;
  input.order = 1;
  input.Ndata = Ndata;
  DenseMatrix data;
  data.SetSize(Ndata, 2);
  for (int k = 0; k < Ndata; k++) {
    data(k, 0) = k * dx + 0.4 * dx * (2.0 * uniformRandomNumber() - 1.0);
  }
  input.xdata = data.Read();
  input.fdata = data.Read() + Ndata;
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
  // Check the values outside the range.
  double xEval = input.xdata[0] - 0.1 * (input.xdata[Ndata-1] - input.xdata[0]);
  int index = table->findInterval(xEval);
  if (index != 0) {
    grvy_printf(GRVY_ERROR, "findIndex below the range failed!: %d != %d\n", index, 0);
    exit(ERROR);
  }
  xEval = input.xdata[Ndata-1] + 0.1 * (input.xdata[Ndata-1] - input.xdata[0]);
  index = table->findInterval(xEval);
  if (index != (Ndata-2)) {
    grvy_printf(GRVY_ERROR, "findIndex above the range failed!: %d != %d\n", index, Ndata-2);
    exit(ERROR);
  }

  delete table;

  if (rank == 0) printf("findInterval test passed.\n");

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), tps.getInputFilename(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();
  Chemistry *chem = srcField->getChemistry();
  assert(srcConfig.numReactions == 1);
  assert(srcConfig.reactionModels[0] = TABULATED_RXN);
  std::string basePath("reactions/reaction1/tabulated");

  if (rank == 0) printf("chemistry initialized.\n");

  std::string fileName;
  tps.getRequiredInput((basePath + "/filename").c_str(), fileName);
  std::string datasetName = "table";
  DenseMatrix refValues;
  Array<int> dims1(2);
  bool success = h5ReadTable(fileName, datasetName, refValues, dims1);
  if (!success) {
    grvy_printf(GRVY_ERROR, "Reading the table file %s failed!\n", fileName.c_str());
    exit(ERROR);
  }

  if (rank == 0) printf("Table read.\n");

  // NOTE(kevin): checks if the returned value is exact interpolation.
  // will need a different test if inexact interpolator is used.
  for (int k = 0; k < dims1[0]; k++) {
    // double xtest = 300.0 * pow(1.0e4 / 300.0, 1.0 * k / Ntest);
    double xtest = refValues(k, 0);
    double fref = refValues(k, 1);
    double ftest[gpudata::MAXREACTIONS];
    chem->computeForwardRateCoeffs(xtest, xtest, ftest);
    double error = abs((fref - ftest[0]) / fref);
    if (error >= scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "Rank %d - %.5E: %.5E\n", rank, xtest, abs((fref - ftest[0]) / fref));
      exit(ERROR);
    }
  }
  if (rank == 0) printf("Exact interpolation test passed.\n");
//  std::cout << "[";
//  for (int k = 0; k < Ntest; k++) {
//    double xtest = 300.0 * pow(1.0e4 / 300.0, 1.0 * k / Ntest);
//    double ftest = table->eval(xtest);
//    std::cout << "[" << xtest << "," << ftest << "]";
//    if (k < Ntest-1) std::cout << ",";
//    std::cout << std::endl;
//  }
//  std::cout << "]" << std::endl;

  // delete table;
}


#ifdef HAVE_GSL
int testGslTableInterpolator2D() {
  int ierr = 0;

  // Test 1: A trivial test
  {
    //   ^
    //   |
    //   *     *
    //   |
    //   |
    //   *  o  *
    //   |
    //   |
    //   *-----*-->
    //
    // Have data at *, interpolate to o

    const unsigned int nx = 2;
    const unsigned int ny = 3;

    double *xx = new double[nx];
    double *yy = new double[ny];

    xx[0] = 0.; xx[1] = 1.;
    yy[0] = -2.; yy[1] = 0.; yy[2] = 2.;

    double *fval = new double[nx * ny];

    fval[0] = 0.;
    fval[1] = 1.;
    fval[2] = -1.;
    fval[3] = 0.;
    fval[4] = -3.;
    fval[5] = -2.;

    {
      GslTableInterpolator2D interp(nx, ny, xx, yy, fval);
      const double fexact = -0.5;
      double x0 = 0.5;
      double y0 = 0.0;
      double f0 = interp.eval(x0, y0);
      if (std::abs(f0 - fexact) > 1e-15) {
        ierr += 1;
      }
    }

    delete[] fval;
    delete[] yy;
    delete[] xx;
  }

  // Test 2: Read thermo data from a plato file
  {
    {
      std::string plato_file("./inputs/argon_lte_thermo_table.dat");
      GslTableInterpolator2D pressure_interp(plato_file, 0, 1, 2);

      // spot check a couple points
      double pexact = 2.0813213725000001e+03;
      double T = 2000;
      double rho = 0.005;
      double p = pressure_interp.eval(T, rho);
      if (std::abs(p - pexact) > 1e-15) {
        ierr += 1;
      }

      pexact = 5.3260479727500002e+05;
      T = 9950.;
      rho = 0.255;
      p = pressure_interp.eval(T, rho);
      if (std::abs(p - pexact) > 1e-15) {
        ierr += 1;
      }
    }

    {
      std::string plato_file("./inputs/argon_lte_thermo_table.dat");
      GslTableInterpolator2D R_interp(plato_file, 0, 1, 6);

      // spot check a couple points
      double Rexact = 208.1321372;
      double T = 2000;
      double rho = 0.005;
      double R = R_interp.eval(T, rho);
      if (std::abs(R - Rexact) > 1e-15) {
        ierr += 1;
      }
    }
  }

  return ierr;
}
#endif  // HAVE_GSL


int main (int argc, char *argv[])
{
  TPS::Tps tps;
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  MPI_Session mpi = tps.getMPISession();
  int rank = mpi.WorldRank();

  testTableInterpolator1D(tps, rank);

  int ierr = 0;
#ifdef HAVE_GSL
  ierr = testGslTableInterpolator2D();
#endif

  return ierr;
}
