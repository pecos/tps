// #include "../src/table.hpp"
// #include "../src/tps.hpp"
#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
// #include "../src/utils.hpp"
#include <fstream>
#include <hdf5.h>

using namespace mfem;
using namespace std;

const double scalarErrorThreshold = 5e-13;
const int nTrials = 10;

double uniformRandomNumber() {
  return (double) rand() / (RAND_MAX);
}

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

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();
  Chemistry *chem = srcField->getChemistry();
  assert(srcConfig.numReactions == 1);
  assert(srcConfig.reactionModels[0] = TABULATED);
  std::string basePath("reactions/reaction1/tabulated");

  std::string fileName;
  tpsP->getRequiredInput((basePath + "/filename").c_str(), filename);
  std::string datasetName = "table";
  DenseMatrix refValues;
  Array<int> dims1 = h5ReadTable(fileName, datasetName, refValues);
  // input.Ndata = dims1[0];
  // for (int k = 0; k < input.Ndata; k++) {
  //   input.xdata[k] = refValues(k, 0);
  //   input.fdata[k] = refValues(k, 1);
  // }
  // input.xLogScale = true;
  // input.fLogScale = true;
  // table = new LinearTable(input);

  // TODO(kevin): write a sort of sanity check for this. The values are manually checked at this point.
   for (int k = 0; k < dims1[0]; k++) {
     // double xtest = 300.0 * pow(1.0e4 / 300.0, 1.0 * k / Ntest);
     double xtest = refValues(k, 0);
     double fref = refValues(k, 1);
     double ftest[gpudata::MAXREACTIONS];
     chem->computeForwardRateCoeffs(xtest, xtest, ftest);
     printf("%.5E: %.5E ?= %.5E", xtest, fref, ftest[0]);
   }
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

  return 0;
}
