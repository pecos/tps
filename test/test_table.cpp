#include "../src/table.hpp"
#include "../src/tps.hpp"
#include "mfem.hpp"
#include <fstream>

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
  const int Ntest = 1000;
  for (int k = 0; k < Ntest; k++) {
    double xEval = input.xdata[0] + (input.xdata[Ndata-1] - input.xdata[0]) * uniformRandomNumber();
    int index = table->findInterval(xEval);
    if ((input.xdata[index] > xEval) || (input.xdata[index+1] < xEval)) {
      grvy_printf(GRVY_ERROR, "findIndex failed!: %.5f < xEval: %.5f < %.5f\n", input.xdata[index], xEval, input.xdata[index+1]);
      exit(ERROR);
    }
  }
//std::cout << "before" << std::endl;
//  TableInput *test;
//  test = new TableInput[10];
//std::cout << "after" << std::endl;
  delete table;

  return 0;
}

