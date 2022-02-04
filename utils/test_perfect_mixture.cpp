/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  TPS::Tps tps(argc, argv);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::USER_DEFINED);
  assert(srcConfig.GetGasModel()==PERFECT_MIXTURE);

  int dim = 3;
  PerfectMixture *mixture = new PerfectMixture( srcConfig, dim);

  return 0;
}
