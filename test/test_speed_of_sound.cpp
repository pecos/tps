/* A utility to transfer tps solutions from one mesh to another */

#include "tps.hpp"
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

  // srand (time(NULL));
  // double dummy = uniformRandomNumber();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);

  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::USER_DEFINED);
  assert(srcConfig.GetGasModel()==PERFECT_MIXTURE);

  int dim = 3;

  /*
    Test non-ambipolar, single-temperature fluid
  */
  {
    srcConfig.ambipolar = false;
    srcConfig.twoTemperature = false;
    PerfectMixture *mixture = new PerfectMixture( srcConfig, dim);

    double airMW = 28.964e-3;

    int numSpecies = mixture->GetNumSpecies();
    int numActiveSpecies = mixture->GetNumActiveSpecies();
    int numEquation = mixture->GetNumEquations();
    Vector X_sp(numSpecies);
    X_sp(0) = 0.0407e-2;
    X_sp(1) = 0.934e-2;
    X_sp(2) = 20.946e-2;
    X_sp(3) = 1.0;
    for (int sp = 0; sp < numSpecies - 1; sp++) X_sp(3) -= X_sp(sp);
    grvy_printf(GRVY_INFO, "\n Mole fraction of dry air.\n");
    grvy_printf(GRVY_INFO, "\n %.8E, %.8E, %.8E, %.8E\n", X_sp(0), X_sp(1), X_sp(2), X_sp(3));


    double rho = 1.2041;
    double nTotal = rho / airMW;
    double T_h = 293.15;

    Vector primitive(numEquation), conserved(numEquation);
    primitive = 0.0;
    primitive(0) = rho;
    primitive(dim + 1) = T_h;
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      primitive(dim + 2 + sp) = nTotal * X_sp(sp);
    }

    double rhoSum = 0.0;
    for (int sp = 0; sp < numSpecies; sp++) {
      rhoSum += nTotal * X_sp(sp) * mixture->GetGasParams(sp, GasParams::SPECIES_MW);
    }
    grvy_printf(GRVY_INFO, "\n Input rho: %.8E ~ Computed rho: %.8E\n", rho, rhoSum);
    grvy_printf(GRVY_INFO, "\n Input air mw: %.8E ~ Computed air mw: %.8E\n", airMW, rhoSum / nTotal);

    double heat_ratio = mixture->computeHeaviesMixtureHeatRatio(&primitive(dim+2), nTotal * X_sp(numSpecies-1));
    grvy_printf(GRVY_INFO, "\n DryAir gamma: 1.4 ~ Computed gamma: %.8E\n", heat_ratio);

    double DryAirSound = sqrt( 1.4 * UNIVERSALGASCONSTANT / airMW * T_h );

    double sound = mixture->ComputeSpeedOfSound(primitive,true);
    grvy_printf(GRVY_INFO, "\n DryAir speed of sound: %.8E ~ Computed value: %.8E\n", DryAirSound, sound);

    double error = abs( (DryAirSound - sound) / DryAirSound );
    grvy_printf(GRVY_INFO, "\n Speed of sound relative deviation: %.8E\n", error);
    if ( error > 1e-4 ) {
      grvy_printf(GRVY_ERROR, "\n Speed of sound deviates from DryAir value.");
      exit(ERROR);
    }

    mixture->GetConservativesFromPrimitives(primitive, conserved);
    double soundFromConserved = mixture->ComputeSpeedOfSound(conserved, false);
    error = abs( (soundFromConserved - sound) / sound );
    if ( error > 1e-15 ) {
      grvy_printf(GRVY_ERROR, "\n Speed of sound is not consistent between primitive and conservative.");
      exit(ERROR);
    }

  }
  grvy_printf(GRVY_INFO, "\nPASS: speed of sound for dry-air at room temperature.\n");

  return 0;
}
