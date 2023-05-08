/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

const double scalarErrorThreshold = 5e-13;
const int nTrials = 10;

double uniformRandomNumber() {
  return (double) rand() / (RAND_MAX);
}

double getRandomPrimitiveState(PerfectMixture *mixture, Vector &primitiveState) {
  const double numEquation = mixture->GetNumEquations();
  const double numSpecies = mixture->GetNumSpecies();
  const double numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool ambipolar = mixture->IsAmbipolar();
  const bool twoTemperature = mixture->IsTwoTemperature();
  // const double dim = mixture->GetDimension();
  const double nvel = mixture->GetNumVels();

  primitiveState.SetSize(numEquation);
  // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
  for (int d = 0; d < nvel; d++) {
    primitiveState(d + 1) = -0.5 + 1.0 * uniformRandomNumber();
  }
  primitiveState(nvel + 1) = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
  if (twoTemperature) primitiveState(numEquation - 1) = 400.0 * ( 0.8 + 0.4 * uniformRandomNumber() );

  Vector n_sp(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++){
    if (ambipolar && (sp == numSpecies - 2)) continue;
    n_sp(sp) = 1.0e0 * uniformRandomNumber();
    if (sp == numSpecies - 1) n_sp(sp) += 2.0e0;
  }
  if (ambipolar) {
    double ne = mixture->computeAmbipolarElectronNumberDensity(&n_sp[0]);
    n_sp(numSpecies - 2) = ne;
  }
  grvy_printf(GRVY_INFO, "\n number densities.\n");
  for (int sp = 0; sp < numSpecies; sp++){
    grvy_printf(GRVY_INFO, "%.8E, ", n_sp(sp));
  }
  grvy_printf(GRVY_INFO, "\n");

  double rho = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) rho += n_sp(sp) * mixture->GetGasParams(sp,GasParams::SPECIES_MW);
  primitiveState(0) = rho;

  for (int sp = 0; sp < numActiveSpecies; sp++) primitiveState(nvel + 2 + sp) = n_sp(sp);

  grvy_printf(GRVY_INFO, "\n random primitive.\n");
  for (int eq = 0; eq < numEquation; eq++) {
    grvy_printf(GRVY_INFO, "%.8E\t", primitiveState(eq));
  }
  grvy_printf(GRVY_INFO, "\n");

  double Yb = n_sp(numSpecies - 1) * mixture->GetGasParams(numSpecies - 1,GasParams::SPECIES_MW) / rho;
  return Yb;
}

void getRandomDirection(const Vector &velocity, Vector &direction) {
  const double dim = velocity.Size();
  Vector unitVel(dim);
  double velNorm = 0.0;
  for (int d = 0; d < dim; d++) velNorm += velocity(d) * velocity(d);
  for (int d = 0; d < dim; d++) unitVel(d) = velocity(d) / sqrt(velNorm);

  direction.SetSize(dim);
  velNorm = 0.0;
  for (int d = 0; d < dim; d++) {
    direction(d) = unitVel(d) * (0.5 + uniformRandomNumber());
    velNorm += direction(d) * direction(d);
  }
  for (int d = 0; d < dim; d++) direction(d) /= sqrt(velNorm);
  grvy_printf(GRVY_INFO, "\n direction.\n");
  for (int d = 0; d < dim; d++) {
    grvy_printf(GRVY_INFO, "%.8E, ", direction(d));
  }
  grvy_printf(GRVY_INFO, "\n");
}

bool testComputeBdrViscousFlux(RunConfiguration &srcConfig, const int dim) {
  bool success = true;

  int nvel = (srcConfig.axisymmetric_) ? dim + 1 : dim;
  PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, nvel);
  ConstantTransport *transport = new ConstantTransport(mixture, srcConfig);
  Equations eqSystem = NS;
  int num_equation = mixture->GetNumEquations();
  int numSpecies = mixture->GetNumSpecies();
  int primFluxSize = (mixture->IsTwoTemperature()) ? numSpecies + nvel + 2 : numSpecies + nvel + 1;

  Fluxes *flux = new Fluxes(mixture, eqSystem, transport, num_equation, dim, srcConfig.axisymmetric_, &srcConfig);
  const double delta = 0.0;
  Vector xyz(3);
  xyz = 0.;

  grvy_printf(GRVY_INFO, "\n Setting a random primitive variable. \n");

  Vector testPrimitives(num_equation), testConserved(num_equation);
  //double Yb = getRandomPrimitiveState(mixture, testPrimitives);
  getRandomPrimitiveState(mixture, testPrimitives);
  mixture->GetConservativesFromPrimitives(testPrimitives, testConserved);

  grvy_printf(GRVY_INFO, "\n Setting a random primitive gradient. \n");

  DenseMatrix gradUp(num_equation,dim);
  gradUp = 0.0;
  for (int eq = 0; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) {
      gradUp(eq, d) = -10.0 + 20.0 * uniformRandomNumber();
      grvy_printf(GRVY_INFO, "%.8E\t", gradUp(eq, d));
    }
    grvy_printf(GRVY_INFO, "\n");
  }
  Vector dir(dim);
  Vector velocity(dim);  // only care dim components if axisymmetric.
  for (int d = 0; d < dim; d++) velocity(d) = testPrimitives(d + 1);
  getRandomDirection(velocity, dir);

  grvy_printf(GRVY_INFO, "\n ComputeViscousFluxes * normal. \n");
  DenseMatrix viscF(num_equation, dim);
  Vector viscFdotNorm(num_equation);
  flux->ComputeViscousFluxes(testConserved, gradUp, xyz, delta, viscF);
  viscF.Mult(dir, viscFdotNorm);
  for (int eq = 0; eq < num_equation; eq++) {
    grvy_printf(GRVY_INFO, "%.8E\t", viscFdotNorm(eq));
  }
  grvy_printf(GRVY_INFO, "\n");

  grvy_printf(GRVY_INFO, "\n ComputeBdrViscousFluxes. \n");
  BoundaryViscousFluxData bcFlux;
  for (int d = 0; d < dim; d++) bcFlux.normal[d] = dir[d];
  // bcFlux.primFlux.SetSize(primFluxSize);
  // bcFlux.primFluxIdxs.SetSize(primFluxSize);
  for (int i = 0; i < primFluxSize; i++) bcFlux.primFlux[i] = 0.0;
  for (int i = 0; i < primFluxSize; i++) bcFlux.primFluxIdxs[i] = false;

  Vector wallViscF(num_equation);
  flux->ComputeBdrViscousFluxes(testConserved, gradUp, xyz, delta, bcFlux, wallViscF);
  for (int eq = 0; eq < num_equation; eq++) {
    grvy_printf(GRVY_INFO, "%.8E\t", wallViscF(eq));
  }
  grvy_printf(GRVY_INFO, "\n");

  grvy_printf(GRVY_INFO, "\n Relative errors. \n");
  Vector error(num_equation - 1);
  double maxError = 0.0;
  for (int eq = 1; eq < num_equation; eq++) {
    error(eq - 1) = abs(viscFdotNorm(eq) - wallViscF(eq));
    if (abs(viscFdotNorm(eq)) > scalarErrorThreshold)
      error(eq - 1) /= abs(viscFdotNorm(eq));
    maxError = max(maxError, error(eq - 1));
    grvy_printf(GRVY_INFO, "%.8E\t", error(eq - 1));
  }
  grvy_printf(GRVY_INFO, "\n");

  success = (maxError < scalarErrorThreshold);

  return success;
}

int main (int argc, char *argv[])
{
  TPS::Tps tps;
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  srand (time(NULL));
  // double dummy = uniformRandomNumber();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);

  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::USER_DEFINED);
  assert(srcConfig.GetGasModel()==PERFECT_MIXTURE);
  assert(srcConfig.GetTranportModel()==CONSTANT);
  assert(!srcConfig.axisymmetric_);
  // srcConfig.ambipolar = false;
  // srcConfig.twoTemperature = false;
  srcConfig.constantTransport.viscosity = uniformRandomNumber();
  srcConfig.constantTransport.bulkViscosity = uniformRandomNumber();
  srcConfig.constantTransport.thermalConductivity = uniformRandomNumber();
  srcConfig.constantTransport.electronThermalConductivity = uniformRandomNumber();
  // srcConfig.constantTransport.diffusivity.SetSize(srcConfig.numSpecies);
  // srcConfig.constantTransport.mtFreq.SetSize(srcConfig.numSpecies);
  for (int sp = 1; sp <= srcConfig.numSpecies; sp++) {
    srcConfig.constantTransport.diffusivity[sp - 1] = uniformRandomNumber();
    srcConfig.constantTransport.mtFreq[sp - 1] = uniformRandomNumber();
  }

  int dim = 2;

  for (int amb = 0; amb < 2; amb++) {
    srcConfig.ambipolar = (bool) amb;
    for (int twoT = 0; twoT < 2; twoT++) {
      srcConfig.twoTemperature = (bool) twoT;
      bool success = true;
      for (int trial = 0; trial < nTrials; trial++)
      {
        for (int sp = 1; sp <= srcConfig.numSpecies; sp++) {
          srcConfig.constantTransport.diffusivity[sp - 1] = uniformRandomNumber();
          srcConfig.constantTransport.mtFreq[sp - 1] = uniformRandomNumber();
        }

        srcConfig.axisymmetric_ = false;

        success = (success && testComputeBdrViscousFlux(srcConfig, dim));
        if (!success) break;

        srcConfig.axisymmetric_ = true;

        success = (success && testComputeBdrViscousFlux(srcConfig, dim));
        if (!success) break;

        for (int sp = 0; sp < srcConfig.numSpecies; sp++) srcConfig.constantTransport.diffusivity[sp] = 0.0;

        srcConfig.axisymmetric_ = false;

        success = (success && testComputeBdrViscousFlux(srcConfig, dim));
        if (!success) break;

        srcConfig.axisymmetric_ = true;

        success = (success && testComputeBdrViscousFlux(srcConfig, dim));
        if (!success) break;
      }
      if (success) {
        grvy_printf(GRVY_INFO, "\nPASS: ambipolar: %s, two-temperature: %s\n",
                    std::string((srcConfig.ambipolar) ? "T" : "F").c_str(),
                    std::string((srcConfig.twoTemperature) ? "T" : "F").c_str());
      } else {
        grvy_printf(GRVY_ERROR, "\nFAIL: ambipolar: %s, two-temperature: %s\n",
                    std::string((srcConfig.ambipolar) ? "T" : "F").c_str(),
                    std::string((srcConfig.twoTemperature) ? "T" : "F").c_str());
        exit(-1);
      }

    }
  }
  return 0;
}
