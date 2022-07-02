/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

double uniformRandomNumber() {
  return (double) rand() / (RAND_MAX);
}

double getRandomPrimitiveState(PerfectMixture *mixture, Vector &primitiveState) {
  const double numEquation = mixture->GetNumEquations();
  const double numSpecies = mixture->GetNumSpecies();
  const double numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool ambipolar = mixture->IsAmbipolar();
  const bool twoTemperature = mixture->IsTwoTemperature();
  const double dim = mixture->GetDimension();

  primitiveState.SetSize(numEquation);
  // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
  for (int d = 0; d < dim; d++) {
    primitiveState(d + 1) = -0.5 + 1.0 * uniformRandomNumber();
  }
  primitiveState(dim + 1) = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
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

  for (int sp = 0; sp < numActiveSpecies; sp++) primitiveState(dim + 2 + sp) = n_sp(sp);

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

int main (int argc, char *argv[])
{
  TPS::Tps tps;
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  srand (time(NULL));
  //double dummy = uniformRandomNumber();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);

  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::USER_DEFINED);
  assert(srcConfig.GetGasModel()==PERFECT_MIXTURE);
  assert(!srcConfig.axisymmetric_);

  int dim = 3;

  double scalarErrorThreshold = 1e-13;
  double gradErrorThreshold = 5e-7;
  double pressGradAbsThreshold = 1e-8;
  int nTrials = 10;

  /*
    Test non-ambipolar, single-temperature fluid
  */
  for (int trial = 0; trial < nTrials; trial++)
  {
    srcConfig.ambipolar = false;
    srcConfig.twoTemperature = false;
    srcField->packUpGasMixtureInput();
    PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, dim);

    grvy_printf(GRVY_INFO, "\n Setting a random primitive variable. \n");

    Vector testPrimitives(mixture->GetNumEquations());
    double Yb = getRandomPrimitiveState(mixture, testPrimitives);
    // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // for (int d = 0; d < dim; d++) {
    //   testPrimitives[d + 1] = -0.5 + 1.0 * uniformRandomNumber();
    // }
    // testPrimitives[dim + 1] = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
    //
    // double Yb = uniformRandomNumber();
    // Vector Ysp(mixture->GetNumActiveSpecies());
    // double Ysum = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   Ysp[sp] = uniformRandomNumber();
    //   Ysum += Ysp[sp];
    // }
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   double Y = Ysp[sp] / Ysum * (1.0 - Yb);
    //   testPrimitives[dim + 2 + sp] = testPrimitives[0] * Y / mixture->GetGasParams(sp,GasParams::SPECIES_MW);
    // }
    // for (int n = 0; n < mixture->GetNumEquations(); n++){
    //   std::cout << testPrimitives[n] << ",";
    // }
    // std::cout << std::endl;

    grvy_printf(GRVY_INFO, "\n Testing Conversion between conserved/primitive. \n");

    Vector conservedState(mixture->GetNumEquations());
    mixture->GetConservativesFromPrimitives(testPrimitives,conservedState);

    // for (int n = 0; n < mixture->GetNumEquations(); n++){
    //   std::cout << conservedState[n] << ",";
    // }
    // std::cout << std::endl;

    Vector targetPrimitives(mixture->GetNumEquations());
    mixture->GetPrimitivesFromConservatives(conservedState, targetPrimitives);

    // for (int n = 0; n < mixture->GetNumEquations(); n++){
    //   std::cout << targetPrimitives[n] << ",";
    // }
    // std::cout << std::endl;

    double error = 0.0;
    for (int n = 0; n < mixture->GetNumEquations(); n++){
      error += abs( ( testPrimitives[n] - targetPrimitives[n] ) / testPrimitives[n] );
    }

    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Conversions between conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E\n", error);
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << testPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] - testPrimitives[n] << ",";
     }
     std::cout << std::endl;
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputePressure. \n");

    double pressureFromConserved = mixture->ComputePressure(conservedState);
    double pressureFromPrimitive = mixture->ComputePressureFromPrimitives(testPrimitives);
    error = abs( (pressureFromConserved - pressureFromPrimitive) / pressureFromConserved );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Pressure computations from conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      grvy_printf(GRVY_ERROR, "\n %.15E =/= %.15E \n", pressureFromConserved, pressureFromPrimitive );
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputeTemperature. \n");

    double temperature = mixture->ComputeTemperature(conservedState);
    error = abs( (temperature - testPrimitives[dim + 1]) / testPrimitives[dim + 1] );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeTemperature is not equivalent to the reference.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing computeSpeciesPrimitives. \n");

    int numSpecies = mixture->GetNumSpecies();
    int numEquation = mixture->GetNumEquations();
    Vector X_sp(numSpecies), Y_sp(numSpecies), n_sp(numSpecies);
    mixture->computeSpeciesPrimitives(conservedState, X_sp, Y_sp, n_sp);
    double Xsum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Xsum += X_sp[sp];
    }
    if (abs(Xsum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mole fraction.");
      exit(ERROR);
    }
    double Ysum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Ysum += Y_sp[sp];
    }
    if (abs(Ysum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mass fraction.");
      exit(ERROR);
    }
    if (abs(Yb - Y_sp[mixture->GetNumSpecies() - 1]) > 1.0e-15) {
      double error = abs(Yb - Y_sp[mixture->GetNumSpecies() - 1]);
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not compute background species properly. Error: %.8E\n", error);
      grvy_printf(GRVY_ERROR, "\n Yb: %.8E\n", Yb);
      grvy_printf(GRVY_ERROR, "\n Ysp: %.8E\n", Y_sp[numSpecies - 1]);
      exit(ERROR);
    }
    for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
      if (abs(n_sp[sp] - testPrimitives[dim + 2 + sp]) > 1.0e-15) {
        grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives is not consistent in number density computation.");
        exit(ERROR);
      }
    }

    grvy_printf(GRVY_INFO, "\n Setting a random primitive gradient. \n");

    DenseMatrix gradUp(numEquation,dim);
    gradUp = 0.0;
    for (int eq = 0; eq < numEquation; eq++) {
      for (int d = 0; d < dim; d++) gradUp(eq, d) = -10.0 + 20.0 * uniformRandomNumber();
    }
    DenseMatrix massFractionGrad(numSpecies,dim);
    DenseMatrix moleFractionGrad(numSpecies,dim);
    mixture->ComputeMassFractionGradient(testPrimitives[0], n_sp, gradUp, massFractionGrad);
    mixture->ComputeMoleFractionGradient(n_sp, gradUp, moleFractionGrad);
    Vector dir(dim);
    Vector velocity(dim);
    for (int d = 0; d < dim; d++) velocity(d) = testPrimitives(d + 1);
    getRandomDirection(velocity, dir);
    // double norm = 0.0;
    // for (int d = 0; d < dim; d++) {
    //   dir(d) = -1.0 + 2.0 * uniformRandomNumber();
    //   norm += dir(d) * dir(d);
    // }
    // dir /= sqrt(norm);

    Vector dUp_dx(numEquation), dY_dx(numSpecies), dX_dx(numSpecies);
    gradUp.Mult(dir, dUp_dx);
    massFractionGrad.Mult(dir, dY_dx);

    moleFractionGrad.Mult(dir, dX_dx);
    double dpdx_prim = mixture->ComputePressureDerivative(dUp_dx, testPrimitives, true);
    double dpdx_cons = mixture->ComputePressureDerivative(dUp_dx, conservedState, false);
    double rel_error = abs( (dpdx_prim - dpdx_cons) / dpdx_cons );
    double abs_error = abs( (dpdx_prim - dpdx_cons) );
    grvy_printf(GRVY_INFO, "\n %.15E =/= %.15E, error: %.8E \n", dpdx_cons, dpdx_prim, rel_error );
    if ((rel_error > scalarErrorThreshold) && (abs_error > pressGradAbsThreshold)) {
      grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not consistent between primitive and conservative.");
      exit(ERROR);
    }

    { // FD verification of pressure gradient.
      grvy_printf(GRVY_INFO, "\n Testing ComputePressureDerivative. \n");

      double delta0 = 1e-3 * testPrimitives(0) / abs(dUp_dx(0));
      for (int eq = 1; eq < numEquation; eq++) delta0 = min(delta0, 0.1 * testPrimitives(0) / abs(dUp_dx(0)));
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta0 * dUp_dx(eq);

      double perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
      double deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta0;

      error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        double delta = delta0 * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);
        perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
        deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta;
        error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMassFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaYdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaYdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      // std::cout << error1 << std::endl;
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMassFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMoleFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaXdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaXdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMoleFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    {
      grvy_printf(GRVY_INFO, "Conserved state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", conservedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      Equations eqSystem = NS;
      TransportProperties *transportPtr = NULL;
      Fluxes *flux = new Fluxes(mixture, eqSystem, transportPtr, numEquation, dim, false);
      grvy_printf(GRVY_INFO, "Initialized flux class. \n");

      DenseMatrix convectiveFlux(numEquation, dim);
      flux->ComputeConvectiveFluxes(conservedState, convectiveFlux);
      grvy_printf(GRVY_INFO, "Computed convective flux. \n");

      Vector normalFlux(numEquation);
      convectiveFlux.Mult(dir, normalFlux);
      grvy_printf(GRVY_INFO, "Computed normal convective flux. \n");

      Vector deducedState(numEquation);
      mixture->computeConservedStateFromConvectiveFlux(normalFlux, dir, deducedState);
      grvy_printf(GRVY_INFO, "Computed conserved state back from normal convective flux. \n");


      grvy_printf(GRVY_INFO, "Pressure: %.15E\n", pressureFromConserved);

      // grvy_printf(GRVY_INFO, "Deducing outside the routine, using known pressure. \n");
      double normalVelocity = 0.0;
      for (int d = 0; d < dim; d++) normalVelocity += testPrimitives(d + 1) * dir(d);
      grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", normalVelocity);
      //
      // double bulkKinEnergy = 0.0;
      // for (int d = 0; d < dim; d++) bulkKinEnergy += testPrimitives(d + 1) * testPrimitives(d + 1);
      // bulkKinEnergy *= 0.5 * testPrimitives(0);
      //
      // double temp = 0.;
      // for (int d = 0; d < dim; d++) temp += normalFlux[1 + d] * dir[d];
      //
      // double A = 1.0;
      // double B = -2.0 * temp;
      // double C = -2.0 * normalFlux[0] * bulkKinEnergy * normalVelocity;
      // for (int d = 0; d < dim; d++) C += normalFlux[d + 1] * normalFlux[d + 1];
      // double p = (-B + sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed with bulk kin energy: %.15E \n", p);
      // p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed with bulk kin energy: %.15E \n", p);
      // grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", (temp - p) / normalFlux[0]);
      //
      // Vector numberDensityFluxes(mixture->GetNumActiveSpecies());
      // double nBFlux = normalFlux(0); // background species number density flux.
      // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++) {
      //   numberDensityFluxes(sp) = normalFlux(dim + 2 + sp) / mixture->GetGasParams(sp, GasParams::SPECIES_MW);
      //   nBFlux -= normalFlux(dim + 2 + sp);
      // }
      // nBFlux /= mixture->GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
      // double nMix = 0.0, cpMix = 0.0;
      // for (int sp = 0; sp < numSpecies - 1; sp++) {
      //   nMix += numberDensityFluxes(sp);
      //   cpMix += numberDensityFluxes(sp) * mixture->getMolarCP(sp);
      // }
      // nMix += nBFlux;
      // cpMix += nBFlux * mixture->getMolarCP(numSpecies - 1);
      // cpMix /= nMix;
      //
      // A = 1.0;
      // B = -2.0 * temp;
      // C = -2.0 * normalFlux[0] * normalFlux[dim + 1];
      // for (int d = 0; d < dim; d++) C += normalFlux[d + 1] * normalFlux[d + 1];
      // C += 2.0 * normalFlux[0] * (nMix * cpMix * testPrimitives(dim + 1));
      // p = (-B + sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed with temps: %.15E \n", p);
      // p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed with temps: %.15E \n", p);
      // grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", (temp - p) / normalFlux[0]);
      //
      // A = 1.0 - 2.0 * cpMix / UNIVERSALGASCONSTANT;
      // B = 2.0 * temp * (cpMix / UNIVERSALGASCONSTANT - 1.0);
      // C = -2.0 * normalFlux[0] * normalFlux[dim + 1];
      // for (int d = 0; d < dim; d++) C += normalFlux[d + 1] * normalFlux[d + 1];
      // p = (-B + sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed: %.15E \n", p);
      // p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);
      // grvy_printf(GRVY_INFO, "pressured computed: %.15E \n", p);
      // grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", (temp - p) / normalFlux[0]);


      grvy_printf(GRVY_INFO, "Deduced state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", deducedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      double error = 0.0;
      for (int n = 0; n < numEquation; n++){
        error = max(error, abs(( conservedState[n] - deducedState[n] ) / conservedState[n]));
      }
      grvy_printf(GRVY_INFO, "Relative error: %.8E\n", error);
      if ( error > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeConservedStateFromConvectiveFlux is not computing the correct state.");
        exit(ERROR);
      }
    }

  }
  grvy_printf(GRVY_INFO, "\nPASS: non-ambipolar, single temperature.\n");

  /*
    Test ambipolar, single-temperature fluid
  */
  for (int trial = 0; trial < nTrials; trial++)
  {
    srcConfig.ambipolar = true;
    srcConfig.twoTemperature = false;
    srcField->packUpGasMixtureInput();
    PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, dim);

    grvy_printf(GRVY_INFO, "\n Setting a random primitive variable. \n");

    Vector testPrimitives(mixture->GetNumEquations());
    double Yb = getRandomPrimitiveState(mixture, testPrimitives);
    // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // for (int d = 0; d < dim; d++) {
    //   testPrimitives[d + 1] = -0.5 + 1.0 * uniformRandomNumber();
    // }
    // testPrimitives[dim + 1] = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
    //
    // double Yb = 0.5 + 0.5 * uniformRandomNumber();
    // Vector Ysp(mixture->GetNumActiveSpecies());
    // Vector nsp(mixture->GetNumActiveSpecies());
    // double Ysum = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   Ysp[sp] = uniformRandomNumber();
    //   Ysum += Ysp[sp];
    // }
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   double Y = Ysp[sp] / Ysum * (1.0 - Yb);
    //   nsp[sp] = testPrimitives[0] * Y / mixture->GetGasParams(sp,GasParams::SPECIES_MW);
    //   testPrimitives[dim + 2 + sp] = nsp[sp];
    // }
    // double ne = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   ne += nsp[sp] * mixture->GetGasParams(sp,GasParams::SPECIES_CHARGES);
    // }
    // Yb -= ne * mixture->GetGasParams(mixture->GetNumSpecies() - 2, GasParams::SPECIES_MW) / testPrimitives[0];

    grvy_printf(GRVY_INFO, "\n Testing Conversion between conserved/primitive. \n");

    Vector conservedState(mixture->GetNumEquations());
    mixture->GetConservativesFromPrimitives(testPrimitives,conservedState);

    Vector targetPrimitives(mixture->GetNumEquations());
    mixture->GetPrimitivesFromConservatives(conservedState, targetPrimitives);

    double error = 0.0;
    for (int n = 0; n < mixture->GetNumEquations(); n++){
      error += abs( ( testPrimitives[n] - targetPrimitives[n] ) / testPrimitives[n] );
    }

    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Conversions between conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E\n", error);
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << testPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] - testPrimitives[n] << ",";
     }
     std::cout << std::endl;
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputePressure. \n");

    double pressureFromConserved = mixture->ComputePressure(conservedState);
    double pressureFromPrimitive = mixture->ComputePressureFromPrimitives(testPrimitives);
    error = abs( (pressureFromConserved - pressureFromPrimitive) / pressureFromConserved );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Pressure computations from conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      grvy_printf(GRVY_ERROR, "\n %.15E =/= %.15E \n", pressureFromConserved, pressureFromPrimitive );
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputeTemperature. \n");

    double temperature = mixture->ComputeTemperature(conservedState);
    error = abs( (temperature - testPrimitives[dim + 1]) / testPrimitives[dim + 1] );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeTemperature is not equivalent to the reference.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing computeSpeciesPrimitives. \n");

    int numSpecies = mixture->GetNumSpecies();
    int numEquation = mixture->GetNumEquations();
    Vector X_sp(numSpecies), Y_sp(numSpecies), n_sp(numSpecies);
    mixture->computeSpeciesPrimitives(conservedState, X_sp, Y_sp, n_sp);
    double Xsum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Xsum += X_sp[sp];
    }
    if (abs(Xsum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mole fraction.");
      exit(ERROR);
    }
    double Ysum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Ysum += Y_sp[sp];
    }
    if (abs(Ysum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mass fraction.");
      exit(ERROR);
    }
    if (abs(Yb - Y_sp[mixture->GetNumSpecies() - 1]) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not compute background species properly.");
      exit(ERROR);
    }

    double netCharge = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      netCharge += n_sp[sp] * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);

      if (sp < mixture->GetNumActiveSpecies()) {
        if (abs(n_sp[sp] - testPrimitives[dim + 2 + sp]) > 1.0e-15) {
          grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives is not consistent in number density computation.");
          exit(ERROR);
        }
      }
    }
    if (abs(netCharge) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not keep charge conservation.");
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Setting a random primitive gradient. \n");

    DenseMatrix gradUp(numEquation,dim);
    gradUp = 0.0;
    for (int eq = 0; eq < numEquation; eq++) {
      for (int d = 0; d < dim; d++) gradUp(eq, d) = -10.0 + 20.0 * uniformRandomNumber();
    }
    DenseMatrix massFractionGrad(numSpecies,dim);
    DenseMatrix moleFractionGrad(numSpecies,dim);
    mixture->ComputeMassFractionGradient(testPrimitives[0], n_sp, gradUp, massFractionGrad);
    mixture->ComputeMoleFractionGradient(n_sp, gradUp, moleFractionGrad);
    Vector dir(dim);
    Vector velocity(dim);
    for (int d = 0; d < dim; d++) velocity(d) = testPrimitives(d + 1);
    getRandomDirection(velocity, dir);
    // double norm = 0.0;
    // for (int d = 0; d < dim; d++) {
    //   dir(d) = -1.0 + 2.0 * uniformRandomNumber();
    //   norm += dir(d) * dir(d);
    // }
    // dir /= sqrt(norm);

    Vector dUp_dx(numEquation), dY_dx(numSpecies), dX_dx(numSpecies);
    gradUp.Mult(dir, dUp_dx);
    massFractionGrad.Mult(dir, dY_dx);
    moleFractionGrad.Mult(dir, dX_dx);
    double dpdx_prim = mixture->ComputePressureDerivative(dUp_dx, testPrimitives, true);
    double dpdx_cons = mixture->ComputePressureDerivative(dUp_dx, conservedState, false);
    double rel_error = abs( (dpdx_prim - dpdx_cons) / dpdx_cons );
    double abs_error = abs( (dpdx_prim - dpdx_cons) );
    grvy_printf(GRVY_INFO, "\n %.15E =/= %.15E, error: %.8E \n", dpdx_cons, dpdx_prim, rel_error );
    if ((rel_error > scalarErrorThreshold) && (abs_error > pressGradAbsThreshold)) {
      grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not consistent between primitive and conservative.");
      exit(ERROR);
    }

    { // FD verification of pressure gradient.
      grvy_printf(GRVY_INFO, "\n Testing ComputePressureDerivative. \n");

      double delta0 = 1e-3 * testPrimitives(0) / abs(dUp_dx(0));
      for (int eq = 1; eq < numEquation; eq++) delta0 = min(delta0, 0.1 * testPrimitives(0) / abs(dUp_dx(0)));
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta0 * dUp_dx(eq);

      double perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
      double deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta0;

      error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        double delta = delta0 * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);
        perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
        deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta;
        error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMassFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaYdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaYdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      // std::cout << error1 << std::endl;
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMassFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMoleFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaXdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaXdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMoleFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    {
      grvy_printf(GRVY_INFO, "Primitive state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", testPrimitives(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      grvy_printf(GRVY_INFO, "Conserved state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", conservedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      Equations eqSystem = NS;
      TransportProperties *transportPtr = NULL;
      Fluxes *flux = new Fluxes(mixture, eqSystem, transportPtr, numEquation, dim, false);
      grvy_printf(GRVY_INFO, "Initialized flux class. \n");

      DenseMatrix convectiveFlux(numEquation, dim);
      flux->ComputeConvectiveFluxes(conservedState, convectiveFlux);
      grvy_printf(GRVY_INFO, "Computed convective flux. \n");

      Vector normalFlux(numEquation);
      convectiveFlux.Mult(dir, normalFlux);
      grvy_printf(GRVY_INFO, "Computed normal convective flux. \n");

      double normalVelocity = 0.0;
      for (int d = 0; d < dim; d++) normalVelocity += testPrimitives(d + 1) * dir(d);
      grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", normalVelocity);

      Vector deducedState(numEquation);
      mixture->computeConservedStateFromConvectiveFlux(normalFlux, dir, deducedState);
      grvy_printf(GRVY_INFO, "Computed conserved state back from normal convective flux. \n");

      grvy_printf(GRVY_INFO, "Deduced state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", deducedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      double error = 0.0;
      for (int n = 0; n < numEquation; n++){
        error = max(error, abs(( conservedState[n] - deducedState[n] ) / conservedState[n]));
      }
      grvy_printf(GRVY_INFO, "Relative error: %.8E\n", error);
      if ( error > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeConservedStateFromConvectiveFlux is not computing the correct state.");
        exit(ERROR);
      }
    }
  }
  grvy_printf(GRVY_INFO, "\nPASS: ambipolar, single temperature.\n");

  /*
    Test non-ambipolar, two-temperature fluid
  */
  for (int trial = 0; trial < nTrials; trial++)
  {
    srcConfig.ambipolar = false;
    srcConfig.twoTemperature = true;
    srcField->packUpGasMixtureInput();
    PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, dim);

    grvy_printf(GRVY_INFO, "\n Setting a random primitive variable. \n");

    Vector testPrimitives(mixture->GetNumEquations());
    double Yb = getRandomPrimitiveState(mixture, testPrimitives);
    // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // for (int d = 0; d < dim; d++) {
    //   testPrimitives[d + 1] = -0.5 + 1.0 * uniformRandomNumber();
    // }
    // testPrimitives[dim + 1] = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // testPrimitives[mixture->GetNumEquations() - 1] = 500.0 * ( 0.8 + 0.4 * uniformRandomNumber() );
    //
    // double Yb = uniformRandomNumber();
    // Vector Ysp(mixture->GetNumActiveSpecies());
    // Vector nsp(mixture->GetNumActiveSpecies());
    // double Ysum = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   Ysp[sp] = uniformRandomNumber();
    //   Ysum += Ysp[sp];
    // }
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   double Y = Ysp[sp] / Ysum * (1.0 - Yb);
    //   nsp[sp] = testPrimitives[0] * Y / mixture->GetGasParams(sp,GasParams::SPECIES_MW);
    //   testPrimitives[dim + 2 + sp] = nsp[sp];
    // }

    grvy_printf(GRVY_INFO, "\n Testing Conversion between conserved/primitive. \n");

    Vector conservedState(mixture->GetNumEquations());
    mixture->GetConservativesFromPrimitives(testPrimitives,conservedState);

    Vector targetPrimitives(mixture->GetNumEquations());
    mixture->GetPrimitivesFromConservatives(conservedState, targetPrimitives);

    double error = 0.0;
    for (int n = 0; n < mixture->GetNumEquations(); n++){
      error += abs( ( testPrimitives[n] - targetPrimitives[n] ) / testPrimitives[n] );
    }

    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Conversions between conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E\n", error);
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << testPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] - testPrimitives[n] << ",";
     }
     std::cout << std::endl;
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputePressure. \n");

    double pressureFromConserved = mixture->ComputePressure(conservedState);
    double pressureFromPrimitive = mixture->ComputePressureFromPrimitives(testPrimitives);
    error = abs( (pressureFromConserved - pressureFromPrimitive) / pressureFromConserved );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Pressure computations from conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      grvy_printf(GRVY_ERROR, "\n %.15E =/= %.15E \n", pressureFromConserved, pressureFromPrimitive );
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputeTemperature. \n");

    double temperature = mixture->ComputeTemperature(conservedState);
    error = abs( (temperature - testPrimitives[dim + 1]) / testPrimitives[dim + 1] );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeTemperature is not equivalent to the reference.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing computeSpeciesPrimitives. \n");

    int numSpecies = mixture->GetNumSpecies();
    int numEquation = mixture->GetNumEquations();
    Vector X_sp(numSpecies), Y_sp(numSpecies), n_sp(numSpecies);
    mixture->computeSpeciesPrimitives(conservedState, X_sp, Y_sp, n_sp);
    double Xsum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Xsum += X_sp[sp];
    }
    if (abs(Xsum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mole fraction.");
      exit(ERROR);
    }
    double Ysum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Ysum += Y_sp[sp];
    }
    if (abs(Ysum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mass fraction.");
      exit(ERROR);
    }
    if (abs(Yb - Y_sp[mixture->GetNumSpecies() - 1]) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not compute background species properly.");
      exit(ERROR);
    }

    for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
      if (abs(n_sp[sp] - testPrimitives[dim + 2 + sp]) > 1.0e-15) {
        grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives is not consistent in number density computation.");
        exit(ERROR);
      }
    }

    double T_h, T_e;
    mixture->computeTemperaturesBase(conservedState, &n_sp[0], n_sp[numSpecies-2], n_sp[numSpecies-1], T_h, T_e);
    if (abs( (T_h - testPrimitives[dim + 1]) / testPrimitives[dim + 1] ) > scalarErrorThreshold) {
      std::cout << T_h << " != " << testPrimitives[dim + 1] << std::endl;
      grvy_printf(GRVY_ERROR, "\n computeTemperaturesBase does not compute heavy-species temperature properly.");
      exit(ERROR);
    }
    if (abs( (T_e - testPrimitives[numEquation - 1]) / testPrimitives[numEquation - 1] ) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeTemperaturesBase does not compute electron temperature properly.");
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Setting a random primitive gradient. \n");

    DenseMatrix gradUp(numEquation,dim);
    gradUp = 0.0;
    for (int eq = 0; eq < numEquation; eq++) {
      for (int d = 0; d < dim; d++) gradUp(eq, d) = -10.0 + 20.0 * uniformRandomNumber();
    }
    DenseMatrix massFractionGrad(numSpecies,dim);
    DenseMatrix moleFractionGrad(numSpecies,dim);
    mixture->ComputeMassFractionGradient(testPrimitives[0], n_sp, gradUp, massFractionGrad);
    mixture->ComputeMoleFractionGradient(n_sp, gradUp, moleFractionGrad);
    Vector dir(dim);
    Vector velocity(dim);
    for (int d = 0; d < dim; d++) velocity(d) = testPrimitives(d + 1);
    getRandomDirection(velocity, dir);
    // double norm = 0.0;
    // for (int d = 0; d < dim; d++) {
    //   dir(d) = -1.0 + 2.0 * uniformRandomNumber();
    //   norm += dir(d) * dir(d);
    // }
    // dir /= sqrt(norm);

    Vector dUp_dx(numEquation), dY_dx(numSpecies), dX_dx(numSpecies);
    gradUp.Mult(dir, dUp_dx);
    massFractionGrad.Mult(dir, dY_dx);
    moleFractionGrad.Mult(dir, dX_dx);
    double dpdx_prim = mixture->ComputePressureDerivative(dUp_dx, testPrimitives, true);
    double dpdx_cons = mixture->ComputePressureDerivative(dUp_dx, conservedState, false);
    double rel_error = abs( (dpdx_prim - dpdx_cons) / dpdx_cons );
    double abs_error = abs( (dpdx_prim - dpdx_cons) );
    grvy_printf(GRVY_INFO, "\n %.15E =/= %.15E, error: %.8E \n", dpdx_cons, dpdx_prim, rel_error );
    if ((rel_error > scalarErrorThreshold) && (abs_error > pressGradAbsThreshold)) {
      grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not consistent between primitive and conservative.");
      exit(ERROR);
    }

    { // FD verification of pressure gradient.
      grvy_printf(GRVY_INFO, "\n Testing ComputePressureDerivative. \n");

      double delta0 = 1e-3 * testPrimitives(0) / abs(dUp_dx(0));
      for (int eq = 1; eq < numEquation; eq++) delta0 = min(delta0, 0.1 * testPrimitives(0) / abs(dUp_dx(0)));
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta0 * dUp_dx(eq);

      double perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
      double deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta0;

      error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        double delta = delta0 * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);
        perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
        deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta;
        error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMassFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaYdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaYdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      // std::cout << error1 << std::endl;
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMassFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMoleFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaXdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaXdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMoleFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    {
      std::cout << "two temperature?: " << mixture->IsTwoTemperature() << std::endl;
      grvy_printf(GRVY_INFO, "Primitive state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", testPrimitives(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      grvy_printf(GRVY_INFO, "Conserved state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", conservedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      grvy_printf(GRVY_INFO, "Pressure: %.15E \n", pressureFromConserved);

      double normalVelocity = 0.0;
      for (int d = 0; d < dim; d++) normalVelocity += testPrimitives(d + 1) * dir(d);
      grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", normalVelocity);

      Equations eqSystem = NS;
      TransportProperties *transportPtr = NULL;
      Fluxes *flux = new Fluxes(mixture, eqSystem, transportPtr, numEquation, dim, false);
      grvy_printf(GRVY_INFO, "Initialized flux class. \n");

      DenseMatrix convectiveFlux(numEquation, dim);
      flux->ComputeConvectiveFluxes(conservedState, convectiveFlux);
      grvy_printf(GRVY_INFO, "Computed convective flux. \n");

      Vector normalFlux(numEquation);
      convectiveFlux.Mult(dir, normalFlux);
      grvy_printf(GRVY_INFO, "Computed normal convective flux. \n");

      Vector deducedState(numEquation);
      mixture->computeConservedStateFromConvectiveFlux(normalFlux, dir, deducedState);
      grvy_printf(GRVY_INFO, "Computed conserved state back from normal convective flux. \n");
      grvy_printf(GRVY_INFO, "Deduced state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", deducedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      double error = 0.0;
      for (int n = 0; n < numEquation; n++){
        error = max(error, abs(( conservedState[n] - deducedState[n] ) / conservedState[n]));
      }
      grvy_printf(GRVY_INFO, "Relative error: %.8E\n", error);
      if ( error > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeConservedStateFromConvectiveFlux is not computing the correct state.");
        exit(ERROR);
      }
    }

  }
  grvy_printf(GRVY_INFO, "\nPASS: non-ambipolar, two temperature.\n");

  /*
    Test ambipolar, two-temperature fluid
  */
  for (int trial = 0; trial < nTrials; trial++)
  {
    srcConfig.ambipolar = true;
    srcConfig.twoTemperature = true;
    srcField->packUpGasMixtureInput();
    PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, dim);

    grvy_printf(GRVY_INFO, "\n Setting a random primitive variable. \n");

    Vector testPrimitives(mixture->GetNumEquations());
    double Yb = getRandomPrimitiveState(mixture, testPrimitives);
    // testPrimitives[0] = 1.784 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // for (int d = 0; d < dim; d++) {
    //   testPrimitives[d + 1] = -0.5 + 1.0 * uniformRandomNumber();
    // }
    // testPrimitives[dim + 1] = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
    // testPrimitives[mixture->GetNumEquations() - 1] = 500.0 * ( 0.8 + 0.4 * uniformRandomNumber() );
    //
    // double Yb = 0.5 + 0.5 * uniformRandomNumber();
    // Vector Ysp(mixture->GetNumActiveSpecies());
    // Vector nsp(mixture->GetNumActiveSpecies());
    // double Ysum = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   Ysp[sp] = uniformRandomNumber();
    //   Ysum += Ysp[sp];
    // }
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   double Y = Ysp[sp] / Ysum * (1.0 - Yb);
    //   nsp[sp] = testPrimitives[0] * Y / mixture->GetGasParams(sp,GasParams::SPECIES_MW);
    //   testPrimitives[dim + 2 + sp] = nsp[sp];
    // }
    // double ne = 0.0;
    // for (int sp = 0; sp < mixture->GetNumActiveSpecies(); sp++){
    //   ne += nsp[sp] * mixture->GetGasParams(sp,GasParams::SPECIES_CHARGES);
    // }
    // Yb -= ne * mixture->GetGasParams(mixture->GetNumSpecies() - 2, GasParams::SPECIES_MW) / testPrimitives[0];

    grvy_printf(GRVY_INFO, "\n Testing Conversion between conserved/primitive. \n");

    Vector conservedState(mixture->GetNumEquations());
    mixture->GetConservativesFromPrimitives(testPrimitives,conservedState);

    Vector targetPrimitives(mixture->GetNumEquations());
    mixture->GetPrimitivesFromConservatives(conservedState, targetPrimitives);

    double error = 0.0;
    for (int n = 0; n < mixture->GetNumEquations(); n++){
      error += abs( ( testPrimitives[n] - targetPrimitives[n] ) / testPrimitives[n] );
    }

    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Conversions between conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E\n", error);
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << testPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] << ",";
     }
     std::cout << std::endl;
     for (int n = 0; n < mixture->GetNumEquations(); n++){
       std::cout << targetPrimitives[n] - testPrimitives[n] << ",";
     }
     std::cout << std::endl;
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputePressure. \n");

    double pressureFromConserved = mixture->ComputePressure(conservedState);
    double pressureFromPrimitive = mixture->ComputePressureFromPrimitives(testPrimitives);
    error = abs( (pressureFromConserved - pressureFromPrimitive) / pressureFromConserved );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n Pressure computations from conserved and primitive variables are not equivalent.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      grvy_printf(GRVY_ERROR, "\n %.15E =/= %.15E \n", pressureFromConserved, pressureFromPrimitive );
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing ComputeTemperature. \n");

    double temperature = mixture->ComputeTemperature(conservedState);
    error = abs( (temperature - testPrimitives[dim + 1]) / testPrimitives[dim + 1] );
    if (error > scalarErrorThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeTemperature is not equivalent to the reference.");
      grvy_printf(GRVY_ERROR, "\n Error: %.15E", error);
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Testing computeSpeciesPrimitives. \n");

    int numSpecies = mixture->GetNumSpecies();
    int numEquation = mixture->GetNumEquations();
    Vector X_sp(numSpecies), Y_sp(numSpecies), n_sp(numSpecies);
    mixture->computeSpeciesPrimitives(conservedState, X_sp, Y_sp, n_sp);
    double Xsum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Xsum += X_sp[sp];
    }
    if (abs(Xsum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mole fraction.");
      exit(ERROR);
    }
    double Ysum = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      Ysum += Y_sp[sp];
    }
    if (abs(Ysum - 1.0) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not conserve mass fraction.");
      exit(ERROR);
    }
    if (abs(Yb - Y_sp[mixture->GetNumSpecies() - 1]) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not compute background species properly.");
      exit(ERROR);
    }

    double netCharge = 0.0;
    for (int sp = 0; sp < mixture->GetNumSpecies(); sp++){
      netCharge += n_sp[sp] * mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);

      if (sp < mixture->GetNumActiveSpecies()) {
        if (abs(n_sp[sp] - testPrimitives[dim + 2 + sp]) > 1.0e-15) {
          grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives is not consistent in number density computation.");
          exit(ERROR);
        }
      }
    }
    if (abs(netCharge) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeSpeciesPrimitives does not keep charge conservation.");
      exit(ERROR);
    }

    double T_h, T_e;
    mixture->computeTemperaturesBase(conservedState, &n_sp[0], n_sp[numSpecies-2], n_sp[numSpecies-1], T_h, T_e);
    if (abs( (T_h - testPrimitives[dim + 1]) / testPrimitives[dim + 1] ) > scalarErrorThreshold) {
      std::cout << T_h << " != " << testPrimitives[dim + 1] << std::endl;
      grvy_printf(GRVY_ERROR, "\n computeTemperaturesBase does not compute heavy-species temperature properly.");
      exit(ERROR);
    }
    if (abs( (T_e - testPrimitives[numEquation - 1]) / testPrimitives[numEquation - 1] ) > 1.0e-15) {
      grvy_printf(GRVY_ERROR, "\n computeTemperaturesBase does not compute electron temperature properly.");
      exit(ERROR);
    }

    grvy_printf(GRVY_INFO, "\n Setting a random primitive gradient. \n");

    DenseMatrix gradUp(numEquation,dim);
    gradUp = 0.0;
    for (int eq = 0; eq < numEquation; eq++) {
      for (int d = 0; d < dim; d++) gradUp(eq, d) = -10.0 + 20.0 * uniformRandomNumber();
    }
    DenseMatrix massFractionGrad(numSpecies,dim);
    DenseMatrix moleFractionGrad(numSpecies,dim);
    mixture->ComputeMassFractionGradient(testPrimitives[0], n_sp, gradUp, massFractionGrad);
    mixture->ComputeMoleFractionGradient(n_sp, gradUp, moleFractionGrad);
    Vector dir(dim);
    Vector velocity(dim);
    for (int d = 0; d < dim; d++) velocity(d) = testPrimitives(d + 1);
    getRandomDirection(velocity, dir);
    // double norm = 0.0;
    // for (int d = 0; d < dim; d++) {
    //   dir(d) = -1.0 + 2.0 * uniformRandomNumber();
    //   norm += dir(d) * dir(d);
    // }
    // dir /= sqrt(norm);

    Vector dUp_dx(numEquation), dY_dx(numSpecies), dX_dx(numSpecies);
    gradUp.Mult(dir, dUp_dx);
    massFractionGrad.Mult(dir, dY_dx);
    moleFractionGrad.Mult(dir, dX_dx);
    double dpdx_prim = mixture->ComputePressureDerivative(dUp_dx, testPrimitives, true);
    double dpdx_cons = mixture->ComputePressureDerivative(dUp_dx, conservedState, false);
    double rel_error = abs( (dpdx_prim - dpdx_cons) / dpdx_cons );
    double abs_error = abs( (dpdx_prim - dpdx_cons) );
    grvy_printf(GRVY_INFO, "\n %.15E =/= %.15E, error: %.8E \n", dpdx_cons, dpdx_prim, rel_error );
    if ((rel_error > scalarErrorThreshold) && (abs_error > pressGradAbsThreshold)) {
      grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not consistent between primitive and conservative.");
      exit(ERROR);
    }

    { // FD verification of pressure gradient.
      grvy_printf(GRVY_INFO, "\n Testing ComputePressureDerivative. \n");

      double delta0 = 1e-3 * testPrimitives(0) / abs(dUp_dx(0));
      for (int eq = 1; eq < numEquation; eq++) delta0 = min(delta0, 0.1 * testPrimitives(0) / abs(dUp_dx(0)));
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta0 * dUp_dx(eq);

      double perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
      double deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta0;

      error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        double delta = delta0 * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);
        perturbedPressure = mixture->ComputePressureFromPrimitives(perturbedPrimitive);
        deltaPdeltax = (perturbedPressure - pressureFromPrimitive) / delta;
        error = abs( (deltaPdeltax - dpdx_prim) / dpdx_prim );

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n ComputePressureDerivative is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMassFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaYdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaYdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaYdeltax(sp) = (perturbedY(sp) - Y_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaYdeltax(sp) - dY_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      // std::cout << error1 << std::endl;
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMassFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    { // FD verification of mass fraction gradient.
      grvy_printf(GRVY_INFO, "\n Testing computeMoleFractionGradient. \n");

      double gradNorm = sqrt( dY_dx * dY_dx );
      double delta = 1.0e-3 / gradNorm;
      Vector perturbedPrimitive(numEquation);
      for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

      Vector perturbedConserved(numEquation);
      mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

      Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
      mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

      Vector deltaXdeltax(numSpecies);
      for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

      error = 0.0;
      for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
      error = sqrt(error) / gradNorm;
      double error1 = error;
      for (int fd = 1; fd < 30; fd++) {
        error1 = error;
        double factor = pow(10.0, (- 0.25 * fd));
        delta = 1.0e-3 / gradNorm * factor;
        for (int eq = 0; eq < numEquation; eq++) perturbedPrimitive(eq) = testPrimitives(eq) + delta * dUp_dx(eq);

        Vector perturbedConserved(numEquation);
        mixture->GetConservativesFromPrimitives(perturbedPrimitive, perturbedConserved);

        Vector perturbedX(numSpecies), perturbedY(numSpecies), perturbedN(numSpecies);
        mixture->computeSpeciesPrimitives(perturbedConserved, perturbedX, perturbedY, perturbedN);

        Vector deltaXdeltax(numSpecies);
        for (int sp = 0; sp < numSpecies; sp++) deltaXdeltax(sp) = (perturbedX(sp) - X_sp(sp)) / delta;

        error = 0.0;
        for (int sp = 0; sp < numSpecies; sp++) error += pow(deltaXdeltax(sp) - dX_dx(sp), 2);
        error = sqrt(error) / gradNorm;

        grvy_printf(GRVY_INFO, "\n Delta : %.8E - FD-Error : %.8E", factor, error);

        if (error > error1) break;
      }
      grvy_printf(GRVY_INFO, "\n");
      if ( error1 > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeMoleFractionGradient is not computing the correct gradient.");
        exit(ERROR);
      }
    }

    {
      std::cout << "two temperature?: " << mixture->IsTwoTemperature() << std::endl;
      grvy_printf(GRVY_INFO, "Primitive state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", testPrimitives(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      grvy_printf(GRVY_INFO, "Conserved state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", conservedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      grvy_printf(GRVY_INFO, "Pressure: %.15E \n", pressureFromConserved);

      double normalVelocity = 0.0;
      for (int d = 0; d < dim; d++) normalVelocity += testPrimitives(d + 1) * dir(d);
      grvy_printf(GRVY_INFO, "Normal velocity: %.8E \n", normalVelocity);

      Equations eqSystem = NS;
      TransportProperties *transportPtr = NULL;
      Fluxes *flux = new Fluxes(mixture, eqSystem, transportPtr, numEquation, dim, false);
      grvy_printf(GRVY_INFO, "Initialized flux class. \n");

      DenseMatrix convectiveFlux(numEquation, dim);
      flux->ComputeConvectiveFluxes(conservedState, convectiveFlux);
      grvy_printf(GRVY_INFO, "Computed convective flux. \n");

      Vector normalFlux(numEquation);
      convectiveFlux.Mult(dir, normalFlux);
      grvy_printf(GRVY_INFO, "Computed normal convective flux. \n");

      Vector deducedState(numEquation);
      mixture->computeConservedStateFromConvectiveFlux(normalFlux, dir, deducedState);
      grvy_printf(GRVY_INFO, "Computed conserved state back from normal convective flux. \n");
      grvy_printf(GRVY_INFO, "Deduced state \n");
      for (int eq = 0; eq < numEquation; eq++) {
        grvy_printf(GRVY_INFO, "%.8E, ", deducedState(eq));
      }
      grvy_printf(GRVY_INFO, "\n");

      double error = 0.0;
      for (int n = 0; n < numEquation; n++){
        error = max(error, abs(( conservedState[n] - deducedState[n] ) / conservedState[n]));
      }
      grvy_printf(GRVY_INFO, "Relative error: %.8E\n", error);
      if ( error > gradErrorThreshold ) {
        grvy_printf(GRVY_ERROR, "\n computeConservedStateFromConvectiveFlux is not computing the correct state.");
        exit(ERROR);
      }
    }

  }
  grvy_printf(GRVY_INFO, "\nPASS: ambipolar, two temperature.\n");

  return 0;
}
