#include <fstream>

#include "dataStructures.hpp"
#include "run_configuration.hpp"
#include "lte_mixture.hpp"
#include "lte_transport_properties.hpp"

using namespace mfem;
using namespace std;

int checkPressureFromPrimitives(LteMixture *lte_mix, double rtol) {
  int ierr = 0;
  Vector Up(5);
  double p, pexact;

  // rho = 0.005, T = 2000
  Up[0] = 0.005; Up[4] = 2000.0;
  pexact = Up[0] * 208.1321372 * Up[4];
  p = lte_mix->ComputePressureFromPrimitives(Up);
  if (std::abs(p - pexact)/pexact > rtol) {
    std::cout << "Incorrect pressure at rho = " << Up[0] << ", T = " << Up[4] << std::endl;
    ierr += 1;
  }

  // rho = 0.255, T = 12000
  Up[0] = 0.255; Up[4] = 12000.0;
  pexact = Up[0] * 217.82066155 * Up[4];
  p = lte_mix->ComputePressureFromPrimitives(Up);
  if (std::abs(p - pexact)/pexact > rtol) {
    std::cout << "Incorrect pressure at rho = " << Up[0] << ", T = " << Up[4] << std::endl;
    ierr += 1;
  }

  return ierr;
}

int checkTemperature(LteMixture *lte_mix, double rtol) {
  int ierr = 0;

  // test temperature calc from conserved state
  {
    Vector U(5);
    double rho, energy;
    double T, Texact;

    U = 0.;

    // rho = 0.005, T = 2000, no KE
    rho = 0.005;
    Texact = 2000.;
    energy = lte_mix->evaluateInternalEnergy(Texact, rho);
    U[0] = rho; U[4] = rho*energy;
    T = lte_mix->ComputeTemperature(U);
    if (std::abs(T - Texact)/Texact > rtol) {
      std::cout << "Incorrect temperature at rho = " << U[0] << ", rho*E = " << U[4] << std::endl;
      ierr += 1;
    }

    // rho = 0.255, T = 12000, no KE
    rho = 0.255;
    Texact = 12000.;
    energy = lte_mix->evaluateInternalEnergy(Texact, rho);
    U[0] = rho; U[4] = rho*energy;
    T = lte_mix->ComputeTemperature(U);
    if (std::abs(T - Texact)/Texact > rtol) {
      std::cout << "Incorrect temperature at rho = " << U[0] << ", rho*E = " << U[4] << std::endl;
      ierr += 1;
    }

    // rho = 0.255, T = 12000, some KE
    rho = 0.255;
    Texact = 12000.;
    energy = lte_mix->evaluateInternalEnergy(Texact, rho);
    U[0] = rho; U[1] = 0.5; U[2] = 10.0; U[3] = -0.5;
    U[4] = rho*energy + 0.5*(U[1]*U[1] + U[2]*U[2] + U[3]*U[3])/rho;
    T = lte_mix->ComputeTemperature(U);
    if (std::abs(T - Texact)/Texact > rtol) {
      std::cout << "Incorrect temperature at rho = " << U[0] << ", rho*E = " << U[4] << std::endl;
      ierr += 1;
    }
  }

  // test temperature calc from rho, p
  {
    double R, rho, p;
    double T, Texact;

    // rho = 0.005, T = 2000
    rho = 0.005; Texact = 2000.;
    R = lte_mix->evaluateGasConstant(Texact, rho);
    p = rho * R * Texact;
    T = lte_mix->ComputeTemperatureFromDensityPressure(rho, p);
    if (std::abs(T - Texact)/Texact > rtol) {
      std::cout << "Incorrect temperature at rho = " << rho << ", p = " << p << std::endl;
      ierr += 1;
    }

    // rho = 0.005, T = 2000
    rho = 0.255; Texact = 12000.;
    R = lte_mix->evaluateGasConstant(Texact, rho);
    p = rho * R * Texact;
    T = lte_mix->ComputeTemperatureFromDensityPressure(rho, p);
    if (std::abs(T - Texact)/Texact > rtol) {
      std::cout << "Incorrect temperature at rho = " << rho << ", p = " << p << std::endl;
      ierr += 1;
    }
  }

  return ierr;
}

int checkTransport(LteTransport *lte_trans, LteMixture *lte_mix, double rtol) {
  int ierr = 0;

  // low T case
  {
    double rho = 1.225; // transport props for this test do not actually depend on density
    double T = 350.0;
    Vector Up(5);
    Up = 0.;
    Up[0] = rho;
    Up[4] = T;

    Vector U;
    lte_mix->GetConservativesFromPrimitives(Up, U);

    DenseMatrix gradUp(5,3);
    Vector E(3);

    Vector transport;
    DenseMatrix diffusion;

    lte_trans->ComputeFluxTransportProperties(U, gradUp, E, -1, transport, diffusion);

    const double muexact = 2.0656339881365003e-05;
    const double mu = transport[FluxTrns::VISCOSITY];

    if (std::abs(mu - muexact)/muexact > rtol) {
      std::cout << "Incorrect viscosity at rho = " << rho << ", T = " << T << std::endl;
      ierr += 1;
    }
  }

  return ierr;
}

int main (int argc, char *argv[]) {
  int ierr = 0;
  double rtol = 10. * std::numeric_limits<double>::epsilon();

  // Need a minimal configuration
  RunConfiguration config;
  config.lteMixtureInput.f = LTE_FLUID;
  config.lteMixtureInput.thermo_file_name = "./inputs/argon_lte_thermo_table.dat";
  config.lteMixtureInput.e_rev_file_name = "./inputs/argon_lte_e_rev_table.dat";

  // NB: This transport file is inconsistent with the mixture thermo
  // file.  It is ok here b/c the tests don't require consistency, but
  // obviously any real simulation should be run with consistent
  // transport and thermo
  config.lteMixtureInput.trans_file_name = "./inputs/air_simple_transport_table.dat";

  //-------------------------------------------------------------------
  // Spot check mixture class
  //-------------------------------------------------------------------
  LteMixture *lte_mix = new LteMixture(config, 3, 3);

  // check that we can compute pressure from rho and T
  ierr += checkPressureFromPrimitives(lte_mix, rtol);

  // check that we can compute temperature from conserved state
  ierr += checkTemperature(lte_mix, rtol);

  //-------------------------------------------------------------------
  // Spot check transport class
  //-------------------------------------------------------------------
  LteTransport *lte_trans = new LteTransport(lte_mix, config);

  // check that we can compute transport
  ierr += checkTransport(lte_trans, lte_mix, rtol);

  // clean up
  delete lte_trans;
  delete lte_mix;

  return ierr;
}
