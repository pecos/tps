#include <fstream>

#include "dataStructures.hpp"
#include "run_configuration.hpp"
#include "lte_mixture.hpp"

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

int main (int argc, char *argv[]) {
  int ierr = 0;
  double rtol = 10. * std::numeric_limits<double>::epsilon();

  RunConfiguration config;
  config.lteMixtureInput.f = LTE_FLUID;
  config.lteMixtureInput.thermo_file_name = "./inputs/argon_lte_thermo_table.dat";

  LteMixture *lte_mix = new LteMixture(config, 3, 3);

  // check that we can compute pressure from rho and T
  ierr += checkPressureFromPrimitives(lte_mix, rtol);

  // check that we can compute temperature from conserved state
  ierr += checkTemperature(lte_mix, rtol);

  delete lte_mix;
  return ierr;
}
