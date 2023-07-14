#ifndef FLOW_INITIAL_CONDITIONS_HPP_
#define FLOW_INITIAL_CONDITIONS_HPP_

#include <mfem.hpp>

void initializeEulerVortex(const mfem::Vector &x, mfem::Vector &y) {
  MFEM_ASSERT(x.Size() == 2, "");
  int equations = 4;

  int problem = 2;
  DryAir *eqState = new DryAir();
  const double gamma = eqState->GetSpecificHeatRatio();
  const double Rg = eqState->GetGasConstant();

  double radius = 0, Minf = 0, beta = 0;
  if (problem == 1) {
    // "Fast vortex"
    radius = 0.1;
    Minf = 0.5;
    beta = 1. / 5.;
  } else if (problem == 2) {
    // "Slow vortex"
    radius = 0.2;
    Minf = 0.05;
    beta = 1. / 50.;
  } else {
    mfem_error(
        "Cannot recognize problem."
        "Options are: 1 - fast vortex, 2 - slow vortex");
  }

  // int numVortices = 3;
  // Vector xc(numVortices), yc(numVortices);
  // yc = 0.;
  // for (int i = 0; i < numVortices; i++) {
  //   xc[i] = 2. * M_PI / static_cast<double>(numVortices + 1);
  //   xc[i] += static_cast<double>(i) * 2. * M_PI / static_cast<double>(numVortices);
  // }

  int numVortices = 1;
  Vector xc(1), yc(1);
  xc = yc = 0.;

  const double Cp = Rg * gamma / (gamma - 1);

  const double Tinf = 300.;
  const double Pinf = 100000.;
  const double rho_inf = Pinf / Rg / Tinf;

  const double ainf = std::sqrt(gamma * Rg * Tinf);
  const double Uinf = Minf * ainf;
  // std::cout << "Uinf = " << Uinf << std::endl;

  const double norm_rad_sq = ((x(0) - xc[0]) * (x(0) - xc[0]) + (x(1) - yc[0]) * (x(1) - yc[0])) / radius / radius;

  const double du = -Uinf * beta * ((x(1) - yc[0]) / radius) * std::exp(-0.5 * norm_rad_sq);
  const double dv = Uinf * beta * ((x(0) - xc[0]) / radius) * std::exp(-0.5 * norm_rad_sq);
  const double dT = 0.5 * (Uinf * beta) * (Uinf * beta) * std::exp(-norm_rad_sq) / Cp;

  const double velX = Uinf + du;
  const double velY = dv;
  const double T0 = Tinf - dT;
  const double rho0 = rho_inf * std::pow(T0 / Tinf, 1. / (gamma - 1));
  const double p0 = rho0 * Rg * T0;

  // const double Tt = 300.;
  // const double Pt = 102200;

  // const double funcGamma = 1. + 0.5 * (gamma - 1.) * Minf * Minf;

  // const double temp_inf = Tt / funcGamma;
  // const double pres_inf = Pt * pow(funcGamma, gamma / (gamma - 1.));
  // const double vel_inf = Minf * sqrt(gamma * Rg * temp_inf);
  // const double den_inf = pres_inf / (Rg * temp_inf);

  // std::cout << "vel_inf = " << vel_inf << std::endl;

  // double r2rad = 0.0;

  // const double shrinv1 = 1.0 / (gamma - 1.);

  // double velX = 0.;
  // double velY = 0.;
  // double temp = 0.;
  // for (int i = 0; i < numVortices; i++) {
  //   r2rad = (x(0) - xc[i]) * (x(0) - xc[i]);
  //   r2rad += (x(1) - yc[i]) * (x(1) - yc[i]);
  //   r2rad /= radius * radius;
  //   velX -= beta * (x(1) - yc[i]) / radius * exp(-0.5 * r2rad);
  //   velY += beta * (x(0) - xc[i]) / radius * exp(-0.5 * r2rad);
  //   temp += exp(-r2rad);
  // }

  // velX = vel_inf * (1 - velX);
  // velY = vel_inf * velY;
  const double vel2 = velX * velX + velY * velY;

  // const double specific_heat = Rg * gamma * shrinv1;
  // temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / specific_heat * temp;

  // const double den = den_inf * pow(temp / temp_inf, shrinv1);
  // const double pres = den * Rg * temp;
  // const double energy = shrinv1 * pres / den + 0.5 * vel2;

  const double den = rho0;
  const double energy = p0 / den / (gamma - 1) + 0.5 * vel2;

  y(0) = den;
  y(1) = den * velX;
  y(2) = den * velY;
  y(3) = den * energy;

  delete eqState;
}

#endif  // FLOW_INITIAL_CONDITIONS_HPP_
