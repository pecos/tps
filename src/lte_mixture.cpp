// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-
/** @file
 * @copydoc lte_mixture.hpp
 */

#include "lte_mixture.hpp"

#ifndef _GPU_
LteMixture::LteMixture(RunConfiguration &_runfile, int _dim, int nvel)
    : GasMixture(_runfile.lteMixtureInput.f, _dim, nvel, _runfile.const_plasma_conductivity_) {
  numSpecies = 1;
  ambipolar = false;
  twoTemperature_ = false;

  SetNumActiveSpecies();
  SetNumEquations();
  SetSpeciesStateIndices();
#ifdef HAVE_GSL
  energy_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                             1,                                            /* density column */
                                             3 /* energy column */);
  R_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                        1,                                            /* density column */
                                        6 /* mixture gas constant column */);

  c_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                        1,                                            /* density column */
                                        8 /* speed of sound column */);

  T_table_ = new GslTableInterpolator2D(_runfile.lteMixtureInput.e_rev_file_name, /* (energy,density) -> temp data */
                                        0,                                        /* energy column */
                                        1,                                        /* density column */
                                        2,                                        /* temperature column */
                                        3);                                       /* number of columns */
#else
  energy_table_ = NULL;
  R_table_ = NULL;
  T_table_ = NULL;
  mfem_error("2D LTE mixture tables require GSL support.");
#endif
}
#endif

#ifdef _GPU_
MFEM_HOST_DEVICE LteMixture::LteMixture(WorkingFluid f, int _dim, int nvel, double pc, TableInput energy_table_input,
                                        TableInput R_table_input, TableInput c_table_input, TableInput T_table_input)
    : GasMixture(f, _dim, nvel, pc),
      energy_table_(LinearTable(energy_table_input)),
      R_table_(LinearTable(R_table_input)),
      c_table_(LinearTable(c_table_input)),
      T_table_(LinearTable(T_table_input)) {
  numSpecies = 1;
  ambipolar = false;
  twoTemperature_ = false;

  SetNumActiveSpecies();
  SetNumEquations();
  SetSpeciesStateIndices();
}
#else
MFEM_HOST_DEVICE LteMixture::LteMixture(WorkingFluid f, int _dim, int nvel, double pc, TableInput energy_table_input,
                                        TableInput R_table_input, TableInput c_table_input, TableInput T_table_input)
    : GasMixture(f, _dim, nvel, pc) {
  numSpecies = 1;
  ambipolar = false;
  twoTemperature_ = false;

  SetNumActiveSpecies();
  SetNumEquations();
  SetSpeciesStateIndices();

  energy_table_ = new LinearTable(energy_table_input);
  R_table_ = new LinearTable(R_table_input);
  c_table_ = new LinearTable(c_table_input);
  T_table_ = new LinearTable(T_table_input);
}
#endif

MFEM_HOST_DEVICE LteMixture::~LteMixture() {
#ifndef _GPU_
  delete energy_table_;
  delete R_table_;
  delete c_table_;
  delete T_table_;
#endif
}

/// Compute pressure from conserved state
double LteMixture::ComputePressure(const Vector &state, double *electronPressure) {
  return ComputePressure(state.GetData(), electronPressure);
}

MFEM_HOST_DEVICE double LteMixture::ComputePressure(const double *state, double *electronPressure) {
  const double rho = state[0];
  // const double T = ComputeTemperature(state);
  double T;
  const bool success = ComputeTemperatureInternal(state, T);
  assert(success);
#ifdef _GPU_
  const double R = R_table_.eval(T, rho);
#else
  const double R = R_table_->eval(T, rho);
#endif
  return rho * R * T;
}

/// Compute pressure from primitive state
double LteMixture::ComputePressureFromPrimitives(const Vector &Up) {
  return ComputePressureFromPrimitives(Up.GetData());
}

MFEM_HOST_DEVICE double LteMixture::ComputePressureFromPrimitives(const double *Up) {
  const double rho = Up[0];
  const double T = Up[1 + nvel_];
#ifdef _GPU_
  const double R = R_table_.eval(T, rho);
#else
  const double R = R_table_->eval(T, rho);
#endif
  return rho * R * T;
}

/** \brief Compute temperature from conserved state
 *
 * This requires a nonlinear solve the nonlinear equation e(U) =
 * e(rho, T) for T, where e(U) is the internal energy computed from
 * the conserved state vector that is passed in and e(rho, T) is the
 * internal energy given by the thermodynamic equilibrium look-up
 * table.
 */
MFEM_HOST_DEVICE bool LteMixture::ComputeTemperatureInternal(const double *state, double &T) {
  const double rho = state[0];

  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) den_vel2 += state[d + 1] * state[d + 1];
  den_vel2 /= rho;

  const double energy = (state[1 + nvel_] - 0.5 * den_vel2) / rho;

#ifdef _GPU_
  T = T_table_.eval(energy, rho);
  double res = energy - energy_table_.eval(T, rho);
#else
  T = T_table_->eval(energy, rho);
  double res = energy - energy_table_->eval(T, rho);
#endif
  const double res0 = abs(res);

  bool converged = false;
  const double atol = 1e-18;
  const double rtol = 1e-12;
  const double dT_atol = 1e-12;
  const double dT_rtol = 1e-8;

  converged = ((abs(res) < atol) || (abs(res) / abs(res0) < rtol));

  const int niter_max = 20;
  int niter = 0;
  //printf("------------------------------------\n"); fflush(stdout);
  //printf("Iter %d: res = %.6e, T = %.6e\n", niter, res, T); fflush(stdout);

  // Newton loop
  while (!converged && (niter < niter_max)) {
    // get derivative of e wrt T
#ifdef _GPU_
    double dedT = energy_table_.eval_x(T, rho);
#else
    double dedT = energy_table_->eval_x(T, rho);
#endif

    // Newton step
    double dT = res / dedT;
    T += dT;
    // clip instead? assert(T > 0);
    //if (T <= 280.0) {
    //  printf("Iter %d: res = %.6e, T = %.6e\n", niter, res, T); fflush(stdout);
    //  std::cout << state[0] << " " << state[1] << " " << state[2] << " " <<state[3] << " " <<state[4] << endl;      
    //}

    // Update residual
#ifdef _GPU_
    res = energy - energy_table_.eval(T, rho);
#else
    res = energy - energy_table_->eval(T, rho);
#endif

    // Declare convergence if residual OR dT is sufficient small
    converged = ((abs(res) < atol) || (abs(res) / res0 < rtol) || (abs(dT) < dT_atol) || (abs(dT) / T < dT_rtol));
    niter++;
    //printf("Iter %d: res = %.6e, T = %.6e\n", niter, res, T); fflush(stdout);
  }

  // clip
  //if (!converged) {
  //  converged = true;
  //  T = 298.0;
  //}
  
  if (!converged) {
    return false;
  }
  return true;
}

double LteMixture::ComputeTemperature(const Vector &state) { return ComputeTemperature(state.GetData()); }

MFEM_HOST_DEVICE double LteMixture::ComputeTemperature(const double *state) {
  double T;
  const bool success = ComputeTemperatureInternal(state, T);
  assert(success);
  return T;
}

/** \brief Compute temperature from density and pressure
 *
 * This requires solving the equation p = rho * R(rho, T) * T for T,
 * where rho and p are the input density and pressure and R(rho, T) is
 * the mixture gas constant given by the thermodynamic equilibrium
 * look-up table.
 */
MFEM_HOST_DEVICE double LteMixture::ComputeTemperatureFromDensityPressure(const double rho, const double p) {
  double T;

  // TODO(trevilo): How should we set initial guess?
  T = p / (rho * 208.);

#ifdef _GPU_
  double R = R_table_.eval(T, rho);
#else
  double R = R_table_->eval(T, rho);
#endif
  double res = p - rho * R * T;
  const double res0 = abs(res);

  bool converged = false;
  const double atol = 1e-18;
  const double rtol = 1e-12;
  const double dT_atol = 1e-12;
  const double dT_rtol = 1e-8;

  converged = ((abs(res) < atol) || (abs(res) / abs(res0) < rtol));

  const int niter_max = 20;
  int niter = 0;
  // printf("------------------------------------\n"); fflush(stdout);
  // printf("Iter %d: res = %.6e, T = %.6e\n", niter, res, T); fflush(stdout);

  // Newton loop
  while (!converged && (niter < niter_max)) {
    // get derivative of p wrt T
#ifdef _GPU_
    double R_T = R_table_.eval_x(T, rho);
#else
    double R_T = R_table_->eval_x(T, rho);
#endif
    double dpdT = rho * R + rho * R_T * T;

    // Newton step
    double dT = res / dpdT;
    T += dT;
    //HERE HERE HERE assert(T > 0);

    // Update residual
#ifdef _GPU_
    R = R_table_.eval(T, rho);
#else
    R = R_table_->eval(T, rho);
#endif
    res = p - rho * R * T;

    // Declare convergence if residual OR dT is sufficient small
    converged = ((abs(res) < atol) || (abs(res) / res0 < rtol) || (abs(dT) < dT_atol) || (abs(dT) / T < dT_rtol));
    niter++;
    // printf("Iter %d: res = %.6e, T = %.6e\n", niter, res, T); fflush(stdout);
  }

  if (!converged) {
    printf("WARNING: temperature did not converge.");
  } else {
    // printf("Temperature converged to res = %.6e in %d steps\n", res, niter); fflush(stdout);
  }

  return T;
}

// TODO(trevilo): move this into the base class
void LteMixture::computeSpeciesEnthalpies(const Vector &state, Vector &speciesEnthalpies) {
  speciesEnthalpies.SetSize(numSpecies);
  speciesEnthalpies = 0.0;
  return;
}

/// Compute primitive variables (rho, u, T) from conserved (rho, rho*u, rho*E)
void LteMixture::GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit) {
  primit.SetSize(conserv.Size());
  GetPrimitivesFromConservatives(conserv.GetData(), primit.GetData());
}

MFEM_HOST_DEVICE void LteMixture::GetPrimitivesFromConservatives(const double *conserv, double *primit) {
  const double T = ComputeTemperature(conserv);

  for (int i = 0; i < num_equation; i++) {
    primit[i] = conserv[i];
  }

  for (int d = 0; d < nvel_; d++) primit[1 + d] /= conserv[0];

  primit[nvel_ + 1] = T;
}

/// Compute conserved variables (rho, rho*u, rho*E) from primitive (rho, u, T)
void LteMixture::GetConservativesFromPrimitives(const Vector &primit, Vector &conserv) {
  conserv.SetSize(primit.Size());
  GetConservativesFromPrimitives(primit.GetData(), conserv.GetData());
}

MFEM_HOST_DEVICE void LteMixture::GetConservativesFromPrimitives(const double *primit, double *conserv) {
  for (int i = 0; i < num_equation; i++) {
    conserv[i] = primit[i];
  }

  double v2 = 0.;
  for (int d = 0; d < nvel_; d++) {
    v2 += primit[1 + d] * primit[1 + d];
    conserv[1 + d] *= primit[0];
  }

  const double rho = primit[0];
  const double T = primit[1 + nvel_];
#ifdef _GPU_
  const double energy = energy_table_.eval(T, rho);
#else
  const double energy = energy_table_->eval(T, rho);
#endif

  // total energy
  conserv[1 + nvel_] = primit[0] * (energy + 0.5 * v2);
}

/// Compute the speed of sound (from look-up table)
double LteMixture::ComputeSpeedOfSound(const Vector &Uin, bool primitive) {
  return ComputeSpeedOfSound(Uin.GetData(), primitive);
}

/// Compute the speed of sound (from look-up table)
MFEM_HOST_DEVICE double LteMixture::ComputeSpeedOfSound(const double *Uin, bool primitive) {
  const double rho = Uin[0];
  double T;
  if (primitive) {
    T = Uin[1 + nvel_];
  } else {
    // T = ComputeTemperature(Uin);
    //if (Uin[4] > 1e12) {
    //  std::cout << "BAD ComputeSoS Uin:" << Uin[4] << endl;
    //}
    const bool success = ComputeTemperatureInternal(Uin, T);
    assert(success);

    // HACK HACK HACK
    //T = 298.15;
  }
#ifdef _GPU_
  return c_table_.eval(T, rho);
#else
  return c_table_->eval(T, rho);
#endif
}

/// Compute the maximum characteristic speed (u+a)
double LteMixture::ComputeMaxCharSpeed(const Vector &state) { return ComputeMaxCharSpeed(state.GetData()); }

/// Compute the maximum characteristic speed (u+a)
MFEM_HOST_DEVICE double LteMixture::ComputeMaxCharSpeed(const double *state) {
  const double den = state[0];

  double den_vel2 = 0;
  for (int d = 0; d < nvel_; d++) {
    den_vel2 += state[d + 1] * state[d + 1];
  }
  den_vel2 /= den;

  //std::cout << "ComputeSoS lte_mixture 1" << endl;
  const double sound = ComputeSpeedOfSound(state, false);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

// only used in non-reflecting BCs... don't implement for now
double LteMixture::ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive) {
  assert(false);
  return 0.0;
}

/// Check if the density, temperature, and pressure are positive
bool LteMixture::StateIsPhysical(const Vector &state) {
  const double rho = state[0];
  const double T = ComputeTemperature(state);
#ifdef _GPU_
  double R = R_table_.eval(T, rho);
#else
  double R = R_table_->eval(T, rho);
#endif
  const double p = rho * R * T;
  if (rho <= 0) return false;
  if (T <= 0) return false;
  if (p <= 0) return false;
  return true;
}

// BC related functions
void LteMixture::computeStagnantStateWithTemp(const Vector &stateIn, const double Temp, Vector &stateOut) {
  stateOut.SetSize(num_equation);
  computeStagnantStateWithTemp(stateIn.GetData(), Temp, stateOut.GetData());
}

MFEM_HOST_DEVICE void LteMixture::computeStagnantStateWithTemp(const double *stateIn, const double Temp,
                                                               double *stateOut) {
  for (int i = 0; i < num_equation; i++) {
    stateOut[i] = stateIn[i];
  }

  for (int d = 0; d < nvel_; d++) stateOut[1 + d] = 0.;

  const double rho = stateIn[0];
  const double T = Temp;
#ifdef _GPU_
  const double energy = energy_table_.eval(T, rho);
#else
  const double energy = energy_table_->eval(T, rho);
#endif
  stateOut[1 + nvel_] = rho * energy;
}

void LteMixture::modifyEnergyForPressure(const Vector &stateIn, Vector &stateOut, const double &p,
                                         bool modifyElectronEnergy) {
  modifyEnergyForPressure(stateIn.GetData(), stateOut.GetData(), p, modifyElectronEnergy);
}

MFEM_HOST_DEVICE void LteMixture::modifyEnergyForPressure(const double *stateIn, double *stateOut, const double &p,
                                                          bool modifyElectronEnergy) {
  for (int eq = 0; eq < num_equation; eq++) stateOut[eq] = stateIn[eq];

  const double rho = stateIn[0];
  double ke = 0.;
  for (int d = 0; d < nvel_; d++) ke += stateIn[1 + d] * stateIn[1 + d];
  ke *= 0.5 / rho;

  const double T = ComputeTemperatureFromDensityPressure(rho, p);
#ifdef _GPU_
  const double energy = energy_table_.eval(T, rho);
#else
  const double energy = energy_table_->eval(T, rho);
#endif

  stateOut[1 + nvel_] = rho * energy + ke;
}
