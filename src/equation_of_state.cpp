// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

#include "equation_of_state.hpp"

// EquationOfState::EquationOfState() {}

GasMixture::GasMixture(WorkingFluid _fluid, int _dim) {
  fluid = _fluid;
  dim = _dim;

  bulk_visc_mult = 0.;
  visc_mult = 1;
}

DryAir::DryAir(RunConfiguration& _runfile, int _dim) : GasMixture(WorkingFluid::DRY_AIR, _dim) {
  setNumEquations();

  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;

  Pr = 0.71;
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
  Sc = 0.71;

  // add extra equation for passive scalar
  if (_runfile.GetEquationSystem() == Equations::NS_PASSIVE) {
    Nconservative++;
    Nprimitive++;
    num_equations++;
  }
}

DryAir::DryAir() {
  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;
  visc_mult = 1.;
  Pr = 0.71;
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
  Sc = 0.71;
}

DryAir::DryAir(int _dim, int _num_equation) {
  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;
  visc_mult = 1.;
  Pr = 0.71;
  cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
  Sc = 0.71;

  dim = _dim;
  num_equations = _num_equation;
}

void DryAir::setNumEquations() {
  Nconservative = dim + 2;
  Nprimitive = Nconservative;
  num_equations = Nconservative;
}

// void EquationOfState::setFluid(WorkingFluid _fluid) {
//   fluid = _fluid;
//   switch (fluid) {
//     case DRY_AIR:
//       gas_constant = 287.058;
//       // gas_constant = 1.; // for comparison against ex18
//       specific_heat_ratio = 1.4;
//       visc_mult = 1.;
//       Pr = 0.71;
//       cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
//       Sc = 0.71;
//       break;
//     default:
//       break;
//   }
// }

bool DryAir::StateIsPhysical(const mfem::Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  if (den < 0) {
    cout << "Negative density: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  if (den_energy <= 0) {
    cout << "Negative energy: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }

  double den_vel2 = 0;
  for (int i = 0; i < dim; i++) {
    den_vel2 += den_vel(i) * den_vel(i);
  }
  den_vel2 /= den;

  const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

  if (pres <= 0) {
    cout << "Negative pressure: " << pres << ", state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  return true;
}

// Compute the maximum characteristic speed.
double DryAir::ComputeMaxCharSpeed(const Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state);
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

void DryAir::GetConservativesFromPrimitives(const Vector& primit, Vector& conserv) {
  conserv = primit;

  double v2 = 0.;
  for (int d = 0; d < dim; d++) {
    v2 += primit[1 + d] * primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  // total energy
  conserv[1 + dim] = gas_constant * primit[0] * primit[1 + dim] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;

  // case of passive scalar
  if (num_equations > dim + 2) {
    for (int n = 0; n < num_equations - dim - 2; n++) {
      conserv[dim + 2 + n] *= primit[0];
    }
  }
}

void DryAir::GetPrimitivesFromConservatives(const Vector& conserv, Vector& primit) {
  double T = ComputeTemperature(conserv);
  primit = conserv;

  for (int d = 0; d < dim; d++) primit[1 + d] /= conserv[0];

  primit[dim + 1] = T;

  // case of passive scalar
  if (num_equations > dim + 2) {
    for (int n = 0; n < num_equations - dim - 2; n++) {
      primit[dim + 2 + n] /= primit[0];
    }
  }
}

double DryAir::ComputeSpeedOfSound(const mfem::Vector& Uin, bool primitive) {
  double T;

  if (primitive) {
    T = Uin[1 + dim];
  } else {
    // conservatives passed in
    T = ComputeTemperature(Uin);
  }

  return sqrt(specific_heat_ratio * gas_constant * T);
}

double DryAir::ComputePressureDerivative(const Vector& dUp_dx, const Vector& Uin, bool primitive) {
  double T, p;
  if (primitive) {
    T = Uin[1 + dim];
  } else {
    T = ComputeTemperature(Uin);
  }

  return gas_constant * (T * dUp_dx[0] + Uin[0] * dUp_dx[1 + dim]);
}

double DryAir::ComputePressureFromPrimitives(const mfem::Vector& Up) { return gas_constant * Up[0] * Up[1 + dim]; }

void DryAir::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up) {
  double* pGridFunc = press->HostWrite();
  const double* UpData = Up->HostRead();

  const int nnode = press->FESpace()->GetNDofs();

  for (int n = 0; n < nnode; n++) {
    double tmp = UpData[n + (1 + dim) * nnode];
    double rho = UpData[n];
    pGridFunc[n] = rho * gas_constant * tmp;
  }
}

void DryAir::computeStagnationState(const mfem::Vector& stateIn, mfem::Vector& stagnationState) {
  const double p = ComputePressure(stateIn);

  stagnationState.SetSize(num_equations);
  stagnationState = stateIn;

  // zero momentum
  for (int d = 0; d < dim; d++) stagnationState(1 + d) = 0.;

  // total energy
  stagnationState(1 + dim) = p / (specific_heat_ratio - 1.);
}

void DryAir::computeStagnantStateWithTemp(const mfem::Vector& stateIn, const double Temp, mfem::Vector& stateOut) {
  stateOut.SetSize(num_equations);
  stateOut = stateIn;

  for (int d = 0; d < dim; d++) stateOut(1 + d) = 0.;

  stateOut(1 + dim) = gas_constant / (specific_heat_ratio - 1.) * stateIn(0) * Temp;
}

void DryAir::modifyEnergyForPressure(const mfem::Vector& stateIn, mfem::Vector& stateOut, const double& p)
{
  stateOut.SetSize(num_equations);
  stateOut = stateIn;
  
  double ke = 0.;
  for (int d = 0; d < dim; d++) ke += stateIn(1 + d) * stateIn(1 + d);
  ke *= 0.5 / stateIn(0);
  
  stateOut(1 + dim) = p / (specific_heat_ratio - 1.) + ke;
}

// GPU FUNCTIONS
/*#ifdef _GPU_
double EquationOfState::pressure( double *state,
                                  double *KE,
                                  const double gamma,
                                  const int dim,
                                  const int num_equations)
{
  double p = 0.;
  for(int k=0;k<dim;k++) p += KE[k];
  return (gamma-1.)*(state[num_equations-1] - p);
}
#endif*/
