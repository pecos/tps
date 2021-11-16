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

EquationOfState::EquationOfState() {}

void EquationOfState::setFluid(WorkingFluid _fluid) {
  fluid = _fluid;
  switch (fluid) {
    case DRY_AIR:
      gas_constant = 287.058;
      // gas_constant = 1.; // for comparison against ex18
      specific_heat_ratio = 1.4;
      visc_mult = 1.;
      Pr = 0.71;
      cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
      break;
    default:
      break;
  }
}

bool EquationOfState::StateIsPhysical(const mfem::Vector& state, const int dim) {
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
double EquationOfState::ComputeMaxCharSpeed(const Vector& state, const int dim) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state, dim);
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

void EquationOfState::GetConservativesFromPrimitives(const Vector& primit, Vector& conserv, const int& dim,
                                                     const int& num_equations) {
  conserv = primit;

  double v2 = 0.;
  for (int d = 0; d < dim; d++) {
    v2 += primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  conserv[num_equations - 1] = primit[num_equations - 1] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;
}

void EquationOfState::GetPrimitivesFromConservatives(const Vector& conserv, Vector& primit, const int& dim,
                                                     const int& num_equations) {
  double p = ComputePressure(conserv, dim);
  primit = conserv;

  for (int d = 0; d < dim; d++) primit[1 + d] /= conserv[0];

  primit[num_equations - 1] = p;
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
