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

#include "riemann_solver.hpp"

using namespace mfem;

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(int &_num_equation, GasMixture *_mixture, Equations &_eqSystem, Fluxes *_fluxClass,
                             bool _useRoe, bool axisym)
    : num_equation(_num_equation),
      mixture(_mixture),
      eqSystem(_eqSystem),
      fluxClass(_fluxClass),
      useRoe(_useRoe),
      axisymmetric_(axisym) {
  flux1.SetSize(num_equation);
  flux2.SetSize(num_equation);
}

// Compute the scalar F(u).n
void RiemannSolver::ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &fluxN) {
  // NOTE: nor in general is not a unit normal
  const int dim = nor.Size();
  const int nvel = (axisymmetric_ ? 3 : dim);

  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, nvel);
  const double den_energy = state(1 + nvel);

  // MFEM_ASSERT(eqState->StateIsPhysical(state, dim), "");
  const double pres = mixture->ComputePressure(state);

  double den_velN = 0;
  for (int d = 0; d < dim; d++) {
    den_velN += den_vel(d) * nor(d);
  }

  fluxN(0) = den_velN;
  for (int d = 0; d < nvel; d++) {
    fluxN(1 + d) = den_velN * den_vel(d) / den;
  }
  for (int d = 0; d < dim; d++) {
    fluxN(1 + d) += pres * nor(d);
  }

  const double H = (den_energy + pres) / den;
  fluxN(1 + nvel) = den_velN * H;

  if (eqSystem == NS_PASSIVE) fluxN(num_equation - 1) = den_velN * state(num_equation - 1) / state(0);
}

// Compute the Jacobian of F(u).n
void RiemannSolver::ComputeFluxDotNJacobian(const Vector &state, const Vector &nor, DenseMatrix &fluxN_state) {
  fluxN_state.SetSize(num_equation, num_equation);

  // NOTE: nor in general is not a unit normal
  const int dim = nor.Size();
  const int nvel = (axisymmetric_ ? 3 : dim);

  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, nvel);
  const double den_energy = state(1 + nvel);

  // MFEM_ASSERT(eqState->StateIsPhysical(state, dim), "");
  const double pres = mixture->ComputePressure(state);

  // First, evaluate the derivative of the pressure with respect to the conserved state.
  // NB: this code is only valid for a perfect gas, and really doesn't belong here anyway.
  // TODO(trevilo): Generalize beyond perfect gas and move to mixture class.
  const double gam = mixture->GetSpecificHeatRatio();
  const double gm1 = gam - 1;
  Vector p_U(num_equation);
  p_U[0] = 0;
  for (int d = 0; d < nvel; d++) {
    p_U[0] += (den_vel(d)/den)*(den_vel(d)/den);
  }
  p_U[0] *= 0.5*gm1;

  for (int d = 0; d < nvel; d++) {
    p_U[d+1] = -gm1*den_vel(d)/den;
  }

  p_U[nvel+1] = gm1;

  // Second compute derivative of rho*u_n (normal velocity) wrt conserved state
  Vector den_velN_U(num_equation);
  den_velN_U = 0.0;
  for (int d = 0; d < dim; d++) {
    den_velN_U[d+1] = nor(d);
  }

  // Third, Jacobian of total enthalpy
  const double H = (den_energy + pres) / den;
  Vector H_U(num_equation);
  H_U[0] = -H/den + p_U[0]/den;
  for (int d = 0; d < nvel; d++) {
    H_U[1 + d] = p_U[1 + d] / den;
  }
  H_U[1 + nvel] = (1. + p_U[1 + nvel])/den;


  // Fourth, compute Jacobian, using results from above
  double den_velN = 0;
  for (int d = 0; d < dim; d++) {
    den_velN += den_vel(d) * nor(d);
  }


  // Cons of mass equation
  for (int jeqn = 0; jeqn < num_equation; jeqn++) {
    fluxN_state(0,jeqn) = den_velN_U[jeqn];
  }

  // Cons of momentum eqns

  // velocity part
  for (int d = 0; d < nvel; d++) {
    fluxN_state(1 + d, 0) = -(den_vel(d)/den)*(den_velN/den);

    for (int k = 0; k < nvel; k++) {
      fluxN_state(1 + d, 1 + k) = (den_vel(d)/den)*den_velN_U[1+k];
    }
    fluxN_state(1 + d, 1 + d) += den_velN/den;
    fluxN_state(1 + d, 1 + nvel) = 0;
  }

  // pressure part
  for (int d = 0; d < dim; d++) {
    fluxN_state(1 + d, 0) += p_U[0]*nor(d);

    for (int k = 0; k < nvel; k++) {
      fluxN_state(1 + d, 1 + k) += p_U[1+k]*nor(d);
    }

    fluxN_state(1 + d, 1 + nvel) += p_U[1 + nvel]*nor(d);
  }

  // cons of energy
  for (int k = 0; k < num_equation; k++) {
    fluxN_state(1 + nvel, k) = den_velN_U[k] * H + den_velN * H_U[k];
  }

  assert(eqSystem != NS_PASSIVE);
}

void RiemannSolver::Eval(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux, bool LF) {
  if (useRoe && !LF) {
    Eval_Roe(state1, state2, nor, flux);
  } else {
    Eval_LF(state1, state2, nor, flux);
  }
}

void RiemannSolver::Jacobian(const Vector &state1, const Vector &state2, const Vector &nor,
                             DenseMatrix &flux_state1, DenseMatrix &flux_state2, bool LF) {
  if (useRoe && !LF) {
    assert(false);
  } else {
    Jacobian_LF(state1, state2, nor, flux_state1, flux_state2);
  }
}

void RiemannSolver::Eval_LF(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux) {
  // NOTE: nor in general is not a unit normal
  const int dim = nor.Size();
  const int nvel = (axisymmetric_ ? 3 : dim);

  // MFEM_ASSERT(eqState->StateIsPhysical(state1, dim), "");
  // MFEM_ASSERT(eqState->StateIsPhysical(state2, dim), "");

  const double maxE1 = mixture->ComputeMaxCharSpeed(state1);
  const double maxE2 = mixture->ComputeMaxCharSpeed(state2);

  const double maxE = max(maxE1, maxE2);

  ComputeFluxDotN(state1, nor, flux1);
  ComputeFluxDotN(state2, nor, flux2);

  double normag = 0;
  for (int i = 0; i < dim; i++) {
    normag += nor(i) * nor(i);
  }
  normag = sqrt(normag);

  for (int i = 0; i < num_equation; i++) {
    flux(i) = 0.5 * (flux1(i) + flux2(i)) - 0.5 * maxE * (state2(i) - state1(i)) * normag;
  }
}

void RiemannSolver::Jacobian_LF(const Vector &state1, const Vector &state2, const Vector &nor,
                                DenseMatrix &flux_state1, DenseMatrix &flux_state2) {
  flux_state1.SetSize(num_equation, num_equation);
  flux_state2.SetSize(num_equation, num_equation);

  // NOTE: nor in general is not a unit normal
  const int dim = nor.Size();
  const int nvel = (axisymmetric_ ? 3 : dim);

  const double maxE1 = mixture->ComputeMaxCharSpeed(state1);
  const double maxE2 = mixture->ComputeMaxCharSpeed(state2);

  const double maxE = max(maxE1, maxE2);

  ComputeFluxDotNJacobian(state1, nor, flux1_state1);
  ComputeFluxDotNJacobian(state2, nor, flux2_state2);

  double normag = 0;
  for (int i = 0; i < dim; i++) {
    normag += nor(i) * nor(i);
  }
  normag = sqrt(normag);

  flux_state1.Set(0.5, flux1_state1);
  flux_state2.Set(0.5, flux2_state2);

  // NB: The following neglects the derivative of maxE wrt the state
  for (int i = 0; i < num_equation; i++) {
    //flux(i) = 0.5 * (flux1(i) + flux2(i)) - 0.5 * maxE * (state2(i) - state1(i)) * normag;
    flux_state1(i, i) += 0.5 * maxE * normag;
    flux_state2(i, i) -= 0.5 * maxE * normag;
  }
}

void RiemannSolver::Eval_Roe(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux) {
  const int dim = nor.Size();
  assert(!axisymmetric_);  // Roe doesn't support axisymmetric yet

  int NS_eq = 2 + dim;  // number of NS equations (without species, passive scalars etc.)

  double normag = 0;
  for (int i = 0; i < dim; i++) normag += nor(i) * nor(i);
  normag = sqrt(normag);

  Vector unitN(dim);
  for (int d = 0; d < dim; d++) unitN[d] = nor[d] / normag;

  DenseMatrix fluxes1(num_equation, dim), fluxes2(num_equation, dim);
  fluxClass->ComputeConvectiveFluxes(state1, fluxes1);
  fluxClass->ComputeConvectiveFluxes(state2, fluxes2);
  Vector meanFlux(num_equation);
  for (int eq = 0; eq < NS_eq; eq++) {
    meanFlux[eq] = 0.;
    for (int d = 0; d < dim; d++) meanFlux[eq] += (fluxes1(eq, d) + fluxes2(eq, d)) * unitN[d];
  }

  // Roe (Lohner)
  double r = sqrt(state1[0] * state2[0]);
  Vector vel(dim);
  for (int i = 0; i < dim; i++) {
    vel[i] = state1[i + 1] / sqrt(state1[0]) + state2[i + 1] / sqrt(state2[0]);
    vel[i] /= sqrt(state1[0]) + sqrt(state2[0]);
  }
  double qk = 0.;
  for (int d = 0; d < dim; d++) qk += vel[d] * unitN[d];

  double p1 = mixture->ComputePressure(state1);
  double p2 = mixture->ComputePressure(state2);
  double H = (state1[1 + dim] + p1) / sqrt(state1[0]) + (state2[1 + dim] + p2) / sqrt(state2[0]);
  H /= sqrt(state1[0]) + sqrt(state2[0]);
  double a2 = 0.4 * (H - 0.5 * (vel[0] * vel[0] + vel[1] * vel[1]));
  double a = sqrt(a2);
  double lamb[3];
  lamb[0] = qk;
  lamb[1] = qk + a;
  lamb[2] = qk - a;

  if (fabs(lamb[0]) < 1e-4) lamb[0] = 1e-4;

  double deltaP = p2 - p1;
  double deltaU = state2[1] / state2[0] - state1[1] / state1[0];
  double deltaV = state2[2] / state2[0] - state1[2] / state1[0];
  double deltaQk = deltaU * unitN[0] + deltaV * unitN[1];

  Vector alpha(NS_eq);
  alpha[0] = 0.5 * (deltaP - r * a * deltaU) / a2;
  alpha[1] = state2[0] - state1[0] - deltaP / a2;
  alpha[2] = r * deltaV;
  alpha[3] = 0.5 * (deltaP + r * a * deltaU) / a2;

  Vector DF1(NS_eq);
  DF1[0] = 1.;
  DF1[1] = vel[0];
  DF1[2] = vel[1];
  DF1[3] = 0.5 * (vel[0] * vel[0] + vel[1] * vel[1]);
  DF1 *= state2[0] - state1[0] - deltaP / a2;
  DF1[1] += r * (deltaU - unitN[0] * deltaQk);
  DF1[2] += r * (deltaV - unitN[1] * deltaQk);
  DF1[3] += r * (vel[0] * deltaU + vel[1] * deltaV - qk * deltaQk);

  DF1 *= fabs(lamb[0]);

  Vector DF4(NS_eq);
  DF4[0] = 1.;
  DF4[1] = vel[0] + unitN[0] * a;
  DF4[2] = vel[1] + unitN[1] * a;
  DF4[3] = H + qk * a;
  DF4 *= fabs(lamb[1]) * (deltaP + r * a * deltaQk) * 0.5 / a2;

  Vector DF5(NS_eq);
  DF5[0] = 1.;
  DF5[1] = vel[0] - unitN[0] * a;
  DF5[2] = vel[1] - unitN[1] * a;
  DF5[3] = H - qk * a;
  DF5 *= fabs(lamb[2]) * (deltaP - r * a * deltaQk) * 0.5 / a2;

  for (int i = 0; i < NS_eq; i++) {
    flux(i) = (meanFlux[i] - (DF1[i] + DF4[i] + DF5[i])) * 0.5 * normag;
  }

  if (eqSystem == NS_PASSIVE)
    flux(num_equation - 1) =
        meanFlux(num_equation - 1) - 0.5 * fabs(qk) * (state2(num_equation - 1) - state1(num_equation - 1));
}
