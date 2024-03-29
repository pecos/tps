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

#include "riemann_solver.hpp"

using namespace mfem;

// Implementation of class RiemannSolver
MFEM_HOST_DEVICE RiemannSolver::RiemannSolver(int _num_equation, GasMixture *_mixture, Equations _eqSystem,
                                              Fluxes *_fluxClass, bool _useRoe, bool axisym)
    : num_equation(_num_equation),
      mixture(_mixture),
      eqSystem(_eqSystem),
      fluxClass(_fluxClass),
      useRoe(_useRoe),
      axisymmetric_(axisym) {}

// Compute the scalar F(u).n
void RiemannSolver::ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &fluxN) {
  ComputeFluxDotN(state.GetData(), nor.GetData(), fluxN.GetData());
}

// Compute the scalar F(u).n
MFEM_HOST_DEVICE void RiemannSolver::ComputeFluxDotN(const double *state, const double *nor, double *fluxN) const {
  const int dim = mixture->GetDimension();

  double fluxes[gpudata::MAXEQUATIONS * gpudata::MAXDIM];  // double fluxes[5 * 3];
  fluxClass->ComputeConvectiveFluxes(state, fluxes);
  for (int eq = 0; eq < num_equation; eq++) {
    fluxN[eq] = 0;
    for (int d = 0; d < dim; d++) {
      fluxN[eq] += fluxes[eq + d * num_equation] * nor[d];
    }
  }
}

void RiemannSolver::Eval(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux, bool LF) {
  if (useRoe && !LF) {
    Eval_Roe(state1, state2, nor, flux);
  } else {
    Eval_LF(state1, state2, nor, flux);
  }
}

MFEM_HOST_DEVICE void RiemannSolver::Eval(const double *state1, const double *state2, const double *nor, double *flux,
                                          bool LF) {
  if (useRoe && !LF) {
    // TODO(kevin): implement MFEM_HOST_DEVICE Eval_Roe.
    printf("Roe RiemannSolver is not implmented for gpu!");
    // Eval_Roe(state1, state2, nor, flux);
  } else {
    Eval_LF(state1, state2, nor, flux);
  }
}

void RiemannSolver::Eval_LF(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux) {
  Eval_LF(state1.GetData(), state2.GetData(), nor.GetData(), flux.GetData());
}

MFEM_HOST_DEVICE void RiemannSolver::Eval_LF(const double *state1, const double *state2, const double *nor,
                                             double *flux) const {
  const int dim = mixture->GetDimension();

  const double maxE1 = mixture->ComputeMaxCharSpeed(state1);
  const double maxE2 = mixture->ComputeMaxCharSpeed(state2);

  // const double maxE = max(maxE1, maxE2);
  const double maxE = fmax(maxE1, maxE2);

  double flux1[gpudata::MAXEQUATIONS];  // double flux1[5];
  double flux2[gpudata::MAXEQUATIONS];  // double flux2[5];

  ComputeFluxDotN(state1, nor, flux1);
  ComputeFluxDotN(state2, nor, flux2);

  double normag = 0;
  for (int i = 0; i < dim; i++) {
    normag += nor[i] * nor[i];
  }
  normag = sqrt(normag);

  for (int i = 0; i < num_equation; i++) {
    flux[i] = 0.5 * (flux1[i] + flux2[i]) - 0.5 * maxE * (state2[i] - state1[i]) * normag;
  }
}

// TODO(kevin): need to write for multiple species and two temperature.
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
