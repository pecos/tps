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

#include <cmath>
#include <limits>

#include "../src/dataStructures.hpp"
#include "../src/equation_of_state.hpp"
#include "../src/fluxes.hpp"
#include "../src/riemann_solver.hpp"

int pingNormalFlux() {
  int ierr = 0;
  int ndim = 3;
  int neqn = 5;
  Equations eqSystem = EULER;
  bool axisym = false;
  bool roe = false;

  // Instantiate objects needed to evaluate the flux
  GasMixture *mixture = new DryAir(ndim, neqn);
  Fluxes *flux = new Fluxes(mixture, eqSystem, neqn, ndim, axisym);
  RiemannSolver *rsolver = new RiemannSolver(neqn, mixture, eqSystem, flux, roe, axisym);

  Vector nhat(ndim);
  Vector U0(neqn), F0(neqn);
  Vector U1(neqn), F1(neqn);
  DenseMatrix F_U0(neqn, neqn), F_U1(neqn, neqn);
  DenseMatrix J_fd(neqn, neqn), J_an(neqn, neqn);
  DenseMatrix err_mat(neqn, neqn);

  // Set the state to something
  const double density = 1.2;
  const double u1 = 20., u2 = 30., u3 = -25.;
  const double Usq = u1 * u1 + u2 * u2 + u3 * u3;
  const double pressure = 101325.;
  const double gamma = mixture->GetSpecificHeatRatio();

  U0[0] = density;
  U0[1] = density * u1;
  U0[2] = density * u2;
  U0[3] = density * u3;
  U0[4] = pressure / (gamma - 1) + 0.5 * density * Usq;

  nhat[0] = std::sqrt(0.5);
  nhat[1] = std::sqrt(0.4);
  nhat[2] = std::sqrt(0.1);

  // Evaluate flux and Jacobian at 'nominal' state
  rsolver->ComputeFluxDotN(U0, nhat, F0);
  rsolver->ComputeFluxDotNJacobian(U0, nhat, F_U0);

  // For this 'ping' test, we take the difference between the
  // first-order finite difference derivative and the average of the
  // analytic derivatives.  If the analytic derivatives are correct
  // (and the function is smooth) this difference converges at
  // O(eps**2), where eps is the step size used for the finite
  // difference.  Thus, the check for this test is whether we observe
  // the correct convergence rate.  This removes the need to supply a
  // tolerance on the discrepancy between the finite difference and
  // analytic results, but also means we have to use a large enough
  // step size that finite difference error (as opposed to rounding
  // error) dominates the discrepancy.  That is why the perturbation
  // as chosen as it is below.


  double perturbation = 2e-2;
  double delta_rel[3];

  // Perturb the state and evaluate finite difference Jacobian approx
  // at three different step sizes
  for (int rep = 0; rep < 3; rep++) {

    for (int i = 0; i < neqn; i++) {
      U1 = U0;
      U1[i] += U0[i]*perturbation;

      rsolver->ComputeFluxDotN(U1, nhat, F1);
      rsolver->ComputeFluxDotNJacobian(U1, nhat, F_U1);

      for (int j = 0; j < neqn; j++) {
        // finite difference
        J_fd(j, i) = (F1[j] - F0[j]) / (U1[i] - U0[i]);

        // 'analytic' (average of results at U0 and U1)
        J_an(j, i) = 0.5*(F_U0(j, i) + F_U1(j, i));
      }
    }

    // Evaluate a relative error metric (based on Frobenius norm)
    Add(J_an, J_fd, -1.0, err_mat);
    delta_rel[rep] = err_mat.FNorm() / J_an.FNorm();

    // Prep for next time through by decreasing the FD step
    perturbation *= 0.5;
  }

  const double rate1 = std::log(delta_rel[1] / delta_rel[0]) / std::log(0.5);
  const double rate2 = std::log(delta_rel[2] / delta_rel[1]) / std::log(0.5);

  std::cout << "RiemannSolver::ComputeFluxDotNJacobian rates = " << rate1 << ", " << rate2 << std::endl;

  // Fail the test if the rate isn't what we expect
  if ( (rate1 < 1.95) || (rate1 > 2.05) ||
       (rate2 < 1.95) || (rate2 > 2.05) ) {
    ierr = 1;
  }

  delete rsolver;
  delete flux;
  delete mixture;

  return ierr;
}

int pingRiemannSolver() {
  int ierr = 0;
  int ndim = 3;
  int neqn = 5;
  Equations eqSystem = EULER;
  bool axisym = false;
  bool roe = false;

  // Instantiate objects needed to evaluate the flux
  GasMixture *mixture = new DryAir(ndim, neqn);
  Fluxes *flux = new Fluxes(mixture, eqSystem, neqn, ndim, axisym);
  RiemannSolver *rsolver = new RiemannSolver(neqn, mixture, eqSystem, flux, roe, axisym);

  Vector nhat(ndim);
  Vector UL0(neqn), UR0(neqn), F0(neqn);
  Vector UL1(neqn), UR1(neqn), F1(neqn);
  DenseMatrix F_UL0(neqn, neqn), F_UR0(neqn, neqn);
  DenseMatrix F_UL1(neqn, neqn), F_UR1(neqn, neqn);
  DenseMatrix JL_fd(neqn, neqn), JL_an(neqn, neqn);
  DenseMatrix JR_fd(neqn, neqn), JR_an(neqn, neqn);
  DenseMatrix err_mat_L(neqn, neqn), err_mat_R(neqn, neqn);

  // Set the state to something
  const double rhoL = 1.2;
  const double uL1 = 20., uL2 = 30., uL3 = -25.;
  const double ULsq = uL1 * uL1 + uL2 * uL2 + uL3 * uL3;
  const double pL = 101325.;

  const double rhoR = 1.2;
  const double uR1 = 20., uR2 = 30., uR3 = -25.;
  const double URsq = uR1 * uR1 + uR2 * uR2 + uR3 * uR3;
  const double pR = 101325.;

  // TODO(trevilo): b/c of a known issue, the Jacobian is not correct when UR-UL != 0
  // const double rhoR = 1.25;
  // const double uR1 = 18., uR2 = 32., uR3 = -22.;
  // const double URsq = uR1 * uR1 + uR2 * uR2 + uR3 * uR3;
  // const double pR = 104325.;

  const double gamma = mixture->GetSpecificHeatRatio();

  UL0[0] = rhoL;
  UL0[1] = rhoL * uL1;
  UL0[2] = rhoL * uL2;
  UL0[3] = rhoL * uL3;
  UL0[4] = pL / (gamma - 1) + 0.5 * rhoL * ULsq;

  UR0[0] = rhoR;
  UR0[1] = rhoR * uR1;
  UR0[2] = rhoR * uR2;
  UR0[3] = rhoR * uR3;
  UR0[4] = pR / (gamma - 1) + 0.5 * rhoR * URsq;

  nhat[0] = std::sqrt(0.5);
  nhat[1] = std::sqrt(0.4);
  nhat[2] = std::sqrt(0.1);

  // Evaluate flux and Jacobian at 'nominal' state
  rsolver->Eval(UL0, UR0, nhat, F0, false);
  rsolver->Jacobian(UL0, UR0, nhat, F_UL0, F_UR0);

  // For this 'ping' test, we take the difference between the
  // first-order finite difference derivative and the average of the
  // analytic derivatives.  If the analytic derivatives are correct
  // (and the function is smooth) this difference converges at
  // O(eps**2), where eps is the step size used for the finite
  // difference.  Thus, the check for this test is whether we observe
  // the correct convergence rate.  This removes the need to supply a
  // tolerance on the discrepancy between the finite difference and
  // analytic results, but also means we have to use a large enough
  // step size that finite difference error (as opposed to rounding
  // error) dominates the discrepancy.  That is why the perturbation
  // as chosen as it is below.


  double perturbation = 2e-2;
  double delta_rel_L[3], delta_rel_R[3];

  // Perturb the state and evaluate finite difference Jacobian approx
  // at three different step sizes
  for (int rep = 0; rep < 3; rep++) {

    // "Left"
    for (int i = 0; i < neqn; i++) {
      UL1 = UL0;
      UL1[i] += UL0[i]*perturbation;

      rsolver->Eval(UL1, UR0, nhat, F1);
      rsolver->Jacobian(UL1, UR0, nhat, F_UL1, F_UR1);

      for (int j = 0; j < neqn; j++) {
        // finite difference
        JL_fd(j, i) = (F1[j] - F0[j]) / (UL1[i] - UL0[i]);

        // 'analytic' (average of results at U0 and U1)
        JL_an(j, i) = 0.5*(F_UL0(j, i) + F_UL1(j, i));
      }
    }

    // Evaluate a relative error metric (based on Frobenius norm)
    Add(JL_an, JL_fd, -1.0, err_mat_L);
    delta_rel_L[rep] = err_mat_L.FNorm() / JL_an.FNorm();

    // "Right"
    for (int i = 0; i < neqn; i++) {
      UR1 = UR0;
      UR1[i] += UR0[i]*perturbation;

      rsolver->Eval(UL0, UR1, nhat, F1);
      rsolver->Jacobian(UL0, UR1, nhat, F_UL1, F_UR1);

      for (int j = 0; j < neqn; j++) {
        // finite difference
        JR_fd(j, i) = (F1[j] - F0[j]) / (UR1[i] - UR0[i]);

        // 'analytic' (average of results at U0 and U1)
        JR_an(j, i) = 0.5*(F_UR0(j, i) + F_UR1(j, i));
      }
    }

    // Evaluate a relative error metric (based on Frobenius norm)
    Add(JR_an, JR_fd, -1.0, err_mat_R);
    delta_rel_R[rep] = err_mat_R.FNorm() / JR_an.FNorm();

    // Prep for next time through by decreasing the FD step
    perturbation *= 0.5;
  }

  const double rateL1 = std::log(delta_rel_L[1] / delta_rel_L[0]) / std::log(0.5);
  const double rateL2 = std::log(delta_rel_L[2] / delta_rel_L[1]) / std::log(0.5);

  const double rateR1 = std::log(delta_rel_R[1] / delta_rel_R[0]) / std::log(0.5);
  const double rateR2 = std::log(delta_rel_R[2] / delta_rel_R[1]) / std::log(0.5);

  std::cout << "RiemannSolver::Jacobian (LF) rates (left) = " << rateL1 << ", " << rateL2 << std::endl;
  std::cout << "RiemannSolver::Jacobian (LF) rates (rght) = " << rateR1 << ", " << rateR2 << std::endl;

  // Fail the test if the rate isn't what we expect
  if ( (rateL1 < 1.95) || (rateL1 > 2.05) ||
       (rateL2 < 1.95) || (rateL2 > 2.05) ) {
    ierr = 1;
  }

  if ( (rateR1 < 1.95) || (rateR1 > 2.05) ||
       (rateR2 < 1.95) || (rateR2 > 2.05) ) {
    ierr = 1;
  }

  delete rsolver;
  delete flux;
  delete mixture;

  return ierr;
}


int main(int argc, char *argv[]) {
  int ierr = 0;

  ierr += pingNormalFlux();
  ierr += pingRiemannSolver();

  return ierr;
}
