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
#ifndef RIEMANN_SOLVER_HPP_
#define RIEMANN_SOLVER_HPP_

#include <tps_config.h>

#include "dataStructures.hpp"
#include "fluxes.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;

// TODO(kevin): In order to avoid repeated primitive variable evaluation,
// Fluxes and RiemannSolver should take Vector Up (on the evaulation point) as input argument,
// and FaceIntegrator should have a pointer to ParGridFunction *Up.
// Also should be able to have Up more than number of equations,
// while gradUp is evaluated only for the first num_equation variables.
// Need to discuss further.

// Implements a simple Rusanov flux
class RiemannSolver {
 private:
  const int num_equation;

  DenseMatrix flux1_state1;
  DenseMatrix flux2_state2;

  GasMixture *mixture;
  Equations eqSystem;
  Fluxes *fluxClass;

  bool useRoe;
  const bool axisymmetric_;

  void Eval_Roe(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux);

  void Jacobian_LF(const Vector &state1, const Vector &state2, const Vector &nor, DenseMatrix &flux_state1,
                   DenseMatrix &flux_state2);

 public:
  MFEM_HOST_DEVICE RiemannSolver(int _num_equation, GasMixture *mixture, Equations _eqSystem, Fluxes *_fluxClass,
                                 bool _useRoe, bool axisym);

  void Eval(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux, bool LF = false);

  MFEM_HOST_DEVICE void Eval(const double *state1, const double *state2, const double *nor, double *flux,
                             bool LF = false);

  void Jacobian(const Vector &state1, const Vector &state2, const Vector &nor, DenseMatrix &flux_state1,
                DenseMatrix &flux_state2, bool LF = false);

  void ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &fluxN);
  void ComputeFluxDotNJacobian(const Vector &state, const Vector &nor, DenseMatrix &fluxN_state);

  MFEM_HOST_DEVICE void ComputeFluxDotN(const double *state, const double *nor, double *fluxN) const;

  void Eval_LF(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux);
  MFEM_HOST_DEVICE void Eval_LF(const double *state1, const double *state2, const double *nor, double *flux) const;

  MFEM_HOST_DEVICE bool isAxisymmetric() const { return axisymmetric_; }
};

#endif  // RIEMANN_SOLVER_HPP_
