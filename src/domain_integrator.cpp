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

#include "domain_integrator.hpp"

// Implementation of class DomainIntegrator
DomainIntegrator::DomainIntegrator(Fluxes *_fluxClass, IntegrationRules *_intRules, int _intRuleType, const int _dim,
                                   const int _num_equation)
    : fluxClass(_fluxClass), dim(_dim), num_equation(_num_equation), intRules(_intRules), intRuleType(_intRuleType) {}

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe, const FiniteElement &test_fe,
                                              ElementTransformation &Tr, DenseMatrix &elmat) {
  // Assemble the form (vec(v), grad(w))
  Vector shape;
  shape.UseDevice(false);
  // DenseMatrix flux(num_equation,dim);
  DenseMatrix dshapedr;
  DenseMatrix dshapedx;

  // Trial space = vector L2 space (mesh dim)
  // Test space  = scalar L2 space

  const int dof_trial = trial_fe.GetDof();
  const int dof_test = test_fe.GetDof();
  const int dim = trial_fe.GetDim();

  shape.SetSize(dof_trial);
  dshapedr.SetSize(dof_test, dim);
  dshapedx.SetSize(dof_test, dim);

  elmat.SetSize(dof_test, dof_trial * dim);
  elmat = 0.0;

  const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
  int intorder = 2 * maxorder;
  if (intRuleType == 1 && trial_fe.GetGeomType() == Geometry::SQUARE) intorder--;  // when Gauss-Lobatto
  const IntegrationRule *ir = &intRules->Get(trial_fe.GetGeomType(), intorder);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    // Calculate the shape functions
    trial_fe.CalcShape(ip, shape);
    shape *= ip.weight;

    // Compute the physical gradients of the test functions
    Tr.SetIntPoint(&ip);
    test_fe.CalcDShape(ip, dshapedr);
    Mult(dshapedr, Tr.AdjugateJacobian(), dshapedx);

    for (int d = 0; d < dim; d++) {
      for (int j = 0; j < dof_test; j++) {
        for (int k = 0; k < dof_trial; k++) {
          elmat(j, k + d * dof_trial) += shape(k) * dshapedx(j, d);
        }
      }
    }
  }
}
