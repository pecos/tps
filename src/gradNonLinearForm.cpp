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
#include "gradNonLinearForm.hpp"

#include "riemann_solver.hpp"

GradNonLinearForm::GradNonLinearForm(ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, const int _dim,
                                     const int _num_equation)
    : ParNonlinearForm(_vfes), vfes(_vfes), intRules(_intRules), dim(_dim), num_equation(_num_equation) {}

void GradNonLinearForm::Mult(const ParGridFunction *Up, Vector &y) {
  Vector x;
  const double *data = Up->GetData();

  // NB: Setting x here accounts for the fact that the space Up lives
  // in is different from the space of the gradient.
  x.SetSize(vfes->GetNDofs() * num_equation * dim);
  x = 0.;
  for (int i = 0; i < vfes->GetNDofs() * num_equation; i++) x(i) = data[i];

  // After setting x as above, base class Mult does the right thing
  ParNonlinearForm::Mult(x, y);
}
