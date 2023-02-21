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

#include "faceGradientIntegration.hpp"

// Implementation of class FaceIntegrator
GradFaceIntegrator::GradFaceIntegrator(IntegrationRules *_intRules, const int _dim, const int _num_equation)
    : dim(_dim), num_equation(_num_equation), intRules(_intRules) {}

void GradFaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                            FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  // Compute the term <nU,[w]> on the interior faces.
  Vector nor(dim);
  Vector mean(num_equation);
  Vector iUp1(num_equation), iUp2(num_equation);
  Vector du1n(dim * num_equation), du2n(dim * num_equation);

  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  Vector shape1(dof1);
  Vector shape2(dof2);

  elvect.SetSize((dof1 + dof2) * num_equation * dim);
  elvect = 0.0;

  // NB: Incoming Vector &elfun is in gradient space.  The first
  // component is the state, and the rest is zero.  See
  // GradNonLinearForm::Mult in gradNonLinearForm.cpp for details.
  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
  DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation * dim, dof2, num_equation);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation * dim);
  DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation * dim, dof2, num_equation * dim);
  elvect1_mat = 0.;
  elvect2_mat = 0.;

  // Integration order calculation from DGTraceIntegrator
  int intorder;
  if (Tr.Elem2No >= 0) {
    intorder = (std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) + 2 * std::max(el1.GetOrder(), el2.GetOrder()));
  } else {
    intorder = Tr.Elem1->OrderW() + 2 * el1.GetOrder();
  }
  if (el1.Space() == FunctionSpace::Pk) {
    intorder++;
  }
  const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip);  // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate U values at the point
    elfun1_mat.MultTranspose(shape1, iUp1);
    elfun2_mat.MultTranspose(shape2, iUp2);

    // Compute average
    // NB: Code below is mathematically equivalent to mean = 0.5*(iUp1
    // + iUp2), but mfem::Vector does not provide operator+
    mean.Set(0.5, iUp1);
    mean.Add(0.5, iUp2);

    // Get the normal vector
    CalcOrtho(Tr.Jacobian(), nor);
    nor *= ip.weight;

    for (int d = 0; d < dim; d++) {
      for (int eq = 0; eq < num_equation; eq++) {
        du1n[eq + d * num_equation] = (mean(eq) - iUp1(eq)) * nor(d);
        du2n[eq + d * num_equation] = (iUp2(eq) - mean(eq)) * nor(d);
      }
    }

    AddMultVWt(shape1, du1n, elvect1_mat);
    AddMultVWt(shape2, du2n, elvect2_mat);
  }
}
