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
#include "element_integrator.hpp"

using namespace mfem;

ElementIntegrator::ElementIntegrator(int dim, int num_eqn, bool axisym, Fluxes *flux, IntegrationRules *int_rules,
                                     ParFiniteElementSpace *vfes, ParGridFunction *gradUp)
    : dim_(dim), num_eqn_(num_eqn), axisym_(axisym), flux_(flux), int_rules_(int_rules), vfes_(vfes), gradUp_(gradUp) {}

void ElementIntegrator::getElementGrad(const int elemNo, const FiniteElement &el, DenseTensor &gradUpElem) {
  const int totDofs = vfes_->GetNDofs();
  const double *dataGradUp = gradUp_->HostRead();

  vfes_->GetElementVDofs(elemNo, vdofs_);
  int eldDof = el.GetDof();
  for (int n = 0; n < eldDof; n++) {
    int index = vdofs_[n];
    for (int eq = 0; eq < num_eqn_; eq++) {
      for (int d = 0; d < dim_; d++) {
        gradUpElem(n, eq, d) = dataGradUp[index + (eq + d * num_eqn_) * totDofs];
      }
    }
  }
}

void ElementIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
                                              Vector &elvec) {
  const int dof = el.GetDof();

  // Init residual contribution to zero
  elvec.SetSize(dof * num_eqn_);
  elvec = 0.0;

  // Storage for basis fcns, derivatives, fluxes
  Vector shape;
  Vector soln;
  DenseMatrix adjJflux;
  DenseMatrix shape_r;
  DenseMatrix shape_x;
  DenseMatrix soln_x;
  DenseMatrix Fflux;
  DenseMatrix Gflux;
  DenseTensor gradUpElem;

  shape.SetSize(dof);
  soln.SetSize(num_eqn_);
  adjJflux.SetSize(num_eqn_, dim_);
  shape_r.SetSize(dof, dim_);
  shape_x.SetSize(dof, dim_);
  soln_x.SetSize(num_eqn_, dim_);
  Fflux.SetSize(num_eqn_, dim_);
  Gflux.SetSize(num_eqn_, dim_);
  gradUpElem.SetSize(dof, num_eqn_, dim_);

  // View elfun and elvec as matrices (make life easy below)
  DenseMatrix elfun_mat(elfun.GetData(), dof, num_eqn_);
  DenseMatrix elvec_mat(elvec.GetData(), dof, num_eqn_);

  // Determine integration rule
  int intorder;
  intorder = Tr.OrderW() + 2 * el.GetOrder();
  if (el.Space() == FunctionSpace::Pk) {
    intorder++;
  }
  const IntegrationRule *ir = &int_rules_->Get(el.GetGeomType(), intorder);

  // for every quadrature point...
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    // evaluate basis functions and derivatives (wrt reference)
    el.CalcShape(ip, shape);
    //el.CalcDShape(ip, shape_r);

    // Mult(shape_r, Tr.InverseJacobian(), shape_x);

    // Interpolate solution
    elfun_mat.MultTranspose(shape, soln);
    // elfun_mat.MultTranspose(shape_x, soln_x);
    // MultAtB(elfun_mat, shape_x, soln_x);

    // Interpolate gradient
    soln_x = 0.;
    for (int eq = 0; eq < num_eqn_; eq++) {
      for (int d = 0; d < dim_; d++) {
        for (int k = 0; k < dof; k++) {
          soln_x(eq, d) += gradUpElem(k, eq, d) * shape(k);
        }
      }
    }

    // Get radius
    double radius = 1;
    if (axisym_) {
      double x[3];
      Vector transip(x, 3);
      Tr.Transform(ip, transip);
      radius = transip[0];
    }

    // Evaluate the fluxes
    Fflux = 0.;
    flux_->ComputeConvectiveFluxes(soln, Fflux);

    Gflux = 0.;
    flux_->ComputeViscousFluxes(soln, soln_x, radius, Gflux);

    // total flux
    Fflux -= Gflux;

    MultABt(Fflux, Tr.AdjugateJacobian(), adjJflux);
    adjJflux *= ip.weight;

    if (axisym_) {
      adjJflux *= radius;
    }

    AddMultABt(shape_r, adjJflux, elvec_mat);
  }
}

void ElementIntegrator::AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
                                            DenseMatrix &elmat) {
  const int dof = el.GetDof();

  // Init Jacobian contribution to zero
  elmat.SetSize(dof * num_eqn_, dof * num_eqn_);
  elmat = 0.0;

  // Storage for basis fcns, derivatives, fluxes
  Vector shape;
  Vector soln;
  DenseMatrix adjJflux;
  DenseTensor adjJflux_U;
  DenseMatrix shape_r;
  DenseMatrix shape_x;
  DenseMatrix soln_x;
  DenseMatrix Fflux;
  DenseMatrix Gflux;
  DenseTensor Jacobian;

  shape.SetSize(dof);
  soln.SetSize(num_eqn_);
  adjJflux.SetSize(num_eqn_, dim_);
  adjJflux_U.SetSize(num_eqn_, dim_, num_eqn_);
  shape_r.SetSize(dof, dim_);
  shape_x.SetSize(dof, dim_);
  soln_x.SetSize(num_eqn_, dim_);
  Fflux.SetSize(num_eqn_, dim_);
  Gflux.SetSize(num_eqn_, dim_);

  // View elfun and elvec as matrices (make life easy below)
  DenseMatrix elfun_mat(elfun.GetData(), dof, num_eqn_);

  // Determine integration rule
  int intorder;
  intorder = Tr.OrderW() + 2 * el.GetOrder();
  if (el.Space() == FunctionSpace::Pk) {
    intorder++;
  }
  const IntegrationRule *ir = &int_rules_->Get(el.GetGeomType(), intorder);

  // for every quadrature point...
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    // evaluate basis functions and derivatives (wrt reference)
    el.CalcShape(ip, shape);
    el.CalcDShape(ip, shape_r);

    // Interpolate solution
    elfun_mat.MultTranspose(shape, soln);

    // Evaluate the (inviscid) flux Jacobian
    flux_->ComputeConvectiveFluxJacobian(soln, Jacobian);

    for (int jeqn = 0; jeqn < num_eqn_; jeqn++) {
      MultABt(Jacobian(jeqn), Tr.AdjugateJacobian(), adjJflux_U(jeqn));
      adjJflux_U(jeqn) *= ip.weight;
    }

    for (int ieqn = 0; ieqn < num_eqn_; ieqn++) {
      for (int idof = 0; idof < dof; idof++) {
        for (int idim = 0; idim < dim_; idim++) {
          // elvect[ieqn*dof + idof] += shape_r(idof, idim) * adjJflux(ieqn, idim);

          for (int jeqn = 0; jeqn < num_eqn_; jeqn++) {
            for (int jdof = 0; jdof < dof; jdof++) {
              elmat(ieqn * dof + idof, jeqn * dof + jdof) +=
                  shape_r(idof, idim) * adjJflux_U(ieqn, idim, jeqn) * shape(jdof);
            }
          }
        }
      }
    }
  }
}
