// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2023, The PECOS Development Team, University of Texas at Austin
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

#include "source_fcn.hpp"

using namespace mfem;

SourceFunction::SourceFunction(const int dim, const int nvel, const int num_equation, const Equations eqSys)
    : dim_(dim), nvel_(nvel), neqn_(num_equation), eqSys_(eqSys) {}

SourceFunction::~SourceFunction() {}

AxisymmetricSourceFunction::AxisymmetricSourceFunction(const int dim, const int nvel, const int num_equation,
                                                       const Equations eqSys, GasMixture* mix,
                                                       TransportProperties* trans)
    : SourceFunction(dim, nvel, num_equation, eqSys), mixture_(mix), transport_(trans) {}

AxisymmetricSourceFunction::~AxisymmetricSourceFunction() {}

void AxisymmetricSourceFunction::evaluate(const Vector& x, const Vector& U, const DenseMatrix& gradUp, Vector& S) {
  S.SetSize(neqn_);
  S = 0.;

  const double radius = x[0];
  assert(radius > 0);

  Vector Up(neqn_);
  mixture_->GetPrimitivesFromConservatives(U, Up);

  const double rho = Up(0);
  const double ur = Up(1);
  // const double uz = Up(2);
  const double ut = Up(3);
  // const double temperature = Up(1 + nvel);

  const double pressure = mixture_->ComputePressureFromPrimitives(Up);

  const double rurut = rho * ur * ut;
  const double rutut = rho * ut * ut;

  double tau_tt, tau_tr;
  if (eqSys_ == EULER) {
    tau_tt = tau_tr = 0.0;
  } else {
    const double ur_r = gradUp(1, 0);
    const double uz_z = gradUp(2, 1);
    const double ut_r = gradUp(3, 0);

    double visc, bulkVisc;
    transport_->GetViscosities(U, Up, visc, bulkVisc);
    bulkVisc -= 2. / 3. * visc;

    // TODO(trevilo): Handle spatially varying viscosity multiplier correctly
    // if (space_vary_viscosity_mult_ != NULL) {
    //   auto *alpha = space_vary_viscosity_mult_->GetData();
    //   visc *= alpha[index];
    // }

    double divV = ur_r + uz_z;
    divV += ur / radius;

    tau_tt = 2.0 * ur / radius * visc;
    tau_tt += bulkVisc * divV;

    tau_tr = ut_r;
    tau_tr -= ut / radius;
    tau_tr *= visc;
  }

  S[1] = (pressure + rutut - tau_tt);
  S[3] = (-rurut + tau_tr);
}
