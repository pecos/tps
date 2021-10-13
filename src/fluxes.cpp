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
#include "fluxes.hpp"

Fluxes::Fluxes(EquationOfState *_eqState, Equations &_eqSystem, const int &_num_equations, const int &_dim)
    : eqState(_eqState), eqSystem(_eqSystem), dim(_dim), num_equations(_num_equations) {
  gradT.SetSize(dim);
  vel.SetSize(dim);
  vtmp.SetSize(dim);
  stress.SetSize(dim, dim);
  Rg = eqState->GetGasConstant();
}

void Fluxes::ComputeTotalFlux(const Vector &state, const DenseMatrix &gradUpi, DenseMatrix &flux) {
  switch (eqSystem) {
    case EULER:
      ComputeConvectiveFluxes(state, flux);
      break;
    case NS: {
      DenseMatrix convF(num_equations, dim);
      ComputeConvectiveFluxes(state, convF);

      DenseMatrix viscF(num_equations, dim);
      ComputeViscousFluxes(state, gradUpi, viscF);
      for (int eq = 0; eq < num_equations; eq++) {
        for (int d = 0; d < dim; d++) flux(eq, d) = convF(eq, d) - viscF(eq, d);
      }
    } break;
    case MHD:
      break;
  }
}

void Fluxes::ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux) {
  const double pres = eqState->ComputePressure(state, dim);

  for (int d = 0; d < dim; d++) {
    flux(0, d) = state(d + 1);
    for (int i = 0; i < dim; i++) {
      flux(1 + i, d) = state(i + 1) * state(d + 1) / state[0];
    }
    flux(1 + d, d) += pres;
  }

  const double H = (state[1 + dim] + pres) / state[0];
  for (int d = 0; d < dim; d++) {
    flux(1 + dim, d) = state(d + 1) * H;
  }
}

void Fluxes::ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux) {
  switch (eqSystem) {
    case NS: {
      const double p = eqState->ComputePressure(state, dim);
      const double temp = p / state[0] / Rg;
      const double visc = eqState->GetViscosity(temp);
      const double bulkViscMult = eqState->GetBulkViscMultiplyer();
      const double k = eqState->GetThermalConductivity(visc);

      // make sure density visc. flux is 0
      for (int d = 0; d < dim; d++) flux(0, d) = 0.;

      double divV = 0.;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) stress(i, j) = gradUp(1 + j, i) + gradUp(1 + i, j);
        divV += gradUp(1 + i, i);
      }

      for (int i = 0; i < dim; i++) stress(i, i) += (bulkViscMult - 2. / 3.) * divV;
      stress *= visc;

      for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) flux(1 + i, j) = stress(i, j);

      // temperature gradient

      for (int d = 0; d < dim; d++) gradT[d] = temp * (gradUp(1 + dim, d) / p - gradUp(0, d) / state[0]);

      for (int d = 0; d < dim; d++) vel(d) = state[1 + d] / state[0];

      stress.Mult(vel, vtmp);
      for (int d = 0; d < dim; d++) {
        flux(1 + dim, d) += vtmp[d];
        flux(1 + dim, d) += k * gradT[d];
      }
    } break;
    default:
      flux = 0.;
      break;
  }
}

void Fluxes::ComputeSplitFlux(const mfem::Vector &state, mfem::DenseMatrix &a_mat, mfem::DenseMatrix &c_mat) {
  const int num_equations = state.Size();
  const int dim = num_equations - 2;

  a_mat.SetSize(num_equations, dim);
  c_mat.SetSize(num_equations, dim);

  const double rho = state(0);
  const Vector rhoV(state.GetData() + 1, dim);
  const double rhoE = state(num_equations - 1);
  // const double p = eqState->ComputePressure(state, dim);
  // cout<<"*"<<p<<" "<<rho<<" "<<rhoE<<endl;

  for (int d = 0; d < dim; d++) {
    for (int i = 0; i < num_equations - 1; i++) a_mat(i, d) = rhoV(d);
    a_mat(num_equations - 1, d) = rhoV(d) / rho;

    c_mat(0, d) = 1.;
    for (int i = 0; i < dim; i++) {
      c_mat(i + 1, d) = rhoV(i) / rho;
    }
    c_mat(num_equations - 1, d) = rhoE /*+p*/;
  }
}

void Fluxes::convectiveFluxes_gpu(const Vector &x, DenseTensor &flux, const double &gamma, const int &dof,
                                  const int &dim, const int &num_equation) {
#ifdef _GPU_
  auto dataIn = x.Read();
  auto d_flux = flux.ReadWrite();

  MFEM_FORALL_2D(n, dof, num_equation, 1, 1, {
    MFEM_SHARED double Un[5];
    MFEM_SHARED double KE[3];
    MFEM_SHARED double p;

    MFEM_FOREACH_THREAD(eq, x, num_equation) {
      Un[eq] = dataIn[n + eq * dof];
      MFEM_SYNC_THREAD;

      if (eq < dim) KE[eq] = 0.5 * Un[1 + eq] * Un[1 + eq] / Un[0];
      if (dim != 3 && eq == 1) KE[2] = 0.;
      MFEM_SYNC_THREAD;

      if (eq == 0) p = EquationOfState::pressure(&Un[0], &KE[0], gamma, dim, num_equation);
      MFEM_SYNC_THREAD;

      double temp;
      for (int d = 0; d < dim; d++) {
        if (eq == 0) d_flux[n + d * dof + eq * dof * dim] = Un[1 + d];
        if (eq > 0 && eq <= dim) {
          temp = Un[eq] * Un[1 + d] / Un[0];
          if (eq - 1 == d) temp += p;
          d_flux[n + d * dof + eq * dof * dim] = temp;
        }
        if (eq == num_equation - 1) {
          d_flux[n + d * dof + eq * dof * dim] = Un[1 + d] * (Un[num_equation - 1] + p) / Un[0];
        }
      }
    }  // end MFEM_FOREACH_THREAD
  });  // end MFEM_FORALL_WD
#endif
}

void Fluxes::viscousFluxes_gpu(const Vector &x, ParGridFunction *gradUp, DenseTensor &flux, const double &gamma,
                               const double &Rg,  // gas constant
                               const double &Pr,  // Prandtl number
                               const double &viscMult, const double &bulkViscMult,
                               const ParGridFunction *spaceVaryViscMult, const linearlyVaryingVisc &linViscData,
                               const int &dof, const int &dim, const int &num_equation) {
#ifdef _GPU_
  const double *dataIn = x.Read();
  double *d_flux = flux.ReadWrite();
  const double *d_gradUp = gradUp->Read();

  const double *d_spaceVaryViscMult;
  if (spaceVaryViscMult != NULL) {
    d_spaceVaryViscMult = spaceVaryViscMult->Read();
  } else {
    d_spaceVaryViscMult = NULL;
  }

  // clang-format off
  MFEM_FORALL_2D(n, dof, num_equation, 1, 1, {
    MFEM_FOREACH_THREAD(eq, x, num_equation) {
      MFEM_SHARED double Un[5];
      MFEM_SHARED double gradUpn[5 * 3];
      MFEM_SHARED double vFlux[5 * 3];
      MFEM_SHARED double linVisc;

      // init. State
      Un[eq] = dataIn[n + eq * dof];

      for (int d = 0; d < dim; d++) {
        gradUpn[eq + d * num_equation] = d_gradUp[n + eq * dof + d * dof * num_equation];
      }
      MFEM_SYNC_THREAD;

      Fluxes::viscousFlux_gpu(&vFlux[0], &Un[0], &gradUpn[0], gamma, Rg, viscMult, bulkViscMult,
                              Pr, eq, num_equation, dim, num_equation);

      MFEM_SYNC_THREAD;

      if (d_spaceVaryViscMult != NULL) {
        if (eq == 0) {
          linVisc = d_spaceVaryViscMult[n];
        }
        MFEM_SYNC_THREAD;

        for (int d = 0; d < dim; d++) {
          vFlux[eq + d * num_equation] *= linVisc;
        }
      }

      // write to global memory
      for (int d = 0; d < dim; d++) {
        d_flux[n + d * dof + eq * dof * dim] -= vFlux[eq + d * num_equation];
      }
    }  // end MFEM_FOREACH_THREAD
  });  // end MFEM_FORALL_2D
#endif
}
// clang-format on
