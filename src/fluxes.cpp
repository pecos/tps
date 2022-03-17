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

Fluxes::Fluxes(GasMixture *_mixture, Equations &_eqSystem, TransportProperties *_transport, const int &_num_equation, const int &_dim, bool axisym)
    : mixture(_mixture),
      eqSystem(_eqSystem),
      transport(_transport),
      dim(_dim),
      nvel(axisym ? 3 : _dim),
      axisymmetric_(axisym),
      num_equation(_num_equation) {
  gradT.SetSize(dim);
  vel.SetSize(dim);
  vtmp.SetSize(dim);
  stress.SetSize(dim, dim);
  // TODO: Ultimately, take Up as input variable.
  // Multi-species cannot use this, since it is not clear which species gas constant is needed.
  // Also we should not repeat primitive computations here.
  Rg = mixture->GetGasConstant();
}

void Fluxes::ComputeTotalFlux(const Vector &state, const DenseMatrix &gradUpi, DenseMatrix &flux) {
  switch (eqSystem) {
    case EULER:
      ComputeConvectiveFluxes(state, flux);
      break;
    case NS:
    case NS_PASSIVE: {
      DenseMatrix convF(num_equation, dim);
      ComputeConvectiveFluxes(state, convF);

      DenseMatrix viscF(num_equation, dim);
      ComputeViscousFluxes(state, gradUpi, 1, viscF);
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) flux(eq, d) = convF(eq, d) - viscF(eq, d);
      }
    } break;
  }
}

void Fluxes::ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux) {
  const double pres = mixture->ComputePressure(state);
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  for (int d = 0; d < dim; d++) {
    flux(0, d) = state(d + 1);
    for (int i = 0; i < nvel; i++) {
      flux(1 + i, d) = state(i + 1) * state(d + 1) / state[0];
    }
    flux(1 + d, d) += pres;
  }

  const double H = (state[1 + nvel] + pres) / state[0];
  for (int d = 0; d < dim; d++) {
    flux(1 + nvel, d) = state(d + 1) * H;
  }

  // Kevin: even NS_PASSIVE will be controlled by this. no need of if statement.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) flux(nvel + 2 + sp, d) = state(nvel + 2 + sp) * state(1 + d) / state(0);
  }

  if (twoTemperature) {
    // TODO: add inviscid flux for electron energy equation.
  }
}

void Fluxes::ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, double radius, DenseMatrix &flux) {
  flux = 0.;
  if (eqSystem == EULER) {
    return;
  }

  // // TODO: Ultimately, take Up as input variable.
  // // TODO: pressure and temperature must be comptued by mixture only.
  // const double p = mixture->ComputePressure(state);
  // const double temp = p / state[0] / UNIVERSALGASCONSTANT * mixture->GetGasParams(0,GasParams::SPECIES_MW);
  // const double temp = p / state[0] / Rg;
  // const double visc = mixture->GetViscosity(state);
  // const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  // const double k = mixture->GetThermalConductivity(state);

  const int numSpecies = mixture->GetNumSpecies();
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  Vector speciesEnthalpies;
  mixture->computeSpeciesEnthalpies(state, speciesEnthalpies);

  Vector transportBuffer;
  DenseMatrix diffusionVelocity;
  transport->ComputeFluxTransportProperties(state, gradUp, transportBuffer, diffusionVelocity);
  const double visc = transportBuffer[GlobalTrnsCoeffs::VISCOSITY];
  const double bulkViscMult = transportBuffer[GlobalTrnsCoeffs::BULK_VISCOSITY];
  double k = transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY];
  if (!twoTemperature) k += transportBuffer[GlobalTrnsCoeffs::ELECTRON_THERMAL_CONDUCTIVITY];

  const double ur = (axisymmetric_ ? state[1] / state[0] : 0);
  const double ut = (axisymmetric_ ? state[3] / state[0] : 0);

  // make sure density visc. flux is 0
  for (int d = 0; d < dim; d++) flux(0, d) = 0.;

  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) stress(i, j) = gradUp(1 + j, i) + gradUp(1 + i, j);
    divV += gradUp(1 + i, i);
  }

  if (axisymmetric_ && radius > 0) {
    divV += ur / radius;
  }

  for (int i = 0; i < dim; i++) stress(i, i) += (bulkViscMult - 2. / 3.) * divV;
  stress *= visc;

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) flux(1 + i, j) = stress(i, j);

  double tau_tr = 0, tau_tz = 0;
  if (axisymmetric_) {
    const double ut_r = gradUp(3, 0);
    const double ut_z = gradUp(3, 1);
    tau_tr = ut_r;
    if (radius > 0) tau_tr -= ut / radius;
    tau_tr *= visc;

    tau_tz = visc * ut_z;

    flux(1 + 2, 0) = tau_tr;
    flux(1 + 2, 1) = tau_tz;
  }

  // temperature gradient
  //       for (int d = 0; d < dim; d++) gradT[d] = temp * (gradUp(1 + dim, d) / p - gradUp(0, d) / state[0]);

  for (int d = 0; d < dim; d++) vel(d) = state[1 + d] / state[0];

  stress.Mult(vel, vtmp);

  for (int d = 0; d < dim; d++) {
    flux(1 + nvel, d) = vtmp[d];
    flux(1 + nvel, d) += k * gradUp(1 + nvel, d);
    // compute diffusive enthalpy flux.
    for (int sp = 0; sp < numSpecies; sp++) {
      flux(1 + nvel, d) -= speciesEnthalpies(sp) * diffusionVelocity(sp, d);
    }
  }

  if (axisymmetric_) {
    flux(1 + nvel, 0) += ut * tau_tr;
    flux(1 + nvel, 1) += ut * tau_tz;
  }

  // if (eqSystem == NS_PASSIVE) {
  //   double Sc = mixture->GetSchmidtNum();
  //   for (int d = 0; d < dim; d++) flux(num_equation - 1, d) = visc / Sc * gradUp(num_equation - 1, d);
  // }
  // NOTE: NS_PASSIVE will not be needed (automatically incorporated).
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // TODO: need to check the sign.
    // NOTE: diffusionVelocity is set to be (numSpecies,dim)-matrix.
    for (int d = 0; d < dim; d++) flux(nvel + 2 + sp, d) = - state[nvel + 2 + sp] * diffusionVelocity(sp, d);
  }

  if (mixture->IsTwoTemperature()) {
    // TODO: add electron heat flux for total energy equation.
    // TODO: viscous flux for electron energy equation.
  }
}

void Fluxes::ComputeSplitFlux(const mfem::Vector &state, mfem::DenseMatrix &a_mat, mfem::DenseMatrix &c_mat) {
  const int num_equation = state.Size();
  const int dim = num_equation - 2;

  a_mat.SetSize(num_equation, dim);
  c_mat.SetSize(num_equation, dim);

  const double rho = state(0);
  const Vector rhoV(state.GetData() + 1, dim);
  const double rhoE = state(num_equation - 1);
  // const double p = eqState->ComputePressure(state, dim);
  // cout<<"*"<<p<<" "<<rho<<" "<<rhoE<<endl;

  for (int d = 0; d < dim; d++) {
    for (int i = 0; i < num_equation - 1; i++) a_mat(i, d) = rhoV(d);
    a_mat(num_equation - 1, d) = rhoV(d) / rho;

    c_mat(0, d) = 1.;
    for (int i = 0; i < dim; i++) {
      c_mat(i + 1, d) = rhoV(i) / rho;
    }
    c_mat(num_equation - 1, d) = rhoE /*+p*/;
  }
}

void Fluxes::convectiveFluxes_gpu(const Vector &x, DenseTensor &flux, const Equations &eqSystem, GasMixture *mixture,
                                  const int &dof, const int &dim, const int &num_equation) {
#ifdef _GPU_
  auto dataIn = x.Read();
  auto d_flux = flux.Write();

  double gamma = mixture->GetSpecificHeatRatio();
  double Sc = mixture->GetSchmidtNum();

  MFEM_FORALL_2D(n, dof, num_equation, 1, 1, {
    MFEM_SHARED double Un[20];
    MFEM_SHARED double KE[3];
    MFEM_SHARED double p;

    MFEM_FOREACH_THREAD(eq, x, num_equation) {
      Un[eq] = dataIn[n + eq * dof];
      MFEM_SYNC_THREAD;

      if (eq < dim) KE[eq] = 0.5 * Un[1 + eq] * Un[1 + eq] / Un[0];
      if (dim != 3 && eq == 1) KE[2] = 0.;
      MFEM_SYNC_THREAD;

      if (eq == 0) p = DryAir::pressure(&Un[0], &KE[0], gamma, dim, num_equation);
      MFEM_SYNC_THREAD;

      double temp;
      for (int d = 0; d < dim; d++) {
        if (eq == 0) d_flux[n + d * dof + eq * dof * dim] = Un[1 + d];
        if (eq > 0 && eq <= dim) {
          temp = Un[eq] * Un[1 + d] / Un[0];
          if (eq - 1 == d) temp += p;
          d_flux[n + d * dof + eq * dof * dim] = temp;
        }
        if (eq == 1 + dim) {
          d_flux[n + d * dof + eq * dof * dim] = Un[1 + d] * (Un[1 + dim] + p) / Un[0];
        }

        if (eq == num_equation - 1 && eqSystem == NS_PASSIVE)
          d_flux[n + d * dof + eq * dof * dim] = Un[num_equation - 1] * Un[1 + d] / Un[0];
      }
    }  // end MFEM_FOREACH_THREAD
  });  // end MFEM_FORALL_WD
#endif
}

void Fluxes::viscousFluxes_gpu(const Vector &x, ParGridFunction *gradUp, DenseTensor &flux, const Equations &eqSystem,
                               GasMixture *mixture, const ParGridFunction *spaceVaryViscMult,
                               const linearlyVaryingVisc &linViscData, const int &dof, const int &dim,
                               const int &num_equation) {
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

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();

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

      Fluxes::viscousFlux_gpu(&vFlux[0], &Un[0], &gradUpn[0], eqSystem, gamma, Rg, viscMult, bulkViscMult,
                              Pr, Sc, eq, num_equation, dim, num_equation);

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
