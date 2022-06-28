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
#include "fluxes.hpp"

MFEM_HOST_DEVICE Fluxes::Fluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                const int _num_equation, const int _dim, bool axisym)
    : mixture(_mixture),
      eqSystem(_eqSystem),
      transport(_transport),
      dim(_dim),
      nvel(axisym ? 3 : _dim),
      axisymmetric_(axisym),
      num_equation(_num_equation) {}

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
  double Pe = 0.0;
  const double pres = mixture->ComputePressure(state, &Pe);
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
    double electronEnthalpy = (state(num_equation - 1) + Pe) / state(0);
    for (int d = 0; d < dim; d++) flux(num_equation - 1, d) = electronEnthalpy * state(1 + d);
    // for (int d = 0; d < dim; d++) flux(num_equation - 1, d) = state(num_equation - 1) * state(1 + d) / state(0);
  }
}

MFEM_HOST_DEVICE void Fluxes::ComputeConvectiveFluxes(const double *state, double *flux) const {
  double Pe = 0.0;
  const double pres = mixture->ComputePressure(state, &Pe);
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  // For consistency with version above, where flux is a DenseMatrix,
  // indexing into flux is column-major

  for (int d = 0; d < dim; d++) {
    flux[0 + d * num_equation] = state[d + 1];
    for (int i = 0; i < nvel; i++) {
      flux[1 + i + d * num_equation] = state[i + 1] * state[d + 1] / state[0];
    }
    flux[1 + d + d * num_equation] += pres;
  }

  const double H = (state[1 + nvel] + pres) / state[0];
  for (int d = 0; d < dim; d++) {
    flux[1 + nvel + d * num_equation] = state[d + 1] * H;
  }

  // Kevin: even NS_PASSIVE will be controlled by this. no need of if statement.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) {
      flux[nvel + 2 + sp + d * num_equation] = state[nvel + 2 + sp] * state[1 + d] / state[0];
    }
  }

  if (twoTemperature) {
    const double electronEnthalpy = (state[num_equation - 1] + Pe) / state[0];
    for (int d = 0; d < dim; d++) {
      flux[num_equation - 1 + d * num_equation] = electronEnthalpy * state[1 + d];
    }
  }
}

// TODO(kevin): check/complete axisymmetric setting for multi-component flow.
void Fluxes::ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, double radius, DenseMatrix &flux) {
  flux = 0.;
  if (eqSystem == EULER) {
    return;
  }

  Vector vel(dim);
  Vector vtmp(dim);
  DenseMatrix stress(dim, dim);

  // // TODO(kevin): Ultimately, take Up as input variable.
  // // TODO(kevin): pressure and temperature must be comptued by mixture only.
  // const double p = mixture->ComputePressure(state);
  // const double temp = p / state[0] / UNIVERSALGASCONSTANT * mixture->GetGasParams(0,GasParams::SPECIES_MW);
  // const double temp = p / state[0] / Rg;
  // const double visc = mixture->GetViscosity(state);
  // const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  // const double k = mixture->GetThermalConductivity(state);

  // TODO(kevin): update E-field with EM coupling.
  Vector Efield(nvel);
  Efield = 0.0;

  const int numSpecies = mixture->GetNumSpecies();
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  Vector speciesEnthalpies(numSpecies);
  mixture->computeSpeciesEnthalpies(state, speciesEnthalpies);

  Vector transportBuffer;
  // NOTE(kevin): in flux, only dim-components of diffusionVelocity will be used.
  DenseMatrix diffusionVelocity(numSpecies, nvel);
  transport->ComputeFluxTransportProperties(state, gradUp, Efield, transportBuffer, diffusionVelocity);
  const double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];

  if (twoTemperature) {
    for (int d = 0; d < dim; d++) {
      double qeFlux = ke * gradUp(num_equation - 1, d);
      flux(1 + nvel, d) += qeFlux;
      flux(num_equation - 1, d) += qeFlux;
      flux(num_equation - 1, d) -= speciesEnthalpies(numSpecies - 2) * diffusionVelocity(numSpecies - 2, d);
    }
  } else {
    k += ke;
  }

  const double ur = (axisymmetric_ ? state[1] / state[0] : 0);
  const double ut = (axisymmetric_ ? state[3] / state[0] : 0);

  // make sure density visc. flux is 0
  for (int d = 0; d < dim; d++) flux(0, d) = 0.;

  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) stress(i, j) = gradUp(1 + j, i) + gradUp(1 + i, j);
    divV += gradUp(1 + i, i);
  }
  stress *= visc;

  if (axisymmetric_ && radius > 0) {
    divV += ur / radius;
  }

  for (int i = 0; i < dim; i++) stress(i, i) += bulkViscosity * divV;

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
    flux(1 + nvel, d) += vtmp[d];
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
    // NOTE: diffusionVelocity is set to be (numSpecies,nvel)-matrix.
    // however only dim-components are used for flux.
    for (int d = 0; d < dim; d++) flux(nvel + 2 + sp, d) = -state[nvel + 2 + sp] * diffusionVelocity(sp, d);
  }
}

MFEM_HOST_DEVICE void Fluxes::ComputeViscousFluxes(const double *state, const double *gradUp, double radius,
                                                   double *flux) {
  for (int d = 0; d < dim; d++) {
    for (int eq = 0; eq < num_equation; eq++) {
      flux[eq + d * num_equation] = 0.;
    }
  }
  if (eqSystem == EULER) {
    return;
  }

  double vel[gpudata::MAXDIM];
  double vtmp[gpudata::MAXDIM];
  double stress[gpudata::MAXDIM * gpudata::MAXDIM];

  // TODO(kevin): update E-field with EM coupling.
  double Efield[gpudata::MAXDIM];
  for (int v = 0; v < nvel; v++) Efield[v] = 0.0;

  const int numSpecies = mixture->GetNumSpecies();
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  double speciesEnthalpies[gpudata::MAXSPECIES];
  mixture->computeSpeciesEnthalpies(state, speciesEnthalpies);

  double transportBuffer[FluxTrns::NUM_FLUX_TRANS];
  // NOTE(kevin): in flux, only dim-components of diffusionVelocity will be used.
  double diffusionVelocity[gpudata::MAXSPECIES * gpudata::MAXDIM];
  transport->ComputeFluxTransportProperties(state, gradUp, Efield, transportBuffer, diffusionVelocity);
  const double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];

  if (twoTemperature) {
    for (int d = 0; d < dim; d++) {
      double qeFlux = ke * gradUp[num_equation - 1 + d * num_equation];
      flux[1 + nvel + d * num_equation] += qeFlux;
      flux[num_equation - 1 + d * num_equation] += qeFlux;
      flux[num_equation - 1 + d * num_equation] -=
          speciesEnthalpies[numSpecies - 2] * diffusionVelocity[numSpecies - 2 + d * numSpecies];
    }
  } else {
    k += ke;
  }

  const double ur = (axisymmetric_ ? state[1] / state[0] : 0);
  const double ut = (axisymmetric_ ? state[3] / state[0] : 0);

  // make sure density visc. flux is 0
  for (int d = 0; d < dim; d++) flux[0 + d * num_equation] = 0.;

  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      stress[i + j * dim] = gradUp[(1 + j) + i * num_equation] + gradUp[(1 + i) + j * num_equation];
    }
    divV += gradUp[(1 + i) + i * num_equation];
  }
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) stress[i + j * dim] *= visc;

  if (axisymmetric_ && radius > 0) {
    divV += ur / radius;
  }

  for (int i = 0; i < dim; i++) stress[i + i * dim] += bulkViscosity * divV;

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) flux[(1 + i) + j * num_equation] = stress[i + j * dim];

  double tau_tr = 0, tau_tz = 0;
  if (axisymmetric_) {
    const double ut_r = gradUp[3 + 0 * num_equation];
    const double ut_z = gradUp[3 + 1 * num_equation];
    tau_tr = ut_r;
    if (radius > 0) tau_tr -= ut / radius;
    tau_tr *= visc;

    tau_tz = visc * ut_z;

    flux[(1 + 2) + 0 * num_equation] = tau_tr;
    flux[(1 + 2) + 1 * num_equation] = tau_tz;
  }

  // temperature gradient
  //       for (int d = 0; d < dim; d++) gradT[d] = temp * (gradUp(1 + dim, d) / p - gradUp(0, d) / state[0]);

  for (int d = 0; d < dim; d++) vel[d] = state[1 + d] / state[0];

  // stress.Mult(vel, vtmp);
  for (int i = 0; i < dim; i++) {
    vtmp[i] = 0.0;
    for (int j = 0; j < dim; j++) vtmp[i] += stress[i + j * dim] * vel[j];
  }

  for (int d = 0; d < dim; d++) {
    flux[(1 + nvel) + d * num_equation] += vtmp[d];
    flux[(1 + nvel) + d * num_equation] += k * gradUp[(1 + nvel) + d * num_equation];
    // compute diffusive enthalpy flux.
    for (int sp = 0; sp < numSpecies; sp++) {
      flux[(1 + nvel) + d * num_equation] -= speciesEnthalpies[sp] * diffusionVelocity[sp + d * numSpecies];
    }
  }

  if (axisymmetric_) {
    flux[(1 + nvel) + 0 * num_equation] += ut * tau_tr;
    flux[(1 + nvel) + 1 * num_equation] += ut * tau_tz;
  }

  // if (eqSystem == NS_PASSIVE) {
  //   double Sc = mixture->GetSchmidtNum();
  //   for (int d = 0; d < dim; d++) flux(num_equation - 1, d) = visc / Sc * gradUp(num_equation - 1, d);
  // }
  // NOTE: NS_PASSIVE will not be needed (automatically incorporated).
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // NOTE: diffusionVelocity is set to be (numSpecies,nvel)-matrix.
    // however only dim-components are used for flux.
    for (int d = 0; d < dim; d++)
      flux[(nvel + 2 + sp) + d * num_equation] = -state[nvel + 2 + sp] * diffusionVelocity[sp + d * numSpecies];
  }
}

void Fluxes::ComputeBdrViscousFluxes(const Vector &state, const DenseMatrix &gradUp, double radius,
                                     const BoundaryViscousFluxData &bcFlux, Vector &normalFlux) {
  normalFlux.SetSize(num_equation);
  normalFlux = 0.;
  if (eqSystem == EULER) {
    return;
  }

  DenseMatrix stress(dim, dim);

  // TODO(kevin): update E-field with EM coupling.
  Vector Efield(nvel);
  Efield = 0.0;

  const int numSpecies = mixture->GetNumSpecies();
  const int numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool twoTemperature = mixture->IsTwoTemperature();

  Vector speciesEnthalpies(numSpecies);
  mixture->computeSpeciesEnthalpies(state, speciesEnthalpies);

  Vector transportBuffer;
  // NOTE(kevin): in flux, only dim-components of diffusionVelocity will be used.
  DenseMatrix diffusionVelocity(numSpecies, nvel);
  transport->ComputeFluxTransportProperties(state, gradUp, Efield, transportBuffer, diffusionVelocity);
  const double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];

  // Primitive viscous fluxes.
  const int primFluxSize = (twoTemperature) ? numSpecies + nvel + 2 : numSpecies + nvel + 1;
  Vector normalPrimFlux(primFluxSize);
  normalPrimFlux = 0.0;

  // Compute the diffusion velocities.
  for (int sp = 0; sp < numSpecies; sp++) {
    // NOTE: diffusionVelocity is set to be (numSpecies,nvel)-matrix.
    // however only dim-components are used for flux.
    for (int d = 0; d < dim; d++) normalPrimFlux(sp) += diffusionVelocity(sp, d) * bcFlux.normal(d);
  }

  // Replace with the prescribed boundary fluxes.
  for (int i = 0; i < numSpecies; i++) {
    if (bcFlux.primFluxIdxs[i]) normalPrimFlux(i) = bcFlux.primFlux(i);
  }

  // Compute the stress.
  const double ur = (axisymmetric_ ? state[1] / state[0] : 0);
  const double ut = (axisymmetric_ ? state[3] / state[0] : 0);

  double divV = 0.;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) stress(i, j) = gradUp(1 + j, i) + gradUp(1 + i, j);
    divV += gradUp(1 + i, i);
  }
  stress *= visc;

  if (axisymmetric_ && radius > 0) {
    divV += ur / radius;
  }

  for (int i = 0; i < dim; i++) stress(i, i) += bulkViscosity * divV;

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) normalPrimFlux(numSpecies + i) += stress(i, j) * bcFlux.normal(j);
  }

  double tau_tr = 0, tau_tz = 0;
  if (axisymmetric_) {
    const double ut_r = gradUp(3, 0);
    const double ut_z = gradUp(3, 1);
    tau_tr = ut_r;
    if (radius > 0) tau_tr -= ut / radius;
    tau_tr *= visc;

    tau_tz = visc * ut_z;

    normalPrimFlux(numSpecies + nvel - 1) += tau_tr * bcFlux.normal(0);
    normalPrimFlux(numSpecies + nvel - 1) += tau_tz * bcFlux.normal(1);
  }

  // Compute the electron heat flux.
  if (twoTemperature) {
    // NOTE(kevin): followed the standard sign of heat flux.
    for (int d = 0; d < dim; d++)
      normalPrimFlux(primFluxSize - 1) -= ke * gradUp(num_equation - 1, d) * bcFlux.normal(d);
    normalPrimFlux(primFluxSize - 1) += speciesEnthalpies(numSpecies - 2) * normalPrimFlux(numSpecies - 2);
  } else {
    k += ke;
  }
  // Compute the heavies heat flux.
  // NOTE(kevin): followed the standard sign of heat flux.
  for (int d = 0; d < dim; d++) normalPrimFlux(numSpecies + nvel) -= k * gradUp(1 + nvel, d) * bcFlux.normal(d);
  for (int sp = 0; sp < numSpecies; sp++) {
    if (twoTemperature && (sp == numSpecies - 2)) continue;
    normalPrimFlux(numSpecies + nvel) += speciesEnthalpies(sp) * normalPrimFlux(sp);
  }

  // Replace with the prescribed boundary fluxes.
  for (int i = numSpecies; i < primFluxSize; i++) {
    if (bcFlux.primFluxIdxs[i]) normalPrimFlux(i) = bcFlux.primFlux(i);
  }

  Vector vel0(nvel);
  for (int d = 0; d < nvel; d++) vel0(d) = state[1 + d] / state[0];

  // convert back to the conservative viscous flux.
  // NOTE(kevin): negative sign comes in from the definition of numerical viscous flux.
  // normalFlux(0) = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    normalFlux(nvel + 2 + sp) = -state[nvel + 2 + sp] * normalPrimFlux(sp);
  }
  for (int d = 0; d < nvel; d++) normalFlux(d + 1) = normalPrimFlux(numSpecies + d);
  for (int d = 0; d < nvel; d++) normalFlux(nvel + 1) += normalPrimFlux(numSpecies + d) * vel0(d);
  normalFlux(nvel + 1) -= normalPrimFlux(numSpecies + nvel);
  if (twoTemperature) {
    normalFlux(nvel + 1) -= normalPrimFlux(primFluxSize - 1);
    normalFlux(num_equation - 1) = -normalPrimFlux(primFluxSize - 1);
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

  MFEM_FORALL(n, dof, {
    double Un[gpudata::MAXEQUATIONS]; // double Un[20];
    double KE[gpudata::MAXDIM]; // double KE[3];
    double p;

    for (int eq = 0; eq < num_equation; eq++) {
      Un[eq] = dataIn[n + eq * dof];
    }
    for (int d = 0; d < dim; d++) KE[d] = 0.5 * Un[1 + d] * Un[1 + d] / Un[0];
    if (dim != 3) KE[2] = 0.;

    p = DryAir::pressure(&Un[0], &KE[0], gamma, dim, num_equation);

    for (int eq = 0; eq < num_equation; eq++) {
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
    }
  });
#endif // _GPU_
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
  MFEM_FORALL(n, dof,
  {
    double Un[gpudata::MAXEQUATIONS]; // double Un[5];
    double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM]; // double gradUpn[5 * 3];
    double vFlux[gpudata::MAXEQUATIONS * gpudata::MAXDIM]; // double vFlux[5 * 3];
    double linVisc;

    // init. State
    for (int eq = 0; eq < num_equation; eq++) {
      Un[eq] = dataIn[n + eq * dof];

      for (int d = 0; d < dim; d++) {
        gradUpn[eq + d * num_equation] = d_gradUp[n + eq * dof + d * dof * num_equation];
      }
    }

    // Fluxes::viscousFlux_gpu(&vFlux[0], &Un[0], &gradUpn[0], eqSystem, gamma, Rg, viscMult, bulkViscMult,
    //                         Pr, Sc, eq, num_equation, dim, num_equation);
    Fluxes::viscousFlux_serial_gpu(&vFlux[0], &Un[0], &gradUpn[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                   num_equation);


    MFEM_SYNC_THREAD;

    if (d_spaceVaryViscMult != NULL) {
      linVisc = d_spaceVaryViscMult[n];

      for (int d = 0; d < dim; d++) {
        for (int eq = 0; eq < num_equation; eq++) {
          vFlux[eq + d * num_equation] *= linVisc;
        }
      }
    }

    // write to global memory
    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) {
        d_flux[n + d * dof + eq * dof * dim] -= vFlux[eq + d * num_equation];
      }
    }
  });  // end MFEM_FORALL
#endif
}
// clang-format on
