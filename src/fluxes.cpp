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

Fluxes::Fluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport, const int _num_equation,
               const int _dim, bool axisym)
    : mixture(_mixture),
      eqSystem(_eqSystem),
      config_(NULL),
      transport(_transport),
      dim(_dim),
      axisymmetric_(axisym),
      num_equation(_num_equation),
      sgs_model_type_(0),
      sgs_model_floor_(0.0),
      sgs_model_const_(0.0) {
  nvel = axisym ? 3 : dim;
  vsd_.enabled = false;
  vsd_.n[0] = vsd_.n[1] = vsd_.n[2] = 0.;
  vsd_.p[0] = vsd_.p[1] = vsd_.p[2] = 0.;
  vsd_.ratio = 1.0;
  vsd_.width = 1.0;
}

Fluxes::Fluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport, const int _num_equation,
               const int _dim, bool axisym, RunConfiguration *config)
    : mixture(_mixture),
      eqSystem(_eqSystem),
      config_(config),
      transport(_transport),
      dim(_dim),
      axisymmetric_(axisym),
      num_equation(_num_equation),
      sgs_model_type_(config->GetSgsModelType()),
      sgs_model_floor_(config->GetSgsFloor()),
      sgs_model_const_(config->GetSgsConstant()) {
  nvel = axisym ? 3 : dim;
  vsd_.enabled = config_->linViscData.isEnabled;
  vsd_.n[0] = vsd_.n[1] = vsd_.n[2] = 0.;
  vsd_.p[0] = vsd_.p[1] = vsd_.p[2] = 0.;
  vsd_.ratio = 1.0;
  vsd_.width = 1.0;

  // Pull viscous sponge data out of config if we are going to use it
  if (vsd_.enabled) {
    for (int d = 0; d < dim; d++) {
      vsd_.n[d] = config_->GetLinearVaryingData().normal(d);
      vsd_.p[d] = config_->GetLinearVaryingData().point0(d);
    }

    // ensure normal is actually a unit normal
    double Nmag = 0;
    for (int d = 0; d < dim; d++) Nmag += vsd_.n[d] * vsd_.n[d];
    Nmag = sqrt(Nmag);
    for (int d = 0; d < dim; d++) vsd_.n[d] /= Nmag;

    vsd_.ratio = config_->GetLinearVaryingData().viscRatio;
    vsd_.width = config_->GetLinearVaryingData().width;
  }
}

MFEM_HOST_DEVICE Fluxes::Fluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                const int _num_equation, const int _dim, bool axisym, int sgs_type, double sgs_floor,
                                double sgs_const, viscositySpongeData vsd)
    : mixture(_mixture),
      eqSystem(_eqSystem),
      config_(NULL),
      transport(_transport),
      dim(_dim),
      axisymmetric_(axisym),
      num_equation(_num_equation),
      sgs_model_type_(sgs_type),
      sgs_model_floor_(sgs_floor),
      sgs_model_const_(sgs_const) {
  nvel = axisym ? 3 : dim;
  vsd_.enabled = vsd.enabled;
  vsd_.n[0] = vsd_.n[1] = vsd_.n[2] = 0.;
  vsd_.p[0] = vsd_.p[1] = vsd_.p[2] = 0.;
  vsd_.ratio = 1.0;
  vsd_.width = 1.0;

  // Pull viscous sponge data out of config if we are going to use it
  if (vsd_.enabled) {
    for (int d = 0; d < dim; d++) {
      vsd_.n[d] = vsd.n[d];
      vsd_.p[d] = vsd.p[d];
    }
    vsd_.ratio = vsd.ratio;
    vsd_.width = vsd.width;
  }
}

void Fluxes::ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux) {
  ComputeConvectiveFluxes(state.GetData(), flux.GetData());
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
void Fluxes::ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, Vector transip, double delta,
                                  double distance, DenseMatrix &flux) {
  ComputeViscousFluxes(state.GetData(), gradUp.GetData(), transip.GetData(), delta, distance, flux.GetData());
}

MFEM_HOST_DEVICE void Fluxes::ComputeViscousFluxes(const double *state, const double *gradUp, double *transip,
                                                   double delta, double distance, double *flux) {
  for (int d = 0; d < dim; d++) {
    for (int eq = 0; eq < num_equation; eq++) {
      flux[eq + d * num_equation] = 0.;
    }
  }
  if (eqSystem == EULER) {
    return;
  }

  // if axisymmetric, radius is the x coordinate
  double radius = -1;
  if (axisymmetric_) {
    radius = transip[0];
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

  transport->ComputeFluxTransportProperties(state, gradUp, Efield, radius, distance, transportBuffer,
                                            diffusionVelocity);
  double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];
  double Pr_Cp = visc / k;

  // subgrid scale model
  if (sgs_model_type_ > 0) {
    double mu_sgs = 0.;
    if (sgs_model_type_ == 1) sgsSmag(state, gradUp, delta, mu_sgs);
    if (sgs_model_type_ == 2) sgsSigma(state, gradUp, delta, mu_sgs);
    bulkViscosity *= (1.0 + mu_sgs / visc);
    visc += mu_sgs;
    k += (mu_sgs / Pr_Cp);
  }

  // viscous sponge
  if (vsd_.enabled) {
    double wgt = 0.;
    viscSpongePlanar(transip, wgt);
    visc *= wgt;
    bulkViscosity *= wgt;
    k *= wgt;
  }

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

void Fluxes::ComputeBdrViscousFluxes(const Vector &state, const DenseMatrix &gradUp, Vector transip, double delta,
                                     double distance, const BoundaryViscousFluxData &bcFlux, Vector &normalFlux) {
  normalFlux.SetSize(num_equation);
  ComputeBdrViscousFluxes(state.GetData(), gradUp.GetData(), transip.GetData(), delta, distance, bcFlux,
                          normalFlux.GetData());
}

MFEM_HOST_DEVICE void Fluxes::ComputeBdrViscousFluxes(const double *state, const double *gradUp, double *transip,
                                                      double delta, double distance,
                                                      const BoundaryViscousFluxData &bcFlux, double *normalFlux) {
  // normalFlux.SetSize(num_equation);
  for (int eq = 0; eq < num_equation; eq++) normalFlux[eq] = 0.;
  if (eqSystem == EULER) {
    return;
  }

  // if axisymmetric, radius is the x coordinate
  double radius = -1;
  if (axisymmetric_) {
    radius = transip[0];
  }

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

  transport->ComputeFluxTransportProperties(state, gradUp, Efield, radius, distance, transportBuffer,
                                            diffusionVelocity);
  double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];
  double Pr_Cp = visc / k;

  // subgrid scale model
  if (sgs_model_type_ > 0) {
    double mu_sgs = 0.;
    if (sgs_model_type_ == 1) sgsSmag(state, gradUp, delta, mu_sgs);
    if (sgs_model_type_ == 2) sgsSigma(state, gradUp, delta, mu_sgs);
    bulkViscosity *= (1.0 + mu_sgs / visc);
    visc += mu_sgs;
    k += (mu_sgs / Pr_Cp);
  }

  // viscous sponge
  if (vsd_.enabled) {
    double wgt = 0.;
    viscSpongePlanar(transip, wgt);
    visc *= wgt;
    bulkViscosity *= wgt;
    k *= wgt;
  }

  // Primitive viscous fluxes.
  const int primFluxSize = (twoTemperature) ? numSpecies + nvel + 2 : numSpecies + nvel + 1;
  double normalPrimFlux[gpudata::MAXEQUATIONS];
  for (int eq = 0; eq < primFluxSize; eq++) normalPrimFlux[eq] = 0.0;

  // Compute the diffusion velocities.
  for (int sp = 0; sp < numSpecies; sp++) {
    // NOTE: diffusionVelocity is set to be (numSpecies,nvel)-matrix.
    // however only dim-components are used for flux.
    for (int d = 0; d < dim; d++) normalPrimFlux[sp] += diffusionVelocity[sp + d * numSpecies] * bcFlux.normal[d];
  }

  // Replace with the prescribed boundary fluxes.
  for (int i = 0; i < numSpecies; i++) {
    if (bcFlux.primFluxIdxs[i]) normalPrimFlux[i] = bcFlux.primFlux[i];
  }

  // Compute the stress.
  const double ur = (axisymmetric_ ? state[1] / state[0] : 0);
  const double ut = (axisymmetric_ ? state[3] / state[0] : 0);

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

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) normalPrimFlux[numSpecies + i] += stress[i + j * dim] * bcFlux.normal[j];
  }

  double tau_tr = 0, tau_tz = 0;
  if (axisymmetric_) {
    const double ut_r = gradUp[3 + 0 * num_equation];
    const double ut_z = gradUp[3 + 1 * num_equation];
    tau_tr = ut_r;
    if (radius > 0) tau_tr -= ut / radius;
    tau_tr *= visc;

    tau_tz = visc * ut_z;

    normalPrimFlux[numSpecies + nvel - 1] += tau_tr * bcFlux.normal[0];
    normalPrimFlux[numSpecies + nvel - 1] += tau_tz * bcFlux.normal[1];
  }

  // Compute the electron heat flux.
  if (twoTemperature) {
    // NOTE(kevin): followed the standard sign of heat flux.
    for (int d = 0; d < dim; d++)
      normalPrimFlux[primFluxSize - 1] -= ke * gradUp[(num_equation - 1) + d * num_equation] * bcFlux.normal[d];
    normalPrimFlux[primFluxSize - 1] += speciesEnthalpies[numSpecies - 2] * normalPrimFlux[numSpecies - 2];
  } else {
    k += ke;
  }
  // Compute the heavies heat flux.
  // NOTE(kevin): followed the standard sign of heat flux.
  for (int d = 0; d < dim; d++)
    normalPrimFlux[numSpecies + nvel] -= k * gradUp[(1 + nvel) + d * num_equation] * bcFlux.normal[d];
  for (int sp = 0; sp < numSpecies; sp++) {
    if (twoTemperature && (sp == numSpecies - 2)) continue;
    normalPrimFlux[numSpecies + nvel] += speciesEnthalpies[sp] * normalPrimFlux[sp];
  }

  // Replace with the prescribed boundary fluxes.
  for (int i = numSpecies; i < primFluxSize; i++) {
    if (bcFlux.primFluxIdxs[i]) normalPrimFlux[i] = bcFlux.primFlux[i];
  }

  double vel0[gpudata::MAXDIM];
  for (int d = 0; d < nvel; d++) vel0[d] = state[1 + d] / state[0];

  // convert back to the conservative viscous flux.
  // NOTE(kevin): negative sign comes in from the definition of numerical viscous flux.
  // normalFlux(0) = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    normalFlux[nvel + 2 + sp] = -state[nvel + 2 + sp] * normalPrimFlux[sp];
  }
  for (int d = 0; d < nvel; d++) normalFlux[d + 1] = normalPrimFlux[numSpecies + d];
  for (int d = 0; d < nvel; d++) normalFlux[nvel + 1] += normalPrimFlux[numSpecies + d] * vel0[d];
  normalFlux[nvel + 1] -= normalPrimFlux[numSpecies + nvel];
  if (twoTemperature) {
    normalFlux[nvel + 1] -= normalPrimFlux[primFluxSize - 1];
    normalFlux[num_equation - 1] = -normalPrimFlux[primFluxSize - 1];
  }
}

/**
Basic Smagorinksy subgrid model with user-specified cutoff grid length
*/
void Fluxes::sgsSmag(const Vector &state, const DenseMatrix &gradUp, double delta, double &mu) {
  sgsSmag(state.GetData(), gradUp.GetData(), delta, mu);
}

MFEM_HOST_DEVICE void Fluxes::sgsSmag(const double *state, const double *gradUp, double delta, double &mu) {
  double Sij[6];
  double Smag = 0.;
  double Cd = sgs_model_const_;  // user-set, defaults to 0.12
  double l_floor;
  double d_model;

  // gradUp is in (eq,dim) form with eq=0 being rho slot
  Sij[0] = gradUp[1 + 0 * num_equation];                                         // gradUp(1, 0);
  Sij[1] = gradUp[2 + 1 * num_equation];                                         // gradUp(2, 1);
  Sij[2] = gradUp[3 + 2 * num_equation];                                         // gradUp(3, 2);
  Sij[3] = 0.5 * (gradUp[1 + 1 * num_equation] + gradUp[2 + 0 * num_equation]);  // 0.5 * (gradUp(1, 1) + gradUp(2, 0));
  Sij[4] = 0.5 * (gradUp[1 + 2 * num_equation] + gradUp[3 + 0 * num_equation]);  // 0.5 * (gradUp(1, 2) + gradUp(3, 0));
  Sij[5] = 0.5 * (gradUp[2 + 2 * num_equation] + gradUp[3 + 1 * num_equation]);  // 0.5 * (gradUp(2, 2) + gradUp(3, 1));

  // strain magnitude with silly sqrt(2) factor
  for (int i = 0; i < 3; i++) Smag += Sij[i] * Sij[i];
  for (int i = 3; i < 6; i++) Smag += 2.0 * Sij[i] * Sij[i];
  Smag = sqrt(2.0 * Smag);

  // eddy viscosity with delta shift
  l_floor = sgs_model_floor_;
  d_model = Cd * max(delta - l_floor, 0.0);
  mu = state[0] * d_model * d_model * Smag;
}

/**
Sigma subgrid model following Nicoud et.al., "Using singular values to build a
subgrid-scale model for large eddy simulations", PoF 2011.
*/
void Fluxes::sgsSigma(const Vector &state, const DenseMatrix &gradUp, double delta, double &mu) {
  sgsSigma(state.GetData(), gradUp.GetData(), delta, mu);
}

MFEM_HOST_DEVICE void Fluxes::sgsSigma(const double *state, const double *gradUp, double delta, double &mu) {
  double Cd = sgs_model_const_;  // user-set, defaults to 0.135
  double sml = 1.0e-12;
  double l_floor, d_model, d4;

  // shifted grid scale, d should really be sqrt of J^T*J
  l_floor = sgs_model_floor_;
  d_model = max((delta - l_floor), sml);

#if !defined(_CUDA_) && !defined(_HIP_) && defined(MFEM_USE_LAPACK)
  // If we have access to lapack---i.e., we're not on the device and
  // we have mfem with lapack support---then compute the singular
  // values of the velocity gradient directly.  This approach is more
  // stable wrt perturbations in the incoming velocity gradient than
  // first computing g^T * g as is done below.  But, we don't always
  // have lapack, and it is more expensive.
  DenseMatrix gmat(3, 3);
  Vector sigma(3);

  // gmat_ij = u_{i,j}
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      gmat(i, j) = gradUp[(i + 1) + j * num_equation];
    }
  }

  gmat.SingularValues(sigma);

  // eddy viscosity
  mu = sigma[2] * (sigma[0] - sigma[1]) * (sigma[1] - sigma[2]);
  mu = max(mu, 0.0);
  mu /= (sigma[0] * sigma[0]);
  mu *= (Cd * Cd * d_model * d_model);
  mu *= state[0];
#else
  // Otherwise, fall back to approach based on eigenvalues of g^T * g.
  // This approach is more sensitive to perturbations in g, because
  // the singular values are not as accurate, but doesn't use any
  // external libraries, so we can always do it.
  double Qij[3][3];
  double B[3][3];
  double ev[3];
  double sigma[3];
  double pi = 3.14159265359;
  double onethird = 1. / 3.;
  double p1, p2, p, q, detB, r, phi;

  // Qij = u_{k,i}*u_{k,j}
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      Qij[i][j] = 0;
      for (int k = 0; k < dim; k++) {
        Qij[i][j] += gradUp[k + 1 + i * num_equation] * gradUp[k + 1 + j * num_equation];
      }
    }
  }

  d4 = pow(d_model, 4);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      Qij[i][j] *= d4;
    }
  }

  // eigenvalues for symmetric pos-def 3x3
  p1 = Qij[0][1] * Qij[0][1] + Qij[0][2] * Qij[0][2] + Qij[1][2] * Qij[1][2];
  q = onethird * (Qij[0][0] + Qij[1][1] + Qij[2][2]);
  p2 = (Qij[0][0] - q) * (Qij[0][0] - q) + (Qij[1][1] - q) * (Qij[1][1] - q) + (Qij[2][2] - q) * (Qij[2][2] - q) +
       2.0 * p1;
  p = std::sqrt(max(p2, 0.0) / 6.0);

  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      B[i][j] = Qij[i][j];
    }
  }
  for (int i = 0; i < dim; i++) {
    B[i][i] -= q;
  }
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      B[i][j] *= (1.0 / max(p, sml));
    }
  }
  detB = B[0][0] * (B[1][1] * B[2][2] - B[2][1] * B[1][2]) - B[0][1] * (B[1][0] * B[2][2] - B[2][0] * B[1][2]) +
         B[0][2] * (B[1][0] * B[2][1] - B[2][0] * B[1][1]);
  r = 0.5 * detB;

  if (r <= -1.0) {
    phi = onethird * pi;
  } else if (r >= 1.0) {
    phi = 0.0;
  } else {
    phi = onethird * acos(r);
  }

  // eigenvalues satisfy eig3 <= eig2 <= eig1 (L^4/T^2)
  ev[0] = q + 2.0 * p * cos(phi);
  ev[2] = q + 2.0 * p * cos(phi + (2.0 * onethird * pi));
  ev[1] = 3.0 * q - ev[0] - ev[2];

  // actual sigma (L^2/T)
  sigma[0] = sqrt(max(ev[0], sml));
  sigma[1] = sqrt(max(ev[1], sml));
  sigma[2] = sqrt(max(ev[2], sml));

  // eddy viscosity
  mu = sigma[2] * (sigma[0] - sigma[1]) * (sigma[1] - sigma[2]);
  mu = max(mu, 0.0);
  mu /= (sigma[0] * sigma[0]);
  mu *= (Cd * Cd);
  mu *= state[0];
#endif

  // shouldnt be necessary
  if (mu != mu) mu = 0.0;
}

/**
Simple planar viscous sponge layer with smooth tanh-transtion using user-specified width and
total amplification.  Note: duplicate in M2
*/
MFEM_HOST_DEVICE void Fluxes::viscSpongePlanar(double *x, double &wgt) {
  double s[3];
  double factor, width, dist;

  // get settings
  factor = max(vsd_.ratio, 1.0);
  width = vsd_.width;

  // distance from plane
  dist = 0.;
  for (int d = 0; d < dim; d++) s[d] = (x[d] - vsd_.p[d]);
  for (int d = 0; d < dim; d++) dist += s[d] * vsd_.n[d];

  // weight
  wgt = 0.5 * (tanh(dist / width - 2.0) + 1.0);
  wgt *= (factor - 1.0);
  wgt += 1.0;
}

// clang-format on
