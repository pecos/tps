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
#ifndef TRANSPORT_PROPERTIES_HPP_
#define TRANSPORT_PROPERTIES_HPP_

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <grvy.h>
#include <tps_config.h>

// #include <general/forall.hpp>
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;

class TransportProperties {
 protected:
  int num_equation;
  int dim;
  int nvel_;
  int numSpecies;
  int numActiveSpecies;
  bool ambipolar;
  bool twoTemperature_;

  GasMixture *mixture;

  // Array<bool> isComputed;
  // SpeciesPrimitiveType speciesPrimitiveType;

  // NOTE: when species primitives (X, Y, n) are in a denominator, this noise is added to avoid dividing-by-zero.
  const double Xeps_ = 1.0e-30;

 public:
  // TransportProperties(GasMixture *_mixture);
  MFEM_HOST_DEVICE TransportProperties(GasMixture *_mixture);

  // virtual ~TransportProperties() {}
  MFEM_HOST_DEVICE virtual ~TransportProperties() {}

  // Currently, diffusion velocity is evaluated together with transport properties,
  // though diffusion velocity can vary according to models we use.
  // In some cases such variations may be independent of transport property models,
  // but mostly changing diffusion velocity involves changes in tranport property models as well.

  // Currently, transport properties are evaluated in flux and source term separately.
  // Flux does not take primitive variables as input, rather evaluate them whenever needed.
  // ComputeFluxTransportProperties also evaluates required primitive variables,
  // but do not return it as output.
  // TODO(kevin): need to discuss whether to reuse computed primitive variables in flux evaluation,
  // or in general evaluation of primitive variables.
  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity) = 0;
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double *transportBuffer,
                                                               double *diffusionVelocity) = 0;

  // Source term will be constructed using ForcingTerms, which have pointers to primitive variables.
  // So we can use them in evaluating transport properties.
  // If this routine evaluate additional primitive variables, can return them just as the routine above.
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp) = 0;
  // NOTE: only for AxisymmetricSource
  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc) = 0;

  // For mixture-averaged diffusion, correct for mass conservation.
  void correctMassDiffusionFlux(const Vector &Y_sp, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE void correctMassDiffusionFlux(const double *Y_sp, double *diffusionVelocity);

  // compute electric conductivity for mixture-averaged diffusions.
  // NOTE: in unit of ELECTRONCHARGE * AVOGADRONUMBER.
  double computeMixtureElectricConductivity(const Vector &mobility, const Vector &n_sp);
  MFEM_HOST_DEVICE double computeMixtureElectricConductivity(const double *mobility, const double *n_sp);

  // These are only for mixture-averaged diffusivity models.
  void addAmbipolarEfield(const Vector &mobility, const Vector &n_sp, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE void addAmbipolarEfield(const double *mobility, const double *n_sp, double *diffusionVelocity);
  void addMixtureDrift(const Vector &mobility, const Vector &n_sp, const Vector &Efield,
                       DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE void addMixtureDrift(const double *mobility, const double *n_sp, const double *Efield,
                                        double *diffusionVelocity);
  double linearAverage(const Vector &X_sp, const Vector &speciesTransport);
  MFEM_HOST_DEVICE double linearAverage(const double *X_sp, const double *speciesTransport);
  // Curtiss-Hirschfelder approximation of diffusivity.
  void CurtissHirschfelder(const Vector &X_sp, const Vector &Y_sp, const DenseMatrix &binaryDiff,
                           Vector &avgDiff);
  // MFEM_HOST_DEVICE void CurtissHirschfelder(const double *X_sp, const double *Y_sp, const double *binaryDiff,
  //                                           double *avgDiff);
};

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

class DryAirTransport : public TransportProperties {
 protected:
  double gas_constant;
  double visc_mult;
  double bulk_visc_mult;
  double thermalConductivity;

  // Prandtl number
  double Pr;         // Prandtl number
  double cp_div_pr;  // cp divided by Pr (used in conductivity calculation)

  // Fick's law
  double Sc;  // Schmidt number

 public:
  DryAirTransport(GasMixture *_mixture, RunConfiguration &_runfile);
  MFEM_HOST_DEVICE DryAirTransport(GasMixture *_mixture, const double viscosity_multiplier,
                                   const double bulk_viscosity);

  MFEM_HOST_DEVICE virtual ~DryAirTransport() {}

  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double *transportBuffer,
                                                               double *diffusionVelocity);
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp) {}

  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc);
};

inline void DryAirTransport::GetViscosities(const Vector &conserved, const Vector &primitive, double &visc,
                                            double &bulkVisc) {
  double temp = primitive[1 + nvel_];
  visc = (1.458e-6 * visc_mult * pow(temp, 1.5) / (temp + 110.4));
  bulkVisc = bulk_visc_mult * visc;
  return;
}

//////////////////////////////////////////////////////
//////// Constant Transport
//////////////////////////////////////////////////////

class ConstantTransport : public TransportProperties {
 protected:
  double viscosity_;
  double bulkViscosity_;
  Vector diffusivity_;
  double thermalConductivity_;
  double electronThermalConductivity_;
  Vector mtFreq_;

  int electronIndex_ = -1;
  const double qeOverkB_ = ELECTRONCHARGE / BOLTZMANNCONSTANT;

 public:
  ConstantTransport(GasMixture *_mixture, RunConfiguration &_runfile);

  MFEM_HOST_DEVICE virtual ~ConstantTransport() {}

  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double *transportBuffer,
                                                               double *diffusionVelocity) {
    exit(-1);
    return;
  }
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);

  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc);
};

inline void ConstantTransport::GetViscosities(const Vector &conserved, const Vector &primitive, double &visc,
                                              double &bulkVisc) {
  visc = viscosity_;
  bulkVisc = bulkViscosity_;
  return;
}

#endif  // TRANSPORT_PROPERTIES_HPP_
