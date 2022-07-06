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
#ifndef ARGON_TRANSPORT_HPP_
#define ARGON_TRANSPORT_HPP_

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <grvy.h>
#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "collision_integrals.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace mfem;
using namespace std;
// using namespace charged;
// using namespace argon;

//////////////////////////////////////////////////////
//////// Argon Minimal Transport (ternary mixture)
//////////////////////////////////////////////////////

class ArgonMinimalTransport : public TransportProperties {
 protected:
  int electronIndex_ = -1;
  int ionIndex_ = -1;
  int neutralIndex_ = -1;

  const double kB_ = BOLTZMANNCONSTANT;
  const double eps0_ = VACUUMPERMITTIVITY;
  const double qe_ = ELECTRONCHARGE;
  const double R_ = UNIVERSALGASCONSTANT;
  const double debyeFactor_ = kB_ * eps0_ / qe_ / qe_;
  // const double PI_ = 4.0 * atan(1.0);
  const double PI_ = PI;
  const double qeOverkB_ = qe_ / kB_;

  // standard Chapman-Enskog coefficients
  const double viscosityFactor_ = 5. / 16. * sqrt(PI_ * kB_);
  const double kOverEtaFactor_ = 15. / 4. * kB_;
  const double diffusivityFactor_ = 3. / 16. * sqrt(2.0 * PI_ * kB_) / AVOGADRONUMBER;
  const double mfFreqFactor_ = 4. / 3. * AVOGADRONUMBER * sqrt(8. * kB_ / PI_);

  // molecular mass per each particle. [kg^-1]
  double mw_[gpudata::MAXSPECIES];
  double muw_[gpudata::MAXSPECIES * gpudata::MAXSPECIES];  // effective mass

  bool thirdOrderkElectron_;

 public:
  ArgonMinimalTransport(GasMixture *_mixture, RunConfiguration &_runfile);
  MFEM_HOST_DEVICE ArgonMinimalTransport(GasMixture *_mixture, const ArgonTransportInput &inputs);
  ArgonMinimalTransport(GasMixture *_mixture);

  MFEM_HOST_DEVICE virtual ~ArgonMinimalTransport() {}

  MFEM_HOST_DEVICE double getMuw(const int &spI, const int &spJ) { return muw_[spI + spJ * numSpecies]; }

  int getIonIndex() { return ionIndex_; }

  virtual collisionInputs computeCollisionInputs(const Vector &primitive, const Vector &n_sp);
  MFEM_HOST_DEVICE collisionInputs computeCollisionInputs(const double *primitive, const double *n_sp);

  // Currently, transport properties are evaluated in flux and source term separately.
  // Flux does not take primitive variables as input, rather evaluate them whenever needed.
  // ComputeFluxTransportProperties also evaluates required primitive variables,
  // but do not return it as output.
  // TODO(kevin): need to discuss whether to reuse computed primitive variables in flux evaluation,
  // or in general evaluation of primitive variables.
  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double *transportBuffer,
                                                               double *diffusionVelocity);

  // Source term will be constructed using ForcingTerms, which have pointers to primitive variables.
  // So we can use them in evaluating transport properties.
  // If this routine evaluate additional primitive variables, can return them just as the routine above.
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);
  MFEM_HOST_DEVICE virtual void ComputeSourceTransportProperties(const double *state, const double *Up,
                                                                 const double *gradUp, const double *Efield,
                                                                 double *globalTransport, double *speciesTransport,
                                                                 double *diffusionVelocity, double *n_sp);

  // NOTE(kevin): only for AxisymmetricSource
  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc);
  MFEM_HOST_DEVICE void GetViscosities(const double *conserved, const double *primitive, double &visc,
                                       double &bulkVisc);

  // virtual double computeThirdOrderElectronThermalConductivity(const Vector &X_sp, const double debyeLength,
  //                                                             const double Te, const double nondimTe);
  MFEM_HOST_DEVICE virtual double computeThirdOrderElectronThermalConductivity(const double *X_sp,
                                                                               const double debyeLength,
                                                                               const double Te, const double nondimTe);

  virtual void computeMixtureAverageDiffusivity(const Vector &state, Vector &diffusivity);
  MFEM_HOST_DEVICE void computeMixtureAverageDiffusivity(const double *state, double *diffusivity);

  // These are used to compute third-order electron thermal conductivity based on standard Chapman--Enskog method.
  MFEM_HOST_DEVICE double L11ee(const double *Q2) { return Q2[0]; }
  MFEM_HOST_DEVICE double L11ea(const double *Q1) { return 6.25 * Q1[0] - 15. * Q1[1] + 12. * Q1[2]; }
  MFEM_HOST_DEVICE double L12ee(const double *Q2) { return 1.75 * Q2[0] - 2.0 * Q2[1]; }
  MFEM_HOST_DEVICE double L12ea(const double *Q1) {
    return 10.9375 * Q1[0] - 39.375 * Q1[1] + 57. * Q1[2] - 30. * Q1[3];
  }
  MFEM_HOST_DEVICE double L22ee(const double *Q2) { return 4.8125 * Q2[0] - 7.0 * Q2[1] + 5. * Q2[2]; }
  MFEM_HOST_DEVICE double L22ea(const double *Q1) {
    return 19.140625 * Q1[0] - 91.875 * Q1[1] + 199.5 * Q1[2] - 210. * Q1[3] + 90. * Q1[4];
  }

  MFEM_HOST_DEVICE void computeEffectiveMass(const double *mw, double *muw);
};

//////////////////////////////////////////////////////
//////// Argon Mixture Transport
//////////////////////////////////////////////////////

class ArgonMixtureTransport : public ArgonMinimalTransport {
 private:
  // int numAtoms_;
  // DenseMatrix composition_;
  // std::map<std::string, int> atomMap_;
  // Array<ArgonSpcs> speciesType_;
  // std::vector<std::string> speciesNames_;
  // std::map<int, int> *mixtureToInputMap_;

  // integer matrix. only upper triangular part will be used.
  // std::vector<std::vector<ArgonColl>> collisionIndex_;
  ArgonColl collisionIndex_[gpudata::MAXSPECIES * gpudata::MAXSPECIES];

  // void identifySpeciesType();
  // void identifyCollisionType();

 public:
  ArgonMixtureTransport(GasMixture *_mixture, RunConfiguration &_runfile);
  MFEM_HOST_DEVICE ArgonMixtureTransport(GasMixture *_mixture, const ArgonTransportInput &inputs);

  MFEM_HOST_DEVICE virtual ~ArgonMixtureTransport() {}

  MFEM_HOST_DEVICE double collisionIntegral(const int _spI, const int _spJ, const int l, const int r, const collisionInputs collInputs);

  // Currently, transport properties are evaluated in flux and source term separately.
  // Flux does not take primitive variables as input, rather evaluate them whenever needed.
  // ComputeFluxTransportProperties also evaluates required primitive variables,
  // but do not return it as output.
  // TODO(kevin): need to discuss whether to reuse computed primitive variables in flux evaluation,
  // or in general evaluation of primitive variables.
  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double *transportBuffer,
                                                               double *diffusionVelocity) {
    exit(-1);
    return;
  }

  // Source term will be constructed using ForcingTerms, which have pointers to primitive variables.
  // So we can use them in evaluating transport properties.
  // If this routine evaluate additional primitive variables, can return them just as the routine above.
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);

  // NOTE(kevin): only for AxisymmetricSource
  virtual void GetViscosities(const Vector &conserved, const Vector &primitive, double &visc, double &bulkVisc);

  virtual double computeThirdOrderElectronThermalConductivity(const Vector &X_sp, const collisionInputs &collInputs);
  //
  // virtual void computeMixtureAverageDiffusivity(const Vector &state, Vector &diffusivity);
};

#endif  // ARGON_TRANSPORT_HPP_
