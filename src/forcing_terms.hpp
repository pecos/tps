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
#ifndef FORCING_TERMS_HPP_
#define FORCING_TERMS_HPP_

#include <grvy.h>
#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

#ifdef HAVE_MASA
#include "masa_handler.hpp"
#endif

using namespace mfem;
using namespace std;

class ForcingTerms {
 protected:
  double time;

  const int &dim;
  const int nvel;
  const int &num_equation;
  const bool axisymmetric_;
  const int &order;
  const int &intRuleType;
  IntegrationRules *intRules;
  ParFiniteElementSpace *vfes;
  ParGridFunction *U_;
  ParGridFunction *Up_;
  ParGridFunction *gradUp_;

  const precomputedIntegrationData &gpuArrays;
  const int *h_num_elems_of_type;

  // added term
  //   ParGridFunction *b;

 public:
  ForcingTerms(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
               IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up,
               ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays, bool axisym);
  virtual ~ForcingTerms();

  void setTime(double _time) { time = _time; }
  virtual void updateTerms(Vector &in) = 0;
  //   virtual void addForcingIntegrals(Vector &in);
};

// Constant pressure gradient term
class ConstantPressureGradient : public ForcingTerms {
 private:
  // RunConfiguration &config;
  Vector pressGrad;
  GasMixture *mixture_;

 public:
  ConstantPressureGradient(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                           IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
                           ParGridFunction *_Up, ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays,
                           RunConfiguration &_config, GasMixture *mixture);
  virtual ~ConstantPressureGradient() {}

  // Terms do not need updating
  virtual void updateTerms(Vector &in);

  // GPU functions
#ifdef _GPU_
  static void updateTerms_gpu(const int numElems, const int offsetElems, const int elDof, const int totalDofs,
                              Vector &pressGrad, Vector &in, const Vector &Up, Vector &gradUp, const int num_equation,
                              const int dim, const precomputedIntegrationData &gpuArrays);
#endif
};

class AxisymmetricSource : public ForcingTerms {
 private:
  GasMixture *mixture;
  TransportProperties *transport_;
  const Equations &eqSystem;
  ParGridFunction *space_vary_viscosity_mult_;

 public:
  AxisymmetricSource(const int &_dim, const int &_num_equation, const int &_order, GasMixture *_mixture,

                     TransportProperties *_transport, const Equations &_eqSystem, const int &_intRuleType,
                     IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
                     ParGridFunction *_Up, ParGridFunction *_gradUp, ParGridFunction *spaceVaryViscMult,
                     const precomputedIntegrationData &gpuArrays, RunConfiguration &_config);
  virtual ~AxisymmetricSource() {}

  virtual void updateTerms(Vector &in);
};

class JouleHeating : public ForcingTerms {
 private:
  const Equations &eqSystem;
  ParGridFunction *joule_heating_;
  GasMixture *mixture_;

 public:
  JouleHeating(const int &_dim, const int &_num_equation, const int &_order, GasMixture *_mixture,
               const Equations &_eqSystem, const int &_intRuleType, IntegrationRules *_intRules,
               ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up, ParGridFunction *_gradUp,
               const precomputedIntegrationData &gpuArrays, RunConfiguration &_config, ParGridFunction *jh_);
  virtual ~JouleHeating() {}

  virtual void updateTerms(Vector &in);
};

class SpongeZone : public ForcingTerms {
 private:
  Fluxes *fluxes;
  GasMixture *mixture;

  SpongeZoneData &szData;
  Vector targetU;

  Array<int> nodesInMixedOutPlane;

  ParGridFunction *sigma;  // linearly varying factor
  Vector radialNormal;
  Array<int> nodesInAnnulus;

  Vector meanNormalFluxes;

  bool singleTemperature_;

  void computeMixedOutValues();
  void addSpongeZoneForcing(Vector &in);

 public:
  SpongeZone(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType, Fluxes *_fluxClass,
             GasMixture *_mixture, IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
             ParGridFunction *_Up, ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays,
             RunConfiguration &_config, const int sz);
  virtual ~SpongeZone();

  virtual void updateTerms(Vector &in);
};

// Forcing that adds a passive scalar
class PassiveScalar : public ForcingTerms {
 private:
  GasMixture *mixture;

  Array<passiveScalarData *> psData_;

 public:
  PassiveScalar(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, GasMixture *_mixture, ParGridFunction *U,
                ParGridFunction *_Up, ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays,
                RunConfiguration &_config);

  // Terms do not need updating
  virtual void updateTerms(Vector &in);

  void updateTerms_gpu(Vector &in, ParGridFunction *Up, Array<passiveScalarData *> &psData, const int nnode,
                       const int num_equation);

  virtual ~PassiveScalar();
};

class HeatSource : public ForcingTerms {
 private:
  GasMixture *mixture_;

  heatSourceData &heatSource_;

  Array<int> nodeList_;

 public:
  HeatSource(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
             heatSourceData &heatSource, GasMixture *_mixture, IntegrationRules *_intRules,
             ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up, ParGridFunction *_gradUp,
             const precomputedIntegrationData &gpuArrays, RunConfiguration &_config);
  virtual ~HeatSource() {}

  virtual void updateTerms(Vector &in);

  void updateTerms_gpu(Vector &in);
};

#ifdef HAVE_MASA
// Manufactured Solution using MASA
class MASA_forcings : public ForcingTerms {
 private:
  void (*evaluateForcing_)(const Vector &, const double, Array<double> &) = 0;

 public:
  MASA_forcings(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up,
                ParGridFunction *_gradUp, const precomputedIntegrationData &gpuArrays, RunConfiguration &_config);

  virtual void updateTerms(Vector &in);
};
#endif  // HAVE_MASA

#endif  // FORCING_TERMS_HPP_
