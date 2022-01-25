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
#ifndef FORCING_TERMS_HPP_
#define FORCING_TERMS_HPP_

#include <tps_config.h>

#include <mfem.hpp>
#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "run_configuration.hpp"

#ifdef _MASA_
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
  const int &order;
  const int &intRuleType;
  IntegrationRules *intRules;
  ParFiniteElementSpace *vfes;
  ParGridFunction *Up;
  ParGridFunction *gradUp;

  const volumeFaceIntegrationArrays &gpuArrays;
  const int *h_numElems;
  const int *h_posDofIds;

  // added term
  //   ParGridFunction *b;

 public:
  ForcingTerms(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
               IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
               ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays);
  ~ForcingTerms();

  void setTime(double _time) { time = _time; }
  virtual void updateTerms(Vector &in) = 0;
  //   virtual void addForcingIntegrals(Vector &in);
};

// Constant pressure gradient term
class ConstantPressureGradient : public ForcingTerms {
 private:
  // RunConfiguration &config;
  Vector pressGrad;
  GasMixture *mixture;

 public:
  ConstantPressureGradient(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                           IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                           ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays,
                           RunConfiguration &_config);
  ~ConstantPressureGradient();

  // Terms do not need updating
  virtual void updateTerms(Vector &in);

  // GPU functions
#ifdef _GPU_
  static void updateTerms_gpu(const int numElems, const int offsetElems, const int elDof, const int totalDofs,
                              Vector &pressGrad, Vector &in, const Vector &Up, Vector &gradUp, const int num_equation,
                              const int dim, const volumeFaceIntegrationArrays &gpuArrays);
#endif
};

class AxisymmetricSource : public ForcingTerms {
 private:
  EquationOfState *eqState;
  const Equations &eqSystem;

 public:
  AxisymmetricSource(const int &_dim, const int &_num_equation, const int &_order,
                     EquationOfState *_eqState, const Equations &_eqSystem,
                     const int &_intRuleType, IntegrationRules *_intRules,
                     ParFiniteElementSpace *_vfes, ParGridFunction *_Up, ParGridFunction *_gradUp,
                     const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config);
  ~AxisymmetricSource();

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

  Vector meanNormalFluxes;

  void computeMixedOutValues();
  void addSpongeZoneForcing(Vector &in);

 public:
  SpongeZone(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType, Fluxes *_fluxClass,
             GasMixture *_mixture, IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
             ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config);
  ~SpongeZone();

  virtual void updateTerms(Vector &in);
};

// Forcing that adds a passive scalar
class PassiveScalar : public ForcingTerms {
 private:
  GasMixture *mixture;

  Array<passiveScalarData *> psData_;

 public:
  PassiveScalar(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, GasMixture *_mixture, ParGridFunction *_Up,
                ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config);

  // Terms do not need updating
  virtual void updateTerms(Vector &in);

  void updateTerms_gpu(Vector &in, ParGridFunction *Up, Array<passiveScalarData *> &psData, const int nnode,
                       const int num_equation);

  ~PassiveScalar();
};

#ifdef _MASA_
// Manufactured Solution using MASA
class MASA_forcings : public ForcingTerms {
 private:
 public:
  MASA_forcings(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config);

  virtual void updateTerms(Vector &in);
};
#endif  // _MASA_

#endif  // FORCING_TERMS_HPP_
