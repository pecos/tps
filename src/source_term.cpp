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
#include "source_term.hpp"

#include <vector>

SourceTerm::SourceTerm(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                       ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays,
                       RunConfiguration &_config, GasMixture *mixture, TransportProperties *transport,
                       Chemistry *chemistry)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, _Up, _gradUp, gpuArrays),
      mixture_(mixture),
      transport_(transport),
      chemistry_(chemistry) {
  numSpecies_ = mixture->GetNumSpecies();
  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  numReactions_ = chemistry_->GetNumReactions();

  ambipolar_ = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  
  // allocate device vectors
#ifdef _GPU_
  reactionsModel_.SetSize(numReactions_);
  for (int r = 0; r < numReactions_; r++) reactionsModel_[r] = chemistry_->GetReaction(r)->GetReactionModel();
  
  reactionConstants_.SetSize(numReactions_);
  for (int r = 0; r < numReactions_; r++) {
    reactionConstants_[r].A_ = chemistry->GetReaction(r)->GetReactionConstants().A_;
    reactionConstants_[r].b_ = chemistry->GetReaction(r)->GetReactionConstants().b_;
    reactionConstants_[r].E_ = chemistry->GetReaction(r)->GetReactionConstants().E_;
  }
#endif
}

SourceTerm::~SourceTerm() {
  //   if (mixture_ != NULL) delete mixture_;
  //   if (transport_ != NULL) delete transport_;
  //   if (chemistry_ != NULL) delete chemistry_;
}

void SourceTerm::updateTerms(mfem::Vector &in) {
#ifdef _GPU_
  updateSourceTerms_gpu(in);
#else
  const double *h_Up = Up->HostRead();
  double *h_in = in.HostReadWrite();

  const int nnodes = vfes->GetNDofs();

  Vector upn(num_equation);
  Vector Un(num_equation);
  for (int n = 0; n < nnodes; n++) {
    for (int eq = 0; eq < num_equation; eq++) upn(eq) = h_Up[n + eq * nnodes];
    mixture_->GetConservativesFromPrimitives(upn, Un);

    double Th = 0., Te = 0.;
    Th = upn[1 + dim];
    if (mixture_->IsTwoTemperature()) {
      Te = upn[num_equation - 1];
    } else {
      Te = Th;
    }

    Vector kfwd, kC;
    chemistry_->computeForwardRateCoeffs(Th, Te, kfwd);
    chemistry_->computeEquilibriumConstants(Th, Te, kC);

    Vector ns;
    mixture_->computeNumberDensities(Un, ns);

    // get reaction rates
    Vector creationRates(numSpecies_);
    chemistry_->computeCreationRate(ns, kfwd, kC, creationRates);

    // add species creation rates
    for (int sp = 0; sp < numActiveSpecies_; sp++) {
      h_in[n + (2 + dim + sp) * nnodes] += creationRates(sp);
    }
  }
#endif // _GPU_
}

#ifdef _GPU_
void SourceTerm::updateSourceTerms_gpu(mfem::Vector& in)
{
  const double *d_Up = Up->Read();
  double *d_in = in.ReadWrite();
  
  const int nnodes = vfes->GetNDofs();
  const int d_num_equations = num_equation;
  const int d_dim = dim;
  
  const WorkingFluid fluid = mixture_->GetWorkingFluid();
  
  const int numSpecies = mixture_->GetNumSpecies();
  const int numActiveSpecies = mixture_->GetNumActiveSpecies();
  const double *gasParams = NULL;
  const double *molarCV = NULL;
  if (fluid == WorkingFluid::USER_DEFINED) {
    gasParams = mixture_->GetGasParam().Read();
    molarCV = mixture_->getMolarCVs().Read();
  }
  const bool ambipolar = mixture_->IsAmbipolar();
  const bool twoTemperature = mixture_->IsTwoTemperature();
  const int electronIndex = chemistry_->GetElectronIndex();
  const int numReactions = chemistry_->GetNumReactions();
  const double *reactantStoich = chemistry_->GetReactantStoichiometryVector().Read();
  const double *productsStoich = chemistry_->getProductStoichiometryVector().Read();
  
  const ReactionModel * reactionsModel = reactionsModel_.Read();
  const reactionConstants *constants = reactionConstants_.Read();
  
  const bool *detailedBalance = chemistry_->GetDetailBalanceArray().Read();
  const double *equilibriumConstantParams = chemistry_->GetEquilibriumConstantsMatrix().Read();
  

  MFEM_FORALL_2D(n, nnodes, d_num_equations,1,1,{
    MFEM_FOREACH_THREAD(eq, x, d_num_equations) {
      MFEM_SHARED double up[20], state[20];
      MFEM_SHARED double nsp[15], creationRates[15];
      MFEM_SHARED double kfwd[20], kC[20]; // WARNING: 20 reactions considered
      
      up[eq] = d_Up[n + eq*nnodes];
      MFEM_SYNC_THREAD;

      // Assuming this will always use a PefectMixture
      PerfectMixture::GetConservativesFromPrimitives_gpu(up,
                                                         state,
                                                         gasParams,
                                                         molarCV,
                                                         ambipolar,
                                                         twoTemperature,
                                                         d_num_equations,
                                                         d_dim,
                                                         numSpecies,
                                                         numActiveSpecies,
                                                         eq, d_num_equations );
      MFEM_SYNC_THREAD;
      
      double Th = 0., Te = 0.;
      Th = up[1 + d_dim];
      if (twoTemperature) {
        Te = up[d_num_equations - 1];
      } else {
        Te = Th;
      }

      Chemistry::computeForwardRateCoeffs_gpu(Th, 
                                              Te, 
                                              &kfwd[0],
                                              electronIndex,
                                              numSpecies,
                                              numReactions,
                                              reactantStoich,
                                              reactionsModel,
                                              constants);

      Chemistry::computeEquilibriumConstants_gpu(Th, 
                                                 Te, 
                                                 &kC[0],
                                                 detailedBalance,
                                                 electronIndex,
                                                 numSpecies,
                                                 numReactions,
                                                 reactantStoich,
                                                 equilibriumConstantParams);

      PerfectMixture::computeNumberDensities_gpu(&state[0], 
                                                 &nsp[0],
                                                 gasParams,
                                                 d_dim,
                                                 numSpecies,
                                                 numActiveSpecies,
                                                 ambipolar,
                                                 eq,
                                                 d_num_equations);
      MFEM_SYNC_THREAD;

      // get reaction rates
      MassActionLaw::computeCreationRate_gpu(&nsp[0],
                                             &kfwd[0],
                                             &kC[0],
                                             &creationRates[0],
                                             gasParams,
                                             reactantStoich,
                                             productsStoich,
                                             numReactions,
                                             numSpecies,
                                             eq,
                                             d_num_equations);
      MFEM_SYNC_THREAD;

      // add species creation rates
      for (int sp = eq; sp < numActiveSpecies; sp += d_num_equations) {
        d_in[n + (2 + d_dim + sp) * nnodes] += creationRates[sp];
      }
    }
  });
}

#endif // _GPU_
