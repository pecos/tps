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
#include "source_term.hpp"

#include <vector>

SourceTerm::SourceTerm(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
                       ParGridFunction *_Up, ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays,
                       RunConfiguration &_config, GasMixture *mixture, GasMixture *d_mixture,
                       TransportProperties *transport, Chemistry *chemistry)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpuArrays,
                   _config.isAxisymmetric()),
      mixture_(mixture),
      d_mixture_(d_mixture),
      transport_(transport),
      chemistry_(chemistry) {
  numSpecies_ = mixture->GetNumSpecies();
  numActiveSpecies_ = mixture->GetNumActiveSpecies();
  numReactions_ = _config.chemistryInput.numReactions;

  ambipolar_ = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
}

SourceTerm::~SourceTerm() {
  //   if (mixture_ != NULL) delete mixture_;
  //   if (transport_ != NULL) delete transport_;
  //   if (chemistry_ != NULL) delete chemistry_;
}

void SourceTerm::updateTerms(mfem::Vector &in) {
#if defined(_HIP_)
  mfem_error("Source term is not supported on hip path!\n");
  exit(-1);
#endif

#if defined(_CUDA_)
  const double *h_Up = Up_->Read();
  const double *h_U = U_->Read();
  const double *h_gradUp = gradUp_->Read();
  double *h_in = in.ReadWrite();

  GasMixture *_mixture = d_mixture_;
#else
  const double *h_Up = Up_->HostRead();
  const double *h_U = U_->HostRead();
  const double *h_gradUp = gradUp_->HostRead();
  double *h_in = in.HostReadWrite();

  GasMixture *_mixture = mixture_;
#endif
  TransportProperties *_transport = transport_;
  Chemistry *_chemistry = chemistry_;
  const int nnodes = vfes->GetNDofs();
  const int _dim = dim;
  const int _nvel = nvel;
  const int _num_equation = num_equation;
  const int _numSpecies = numSpecies_;
  const int _numActiveSpecies = numActiveSpecies_;
  const int _numReactions = numReactions_;

#if defined(_CUDA_)
  MFEM_FORALL(n, nnodes, {
    double upn[gpudata::MAXEQUATIONS];
    double Un[gpudata::MAXEQUATIONS];
    double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double srcTerm[gpudata::MAXEQUATIONS];
#else
  double upn[gpudata::MAXEQUATIONS];
  double Un[gpudata::MAXEQUATIONS];
  double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
  double srcTerm[gpudata::MAXEQUATIONS];
  for (int n = 0; n < nnodes; n++) {
#endif
    for (int eq = 0; eq < _num_equation; eq++) {
      upn[eq] = h_Up[n + eq * nnodes];
      Un[eq] = h_U[n + eq * nnodes];
      srcTerm[eq] = 0.0;
      for (int d = 0; d < _dim; d++)
        gradUpn[eq + d * _num_equation] = h_gradUp[n + eq * nnodes + d * _num_equation * nnodes];
    }
    // TODO(kevin): update E-field with EM coupling.
    // E-field can have azimuthal component.
    double Efield[gpudata::MAXDIM];
    for (int v = 0; v < _nvel; v++) Efield[v] = 0.0;

    double globalTransport[gpudata::MAXSPECIES];
    double speciesTransport[gpudata::MAXSPECIES * SpeciesTrns::NUM_SPECIES_COEFFS];
    // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
    double diffusionVelocity[gpudata::MAXSPECIES * gpudata::MAXDIM];
    for (int v = 0; v < _nvel; v++)
      for (int sp = 0; sp < _numSpecies; sp++) diffusionVelocity[sp + v * _numSpecies] = 0.0;
    double ns[gpudata::MAXSPECIES];
    _transport->ComputeSourceTransportProperties(Un, upn, gradUpn, Efield, globalTransport, speciesTransport,
                                                 diffusionVelocity, ns);

    for (int eq = 0; eq < _num_equation; eq++) srcTerm[eq] = 0.0;

    double Th = 0., Te = 0.;
    Th = upn[1 + _nvel];
    if (_mixture->IsTwoTemperature()) {
      Te = upn[_num_equation - 1];
    } else {
      Te = Th;
    }

    double kfwd[gpudata::MAXREACTIONS], kC[gpudata::MAXREACTIONS];
    _chemistry->computeForwardRateCoeffs(Th, Te, kfwd);
    _chemistry->computeEquilibriumConstants(Th, Te, kC);

    // get reaction rates
    double progressRates[gpudata::MAXREACTIONS], creationRates[gpudata::MAXREACTIONS];
    for (int r = 0; r < _numReactions; r++) {
      progressRates[r] = 0.0;
      creationRates[r] = 0.0;
    }
    _chemistry->computeProgressRate(ns, kfwd, kC, progressRates);
    _chemistry->computeCreationRate(progressRates, creationRates);

    // add species creation rates
    for (int sp = 0; sp < _numActiveSpecies; sp++) {
      srcTerm[2 + _nvel + sp] += creationRates[sp];
    }

    // Terms required for EM-coupling.
    double Jd[gpudata::MAXDIM];  // diffusion current.
    for (int v = 0; v < _nvel; v++) Jd[v] = 0.0;
    if (_mixture->IsAmbipolar()) {  // diffusion current using electric conductivity.
      // const double mho = globalTransport(SrcTrns::ELECTRIC_CONDUCTIVITY);
      // Jd = mho * Efield

    } else {  // diffusion current by definition.
      for (int sp = 0; sp < _numSpecies; sp++) {
        for (int d = 0; d < _nvel; d++)
          Jd[d] += diffusionVelocity[sp + d * _numSpecies] * ns[sp] * MOLARELECTRONCHARGE *
                   _mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES);
      }
    }

    // TODO(kevin): may move axisymmetric source terms to here.

    // TODO(kevin): energy sink for radiative reaction.

    if (_mixture->IsTwoTemperature()) {
      // energy sink from electron-impact reactions.
      for (int r = 0; r < _numReactions; r++) {
        if (_chemistry->isElectronInvolvedAt(r))
          srcTerm[_num_equation - 1] -= _chemistry->getReactionEnergy(r) * progressRates[r];
      }

      // work by electron pressure
      // const double pe = mixture_->computeElectronPressure(ns(numSpecies_ - 2), Te);
      // for (int d = 0; d < dim; d++) srcTerm(num_equation - 1) -= pe * gradUpn(d + 1, d);
      double gradPe[gpudata::MAXDIM];
      _mixture->computeElectronPressureGrad(ns[_numSpecies - 2], Te, gradUpn, gradPe);
      for (int d = 0; d < _dim; d++) srcTerm[_num_equation - 1] += gradPe[d] * upn[d + 1];

      // energy transfer by elastic momentum transfer
      const double me = _mixture->GetGasParams(_numSpecies - 2, GasParams::SPECIES_MW);
      const double ne = ns[_numSpecies - 2];
      for (int sp = 0; sp < _numSpecies; sp++) {
        if (sp == _numSpecies - 2) continue;

        double m_sp = _mixture->GetGasParams(sp, GasParams::SPECIES_MW);
        // kB is converted to R, as number densities are provided in mol.
        double energy = 1.5 * UNIVERSALGASCONSTANT * (Te - Th);
        // TODO(kevin): diffusion-driven term is often neglected.
        // Let's neglect this now and add it later if more refined physics is needed.
        // for (int d = 0; d < dim; d++) {
        //   energy += 0.5 * (m_sp - me) * diffusionVelocity(sp, d) * diffusionVelocity(numSpecies_, d);
        // }
        energy *= 2.0 * me * m_sp / (m_sp + me) / (m_sp + me) * ne *
                  speciesTransport[sp + SpeciesTrns::MF_FREQUENCY * _numSpecies];

        srcTerm[_num_equation - 1] -= energy;
      }

      // TODO(kevin): work by electron diffusion - rho_e V_e * Du/Dt
    }

    // add source term to buffer
    for (int eq = 0; eq < _num_equation; eq++) {
      h_in[n + eq * nnodes] += srcTerm[eq];
    }
#if defined(_CUDA_)
  });  // MFEM_FORALL(n, nnodes, {
#else
  }  // for (int n = 0; n < nnodes; n++)
#endif  // defined(_CUDA_)
}

// NOTE(kevin): previous cpu routine.
// void SourceTerm::updateTerms(mfem::Vector &in) {
//   const double *h_Up = Up_->HostRead();
//   const double *h_U = U_->HostRead();
//   const double *h_gradUp = gradUp_->HostRead();
//   double *h_in = in.HostReadWrite();
//
//   const int nnodes = vfes->GetNDofs();
//
//   Vector upn(num_equation);
//   Vector Un(num_equation);
//   DenseMatrix gradUpn(num_equation, dim);
//   Vector srcTerm(num_equation);
//   for (int n = 0; n < nnodes; n++) {
//     for (int eq = 0; eq < num_equation; eq++) {
//       upn(eq) = h_Up[n + eq * nnodes];
//       Un(eq) = h_U[n + eq * nnodes];
//       for (int d = 0; d < dim; d++) gradUpn(eq, d) = h_gradUp[n + eq * nnodes + d * num_equation * nnodes];
//     }
//     // TODO(kevin): update E-field with EM coupling.
//     // E-field can have azimuthal component.
//     Vector Efield(nvel);
//     Efield = 0.0;
//
//     Vector globalTransport(numSpecies_);
//     DenseMatrix speciesTransport(numSpecies_, SpeciesTrns::NUM_SPECIES_COEFFS);
//     // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
//     DenseMatrix diffusionVelocity(numSpecies_, nvel);
//     diffusionVelocity = 0.0;
//     Vector ns(numSpecies_);
//     transport_->ComputeSourceTransportProperties(Un, upn, gradUpn, globalTransport, Efield, speciesTransport,
//                                                  diffusionVelocity, ns);
//
//     srcTerm = 0.0;
//
//     double Th = 0., Te = 0.;
//     Th = upn[1 + nvel];
//     if (mixture_->IsTwoTemperature()) {
//       Te = upn[num_equation - 1];
//     } else {
//       Te = Th;
//     }
//
//     Vector kfwd, kC;
//     chemistry_->computeForwardRateCoeffs(Th, Te, kfwd);
//     chemistry_->computeEquilibriumConstants(Th, Te, kC);
//
//     // get reaction rates
//     Vector progressRates(numReactions_), creationRates(numSpecies_);
//     progressRates = 0.0;
//     creationRates = 0.0;
//     chemistry_->computeProgressRate(ns, kfwd, kC, progressRates);
//     chemistry_->computeCreationRate(progressRates, creationRates);
//
//     // add species creation rates
//     for (int sp = 0; sp < numActiveSpecies_; sp++) {
//       srcTerm(2 + nvel + sp) += creationRates(sp);
//     }
//
//     // Terms required for EM-coupling.
//     Vector Jd(nvel);  // diffusion current.
//     Jd = 0.0;
//     if (ambipolar_) {  // diffusion current using electric conductivity.
//       // const double mho = globalTransport(SrcTrns::ELECTRIC_CONDUCTIVITY);
//       // Jd = mho * Efield
//
//     } else {  // diffusion current by definition.
//       for (int sp = 0; sp < numSpecies_; sp++) {
//         for (int d = 0; d < nvel; d++)
//           Jd(d) += diffusionVelocity(sp, d) * ns(sp) * MOLARELECTRONCHARGE *
//                    mixture_->GetGasParams(sp, GasParams::SPECIES_CHARGES);
//       }
//     }
//
//     // TODO(kevin): may move axisymmetric source terms to here.
//
//     // TODO(kevin): energy sink for radiative reaction.
//
//     if (twoTemperature_) {
//       // energy sink from electron-impact reactions.
//       for (int r = 0; r < numReactions_; r++) {
//         if (chemistry_->isElectronInvolvedAt(r))
//           srcTerm(num_equation - 1) -= chemistry_->getReactionEnergy(r) * progressRates(r);
//       }
//
//       // work by electron pressure
//       // const double pe = mixture_->computeElectronPressure(ns(numSpecies_ - 2), Te);
//       // for (int d = 0; d < dim; d++) srcTerm(num_equation - 1) -= pe * gradUpn(d + 1, d);
//       Vector gradPe(dim);
//       mixture_->computeElectronPressureGrad(ns(numSpecies_ - 2), Te, gradUpn, gradPe);
//       for (int d = 0; d < dim; d++) srcTerm(num_equation - 1) += gradPe(d) * upn(d + 1);
//
//       // energy transfer by elastic momentum transfer
//       const double me = mixture_->GetGasParams(numSpecies_ - 2, GasParams::SPECIES_MW);
//       const double ne = ns(numSpecies_ - 2);
//       for (int sp = 0; sp < numSpecies_; sp++) {
//         if (sp == numSpecies_ - 2) continue;
//
//         double m_sp = mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
//         // kB is converted to R, as number densities are provided in mol.
//         double energy = 1.5 * UNIVERSALGASCONSTANT * (Te - Th);
//         // TODO(kevin): diffusion-driven term is often neglected.
//         // Let's neglect this now and add it later if more refined physics is needed.
//         // for (int d = 0; d < dim; d++) {
//         //   energy += 0.5 * (m_sp - me) * diffusionVelocity(sp, d) * diffusionVelocity(numSpecies_, d);
//         // }
//         energy *= 2.0 * me * m_sp / (m_sp + me) / (m_sp + me) * ne * speciesTransport(sp, SpeciesTrns::MF_FREQUENCY);
//
//         srcTerm(num_equation - 1) -= energy;
//       }
//
//       // TODO(kevin): work by electron diffusion - rho_e V_e * Du/Dt
//     }
//
//     // add source term to buffer
//     for (int eq = 0; eq < num_equation; eq++) {
//       h_in[n + eq * nnodes] += srcTerm(eq);
//     }
//   }
// }
