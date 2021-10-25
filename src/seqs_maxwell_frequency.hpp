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

#ifndef SEQS_MAXWELL_FREQUENCY_HPP_
#define SEQS_MAXWELL_FREQUENCY_HPP_

#include <tps_config.h>
#include <iostream>
#include <mfem.hpp>

#include "em_options.hpp"

/** Solve Maxwell's equations in the frequency domain
 *
 *  Solve Maxwell's equations in the frequency domain using the
 *  stabilized, electro-quasistatic (SEQS) gauge formulation of
 *  Ostrowski and Hiptmair ["Frequency-stable full Maxwell in
 *  electro-quasistatic gauge", SIAM J. Sci. Comput., Vol. 43, No. 4,
 *  pp. B1008--B1028].  This formulation is designed to avoid
 *  low-frequency instabilities that can arise in full Maxwell
 *  formulations without a priori asserting that either inductive or
 *  capacitive effects are negligible.
 *
 */
class SeqsMaxwellFrequencySolver {
 private:
  mfem::MPI_Session &mpi_;
  ElectromagneticOptions em_opt_;

  mfem::ParMesh *pmesh_;
  int dim_;

  mfem::FiniteElementCollection *hcurl_;
  mfem::FiniteElementCollection *h1_;
  mfem::FiniteElementCollection *hdiv_;

  mfem::ParFiniteElementSpace *Aspace_;
  mfem::ParFiniteElementSpace *pspace_;
  mfem::ParFiniteElementSpace *Bspace_;

  mfem::Array<int> h1_ess_tdof_list_;
  mfem::Array<int> hcurl_ess_tdof_list_;
  mfem::Array<int> port_0_;
  mfem::Array<int> port_1_;

  mfem::Array<int> psi_ess_tdof_list_;

  mfem::PWConstCoefficient *rel_sig_; /**< Relative conductivity = sigma/sigma0 */
  mfem::PWConstCoefficient *rel_eps_; /**< Relative permittivity = eps/eps0 */
  mfem::PWConstCoefficient *rel_mui_; /**< Inverse relative permeability = mu0/mu */
  mfem::PWConstCoefficient *rel_eps_nc_; /**< Relative permeability = eps/eps0 (for non-conducting domains)*/
  mfem::PWConstCoefficient *S_eta2_; /**< */
  mfem::PWConstCoefficient *S_eta2_reg_; /**< */
  mfem::ConstantCoefficient *one_over_sigma_; /**< 1/(non-dim conductivity) = omega*eps0/sigma0 */
  mfem::ConstantCoefficient *eta_squared_;    /**< (non-dim freq)*(non-dim freq) = (omega^2 ell^2)/(c^2) */
  mfem::ConstantCoefficient *neg_eta_squared_;    /**< -1.0*eta_squared_ */

  mfem::ParGridFunction *phi_real_;
  mfem::ParGridFunction *phi_imag_;
  mfem::ParGridFunction *psi_real_;
  mfem::ParGridFunction *psi_imag_;

  mfem::ParGridFunction *V0_real_;
  mfem::ParGridFunction *V1_real_;

  mfem::ParGridFunction *V0_imag_;
  mfem::ParGridFunction *V1_imag_;

  mfem::ParGridFunction *phi_tot_real_;
  mfem::ParGridFunction *phi_tot_imag_;

  mfem::ParGridFunction *A_real_;
  mfem::ParGridFunction *A_imag_;
  mfem::ParGridFunction *n_real_;
  mfem::ParGridFunction *n_imag_;

  mfem::ParGridFunction *B_real_;
  mfem::ParGridFunction *B_imag_;
  mfem::ParGridFunction *E_real_;
  mfem::ParGridFunction *E_imag_;

  // Solve for the electric potential
  void SolveSEQS();

  // Solve for the magnetic vector potential
  void SolveStabMaxwell();

  // Evaluate electric and magnetic fields
  void EvaluateEandB();

  // Evaluate the current
  void EvaluateCurrent();

  // Write paraview data
  void WriteParaview();

 public:
  SeqsMaxwellFrequencySolver(mfem::MPI_Session &mpi, ElectromagneticOptions em_opts);
  ~SeqsMaxwellFrequencySolver();

  /** Initialize SEQS Maxwell problem
   *
   *  Sets up MFEM objects (e.g., ParMesh, ParFiniteElementSpace,
   *  ParBilinearForm etc) required to form the linear system
   *  corresponding to the SEQS Maxwell problem.
   */
  void Initialize();

  /** Solve Maxwell problem
   *
   *  Solve both the electric potential and magnetic vector potential
   *  problems.
   */
  void Solve();
};

#endif  // SEQS_MAXWELL_FREQUENCY_HPP_
