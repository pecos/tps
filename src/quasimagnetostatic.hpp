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

#ifndef QUASIMAGNETOSTATIC_HPP_
#define QUASIMAGNETOSTATIC_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>

#include "em_options.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"

/** Solve quasi-magnetostatic approximation of Maxwell
 *
 *  Solves the quasi-magnetostatic approximation of Maxwell's
 *  equations for a user-specified source current.
 *
 */
class QuasiMagnetostaticSolver3D : public TPS::Solver {
 private:
  mfem::MPI_Session &mpi_;
  ElectromagneticOptions em_opts_;

  // pointer to parent Tps class
  TPS::Tps *tpsP_;

  mfem::ParMesh *pmesh_;
  int dim_;

  int true_size_;
  Array<int> offsets_;

  mfem::FiniteElementCollection *hcurl_;
  mfem::FiniteElementCollection *h1_;
  mfem::FiniteElementCollection *hdiv_;

  mfem::ParFiniteElementSpace *Aspace_;
  mfem::ParFiniteElementSpace *pspace_;
  mfem::ParFiniteElementSpace *Bspace_;

  mfem::Array<int> ess_bdr_tdofs_;

  mfem::ParBilinearForm *K_;
  mfem::ParLinearForm *r_;

  mfem::ParGridFunction *Areal_;
  mfem::ParGridFunction *Aimag_;

  mfem::ParGridFunction *Breal_;
  mfem::ParGridFunction *Bimag_;

  mfem::ParGridFunction *plasma_conductivity_;
  mfem::GridFunctionCoefficient *plasma_conductivity_coef_;

  bool operator_initialized_;
  bool current_initialized_;

  void InterpolateToYAxis() const;

 public:
  QuasiMagnetostaticSolver3D(mfem::MPI_Session &mpi, ElectromagneticOptions em_opts, TPS::Tps *tps);
  ~QuasiMagnetostaticSolver3D();

  /** Initialize quasi-magnetostatic problem
   *
   *  Sets up MFEM objects (e.g., ParMesh, ParFiniteElementSpace,
   *  ParBilinearForm etc) required to form the linear system
   *  corresponding to the quasi-magnetostatic problem.
   */
  void initialize() override;

  /** Initialize current for quasi-magnetostatic problem
   *
   *  Build the source current function and do a divergence free
   *  projection to form the RHS
   */
  void InitializeCurrent();

  void parseSolverOptions() override;

  /** Solve quasi-magnetostatic problem */
  void solve() override;
};

/** Solve quasi-magnetostatic approximation of Maxwell for 2-D axisymmetric
 *
 *  A separate class is used for axisymmetric b/c we use different finite
 *  element spaces from the 3D solver.
 *
 */
class QuasiMagnetostaticSolverAxiSym : public TPS::Solver {
 private:
  mfem::MPI_Session &mpi_;
  ElectromagneticOptions em_opts_;

  // pointer to parent Tps class
  TPS::Tps *tpsP_;

  mfem::ParMesh *pmesh_;
  int dim_;
  int true_size_;
  Array<int> offsets_;

  mfem::FiniteElementCollection *h1_;

  mfem::ParFiniteElementSpace *Atheta_space_;

  mfem::Array<int> ess_bdr_tdofs_;

  mfem::ParBilinearForm *K_;
  mfem::ParLinearForm *r_;

  mfem::ParGridFunction *Atheta_real_;
  mfem::ParGridFunction *Atheta_imag_;

  mfem::ParGridFunction *plasma_conductivity_;
  mfem::GridFunctionCoefficient *plasma_conductivity_coef_;

  mfem::ParGridFunction *joule_heating_;

  bool operator_initialized_;
  bool current_initialized_;

  void InterpolateToYAxis() const;

 public:
  QuasiMagnetostaticSolverAxiSym(mfem::MPI_Session &mpi, ElectromagneticOptions em_opts, TPS::Tps *tps);
  ~QuasiMagnetostaticSolverAxiSym();

  /** Initialize axisymmetric quasi-magnetostatic problem
   *
   *  Sets up MFEM objects (e.g., ParMesh, ParFiniteElementSpace,
   *  ParBilinearForm etc) required to form the linear system
   *  corresponding to the quasi-magnetostatic problem.
   */
  void initialize() override;

  /** Initialize current for axisymmetric quasi-magnetostatic problem
   *
   *  Build the source current function
   */
  void InitializeCurrent();

  void parseSolverOptions() override;

  /** Solve quasi-magnetostatic problem */
  void solve() override;

  mfem::ParMesh *getMesh() const { return pmesh_; }
  mfem::ParGridFunction *getPlasmaConductivityGF() { return plasma_conductivity_; }
  mfem::ParGridFunction *getJouleHeatingGF() { return joule_heating_; }
};

#endif  // QUASIMAGNETOSTATIC_HPP_
