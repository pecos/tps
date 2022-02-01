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

#ifndef QUASIMAGNETOSTATIC_HPP_
#define QUASIMAGNETOSTATIC_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
#include <mfem.hpp>

#include "em_options.hpp"
#include "tps.hpp"

/** Solve quasi-magnetostatic approximation of Maxwell
 *
 *  Solves the quasi-magnetostatic approximation of Maxwell's
 *  equations for a user-specified source current.
 *
 */
class QuasiMagnetostaticSolver : public TPS::Solver {
 private:
  mfem::MPI_Session &_mpi;
  ElectromagneticOptions _em_opts;

  // pointer to parent Tps class
  TPS::Tps *tpsP;

  mfem::ParMesh *_pmesh;
  int _dim;

  mfem::FiniteElementCollection *_hcurl;
  mfem::FiniteElementCollection *_h1;
  mfem::FiniteElementCollection *_hdiv;

  mfem::ParFiniteElementSpace *_Aspace;
  mfem::ParFiniteElementSpace *_pspace;
  mfem::ParFiniteElementSpace *_Bspace;

  mfem::Array<int> _ess_bdr_tdofs;

  mfem::ParBilinearForm *_K;
  mfem::ParLinearForm *_r;

  mfem::ParGridFunction *_A;
  mfem::ParGridFunction *_B;

  bool _operator_initialized;
  bool _current_initialized;

  void InterpolateToYAxis() const;

 public:
  QuasiMagnetostaticSolver(mfem::MPI_Session &mpi, ElectromagneticOptions em_opts, TPS::Tps *tps);
  ~QuasiMagnetostaticSolver();

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

#endif  // QUASIMAGNETOSTATIC_HPP_
