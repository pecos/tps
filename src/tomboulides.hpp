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
#ifndef TOMBOULIDES_HPP_
#define TOMBOULIDES_HPP_
/** @file
 *
 * @brief Implements the time marching scheme from Tomboulides et al,
 * Numerical simulation of low Mach number reactive flows,
 * J. Sci. Comput, 12(2), 1997.
 *
 */

#include "split_flow_base.hpp"

/// Container for forcing terms to be added to velocity equation
class ForcingTerm_T {
 public:
  ForcingTerm_T(mfem::Array<int> attr, mfem::VectorCoefficient *coeff) : attr(attr), coeff(coeff) {
    // nothing here
  }

  ForcingTerm_T(ForcingTerm_T &&obj) {
    // Deep copy the attribute array
    this->attr = obj.attr;

    // Move the coefficient pointer
    this->coeff = obj.coeff;
    obj.coeff = nullptr;
  }

  ~ForcingTerm_T() { delete coeff; }

  mfem::Array<int> attr;
  mfem::VectorCoefficient *coeff;
};

class Tomboulides : public FlowBase {
 protected:
  // Options
  // TODO(trevilo): hardcoded for testing.  Need to set from config,
  // so pass config to ctor
  bool numerical_integ_ = true;
  bool partial_assembly_ = false;

  int pressure_solve_pl_ = 0;
  int pressure_solve_max_iter_ = 100;
  double pressure_solve_rtol_ = 1e-8;

  int mass_inverse_pl_ = 0;
  int mass_inverse_max_iter_ = 200;
  double mass_inverse_rtol_ = 1e-12;

  // To use "numerical integration", quadrature rule must persist
  mfem::IntegrationRules gll_rules;

  // Basic info needed to create fem spaces
  mfem::ParMesh *pmesh_;
  const int vorder_;
  const int porder_;
  const int dim_;

  // Coefficients necessary to take a time step (including dt).
  // Assumed to be externally managed and determined, so just get a
  // reference here.
  const timeCoefficients &coeff_;

  // Object used to build forcing
  mfem::VectorConstantCoefficient *gravity_vec_;
  std::vector<ForcingTerm_T> forcing_terms_;

  /// Velocity FEM objects and fields
  mfem::FiniteElementCollection *vfec_ = nullptr;
  mfem::ParFiniteElementSpace *vfes_ = nullptr;
  mfem::ParGridFunction *u_curr_gf_ = nullptr;
  mfem::ParGridFunction *u_next_gf_ = nullptr;
  mfem::ParGridFunction *curl_gf_ = nullptr;
  mfem::ParGridFunction *curlcurl_gf_ = nullptr;

  /// Presser FEM objects and fields
  mfem::FiniteElementCollection *pfec_ = nullptr;
  mfem::ParFiniteElementSpace *pfes_ = nullptr;
  mfem::ParGridFunction *p_gf_ = nullptr;

  /// mfem::Coefficients used in forming necessary operators
  mfem::GridFunctionCoefficient *rho_coeff_ = nullptr;
  mfem::RatioCoefficient *iorho_coeff_ = nullptr;
  mfem::ConstantCoefficient nlcoeff_;

  // mfem "form" objects used to create operators
  mfem::ParBilinearForm *L_iorho_form_ = nullptr;  // \int (1/\rho) \nabla \phi_i \cdot \nabla \phi_j
  mfem::ParLinearForm *forcing_form_ = nullptr;    // \int \phi_i f
  mfem::ParNonlinearForm *Nconv_form_ = nullptr;   // \int \vphi_i \cdot [(u \cdot \nabla) u]
  mfem::ParBilinearForm *Mv_form_ = nullptr;       // mass matrix = \int \vphi_i \cdot \vphi_j

  // mfem operator objects
  mfem::OperatorHandle L_iorho_op_;
  mfem::OperatorHandle Mv_op_;

  // solver objects
  mfem::ParLORDiscretization *L_iorho_lor_ = nullptr;
  mfem::HypreBoomerAMG *L_iorho_inv_pc_ = nullptr;
  mfem::OrthoSolver *L_iorho_inv_ortho_pc_ = nullptr;
  mfem::CGSolver *L_iorho_inv_ = nullptr;

  mfem::Solver *Mv_inv_pc_ = nullptr;
  mfem::CGSolver *Mv_inv_ = nullptr;

  // Vectors
  mfem::Vector forcing_vec_;
  mfem::Vector u_vec_;
  mfem::Vector um1_vec_;
  mfem::Vector um2_vec_;
  mfem::Vector N_vec_;
  mfem::Vector Nm1_vec_;
  mfem::Vector Nm2_vec_;
  mfem::Vector ustar_vec_;
  mfem::Vector uext_vec_;

 public:
  /// Constructor
  Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder, timeCoefficients &coeff);

  /// Destructor
  ~Tomboulides() final;

  /// Allocate state ParGridFunction objects and initialize interface
  void initializeSelf() final;

  /**
   * @brief Initialize operators
   *
   * Initialize all Coefficient, BilinearForm, etc objects necessary
   * for step().  Note that this initialization necessarily depends
   * upon the thermo_interface_ object being initialized, so this
   * should be call *after* initializeFromThermoChem().
   */
  void initializeOperators() final;

  /// Advance
  void step() final;
};

#endif  // TOMBOULIDES_HPP_
