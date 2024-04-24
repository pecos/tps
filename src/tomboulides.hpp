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

// forward-declaration for Tps support class (tps.hpp)
namespace TPS {
class Tps;
}

// forward-declaration of struct hold temporal coefficients (loMach.hpp)
struct temporalSchemeCoefficients;

#include "dirichlet_bc_helper.hpp"
#include "split_flow_base.hpp"
#include "utils.hpp"

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

class Tomboulides final : public FlowBase {
 protected:
  // true if this is root rank
  bool rank0_;
  bool axisym_;

  // Options
  // TODO(trevilo): hardcoded for testing.  Need to set based on input file.
  bool numerical_integ_ = false;
  // bool partial_assembly_ = false;
  bool partial_assembly_ = true;

  int pressure_solve_pl_ = 0;
  int pressure_solve_max_iter_ = 100;
  double pressure_solve_rtol_ = 1e-8;
  // double pressure_solve_rtol_ = 1e-12;
  double pressure_solve_atol_ = 1e-18;

  int mass_inverse_pl_ = 0;
  int mass_inverse_max_iter_ = 200;
  double mass_inverse_rtol_ = 1e-12;

  int hsolve_pl_ = 0;
  int hsolve_max_iter_ = 200;
  double hsolve_rtol_ = 1e-12;

  // To use "numerical integration", quadrature rule must persist
  mfem::IntegrationRules gll_rules;

  // Options-related structures
  TPS::Tps *tpsP_ = nullptr;

  // Basic info needed to create fem spaces
  mfem::ParMesh *pmesh_;
  const int vorder_;
  const int porder_;
  const int dim_;
  int nvel_;

  // Coefficients necessary to take a time step (including dt).
  // Assumed to be externally managed and determined, so just get a
  // reference here.
  const temporalSchemeCoefficients &coeff_;

  std::string ic_string_;

  StopWatch sw_setup_, sw_vstar_, sw_pp_, sw_helm_, sw_curl_;

  // Object used to build forcing
  mfem::Vector gravity_;
  mfem::VectorConstantCoefficient *gravity_vec_;
  mfem::ScalarVectorProductCoefficient *rad_gravity_vec_;
  std::vector<ForcingTerm_T> forcing_terms_;

  // BCs holders
  std::vector<DirichletBC_T<mfem::VectorCoefficient>> vel_dbcs_;
  std::vector<DirichletBC_T<mfem::Coefficient>> pres_dbcs_;
  std::vector<DirichletBC_T<mfem::Coefficient>> swirl_dbcs_;

  // List of essential boundaries
  mfem::Array<int> vel_ess_attr_;
  mfem::Array<int> pres_ess_attr_;
  mfem::Array<int> swirl_ess_attr_;

  // All essential true dofs.
  mfem::Array<int> vel_ess_tdof_;
  mfem::Array<int> pres_ess_tdof_;
  mfem::Array<int> swirl_ess_tdof_;

  /// Velocity FEM objects and fields
  mfem::FiniteElementCollection *vfec_ = nullptr;
  mfem::ParFiniteElementSpace *vfes_ = nullptr;
  mfem::FiniteElementCollection *sfec_ = nullptr;
  mfem::ParFiniteElementSpace *sfes_ = nullptr;
  mfem::ParGridFunction *u_curr_gf_ = nullptr;
  mfem::ParGridFunction *u_next_gf_ = nullptr;
  mfem::ParGridFunction *curl_gf_ = nullptr;
  mfem::ParGridFunction *curlcurl_gf_ = nullptr;
  mfem::ParGridFunction *resu_gf_ = nullptr;
  mfem::ParGridFunction *pp_div_gf_ = nullptr;
  mfem::ParGridFunction *gradU_gf_ = nullptr;
  mfem::ParGridFunction *gradV_gf_ = nullptr;
  mfem::ParGridFunction *gradW_gf_ = nullptr;
  // mfem::ParGridFunction *buffer_uInlet_ = nullptr;
  mfem::VectorGridFunctionCoefficient *velocity_field_ = nullptr;

  /// Pressure FEM objects and fields
  mfem::FiniteElementCollection *pfec_ = nullptr;
  mfem::ParFiniteElementSpace *pfes_ = nullptr;
  mfem::ParGridFunction *p_gf_ = nullptr;
  mfem::ParGridFunction *resp_gf_ = nullptr;
  mfem::ParGridFunction *pp_div_rad_comp_gf_ = nullptr;

  /// Swirl
  mfem::ParGridFunction *utheta_gf_ = nullptr;
  mfem::ParGridFunction *utheta_next_gf_ = nullptr;
  mfem::ParGridFunction *u_next_rad_comp_gf_ = nullptr;

  /// "total" viscosity, including fluid, turbulence, sponge
  mfem::ParGridFunction *mu_total_gf_ = nullptr;

  /// mfem::Coefficients used in forming necessary operators
  mfem::GridFunctionCoefficient *rho_coeff_ = nullptr;
  mfem::ConstantCoefficient *rho_const_coeff_ = nullptr;
  mfem::RatioCoefficient *iorho_coeff_ = nullptr;
  mfem::ConstantCoefficient nlcoeff_;
  mfem::ConstantCoefficient one_coeff_;
  mfem::ConstantCoefficient Hv_bdfcoeff_;
  mfem::ProductCoefficient *rho_over_dt_coeff_ = nullptr;
  mfem::GridFunctionCoefficient *mu_coeff_ = nullptr;
  mfem::VectorGridFunctionCoefficient *pp_div_coeff_ = nullptr;

  mfem::GradientGridFunctionCoefficient *grad_mu_coeff_ = nullptr;
  mfem::GradientVectorGridFunctionCoefficient *grad_u_next_coeff_ = nullptr;
  mfem::TransposeMatrixCoefficient *grad_u_next_transp_coeff_ = nullptr;
  mfem::MatrixVectorProductCoefficient *gradu_gradmu_coeff_ = nullptr;
  mfem::MatrixVectorProductCoefficient *graduT_gradmu_coeff_ = nullptr;
  mfem::VectorSumCoefficient *twoS_gradmu_coeff_ = nullptr;
  mfem::GridFunctionCoefficient *Qt_coeff_ = nullptr;
  mfem::ScalarVectorProductCoefficient *gradmu_Qt_coeff_ = nullptr;
  mfem::VectorSumCoefficient *S_poisson_coeff_ = nullptr;
  mfem::VectorSumCoefficient *S_mom_coeff_ = nullptr;

  mfem::ProductCoefficient *rad_rho_coeff_ = nullptr;
  mfem::ProductCoefficient *rad_rho_over_dt_coeff_ = nullptr;
  mfem::ProductCoefficient *rad_mu_coeff_ = nullptr;
  mfem::ScalarVectorProductCoefficient *rad_S_poisson_coeff_ = nullptr;
  mfem::ScalarVectorProductCoefficient *rad_S_mom_coeff_ = nullptr;
  mfem::RatioCoefficient *mu_over_rad_coeff_ = nullptr;
  mfem::VectorArrayCoefficient *visc_forcing_coeff_ = nullptr;
  mfem::GridFunctionCoefficient *pp_div_rad_comp_coeff_ = nullptr;

  mfem::ScalarVectorProductCoefficient *rad_pp_div_coeff_ = nullptr;
  std::vector<mfem::ScalarVectorProductCoefficient *> rad_vel_coeff_;

  mfem::GridFunctionCoefficient *utheta_coeff_ = nullptr;
  mfem::ProductCoefficient *utheta2_coeff_ = nullptr;
  mfem::VectorArrayCoefficient *ur_conv_forcing_coeff_ = nullptr;
  mfem::VectorGridFunctionCoefficient *u_next_coeff_ = nullptr;
  mfem::ScalarVectorProductCoefficient *rad_rhou_coeff_ = nullptr;
  mfem::GridFunctionCoefficient *u_next_rad_coeff_ = nullptr;
  mfem::ProductCoefficient *ur_ut_coeff_ = nullptr;
  mfem::ProductCoefficient *rho_ur_ut_coeff_ = nullptr;
  mfem::VectorArrayCoefficient *utheta_vec_coeff_ = nullptr;
  mfem::InnerProductCoefficient *swirl_var_viscosity_coeff_ = nullptr;

  // mfem "form" objects used to create operators
  mfem::ParBilinearForm *L_iorho_form_ = nullptr;  // \int (1/\rho) \nabla \phi_i \cdot \nabla \phi_j
  mfem::ParLinearForm *forcing_form_ = nullptr;    // \int \phi_i f
  mfem::ParNonlinearForm *Nconv_form_ = nullptr;   // \int \vphi_i \cdot [(u \cdot \nabla) u]
  mfem::ParBilinearForm *Ms_form_ = nullptr;       // mass matrix = \int \vphi_i \cdot \vphi_j
  mfem::ParBilinearForm *Mv_form_ = nullptr;       // mass matrix = \int \vphi_i \cdot \vphi_j
  mfem::ParMixedBilinearForm *D_form_ = nullptr;   // divergence = \int \phi_i \nabla \cdot \vphi_j
  mfem::ParMixedBilinearForm *G_form_ = nullptr;   // gradient = \int \vphi_i \cdot \nabla \phi_j
  mfem::ParBilinearForm *Mv_rho_form_ = nullptr;   // mass matrix (density weighted) = \int \rho \vphi_i \cdot \vphi_j
  mfem::ParBilinearForm *Hv_form_ = nullptr;
  mfem::ParLinearForm *pp_div_bdr_form_ = nullptr;
  mfem::ParLinearForm *u_bdr_form_ = nullptr;
  mfem::ParLinearForm *S_poisson_form_ = nullptr;
  mfem::ParLinearForm *S_mom_form_ = nullptr;
  mfem::ParLinearForm *Faxi_poisson_form_ = nullptr;
  mfem::ParLinearForm *ur_conv_axi_form_ = nullptr;

  mfem::ParBilinearForm *Ms_rho_form_ = nullptr;
  mfem::ParBilinearForm *Hs_form_ = nullptr;
  mfem::ParBilinearForm *As_form_ = nullptr;
  mfem::ParLinearForm *rho_ur_ut_form_ = nullptr;
  mfem::ParLinearForm *swirl_var_viscosity_form_ = nullptr;

  // mfem operator objects
  mfem::OperatorHandle L_iorho_op_;
  mfem::OperatorHandle Ms_op_;
  mfem::OperatorHandle Mv_op_;
  mfem::OperatorHandle Mv_rho_op_;
  mfem::OperatorHandle D_op_;
  mfem::OperatorHandle G_op_;
  mfem::OperatorHandle Hv_op_;
  mfem::OperatorHandle Ms_rho_op_;
  mfem::OperatorHandle Hs_op_;
  mfem::OperatorHandle As_op_;

  // solver objects
  mfem::ParLORDiscretization *L_iorho_lor_ = nullptr;
  // mfem::HypreBoomerAMG *L_iorho_inv_pc_ = nullptr;
  mfem::Solver *L_iorho_inv_pc_ = nullptr;
  mfem::OrthoSolver *L_iorho_inv_ortho_pc_ = nullptr;
  mfem::CGSolver *L_iorho_inv_ = nullptr;

  mfem::Solver *Mv_inv_pc_ = nullptr;
  mfem::CGSolver *Mv_inv_ = nullptr;

  mfem::Solver *Mv_rho_inv_pc_ = nullptr;
  mfem::CGSolver *Mv_rho_inv_ = nullptr;

  mfem::Solver *Hv_inv_pc_ = nullptr;
  mfem::CGSolver *Hv_inv_ = nullptr;

  mfem::Solver *Hs_inv_pc_ = nullptr;
  mfem::CGSolver *Hs_inv_ = nullptr;

  // Vectors
  mfem::Vector forcing_vec_;
  mfem::Vector u_vec_;
  mfem::Vector um1_vec_;
  mfem::Vector um2_vec_;
  mfem::Vector u_next_vec_;
  mfem::Vector N_vec_;
  mfem::Vector Nm1_vec_;
  mfem::Vector Nm2_vec_;
  mfem::Vector ustar_vec_;
  mfem::Vector uext_vec_;
  mfem::Vector pp_div_vec_;
  mfem::Vector pp_div_bdr_vec_;
  mfem::Vector u_bdr_vec_;
  mfem::Vector resp_vec_;
  mfem::Vector p_vec_;
  mfem::Vector resu_vec_;
  mfem::Vector tmpR0_;
  mfem::Vector tmpR1_;
  mfem::Vector gradU_;
  mfem::Vector gradV_;
  mfem::Vector gradW_;
  mfem::Vector Qt_vec_;
  mfem::Vector grad_Qt_vec_;
  mfem::Vector rho_vec_;
  mfem::Vector mu_vec_;
  mfem::Vector ress_vec_;
  mfem::Vector S_poisson_vec_;
  mfem::Vector Faxi_poisson_vec_;
  mfem::Vector ur_conv_forcing_vec_;
  mfem::Vector utheta_vec_;
  mfem::Vector utheta_m1_vec_;
  mfem::Vector utheta_m2_vec_;
  mfem::Vector utheta_next_vec_;

  // miscellaneous
  double volume_;
  mfem::ParLinearForm *mass_lform_ = nullptr;

  // helper functions

  /**
   * @brief Zero the mean of the input function
   */
  void meanZero(mfem::ParGridFunction &v);

  /**
   * @brief Update total viscosity using latest inputs
   */
  void updateTotalViscosity();

 public:
  /// Constructor
  Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder, temporalSchemeCoefficients &coeff, TPS::Tps *tps = nullptr);

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

  /**
   * @brief Initialize IO
   *
   * Register the state with the IODataOrganizer object so that it can
   * be read/written.  Must be called after initializeSelf().
   */
  void initializeIO(IODataOrganizer &io) const final;

  /**
   * @brief Initialize Paraview outputs
   *
   * Register fields to be written to paraview visualization files.
   */
  void initializeViz(mfem::ParaViewDataCollection &pvdc) const final;

  /// Advance
  void step() final;

  void screenHeader(std::vector<std::string> &header) const final {
    int nprint = 1;
    header.resize(nprint);
    header[0] = std::string("Max vel.");
  }

  void screenValues(std::vector<double> &values) final {
    int nprint = 1;
    values.resize(nprint);
    values[0] = maxVelocityMagnitude();
  }

  mfem::ParGridFunction *getCurrentVelocity() final { return u_curr_gf_; }

  /// Evaluate error (only when exact solution is known)
  double computeL2Error() const final;

  /// Add a Dirichlet boundary condition to the velocity field
  void addVelDirichletBC(const mfem::Vector &u, mfem::Array<int> &attr);
  void addVelDirichletBC(mfem::VectorCoefficient *coeff, mfem::Array<int> &attr);
  void addVelDirichletBC(void (*f)(const Vector &, double, Vector &), Array<int> &attr);

  /// Add a Dirichlet boundary condition to the pressure field.
  void addPresDirichletBC(double p, mfem::Array<int> &attr);

  /// Add constant swirl
  void addSwirlDirichletBC(double ut, mfem::Array<int> &attr);

  /// Compute maximum velocity magnitude anywhere in the domain
  double maxVelocityMagnitude();

  /// Compute Galerkin projection of velocity gradient
  void evaluateVelocityGradient();
};

#endif  // TOMBOULIDES_HPP_
