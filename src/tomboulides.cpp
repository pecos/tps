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

#include "tomboulides.hpp"

#include <mfem/general/forall.hpp>

#include "io.hpp"
#include "loMach.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;

/// forward declarations
void vel_exact_tgv2d(const Vector &x, double t, Vector &u);

/**
 * @brief Helper function to remove mean from a vector
 */
void Orthogonalize(Vector &v, const ParFiniteElementSpace *pfes) {
  double loc_sum = v.Sum();
  double global_sum = 0.0;
  int loc_size = v.Size();
  int global_size = 0;

  MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
  MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pfes->GetComm());

  v -= global_sum / static_cast<double>(global_size);
}

Tomboulides::Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder, temporalSchemeCoefficients &coeff, TPS::Tps *tps)
    : gll_rules(0, Quadrature1D::GaussLobatto),
      pmesh_(pmesh),
      vorder_(vorder),
      porder_(porder),
      dim_(pmesh->Dimension()),
      coeff_(coeff) {
  // make sure there is room for BC attributes
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    vel_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    vel_ess_attr_ = 0;

    pres_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    pres_ess_attr_ = 0;
  }

  if (tps != nullptr) {
    // if we have Tps object, use it to set other options...

    // Initial condition function... options are
    // 1) "" (Empty string), velocity initialized to zero
    // 2) "tgv2d", velocity initialized using vel_exact_tgv2d function
    // 3) "constant", TODO(trevilo) implement options to read constant
    tps->getInput("loMach/tomboulides/ic", ic_string_, std::string(""));

    // Boundary conditions
    // number of BC regions defined
    int numWalls, numInlets, numOutlets;
    tps->getInput("boundaryConditions/numWalls", numWalls, 0);
    tps->getInput("boundaryConditions/numInlets", numInlets, 0);
    tps->getInput("boundaryConditions/numOutlets", numOutlets, 0);

    // Inlet and outlet not supported yet!
    // TODO(trevilo): Add support
    assert(numInlets == 0);
    assert(numOutlets == 0);

    // Wall Bcs
    for (int i = 1; i <= numWalls; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/wall" + std::to_string(i));

      tps->getRequiredInput((basepath + "/patch").c_str(), patch);
      tps->getRequiredInput((basepath + "/type").c_str(), type);

      if (type == "viscous_isothermal" || type == "viscous_adiabatic" || type == "viscous" || type == "no-slip") {
        Array<int> wall_attr(pmesh_->bdr_attributes.Max());
        wall_attr = 0;
        wall_attr[patch] = 1;

        Vector zero(dim_);
        zero = 0.0;

        Vector velocity_value(dim_);
        tps->getVec((basepath + "/velocity").c_str(), velocity_value, dim_, zero);

        if (pmesh_->GetMyRank() == 0) {
          std::cout << "Tomboulides: Setting Dirichlet velocity on patch = " << patch << std::endl;
        }
        addVelDirichletBC(velocity_value, wall_attr);
      } else {
        if (pmesh_->GetMyRank() == 0) {
          std::cout << "Tomboulides: When reading " << basepath << ", encountered wall type = " << type << std::endl;
          std::cout << "Tomboulides: Tomboulides flow solver does not support this type." << std::endl;
        }
        assert(false);
        exit(1);
      }
    }
  }
}

Tomboulides::~Tomboulides() {
  // miscellaneous
  delete mass_lform_;

  // objects allocated by initializeOperators
  delete pp_div_bdr_form_;
  delete Hv_inv_;
  delete Hv_inv_pc_;
  delete Hv_form_;
  delete Mv_rho_form_;
  delete G_form_;
  delete D_form_;
  delete Mv_inv_;
  delete Mv_inv_pc_;
  delete Mv_form_;
  delete Nconv_form_;
  delete forcing_form_;
  delete L_iorho_inv_;
  delete L_iorho_inv_ortho_pc_;
  delete L_iorho_inv_pc_;
  delete L_iorho_lor_;
  delete L_iorho_form_;
  delete pp_div_coeff_;
  delete mu_coeff_;
  delete rho_over_dt_coeff_;
  delete iorho_coeff_;
  delete rho_coeff_;

  // objects allocated by initalizeSelf
  delete resp_gf_;
  delete p_gf_;
  delete pfes_;
  delete pfec_;
  delete resu_gf_;
  delete pp_div_gf_;
  delete curlcurl_gf_;
  delete curl_gf_;
  delete u_next_gf_;
  delete u_curr_gf_;
  delete vfes_;
  delete vfec_;
}

void Tomboulides::initializeSelf() {
  // Initialize minimal state and interface
  vfec_ = new H1_FECollection(vorder_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);
  u_curr_gf_ = new ParGridFunction(vfes_);
  u_next_gf_ = new ParGridFunction(vfes_);
  curl_gf_ = new ParGridFunction(vfes_);
  curlcurl_gf_ = new ParGridFunction(vfes_);
  resu_gf_ = new ParGridFunction(vfes_);
  pp_div_gf_ = new ParGridFunction(vfes_);

  pfec_ = new H1_FECollection(porder_);
  pfes_ = new ParFiniteElementSpace(pmesh_, pfec_);
  p_gf_ = new ParGridFunction(pfes_);
  resp_gf_ = new ParGridFunction(pfes_);

  *u_curr_gf_ = 0.0;
  *u_next_gf_ = 0.0;
  *curl_gf_ = 0.0;
  *curlcurl_gf_ = 0.0;
  *resu_gf_ = 0.0;
  *p_gf_ = 0.0;
  *resp_gf_ = 0.0;

  toThermoChem_interface.velocity = u_next_gf_;

  // NEEDS TO BE ADDED
  // toTurbModel_interface.gradU = gradU_gf_;
  // toTurbModel_interface.gradV = gradV_gf_;
  // toTurbModel_interface.gradW = gradW_gf_;

  // Add gravity forcing
  assert(dim_ >= 2);
  Vector gravity(dim_);
  gravity = 0.0;

  // TODO(trevilo): Get gravity from input file.  For now, hardcoded
  // to usual value acting in -y direction.
  double *g = gravity.HostWrite();
  // g[1] = -9.81;

  // NB: ForcingTerm_T takes ownership of this vector.  Do not delete it.
  gravity_vec_ = new VectorConstantCoefficient(gravity);
  Array<int> domain_attr(pmesh_->attributes.Max());
  domain_attr = 1;

  forcing_terms_.emplace_back(domain_attr, gravity_vec_);

  // Allocate Vector storage
  const int vfes_truevsize = vfes_->GetTrueVSize();
  const int pfes_truevsize = pfes_->GetTrueVSize();

  forcing_vec_.SetSize(vfes_truevsize);
  u_vec_.SetSize(vfes_truevsize);
  um1_vec_.SetSize(vfes_truevsize);
  um2_vec_.SetSize(vfes_truevsize);
  N_vec_.SetSize(vfes_truevsize);
  Nm1_vec_.SetSize(vfes_truevsize);
  Nm2_vec_.SetSize(vfes_truevsize);
  ustar_vec_.SetSize(vfes_truevsize);
  uext_vec_.SetSize(vfes_truevsize);
  pp_div_vec_.SetSize(vfes_truevsize);
  resu_vec_.SetSize(vfes_truevsize);

  resp_vec_.SetSize(pfes_truevsize);
  p_vec_.SetSize(pfes_truevsize);
  pp_div_bdr_vec_.SetSize(pfes_truevsize);

  // zero vectors for now
  forcing_vec_ = 0.0;
  u_vec_ = 0.0;
  um1_vec_ = 0.0;
  um2_vec_ = 0.0;
  N_vec_ = 0.0;
  Nm1_vec_ = 0.0;
  Nm2_vec_ = 0.0;
  ustar_vec_ = 0.0;
  uext_vec_ = 0.0;
  pp_div_vec_ = 0.0;
  resu_vec_ = 0.0;

  resp_vec_ = 0.0;
  p_vec_ = 0.0;
  pp_div_bdr_vec_ = 0.0;

  // set IC if we have one at this point
  if (!ic_string_.empty()) {
    if (ic_string_ == "tgv2d") {
      std::cout << "Setting tgv2d IC..." << std::endl;
      VectorFunctionCoefficient u_excoeff(2, vel_exact_tgv2d);
      u_excoeff.SetTime(0.0);
      u_curr_gf_->ProjectCoefficient(u_excoeff);
    }
  }
}

void Tomboulides::initializeOperators() {
  assert(thermo_interface_ != NULL);

  // Create all the Coefficient objects we need
  rho_coeff_ = new GridFunctionCoefficient(thermo_interface_->density);
  iorho_coeff_ = new RatioCoefficient(1.0, *rho_coeff_);

  Hv_bdfcoeff_.constant = 1.0 / coeff_.dt;
  rho_over_dt_coeff_ = new ProductCoefficient(Hv_bdfcoeff_, *rho_coeff_);
  mu_coeff_ = new GridFunctionCoefficient(thermo_interface_->viscosity);
  pp_div_coeff_ = new VectorGridFunctionCoefficient(pp_div_gf_);

  // Integration rules (only used if numerical_integ_ is true).  When
  // this is the case, the quadrature degree set such that the
  // Gauss-Lobatto quad pts correspond to the Gauss-Lobatto nodes.
  // For most terms this will result in an under-integration, but it
  // has the nice consequence that the mass matrix is diagonal.
  const IntegrationRule &ir_ni_v = gll_rules.Get(vfes_->GetFE(0)->GetGeomType(), 2 * vorder_ - 1);
  const IntegrationRule &ir_ni_p = gll_rules.Get(pfes_->GetFE(0)->GetGeomType(), 2 * porder_ - 1);

  // Empty array, use where we want operators without BCs
  Array<int> empty;

  // Get Dirichlet dofs
  vfes_->GetEssentialTrueDofs(vel_ess_attr_, vel_ess_tdof_);

  // Create the operators

  // Variable coefficient Laplacian: \nabla \cdot ( (1/\rho) \nabla )
  L_iorho_form_ = new ParBilinearForm(pfes_);
  auto *L_iorho_blfi = new DiffusionIntegrator(*iorho_coeff_);
  if (numerical_integ_) {
    L_iorho_blfi->SetIntRule(&ir_ni_p);
  }
  L_iorho_form_->AddDomainIntegrator(L_iorho_blfi);
  if (partial_assembly_) {
    L_iorho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  L_iorho_form_->Assemble();
  // TODO(trevilo): BC infrastructure
  L_iorho_form_->FormSystemMatrix(empty, L_iorho_op_);

  // Variable coefficient Laplacian inverse
  L_iorho_lor_ = new ParLORDiscretization(*L_iorho_form_, empty);
  L_iorho_inv_pc_ = new HypreBoomerAMG(L_iorho_lor_->GetAssembledMatrix());
  L_iorho_inv_pc_->SetPrintLevel(0);
  L_iorho_inv_ortho_pc_ = new OrthoSolver(pfes_->GetComm());
  L_iorho_inv_ortho_pc_->SetSolver(*L_iorho_inv_pc_);

  L_iorho_inv_ = new CGSolver(pfes_->GetComm());
  L_iorho_inv_->iterative_mode = true;
  L_iorho_inv_->SetOperator(*L_iorho_op_);
  // TODO(trevilo): BC infrastructure (b/c empty, force ortho for now)
  L_iorho_inv_->SetPreconditioner(*L_iorho_inv_ortho_pc_);
  L_iorho_inv_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_->SetRelTol(pressure_solve_rtol_);
  L_iorho_inv_->SetMaxIter(pressure_solve_max_iter_);

  // Forcing term in the velocity equation: du/dt + ... = ... + f
  forcing_form_ = new ParLinearForm(vfes_);
  for (auto &force : forcing_terms_) {
    auto *fdlfi = new VectorDomainLFIntegrator(*force.coeff);
    if (numerical_integ_) {
      fdlfi->SetIntRule(&ir_ni_v);
    }
    forcing_form_->AddDomainIntegrator(fdlfi);
  }

  // The nonlinear (i.e., convection) term
  // Coefficient is -1 so we can just add to rhs
  nlcoeff_.constant = -1.0;
  Nconv_form_ = new ParNonlinearForm(vfes_);
  auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff_);
  if (numerical_integ_) {
    nlc_nlfi->SetIntRule(&ir_ni_v);
  }
  Nconv_form_->AddDomainIntegrator(nlc_nlfi);
  if (partial_assembly_) {
    Nconv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    Nconv_form_->Setup();
  }

  // Mass matrix for the velocity
  Mv_form_ = new ParBilinearForm(vfes_);
  auto *mv_blfi = new VectorMassIntegrator;
  if (numerical_integ_) {
    mv_blfi->SetIntRule(&ir_ni_v);
  }
  Mv_form_->AddDomainIntegrator(mv_blfi);
  if (partial_assembly_) {
    Mv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mv_form_->Assemble();
  Mv_form_->FormSystemMatrix(empty, Mv_op_);

  // Mass matrix (density weighted) for the velocity
  Mv_rho_form_ = new ParBilinearForm(vfes_);
  auto *mvr_blfi = new VectorMassIntegrator(*rho_coeff_);
  if (numerical_integ_) {
    mvr_blfi->SetIntRule(&ir_ni_v);
  }
  Mv_rho_form_->AddDomainIntegrator(mvr_blfi);
  if (partial_assembly_) {
    Mv_rho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);

  // Inverse (unweighted) mass operator
  if (partial_assembly_) {
    Vector diag_pa(vfes_->GetTrueVSize());
    Mv_form_->AssembleDiagonal(diag_pa);
    Mv_inv_pc_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    Mv_inv_pc_ = new HypreSmoother(*Mv_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Mv_inv_pc_)->SetType(HypreSmoother::Jacobi, 1);
  }
  Mv_inv_ = new CGSolver(vfes_->GetComm());
  Mv_inv_->iterative_mode = false;
  Mv_inv_->SetOperator(*Mv_op_);
  Mv_inv_->SetPreconditioner(*Mv_inv_pc_);
  Mv_inv_->SetPrintLevel(mass_inverse_pl_);
  Mv_inv_->SetRelTol(mass_inverse_rtol_);
  Mv_inv_->SetMaxIter(mass_inverse_max_iter_);

  // Divergence operator
  D_form_ = new ParMixedBilinearForm(vfes_, pfes_);
  auto *vd_mblfi = new VectorDivergenceIntegrator();
  if (numerical_integ_) {
    vd_mblfi->SetIntRule(&ir_ni_v);
  }
  D_form_->AddDomainIntegrator(vd_mblfi);
  if (partial_assembly_) {
    D_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  D_form_->Assemble();
  D_form_->FormRectangularSystemMatrix(empty, empty, D_op_);

  // Gradient
  G_form_ = new ParMixedBilinearForm(pfes_, vfes_);
  auto *g_mblfi = new GradientIntegrator();
  if (numerical_integ_) {
    g_mblfi->SetIntRule(&ir_ni_v);
  }
  G_form_->AddDomainIntegrator(g_mblfi);
  if (partial_assembly_) {
    G_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  G_form_->Assemble();
  G_form_->FormRectangularSystemMatrix(empty, empty, G_op_);

  // Helmholtz
  Hv_form_ = new ParBilinearForm(vfes_);
  auto *hmv_blfi = new VectorMassIntegrator(*rho_over_dt_coeff_);
  auto *hdv_blfi = new VectorDiffusionIntegrator(*mu_coeff_);
  if (numerical_integ_) {
    hmv_blfi->SetIntRule(&ir_ni_v);
    hdv_blfi->SetIntRule(&ir_ni_v);
  }
  Hv_form_->AddDomainIntegrator(hmv_blfi);
  Hv_form_->AddDomainIntegrator(hdv_blfi);
  if (partial_assembly_) {
    // Partial assembly is not supported for variable coefficient
    // VectorMassIntegrator (as of mfem 4.5.2 at least)
    assert(false);
    Hv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(vel_ess_tdof_, Hv_op_);

  // Helmholtz solver
  if (partial_assembly_) {
    Vector diag_pa(vfes_->GetTrueVSize());
    Hv_form_->AssembleDiagonal(diag_pa);
    Hv_inv_pc_ = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof_);
  } else {
    Hv_inv_pc_ = new HypreSmoother(*Hv_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Hv_inv_pc_)->SetType(HypreSmoother::Jacobi, 1);
  }
  Hv_inv_ = new CGSolver(vfes_->GetComm());
  Hv_inv_->iterative_mode = true;
  Hv_inv_->SetOperator(*Hv_op_);
  Hv_inv_->SetPreconditioner(*Hv_inv_pc_);
  Hv_inv_->SetPrintLevel(hsolve_pl_);
  Hv_inv_->SetRelTol(hsolve_rtol_);
  Hv_inv_->SetMaxIter(hsolve_max_iter_);

  //
  pp_div_bdr_form_ = new ParLinearForm(pfes_);
  auto *ppd_bnlfi = new BoundaryNormalLFIntegrator(*pp_div_coeff_);
  if (numerical_integ_) {
    ppd_bnlfi->SetIntRule(&ir_ni_p);
  }
  pp_div_bdr_form_->AddBoundaryIntegrator(ppd_bnlfi, vel_ess_attr_);

  // Ensure u_vec_ consistent with u_curr_gf_
  u_curr_gf_->GetTrueDofs(u_vec_);
}

void Tomboulides::initializeIO(IODataOrganizer &io) const {
  io.registerIOFamily("Velocity", "/velocity", u_curr_gf_, false);
  io.registerIOVar("/velocity", "x-comp", 0);
  if (dim_ >= 2) io.registerIOVar("/velocity", "y-comp", 1);
  if (dim_ == 3) io.registerIOVar("/velocity", "z-comp", 2);
}

void Tomboulides::step() {
  //------------------------------------------------------------------------
  // The time step is split into 4 large chunks:
  //
  // 1) Update all operators that may have been invalidated by updates
  // outside of this class (e.g., changes to the density b/c the
  // thermochemistry model took a step
  //
  // 2) Compute vstar/dt according to eqn (2.3) from Tomboulides
  //
  // 3) Solve the Poisson equation for the pressure.  This is a
  // generalized (for variable dynamic viscosity) version of eqn (2.6)
  // from Tomboulides.
  //
  // 4) Solve the Helmholtz equation for the velocity.  This is a
  // generalized version of eqns (2.8) and (2.9) from Tomboulides.
  //------------------------------------------------------------------------

  // For convenience, get time and dt
  const double time = coeff_.time;
  const double dt = coeff_.dt;

  // TODO(trevilo): Have to implement some BC infrastructure
  Array<int> empty;

  //------------------------------------------------------------------------
  // Step 1: Update any operators invalidated by changing data
  // external to this class
  // ------------------------------------------------------------------------

  // Update the variable coefficient Laplacian to account for change
  // in density
  L_iorho_form_->Update();
  L_iorho_form_->Assemble();
  L_iorho_form_->FormSystemMatrix(empty, L_iorho_op_);

  // Update inverse for the variable coefficient Laplacian
  // TODO(trevilo): Really need to delete and recreate???
  delete L_iorho_inv_;
  delete L_iorho_inv_ortho_pc_;
  delete L_iorho_inv_pc_;
  delete L_iorho_lor_;

  L_iorho_lor_ = new ParLORDiscretization(*L_iorho_form_, empty);
  L_iorho_inv_pc_ = new HypreBoomerAMG(L_iorho_lor_->GetAssembledMatrix());
  L_iorho_inv_pc_->SetPrintLevel(0);
  L_iorho_inv_ortho_pc_ = new OrthoSolver(pfes_->GetComm());
  L_iorho_inv_ortho_pc_->SetSolver(*L_iorho_inv_pc_);

  L_iorho_inv_ = new CGSolver(pfes_->GetComm());
  L_iorho_inv_->iterative_mode = true;
  L_iorho_inv_->SetOperator(*L_iorho_op_);
  // TODO(trevilo): BC infrastructure (b/c empty, force ortho for now)
  L_iorho_inv_->SetPreconditioner(*L_iorho_inv_ortho_pc_);
  L_iorho_inv_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_->SetRelTol(pressure_solve_rtol_);
  L_iorho_inv_->SetMaxIter(pressure_solve_max_iter_);

  // Update density weighted mass
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);

  // Update the Helmholtz operator and inverse
  Hv_bdfcoeff_.constant = coeff_.bd0 / dt;
  Hv_form_->Update();
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(vel_ess_tdof_, Hv_op_);

  Hv_inv_->SetOperator(*Hv_op_);
  if (partial_assembly_) {
    // TODO(trevilo): Support partial assembly
    assert(false);
  }

  //------------------------------------------------------------------------
  // Step 2: Compute vstar / dt (as in eqn 2.3 from Tomboulides)
  // ------------------------------------------------------------------------

  // Evaluate the forcing at the end of the time step
  for (auto &force : forcing_terms_) {
    force.coeff->SetTime(time + dt);
  }
  forcing_form_->Assemble();
  forcing_form_->ParallelAssemble(forcing_vec_);

  // Evaluate the nonlinear---i.e. convection---term in the velocity eqn
  Nconv_form_->Mult(u_vec_, N_vec_);
  Nconv_form_->Mult(um1_vec_, Nm1_vec_);
  Nconv_form_->Mult(um2_vec_, Nm2_vec_);

  {
    const auto d_N = N_vec_.Read();
    const auto d_Nm1 = Nm1_vec_.Read();
    const auto d_Nm2 = Nm2_vec_.Read();
    auto d_force = forcing_vec_.ReadWrite();
    const auto ab1 = coeff_.ab1;
    const auto ab2 = coeff_.ab2;
    const auto ab3 = coeff_.ab3;
    MFEM_FORALL(i, forcing_vec_.Size(), { d_force[i] += (ab1 * d_N[i] + ab2 * d_Nm1[i] + ab3 * d_Nm2[i]); });
  }

  // vstar / dt = M^{-1} (extrapolated nonlinear + forcing) --- eqn.
  Mv_inv_->Mult(forcing_vec_, ustar_vec_);
  if (!Mv_inv_->GetConverged()) {
    exit(1);
  }
  // TODO(trevilo): track linear solve
  // iter_mvsolve = MvInv->GetNumIterations();
  // res_mvsolve = MvInv->GetFinalNorm();

  // vstar / dt += BDF contribution
  {
    const double bd1idt = -coeff_.bd1 / dt;
    const double bd2idt = -coeff_.bd2 / dt;
    const double bd3idt = -coeff_.bd3 / dt;
    const auto d_u = u_vec_.Read();
    const auto d_um1 = um1_vec_.Read();
    const auto d_um2 = um2_vec_.Read();
    auto d_ustar = ustar_vec_.ReadWrite();
    MFEM_FORALL(i, ustar_vec_.Size(), { d_ustar[i] += bd1idt * d_u[i] + bd2idt * d_um1[i] + bd3idt * d_um2[i]; });
  }

  // At this point ustar_vec_ holds vstar/dt!

  //------------------------------------------------------------------------
  // Step 3: Poisson
  // ------------------------------------------------------------------------

  // Extrapolate the velocity field (and store in u_next_gf_)
  {
    const auto d_u = u_vec_.Read();
    const auto d_um1 = um1_vec_.Read();
    const auto d_um2 = um2_vec_.Read();
    auto d_uext = uext_vec_.Write();
    const auto ab1 = coeff_.ab1;
    const auto ab2 = coeff_.ab2;
    const auto ab3 = coeff_.ab3;
    MFEM_FORALL(i, uext_vec_.Size(), { d_uext[i] = ab1 * d_u[i] + ab2 * d_um1[i] + ab3 * d_um2[i]; });
  }
  u_next_gf_->SetFromTrueDofs(uext_vec_);

  // Evaluate the double curl of the extrapolated velocity field
  if (dim_ == 2) {
    ComputeCurl2D(*u_next_gf_, *curl_gf_, false);
    ComputeCurl2D(*curl_gf_, *curlcurl_gf_, true);
  } else {
    ComputeCurl3D(*u_next_gf_, *curl_gf_);
    ComputeCurl3D(*curl_gf_, *curlcurl_gf_);
  }

  curlcurl_gf_->GetTrueDofs(pp_div_vec_);

  // Negate b/c term that appears in residual is -curl curl u
  pp_div_vec_.Neg();

  // TODO(trevilo): Generalize the calculation below of the viscous
  // contribution to the rhs of the pressure Poisson equation.  The
  // current calculation is incomplete in two ways.  First, it
  // neglects the contribution of gradient of the thermal divergence
  // (\nabla Q).  Second, it is restricted to constant properties.

  // Restriction 1: This doesn't handle variable nu
  auto rho = thermo_interface_->density->HostRead();
  auto mu = thermo_interface_->viscosity->HostRead();
  const double nu = mu[0] / rho[0];
  pp_div_vec_ *= nu;

  // Restriction 2: grad(Qt) term isn't here
  // pp_div_vec_ += (4/3) grad(Qt) contribution

  // Add ustar/dt contribution
  pp_div_vec_ += ustar_vec_;

  D_op_->Mult(pp_div_vec_, resp_vec_);

  // TODO(trevilo): Add Qt term
  // Ms->AddMult(Qt, resp, -bd0 / dt);

  // Negate residual s.t. sign is consistent with
  // -\nabla ( (1/\rho) \cdot \nabla p ) on LHS
  resp_vec_.Neg();

  // TODO(trevilo): Add boundary terms to residual
  pp_div_gf_->SetFromTrueDofs(pp_div_vec_);
  pp_div_bdr_form_->Assemble();
  pp_div_bdr_form_->ParallelAssemble(pp_div_bdr_vec_);
  resp_vec_.Add(1.0, pp_div_bdr_vec_);

  // TODO(trevilo): Only do this if no pressure BCs
  // Since now we don't have BCs at all, have to do it
  Orthogonalize(resp_vec_, pfes_);

  // for (auto &pres_dbc : pres_dbcs) {
  //   pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
  // }

  // Isn't this the same as SetFromTrueDofs????
  pfes_->GetRestrictionMatrix()->MultTranspose(resp_vec_, *resp_gf_);

  Vector X1, B1;
  if (partial_assembly_) {
    // TODO(trevilo): Support partial assembly here
    assert(false);
    // auto *SpC = Sp.As<ConstrainedOperator>();
    // EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
  } else {
    L_iorho_form_->FormLinearSystem(empty, *p_gf_, *resp_gf_, L_iorho_op_, X1, B1, 1);
  }

  L_iorho_inv_->Mult(B1, X1);
  if (!L_iorho_inv_->GetConverged()) {
    exit(1);
  }

  // iter_spsolve = SpInv->GetNumIterations();
  // res_spsolve = SpInv->GetFinalNorm();
  L_iorho_form_->RecoverFEMSolution(X1, *resp_gf_, *p_gf_);

  // If the boundary conditions on the pressure are pure Neumann remove the
  // nullspace by removing the mean of the pressure solution. This is also
  // ensured by the OrthoSolver wrapper for the preconditioner which removes
  // the nullspace after every application.
  meanZero(*p_gf_);
  p_gf_->GetTrueDofs(p_vec_);

  //------------------------------------------------------------------------
  // Step 4: Helmholtz solve for the velocity
  //------------------------------------------------------------------------
  resu_vec_ = 0.0;

  // Variable viscosity term
  // TODO(trevilo): Add variable viscosity terms
  // S_mom_form->Assemble();
  // S_mom_form->ParallelAssemble(resu);

  // -grad(p)
  G_op_->AddMult(p_vec_, resu_vec_, -1.0);

  // TODO(trevilo): Add grad(mu * Qt) term
  // Qt *= mun_next; // NB: pointwise multiply
  // G->AddMult(Qt, resu, 1.0 / 3.0);

  // rho * vstar / dt term
  Mv_rho_op_->AddMult(ustar_vec_, resu_vec_);

  for (auto &vel_dbc : vel_dbcs_) {
    u_next_gf_->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
  }

  vfes_->GetRestrictionMatrix()->MultTranspose(resu_vec_, *resu_gf_);

  Vector X2, B2;
  if (partial_assembly_) {
    // TODO(trevilo): Add partial assembly support
    assert(false);
    // auto *HC = H.As<ConstrainedOperator>();
    // EliminateRHS(*Hv_form_, *HC, vel_ess_tdof, un_next_gf, resu_gf, X2, B2, 1);
  } else {
    Hv_form_->FormLinearSystem(vel_ess_tdof_, *u_next_gf_, *resu_gf_, Hv_op_, X2, B2, 1);
  }

  Hv_inv_->Mult(B2, X2);
  if (!Hv_inv_->GetConverged()) {
    exit(1);
  }
  // iter_hsolve = HInv->GetNumIterations();
  // res_hsolve = HInv->GetFinalNorm();
  Hv_form_->RecoverFEMSolution(X2, *resu_gf_, *u_next_gf_);
  u_next_gf_->GetTrueDofs(u_next_vec_);

  // Rotate values in solution history
  um2_vec_ = um1_vec_;
  um1_vec_ = u_vec_;

  // Update the current solution and corresponding GridFunction
  u_next_gf_->GetTrueDofs(u_next_vec_);
  u_vec_ = u_next_vec_;
  u_curr_gf_->SetFromTrueDofs(u_vec_);
}

double Tomboulides::computeL2Error() const {
  double err = -1.0;
  if (ic_string_ == "tgv2d") {
    std::cout << "Evaluating TGV2D error..." << std::endl;
    VectorFunctionCoefficient u_excoeff(2, vel_exact_tgv2d);
    u_excoeff.SetTime(coeff_.time);
    err = u_curr_gf_->ComputeL2Error(u_excoeff);
  }
  return err;
}

void Tomboulides::meanZero(ParGridFunction &v) {
  // Make sure not to recompute the inner product linear form every
  // application.
  if (mass_lform_ == nullptr) {
    one_coeff_.constant = 1.0;
    mass_lform_ = new ParLinearForm(v.ParFESpace());
    auto *dlfi = new DomainLFIntegrator(one_coeff_);
    if (numerical_integ_) {
      const IntegrationRule &ir_ni = gll_rules.Get(vfes_->GetFE(0)->GetGeomType(), 2 * vorder_ - 1);
      dlfi->SetIntRule(&ir_ni);
    }
    mass_lform_->AddDomainIntegrator(dlfi);
    mass_lform_->Assemble();

    ParGridFunction one_gf(v.ParFESpace());
    one_gf.ProjectCoefficient(one_coeff_);

    volume_ = mass_lform_->operator()(one_gf);
  }

  double integ = mass_lform_->operator()(v);

  v -= integ / volume_;
}

/// Add a Dirichlet boundary condition to the velocity field
void Tomboulides::addVelDirichletBC(const Vector &u, Array<int> &attr) {
  assert(u.Size() == dim_);
  vel_dbcs_.emplace_back(attr, new VectorConstantCoefficient(u));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!vel_ess_attr_[i]);  // if vel_ess_attr[i] already set, fail b/c duplicate
      vel_ess_attr_[i] = 1;
    }
  }
}

/// Add a Dirichlet boundary condition to the pressure field.
void Tomboulides::addPresDirichletBC(double p, Array<int> &attr) {}

// Non-class functions that are only used in this file below here

/// Used to set the velocity IC (and to check error)
void vel_exact_tgv2d(const Vector &x, double t, Vector &u) {
  const double nu = 1.0;
  const double F = std::exp(-2 * nu * t);

  u(0) = F * std::sin(x[0]) * std::cos(x[1]);
  u(1) = -F * std::cos(x[0]) * std::sin(x[1]);
}
