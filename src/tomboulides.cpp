#include "tomboulides.hpp"

#include <mfem/general/forall.hpp>

#include "thermo_chem_base.hpp"

using namespace mfem;

Tomboulides::Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder, timeCoefficients &coeff)
    : gll_rules(0, Quadrature1D::GaussLobatto),
      pmesh_(pmesh),
      vorder_(vorder),
      porder_(porder),
      dim_(pmesh->Dimension()),
      coeff_(coeff) {}

Tomboulides::~Tomboulides() {
  // objects allocated by initializeOperators
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
  delete iorho_coeff_;
  delete rho_coeff_;

  // objects allocated by initalizeSelf
  delete p_gf_;
  delete pfes_;
  delete pfec_;
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

  pfec_ = new H1_FECollection(porder_);
  pfes_ = new ParFiniteElementSpace(pmesh_, pfec_);
  p_gf_ = new ParGridFunction(pfes_);

  *u_curr_gf_ = 0.0;
  *u_next_gf_ = 0.0;
  *p_gf_ = 0.0;

  interface.velocity = u_next_gf_;

  // Add gravity forcing
  assert(dim_ >= 2);
  Vector gravity(dim_);
  gravity = 0.0;

  // TODO(trevilo): Get gravity from input file.  For now, hardcoded
  // to usual value acting in -y direction.
  double *g = gravity.HostWrite();
  g[1] = -9.81;

  // NB: ForcingTerm_T takes ownership of this vector.  Do not delete it.
  gravity_vec_ = new VectorConstantCoefficient(gravity);
  Array<int> domain_attr(pmesh_->attributes.Max());
  domain_attr = 1;

  forcing_terms_.emplace_back(domain_attr, gravity_vec_);

  // Allocate Vector storage
  const int vfes_truevsize = vfes_->GetTrueVSize();
  // const int pfes_truevsize = pfes->GetTrueVSize();
  forcing_vec_.SetSize(vfes_truevsize);
  u_vec_.SetSize(vfes_truevsize);
  um1_vec_.SetSize(vfes_truevsize);
  um2_vec_.SetSize(vfes_truevsize);
  N_vec_.SetSize(vfes_truevsize);
  Nm1_vec_.SetSize(vfes_truevsize);
  Nm2_vec_.SetSize(vfes_truevsize);
  ustar_vec_.SetSize(vfes_truevsize);

  // zero vectors for now
  forcing_vec_ = 0.0;
  u_vec_ = 0.0;
  um1_vec_ = 0.0;
  um2_vec_ = 0.0;
  N_vec_ = 0.0;
  Nm1_vec_ = 0.0;
  Nm2_vec_ = 0.0;
  ustar_vec_ = 0.0;
}

void Tomboulides::initializeOperators() {
  assert(thermo_interface_ != NULL);

  // Create all the Coefficient objects we need
  rho_coeff_ = new GridFunctionCoefficient(thermo_interface_->density);
  iorho_coeff_ = new RatioCoefficient(1.0, *rho_coeff_);

  // Integration rules (only used if numerical_integ_ is true).  When
  // this is the case, the quadrature degree set such that the
  // Gauss-Lobatto quad pts correspond to the Gauss-Lobatto nodes.
  // For most terms this will result in an under-integration, but it
  // has the nice consequence that the mass matrix is diagonal.
  const IntegrationRule &ir_ni_v = gll_rules.Get(vfes_->GetFE(0)->GetGeomType(), 2 * vorder_ - 1);
  const IntegrationRule &ir_ni_p = gll_rules.Get(pfes_->GetFE(0)->GetGeomType(), 2 * porder_ - 1);

  // Empty array, use where we want operators without BCs
  Array<int> empty;

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
}
