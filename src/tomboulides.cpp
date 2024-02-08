#include "tomboulides.hpp"

#include <mfem/general/forall.hpp>

#include "thermo_chem_base.hpp"

using namespace mfem;

/**
 * @brief Helper function to compute the curl of a vector field (2D version)
 *
 * Computes an approximation of the curl of a vector field.  Input
 * should be a vector function where each component is in the same
 * H1-conforming scalar FE space.  The output is in the same space.
 */
void ComputeCurl2D(ParGridFunction &u, ParGridFunction &cu, bool assume_scalar) {
  FiniteElementSpace *fes = u.FESpace();

  // AccumulateAndCountZones.
  Array<int> zones_per_vdof;
  zones_per_vdof.SetSize(fes->GetVSize());
  zones_per_vdof = 0;

  cu = 0.0;

  // Local interpolation.
  int elndofs;
  Array<int> vdofs;
  Vector vals;
  Vector loc_data;
  int vdim = fes->GetVDim();
  DenseMatrix grad_hat;
  DenseMatrix dshape;
  DenseMatrix grad;
  Vector curl;

  for (int e = 0; e < fes->GetNE(); ++e) {
    fes->GetElementVDofs(e, vdofs);
    u.GetSubVector(vdofs, loc_data);
    vals.SetSize(vdofs.Size());
    ElementTransformation *tr = fes->GetElementTransformation(e);
    const FiniteElement *el = fes->GetFE(e);
    elndofs = el->GetDof();
    int dim = el->GetDim();
    dshape.SetSize(elndofs, dim);

    for (int dof = 0; dof < elndofs; ++dof) {
      // Project.
      const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
      tr->SetIntPoint(&ip);

      // Eval and GetVectorGradientHat.
      el->CalcDShape(tr->GetIntPoint(), dshape);
      grad_hat.SetSize(vdim, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
      MultAtB(loc_data_mat, dshape, grad_hat);

      const DenseMatrix &Jinv = tr->InverseJacobian();
      grad.SetSize(grad_hat.Height(), Jinv.Width());
      Mult(grad_hat, Jinv, grad);

      if (assume_scalar) {
        curl.SetSize(2);
        curl(0) = grad(0, 1);
        curl(1) = -grad(0, 0);
      } else {
        curl.SetSize(2);
        curl(0) = grad(1, 0) - grad(0, 1);
        curl(1) = 0.0;
      }

      for (int j = 0; j < curl.Size(); ++j) {
        vals(elndofs * j + dof) = curl(j);
      }
    }

    // Accumulate values in all dofs, count the zones.
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      cu(ldof) += vals[j];
      zones_per_vdof[ldof]++;
    }
  }

  // Communication.

  // Count the zones globally.
  GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
  gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
  gcomm.Bcast(zones_per_vdof);

  // Accumulate for all vdofs.
  gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
  gcomm.Bcast<double>(cu.GetData());

  // Compute means.
  for (int i = 0; i < cu.Size(); i++) {
    const int nz = zones_per_vdof[i];
    if (nz) {
      cu(i) /= nz;
    }
  }
}

/**
 * @brief Helper function to compute the curl of a vector field (3D version)
 *
 * Computes an approximation of the curl of a vector field.  Input
 * should be a vector function where each component is in the same
 * H1-conforming scalar FE space.  The output is in the same space.
 */
void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu) {
  FiniteElementSpace *fes = u.FESpace();

  // AccumulateAndCountZones.
  Array<int> zones_per_vdof;
  zones_per_vdof.SetSize(fes->GetVSize());
  zones_per_vdof = 0;

  cu = 0.0;

  // Local interpolation.
  int elndofs;
  Array<int> vdofs;
  Vector vals;
  Vector loc_data;
  int vdim = fes->GetVDim();
  DenseMatrix grad_hat;
  DenseMatrix dshape;
  DenseMatrix grad;
  Vector curl;

  for (int e = 0; e < fes->GetNE(); ++e) {
    fes->GetElementVDofs(e, vdofs);
    u.GetSubVector(vdofs, loc_data);
    vals.SetSize(vdofs.Size());
    ElementTransformation *tr = fes->GetElementTransformation(e);
    const FiniteElement *el = fes->GetFE(e);
    elndofs = el->GetDof();
    int dim = el->GetDim();
    dshape.SetSize(elndofs, dim);

    for (int dof = 0; dof < elndofs; ++dof) {
      // Project.
      const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
      tr->SetIntPoint(&ip);

      // Eval and GetVectorGradientHat.
      el->CalcDShape(tr->GetIntPoint(), dshape);
      grad_hat.SetSize(vdim, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
      MultAtB(loc_data_mat, dshape, grad_hat);

      const DenseMatrix &Jinv = tr->InverseJacobian();
      grad.SetSize(grad_hat.Height(), Jinv.Width());
      Mult(grad_hat, Jinv, grad);

      curl.SetSize(3);
      curl(0) = grad(2, 1) - grad(1, 2);
      curl(1) = grad(0, 2) - grad(2, 0);
      curl(2) = grad(1, 0) - grad(0, 1);

      for (int j = 0; j < curl.Size(); ++j) {
        vals(elndofs * j + dof) = curl(j);
      }
    }

    // Accumulate values in all dofs, count the zones.
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      cu(ldof) += vals[j];
      zones_per_vdof[ldof]++;
    }
  }

  // Communication

  // Count the zones globally.
  GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
  gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
  gcomm.Bcast(zones_per_vdof);

  // Accumulate for all vdofs.
  gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
  gcomm.Bcast<double>(cu.GetData());

  // Compute means.
  for (int i = 0; i < cu.Size(); i++) {
    const int nz = zones_per_vdof[i];
    if (nz) {
      cu(i) /= nz;
    }
  }
}

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

  pfec_ = new H1_FECollection(porder_);
  pfes_ = new ParFiniteElementSpace(pmesh_, pfec_);
  p_gf_ = new ParGridFunction(pfes_);

  *u_curr_gf_ = 0.0;
  *u_next_gf_ = 0.0;
  *curl_gf_ = 0.0;
  *curlcurl_gf_ = 0.0;
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
  uext_vec_.SetSize(vfes_truevsize);

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
}
