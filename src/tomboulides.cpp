
#include "tomboulides.hpp"

#include "thermo_chem_base.hpp"

using namespace mfem;

Tomboulides::Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder)
    : gll_rules(0, Quadrature1D::GaussLobatto),
      pmesh_(pmesh),
      vorder_(vorder),
      porder_(porder),
      dim_(pmesh->Dimension()) {}

Tomboulides::~Tomboulides() {
  // objects allocated by initializeOperators
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
}

void Tomboulides::initializeOperators() {
  assert(thermo_interface_ != NULL);

  // Create all the Coefficient objects we need
  rho_coeff_ = new GridFunctionCoefficient(thermo_interface_->density);
  iorho_coeff_ = new RatioCoefficient(1.0, *rho_coeff_);

  // const IntegrationRule &ir_ni_v = gll_rules.Get(vfes_->GetFE(0)->GetGeomType(), 2 * vorder_ - 1);
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
}

/// Advance
void Tomboulides::step() {
  // TODO(trevilo): BC infrastructure
  Array<int> empty;

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
}
