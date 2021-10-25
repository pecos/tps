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

#include <grvy.h>
#include <hdf5.h>

#include "logger.hpp"
#include "utils.hpp"

#include "seqs_maxwell_frequency.hpp"

using namespace mfem;

SeqsMaxwellFrequencySolver::SeqsMaxwellFrequencySolver(MPI_Session &mpi, ElectromagneticOptions em_opts)
  :
  mpi_(mpi),
  em_opt_(em_opts) {
  // dump options to screen for user inspection
  if (mpi_.Root()) {
    em_opt_.print(std::cout);
  }

  pmesh_  = NULL;
  hcurl_  = NULL;
  h1_     = NULL;
  hdiv_   = NULL;
  Aspace_ = NULL;
  pspace_ = NULL;
  Bspace_ = NULL;

  rel_sig_ = NULL;
  rel_eps_ = NULL;
  rel_mui_ = NULL;
  rel_eps_nc_ = NULL;
  S_eta2_  = NULL;
  S_eta2_reg_  = NULL;
  one_over_sigma_ = NULL;
  eta_squared_ = NULL;
  neg_eta_squared_ = NULL;

  phi_real_ = NULL;
  phi_imag_ = NULL;
  psi_real_ = NULL;
  psi_imag_ = NULL;

  V0_real_ = NULL;
  V1_real_ = NULL;

  V0_imag_ = NULL;
  V1_imag_ = NULL;

  phi_tot_real_ = NULL;
  phi_tot_imag_ = NULL;

  A_real_ = NULL;
  A_imag_ = NULL;
  n_real_ = NULL;
  n_imag_ = NULL;

  B_real_ = NULL;
  B_imag_ = NULL;
  E_real_ = NULL;
  E_imag_ = NULL;
}

SeqsMaxwellFrequencySolver::~SeqsMaxwellFrequencySolver() {
  delete E_imag_;
  delete E_real_;
  delete B_imag_;
  delete B_real_;
  delete n_imag_;
  delete n_real_;
  delete A_imag_;
  delete A_real_;
  delete phi_tot_real_;
  delete phi_tot_imag_;
  delete V1_imag_;
  delete V0_imag_;
  delete V1_real_;
  delete V0_real_;
  delete psi_imag_;
  delete psi_real_;
  delete phi_imag_;
  delete phi_real_;
  delete neg_eta_squared_;
  delete eta_squared_;
  delete one_over_sigma_;
  delete S_eta2_reg_;
  delete S_eta2_;
  delete rel_eps_nc_;
  delete rel_mui_;
  delete rel_eps_;
  delete rel_sig_;
  delete Bspace_;
  delete pspace_;
  delete Aspace_;
  delete hdiv_;
  delete h1_;
  delete hcurl_;
  delete pmesh_;
}


void SeqsMaxwellFrequencySolver::Initialize() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing SEQS Maxwell solver.\n");

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // 1a) Read the serial mesh (on each mpi rank)
  Mesh *mesh = new Mesh(em_opt_.mesh_file, 1, 1);
  dim_ = mesh->Dimension();
  if (dim_ != 3) {
    if (verbose) {
      grvy_printf(gerror, "[ERROR] SEQS solver only supported for 3D.");
    }
    exit(1);
  }

  // 1b) Refine the serial mesh, if requested
  if (verbose && (em_opt_.ref_levels > 0)) {
    grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", em_opt_.ref_levels);
  }
  for (int l = 0; l < em_opt_.ref_levels; l++) {
    mesh->UniformRefinement();
  }

  // 1c) Partition the mesh
  pmesh_ = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;  // no longer need the serial mesh
  pmesh_->ReorientTetMesh();

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------
  hcurl_ = new ND_FECollection(em_opt_.order  , dim_);
  h1_    = new H1_FECollection(em_opt_.order  , dim_);
  hdiv_  = new RT_FECollection(em_opt_.order-1, dim_);

  Aspace_ = new ParFiniteElementSpace(pmesh_, hcurl_);
  pspace_ = new ParFiniteElementSpace(pmesh_, h1_   );
  Bspace_ = new ParFiniteElementSpace(pmesh_, hdiv_ );


  //-----------------------------------------------------
  // 3) Prepare volume and boundary attributes
  //-----------------------------------------------------

  // For the volume, we need to know what dofs (in the H1 space) are
  // part of the conductor.  These are determined below.
  Array<int> conductor_marker;
  conductor_marker.SetSize(pspace_->GetVSize());
  conductor_marker = 0;
  for (int ielem=0; ielem < pmesh_->GetNE(); ielem++) {
    for (int icond=0; icond < em_opt_.conductor_domains.Size(); icond++) {
      if (pmesh_->GetAttribute(ielem) == em_opt_.conductor_domains[icond]) {
        Array<int> elem_vdofs;
        pspace_->GetElementVDofs(ielem, elem_vdofs);
        mfem_mark_dofs(elem_vdofs, conductor_marker);
      }
    }
  }

  // Synchronize does bitwise OR, such that, for shared dofs, if they
  // belong to an element in the conductor on any rank, they are marked
  pspace_->Synchronize(conductor_marker);

  // Build dof list from marker array
  Array<int> conductor_true_marker, conductor_true_list;
  pspace_->GetRestrictionMatrix()->BooleanMult(conductor_marker, conductor_true_marker);
  pspace_->MarkerToList(conductor_true_marker, conductor_true_list);

  // For the boundaries, we need to know 1) which boundaries are
  // essential and which are Neumann and 2) among the essential
  // boundaries, which have non-zero electric potential.
  assert(pmesh_->bdr_attributes.Size());  // have to have some boundary attributes

  Array<int> ess_bdr;
  ess_bdr.SetSize(pmesh_->bdr_attributes.Max());
  ess_bdr = 1;  // start with everything being Dirichlet

  // Neumann on outer part (if uncommented)
  for (int i=0; i < em_opt_.neumann_bc_attr.Size(); i++) {
    ess_bdr[em_opt_.neumann_bc_attr[i]-1] = 0;
  }

  pspace_->GetEssentialTrueDofs(ess_bdr, h1_ess_tdof_list_);

  // Set the boundary indicators for the ports
  port_0_.SetSize(pmesh_->bdr_attributes.Max());
  port_0_ = 0;
  port_0_[em_opt_.port0-1] = 1;

  port_1_.SetSize(pmesh_->bdr_attributes.Max());
  port_1_ = 0;
  port_1_[em_opt_.port1-1] = 1;

  // Finally, get list of essential dofs for psi, which includes those
  // in the conductor as well as the phi Dirichlet dofs
  psi_ess_tdof_list_.SetSize(0);
  psi_ess_tdof_list_.Append(conductor_true_list);
  psi_ess_tdof_list_.Append(h1_ess_tdof_list_);

  Array<int> A_ess_bdr;
  A_ess_bdr.SetSize(pmesh_->bdr_attributes.Max());
  A_ess_bdr = 1;
  // Neumann?
  Aspace_->GetEssentialTrueDofs(A_ess_bdr, hcurl_ess_tdof_list_);


  //-----------------------------------------------------
  // 3) Prepare material parameters and other coefficients
  //-----------------------------------------------------

  // TODO(trevilo): Here, we assume all subdomains have the same
  // permittivity and (if they are conducting) the same conductivity.
  // Need to allow user to specify different relative permittivities
  // and conductivities for different subdomains.

  Vector rsig(pmesh_->attributes.Max());
  Vector reps(pmesh_->attributes.Max());
  Vector rmui(pmesh_->attributes.Max());
  Vector reps_nc(pmesh_->attributes.Max());
  Vector se2(pmesh_->attributes.Max());
  Vector se2_reg(pmesh_->attributes.Max());
  rsig = 0.0;     // domains are non-conducting by default
  se2  = 0.0;     // domains are non-conducting by default
  se2_reg = 1.0;  // not sure how to set this value (1 seems to do pretty well though)
  reps = 1.0;     // domains have relative permeability 1 by default
  rmui = 1.0;     // domains have relative permeability 1 by default
  reps_nc = 1.0;  // non-conducting domains have relative permeability 1 by default

  for (int icond=0; icond < em_opt_.conductor_domains.Size(); icond++) {
    rsig(em_opt_.conductor_domains[icond]-1) = 1.0;
    reps_nc(em_opt_.conductor_domains[icond]-1) = 0.0;
    se2(em_opt_.conductor_domains[icond]-1) = em_opt_.nd_conductivity*em_opt_.nd_frequency*em_opt_.nd_frequency;
    se2_reg(em_opt_.conductor_domains[icond]-1) = em_opt_.nd_conductivity*em_opt_.nd_frequency*em_opt_.nd_frequency;
  }
  rel_sig_    = new PWConstCoefficient(rsig);
  rel_eps_    = new PWConstCoefficient(reps);
  rel_mui_    = new PWConstCoefficient(rmui);
  rel_eps_nc_ = new PWConstCoefficient(reps_nc);

  one_over_sigma_ = new ConstantCoefficient(1.0/em_opt_.nd_conductivity);
  eta_squared_ = new ConstantCoefficient(em_opt_.nd_frequency*em_opt_.nd_frequency);

  S_eta2_ = new PWConstCoefficient(se2);
  S_eta2_reg_ = new PWConstCoefficient(se2_reg);
}

void SeqsMaxwellFrequencySolver::Solve() {
  SolveSEQS();
  SolveStabMaxwell();
  EvaluateEandB();
  EvaluateCurrent();
  WriteParaview();
}

void SeqsMaxwellFrequencySolver::SolveSEQS() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Solving stabilized electro-quasistatic problem.\n");

  if (verbose) grvy_printf(ginfo, "... Initializing solution functions.\n");
  phi_real_ = new ParGridFunction(pspace_);
  phi_imag_ = new ParGridFunction(pspace_);
  psi_real_ = new ParGridFunction(pspace_);
  psi_imag_ = new ParGridFunction(pspace_);

  *phi_real_ = 0.0;
  *phi_imag_ = 0.0;
  *psi_real_ = 0.0;
  *psi_imag_ = 0.0;

  ConstantCoefficient one(1.0);

  V0_real_ = new ParGridFunction(pspace_);
  *V0_real_ = 0.0;
  V0_real_->ProjectBdrCoefficient(one, port_0_);

  V0_imag_ = new ParGridFunction(pspace_);
  *V0_imag_ = 0.0;
  V0_imag_->ProjectBdrCoefficient(one, port_0_);

  V1_real_ = new ParGridFunction(pspace_);
  *V1_real_ = 0.0;
  V1_real_->ProjectBdrCoefficient(one, port_1_);

  V1_imag_ = new ParGridFunction(pspace_);
  *V1_imag_ = 0.0;
  V1_imag_->ProjectBdrCoefficient(one, port_1_);

  if (verbose) grvy_printf(ginfo, "... Initializing right hand side.\n");
  ParLinearForm *bp_real = new ParLinearForm(pspace_);
  ParLinearForm *bp_imag = new ParLinearForm(pspace_);
  ParLinearForm *bs_real = new ParLinearForm(pspace_);
  ParLinearForm *bs_imag = new ParLinearForm(pspace_);

  *bp_real = 0.0;
  *bp_imag = 0.0;
  *bs_real = 0.0;
  *bs_imag = 0.0;

  {
    ParBilinearForm *Kpp_real = new ParBilinearForm(pspace_);
    Kpp_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
    Kpp_real->Assemble();
    Kpp_real->Finalize();
    Kpp_real->AddMult(*V0_real_, *bp_real, -em_opt_.Vstat0_real);
    Kpp_real->AddMult(*V1_real_, *bp_real, -em_opt_.Vstat1_real);
    Kpp_real->AddMult(*V0_imag_, *bp_imag, -em_opt_.Vstat0_imag);
    Kpp_real->AddMult(*V1_imag_, *bp_imag, -em_opt_.Vstat1_imag);
    delete Kpp_real;

    ParBilinearForm *Kpp_imag = new ParBilinearForm(pspace_);
    Kpp_imag->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
    Kpp_imag->Assemble();
    Kpp_imag->Finalize();
    Kpp_imag->AddMult(*V0_real_, *bp_imag, -em_opt_.Vstat0_real);
    Kpp_imag->AddMult(*V1_real_, *bp_imag, -em_opt_.Vstat1_real);
    Kpp_imag->AddMult(*V0_imag_, *bp_real,  em_opt_.Vstat0_imag);
    Kpp_imag->AddMult(*V1_imag_, *bp_real,  em_opt_.Vstat1_imag);
    delete Kpp_imag;

    ParBilinearForm *Kss_real = new ParBilinearForm(pspace_);
    Kss_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Kss_real->Assemble();
    Kss_real->Finalize();
    Kss_real->AddMult(*V0_real_, *bs_real, -em_opt_.Vstat0_real);
    Kss_real->AddMult(*V1_real_, *bs_real, -em_opt_.Vstat1_real);
    Kss_real->AddMult(*V0_imag_, *bs_imag, -em_opt_.Vstat0_imag);
    Kss_real->AddMult(*V1_imag_, *bs_imag, -em_opt_.Vstat1_imag);
    delete Kss_real;
  }

  // True DOFs (for setting up block vectors and matrices below)
  const int true_psize = pspace_->TrueVSize();

  Array<int> true_offsets(4+1);  // num variables (+1)
  true_offsets[0] = 0;
  true_offsets[1] = true_psize;
  true_offsets[2] = true_psize;
  true_offsets[3] = true_psize;
  true_offsets[4] = true_psize;
  true_offsets.PartialSum();


  BlockVector u(true_offsets), b(true_offsets);

  phi_real_->ParallelAssemble(u.GetBlock(0));
  phi_imag_->ParallelAssemble(u.GetBlock(1));
  psi_real_->ParallelAssemble(u.GetBlock(2));
  psi_imag_->ParallelAssemble(u.GetBlock(3));

  bp_real->ParallelAssemble(b.GetBlock(0));
  bp_imag->ParallelAssemble(b.GetBlock(1));
  bs_real->ParallelAssemble(b.GetBlock(2));
  bs_imag->ParallelAssemble(b.GetBlock(3));

  { // begin scope for SEQS solver
    if (verbose) grvy_printf(ginfo, "... Initializing SEQS operators.\n");

    // 11. Set up required bilinear forms
    ParBilinearForm *Kpp_real = new ParBilinearForm(pspace_);
    Kpp_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
    Kpp_real->Assemble();

#if (MFEM_VERSION >= 40300)
    // In mfem v4.3, ParBilinearForm::EliminateVDofsInRHS is
    // implemented and calls EliminateBC to accomplish the desired
    // modification of the RHS
    OperatorPtr Kpp_real_op;
    Kpp_real->FormSystemMatrix(h1_ess_tdof_list_, Kpp_real_op);
    Kpp_real->EliminateVDofsInRHS(h1_ess_tdof_list_, u.GetBlock(0), b.GetBlock(0));
    Kpp_real->EliminateVDofsInRHS(h1_ess_tdof_list_, u.GetBlock(1), b.GetBlock(1));
#else
    // In mfem v4.2 (and before) ParBilinearForm::EliminateVDofsInRHS
    // isn't explicitly implemented, so becomes call to
    // BilinearForm::EliminateVDofsInRHS, which leads to seg faults or
    // incorrect results.  Instead, we have to explicitly call
    // ParallelAssemble and EliminateBC as below.
    Kpp_real->Finalize(0);

    OperatorPtr Kpp_real_op(Operator::Hypre_ParCSR);
    Kpp_real->ParallelAssemble(Kpp_real_op);

    OperatorPtr Kpp_real_op_elim(Operator::Hypre_ParCSR);
    Kpp_real_op_elim.EliminateRowsCols(Kpp_real_op, h1_ess_tdof_list_);
    Kpp_real_op.EliminateBC(Kpp_real_op_elim, h1_ess_tdof_list_, u.GetBlock(0), b.GetBlock(0));
    Kpp_real_op.EliminateBC(Kpp_real_op_elim, h1_ess_tdof_list_, u.GetBlock(1), b.GetBlock(1));
#endif

    ParBilinearForm *Kpp_imag = new ParBilinearForm(pspace_);
    Kpp_imag->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
    Kpp_imag->Assemble();

    OperatorPtr Kpp_imag_op;
    Kpp_imag->FormSystemMatrix(h1_ess_tdof_list_, Kpp_imag_op);

    ParBilinearForm *Kps_real = new ParBilinearForm(pspace_);
    Kps_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
    Kps_real->Assemble();

    OperatorPtr Kps_real_op;
    Kps_real->FormSystemMatrix(h1_ess_tdof_list_, Kps_real_op);

    ParBilinearForm *Kps_imag = new ParBilinearForm(pspace_);
    Kps_imag->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
    Kps_imag->Assemble();

    OperatorPtr Kps_imag_op;
    Kps_imag->FormSystemMatrix(h1_ess_tdof_list_, Kps_imag_op);

    ParBilinearForm *Ksp_real = new ParBilinearForm(pspace_);
    Ksp_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Ksp_real->Assemble();

    OperatorPtr Ksp_real_op;
    Ksp_real->FormSystemMatrix(h1_ess_tdof_list_, Ksp_real_op);

    ParBilinearForm *Kss_real = new ParBilinearForm(pspace_);
    Kss_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Kss_real->Assemble();

#if (MFEM_VERSION >= 40300)
    // See comments above about why #if necessary here
    OperatorPtr Kss_real_op;
    Kss_real->FormSystemMatrix(psi_ess_tdof_list_, Kss_real_op);
    Kss_real->EliminateVDofsInRHS(psi_ess_tdof_list_, u.GetBlock(2), b.GetBlock(2));
    Kss_real->EliminateVDofsInRHS(psi_ess_tdof_list_, u.GetBlock(3), b.GetBlock(3));
#else
    Kss_real->Finalize(0);

    OperatorPtr Kss_real_op(Operator::Hypre_ParCSR);
    Kss_real->ParallelAssemble(Kss_real_op);

    OperatorPtr Kss_real_op_elim(Operator::Hypre_ParCSR);
    Kss_real_op_elim.EliminateRowsCols(Kss_real_op, psi_ess_tdof_list_);
    Kss_real_op.EliminateBC(Kss_real_op_elim, psi_ess_tdof_list_, u.GetBlock(2), b.GetBlock(2));
    Kss_real_op.EliminateBC(Kss_real_op_elim, psi_ess_tdof_list_, u.GetBlock(3), b.GetBlock(3));
#endif

    // Put the operators formed above into a single block operator object
    BlockOperator *maxwellOp = new BlockOperator(true_offsets);

    // Kpp block
    HypreParMatrix *kppr = Kpp_real_op.As<HypreParMatrix>();
    HypreParMatrix *kppi = Kpp_imag_op.As<HypreParMatrix>();

    // Elim rows from off-diagonal block (b/c FormSystemMatrix call
    // above leave 1 on diagonal and doesn't seem to respect
    // SetDiagonalPolicy call).  This logic is repeated for all
    // off-diagonal blocks below.
    kppi->EliminateRows(h1_ess_tdof_list_);

    maxwellOp->SetBlock(0, 0, kppr);
    maxwellOp->SetBlock(0, 1, kppi, -1.0);
    maxwellOp->SetBlock(1, 0, kppi);
    maxwellOp->SetBlock(1, 1, kppr);

    // Kps block
    HypreParMatrix *kpsr = Kps_real_op.As<HypreParMatrix>();
    HypreParMatrix *kpsi = Kps_imag_op.As<HypreParMatrix>();

    kpsr->EliminateRows(h1_ess_tdof_list_);
    kpsi->EliminateRows(h1_ess_tdof_list_);

    maxwellOp->SetBlock(0, 2, kpsr);
    maxwellOp->SetBlock(0, 3, kpsi, -1.0);
    maxwellOp->SetBlock(1, 2, kpsi);
    maxwellOp->SetBlock(1, 3, kpsr);

    // Ksp block
    HypreParMatrix *kspr = Ksp_real_op.As<HypreParMatrix>();
    kspr->EliminateRows(psi_ess_tdof_list_);

    maxwellOp->SetBlock(2, 0, kspr);
    maxwellOp->SetBlock(3, 1, kspr);

    // Kss block
    HypreParMatrix *kssr = Kss_real_op.As<HypreParMatrix>();

    maxwellOp->SetBlock(2, 2, kssr);
    maxwellOp->SetBlock(3, 3, kssr);


    // 12. Set up the preconditioner
    if (verbose) grvy_printf(ginfo, "... Initializing preconditioner.\n");

    // phi-phi block
    ParBilinearForm *Kpp_pre = new ParBilinearForm(pspace_);
    Kpp_pre->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
    Kpp_pre->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
    Kpp_pre->Assemble();
    OperatorPtr Kpp_pre_op;
    Kpp_pre->FormSystemMatrix(h1_ess_tdof_list_, Kpp_pre_op);

    Operator *Pppr = new HypreBoomerAMG(*Kpp_pre_op.As<HypreParMatrix>());

    // TODO(trevilo): should this be -1?
    // Patterned off mfem/examples/ex22p.cpp, but not clear to me why
    // this is a good choice.
    // Operator *Pppi = new ScaledOperator(Pppr, -1.0);
    Operator *Pppi = new ScaledOperator(Pppr, 1.0);

    // psi-psi block
    ParBilinearForm *Kss_pre = new ParBilinearForm(pspace_);
    Kss_pre->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Kss_pre->Assemble();
    OperatorPtr Kss_pre_op;
    Kss_pre->FormSystemMatrix(psi_ess_tdof_list_, Kss_pre_op);

    Operator *Pssr = new HypreBoomerAMG(*Kss_pre_op.As<HypreParMatrix>());

    // TODO(trevilo): should this be -1?  Patterned off
    // mfem/examples/ex22p.cpp, but not clear to me why this is a good
    // choice.  Empirically, +1.0 works better, so going with that for
    // the moment.
    // Operator *Pssi = new ScaledOperator(Pssr, -1.0);
    Operator *Pssi = new ScaledOperator(Pssr, 1.0);

    BlockDiagonalPreconditioner BDP(true_offsets);
    BDP.SetDiagonalBlock(0, Pppr);
    BDP.SetDiagonalBlock(1, Pppi);
    BDP.SetDiagonalBlock(2, Pssr);
    BDP.SetDiagonalBlock(3, Pssi);
    BDP.owns_blocks = 1;

    // 13. Solve the thing (finally!)
    if (verbose) grvy_printf(ginfo, "... Solving SEQS linear system.\n");

    BiCGSTABSolver solver(MPI_COMM_WORLD);
    solver.SetPreconditioner(BDP);
    solver.SetOperator(*maxwellOp);
    solver.SetRelTol(em_opt_.rtol);
    solver.SetAbsTol(em_opt_.atol);
    solver.SetMaxIter(em_opt_.max_iter);
    solver.SetPrintLevel(0);
    solver.Mult(b, u);

    // Recover the solution
    phi_real_->Distribute(&(u.GetBlock(0)));
    phi_imag_->Distribute(&(u.GetBlock(1)));
    psi_real_->Distribute(&(u.GetBlock(2)));
    psi_imag_->Distribute(&(u.GetBlock(3)));

    // clean up
    delete Kss_pre;
    delete Kpp_pre;
    delete maxwellOp;
    delete Kss_real;
    delete Ksp_real;
    delete Kps_imag;
    delete Kps_real;
    delete Kpp_imag;
    delete Kpp_real;
  }  // end of scope for solver

  // clean up
  delete bs_imag;
  delete bs_real;
  delete bp_imag;
  delete bp_real;

  // Combine the three contributions to the electric potential into a
  // single function, which will be used by the magnetic vector
  // potential solve
  phi_tot_real_ = new ParGridFunction(pspace_);
  *phi_tot_real_ = 0.0;
  add(*phi_tot_real_ , em_opt_.Vstat0_real, *V0_real_, *phi_tot_real_);
  add(*phi_tot_real_ , em_opt_.Vstat1_real, *V1_real_, *phi_tot_real_);
  *phi_tot_real_ += *phi_real_;
  *phi_tot_real_ += *psi_real_;

  phi_tot_imag_ = new ParGridFunction(pspace_);
  *phi_tot_imag_  = 0.0;
  add(*phi_tot_imag_ , em_opt_.Vstat0_imag, *V0_imag_, *phi_tot_imag_);
  add(*phi_tot_imag_ , em_opt_.Vstat1_imag, *V1_imag_, *phi_tot_imag_);
  *phi_tot_imag_ += *phi_imag_;
  *phi_tot_imag_ += *psi_imag_;


  if (verbose) grvy_printf(ginfo, "Stabilized electro-quasistatic finished.\n");
}

void SeqsMaxwellFrequencySolver::SolveStabMaxwell() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Solving stabilized Maxwell problem for A.\n");

  if (verbose) grvy_printf(ginfo, "... Initializing solution functions.\n");
  A_real_ = new ParGridFunction(Aspace_);
  A_imag_ = new ParGridFunction(Aspace_);
  n_real_ = new ParGridFunction(pspace_);
  n_imag_ = new ParGridFunction(pspace_);

  *A_real_ = 0.0;
  *A_imag_ = 0.0;
  *n_real_ = 0.0;
  *n_imag_ = 0.0;

  if (verbose) grvy_printf(ginfo, "... Initializing right hand side.\n");
  ParLinearForm *bA_real = new ParLinearForm(Aspace_);
  ParLinearForm *bA_imag = new ParLinearForm(Aspace_);
  ParLinearForm *bn_real = new ParLinearForm(pspace_);
  ParLinearForm *bn_imag = new ParLinearForm(pspace_);

  *bA_real = 0.0;
  *bA_imag = 0.0;
  *bn_real = 0.0;
  *bn_imag = 0.0;

  // build RHS
  {
    ParMixedBilinearForm *Kap_real = new ParMixedBilinearForm(pspace_, Aspace_);
    Kap_real->AddDomainIntegrator(new MixedVectorGradientIntegrator(*S_eta2_));
    Kap_real->Assemble();
    Kap_real->Finalize();
    Kap_real->AddMult(*phi_tot_real_, *bA_real, -1.0);
    Kap_real->AddMult(*phi_tot_imag_, *bA_imag, -1.0);
    delete Kap_real;

    ParMixedBilinearForm *Kap_imag = new ParMixedBilinearForm(pspace_, Aspace_);
    Kap_imag->AddDomainIntegrator(new MixedVectorGradientIntegrator(*eta_squared_));
    Kap_imag->Assemble();
    Kap_imag->Finalize();
    Kap_imag->AddMult(*phi_tot_real_, *bA_imag, -1.0);
    Kap_imag->AddMult(*phi_tot_imag_, *bA_real, +1.0);
    delete Kap_imag;
  }

  // True DOFs (for setting up block vectors and matrices below)
  const int true_nsize = pspace_->TrueVSize();
  const int true_Asize = Aspace_->TrueVSize();

  Array<int> sm_offsets(4+1);  // num variables (+1)
  sm_offsets[0] = 0;
  sm_offsets[1] = true_Asize;
  sm_offsets[2] = true_Asize;
  sm_offsets[3] = true_nsize;
  sm_offsets[4] = true_nsize;
  sm_offsets.PartialSum();

  BlockVector X(sm_offsets), R(sm_offsets);

  A_real_->ParallelAssemble(X.GetBlock(0));
  A_imag_->ParallelAssemble(X.GetBlock(1));
  n_real_->ParallelAssemble(X.GetBlock(2));
  n_imag_->ParallelAssemble(X.GetBlock(3));

  bA_real->ParallelAssemble(R.GetBlock(0));
  bA_imag->ParallelAssemble(R.GetBlock(1));
  bn_real->ParallelAssemble(R.GetBlock(2));
  bn_imag->ParallelAssemble(R.GetBlock(3));

  if (verbose) grvy_printf(ginfo, "... Initializing stabilized Maxwell operators.\n");

  {
    ParBilinearForm *Kaa_real = new ParBilinearForm(Aspace_);
    Kaa_real->AddDomainIntegrator(new CurlCurlIntegrator(*rel_mui_));
    Kaa_real->AddDomainIntegrator(new VectorFEMassIntegrator(*neg_eta_squared_));
    Kaa_real->Assemble();
#if (MFEM_VERSION >= 40300)
    // See comments above about why #if necessary here
    OperatorPtr Kaa_real_op;
    Kaa_real->FormSystemMatrix(hcurl_ess_tdof_list, Kaa_real_op);
    Kaa_real->EliminateVDofsInRHS(hcurl_ess_tdof_list_, X.GetBlock(0), R.GetBlock(0));
    Kaa_real->EliminateVDofsInRHS(hcurl_ess_tdof_list_, X.GetBlock(1), R.GetBlock(1));
#else
    Kaa_real->Finalize(0);

    OperatorPtr Kaa_real_op(Operator::Hypre_ParCSR);
    Kaa_real->ParallelAssemble(Kaa_real_op);

    OperatorPtr Kaa_real_op_elim(Operator::Hypre_ParCSR);
    Kaa_real_op_elim.EliminateRowsCols(Kaa_real_op, hcurl_ess_tdof_list_);
    Kaa_real_op.EliminateBC(Kaa_real_op_elim, hcurl_ess_tdof_list_, X.GetBlock(0), R.GetBlock(0));
    Kaa_real_op.EliminateBC(Kaa_real_op_elim, hcurl_ess_tdof_list_, X.GetBlock(1), R.GetBlock(1));
#endif

    ParBilinearForm *Kaa_imag = new ParBilinearForm(Aspace_);
    Kaa_imag->AddDomainIntegrator(new VectorFEMassIntegrator(*S_eta2_));
    Kaa_imag->Assemble();
    OperatorPtr Kaa_imag_op;
    Kaa_imag->FormSystemMatrix(hcurl_ess_tdof_list_, Kaa_imag_op);
    // todo: eliminate rows

    ParMixedBilinearForm *Kan_imag = new ParMixedBilinearForm(pspace_, Aspace_);
    Kan_imag->AddDomainIntegrator(new MixedVectorGradientIntegrator(*S_eta2_));
    Kan_imag->Assemble();
    OperatorPtr Kan_imag_op;
    Kan_imag->FormRectangularSystemMatrix(psi_ess_tdof_list_, hcurl_ess_tdof_list_, Kan_imag_op);

    ParMixedBilinearForm *Kan_real = new ParMixedBilinearForm(pspace_, Aspace_);
    Kan_real->AddDomainIntegrator(new MixedVectorGradientIntegrator(*neg_eta_squared_));
    Kan_real->Assemble();
    OperatorPtr Kan_real_op;
    Kan_real->FormRectangularSystemMatrix(psi_ess_tdof_list_, hcurl_ess_tdof_list_, Kan_real_op);

    ParMixedBilinearForm *Kna_real = new ParMixedBilinearForm(Aspace_, pspace_);
    Kna_real->AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator(*rel_eps_nc_));
    Kna_real->Assemble();
    OperatorPtr Kna_real_op;
    Kna_real->FormRectangularSystemMatrix(hcurl_ess_tdof_list_, psi_ess_tdof_list_, Kna_real_op);

    ParBilinearForm *Knn_real = new ParBilinearForm(pspace_);
    Knn_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Knn_real->Assemble();

#if (MFEM_VERSION >= 40300)
    OperatorPtr Knn_real_op;
    Knn_real->FormSystemMatrix(psi_ess_tdof_list_, Knn_real_op);
    Knn_real->EliminateVDofsInRHS(psi_ess_tdof_list_, X.GetBlock(2), R.GetBlock(2));
    Knn_real->EliminateVDofsInRHS(psi_ess_tdof_list_, X.GetBlock(3), R.GetBlock(3));
#else
    Knn_real->Finalize(0);

    OperatorPtr Knn_real_op(Operator::Hypre_ParCSR);
    Knn_real->ParallelAssemble(Knn_real_op);

    OperatorPtr Knn_real_op_elim(Operator::Hypre_ParCSR);
    Knn_real_op_elim.EliminateRowsCols(Knn_real_op, psi_ess_tdof_list_);
    Knn_real_op.EliminateBC(Knn_real_op_elim, psi_ess_tdof_list_, X.GetBlock(2), R.GetBlock(2));
    Knn_real_op.EliminateBC(Knn_real_op_elim, psi_ess_tdof_list_, X.GetBlock(3), R.GetBlock(3));
#endif

    // Put the operators formed above into a single block operator object
    BlockOperator *maxwellOp = new BlockOperator(sm_offsets);

    // Kaa block
    HypreParMatrix *kaar = Kaa_real_op.As<HypreParMatrix>();
    HypreParMatrix *kaai = Kaa_imag_op.As<HypreParMatrix>();

    // Elim rows from off-diagonal block (b/c FormSystemMatrix call
    // above leave 1 on diagonal and doesn't seem to respect
    // SetDiagonalPolicy call).  This logic is repeated for all
    // off-diagonal blocks below.
    kaai->EliminateRows(hcurl_ess_tdof_list_);

    maxwellOp->SetBlock(0, 0, kaar);
    maxwellOp->SetBlock(0, 1, kaai, -1.0);
    maxwellOp->SetBlock(1, 0, kaai);
    maxwellOp->SetBlock(1, 1, kaar);

    // Kan block
    HypreParMatrix *kanr = Kan_real_op.As<HypreParMatrix>();
    HypreParMatrix *kani = Kan_imag_op.As<HypreParMatrix>();

    maxwellOp->SetBlock(0, 2, kanr);
    maxwellOp->SetBlock(0, 3, kani, -1.0);
    maxwellOp->SetBlock(1, 2, kani);
    maxwellOp->SetBlock(1, 3, kanr);

    // Kna block
    HypreParMatrix *knar = Kna_real_op.As<HypreParMatrix>();

    maxwellOp->SetBlock(2, 0, knar);
    maxwellOp->SetBlock(3, 1, knar);

    // Knn block
    HypreParMatrix *knnr = Knn_real_op.As<HypreParMatrix>();

    maxwellOp->SetBlock(2, 2, knnr);
    maxwellOp->SetBlock(3, 3, knnr);

    // A-A block
    ParBilinearForm *Kaa_pre = new ParBilinearForm(Aspace_);
    Kaa_pre->AddDomainIntegrator(new CurlCurlIntegrator(*rel_mui_));
    Kaa_pre->AddDomainIntegrator(new VectorFEMassIntegrator(*S_eta2_reg_));
    Kaa_pre->Assemble();
    OperatorPtr Kaa_pre_op;
    Kaa_pre->FormSystemMatrix(hcurl_ess_tdof_list_, Kaa_pre_op);

    Operator *Paar = new HypreAMS(*Kaa_pre_op.As<HypreParMatrix>(), Aspace_);
    // Operator *Paai = new ScaledOperator(Paar,-1.0);
    Operator *Paai = new ScaledOperator(Paar, 1.0);

    // psi-psi block
    ParBilinearForm *Knn_pre = new ParBilinearForm(pspace_);
    Knn_pre->AddDomainIntegrator(new DiffusionIntegrator(*rel_eps_nc_));
    Knn_pre->Assemble();
    OperatorPtr Knn_pre_op;
    Knn_pre->FormSystemMatrix(psi_ess_tdof_list_, Knn_pre_op);

    Operator *Pnnr = new HypreBoomerAMG(*Knn_pre_op.As<HypreParMatrix>());
    // Operator *Pnni = new ScaledOperator(Pnnr,-1.0);
    Operator *Pnni = new ScaledOperator(Pnnr, 1.0);

    BlockDiagonalPreconditioner BDP(sm_offsets);
    BDP.SetDiagonalBlock(0, Paar);
    BDP.SetDiagonalBlock(1, Paai);
    BDP.SetDiagonalBlock(2, Pnnr);
    BDP.SetDiagonalBlock(3, Pnni);
    BDP.owns_blocks = 1;

    // Solve the thing (finally!)
    FGMRESSolver solver(MPI_COMM_WORLD);
    solver.SetPreconditioner(BDP);
    solver.SetOperator(*maxwellOp);
    solver.SetRelTol(em_opt_.rtol);
    solver.SetAbsTol(em_opt_.atol);
    solver.SetMaxIter(em_opt_.max_iter);
    solver.SetPrintLevel(1);
    solver.Mult(R, X);

    // Recover the solution
    A_real_->Distribute(&(X.GetBlock(0)));
    A_imag_->Distribute(&(X.GetBlock(1)));
    n_real_->Distribute(&(X.GetBlock(2)));
    n_imag_->Distribute(&(X.GetBlock(3)));

    // clean up
    delete Knn_pre;
    delete Kaa_pre;
    delete maxwellOp;
    delete Knn_real;
    delete Kna_real;
    delete Kan_real;
    delete Kan_imag;
    delete Kaa_imag;
    delete Kaa_real;
    delete bA_real;
    delete bA_imag;
    delete bn_imag;
    delete bn_real;
  }
}

void SeqsMaxwellFrequencySolver::EvaluateEandB() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Evaluating the magnetic field.\n");

  assert(A_real_ != NULL);
  assert(A_imag_ != NULL);

  // First, we evaluate the B field, which is the curl of the magnetic
  // vector potential.  The full magnetic vector potential is given by
  // A + grad(n).  Thus, B = curl(A + grad(n)) = curl(A).
  B_real_ = new ParGridFunction(Bspace_);
  *B_real_ = 0;

  B_imag_ = new ParGridFunction(Bspace_);
  *B_imag_ = 0;

  ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(Aspace_, Bspace_);
  curl->AddDomainInterpolator(new CurlInterpolator);
  curl->Assemble();
  curl->Finalize();

  curl->Mult(*A_real_, *B_real_);
  curl->Mult(*A_imag_, *B_imag_);

  delete curl;

  if (verbose) grvy_printf(ginfo, "Evaluating the electric field.\n");
  assert(phi_tot_real_ != NULL);
  assert(phi_tot_imag_ != NULL);
  assert(n_real_ != NULL);
  assert(n_imag_ != NULL);

  // Second, we evaluate the E field, which is the opposite of the
  // gradient of the electric potential minus the time derivative of
  // the magnetic vector potential:
  //
  // E = -grad(phi_tot) - dA_tot/dt,
  //
  // where phi_tot and A_tot are the "total" potentials (i.e. phi_tot
  // = phi+psi+V_0+V_1) and (A_tot = A + grad(n)).  In the frequency
  // domain, we have
  //
  // E = -grad(phi_tot) - j*omega*A_tot.
  //
  // After non-dimensionalizing, this becomes
  //
  // E = -grad(phi_tot) - j*A_tot,
  //
  // such that
  //
  // E_real = -grad(phi_tot_real) + A_tot_imag
  // E_imag = -grad(phi_tot_imag) - A_tot_real
  E_real_ = new ParGridFunction(Aspace_);
  *E_real_ = 0;

  E_imag_ = new ParGridFunction(Aspace_);
  *E_imag_ = 0;

  ParDiscreteLinearOperator *grad = new ParDiscreteLinearOperator(pspace_, Aspace_);
  grad->AddDomainInterpolator(new GradientInterpolator);
  grad->Assemble();
  grad->Finalize();

  grad->AddMult(*phi_tot_real_, *E_real_, -1.0);
  grad->AddMult(*phi_tot_imag_, *E_imag_, -1.0);

  *E_real_ += *A_imag_;
  *E_imag_ -= *A_real_;

  grad->AddMult(*n_imag_, *E_real_, +1.0);
  grad->AddMult(*n_real_, *E_imag_, -1.0);

  delete grad;
}

void SeqsMaxwellFrequencySolver::EvaluateCurrent() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Evaluating the current.\n");

  ParLinearForm *bp_real = new ParLinearForm(pspace_);
  *bp_real = 0.0;

  ParLinearForm *bp_imag = new ParLinearForm(pspace_);
  *bp_imag = 0.0;

  // First, the "static" part
  ParBilinearForm *Kpp_real = new ParBilinearForm(pspace_);
  Kpp_real->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
  Kpp_real->Assemble();
  Kpp_real->Finalize();
  Kpp_real->AddMult(*phi_tot_real_, *bp_real, 1.0);
  Kpp_real->AddMult(*phi_tot_imag_, *bp_imag, 1.0);
  delete Kpp_real;

  ParBilinearForm *Kpp_imag = new ParBilinearForm(pspace_);
  Kpp_imag->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
  Kpp_imag->Assemble();
  Kpp_imag->Finalize();
  Kpp_imag->AddMult(*phi_tot_imag_, *bp_real, -1.0);
  Kpp_imag->AddMult(*phi_tot_real_, *bp_imag,  1.0);
  delete Kpp_imag;

  // call operator() for linear forms
  const double I_stat_real = (*bp_real)(*V0_real_);
  const double I_stat_imag = (*bp_imag)(*V0_real_);

  if (verbose) grvy_printf(ginfo, "I_stat_real = %.8e\n", I_stat_real);
  if (verbose) grvy_printf(ginfo, "I_stat_imag = %.8e\n", I_stat_imag);

  // these linear forms are re-used, so zero them out
  *bp_real = 0.0;
  *bp_imag = 0.0;


  // Second, the "induced" part
  ParMixedBilinearForm *Kna_imag = new ParMixedBilinearForm(Aspace_, pspace_);
  Kna_imag->AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator(*rel_sig_));  // should be -sigma
  Kna_imag->Assemble();
  Kna_imag->Finalize();

  Kna_imag->AddMult(*A_real_, *bp_imag, -1.0);  // sign flip accounts for -sigma (see above)
  Kna_imag->AddMult(*A_imag_, *bp_real,  1.0);  // sign flip accounts for -sigma (see above)
  delete Kna_imag;

  ParMixedBilinearForm *Kna_real = new ParMixedBilinearForm(Aspace_, pspace_);
  Kna_real->AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator(*one_over_sigma_));  // should be -1/sigma
  Kna_real->Assemble();
  Kna_real->Finalize();

  Kna_real->AddMult(*A_real_, *bp_real, 1.0);  // sign flip accounts for -1/sigma (see above)
  Kna_real->AddMult(*A_imag_, *bp_imag, 1.0);  // sign flip accounts for -1/sigma (see above)
  delete Kna_real;

  ParBilinearForm *Knn_real = new ParBilinearForm(pspace_);
  Knn_real->AddDomainIntegrator(new DiffusionIntegrator(*one_over_sigma_));
  Knn_real->Assemble();
  Knn_real->Finalize();

  Knn_real->AddMult(*n_real_, *bp_real, -1.0);
  Knn_real->AddMult(*n_imag_, *bp_imag, -1.0);
  delete Knn_real;

  ParBilinearForm *Knn_imag = new ParBilinearForm(pspace_);
  Knn_imag->AddDomainIntegrator(new DiffusionIntegrator(*rel_sig_));
  Knn_imag->Assemble();
  Knn_imag->Finalize();

  Knn_imag->AddMult(*n_real_, *bp_imag,  1.0);
  Knn_imag->AddMult(*n_imag_, *bp_real, -1.0);
  delete Knn_imag;

  const double I_ind_real = (*bp_real)(*V0_real_);
  const double I_ind_imag = (*bp_imag)(*V0_real_);

  delete bp_imag;
  delete bp_real;

  if (verbose) grvy_printf(ginfo, "I_ind_real  = %.8e\n", I_ind_real);
  if (verbose) grvy_printf(ginfo, "I_ind_imag  = %.8e\n", I_ind_imag);

  if (verbose) grvy_printf(ginfo, "I_tot_real  = %.8e\n", I_stat_real + I_ind_real);
  if (verbose) grvy_printf(ginfo, "I_tot_imag  = %.8e\n", I_stat_imag + I_ind_imag);
}

void SeqsMaxwellFrequencySolver::WriteParaview() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Writing Maxwell solution to paraview output.\n");
  ParaViewDataCollection paraview_dc("seqs", pmesh_);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(em_opt_.order);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);
  // paraview_dc.RegisterField("phi_tot_real", phi_tot_real_);
  // paraview_dc.RegisterField("phi_tot_imag", phi_tot_imag_);
  // paraview_dc.RegisterField("A_real", A_real_);
  // paraview_dc.RegisterField("A_imag", A_imag_);
  paraview_dc.RegisterField("E_real", E_real_);
  paraview_dc.RegisterField("E_imag", E_imag_);
  paraview_dc.RegisterField("B_real", B_real_);
  paraview_dc.RegisterField("B_imag", B_imag_);
  paraview_dc.Save();
}
