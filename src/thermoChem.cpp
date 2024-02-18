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

// #include "turbModel.hpp"
#include "thermoChem.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

// temporary
double temp_ic(const Vector &coords, double t);
double temp_wall(const Vector &x, double t);
double temp_wallBox(const Vector &x, double t);
double temp_inlet(const Vector &x, double t);

ThermoChem::ThermoChem(mfem::ParMesh *pmesh, RunConfiguration *config, LoMachOptions *loMach_opts,
                       temporalSchemeCoefficients &timeCoeff)
    : pmesh_(pmesh), loMach_opts_(loMach_opts), config_(config), timeCoeff_(timeCoeff) {}

void ThermoChem::initializeSelf() {
  // groupsMPI = pmesh->GetComm();
  rank = pmesh_->GetMyRank();
  rank0 = false;
  if (rank == 0) {
    rank0 = true;
  }
  dim = pmesh_->Dimension();
  nvel = dim;

  // config settings for thermoChem
  if (config_->const_visc > 0.0) {
    constantViscosity = true;
  }
  if (config_->const_dens > 0.0) {
    constantDensity = true;
  }
  if (constantViscosity == true && constantDensity == true) {
    incompressibleSolve = true;
  }

  bool verbose = rank0;
  if (verbose) grvy_printf(ginfo, "Initializing ThermoChem solver.\n");

  if (loMach_opts_->uOrder == -1) {
    order = std::max(config_->solOrder, 1);
    porder = order;
    double no;
    no = ceil(((double)order * 1.5));
    norder = int(no);
  } else {
    order = loMach_opts_->uOrder;
    porder = loMach_opts_->pOrder;
    norder = loMach_opts_->nOrder;
  }

  static_rho = config_->const_dens;
  dyn_vis = config_->const_visc;
  ambientPressure = config_->amb_pres;
  thermoPressure = config_->amb_pres;
  tPm1 = thermoPressure;
  tPm2 = thermoPressure;
  dtP = 0.0;

  // HARD CODE HARDCODE get for dry air
  Rgas = 287.0;
  Pr = 0.76;
  Cp = 1000.5;
  gamma = 1.4;

  mixture = NULL;

  // HARD CODE HARD CODE HARD CODE
  config_->dryAirInput.specific_heat_ratio = 1.4;
  config_->dryAirInput.gas_constant = 287.058;
  config_->dryAirInput.f = config_->workFluid;
  config_->dryAirInput.eq_sys = config_->eqSystem;
  mixture = new DryAir(*config_, dim, nvel);
  transportPtr = new DryAirTransport(mixture, *config_);

  MaxIters = config_->GetNumIters();
  num_equation = 1;  // hard code for now, fix with species
  /*
  numSpecies = mixture->GetNumSpecies();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
  */

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec = new H1_FECollection(order);
  sfes = new ParFiniteElementSpace(pmesh_, sfec);

  // vector
  vfec = new H1_FECollection(order, dim);
  vfes = new ParFiniteElementSpace(pmesh_, vfec, dim);

  // pressure
  pfec = new H1_FECollection(porder);
  pfes = new ParFiniteElementSpace(pmesh_, pfec);

  // Check if fully periodic mesh
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    temp_ess_attr.SetSize(pmesh_->bdr_attributes.Max());
    temp_ess_attr = 0;

    Qt_ess_attr.SetSize(pmesh_->bdr_attributes.Max());
    Qt_ess_attr = 0;
  }
  if (verbose) grvy_printf(ginfo, "ThermoChem paces constructed...\n");

  int vfes_truevsize = vfes->GetTrueVSize();
  int sfes_truevsize = sfes->GetTrueVSize();

  gradT.SetSize(vfes_truevsize);
  gradT = 0.0;
  gradT_gf.SetSpace(vfes);

  Qt.SetSize(sfes_truevsize);
  Qt = 0.0;
  Qt_gf.SetSpace(sfes);

  Tn.SetSize(sfes_truevsize);
  Tn_next.SetSize(sfes_truevsize);

  Tnm1.SetSize(sfes_truevsize);
  Tnm2.SetSize(sfes_truevsize);

  // forcing term
  fTn.SetSize(sfes_truevsize);

  // temp convection terms
  NTn.SetSize(sfes_truevsize);
  NTnm1.SetSize(sfes_truevsize);
  NTnm2.SetSize(sfes_truevsize);
  NTn = 0.0;
  NTnm1 = 0.0;
  NTnm2 = 0.0;

  Text.SetSize(sfes_truevsize);
  Text_bdr.SetSize(sfes_truevsize);
  Text_gf.SetSpace(sfes);
  t_bdr.SetSize(sfes_truevsize);

  resT_gf.SetSpace(sfes);
  resT.SetSize(sfes_truevsize);
  resT = 0.0;

  Tn_next_gf.SetSpace(sfes);
  Tn_gf.SetSpace(sfes);
  Tnm1_gf.SetSpace(sfes);
  Tnm2_gf.SetSpace(sfes);

  rn.SetSize(sfes_truevsize);
  rn = 1.0;
  rn_gf.SetSpace(sfes);
  rn_gf = 1.0;

  visc.SetSize(sfes_truevsize);
  visc = 1.0e-12;
  viscMult.SetSize(sfes_truevsize);
  viscMult = 1.0;

  kappa.SetSize(sfes_truevsize);
  kappa = 1.0e-12;

  kappa_gf.SetSpace(sfes);
  kappa_gf = 0.0;

  subgridVisc.SetSize(sfes_truevsize);
  subgridVisc = 1.0e-15;

  eddyVisc_gf.SetSize(sfes_truevsize);
  eddyVisc_gf = 1.0e-15;

  viscTotal_gf.SetSpace(sfes);
  viscTotal_gf = 0.0;

  visc_gf.SetSpace(sfes);
  visc_gf = 0.0;

  tmpR0.SetSize(sfes_truevsize);
  tmpR0a.SetSize(sfes_truevsize);
  tmpR0b.SetSize(sfes_truevsize);
  tmpR0c.SetSize(sfes_truevsize);
  tmpR1.SetSize(vfes_truevsize);
  R0PM0_gf.SetSpace(sfes);
  R0PM1_gf.SetSpace(pfes);

  rhoDt.SetSpace(sfes);
  viscMult_gf.SetSpace(sfes);

  if (verbose) grvy_printf(ginfo, "ThermoChem vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf;
  toFlow_interface_.viscosity = &visc_gf;
  toFlow_interface_.thermal_divergence = &Qt_gf;
  toTurbModel_interface_.density = &rn_gf;
}

void ThermoChem::setup(double dt) {
  // if (verbose) grvy_printf(ginfo, "in Setup...\n");

  Sdof = sfes->GetNDofs();
  Pdof = pfes->GetNDofs();

  SdofInt = sfes->GetTrueVSize();
  PdofInt = pfes->GetTrueVSize();

  // Initial conditions
  if (config_->useICFunction == true) {
    FunctionCoefficient t_ic_coef(temp_ic);
    if (!config_->restart) Tn_gf.ProjectCoefficient(t_ic_coef);
    if (rank0) {
      std::cout << "Using initial condition function" << endl;
    }
  } else if (config_->useICBoxFunction == true) {
    FunctionCoefficient t_ic_coef(temp_wallBox);
    if (!config_->restart) Tn_gf.ProjectCoefficient(t_ic_coef);
    if (rank0) {
      std::cout << "Using initial condition box function" << endl;
    }
  } else {
    ConstantCoefficient t_ic_coef;
    // t_ic_coef.constant = config.initRhoRhoVp[4] / (Rgas * config.initRhoRhoVp[0]);
    t_ic_coef.constant = config_->initRhoRhoVp[4];
    if (!config_->restart) Tn_gf.ProjectCoefficient(t_ic_coef);
    if (rank0) {
      std::cout << "Initial temperature set from input file: " << config_->initRhoRhoVp[4] << endl;
    }
  }
  Tn_gf.GetTrueDofs(Tn);
  Tnm1_gf.SetFromTrueDofs(Tn);
  Tnm2_gf.SetFromTrueDofs(Tn);
  Tnm1_gf.GetTrueDofs(Tnm1);
  Tnm2_gf.GetTrueDofs(Tnm2);

  // Boundary conditions
  ConstantCoefficient Qt_bc_coef;
  Array<int> Tattr(pmesh_->bdr_attributes.Max());
  Array<int> Qattr(pmesh_->bdr_attributes.Max());

  // inlet bc
  Tattr = 0.0;
  for (int i = 0; i < numInlets; i++) {
    for (int iFace = 1; iFace < pmesh_->bdr_attributes.Max() + 1; iFace++) {
      if (iFace == config_->inletPatchType[i].first) {
        Tattr[iFace - 1] = 1;
      }
    }
  }

  for (int i = 0; i < numInlets; i++) {
    if (config_->inletPatchType[i].second == InletType::UNI_DENS_VEL) {
      if (rank0) std::cout << "Caught uniform inlet conditions " << endl;
      buffer_tInletInf = new ParGridFunction(sfes);
      buffer_tInlet = new ParGridFunction(sfes);
      tInletField = new GridFunctionCoefficient(buffer_tInlet);
      AddTempDirichletBC(tInletField, Tattr);

    } else if (config_->inletPatchType[i].second == InletType::INTERPOLATE) {
      if (rank0) std::cout << "Caught interpolated inlet conditions " << endl;
      // interpolate from external file
      buffer_tInletInf = new ParGridFunction(sfes);
      buffer_tInlet = new ParGridFunction(sfes);
      interpolateInlet();
      if (rank0) std::cout << "ThermoChem interpolation complete... " << endl;

      tInletField = new GridFunctionCoefficient(buffer_tInlet);
      AddTempDirichletBC(tInletField, Tattr);
      if (rank0) std::cout << "Interpolated temperature conditions applied " << endl;
    }
  }
  if (rank0) std::cout << "ThermoChem inlet bc's completed: " << numInlets << endl;

  // outlet bc
  Tattr = 0.0;

  // Wall bc
  Vector wallBCs;
  wallBCs.SetSize(numWalls);
  for (int i = 0; i < numWalls; i++) {
    wallBCs[i] = config_->wallBC[i].Th;
  }

  // for multiple walls at different temps => this is absolutely terrible
  // ConstantCoefficient t_bc_coef(wallBCs);
  if (numWalls > 0) {
    t_bc_coef0.constant = wallBCs[0];
  }
  if (numWalls > 1) {
    t_bc_coef1.constant = wallBCs[1];
  }
  if (numWalls > 2) {
    t_bc_coef2.constant = wallBCs[2];
  }
  if (numWalls > 3) {
    t_bc_coef3.constant = wallBCs[3];
  }

  for (int i = 0; i < numWalls; i++) {
    Tattr = 0;
    // t_bc_coef.constant = config.wallBC[i].Th;
    for (int iFace = 1; iFace < pmesh_->bdr_attributes.Max() + 1; iFace++) {
      if (config_->wallPatchType[i].second == WallType::VISC_ISOTH) {
        if (iFace == config_->wallPatchType[i].first) {
          Tattr[iFace - 1] = 1;
          if (rank0) std::cout << "Dirichlet wall temperature BC: " << iFace << endl;

          if (config_->useWallFunction == true) {
            AddTempDirichletBC(temp_wall, Tattr);
            if (rank0) std::cout << "Using wall temp function bc for wall number: " << i + 1 << endl;
          } else if (config_->useWallBox == true) {
            AddTempDirichletBC(temp_wallBox, Tattr);
            if (rank0) std::cout << "Using box temp bc for wall number: " << i + 1 << endl;
          } else {
            if (i == 0) {
              AddTempDirichletBC(&t_bc_coef0, Tattr);
            }
            if (i == 1) {
              AddTempDirichletBC(&t_bc_coef1, Tattr);
            }
            if (i == 2) {
              AddTempDirichletBC(&t_bc_coef2, Tattr);
            }
            if (i == 3) {
              AddTempDirichletBC(&t_bc_coef3, Tattr);
            }

            if (rank0) std::cout << i + 1 << ") wall temperature BC: " << config_->wallBC[i].Th << endl;
          }
        }
      } else if (config_->wallPatchType[i].second == WallType::VISC_ADIAB) {
        if (iFace == config_->wallPatchType[i].first) {
          Tattr[iFace - 1] = 0;  // no essential
          if (rank0) std::cout << "Insulated wall temperature BC: " << iFace << endl;
        }
      }
    }
  }
  if (rank0) std::cout << "Temp wall bc completed: " << numWalls << endl;

  // Qt wall bc dirichlet (divU = 0)
  Qattr = 0;
  for (int i = 0; i < numWalls; i++) {
    Qt_bc_coef.constant = 0.0;
    for (int iFace = 1; iFace < pmesh_->bdr_attributes.Max() + 1; iFace++) {
      if (config_->wallPatchType[i].second == WallType::VISC_ISOTH ||
          config_->wallPatchType[i].second == WallType::VISC_ADIAB) {
        if (iFace == config_->wallPatchType[i].first) {
          Qattr[iFace - 1] = 1;
          if (rank0) std::cout << "Dirichlet wall Qt BC: " << iFace << endl;
        }
      }
    }
  }
  buffer_qbc = new ConstantCoefficient(Qt_bc_coef);
  AddQtDirichletBC(buffer_qbc, Qattr);

  sfes->GetEssentialTrueDofs(temp_ess_attr, temp_ess_tdof);
  sfes->GetEssentialTrueDofs(Qt_ess_attr, Qt_ess_tdof);
  if (rank0) std::cout << "ThermoChem Essential true dof step" << endl;

  Array<int> empty;

  // GLL integration rule (Numerical Integration)
  // unsteady: p+p [+p] = 2p [3p]
  // convection: p+p+(p-1) [+p] = 3p-1 [4p-1]
  // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
  const IntegrationRule &ir_i = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2 * order + 1);
  const IntegrationRule &ir_nli = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 4 * order);
  const IntegrationRule &ir_di = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 3 * order - 1);
  if (rank0) std::cout << "Integration rules set" << endl;

  // coefficients for operators
  Rho = new GridFunctionCoefficient(&rn_gf);
  viscField = new GridFunctionCoefficient(&visc_gf);
  kappaField = new GridFunctionCoefficient(&kappa_gf);
  {
    double *data = rhoDt.HostReadWrite();
    double *Rdata = rn_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = Rdata[i] / dt;
    }
  }
  rhoDtField = new GridFunctionCoefficient(&rhoDt);

  // thermal_diff_coeff.constant = thermal_diff;
  thermal_diff_coeff = new GridFunctionCoefficient(&kappa_gf);
  gradT_coeff = new GradientGridFunctionCoefficient(&Tn_next_gf);
  kap_gradT_coeff = new ScalarVectorProductCoefficient(*thermal_diff_coeff, *gradT_coeff);

  // gradient of scalar
  G_form = new ParMixedBilinearForm(sfes, vfes);
  auto *g_mblfi = new GradientIntegrator();
  if (numerical_integ) {
    g_mblfi->SetIntRule(&ir_i);
  }
  G_form->AddDomainIntegrator(g_mblfi);
  if (partial_assembly) {
    G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  G_form->Assemble();
  G_form->FormRectangularSystemMatrix(empty, empty, G);
  if (rank0) std::cout << "ThermoChem gradient operator set" << endl;

  // mass matrix for vector
  Mv_form = new ParBilinearForm(vfes);
  auto *mv_blfi = new VectorMassIntegrator;
  if (numerical_integ) {
    mv_blfi->SetIntRule(&ir_i);
  }
  Mv_form->AddDomainIntegrator(mv_blfi);
  if (partial_assembly) {
    Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mv_form->Assemble();
  Mv_form->FormSystemMatrix(empty, Mv);

  // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho u \cdot \nabla \phi_j
  un_next_coeff = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhon_next_coeff = new GridFunctionCoefficient(&rn_gf);
  rhou_coeff = new ScalarVectorProductCoefficient(*rhon_next_coeff, *un_next_coeff);

  At_form = new ParBilinearForm(sfes);
  auto *at_blfi = new ConvectionIntegrator(*rhou_coeff);
  if (numerical_integ) {
    at_blfi->SetIntRule(&ir_nli);
  }
  At_form->AddDomainIntegrator(at_blfi);
  if (partial_assembly) {
    At_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  At_form->Assemble();
  At_form->FormSystemMatrix(empty, At);
  if (rank0) std::cout << "ThermoChem At operator set" << endl;

  // mass matrix
  Ms_form = new ParBilinearForm(sfes);
  auto *ms_blfi = new MassIntegrator;
  if (numerical_integ) {
    ms_blfi->SetIntRule(&ir_i);
  }
  Ms_form->AddDomainIntegrator(ms_blfi);
  if (partial_assembly) {
    Ms_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ms_form->Assemble();
  Ms_form->FormSystemMatrix(empty, Ms);

  // mass matrix with rho
  MsRho_form = new ParBilinearForm(sfes);
  auto *msrho_blfi = new MassIntegrator(*Rho);
  if (numerical_integ) {
    msrho_blfi->SetIntRule(&ir_i);
  }
  MsRho_form->AddDomainIntegrator(msrho_blfi);
  if (partial_assembly) {
    MsRho_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MsRho_form->Assemble();
  MsRho_form->FormSystemMatrix(empty, MsRho);
  if (rank0) std::cout << "ThermoChem MsRho operator set" << endl;

  // temperature laplacian
  Lt_form = new ParBilinearForm(sfes);
  Lt_coeff.constant = -1.0;
  auto *lt_blfi = new DiffusionIntegrator(Lt_coeff);
  if (numerical_integ) {
    lt_blfi->SetIntRule(&ir_di);
  }
  Lt_form->AddDomainIntegrator(lt_blfi);
  if (partial_assembly) {
    Lt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Lt_form->Assemble();
  Lt_form->FormSystemMatrix(empty, Lt);
  if (rank0) std::cout << "ThermoChem Lt operator set" << endl;

  // Ht_lincoeff.constant = kin_vis / Pr;
  Ht_bdfcoeff.constant = 1.0 / dt;
  Ht_form = new ParBilinearForm(sfes);
  hmt_blfi = new MassIntegrator(*rhoDtField);
  hdt_blfi = new DiffusionIntegrator(*kappaField);

  if (numerical_integ) {
    hmt_blfi->SetIntRule(&ir_di);
    hdt_blfi->SetIntRule(&ir_di);
  }
  Ht_form->AddDomainIntegrator(hmt_blfi);
  Ht_form->AddDomainIntegrator(hdt_blfi);
  if (partial_assembly) {
    Ht_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ht_form->Assemble();
  Ht_form->FormSystemMatrix(temp_ess_tdof, Ht);
  if (rank0) std::cout << "ThermoChem Ht operator set" << endl;

  // temp boundary terms
  Text_gfcoeff = new GridFunctionCoefficient(&Text_gf);
  Text_bdr_form = new ParLinearForm(sfes);
  auto *text_blfi = new BoundaryLFIntegrator(*Text_gfcoeff);
  if (numerical_integ) {
    text_blfi->SetIntRule(&ir_i);
  }
  Text_bdr_form->AddBoundaryIntegrator(text_blfi, temp_ess_attr);

  t_bdr_form = new ParLinearForm(sfes);
  for (auto &temp_dbc : temp_dbcs) {
    auto *tbdr_blfi = new BoundaryLFIntegrator(*temp_dbc.coeff);
    if (numerical_integ) {
      tbdr_blfi->SetIntRule(&ir_i);
    }
    t_bdr_form->AddBoundaryIntegrator(tbdr_blfi, temp_dbc.attr);
  }

  if (partial_assembly) {
    Vector diag_pa(vfes->GetTrueVSize());
    Mv_form->AssembleDiagonal(diag_pa);
    MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MvInv = new CGSolver(vfes->GetComm());
  MvInv->iterative_mode = false;
  MvInv->SetOperator(*Mv);
  MvInv->SetPreconditioner(*MvInvPC);
  MvInv->SetPrintLevel(pl_mvsolve);
  MvInv->SetRelTol(config_->solver_tol);
  MvInv->SetMaxIter(config_->solver_iter);
  if (rank0) std::cout << "ThermoChem MvInv operator set" << endl;

  if (partial_assembly) {
    Vector diag_pa(sfes->GetTrueVSize());
    Ms_form->AssembleDiagonal(diag_pa);
    MsInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsInvPC = new HypreSmoother(*Ms.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MsInv = new CGSolver(sfes->GetComm());
  MsInv->iterative_mode = false;
  MsInv->SetOperator(*Ms);
  MsInv->SetPreconditioner(*MsInvPC);
  MsInv->SetPrintLevel(pl_mtsolve);
  MsInv->SetRelTol(config_->solver_tol);
  MsInv->SetMaxIter(config_->solver_iter);

  if (partial_assembly) {
    Vector diag_pa(sfes->GetTrueVSize());
    Ht_form->AssembleDiagonal(diag_pa);
    HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
  } else {
    HtInvPC = new HypreSmoother(*Ht.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HtInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  HtInv = new CGSolver(sfes->GetComm());

  // make these options instead?
  // HtInv = new GMRESSolver(sfes->GetComm());
  // HtInv = new BiCGSTABSolver(sfes->GetComm());

  HtInv->iterative_mode = true;
  HtInv->SetOperator(*Ht);
  HtInv->SetPreconditioner(*HtInvPC);
  HtInv->SetPrintLevel(pl_htsolve);
  HtInv->SetRelTol(config_->solver_tol);
  HtInv->SetMaxIter(config_->solver_iter);
  if (rank0) std::cout << "Temperature operators set" << endl;

  // Qt .....................................
  Mq_form = new ParBilinearForm(sfes);
  auto *mq_blfi = new MassIntegrator;
  if (numerical_integ) {
    mq_blfi->SetIntRule(&ir_i);
  }
  Mq_form->AddDomainIntegrator(mq_blfi);
  if (partial_assembly) {
    Mq_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mq_form->Assemble();
  Mq_form->FormSystemMatrix(Qt_ess_tdof, Mq);
  // Mq_form->FormSystemMatrix(empty, Mq);
  if (rank0) std::cout << "ThermoChem Mq operator set" << endl;

  if (partial_assembly) {
    Vector diag_pa(sfes->GetTrueVSize());
    Mq_form->AssembleDiagonal(diag_pa);
    MqInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MqInvPC = new HypreSmoother(*Mq.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MqInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MqInv = new CGSolver(sfes->GetComm());
  MqInv->iterative_mode = false;
  MqInv->SetOperator(*Mq);
  MqInv->SetPreconditioner(*MqInvPC);
  MqInv->SetPrintLevel(pl_mtsolve);
  MqInv->SetRelTol(config_->solver_tol);
  MqInv->SetMaxIter(config_->solver_iter);

  LQ_form = new ParBilinearForm(sfes);
  auto *lqd_blfi = new DiffusionIntegrator(*kappaField);
  if (numerical_integ) {
    lqd_blfi->SetIntRule(&ir_di);
  }
  LQ_form->AddDomainIntegrator(lqd_blfi);
  if (partial_assembly) {
    LQ_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  LQ_form->Assemble();
  LQ_form->FormSystemMatrix(empty, LQ);

  LQ_bdry = new ParLinearForm(sfes);
  auto *lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff, 2, -1);
  if (numerical_integ) {
    lq_bdry_lfi->SetIntRule(&ir_di);
  }
  LQ_bdry->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr);
  if (rank0) std::cout << "ThermoChem LQ operator set" << endl;

  // remaining initialization
  if (config_->resetTemp == true) {
    double *data = Tn_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = config_->initRhoRhoVp[4];
    }
    if (rank0) std::cout << "Reset temperature to IC" << endl;
  }

  // copy IC to other temp containers
  Tn_gf.GetTrueDofs(Tn);
  Tn_next_gf.SetFromTrueDofs(Tn);
  Tn_next_gf.GetTrueDofs(Tn_next);

  // Temp filter
  filter_alpha = loMach_opts_->filterWeight;
  filter_cutoff_modes = loMach_opts_->nFilter;

  if (loMach_opts_->filterTemp == true) {
    sfec_filter = new H1_FECollection(order - filter_cutoff_modes);
    sfes_filter = new ParFiniteElementSpace(pmesh_, sfec_filter);

    Tn_NM1_gf.SetSpace(sfes_filter);
    Tn_NM1_gf = 0.0;

    Tn_filtered_gf.SetSpace(sfes);
    Tn_filtered_gf = 0.0;
  }

  // viscous sponge
  ParGridFunction coordsDof(vfes);
  pmesh_->GetNodes(coordsDof);
  if (config_->linViscData.isEnabled) {
    if (rank0) std::cout << "Viscous sponge active" << endl;
    double *vMult = viscMult.HostReadWrite();
    double *hcoords = coordsDof.HostReadWrite();
    double wgt = 0.;
    for (int n = 0; n < sfes->GetNDofs(); n++) {
      double coords[3];
      for (int d = 0; d < dim; d++) {
        coords[d] = hcoords[n + d * sfes->GetNDofs()];
      }
      viscSpongePlanar(coords, wgt);
      vMult[n] = wgt + (config_->linViscData.uniformMult - 1.0);
    }
  }
}

/**
   Rotate temperature state in registers
 */
void ThermoChem::UpdateTimestepHistory(double dt) {
  // temperature
  NTnm2 = NTnm1;
  NTnm1 = NTn;
  Tnm2 = Tnm1;
  Tnm1 = Tn;
  Tn_next_gf.GetTrueDofs(Tn_next);
  Tn = Tn_next;
  Tn_gf.SetFromTrueDofs(Tn);
}

void ThermoChem::step() {
  // update bdf coefficients
  bd0 = timeCoeff_.bd0;
  bd1 = timeCoeff_.bd1;
  bd2 = timeCoeff_.bd2;
  bd3 = timeCoeff_.bd3;
  dt = timeCoeff_.dt;
  time = timeCoeff_.time;

  // Set current time for velocity Dirichlet boundary conditions.
  for (auto &temp_dbc : temp_dbcs) {
    temp_dbc.coeff->SetTime(time + dt);
  }

  if (loMach_opts_->solveTemp == true) {
    resT = 0.0;

    // convection
    computeExplicitTempConvectionOP(true);  // ->tmpR0
    resT.Set(-1.0, tmpR0);

    // for unsteady term, compute and add known part of BDF unsteady term
    tmpR0.Set(bd1 / dt, Tn);
    tmpR0.Add(bd2 / dt, Tnm1);
    tmpR0.Add(bd3 / dt, Tnm2);

    // multScalarScalarIP(rn,&tmpR0);
    MsRho->Mult(tmpR0, tmpR0b);
    resT.Add(-1.0, tmpR0b);
    // multConstScalarIP(Cp,&resT);

    // dPo/dt
    tmpR0 = (dtP / Cp);
    Ms->Mult(tmpR0, tmpR0b);
    resT.Add(1.0, tmpR0b);

    // Add natural boundary terms here later

    // update Helmholtz operator, unsteady here, kappa in updateDiffusivity
    {
      double *data = rhoDt.HostReadWrite();
      double *Rdata = rn_gf.HostReadWrite();
      // double coeff = Cp*bd0/dt;
      double coeff = bd0 / dt;
      for (int i = 0; i < Sdof; i++) {
        data[i] = coeff * Rdata[i];
      }
    }
    Ht_form->Update();
    Ht_form->Assemble();
    Ht_form->FormSystemMatrix(temp_ess_tdof, Ht);

    HtInv->SetOperator(*Ht);
    if (partial_assembly) {
      delete HtInvPC;
      Vector diag_pa(sfes->GetTrueVSize());
      Ht_form->AssembleDiagonal(diag_pa);
      HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
      HtInv->SetPreconditioner(*HtInvPC);
    }

    for (auto &temp_dbc : temp_dbcs) {
      Tn_next_gf.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr);
    }
    sfes->GetRestrictionMatrix()->MultTranspose(resT, resT_gf);

    Vector Xt2, Bt2;
    if (partial_assembly) {
      auto *HC = Ht.As<ConstrainedOperator>();
      EliminateRHS(*Ht_form, *HC, temp_ess_tdof, Tn_next_gf, resT_gf, Xt2, Bt2, 1);
    } else {
      Ht_form->FormLinearSystem(temp_ess_tdof, Tn_next_gf, resT_gf, Ht, Xt2, Bt2, 1);
    }

    // solve helmholtz eq for temp
    HtInv->Mult(Bt2, Xt2);
    iter_htsolve = HtInv->GetNumIterations();
    res_htsolve = HtInv->GetFinalNorm();
    Ht_form->RecoverFEMSolution(Xt2, resT_gf, Tn_next_gf);
    Tn_next_gf.GetTrueDofs(Tn_next);

    // explicit filter
    if (loMach_opts_->filterTemp == true) {
      const auto filter_alpha_ = filter_alpha;
      Tn_NM1_gf.ProjectGridFunction(Tn_next_gf);
      Tn_filtered_gf.ProjectGridFunction(Tn_NM1_gf);
      const auto d_Tn_filtered_gf = Tn_filtered_gf.Read();
      auto d_Tn_gf = Tn_next_gf.ReadWrite();
      MFEM_FORALL(i, Tn_next_gf.Size(),
                  { d_Tn_gf[i] = (1.0 - filter_alpha_) * d_Tn_gf[i] + filter_alpha_ * d_Tn_filtered_gf[i]; });
      Tn_next_gf.GetTrueDofs(Tn_next);
    }
  }

  // prepare for external use
  updateDensity(1.0);
  // computeQt();
  computeQtTO();

  UpdateTimestepHistory(dt);
}

void ThermoChem::computeExplicitTempConvectionOP(bool extrap) {
  Array<int> empty;
  At_form->Update();
  At_form->Assemble();
  At_form->FormSystemMatrix(empty, At);
  if (extrap == true) {
    At->Mult(Tn, NTn);
  } else {
    At->Mult(Text, NTn);
  }

  // ab predictor
  if (extrap == true) {
    tmpR0.Set(ab1, NTn);
    tmpR0.Add(ab2, NTnm1);
    tmpR0.Add(ab3, NTnm2);
  } else {
    tmpR0.Set(1.0, NTn);
  }
}

/**
   Update boundary conditions for Temperature, useful for
   ramping a dirichlet condition over time
 */
void ThermoChem::updateBC(int current_step) {
  if (numInlets > 0) {
    double *dTInf = buffer_tInletInf->HostReadWrite();
    double *dT = buffer_tInlet->HostReadWrite();
    int nRamp = config_->rampStepsInlet;
    double wt = std::pow(std::min((double)current_step / (double)nRamp, 1.0), 1.0);
    for (int i = 0; i < Sdof; i++) {
      dT[i] = wt * dTInf[i] + (1.0 - wt) * config_->initRhoRhoVp[nvel + 1];
    }
  }
}

void ThermoChem::extrapolateState(int current_step, std::vector<double> ab) {
  // update ab coefficients
  ab1 = timeCoeff_.ab1;
  ab2 = timeCoeff_.ab2;
  ab3 = timeCoeff_.ab3;

  // extrapolated temp at {n+1}
  Text.Set(ab1, Tn);
  Text.Add(ab2, Tnm1);
  Text.Add(ab3, Tnm2);

  // store ext in _next containers, remove Text, Uext completely later
  Tn_next_gf.SetFromTrueDofs(Text);
  Tn_next_gf.GetTrueDofs(Tn_next);
}

// update thermodynamic pressure
void ThermoChem::updateThermoP(double dt) {
  if (config_->isOpen != true) {
    double allMass, PNM1;
    double myMass = 0.0;
    double *Tdata = Tn.HostReadWrite();
    double *data = tmpR0.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] = 1.0 / Tdata[i];
    }
    // multConstScalarIP((thermoPressure/Rgas),&tmpR0);
    tmpR0 *= (thermoPressure / Rgas);
    Ms->Mult(tmpR0, tmpR0b);
    for (int i = 0; i < SdofInt; i++) {
      myMass += tmpR0b[i];
    }
    MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PNM1 = thermoPressure;
    thermoPressure = systemMass / allMass * PNM1;
    dtP = (thermoPressure - PNM1) / dt;
    // if( rank0 == true ) std::cout << " Original closed system mass: " << systemMass << " [kg]" << endl;
    // if( rank0 == true ) std::cout << " New (closed) system mass: " << allMass << " [kg]" << endl;
    // if( rank0 == true ) std::cout << " ThermoP: " << thermoPressure << " dtP: " << dtP << endl;
  }
}

/**
Simple planar viscous sponge layer with smooth tanh-transtion using user-specified width and
total amplification.  Note: duplicate in M2
*/
// MFEM_HOST_DEVICE
void ThermoChem::viscSpongePlanar(double *x, double &wgt) {
  double normal[3];
  double point[3];
  double s[3];
  double factor, width, dist, wgt0;

  for (int d = 0; d < dim; d++) normal[d] = config_->linViscData.normal[d];
  for (int d = 0; d < dim; d++) point[d] = config_->linViscData.point0[d];
  width = config_->linViscData.width;
  factor = config_->linViscData.viscRatio;

  // get settings
  factor = max(factor, 1.0);
  // width = vsd_.width;

  // distance from plane
  dist = 0.;
  for (int d = 0; d < dim; d++) s[d] = (x[d] - point[d]);
  for (int d = 0; d < dim; d++) dist += s[d] * normal[d];

  // weight
  wgt0 = 0.5 * (tanh(0.0 / width - 2.0) + 1.0);
  wgt = 0.5 * (tanh(dist / width - 2.0) + 1.0);
  wgt = (wgt - wgt0) * 1.0 / (1.0 - wgt0);
  wgt = std::max(wgt, 0.0);
  wgt *= (factor - 1.0);
  wgt += 1.0;

  // add cylindrical area
  double cylX = config_->linViscData.cylXradius;
  double cylY = config_->linViscData.cylYradius;
  double cylZ = config_->linViscData.cylZradius;
  double wgtCyl;
  if (config_->linViscData.cylXradius > 0.0) {
    dist = x[1] * x[1] + x[2] * x[2];
    dist = std::sqrt(dist);
    dist = dist - cylX;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);
  } else if (config_->linViscData.cylYradius > 0.0) {
    dist = x[0] * x[0] + x[2] * x[2];
    dist = std::sqrt(dist);
    dist = dist - cylY;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);
  } else if (config_->linViscData.cylZradius > 0.0) {
    dist = x[0] * x[0] + x[1] * x[1];
    dist = std::sqrt(dist);
    dist = dist - cylZ;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);
  }

  // add annulus sponge => NOT GENERAL, only for annulus aligned with y
  /**/
  double centerAnnulus[3];
  // for (int d = 0; d < dim; d++) normalAnnulus[d] = config.linViscData.normalA[d];
  for (int d = 0; d < dim; d++) centerAnnulus[d] = config_->linViscData.pointA[d];
  double rad1 = config_->linViscData.annulusRadius;
  double rad2 = config_->linViscData.annulusThickness;
  double factorA = config_->linViscData.viscRatioAnnulus;

  // distance to center
  double dist1 = x[0] * x[0] + x[2] * x[2];
  dist1 = std::sqrt(dist1);

  // coordinates of nearest point on annulus center ring
  for (int d = 0; d < dim; d++) s[d] = (rad1 / dist1) * x[d];
  s[1] = centerAnnulus[1];

  // distance to ring
  for (int d = 0; d < dim; d++) s[d] = x[d] - s[d];
  double dist2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
  dist2 = std::sqrt(dist2);

  // if (x[1] <= centerAnnulus[1]+rad2) {
  //   std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << endl;
  // }

  // if (dist2 <= rad2) {
  //   std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << " fA: " << factorA << endl;
  // }

  // sponge weight
  double wgtAnn;
  wgtAnn = 0.5 * (tanh(10.0 * (1.0 - dist2 / rad2)) + 1.0);
  wgtAnn = (wgtAnn - wgt0) * 1.0 / (1.0 - wgt0);
  wgtAnn = std::max(wgtAnn, 0.0);
  // if (dist2 <= rad2) wgtAnn = 1.0; // testing....
  wgtAnn *= (factorA - 1.0);
  wgtAnn += 1.0;
  wgt = std::max(wgt, wgtAnn);
}

void ThermoChem::updateDiffusivity() {
  // viscosity
  if (constantViscosity != true) {
    double *dataVisc = visc_gf.HostReadWrite();
    // double *dataBVisc = bulkVisc_gf->HostReadWrite();
    double *Tdata = Tn_gf.HostReadWrite();
    double *Rdata = rn_gf.HostReadWrite();
    double visc[2];
    // double prim[nvel + 2];
    double prim[5];
    for (int i = 0; i < nvel + 2; i++) {
      prim[i] = 0.0;
    }
    for (int i = 0; i < Sdof; i++) {
      prim[1 + nvel] = Tdata[i];
      transportPtr->GetViscosities(prim, prim, visc);  // returns dynamic
      dataVisc[i] = visc[0];
      // dataBVisc[i] = visc[1];
    }
  } else {
    double *dataVisc = visc_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      dataVisc[i] = dyn_vis;
    }
  }

  if (config_->sgsModelType > 0) {
    const double *dataSubgrid = turbModel_interface_->eddy_viscosity->HostRead();
    double *dataVisc = visc_gf.HostReadWrite();
    double *data = viscTotal_gf.HostReadWrite();
    double *Rdata = rn_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = dataVisc[i] + Rdata[i] * dataSubgrid[i];
    }
  }

  if (config_->linViscData.isEnabled) {
    double *vMult = viscMult.HostReadWrite();
    double *dataVisc = viscTotal_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      dataVisc[i] *= viscMult[i];
    }
  }

  // thermal diffusivity: storing as kappa eventhough does NOT
  // include Cp for now
  {
    double *data = kappa_gf.HostReadWrite();
    double *dataVisc = viscTotal_gf.HostReadWrite();
    // double Ctmp = Cp / Pr;
    double Ctmp = 1.0 / Pr;
    for (int i = 0; i < Sdof; i++) {
      data[i] = Ctmp * dataVisc[i];
    }
  }

  viscTotal_gf.SetFromTrueDofs(visc);
  kappa_gf.SetFromTrueDofs(kappa);
}

void ThermoChem::updateDensity(double tStep) {
  Array<int> empty;

  // set rn
  if (constantDensity != true) {
    if (tStep == 1.0) {
      multConstScalarInv((thermoPressure / Rgas), Tn_next, &rn);
    } else if (tStep == 0.5) {
      multConstScalarInv((thermoPressure / Rgas), Tn_next, &tmpR0a);
      multConstScalarInv((thermoPressure / Rgas), Tn, &tmpR0b);
      {
        double *data = rn.HostWrite();
        double *data1 = tmpR0a.HostWrite();
        double *data2 = tmpR0b.HostWrite();
        for (int i = 0; i < SdofInt; i++) {
          data[i] = 0.5 * (data1[i] + data2[i]);
        }
      }
    } else {
      multConstScalarInv((thermoPressure / Rgas), Tn, &rn);
    }

  } else {
    double *data = rn.HostWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] = static_rho;
    }
  }
  rn_gf.SetFromTrueDofs(rn);

  MsRho_form->Update();
  MsRho_form->Assemble();
  MsRho_form->FormSystemMatrix(empty, MsRho);

  // project to p-space in case not same as vel-temp
  R0PM0_gf.SetFromTrueDofs(rn);
  R0PM1_gf.ProjectGridFunction(R0PM0_gf);
  {
    double *data = bufferInvRho->HostReadWrite();
    double *rho = R0PM1_gf.HostReadWrite();
    for (int i = 0; i < Pdof; i++) {
      data[i] = 1.0 / rho[i];
    }
  }
}

void ThermoChem::updateGradientsOP(double tStep) {
  // gradient of temperature
  if (tStep == 0.0) {
    G->Mult(Tn, tmpR1);
  } else {
    G->Mult(Tn_next, tmpR1);
  }
  MvInv->Mult(tmpR1, gradT);
}

void ThermoChem::computeSystemMass() {
  systemMass = 0.0;
  double myMass = 0.0;
  double *Tdata = Tn.HostReadWrite();
  double *Rdata = rn.HostReadWrite();
  for (int i = 0; i < SdofInt; i++) {
    Rdata[i] = thermoPressure / (Rgas * Tdata[i]);
  }
  Ms->Mult(rn, tmpR0);
  for (int i = 0; i < SdofInt; i++) {
    myMass += tmpR0[i];
  }
  MPI_Allreduce(&myMass, &systemMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank0 == true) std::cout << " Closed system mass: " << systemMass << " [kg]" << endl;
}

/**
   Interpolation of inlet bc from external file using
   Gaussian interpolation.  Not at all general and only for use
   with torch
 */
void ThermoChem::interpolateInlet() {
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  int Sdof = sfes->GetNDofs();

  string fname;
  fname = "./inputs/inletPlane.csv";
  const int fname_length = fname.length();
  char *char_array = new char[fname_length + 1];
  // strcpy(char_array, fname.c_str());
  snprintf(char_array, fname.c_str());
  int nCount = 0;

  // open, find size
  if (rank0) {
    std::cout << " Attempting to open inlet file for counting... " << fname << endl;
    fflush(stdout);

    FILE *inlet_file;
    if (inlet_file = fopen("./inputs/inletPlane.csv", "r")) {
      std::cout << " ...and open" << endl;
      fflush(stdout);
    } else {
      std::cout << " ...CANNOT OPEN FILE" << endl;
      fflush(stdout);
    }

    char *line = NULL;
    int ch = 0;
    size_t len = 0;
    ssize_t read;

    std::cout << " ...starting count" << endl;
    fflush(stdout);
    for (ch = getc(inlet_file); ch != EOF; ch = getc(inlet_file)) {
      if (ch == '\n') {
        nCount++;
      }
    }
    fclose(inlet_file);
    std::cout << " final count: " << nCount << endl;
    fflush(stdout);
  }

  // broadcast size
  MPI_Bcast(&nCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // set size
  struct inlet_profile {
    double x, y, z, rho, temp, u, v, w;
  };
  struct inlet_profile inlet[nCount];
  double junk;

  // paraview slice output from mean
  // 0)"dens",1)"rms:0",2)"rms:1",3)"rms:2",4)"rms:3",5)"rms:4",6)"rms:5",7)"temp",8)"vel:0",9)"vel:1",10)"vel:2",11)"Points:0",12)"Points:1",13)"Points:2"

  // mean output from plane interpolation
  // 0) no, 1) x, 2) y, 3) z, 4) rho, 5) u, 6) v, 7) w

  // open, read data
  if (rank0) {
    std::cout << " Attempting to read line-by-line..." << endl;
    fflush(stdout);

    ifstream file;
    file.open("./inputs/inletPlane.csv");
    string line;
    getline(file, line);
    int nLines = 0;
    while (getline(file, line)) {
      stringstream sline(line);
      int entry = 0;
      while (sline.good()) {
        string substr;
        getline(sline, substr, ',');  // csv delimited
        double buffer = std::stod(substr);
        entry++;

        // using paraview dump
        /*
        if (entry == 1) {
          inlet[nLines].rho = buffer;
        } else if (entry == 8) {
          inlet[nLines].temp = buffer;
        } else if (entry == 9) {
          inlet[nLines].u = buffer;
        } else if (entry == 10) {
          inlet[nLines].v = buffer;
        } else if (entry == 11) {
          inlet[nLines].w = buffer;
        } else if (entry == 12) {
          inlet[nLines].x = buffer;
        } else if (entry == 13) {
          inlet[nLines].y = buffer;
        } else if (entry == 14) {
          inlet[nLines].z = buffer;
        }
        */

        // using code plane interp
        /**/
        if (entry == 2) {
          inlet[nLines].x = buffer;
        } else if (entry == 3) {
          inlet[nLines].y = buffer;
        } else if (entry == 4) {
          inlet[nLines].z = buffer;
        } else if (entry == 5) {
          inlet[nLines].rho = buffer;
          // to prevent nans in temp in out-of-domain plane regions
          inlet[nLines].temp = thermoPressure / (Rgas * std::max(buffer, 1.0));
        } else if (entry == 6) {
          inlet[nLines].u = buffer;
        } else if (entry == 7) {
          inlet[nLines].v = buffer;
        } else if (entry == 8) {
          inlet[nLines].w = buffer;
        }
        /**/
      }
      // std::cout << inlet[nLines].rho << " " << inlet[nLines].temp << " " << inlet[nLines].u << " " << inlet[nLines].x
      // << " " << inlet[nLines].y << " " << inlet[nLines].z << endl; fflush(stdout);
      nLines++;
    }
    file.close();

    // std::cout << " stream method ^" << endl; fflush(stdout);
    std::cout << " " << endl;
    fflush(stdout);
  }

  // broadcast data
  MPI_Bcast(&inlet, nCount * 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank0) {
    std::cout << " communicated inlet data" << endl;
    fflush(stdout);
  }

  // width of interpolation stencil
  double radius;
  double torch_radius = 28.062 / 1000.0;
  radius = 2.0 * torch_radius / 300.0;  // 300 from nxn interp plane

  ParGridFunction coordsDof(vfes);
  pmesh_->GetNodes(coordsDof);
  double *dataTemp = buffer_tInlet->HostWrite();
  double *dataTempInf = buffer_tInletInf->HostWrite();
  for (int n = 0; n < sfes->GetNDofs(); n++) {
    // get coords
    auto hcoords = coordsDof.HostRead();
    double xp[3];
    for (int d = 0; d < dim; d++) {
      xp[d] = hcoords[n + d * vfes->GetNDofs()];
    }

    int iCount = 0;
    double dist, wt;
    double wt_tot = 0.0;
    double val_rho = 0.0;
    double val_u = 0.0;
    double val_v = 0.0;
    double val_w = 0.0;
    double val_T = 0.0;
    double dmin = 1.0e15;

    for (int j = 0; j < nCount; j++) {
      dist = (xp[0] - inlet[j].x) * (xp[0] - inlet[j].x) + (xp[1] - inlet[j].y) * (xp[1] - inlet[j].y) +
             (xp[2] - inlet[j].z) * (xp[2] - inlet[j].z);
      dist = sqrt(dist);
      // std::cout << " Gaussian interpolation, point " << n << " with distance " << dist << endl; fflush(stdout);

      // exclude points outside the domain
      if (inlet[j].rho < 1.0e-8) {
        continue;
      }

      // gaussian
      if (dist <= 1.0 * radius) {
        // std::cout << " Caught an interp point " << dist << endl; fflush(stdout);
        wt = exp(-(dist * dist) / (radius * radius));
        wt_tot = wt_tot + wt;
        val_rho = val_rho + wt * inlet[j].rho;
        val_u = val_u + wt * inlet[j].u;
        val_v = val_v + wt * inlet[j].v;
        val_w = val_w + wt * inlet[j].w;
        val_T = val_T + wt * inlet[j].temp;
        iCount++;
        // std::cout << "* " << n << " "  << iCount << " " << wt_tot << " " << val_T << endl; fflush(stdout);
      }
      // nearest, just for testing
      // if(dist <= dmin) {
      //  dmin = dist;
      //  wt_tot = 1.0;
      //   val_rho = inlet[j].rho;
      //   val_u = inlet[j].u;
      //   val_v = inlet[j].v;
      //   val_w = inlet[j].w;
      //   iCount = 1;
      // }
    }

    // if(rank0) { std::cout << " attempting to record interpolated values in buffer" << endl; fflush(stdout); }
    if (wt_tot > 0.0) {
      dataTemp[n] = val_T / wt_tot;
      dataTempInf[n] = val_T / wt_tot;
      // std::cout << n << " point set to: " << dataVel[n + 0*Sdof] << " " << dataTemp[n] << endl; fflush(stdout);
    } else {
      dataTemp[n] = 0.0;
      dataTempInf[n] = 0.0;
      // std::cout << " iCount of zero..." << endl; fflush(stdout);
    }
  }
}

void ThermoChem::uniformInlet() {
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  int Sdof = sfes->GetNDofs();

  ParGridFunction coordsDof(vfes);
  pmesh_->GetNodes(coordsDof);
  double *dataTemp = buffer_tInlet->HostWrite();
  double *dataTempInf = buffer_tInletInf->HostWrite();

  Vector inlet_vec(3);
  double inlet_temp, inlet_pres;
  inlet_temp = config_->amb_pres / (Rgas * config_->densInlet);
  inlet_pres = config_->amb_pres;

  for (int n = 0; n < sfes->GetNDofs(); n++) {
    auto hcoords = coordsDof.HostRead();
    double xp[3];
    for (int d = 0; d < dim; d++) {
      xp[d] = hcoords[n + d * vfes->GetNDofs()];
    }
    dataTemp[n] = inlet_temp;
    dataTempInf[n] = inlet_temp;
  }
}

void ThermoChem::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr) {
  temp_dbcs.emplace_back(attr, coeff);

  if (verbose && pmesh_->GetMyRank() == 0) {
    mfem::out << "Adding Temperature Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((temp_ess_attr[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      temp_ess_attr[i] = 1;
    }
  }
}

void ThermoChem::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void ThermoChem::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr) {
  Qt_dbcs.emplace_back(attr, coeff);

  if (verbose && pmesh_->GetMyRank() == 0) {
    mfem::out << "Adding Qt Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((Qt_ess_attr[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      Qt_ess_attr[i] = 1;
    }
  }
}

void ThermoChem::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddQtDirichletBC(new FunctionCoefficient(f), attr);
}

/*
void ThermoChem::computeQt() {

  if (incompressibleSolve == true) {
    Qt = 0.0;

  } else {

     // laplace(T)
     double TdivFactor = 0.0;
     if (loMach_opts_.thermalDiv == true) {
       Lt->Mult(Tn_next,tmpR0);
       MsInv->Mult(tmpR0, Qt);
     }

     //multScalarScalar(rn,viscSml,&tmpR0);
     multScalarScalarIP(viscSml,&Qt);
     double tmp = Rgas / (Pr * thermoPressure);
     multConstScalarIP(tmp,&Qt);

     // add closed-domain pressure term
     if (config.isOpen != true) {
       double *data = Qt.HostReadWrite();
       for (int i = 0; i < SdofInt; i++) {
         data[i] += ((gamma - 1.0)/gamma - Cp) * 1.0 / (Cp*thermoPressure) * dtP;
       }
     }

   }

   Qt_gf.SetFromTrueDofs(Qt);

}
*/

void ThermoChem::computeQtTO() {
  Array<int> empty;

  if (incompressibleSolve == true) {
    Qt = 0.0;

  } else {
    tmpR0 = 0.0;
    LQ_bdry->Update();
    LQ_bdry->Assemble();
    LQ_bdry->ParallelAssemble(tmpR0);
    tmpR0.Neg();

    LQ_form->Update();
    LQ_form->Assemble();
    LQ_form->FormSystemMatrix(empty, LQ);
    LQ->AddMult(Tn_next, tmpR0);  // tmpR0 += LQ{Tn_next}
    MqInv->Mult(tmpR0, Qt);

    Qt *= -Rgas / thermoPressure;

    /*
    for (int be = 0; be < pmesh->GetNBE(); be++) {
      int bAttr = pmesh.GetBdrElement(be)->GetAttribute();
      if (bAttr == WallType::VISC_ISOTH || bAttr = WallType::VISC_ADIAB) {
        Array<int> vdofs;
        sfes->GetBdrElementVDofs(be, vdofs);
        for (int i = 0; i < vdofs.Size(); i++) {
          Qt[vdofs[i]] = 0.0;
        }
      }
    }
    */
  }

  Qt_gf.SetFromTrueDofs(Qt);
}

///~ BC hard-codes ~///

double temp_ic(const Vector &coords, double t) {
  double Thi = 400.0;
  double Tlo = 200.0;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);
  double temp;
  temp = Tlo + 0.5 * (y + 1.0) * (Thi - Tlo);
  // temp = 0.5 * (Thi + Tlo);

  return temp;
}

double temp_wall(const Vector &coords, double t) {
  double Thi = 400.0;
  double Tlo = 200.0;
  double Tmean, tRamp, wt;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);
  double temp;

  Tmean = 0.5 * (Thi + Tlo);
  tRamp = 100.0;
  wt = min(t / tRamp, 1.0);
  // wt = 0.0; // HACK
  wt = 1.0;  // HACK

  if (y > 0.0) {
    temp = Tmean + wt * (Thi - Tmean);
  }
  if (y < 0.0) {
    temp = Tmean + wt * (Tlo - Tmean);
  }

  return temp;
}

double temp_wallBox(const Vector &coords, double t) {
  double Thi = 480.0;
  double Tlo = 120.0;
  double Tmean, tRamp, wt;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);
  double temp;

  Tmean = 0.5 * (Thi + Tlo);
  // tRamp = 10.0;
  // wt = min(t/tRamp,1.0);
  // wt = 1.0;

  temp = Tmean + x * (Thi - Tlo);

  // hack
  // temp = 310.0;

  // std::cout << " *** temp_wallBox: " << temp << " at " << x << endl;

  return temp;
}

double temp_inlet(const Vector &coords, double t) {
  double Thi = 400.0;
  double Tlo = 200.0;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);
  double temp;
  temp = Tlo + (y + 0.5) * (Thi - Tlo);
  return temp;
}
