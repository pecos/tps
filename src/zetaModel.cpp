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

#include "zetaModel.hpp"

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

ZetaModel::ZetaModel(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff,
                     TPS::Tps *tps, ParGridFunction *gridScale)
    : tpsP_(tps), loMach_opts_(loMach_opts), pmesh_(pmesh), time_coeff_(time_coeff) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  dim_ = pmesh_->Dimension();
  nvel_ = dim_;
  order_ = loMach_opts_->order;
  gridScale_gf_ = gridScale;

  // read-ins, if any
  // example: tpsP_->getInput("loMach/sgsModelConstant", sgs_model_const_, sgs_const);

  tpsP_->getInput("ransModel/tke", tke_ic_, 1.0e-8);
  tpsP_->getInput("ransModel/tdr", tke_ic_, 1.0e-10);

  tpsP_->getInput("ransModel/tke-min", tke_min_, 1.0e-11);
  tpsP_->getInput("ransModel/tdr-min", tdr_min_, 1.0e-14);
  tpsP_->getInput("ransModel/zeta-min", zeta_min_, 1.0e-12);
  tpsP_->getInput("ransModel/f-min", fRate_min_, 1.0e-14);
  tpsP_->getInput("ransModel/tts-min", tts_min_, 1.0e-12);
  tpsP_->getInput("ransModel/tls-min", tls_min_, 1.0e-12);
}

ZetaModel::~ZetaModel() {
  delete sfec_;
  delete sfes_;
  delete vfec_;
  delete vfes_;
}

void ZetaModel::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing zeta-f RANS model solver.\n");

  //-----------------------------------------------------
  // 1) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  // vector
  vfec_ = new H1_FECollection(order_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  // Check if fully periodic mesh
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    tke_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    tke_ess_attr_ = 0;

    tdr_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    tdr_ess_attr_ = 0;

    zeta_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    zeta_ess_attr_ = 0;

    fRate_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    fRate_ess_attr_ = 0;
  }

  int vfes_truevsize = vfes_->GetTrueVSize();
  int sfes_truevsize = sfes_->GetTrueVSize();

  Sdof_ = sfes_->GetNDofs();
  SdofInt_ = sfes_->GetTrueVSize();

  eddyVisc_gf_.SetSpace(sfes_);
  eddyVisc_.SetSize(sfes_truevsize);

  tke_gf_.SetSpace(sfes_);
  tke_.SetSize(sfes_truevsize);

  tdr_gf_.SetSpace(sfes_);
  tdr_.SetSize(sfes_truevsize);

  zeta_gf_.SetSpace(sfes_);
  zeta_.SetSize(sfes_truevsize);

  tke_next_gf_.SetSpace(sfes_);
  tke_next_.SetSize(sfes_truevsize);

  tdr_next_gf_.SetSpace(sfes_);
  tdr_next_.SetSize(sfes_truevsize);

  zeta_next_gf_.SetSpace(sfes_);
  zeta_next_.SetSize(sfes_truevsize);

  fRate_gf_.SetSpace(sfes_);
  fRate_.SetSize(sfes_truevsize);

  tke_nm1_.SetSize(sfes_truevsize);
  tke_nm2_.SetSize(sfes_truevsize);
  Ntke_.SetSize(sfes_truevsize);
  Ntke_nm1_.SetSize(sfes_truevsize);
  Ntke_nm2_.SetSize(sfes_truevsize);

  tdr_nm1_.SetSize(sfes_truevsize);
  tdr_nm2_.SetSize(sfes_truevsize);
  Ntdr_.SetSize(sfes_truevsize);
  Ntdr_nm1_.SetSize(sfes_truevsize);
  Ntdr_nm2_.SetSize(sfes_truevsize);

  zeta_nm1_.SetSize(sfes_truevsize);
  zeta_nm2_.SetSize(sfes_truevsize);
  Nzeta_.SetSize(sfes_truevsize);
  Nzeta_nm1_.SetSize(sfes_truevsize);
  Nzeta_nm2_.SetSize(sfes_truevsize);

  tls_gf_.SetSpace(sfes_);
  tls_.SetSize(sfes_truevsize);
  tls2_gf_.SetSpace(sfes_);
  tls2_.SetSize(sfes_truevsize);

  tts_gf_.SetSpace(sfes_);
  tts_.SetSize(sfes_truevsize);

  prod_gf_.SetSpace(sfes_);
  prod_.SetSize(sfes_truevsize);

  // tdrWall_gf_.SetSpace(sfes_);
  // fWall_gf_.SetSpace(sfes_);

  res_gf_.SetSpace(sfes_);
  res_.SetSize(sfes_truevsize);

  rho_.SetSize(sfes_truevsize);
  mu_.SetSize(sfes_truevsize);
  vel_.SetSize(vfes_truevsize);
  gradU_.SetSize(vfes_truevsize);
  gradV_.SetSize(vfes_truevsize);
  gradW_.SetSize(vfes_truevsize);

  tmpR0_.SetSize(sfes_truevsize);
  tmpR0a_.SetSize(sfes_truevsize);
  tmpR0b_.SetSize(sfes_truevsize);
  tmpR0c_.SetSize(sfes_truevsize);

  strain_.SetSize(6 * sfes_truevsize);
  sMag_.SetSize(sfes_truevsize);

  if (rank0_) grvy_printf(ginfo, "zeta-f vectors and gf initialized...\n");

  // exports
  toFlow_interface_.eddy_viscosity = &eddyVisc_gf_;
  toThermoChem_interface_.eddy_viscosity = &eddyVisc_gf_;

  //-----------------------------------------------------
  // 2) Set the initial condition
  //-----------------------------------------------------

  ConstantCoefficient tke_ic_coef;
  ConstantCoefficient tdr_ic_coef;
  ConstantCoefficient zeta_ic_coef;
  tke_ic_coef.constant = tke_ic_;
  tdr_ic_coef.constant = tdr_ic_;
  zeta_ic_coef.constant = 2.0 / 3.0;
  tke_gf_.ProjectCoefficient(tke_ic_coef);
  tdr_gf_.ProjectCoefficient(tdr_ic_coef);
  zeta_gf_.ProjectCoefficient(zeta_ic_coef);

  /*
  ConstantCoefficient tdrWall_bc_coef;
  ConstantCoefficient fWall_bc_coef;
  tdrWall_bc_coef.constant = 0.0;
  fWall_bc_coef.constant = 0.0;
  tdrWall_gf_.ProjectCoefficient(tdrWall_bc_coef);
  fWall_gf_.ProjectCoefficient(fWall_bc_coef);
  */

  //-----------------------------------------------------
  // 3) Set the boundary conditions
  //-----------------------------------------------------
  int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // inlet bc :: at least to begin assume epsi is zero-gradient
  {
    Array<int> attr_inlet(pmesh_->bdr_attributes.Max());
    attr_inlet = 0;
    for (int i = 1; i <= numInlets; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/inlet" + std::to_string(i));

      tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
      tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

      if (type == "uniform") {
        Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
        inlet_attr = 0;
        inlet_attr[patch - 1] = 1;
        double tke_value;
        tpsP_->getRequiredInput((basepath + "/tke").c_str(), tke_value);
        AddTKEDirichletBC(tke_value, inlet_attr);
      } else {
        if (rank0_) {
          std::cout << "ERROR: zeta-f inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      }
    }
  }

  // outlet bc :: homogeneous Neumann on all, so nothing needed here
  {
    Array<int> attr_outlet(pmesh_->bdr_attributes.Max());
    attr_outlet = 0;
  }

  // Wall BCs
  {
    Array<int> attr_wall(pmesh_->bdr_attributes.Max());
    attr_wall = 0;

    for (int i = 1; i <= numWalls; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/wall" + std::to_string(i));

      tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
      tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

      // for now, assume all BC's fall into this catergory
      {
        attr_wall = 0;
        attr_wall[patch - 1] = 1;

        ConstantCoefficient *tke_wall_coeff = new ConstantCoefficient();
        tke_wall_coeff->constant = 0.0;
        AddTKEDirichletBC(tke_wall_coeff, attr_wall);

        // tdr handled through Hs_bdry

        ConstantCoefficient *zeta_wall_coeff = new ConstantCoefficient();
        zeta_wall_coeff->constant = 0.0;
        AddZETADirichletBC(zeta_wall_coeff, attr_wall);

        // f handled through Hs_bdry
      }
    }
  }

  sfes_->GetEssentialTrueDofs(tke_ess_attr_, tke_ess_tdof_);
  sfes_->GetEssentialTrueDofs(tdr_ess_attr_, tdr_ess_tdof_);
  sfes_->GetEssentialTrueDofs(zeta_ess_attr_, zeta_ess_tdof_);
  sfes_->GetEssentialTrueDofs(fRate_ess_attr_, fRate_ess_tdof_);
  if (rank0_) std::cout << "Zeta-f RANS model  essential true dof step" << endl;
}

void ZetaModel::initializeOperators() {
  dt_ = time_coeff_.dt;
  Array<int> empty;

  // GLL integration rule (Numerical Integration)
  const IntegrationRule &ir_i = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 2 * order_ + 1);
  const IntegrationRule &ir_nli = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 4 * order_);
  const IntegrationRule &ir_di = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 3 * order_ - 1);

  // coefficients for operators
  zero_coeff_ = new ConstantCoefficient(0.0);
  unity_coeff_ = new ConstantCoefficient(1.0);
  posTwo_coeff_ = new ConstantCoefficient(2.0);
  negTwo_coeff_ = new ConstantCoefficient(-2.0);
  delta_coeff_ = new GridFunctionCoefficient(gridScale_gf_);
  rho_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->density);
  mu_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->viscosity);
  tts_coeff_ = new GridFunctionCoefficient(&tts_gf_);
  tls2_coeff_ = new GridFunctionCoefficient(&tls2_gf_);
  prod_coeff_ = new GridFunctionCoefficient(&prod_gf_);
  tke_coeff_ = new GridFunctionCoefficient(&tke_gf_);

  // boundary-condition related
  nu_coeff_ = new RatioCoefficient(*mu_coeff_, *rho_coeff_);
  nu_delta_coeff_ = new RatioCoefficient(*nu_coeff_, *delta_coeff_);
  gradTKE_coeff_ = new GradientGridFunctionCoefficient(&tke_gf_);
  two_nu_delta_coeff_ = new ProductCoefficient(*nu_delta_coeff_, *posTwo_coeff_);
  tdr_wall_coeff_ = new ScalarVectorProductCoefficient(*two_nu_delta_coeff_, *gradTKE_coeff_);
  gradZeta_coeff_ = new GradientGridFunctionCoefficient(&zeta_gf_);
  two_nuNeg_delta_coeff_ = new ProductCoefficient(*nu_delta_coeff_, *negTwo_coeff_);
  fRate_wall_coeff_ = new ScalarVectorProductCoefficient(*two_nuNeg_delta_coeff_, *gradZeta_coeff_);

  // convection-related
  vel_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rho_coeff_, *vel_coeff_);

  // diffusion-related
  scalar_diff_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->viscosity);
  mut_coeff_ = new GridFunctionCoefficient(&eddyVisc_gf_);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);

  tke_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0 / sigmaK_);
  tdr_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0 / sigmaO_);
  zeta_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0 / sigmaZ_);

  tke_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *tke_diff_sum_coeff_);
  tdr_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *tdr_diff_sum_coeff_);
  zeta_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *zeta_diff_sum_coeff_);
  unity_diff_total_coeff_ = new ProductCoefficient(*unity_coeff_, *unity_diff_coeff_);

  // unsteady- and destruction-related
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ /= dt_;
  rhoDt_coeff_ = new GridFunctionCoefficient(&rhoDt_gf_);
  rhoTTS_coeff_ = new RatioCoefficient(*rho_coeff_, *tts_coeff_);
  Ce2_coeff_ = new ConstantCoefficient(Ce2_);
  Ce2rhoTTS_coeff_ = new ProductCoefficient(*Ce2_coeff_, *rhoTTS_coeff_);
  Pk_coeff_ = new RatioCoefficient(*prod_coeff_, *tke_coeff_);
  tke_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *rhoTTS_coeff_);
  tdr_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *Ce2rhoTTS_coeff_);
  zeta_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *Pk_coeff_);
  f_diag_coeff_ = new RatioCoefficient(*unity_coeff_, *tls2_coeff_);
  f_diag_total_coeff_ = new SumCoefficient(*f_diag_coeff_, *zero_coeff_);

  // operators
  As_form_ = new ParBilinearForm(sfes_);
  auto *as_blfi = new ConvectionIntegrator(*rhou_coeff_);
  if (numerical_integ_) {
    as_blfi->SetIntRule(&ir_nli);
  }
  As_form_->AddDomainIntegrator(as_blfi);
  if (partial_assembly_) {
    As_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  As_form_->Assemble();
  As_form_->FormSystemMatrix(empty, As_);

  // mass matrix
  Ms_form_ = new ParBilinearForm(sfes_);
  auto *ms_blfi = new MassIntegrator;
  if (numerical_integ_) {
    ms_blfi->SetIntRule(&ir_i);
  }
  Ms_form_->AddDomainIntegrator(ms_blfi);
  if (partial_assembly_) {
    Ms_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ms_form_->Assemble();
  Ms_form_->FormSystemMatrix(empty, Ms_);

  // mass matrix with rho
  MsRho_form_ = new ParBilinearForm(sfes_);
  auto *msrho_blfi = new MassIntegrator(*rho_coeff_);
  if (numerical_integ_) {
    msrho_blfi->SetIntRule(&ir_i);
    // msrho_blfi->SetIntRule(&ir_di);
  }
  MsRho_form_->AddDomainIntegrator(msrho_blfi);
  if (partial_assembly_) {
    MsRho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);

  // helmholtz
  Hs_form_ = new ParBilinearForm(sfes_);
  auto *hmt_blfi = new MassIntegrator(*diag_coeff_);
  auto *hdt_blfi = new DiffusionIntegrator(*diff_total_coeff_);

  if (numerical_integ_) {
    hmt_blfi->SetIntRule(&ir_di);
    hdt_blfi->SetIntRule(&ir_di);
  }
  Hs_form_->AddDomainIntegrator(hmt_blfi);
  Hs_form_->AddDomainIntegrator(hdt_blfi);
  if (partial_assembly_) {
    Hs_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Hs_form_->Assemble();
  // Hs_form_->FormSystemMatrix(temp_ess_tdof_, Hs_);
  Hs_form_->FormSystemMatrix(*ess_tdof_, Hs_);
  if (rank0_) std::cout << "Zeta-F Hs operator set" << endl;

  Hs_bdry_ = new ParLinearForm(sfes_);
  auto *hs_bdry_lfi = new BoundaryNormalLFIntegrator(*wall_coeff_, 2, -1);
  if (numerical_integ_) {
    hs_bdry_lfi->SetIntRule(&ir_di);
  }
  Hs_bdry_->AddBoundaryIntegrator(hs_bdry_lfi, *ess_attr_);

  // inverse operators:: linear solves
  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Ms_form_->AssembleDiagonal(diag_pa);
    MsInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsInvPC_ = new HypreSmoother(*Ms_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  MsInv_ = new CGSolver(sfes_->GetComm());
  MsInv_->iterative_mode = false;
  MsInv_->SetOperator(*Ms_);
  MsInv_->SetPreconditioner(*MsInvPC_);
  MsInv_->SetPrintLevel(pl_solve_);
  MsInv_->SetRelTol(rtol_);
  MsInv_->SetMaxIter(max_iter_);

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Hs_form_->AssembleDiagonal(diag_pa);
    // HsInvPC_ = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof_);
    HsInvPC_ = new OperatorJacobiSmoother(diag_pa, *ess_tdof_);
  } else {
    HsInvPC_ = new HypreSmoother(*Hs_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HsInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  HsInv_ = new CGSolver(sfes_->GetComm());
  HsInv_->iterative_mode = true;
  HsInv_->SetOperator(*Hs_);
  HsInv_->SetPreconditioner(*HsInvPC_);
  HsInv_->SetPrintLevel(pl_solve_);
  HsInv_->SetRelTol(rtol_);
  HsInv_->SetMaxIter(max_iter_);

  if (rank0_) std::cout << "zeta-f operators set" << endl;
}

void ZetaModel::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("muT", &eddyVisc_gf_);
  pvdc.RegisterField("tke", &tke_gf_);
  pvdc.RegisterField("tdr", &tdr_gf_);
  pvdc.RegisterField("zeta", &zeta_gf_);
  pvdc.RegisterField("f", &fRate_gf_);
  pvdc.RegisterField("tts", &tts_gf_);
  pvdc.RegisterField("tls", &tls_gf_);
  pvdc.RegisterField("Pk", &prod_gf_);
}

void ZetaModel::setup() {
  // By calling step here, we initialize the eddy viscosity using
  // the initial velocity gradient
  this->step();
}

void ZetaModel::step() {
  // update time and rotate storage
  dt_ = time_coeff_.dt;
  time_ = time_coeff_.time;
  UpdateTimestepHistory(dt_);

  // gather necessary information from other classes
  (flow_interface_->velocity)->GetTrueDofs(vel_);
  (flow_interface_->gradU)->GetTrueDofs(gradU_);
  (flow_interface_->gradV)->GetTrueDofs(gradV_);
  (flow_interface_->gradW)->GetTrueDofs(gradW_);
  (thermoChem_interface_->density)->GetTrueDofs(rho_);
  (thermoChem_interface_->viscosity)->GetTrueDofs(mu_);

  // Set current time for Dirichlet boundary conditions.
  for (auto &tke_dbc : tke_dbcs_) {
    tke_dbc.coeff->SetTime(time_ + dt_);
  }
  for (auto &tdr_dbc : tdr_dbcs_) {
    tdr_dbc.coeff->SetTime(time_ + dt_);
  }
  for (auto &fRate_dbc : fRate_dbcs_) {
    fRate_dbc.coeff->SetTime(time_ + dt_);
  }
  for (auto &zeta_dbc : zeta_dbcs_) {
    zeta_dbc.coeff->SetTime(time_ + dt_);
  }

  // preliminaries
  extrapolateState();
  computeStrain();
  updateTTS();
  updateTLS();
  updateProd();
  // updateBC();

  // actual solves
  tkeStep();
  tdrStep();
  fStep();
  zetaStep();

  // final calc of eddy visc
  updateMuT();
}

void ZetaModel::updateMuT() {
  const double *dTKE = tke_.HostRead();
  const double *dTTS = tts_.HostRead();
  const double *dZeta = zeta_.HostRead();
  const double *dRho = rho_.HostRead();
  double *muT = eddyVisc_.HostReadWrite();

  for (int i = 0; i < SdofInt_; i++) muT[i] = Cmu_ * dRho[i];
  for (int i = 0; i < SdofInt_; i++) muT[i] *= dZeta[i];
  for (int i = 0; i < SdofInt_; i++) muT[i] *= dTKE[i];
  for (int i = 0; i < SdofInt_; i++) muT[i] *= dTTS[i];

  eddyVisc_gf_.SetFromTrueDofs(eddyVisc_);
}

void ZetaModel::computeStrain() {
  const double *dGradU = gradU_.HostRead();
  const double *dGradV = gradV_.HostRead();
  const double *dGradW = gradW_.HostRead();
  double *Sij = strain_.HostReadWrite();
  double *dSmag = sMag_.HostReadWrite();
  double Smin = 1.0e-12;

  for (int i = 0; i < SdofInt_; i++) {
    DenseMatrix gradUp;
    gradUp.SetSize(nvel_, dim_);
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(0, dir) = dGradU[i + dir * SdofInt_];
    }
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(1, dir) = dGradV[i + dir * SdofInt_];
    }
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(2, dir) = dGradW[i + dir * SdofInt_];
    }

    Sij[i + 0 * SdofInt_] = gradUp(0, 0);
    Sij[i + 1 * SdofInt_] = gradUp(1, 1);
    Sij[i + 3 * SdofInt_] = 0.5 * (gradUp(0, 1) + gradUp(1, 0));

    // TODO: make general for 2d * 2d-axisym
    Sij[i + 2 * SdofInt_] = gradUp(2, 2);
    Sij[i + 4 * SdofInt_] = 0.5 * (gradUp(0, 2) + gradUp(2, 0));
    Sij[i + 5 * SdofInt_] = 0.5 * (gradUp(1, 2) + gradUp(2, 1));
  }

  // NOTE:  not including sqrt(2) factor here
  for (int i = 0; i < SdofInt_; i++) {
    dSmag[i] = 0.0;
    for (int j = 0; j < dim_; j++) dSmag[i] += Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];
    for (int j = dim_; j < 2 * dim_; j++) dSmag[i] += 2.0 * Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];
    dSmag[i] = sqrt(dSmag[i]);
    dSmag[i] = std::max(dSmag[i], Smin);
  }
}

void ZetaModel::updateTTS() {
  const double *dTKE = tke_.HostRead();
  const double *dTDR = tdr_.HostRead();
  const double *dZeta = zeta_.HostRead();
  const double *dSmag = sMag_.HostRead();
  const double *dMu = mu_.HostRead();
  const double *dRho = rho_.HostRead();
  double *dTTS = tts_.HostReadWrite();

  double Ctime;
  Ctime = 0.6 / (std::sqrt(0.6) * std::sqrt(2.0) * Cmu_);
  for (int i = 0; i < SdofInt_; i++) {
    double T1, T2, T3;
    T1 = dTKE[i] / dTDR[i];
    T2 = Ctime / (dSmag[i] * dZeta[i]);
    T3 = Ct_ * std::sqrt((dMu[i] / (dRho[i]) * 1.0 / dTDR[i]));
    dTTS[i] = std::min(T1, T2);
    dTTS[i] = std::max(dTTS[i], T3);
    dTTS[i] = std::max(dTTS[i], tts_min_);
  }
  tts_gf_.SetFromTrueDofs(tts_);
}

void ZetaModel::updateTLS() {
  const double *dTKE = tke_.HostRead();
  const double *dTDR = tdr_.HostRead();
  const double *dZeta = zeta_.HostRead();
  const double *dSmag = sMag_.HostRead();
  const double *dMu = mu_.HostRead();
  const double *dRho = rho_.HostRead();
  double *dTLS = tls_.HostReadWrite();

  double Clength;
  Clength = 1.0 / (std::sqrt(0.6) * std::sqrt(2.0) * Cmu_);
  for (int i = 0; i < SdofInt_; i++) {
    double L1, L2, L3;
    L1 = std::pow(dTKE[i], 1.5) / dTDR[i];
    L2 = Clength * std::sqrt(dTKE[i]) / (dSmag[i] * dZeta[i]);
    L3 = Cn_ * std::pow((std::pow(dMu[i] / dRho[i], 3.0) * 1.0 / dTDR[i]), 0.25);
    dTLS[i] = std::min(L1, L2);
    dTLS[i] = Cl_ * std::max(dTLS[i], L3);
    dTLS[i] = std::max(dTLS[i], tls_min_);
  }
  tls_gf_.SetFromTrueDofs(tls_);

  double *dTLS2 = tls2_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dTLS2[i] = dTLS[i] * dTLS[i];
  }
  tls2_gf_.SetFromTrueDofs(tls2_);
}

void ZetaModel::updateProd() {
  const double *dGradU = gradU_.HostRead();
  const double *dGradV = gradV_.HostRead();
  const double *dGradW = gradW_.HostRead();
  const double *Sij = strain_.HostReadWrite();
  const double *dTKE = tke_.HostRead();
  const double *dmuT = eddyVisc_.HostRead();
  const double *dRho = rho_.HostRead();
  double *Pk = prod_.HostReadWrite();

  double twoThirds = 2.0 / 3.0;
  for (int i = 0; i < SdofInt_; i++) {
    DenseMatrix gradUp;
    gradUp.SetSize(nvel_, dim_);
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(0, dir) = dGradU[i + dir * SdofInt_];
    }
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(1, dir) = dGradV[i + dir * SdofInt_];
    }
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(2, dir) = dGradW[i + dir * SdofInt_];
    }

    double divU = 0.0;
    for (int j = 0; j < dim_; j++) divU += gradUp(j, j);

    DenseMatrix tau;
    tau.SetSize(nvel_, dim_);
    tau(0, 0) = 2.0 * Sij[i + 0 * SdofInt_];
    tau(1, 1) = 2.0 * Sij[i + 1 * SdofInt_];
    tau(2, 2) = 2.0 * Sij[i + 2 * SdofInt_];
    tau(0, 1) = 2.0 * Sij[i + 3 * SdofInt_];
    tau(1, 0) = 2.0 * Sij[i + 3 * SdofInt_];
    tau(0, 2) = 2.0 * Sij[i + 4 * SdofInt_];
    tau(1, 2) = 2.0 * Sij[i + 5 * SdofInt_];
    tau(2, 0) = 2.0 * Sij[i + 4 * SdofInt_];
    tau(2, 1) = 2.0 * Sij[i + 5 * SdofInt_];

    for (int j = 0; j < dim_; j++) {
      tau(j, j) -= twoThirds * divU;
    }
    tau *= dmuT[i];

    for (int j = 0; j < dim_; j++) {
      tau(j, j) -= twoThirds * dRho[i] * dTKE[i];
    }

    Pk[i] = 0.0;
    for (int j = 0; j < dim_; j++) {
      for (int k = 0; k < dim_; k++) {
        Pk[i] += tau(j, k) * gradUp(j, k);
      }
    }
  }

  prod_gf_.SetFromTrueDofs(prod_);
}

/// extrapolated states to {n+1}
void ZetaModel::extrapolateState() {
  tke_next_.Set(time_coeff_.ab1, tke_);
  tke_next_.Add(time_coeff_.ab2, tke_nm1_);
  tke_next_.Add(time_coeff_.ab3, tke_nm2_);

  tdr_next_.Set(time_coeff_.ab1, tdr_);
  tdr_next_.Add(time_coeff_.ab2, tdr_nm1_);
  tdr_next_.Add(time_coeff_.ab3, tdr_nm2_);

  zeta_next_.Set(time_coeff_.ab1, zeta_);
  zeta_next_.Add(time_coeff_.ab2, zeta_nm1_);
  zeta_next_.Add(time_coeff_.ab3, zeta_nm2_);
}

void ZetaModel::UpdateTimestepHistory(double dt) {
  // nonlinear products
  Ntke_nm2_ = Ntke_nm1_;
  Ntke_nm1_ = Ntke_;

  Ntdr_nm2_ = Ntdr_nm1_;
  Ntdr_nm1_ = Ntdr_;

  Nzeta_nm2_ = Nzeta_nm1_;
  Nzeta_nm1_ = Nzeta_;

  // scalars
  tke_nm2_ = tke_nm1_;
  tke_nm1_ = tke_;

  tdr_nm2_ = tdr_nm1_;
  tdr_nm1_ = tdr_;

  zeta_nm2_ = zeta_nm1_;
  zeta_nm1_ = zeta_;

  // copy previous {n+1} to {n}
  tke_next_gf_.GetTrueDofs(tke_next_);
  tke_ = tke_next_;
  tke_gf_.SetFromTrueDofs(tke_);

  tdr_next_gf_.GetTrueDofs(tdr_next_);
  tdr_ = tdr_next_;
  tdr_gf_.SetFromTrueDofs(tdr_);

  zeta_next_gf_.GetTrueDofs(zeta_next_);
  zeta_ = zeta_next_;
  zeta_gf_.SetFromTrueDofs(zeta_);
}

void ZetaModel::convection(string scalar) {
  Array<int> empty;
  As_form_->Update();
  As_form_->Assemble();
  As_form_->FormSystemMatrix(empty, As_);

  if (scalar == "tke") {
    As_->Mult(tke_, Ntke_);
    tmpR0_.Set(time_coeff_.ab1, Ntke_);
    tmpR0_.Add(time_coeff_.ab2, Ntke_nm1_);
    tmpR0_.Add(time_coeff_.ab3, Ntke_nm2_);
  } else if (scalar == "tdr") {
    As_->Mult(tdr_, Ntdr_);
    tmpR0_.Set(time_coeff_.ab1, Ntdr_);
    tmpR0_.Add(time_coeff_.ab2, Ntdr_nm1_);
    tmpR0_.Add(time_coeff_.ab3, Ntdr_nm2_);
  } else if (scalar == "zeta") {
    As_->Mult(zeta_, Nzeta_);
    tmpR0_.Set(time_coeff_.ab1, Nzeta_);
    tmpR0_.Add(time_coeff_.ab2, Nzeta_nm1_);
    tmpR0_.Add(time_coeff_.ab3, Nzeta_nm2_);
  } else {
    std::cout << "ERROR: wrong scalar passed to zeta-f convection" << endl;
    assert(false);
    exit(1);
  }
}

void ZetaModel::tkeStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection (->tmpR0_)
  convection("tke");
  res_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, tke_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, tke_nm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, tke_nm2_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // production
  Ms_->AddMult(prod_, res_, +1.0);

  // destruction => include in lhs
  // MsRho_->AddMult(tdr_, res_, -1.0);

  // Update Helmholtz operator
  // rhoDt_gf_ = *rho_gf_;
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  diag_coeff_ = tke_diag_coeff_;
  diff_total_coeff_ = tke_diff_total_coeff_;
  ess_tdof_ = &tke_ess_tdof_;
  ess_attr_ = &tke_ess_attr_;

  Hs_form_->Update();
  Hs_form_->Assemble();
  Hs_form_->FormSystemMatrix(tke_ess_tdof_, Hs_);

  HsInv_->SetOperator(*Hs_);
  if (partial_assembly_) {
    delete HsInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hs_form_->AssembleDiagonal(diag_pa);
    HsInvPC_ = new OperatorJacobiSmoother(diag_pa, tke_ess_tdof_);
    HsInv_->SetPreconditioner(*HsInvPC_);
  }

  // Prepare for the solve
  for (auto &tke_dbc : tke_dbcs_) {
    tke_next_gf_.ProjectBdrCoefficient(*tke_dbc.coeff, tke_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Hs_.As<ConstrainedOperator>();
    EliminateRHS(*Hs_form_, *HC, tke_ess_tdof_, tke_next_gf_, res_gf_, Xt2, Bt2, 1);
  } else {
    Hs_form_->FormLinearSystem(tke_ess_tdof_, tke_next_gf_, res_gf_, Hs_, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for temp
  HsInv_->Mult(Bt2, Xt2);
  assert(HsInv_->GetConverged());

  Hs_form_->RecoverFEMSolution(Xt2, res_gf_, tke_next_gf_);
  tke_next_gf_.GetTrueDofs(tke_next_);

  // hard-clip
  double *dTKE = tke_next_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dTKE[i] = std::max(dTKE[i], tke_min_);
  }
  tke_next_gf_.SetFromTrueDofs(tke_next_);
}

void ZetaModel::tdrStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection (->tmpR0_)
  convection("tdr");
  res_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, tdr_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, tdr_nm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, tdr_nm2_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // production
  const double *dp = prod_.HostRead();
  const double *dz = zeta_.HostRead();
  double *data = tmpR0_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    double ceps1 = 1.4 * (1.0 + 0.012 / dz[i]);
    data[i] = ceps1 * dp[i];
  }
  tmpR0_ /= tts_;
  Ms_->AddMult(tmpR0_, res_, +1.0);

  // destruction => include in lhs
  // tmpR0.Set(Ce2_,tdr_);
  // tmpR0 /= TTS;
  // MsRho_->AddMult(tmpR0_, res_, -1.0);

  // Update Helmholtz operator
  // rhoDt_gf_ = *rho_gf_;
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  diag_coeff_ = tdr_diag_coeff_;
  diff_total_coeff_ = tdr_diff_total_coeff_;
  ess_tdof_ = &tdr_ess_tdof_;
  ess_attr_ = &tdr_ess_attr_;
  wall_coeff_ = tdr_wall_coeff_;

  Hs_bdry_->Update();
  Hs_bdry_->Assemble();
  Hs_bdry_->ParallelAssemble(tmpR0_);
  res_.Add(-1.0, tmpR0_);

  Hs_form_->Update();
  Hs_form_->Assemble();
  Hs_form_->FormSystemMatrix(tdr_ess_tdof_, Hs_);

  HsInv_->SetOperator(*Hs_);
  if (partial_assembly_) {
    delete HsInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hs_form_->AssembleDiagonal(diag_pa);
    HsInvPC_ = new OperatorJacobiSmoother(diag_pa, tdr_ess_tdof_);
    HsInv_->SetPreconditioner(*HsInvPC_);
  }

  // Prepare for the solve
  for (auto &tdr_dbc : tdr_dbcs_) {
    tdr_next_gf_.ProjectBdrCoefficient(*tdr_dbc.coeff, tdr_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Hs_.As<ConstrainedOperator>();
    EliminateRHS(*Hs_form_, *HC, tdr_ess_tdof_, tdr_next_gf_, res_gf_, Xt2, Bt2, 1);
  } else {
    Hs_form_->FormLinearSystem(tdr_ess_tdof_, tdr_next_gf_, res_gf_, Hs_, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for temp
  HsInv_->Mult(Bt2, Xt2);
  assert(HsInv_->GetConverged());

  Hs_form_->RecoverFEMSolution(Xt2, res_gf_, tdr_next_gf_);
  tdr_next_gf_.GetTrueDofs(tdr_next_);

  // hard-clip
  double *dTDR = tdr_next_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dTDR[i] = std::max(dTDR[i], tdr_min_);
  }
  tdr_next_gf_.SetFromTrueDofs(tdr_next_);
}

void ZetaModel::zetaStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection
  convection("zeta");  // ->tmpR0_
  res_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, zeta_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, zeta_nm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, zeta_nm2_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // production
  MsRho_->AddMult(fRate_, res_, +1.0);

  // destruction => include in lhs
  // tmpR0.Set(1.0,prod_);
  // tmpR0 /= TKEn_;
  // tmpR0 *= fRate_;
  // Ms_->AddMult(tmpR0_, res_, -1.0);

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  // rhoDt_gf_ = *rho_gf_;
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  diag_coeff_ = zeta_diag_coeff_;
  diff_total_coeff_ = zeta_diff_total_coeff_;
  ess_tdof_ = &zeta_ess_tdof_;
  ess_attr_ = &zeta_ess_attr_;

  Hs_form_->Update();
  Hs_form_->Assemble();
  Hs_form_->FormSystemMatrix(zeta_ess_tdof_, Hs_);

  HsInv_->SetOperator(*Hs_);
  if (partial_assembly_) {
    delete HsInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hs_form_->AssembleDiagonal(diag_pa);
    HsInvPC_ = new OperatorJacobiSmoother(diag_pa, zeta_ess_tdof_);
    HsInv_->SetPreconditioner(*HsInvPC_);
  }

  // Prepare for the solve
  for (auto &zeta_dbc : zeta_dbcs_) {
    zeta_next_gf_.ProjectBdrCoefficient(*zeta_dbc.coeff, zeta_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Hs_.As<ConstrainedOperator>();
    EliminateRHS(*Hs_form_, *HC, zeta_ess_tdof_, zeta_next_gf_, res_gf_, Xt2, Bt2, 1);
  } else {
    Hs_form_->FormLinearSystem(zeta_ess_tdof_, zeta_next_gf_, res_gf_, Hs_, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for temp
  HsInv_->Mult(Bt2, Xt2);
  assert(HsInv_->GetConverged());

  Hs_form_->RecoverFEMSolution(Xt2, res_gf_, zeta_next_gf_);
  zeta_next_gf_.GetTrueDofs(zeta_next_);

  // hard-clip
  double *dZ = zeta_next_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dZ[i] = std::max(dZ[i], zeta_min_);
  }
  zeta_next_gf_.SetFromTrueDofs(zeta_next_);
}

void ZetaModel::fStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // assemble source
  tmpR0b_.Set(C2_, prod_);
  tmpR0b_ /= rho_;
  tmpR0b_ /= tdr_;
  tmpR0b_ += (C1_ - 1.0);

  tmpR0a_.Set(1.0, zeta_);
  tmpR0a_ -= 2.0 / 3.0;
  tmpR0a_ *= tmpR0b_;
  tmpR0a_ /= tts_;
  tmpR0a_ /= tls2_;

  Ms_->AddMult(tmpR0a_, res_, -1.0);

  // Update Helmholtz operator
  diag_coeff_ = f_diag_total_coeff_;
  diff_total_coeff_ = unity_diff_total_coeff_;
  ess_tdof_ = &fRate_ess_tdof_;
  ess_attr_ = &fRate_ess_attr_;
  wall_coeff_ = fRate_wall_coeff_;

  Hs_bdry_->Update();
  Hs_bdry_->Assemble();
  Hs_bdry_->ParallelAssemble(tmpR0_);
  res_.Add(-1.0, tmpR0_);

  Hs_form_->Update();
  Hs_form_->Assemble();
  Hs_form_->FormSystemMatrix(fRate_ess_tdof_, Hs_);

  HsInv_->SetOperator(*Hs_);
  if (partial_assembly_) {
    delete HsInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hs_form_->AssembleDiagonal(diag_pa);
    HsInvPC_ = new OperatorJacobiSmoother(diag_pa, fRate_ess_tdof_);
    HsInv_->SetPreconditioner(*HsInvPC_);
  }

  // Prepare for the solve
  for (auto &fRate_dbc : fRate_dbcs_) {
    fRate_gf_.ProjectBdrCoefficient(*fRate_dbc.coeff, fRate_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Hs_.As<ConstrainedOperator>();
    EliminateRHS(*Hs_form_, *HC, fRate_ess_tdof_, fRate_gf_, res_gf_, Xt2, Bt2, 1);
  } else {
    Hs_form_->FormLinearSystem(fRate_ess_tdof_, fRate_gf_, res_gf_, Hs_, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for temp
  HsInv_->Mult(Bt2, Xt2);
  assert(HsInv_->GetConverged());

  Hs_form_->RecoverFEMSolution(Xt2, res_gf_, fRate_gf_);
  fRate_gf_.GetTrueDofs(fRate_);

  // hard-clip
  double *df = fRate_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    df[i] = std::max(df[i], fRate_min_);
  }
  fRate_gf_.SetFromTrueDofs(fRate_);
}

/// TODO: pull in tensor gridscale and calc Mnn^T for wall-normal spacing
void ZetaModel::updateBC(int step) {
  /*
  gridScale_gf_.GetTrueDofs(tmpR0b_);
  const double *dTKE = tke_.HostRead();
  const double *dZeta = zeta_.HostRead();
  const double *dMu = mu_.HostRead();
  const double *delta = tmpR0b_.HostRead();
  double *wallBC = tmpR0_.HostReadWrite();

  //gradk * n *delta

  tdrWall_gf_.GetTrueDofs(tmpR0_);
  for (int i = 0; i < SdofInt_; i++) {
  }
  tdrWall_gf_.SetFromTrueDofs(tmpR0_);

  fWall_gf_.GetTrueDofs(tmpR0_);
  for (int i = 0; i < SdofInt_; i++) {
  }
  fWall_gf_.SetFromTrueDofs(tmpR0_);
  */
}

/// Add a Dirichlet boundary condition to scalar fields
void ZetaModel::AddTKEDirichletBC(const double &tke, Array<int> &attr) {
  tke_dbcs_.emplace_back(attr, new ConstantCoefficient(tke));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!tke_ess_attr_[i]);
      tke_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddTKEDirichletBC(Coefficient *coeff, Array<int> &attr) {
  tke_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!tke_ess_attr_[i]);
      tke_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddTDRDirichletBC(const double &tdr, Array<int> &attr) {
  tdr_dbcs_.emplace_back(attr, new ConstantCoefficient(tdr));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!tdr_ess_attr_[i]);
      tdr_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddTDRDirichletBC(Coefficient *coeff, Array<int> &attr) {
  tdr_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!tdr_ess_attr_[i]);
      tdr_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddZETADirichletBC(const double &zeta, Array<int> &attr) {
  zeta_dbcs_.emplace_back(attr, new ConstantCoefficient(zeta));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!zeta_ess_attr_[i]);
      zeta_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddZETADirichletBC(Coefficient *coeff, Array<int> &attr) {
  zeta_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!zeta_ess_attr_[i]);
      zeta_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddFRATEDirichletBC(const double &fRate, Array<int> &attr) {
  fRate_dbcs_.emplace_back(attr, new ConstantCoefficient(fRate));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!fRate_ess_attr_[i]);
      fRate_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddFRATEDirichletBC(Coefficient *coeff, Array<int> &attr) {
  fRate_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!fRate_ess_attr_[i]);
      fRate_ess_attr_[i] = 1;
    }
  }
}
