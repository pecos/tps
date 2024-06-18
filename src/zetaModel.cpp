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

ZetaModel::ZetaModel(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps,
                                               ParGridFunction *gridScale)
    : tpsP_(tps), loMach_opts_(loMach_opts), pmesh_(pmesh) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  dim_ = pmesh_->Dimension();
  nvel_ = dim_;
  order_ = loMach_opts_->order;
  gridScale_ = gridScale;

  // read-ins, if any
  // example: tpsP_->getInput("loMach/sgsModelConstant", sgs_model_const_, sgs_const);

  tpsP_->getInput("initialConditions/tke", tke_ic_, 1.0e-8);
  tpsP_->getInput("initialConditions/tdr", tke_ic_, 1.0e-10);
  
}

ZetaModel::~ZetaModel() {
  delete TDRwall_coeff_;
  delete Fwall_coeff_;  
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

  fRate_gf_.SetSpace(sfes_);
  fRate_.SetSize(sfes_truevsize);  
  
  tls_gf_.SetSpace(sfes_);
  tls_.SetSize(sfes_truevsize);  

  tts_gf_.SetSpace(sfes_);
  tts_.SetSize(sfes_truevsize);

  prod_gf_.SetSpace(sfes_);
  prod_.SetSize(sfes_truevsize);   

  tdrWall_gf_.SetSpace(sfes_);
  fWall_gf_.SetSpace(sfes_);  
  
  vel_.SetSize(vfes_truevsize);  
  gradU_.SetSize(vfes_truevsize);
  gradV_.SetSize(vfes_truevsize);
  gradW_.SetSize(vfes_truevsize);
  
  rn_.SetSize(sfes_truevsize);
  delta_.SetSize(sfes_truevsize);  

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
  zeta_ic_coef.constant = 2.0/3.0;  
  tke_gf_.ProjectCoefficient(tke_ic_coef);
  tdr_gf_.ProjectCoefficient(tdr_ic_coef);
  zeta_gf_.ProjectCoefficient(zeta_ic_coef);    

  ConstantCoefficient tdrWall_bc_coef;
  ConstantCoefficient fWall_bc_coef;
  tdrWall_bc_coef.constant = 0.0;
  fWall_bc_coef.constant = 0.0;    
  tdrWall_gf_.ProjectCoefficient(tdrWall_bc_coef);
  fWall_gf_.ProjectCoefficient(fWall_bc_coef);
  
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

        ConstantCoefficient *TKEwall_coeff = new ConstantCoefficient();
        TKEwall_coeff->constant = 0.0;
        AddTKEDirichletBC(TKEwall_coeff, attr_wall);

        TDRwall_coeff_ = new GridFunctionCoefficient(&tdrWall_gf_);
        AddTDRDirichletBC(TDRwall_coeff_, attr_wall);

        ConstantCoefficient *ZETAwall_coeff = new ConstantCoefficient();
        ZETAwall_coeff->constant = 0.0;
        AddZETADirichletBC(ZETAwall_coeff, attr_wall);

        Fwall_coeff_ = new GridFunctionCoefficient(&fWall_gf_);	
        AddFDirichletBC(Fwall_coeff_, attr_wall);		
      }
    }

  }

  sfes_->GetEssentialTrueDofs(tke_ess_attr_, tke_ess_tdof_);
  sfes_->GetEssentialTrueDofs(tdr_ess_attr_, tdr_ess_tdof_);
  sfes_->GetEssentialTrueDofs(zeta_ess_attr_, zeta_ess_tdof_);
  sfes_->GetEssentialTrueDofs(f_ess_attr_, f_ess_tdof_);    
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
  rhoDt = rn_gf_;
  rhoDt /= dt_;
  rho_over_dt_coeff_ = new GridFunctionCoefficient(&rhoDt);  
  rho_coeff_ = new GridFunctionCoefficient(&rn_gf_);

  un_next_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhon_next_coeff_ = new GridFunctionCoefficient(&rn_gf_);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rhon_next_coeff_, *un_next_coeff_);
  
  scalar_diff_coeff_ = new GridFunctionCoefficient(&mu_gf_);
  mut_coeff_ = new GridFunctionCoefficient(&eddyVisc_gf_);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);  
  
  tke_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0/sigmaK);
  tdr_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0/sigmaO);
  zeta_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0, 1.0/sigmaZ);  
  
  tke_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *tke_diff_sum_coeff_);
  tdr_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *tdr_diff_sum_coeff_);
  zeta_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *zeta_diff_sum_coeff_);
    
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
  
  Ht_form_ = new ParBilinearForm(sfes_);
  auto *hmt_blfi = new MassIntegrator(*rho_over_dt_coeff_);
  auto *hdt_blfi = new DiffusionIntegrator(*thermal_diff_total_coeff_);

  if (numerical_integ_) {
    hmt_blfi->SetIntRule(&ir_di);
    hdt_blfi->SetIntRule(&ir_di);
  }
  Ht_form_->AddDomainIntegrator(hmt_blfi);
  Ht_form_->AddDomainIntegrator(hdt_blfi);
  if (partial_assembly_) {
    Ht_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ht_form_->Assemble();
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem Ht operator set" << endl;

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
    Ht_form_->AssembleDiagonal(diag_pa);
    HtInvPC_ = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof_);
  } else {
    HtInvPC_ = new HypreSmoother(*Ht_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HtInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  HtInv_ = new CGSolver(sfes_->GetComm());
  HtInv_->iterative_mode = true;
  HtInv_->SetOperator(*Ht_);
  HtInv_->SetPreconditioner(*HtInvPC_);
  HtInv_->SetPrintLevel(pl_solve_);
  HtInv_->SetRelTol(rtol_);
  HtInv_->SetMaxIter(max_iter_);

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

void AlgebraicSubgridModels::step() {
  // std::cout << " giddy up" << endl;

  // gather necessary information from other classes
  (flow_interface_->velocity)->GetTrueDofs(vel_);  
  (flow_interface_->gradU)->GetTrueDofs(gradU_);
  (flow_interface_->gradV)->GetTrueDofs(gradV_);
  (flow_interface_->gradW)->GetTrueDofs(gradW_);
  (thermoChem_interface_->density)->GetTrueDofs(rn_);

  eddyVisc_ = 0.0;
  const double *dGradU = gradU_.HostRead();
  const double *dGradV = gradV_.HostRead();
  const double *dGradW = gradW_.HostRead();
  const double *rho = rn_.HostRead();
  const double *Un = vel_.HostRead();  
  const double *del = gridScale_->HostRead();
  double *data = eddyVisc_.HostReadWrite();

  // solve order:
  // 1. update T,L,Pk
  // 2. advance tke
  // 3. advance tdr
  // 4. solve f
  // 5. advance zeta  
  
  if (sModel_ == 1) {
    // std::cout << "In Smag loop " << endl;
    for (int i = 0; i < SdofInt_; i++) {
      double nu_sgs = 0.;
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
      sgsSmag(gradUp, del[i], nu_sgs);
      // std::cout << "gradU diag:" << gradUp(0,0) << " " << gradUp(1,1) << " " << gradUp(1,1) << "| nuT: " << nu_sgs <<
      // endl;
      data[i] = rho[i] * nu_sgs;
    }

  } else if (sModel_ == 2) {
    for (int i = 0; i < SdofInt_; i++) {
      double nu_sgs = 0.;
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
      sgsSigma(gradUp, del[i], nu_sgs);
      data[i] = rho[i] * nu_sgs;
    }
  }

  subgridVisc_gf_.SetFromTrueDofs(subgridVisc_);
}

/**
Basic Smagorinksy subgrid model with user-specified coefficient
*/
void AlgebraicSubgridModels::sgsSmag(const DenseMatrix &gradUp, double delta, double &nu) {
  Vector Sij(6);
  double Smag = 0.;
  double Cd;
  // double l_floor; // TODO(swh): re-instate
  double d_model;
  Cd = sgs_model_const_;

  // gradUp is in (eq,dim) form
  Sij[0] = gradUp(0, 0);
  Sij[1] = gradUp(1, 1);
  Sij[2] = gradUp(2, 2);
  Sij[3] = 0.5 * (gradUp(0, 1) + gradUp(1, 0));
  Sij[4] = 0.5 * (gradUp(0, 2) + gradUp(2, 0));
  Sij[5] = 0.5 * (gradUp(1, 2) + gradUp(2, 1));

  // strain magnitude with silly sqrt(2) factor
  for (int i = 0; i < 3; i++) Smag += Sij[i] * Sij[i];
  for (int i = 3; i < 6; i++) Smag += 2.0 * Sij[i] * Sij[i];
  Smag = sqrt(2.0 * Smag);

  // eddy viscosity with delta shift
  // l_floor = config->GetSgsFloor();
  // d_model = Cd * max(delta-l_floor,0.0);
  d_model = Cd * delta;
  nu = d_model * d_model * Smag;
}

/**
NOT TESTED: Sigma subgrid model following Nicoud et.al., "Using singular values to build a
subgrid-scale model for large eddy simulations", PoF 2011.
*/
void AlgebraicSubgridModels::sgsSigma(const DenseMatrix &gradUp, double delta, double &nu) {
  DenseMatrix Qij(dim_, dim_);
  DenseMatrix du(dim_, dim_);
  DenseMatrix B(dim_, dim_);
  Vector ev(dim_);
  Vector sigma(dim_);
  double Cd;
  double sml = 1.0e-12;
  double pi = 3.14159265359;
  double onethird = 1. / 3.;
  double d_model, d4;
  double p1, p2, p, q, detB, r, phi;
  Cd = sgs_model_const_;

  // Qij = u_{k,i}*u_{k,j}
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      Qij(i, j) = 0.;
    }
  }
  for (int k = 0; k < dim_; k++) {
    for (int j = 0; j < dim_; j++) {
      for (int i = 0; i < dim_; i++) {
        Qij(i, j) += gradUp(k, i) * gradUp(k, j);
      }
    }
  }

  // shifted grid scale, d should really be sqrt of J^T*J
  // l_floor = config->GetSgsFloor();
  // d_model = max((delta-l_floor), sml);
  d_model = delta;
  d4 = pow(d_model, 4);
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      Qij(i, j) *= d4;
    }
  }

  // eigenvalues for symmetric pos-def 3x3
  p1 = Qij(0, 1) * Qij(0, 1) + Qij(0, 2) * Qij(0, 2) + Qij(1, 2) * Qij(1, 2);
  q = onethird * (Qij(0, 0) + Qij(1, 1) + Qij(2, 2));
  p2 = (Qij(0, 0) - q) * (Qij(0, 0) - q) + (Qij(1, 1) - q) * (Qij(1, 1) - q) + (Qij(2, 2) - q) * (Qij(2, 2) - q) +
       2.0 * p1;
  p = std::sqrt(max(p2, 0.0) / 6.0);
  // cout << "p1, q, p2, p: " << p1 << " " << q << " " << p2 << " " << p << endl; fflush(stdout);

  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      B(i, j) = Qij(i, j);
    }
  }
  for (int i = 0; i < dim_; i++) {
    B(i, i) -= q;
  }
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      B(i, j) *= (1.0 / max(p, sml));
    }
  }
  detB = B(0, 0) * (B(1, 1) * B(2, 2) - B(2, 1) * B(1, 2)) - B(0, 1) * (B(1, 0) * B(2, 2) - B(2, 0) * B(1, 2)) +
         B(0, 2) * (B(1, 0) * B(2, 1) - B(2, 0) * B(1, 1));
  r = 0.5 * detB;
  // cout << "r: " << r << endl; fflush(stdout);

  if (r <= -1.0) {
    phi = onethird * pi;
  } else if (r >= 1.0) {
    phi = 0.0;
  } else {
    phi = onethird * acos(r);
  }

  // eigenvalues satisfy eig3 <= eig2 <= eig1 (L^4/T^2)
  ev[0] = q + 2.0 * p * cos(phi);
  ev[2] = q + 2.0 * p * cos(phi + (2.0 * onethird * pi));
  ev[1] = 3.0 * q - ev[0] - ev[2];

  // actual sigma (L^2/T)
  sigma[0] = sqrt(max(ev[0], sml));
  sigma[1] = sqrt(max(ev[1], sml));
  sigma[2] = sqrt(max(ev[2], sml));
  // cout << "sigma: " << sigma[0] << " " << sigma[1] << " " << sigma[2] << endl; fflush(stdout);

  // eddy viscosity
  nu = sigma[2] * (sigma[0] - sigma[1]) * (sigma[1] - sigma[2]);
  nu = max(nu, 0.0);
  nu /= (sigma[0] * sigma[0]);
  nu *= (Cd * Cd);
  // cout << "mu: " << mu << endl; fflush(stdout);

  // shouldnt be necessary
  if (nu != nu) nu = 0.0;
}
