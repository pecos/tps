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

  tpsP_->getInput("ransModel/tke-ic", tke_ic_, 1.0e-4);
  tpsP_->getInput("ransModel/tdr-ic", tdr_ic_, 1.0e-8);

  tpsP_->getInput("ransModel/tke-min", tke_min_, 1.0e-11);
  tpsP_->getInput("ransModel/tdr-min", tdr_min_, 1.0e-12);
  tpsP_->getInput("ransModel/zeta-min", zeta_min_, 1.0e-12);
  tpsP_->getInput("ransModel/v2-min", v2_min_, 1.0e-12);  
  tpsP_->getInput("ransModel/f-min", fRate_min_, 1.0e-14);
  tpsP_->getInput("ransModel/tts-min", tts_min_, 1.0e-12);
  tpsP_->getInput("ransModel/tls-min", tls_min_, 1.0e-12);
  tpsP_->getInput("ransModel/tts-max", tts_max_, 100.0);
  tpsP_->getInput("ransModel/tls-max", tls_max_, 100.0);
  tpsP_->getInput("ransModel/mut-min", mut_min_, 1.0e-12);
  tpsP_->getInput("ransModel/destruction", des_wgt_, 1.0);
  tpsP_->getInput("ransModel/tls-coeff", Cl_, 0.23);
  tpsP_->getInput("ransModel/f-order", forder_, order_);
}

ZetaModel::~ZetaModel() {

  delete HkInv_;
  delete HkInvPC_;
  delete HeInv_;
  delete HeInvPC_;
  delete HvInv_;
  delete HvInvPC_;
  delete HfInv_;
  delete HfInvPC_;
  delete HzInv_;
  delete HzInvPC_;    
  delete MsInv_;
  delete MsInvPC_;
  delete MsRhoInv_;
  delete MsRhoInvPC_;  
  delete Hk_form_;
  delete He_form_;
  delete Hv_form_;
  delete Hf_form_;
  delete Hz_form_;
  delete Lk_form_;
  delete Lf_form_;  
  delete MsRho_form_;
  delete Ms_form_;
  delete Mf_form_;  
  delete As_form_;
  //delete He_bdry_;
  
  delete zero_coeff_;
  delete unity_coeff_;
  delete posTwo_coeff_;
  delete negTwo_coeff_;
  delete delta_coeff_;
  delete rho_coeff_;
  delete mu_coeff_;
  delete tts_coeff_;
  delete tls2_coeff_;
  delete prod_coeff_;
  delete tke_coeff_;
  delete nu_coeff_;
  delete nu_delta_coeff_;
  delete gradTKE_coeff_;
  delete two_nu_delta_coeff_;
  delete tdr_wall_coeff_;
  delete gradZeta_coeff_;
  delete two_nuNeg_delta_coeff_;
  delete fRate_wall_coeff_;
  delete vel_coeff_;
  delete rhou_coeff_;
  delete scalar_diff_coeff_;
  delete mut_coeff_;
  delete mult_coeff_;
  delete tke_diff_sum_coeff_;
  delete tdr_diff_sum_coeff_;
  delete zeta_diff_sum_coeff_;
  delete tke_diff_total_coeff_;
  delete tdr_diff_total_coeff_;
  delete zeta_diff_total_coeff_;
  delete unity_diff_total_coeff_;
  delete rhoDt_coeff_;
  delete rhoTTS_coeff_;
  delete Ce2_coeff_;
  delete Ce2rhoTTS_coeff_;
  delete Pk_coeff_;
  delete ek_coeff_;  
  delete tke_diag_coeff_;
  delete tdr_diag_coeff_;
  delete zeta_diag_coeff_;
  delete v2_diag_coeff_;  
  delete f_diag_coeff_;
  delete f_diag_total_coeff_;

  delete ffec_;
  delete ffes_;  
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

  // f-rate
  ffec_ = new H1_FECollection(forder_);
  ffes_ = new ParFiniteElementSpace(pmesh_, ffec_);
  
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

    v2_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    v2_ess_attr_ = 0;
    
    fRate_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    fRate_ess_attr_ = 0;
  }

  int vfes_truevsize = vfes_->GetTrueVSize();
  int sfes_truevsize = sfes_->GetTrueVSize();
  int ffes_truevsize = ffes_->GetTrueVSize();  

  Sdof_ = sfes_->GetNDofs();
  SdofInt_ = sfes_->GetTrueVSize();

  eddyVisc_gf_.SetSpace(sfes_);
  eddyVisc_.SetSize(sfes_truevsize);
  eddyVisc_gf_ = 1.0e-2;
  eddyVisc_gf_.GetTrueDofs(eddyVisc_);  

  tke_gf_.SetSpace(sfes_);
  tke_.SetSize(sfes_truevsize);
  tke_gf_ = tke_min_;
  tke_gf_.GetTrueDofs(tke_);
  
  tdr_gf_.SetSpace(sfes_);
  tdr_.SetSize(sfes_truevsize);
  tdr_gf_ = tdr_min_;
  tdr_gf_.GetTrueDofs(tdr_);  
  
  zeta_gf_.SetSpace(sfes_);
  zeta_.SetSize(sfes_truevsize);
  zeta_gf_ = 2.0/3.0;
  zeta_gf_.GetTrueDofs(zeta_);  

  v2_gf_.SetSpace(sfes_);
  v2_.SetSize(sfes_truevsize);
  v2_gf_ = 2.0/3.0 * tke_min_;
  v2_gf_.GetTrueDofs(v2_);  
  
  fRate_gf_.SetSpace(ffes_);
  fRate_.SetSize(ffes_truevsize);
  fRate_gf_ = 0.0;
  fRate_ = 0.0;
  
  tke_next_gf_.SetSpace(sfes_);
  tke_next_.SetSize(sfes_truevsize);
  tke_next_gf_ = tke_gf_;
  tke_next_gf_.GetTrueDofs(tke_next_);  

  tdr_next_gf_.SetSpace(sfes_);
  tdr_next_.SetSize(sfes_truevsize);
  tdr_next_gf_ = tdr_gf_;
  tdr_next_gf_.GetTrueDofs(tdr_next_); 

  zeta_next_gf_.SetSpace(sfes_);
  zeta_next_.SetSize(sfes_truevsize);
  zeta_next_gf_ = zeta_gf_;
  zeta_next_gf_.GetTrueDofs(zeta_next_);    

  v2_next_gf_.SetSpace(sfes_);
  v2_next_.SetSize(sfes_truevsize);
  v2_next_gf_ = v2_gf_;
  v2_next_gf_.GetTrueDofs(v2_next_);    
  
  tke_nm1_.SetSize(sfes_truevsize);
  tke_nm2_.SetSize(sfes_truevsize);
  Ntke_.SetSize(sfes_truevsize);
  Ntke_nm1_.SetSize(sfes_truevsize);
  Ntke_nm2_.SetSize(sfes_truevsize);
  tke_nm1_ = tke_min_;
  tke_nm2_ = tke_min_;
  Ntke_ = 0.0;
  Ntke_nm1_ = 0.0;
  Ntke_nm2_ = 0.0;
  
  tdr_nm1_.SetSize(sfes_truevsize);
  tdr_nm2_.SetSize(sfes_truevsize);
  Ntdr_.SetSize(sfes_truevsize);
  Ntdr_nm1_.SetSize(sfes_truevsize);
  Ntdr_nm2_.SetSize(sfes_truevsize);
  tdr_nm1_ = tdr_min_;
  tdr_nm2_ = tdr_min_;
  Ntdr_ = 0.0;
  Ntdr_nm1_ = 0.0;
  Ntdr_nm2_ = 0.0;
  
  zeta_nm1_.SetSize(sfes_truevsize);
  zeta_nm2_.SetSize(sfes_truevsize);
  Nzeta_.SetSize(sfes_truevsize);
  Nzeta_nm1_.SetSize(sfes_truevsize);
  Nzeta_nm2_.SetSize(sfes_truevsize);
  zeta_nm1_ = 2.0/3.0;
  zeta_nm2_ = 2.0/3.0;
  Nzeta_ = 0.0;
  Nzeta_nm1_ = 0.0;
  Nzeta_nm2_ = 0.0;

  v2_nm1_.SetSize(sfes_truevsize);
  v2_nm2_.SetSize(sfes_truevsize);
  Nv2_.SetSize(sfes_truevsize);
  Nv2_nm1_.SetSize(sfes_truevsize);
  Nv2_nm2_.SetSize(sfes_truevsize);
  v2_nm1_ = 2.0/3.0 * tke_min_;
  v2_nm2_ = 2.0/3.0 * tke_min_;
  Nv2_ = 0.0;
  Nv2_nm1_ = 0.0;
  Nv2_nm2_ = 0.0;  
  
  tls_gf_.SetSpace(sfes_);
  tls_.SetSize(sfes_truevsize);
  tls_gf_ = 1.0;
  tls_gf_.GetTrueDofs(tls_);
  
  tls2_gf_.SetSpace(ffes_);
  tls2_.SetSize(ffes_truevsize);
  tls2_gf_ = 1.0;
  tls2_gf_.GetTrueDofs(tls2_);  

  tts_gf_.SetSpace(sfes_);
  tts_.SetSize(sfes_truevsize);
  tts_gf_ = 1.0;
  tts_gf_.GetTrueDofs(tts_);
  tts_strain_.SetSize(sfes_truevsize);
  tts_kol_.SetSize(sfes_truevsize);    
  
  prod_gf_.SetSpace(sfes_);
  prod_.SetSize(sfes_truevsize);
  prod_gf_ = 1.0e-8;
  prod_gf_.GetTrueDofs(prod_);

  prod_next_gf_.SetSpace(sfes_);    
  prod_next_.SetSize(sfes_truevsize);  
  prod_nm1_.SetSize(sfes_truevsize);
  prod_nm2_.SetSize(sfes_truevsize);
  prod_next_.Set(1.0, prod_);  
  prod_nm1_.Set(1.0, prod_);
  prod_nm2_.Set(1.0, prod_);    

  tts_next_gf_.SetSpace(sfes_);    
  tts_next_.SetSize(sfes_truevsize);  
  tts_nm1_.SetSize(sfes_truevsize);
  tts_nm2_.SetSize(sfes_truevsize);
  tts_next_.Set(1.0, tts_);  
  tts_nm1_.Set(1.0, tts_);
  tts_nm2_.Set(1.0, tts_);      

  tls2_next_gf_.SetSpace(sfes_);  
  tls2_next_.SetSize(sfes_truevsize);  
  tls2_nm1_.SetSize(sfes_truevsize);
  tls2_nm2_.SetSize(sfes_truevsize);
  tls2_next_.Set(1.0, tls2_);  
  tls2_nm1_.Set(1.0, tls2_);
  tls2_nm2_.Set(1.0, tls2_);          
  
  tdr_wall_gf_.SetSpace(sfes_);
  // fWall_gf_.SetSpace(sfes_);

  res_gf_.SetSpace(sfes_);
  res_.SetSize(sfes_truevsize);

  resf_gf_.SetSpace(ffes_);
  resf_.SetSize(ffes_truevsize);
  
  rho_.SetSize(sfes_truevsize);
  mu_.SetSize(sfes_truevsize);
  mult_.SetSize(sfes_truevsize);  
  vel_.SetSize(vfes_truevsize);
  gradU_.SetSize(vfes_truevsize);
  gradV_.SetSize(vfes_truevsize);
  gradW_.SetSize(vfes_truevsize);

  tmpR0_.SetSize(sfes_truevsize);
  tmpR0a_.SetSize(sfes_truevsize);
  tmpR0b_.SetSize(sfes_truevsize);
  tmpR0c_.SetSize(sfes_truevsize);
  ftmpR0_.SetSize(ffes_truevsize);  

  rhoDt_gf_.SetSpace(sfes_);

  sMag_gf_.SetSpace(sfes_);
  sMag_gf_ = 1.0e-12;  
  strain_.SetSize(6 * sfes_truevsize);
  sMag_.SetSize(sfes_truevsize);  
  strain_ = 1.0e-14;
  sMag_ = 1.0e-12;
  
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
  ConstantCoefficient v2_ic_coef;  
  tke_ic_coef.constant = tke_ic_;
  tdr_ic_coef.constant = tdr_ic_;
  zeta_ic_coef.constant = 2.0 / 3.0;
  v2_ic_coef.constant = 2.0 / 3.0 * tke_ic_;  
  
  tke_gf_.ProjectCoefficient(tke_ic_coef);
  tdr_gf_.ProjectCoefficient(tdr_ic_coef);
  zeta_gf_.ProjectCoefficient(zeta_ic_coef);
  v2_gf_.ProjectCoefficient(v2_ic_coef);  
  tke_gf_.GetTrueDofs(tke_);
  tdr_gf_.GetTrueDofs(tdr_);
  zeta_gf_.GetTrueDofs(zeta_);
  v2_gf_.GetTrueDofs(v2_);    
  
  tke_next_gf_.ProjectCoefficient(tke_ic_coef);
  tdr_next_gf_.ProjectCoefficient(tdr_ic_coef);
  zeta_next_gf_.ProjectCoefficient(zeta_ic_coef);
  v2_next_gf_.ProjectCoefficient(v2_ic_coef);  
  tke_next_gf_.GetTrueDofs(tke_next_);
  tdr_next_gf_.GetTrueDofs(tdr_next_);
  zeta_next_gf_.GetTrueDofs(zeta_next_);
  v2_next_gf_.GetTrueDofs(v2_next_);  

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
        AddV2DirichletBC(2.0/3.0*tke_value, inlet_attr);	
      // TODO: Currently just applies a uniform BC as well
      } else if (type == "interpolate") {
        if (rank0_) {
          std::cout << "Zeta Model: Setting interpolated Dirichlet TKE on patch = " << patch << std::endl;
        }
        Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
        inlet_attr = 0;
        inlet_attr[patch - 1] = 1;
        double tke_value;
        tpsP_->getRequiredInput((basepath + "/tke").c_str(), tke_value);
        AddTKEDirichletBC(tke_value, inlet_attr);
        AddV2DirichletBC(2.0/3.0*tke_value, inlet_attr);	
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

      // for now, assume all wall BC's fall into this catergory
      {
        attr_wall = 0;
        attr_wall[patch - 1] = 1;

        //ConstantCoefficient *tke_wall_coeff = new ConstantCoefficient();
        //tke_wall_coeff->constant = 0.0;
        //AddTKEDirichletBC(tke_wall_coeff, attr_wall);
	AddTKEDirichletBC(0.0, attr_wall);
	AddV2DirichletBC(0.0, attr_wall);	

        // tdr handled through Hs_bdry
        // ConstantCoefficient *tdr_wall_coeff = new ConstantCoefficient();
        // tdr_wall_coeff->constant = 0.0;
	//>>>
        AddTDRDirichletBC(0.0, attr_wall);
	
        //tdr_bc_ = new GridFunctionCoefficient(&tdr_wall_gf_);
        //AddTDRDirichletBC(tdr_bc_, attr_wall);	

        //ConstantCoefficient *zeta_wall_coeff = new ConstantCoefficient();
        //zeta_wall_coeff->constant = 0.0;
        //AddZETADirichletBC(zeta_wall_coeff, attr_wall);
        AddZETADirichletBC(0.0, attr_wall);	

        //ConstantCoefficient *fRate_wall_coeff = new ConstantCoefficient();
        //fRate_wall_coeff->constant = 0.0;
        //AddFRATEDirichletBC(fRate_wall_coeff, attr_wall);
        AddFRATEDirichletBC(0.0, attr_wall);	
	
      }
    }
  }

  sfes_->GetEssentialTrueDofs(tke_ess_attr_, tke_ess_tdof_);
  sfes_->GetEssentialTrueDofs(tdr_ess_attr_, tdr_ess_tdof_);
  sfes_->GetEssentialTrueDofs(zeta_ess_attr_, zeta_ess_tdof_);
  sfes_->GetEssentialTrueDofs(v2_ess_attr_, v2_ess_tdof_);  
  ffes_->GetEssentialTrueDofs(fRate_ess_attr_, fRate_ess_tdof_);
  if (rank0_) std::cout << "Zeta-f RANS model  essential true dof step" << endl;
  
}

void ZetaModel::initializeOperators() {
  if (rank0_) std::cout << "... here we go ..." << endl;
  
  dt_ = time_coeff_.dt;
  Array<int> empty;

  // GLL integration rule (Numerical Integration)
  const IntegrationRule &ir_lump = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 2 * order_ - 1);    
  
  const IntegrationRule &ir_i = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 2 * order_ + 1);
  const IntegrationRule &ir_nli = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 4 * order_);
  const IntegrationRule &ir_di = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 3 * order_ - 1);

  const IntegrationRule &ir_if = gll_rules_.Get(ffes_->GetFE(0)->GetGeomType(), 2 * forder_ + 1);
  const IntegrationRule &ir_dif = gll_rules_.Get(ffes_->GetFE(0)->GetGeomType(), 3 * forder_ - 1);

  // coefficients for operators
  zero_coeff_ = new ConstantCoefficient(0.0);
  unity_coeff_ = new ConstantCoefficient(1.0);
  posTwo_coeff_ = new ConstantCoefficient(2.0);
  negTwo_coeff_ = new ConstantCoefficient(-2.0);
  delta_coeff_ = new GridFunctionCoefficient(gridScale_gf_);
  rho_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->density);
  mu_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->viscosity);
  tts_coeff_ = new GridFunctionCoefficient(&tts_next_gf_);
  tls2_coeff_ = new GridFunctionCoefficient(&tls2_next_gf_);
  prod_coeff_ = new GridFunctionCoefficient(&prod_next_gf_);
  tke_coeff_ = new GridFunctionCoefficient(&tke_next_gf_);
  tdr_coeff_ = new GridFunctionCoefficient(&tdr_next_gf_);  

  // boundary-condition related
  nu_coeff_ = new RatioCoefficient(*mu_coeff_, *rho_coeff_);
  nu_delta_coeff_ = new RatioCoefficient(*nu_coeff_, *delta_coeff_);
  gradTKE_coeff_ = new GradientGridFunctionCoefficient(&tke_next_gf_);
  two_nu_delta_coeff_ = new ProductCoefficient(*nu_delta_coeff_, *posTwo_coeff_);
  tdr_wall_coeff_ = new ScalarVectorProductCoefficient(*two_nu_delta_coeff_, *gradTKE_coeff_);
  gradZeta_coeff_ = new GradientGridFunctionCoefficient(&zeta_next_gf_);
  two_nuNeg_delta_coeff_ = new ProductCoefficient(*nu_delta_coeff_, *negTwo_coeff_);
  fRate_wall_coeff_ = new ScalarVectorProductCoefficient(*two_nuNeg_delta_coeff_, *gradZeta_coeff_);
  tdr_wall_eval_coeff_ = new GridFunctionCoefficient(&tdr_wall_gf_);

  // convection-related
  vel_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rho_coeff_, *vel_coeff_);

  // diffusion-related
  scalar_diff_coeff_ = new GridFunctionCoefficient(thermoChem_interface_->viscosity);
  mut_coeff_ = new GridFunctionCoefficient(&eddyVisc_gf_);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);
  tke_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0 / sigmaK_, 1.0);
  tdr_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0 / sigmaE_, 1.0);
  zeta_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *scalar_diff_coeff_, 1.0 / sigmaZ_, 1.0);
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
  ek_coeff_ = new RatioCoefficient(*tdr_coeff_, *tke_coeff_); 
  ek_rho_coeff_ = new ProductCoefficient(*ek_coeff_, *rho_coeff_);   
  tke_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *rhoTTS_coeff_, 1.0, des_wgt_);
  //tke_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *ek_rho_coeff_, 1.0, des_wgt_);  
  tdr_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *Ce2rhoTTS_coeff_, 1.0, des_wgt_);    
  zeta_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *Pk_coeff_, 1.0, 1.0);
  v2_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *rhoTTS_coeff_, 1.0, 6.0*des_wgt_);
  //v2_diag_coeff_ = new SumCoefficient(*rhoDt_coeff_, *ek_rho_coeff_, 1.0, 6.0*des_wgt_);  
  f_diag_coeff_ = new RatioCoefficient(1.0, *tls2_coeff_);
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

  Mf_form_ = new ParBilinearForm(ffes_);
  auto *mf_blfi = new MassIntegrator;
  if (numerical_integ_) {
    mf_blfi->SetIntRule(&ir_if);
  }
  Mf_form_->AddDomainIntegrator(mf_blfi);
  if (partial_assembly_) {
    Mf_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mf_form_->Assemble();
  Mf_form_->FormSystemMatrix(empty, Mf_);

  // diffusion of tke for tdr bc
  Lk_form_ = new ParBilinearForm(sfes_);
  // auto *lkd_blfi = new DiffusionIntegrator(*tke_diff_total_coeff_);
  auto *lkd_blfi = new DiffusionIntegrator(*scalar_diff_coeff_);
  if (numerical_integ_) {
    lkd_blfi->SetIntRule(&ir_di);
  }
  Lk_form_->AddDomainIntegrator(lkd_blfi);
  Lk_form_->Assemble();
  Lk_form_->FormSystemMatrix(empty, Lk_);  

  // testing...
  /*
  Lf_form_ = new ParBilinearForm(sfes_);
  auto *lfd_blfi = new DiffusionIntegrator(*tls2_coeff_);
  if (numerical_integ_) {
    lfd_blfi->SetIntRule(&ir_di);
  }
  Lf_form_->AddDomainIntegrator(lfd_blfi);
  Lf_form_->Assemble();
  Lf_form_->FormSystemMatrix(empty, Lf_);
  */
  
  // Helmholtz operators for lhs of all scalars  
  Hk_form_ = new ParBilinearForm(sfes_);
  auto *hmk_blfi = new MassIntegrator(*tke_diag_coeff_);
  //auto *hmk_blfi = new MassIntegrator(*rhoDt_coeff_);  
  auto *hdk_blfi = new DiffusionIntegrator(*tke_diff_total_coeff_);
  if (numerical_integ_) {
    hmk_blfi->SetIntRule(&ir_di);
    hdk_blfi->SetIntRule(&ir_i);
  }
  Hk_form_->AddDomainIntegrator(hmk_blfi);
  Hk_form_->AddDomainIntegrator(hdk_blfi);
  Hk_form_->Assemble();
  Hk_form_->FormSystemMatrix(tke_ess_tdof_, Hk_);

  He_form_ = new ParBilinearForm(sfes_);
  auto *hme_blfi = new MassIntegrator(*tdr_diag_coeff_);
  //auto *hme_blfi = new MassIntegrator(*rhoDt_coeff_);
  auto *hde_blfi = new DiffusionIntegrator(*tdr_diff_total_coeff_);
  if (numerical_integ_) {
    hme_blfi->SetIntRule(&ir_di);
    hde_blfi->SetIntRule(&ir_i);
    //hde_blfi->SetIntRule(&ir_lump);
  }
  He_form_->AddDomainIntegrator(hme_blfi);
  He_form_->AddDomainIntegrator(hde_blfi);
  He_form_->Assemble();
  He_form_->FormSystemMatrix(tdr_ess_tdof_, He_);

  Hv_form_ = new ParBilinearForm(sfes_);
  auto *hmv_blfi = new MassIntegrator(*v2_diag_coeff_);
  //auto *hmv_blfi = new MassIntegrator(*rhoDt_coeff_);  
  auto *hdv_blfi = new DiffusionIntegrator(*tke_diff_total_coeff_); // NOTE: not an error
  if (numerical_integ_) {
    hmv_blfi->SetIntRule(&ir_di);
    hdv_blfi->SetIntRule(&ir_i);
  }
  Hv_form_->AddDomainIntegrator(hmv_blfi);
  Hv_form_->AddDomainIntegrator(hdv_blfi);
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(v2_ess_tdof_, Hv_);

  Hf_form_ = new ParBilinearForm(ffes_);
  // dividing all by L^2
  auto *hmf_blfi = new MassIntegrator(*f_diag_coeff_);
  auto *hdf_blfi = new DiffusionIntegrator(*unity_diff_coeff_);
  // div(L^2 df) - f =...
  //auto *hmf_blfi = new MassIntegrator(*unity_coeff_);
  //auto *hdf_blfi = new DiffusionIntegrator(*tls2_coeff_);
  if (numerical_integ_) {
    hmf_blfi->SetIntRule(&ir_dif);
    hdf_blfi->SetIntRule(&ir_if);
  }
  Hf_form_->AddDomainIntegrator(hmf_blfi);
  Hf_form_->AddDomainIntegrator(hdf_blfi);
  Hf_form_->Assemble();
  Hf_form_->FormSystemMatrix(fRate_ess_tdof_, Hf_);

  Hz_form_ = new ParBilinearForm(sfes_);
  auto *hmz_blfi = new MassIntegrator(*zeta_diag_coeff_);
  auto *hdz_blfi = new DiffusionIntegrator(*zeta_diff_total_coeff_);
  if (numerical_integ_) {
    hmz_blfi->SetIntRule(&ir_di);
    hdz_blfi->SetIntRule(&ir_di);
  }
  Hz_form_->AddDomainIntegrator(hmz_blfi);
  Hz_form_->AddDomainIntegrator(hdz_blfi);
  Hz_form_->Assemble();
  Hz_form_->FormSystemMatrix(fRate_ess_tdof_, Hz_);

  // boundary terms for tdr
  /*
  He_bdry_ = new ParLinearForm(sfes_);
  auto *he_bdry_lfi = new BoundaryNormalLFIntegrator(*tdr_wall_coeff_, 2, 1);  
  if (numerical_integ_) {
    he_bdry_lfi->SetIntRule(&ir_di);
  }
  He_bdry_->AddBoundaryIntegrator(he_bdry_lfi, tdr_ess_attr_);
  if (rank0_) std::cout << "... check 8 ..." << endl;
  */
  
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
    MsRho_form_->AssembleDiagonal(diag_pa);
    MsRhoInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsRhoInvPC_ = new HypreSmoother(*MsRho_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsRhoInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  MsRhoInv_ = new CGSolver(sfes_->GetComm());
  MsRhoInv_->iterative_mode = false;
  MsRhoInv_->SetOperator(*Ms_);
  MsRhoInv_->SetPreconditioner(*MsInvPC_);
  MsRhoInv_->SetPrintLevel(pl_solve_);
  MsRhoInv_->SetRelTol(rtol_);
  MsRhoInv_->SetMaxIter(max_iter_);
  
  HkInvPC_ = new HypreSmoother(*Hk_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HkInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  HkInv_ = new CGSolver(sfes_->GetComm());
  HkInv_->iterative_mode = true;
  HkInv_->SetOperator(*Hk_);
  HkInv_->SetPreconditioner(*HkInvPC_);
  HkInv_->SetPrintLevel(pl_solve_);
  HkInv_->SetRelTol(rtol_);
  HkInv_->SetMaxIter(max_iter_);

  HeInvPC_ = new HypreSmoother(*He_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HeInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  HeInv_ = new CGSolver(sfes_->GetComm());
  HeInv_->iterative_mode = true;
  HeInv_->SetOperator(*He_);
  HeInv_->SetPreconditioner(*HeInvPC_);
  HeInv_->SetPrintLevel(pl_solve_);
  HeInv_->SetRelTol(rtol_);
  HeInv_->SetMaxIter(max_iter_);

  HvInvPC_ = new HypreSmoother(*Hv_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HvInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  HvInv_ = new CGSolver(sfes_->GetComm());
  HvInv_->iterative_mode = true;
  HvInv_->SetOperator(*Hv_);
  HvInv_->SetPreconditioner(*HvInvPC_);
  HvInv_->SetPrintLevel(pl_solve_);
  HvInv_->SetRelTol(rtol_);
  HvInv_->SetMaxIter(max_iter_);

  HfInvPC_ = new HypreSmoother(*Hf_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HfInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  HfInv_ = new CGSolver(ffes_->GetComm());
  // HfInv_ = new GMRESSolver(sfes_->GetComm());
  HfInv_->iterative_mode = true;
  HfInv_->SetOperator(*Hf_);
  HfInv_->SetPreconditioner(*HfInvPC_);
  HfInv_->SetPrintLevel(pl_solve_);
  HfInv_->SetRelTol(f_rtol_);
  HfInv_->SetMaxIter(f_max_iter_);

  HzInvPC_ = new HypreSmoother(*Hz_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HzInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  HzInv_ = new CGSolver(sfes_->GetComm());
  HzInv_->iterative_mode = true;
  HzInv_->SetOperator(*Hz_);
  HzInv_->SetPreconditioner(*HzInvPC_);
  HzInv_->SetPrintLevel(pl_solve_);
  HzInv_->SetRelTol(rtol_);
  HzInv_->SetMaxIter(max_iter_);  
  
  if (rank0_) std::cout << "zeta-f operators set" << endl;
}

void ZetaModel::initializeIO(IODataOrganizer &io) {
  io.registerIOFamily("tke", "/tke", &tke_gf_, true, true, sfec_);
  io.registerIOVar("/tke", "tke", 0);
  io.registerIOFamily("tdr", "/tdr", &tdr_gf_, true, true, sfec_);
  io.registerIOVar("/tdr", "tdr", 0);  
  io.registerIOFamily("v2", "/v2", &v2_gf_, true, true, sfec_);
  io.registerIOVar("/v2", "v2", 0);  
  io.registerIOFamily("zeta", "/zeta", &zeta_gf_, true, true, sfec_);  
  io.registerIOVar("/zeta", "zeta", 0);

  // should not be necessary with a computeMuT (and strain/tts) call in setup, but
  // does not seem to works
  io.registerIOFamily("eddy viscosity", "/muT", &eddyVisc_gf_, true, true, sfec_);  
  io.registerIOVar("/muT", "muT", 0);  
}

void ZetaModel::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("Pk", &prod_gf_);
  pvdc.RegisterField("TTS", &tts_gf_);
  pvdc.RegisterField("TLS", &tls_gf_);
  pvdc.RegisterField("|S|", &sMag_gf_);      
  pvdc.RegisterField("muT", &eddyVisc_gf_);
  pvdc.RegisterField("TKE", &tke_gf_);
  pvdc.RegisterField("TDR", &tdr_gf_);
  pvdc.RegisterField("v2", &v2_gf_);  
  pvdc.RegisterField("zeta", &zeta_gf_);
  pvdc.RegisterField("f", &fRate_gf_);
}

void ZetaModel::setup() {
  
  // populate  registers after restarts
  tke_gf_.GetTrueDofs(tke_);
  tdr_gf_.GetTrueDofs(tdr_);
  zeta_gf_.GetTrueDofs(zeta_);
  v2_gf_.GetTrueDofs(v2_);  
  
  tke_next_gf_.SetFromTrueDofs(tke_);
  tdr_next_gf_.SetFromTrueDofs(tdr_);
  zeta_next_gf_.SetFromTrueDofs(zeta_);
  v2_next_gf_.SetFromTrueDofs(v2_);  

  tke_next_gf_.GetTrueDofs(tke_next_);
  tdr_next_gf_.GetTrueDofs(tdr_next_);
  zeta_next_gf_.GetTrueDofs(zeta_next_);
  v2_next_gf_.GetTrueDofs(v2_next_);

  // call twice to fill both {n-1} and {n-2} 
  updateTimestepHistory();
  updateTimestepHistory();
  
  // initial mu_t
  /*
  computeStrain();
  updateTTS();
  updateMuT();
  */

}

void ZetaModel::step() {
  // update time
  dt_ = time_coeff_.dt;
  time_ = time_coeff_.time;

  // gather necessary information from other classes
  (flow_interface_->velocity)->GetTrueDofs(vel_);
  (flow_interface_->gradU)->GetTrueDofs(gradU_);
  (flow_interface_->gradV)->GetTrueDofs(gradV_);
  (flow_interface_->gradW)->GetTrueDofs(gradW_);
  (thermoChem_interface_->density)->GetTrueDofs(rho_);
  (thermoChem_interface_->viscosity)->GetTrueDofs(mu_);
  (sponge_interface_->diff_multiplier)->GetTrueDofs(mult_);  

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
  //updateBC();      
  extrapolateState();
  computeStrain();
  updateTTS();
  updateTLS();    
  updateProd();
  updateMsRho();
  extrapolateRHS();  

  // TKE step
  tkeStep();

  // TDR step
  tdrStep();

  // f-Rate
  fStep();

  // zeta or v2 step
  // zetaStep();
  v2Step();
  updateZeta();
  
  // final calc of eddy visc at {n+1}
  updateTTS();  
  updateMuT();

  // rotate storage
  updateTimestepHistory();  
}

void ZetaModel::updateMuT() {
  eddyVisc_.Set(Cmu_,rho_);
  {
    double twoThirds = 2.0/3.0;
    const double *dv2 = v2_next_.HostRead();
    const double *dk = tke_next_.HostRead();        
    const double *dTTS = tts_.HostRead();
    const double *dTTS_strain = tts_strain_.HostRead();
    double *muT = eddyVisc_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) muT[i] *= std::min(dv2[i], twoThirds*dk[i]);
    for (int i = 0; i < SdofInt_; i++) muT[i] *= std::min(dTTS[i], dTTS_strain[i]);
    for (int i = 0; i < SdofInt_; i++) muT[i] = std::max(muT[i], mut_min_);
  }  
  eddyVisc_gf_.SetFromTrueDofs(eddyVisc_);
  
  resf_gf_.ProjectGridFunction(eddyVisc_gf_);
  eddyVisc_gf_.ProjectGridFunction(resf_gf_);  
  eddyVisc_gf_.GetTrueDofs(eddyVisc_);
  
}

void ZetaModel::computeStrain() {
  const double *dGradU = gradU_.HostRead();
  const double *dGradV = gradV_.HostRead();
  const double *dGradW = gradW_.HostRead();
  double *Sij = strain_.HostReadWrite();
  double *dSmag = sMag_.HostReadWrite();
  double Smin = 1.0e-14;

  for (int i = 0; i < SdofInt_; i++) {
    DenseMatrix gradUp;
    gradUp.SetSize(nvel_, dim_);
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(0, dir) = dGradU[i + dir * SdofInt_];
    }
    for (int dir = 0; dir < dim_; dir++) {
      gradUp(1, dir) = dGradV[i + dir * SdofInt_];
    }
    if(dim_ == 3) {
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(2, dir) = dGradW[i + dir * SdofInt_];
      }
    }

    Sij[i + 0 * SdofInt_] = gradUp(0, 0);
    Sij[i + 1 * SdofInt_] = gradUp(1, 1);
    if(dim_ == 3) {
      Sij[i + 2 * SdofInt_] = gradUp(2, 2);
      Sij[i + 3 * SdofInt_] = 0.5 * (gradUp(0, 1) + gradUp(1, 0));
      Sij[i + 4 * SdofInt_] = 0.5 * (gradUp(0, 2) + gradUp(2, 0));
      Sij[i + 5 * SdofInt_] = 0.5 * (gradUp(1, 2) + gradUp(2, 1));      
    } else {
      Sij[i + 2 * SdofInt_] = 0.5 * (gradUp(0, 1) + gradUp(1, 0));
    }
  }

  // NOTE:  not including sqrt(2) factor here
  for (int i = 0; i < SdofInt_; i++) {
    dSmag[i] = 0.0;
    if(dim_ == 3) {
      for (int j = 0; j < dim_; j++) dSmag[i] += Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];
      for (int j = dim_; j < 2*dim_; j++) dSmag[i] += 2.0 * Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];      
    } else {
      for (int j = 0; j < dim_; j++) dSmag[i] += Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];
      for (int j = dim_; j < 3; j++) dSmag[i] += 2.0 * Sij[i + j * SdofInt_] * Sij[i + j * SdofInt_];      
    }
    dSmag[i] = sqrt(std::max(dSmag[i],0.0));
    dSmag[i] = std::max(dSmag[i], Smin);
  }
  sMag_gf_.SetFromTrueDofs(sMag_);
}

void ZetaModel::updateTTS() {
  const double *dTKE = tke_next_.HostRead();
  const double *dTDR = tdr_next_.HostRead();
  const double *dv2 = v2_next_.HostRead();  
  // const double *dZeta = zeta_next_.HostRead();
  const double *dSmag = sMag_.HostRead();
  const double *dMu = mu_.HostRead();
  const double *dRho = rho_.HostRead();
  double *dTTS = tts_.HostReadWrite();
  double *dTTS_strain = tts_strain_.HostReadWrite();
  double *dTTS_kol = tts_kol_.HostReadWrite();  

  double Ctime;
  Ctime = 0.6 / (std::sqrt(6.0) * Cmu_);
  for (int i = 0; i < SdofInt_; i++) {
    double T1, T2, T3;
    T1 = dTKE[i] / std::max(dTDR[i], tdr_min_);
    T2 = Ctime * dTKE[i] / (dSmag[i] * std::max(dv2[i], v2_min_));
    T3 = Ct_ * std::sqrt( std::max(dMu[i] / (dRho[i] * std::max(dTDR[i], tdr_min_)),0.0) );

    // zeta ordering
    //dTTS[i] = std::min(T1, T2);
    //dTTS[i] = std::max(dTTS[i], T3);

    // order consistent with "code-friendly" version
    //dTTS[i] = std::max(T1, T3);
    //dTTS[i] = std::min(dTTS[i], T2);    
    //dTTS[i] = std::max(dTTS[i], tts_min_);
    //dTTS[i] = std::min(dTTS[i], tts_max_);

    // restrain source terms
    dTTS[i] = std::max(T1, T3);
    // dTTS[i] = std::min(T1, T2);    
    dTTS[i] = std::max(dTTS[i], tts_min_);
    dTTS[i] = std::min(dTTS[i], tts_max_);
    dTTS_kol[i] = T3;    
    dTTS_strain[i] = T2;
    dTTS_strain[i] = std::max(dTTS_strain[i], tts_min_);
    dTTS_strain[i] = std::min(dTTS_strain[i], tts_max_);        
  }
  tts_gf_.SetFromTrueDofs(tts_);
}

void ZetaModel::updateTLS() {
  const double *dTKE = tke_next_.HostRead();
  const double *dTDR = tdr_next_.HostRead();
  const double *dv2 = v2_next_.HostRead();    
  // const double *dZeta = zeta_next_.HostRead();
  const double *dSmag = sMag_.HostRead();
  const double *dMu = mu_.HostRead();
  const double *dRho = rho_.HostRead();
  double *dTLS = tls_.HostReadWrite();

  double Clength;
  Clength = 1.0 / (std::sqrt(6.0) * Cmu_);
  for (int i = 0; i < SdofInt_; i++) {
    double L1, L2, L3;
    L1 = std::pow(dTKE[i], 1.5) / std::max(dTDR[i], tdr_min_);
    L2 = Clength * std::pow(dTKE[i], 1.5) / (dSmag[i] * std::max(dv2[i], v2_min_));
    L3 = Cn_ * std::pow((std::pow(dMu[i] / dRho[i], 3.0) / std::max(dTDR[i], tdr_min_)), 0.25);    
    dTLS[i] = std::min(L1, L2);
    dTLS[i] = Cl_ * std::max(dTLS[i], L3);    
    dTLS[i] = std::max(dTLS[i], tls_min_);
    dTLS[i] = std::min(dTLS[i], tls_max_);  
  }
  tls_gf_.SetFromTrueDofs(tls_);
  
  // for convience in assembling coeffs
  tmpR0_.Set(1.0,tls_);
  tmpR0_ *= tls_;
  res_gf_.SetFromTrueDofs(tmpR0_);
  tls2_gf_.ProjectGridFunction(res_gf_);
  tls2_gf_.GetTrueDofs(tls2_);    
}

void ZetaModel::updateMsRho() {
  Array<int> empty;
  MsRho_form_->Update();
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);
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
    if(dim_ == 3) {
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(2, dir) = dGradW[i + dir * SdofInt_];
      }
    }

    double divU = 0.0;
    for (int j = 0; j < dim_; j++) divU += gradUp(j, j);

    DenseMatrix tau;
    tau.SetSize(nvel_, dim_);

    if(dim_ == 3) {
      tau(0, 0) = 2.0 * Sij[i + 0 * SdofInt_];
      tau(1, 1) = 2.0 * Sij[i + 1 * SdofInt_];
      tau(2, 2) = 2.0 * Sij[i + 2 * SdofInt_];
      tau(0, 1) = 2.0 * Sij[i + 3 * SdofInt_];
      tau(1, 0) = 2.0 * Sij[i + 3 * SdofInt_];
      tau(0, 2) = 2.0 * Sij[i + 4 * SdofInt_];
      tau(1, 2) = 2.0 * Sij[i + 5 * SdofInt_];
      tau(2, 0) = 2.0 * Sij[i + 4 * SdofInt_];
      tau(2, 1) = 2.0 * Sij[i + 5 * SdofInt_];    
    } else {
      tau(0, 0) = 2.0 * Sij[i + 0 * SdofInt_];
      tau(1, 1) = 2.0 * Sij[i + 1 * SdofInt_];
      tau(0, 1) = 2.0 * Sij[i + 2 * SdofInt_];
      tau(1, 0) = 2.0 * Sij[i + 2 * SdofInt_];      
    }

    // for (int j = 0; j < dim_; j++) {
    //  tau(j, j) -= twoThirds * divU;
    // }
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

    Pk[i] = std::max(Pk[i], 0.0);
    
  }

  prod_gf_.SetFromTrueDofs(prod_);

  resf_gf_.ProjectGridFunction(prod_gf_);
  prod_gf_.ProjectGridFunction(resf_gf_);  
  prod_gf_.GetTrueDofs(prod_);
  
}

/// extrapolated states to {n+1}
void ZetaModel::extrapolateState() {
  tke_next_.Set(time_coeff_.ab1, tke_);
  tke_next_.Add(time_coeff_.ab2, tke_nm1_);
  tke_next_.Add(time_coeff_.ab3, tke_nm2_);
  tke_next_gf_.GetTrueDofs(tke_next_);    

  tdr_next_.Set(time_coeff_.ab1, tdr_);
  tdr_next_.Add(time_coeff_.ab2, tdr_nm1_);
  tdr_next_.Add(time_coeff_.ab3, tdr_nm2_);
  tdr_next_gf_.GetTrueDofs(tdr_next_);      

  zeta_next_.Set(time_coeff_.ab1, zeta_);
  zeta_next_.Add(time_coeff_.ab2, zeta_nm1_);
  zeta_next_.Add(time_coeff_.ab3, zeta_nm2_);
  zeta_next_gf_.GetTrueDofs(zeta_next_);        

  v2_next_.Set(time_coeff_.ab1, v2_);
  v2_next_.Add(time_coeff_.ab2, v2_nm1_);
  v2_next_.Add(time_coeff_.ab3, v2_nm2_);
  v2_next_gf_.GetTrueDofs(v2_next_);        
}

void ZetaModel::extrapolateRHS() {
  prod_next_.Set(time_coeff_.ab1, prod_);
  prod_next_.Add(time_coeff_.ab2, prod_nm1_);
  prod_next_.Add(time_coeff_.ab3, prod_nm2_);
  prod_next_gf_.SetFromTrueDofs(prod_next_);

  tts_next_.Set(time_coeff_.ab1, tts_);
  tts_next_.Add(time_coeff_.ab2, tts_nm1_);
  tts_next_.Add(time_coeff_.ab3, tts_nm2_);
  tts_next_gf_.SetFromTrueDofs(tts_next_);  

  tls2_next_.Set(time_coeff_.ab1, tls2_);
  tls2_next_.Add(time_coeff_.ab2, tls2_nm1_);
  tls2_next_.Add(time_coeff_.ab3, tls2_nm2_);
  tls2_next_gf_.SetFromTrueDofs(tls2_next_);  
}

void ZetaModel::updateZeta() {
  zeta_next_.Set(1.0, v2_next_);
  {
    const double *dk = tke_next_.HostRead();
    double *dz = zeta_next_.HostReadWrite();    
    for (int i = 0; i < SdofInt_; i++) {  
      dz[i] /= std::max(dk[i], tke_min_);
    }
  }
  zeta_next_gf_.SetFromTrueDofs(zeta_next_);
}

void ZetaModel::updateTimestepHistory() {

  dt_nm2_ = dt_nm1_;
  dt_nm1_ = dt_;  
  
  // nonlinear products
  Ntke_nm2_ = Ntke_nm1_;
  Ntke_nm1_ = Ntke_;

  Ntdr_nm2_ = Ntdr_nm1_;
  Ntdr_nm1_ = Ntdr_;

  Nzeta_nm2_ = Nzeta_nm1_;
  Nzeta_nm1_ = Nzeta_;

  Nv2_nm2_ = Nv2_nm1_;
  Nv2_nm1_ = Nv2_;

  // rhs terms
  prod_nm2_ = prod_nm1_;
  prod_nm1_ = prod_;  

  tts_nm2_ = tts_nm1_;
  tts_nm1_ = tts_;  

  tls2_nm2_ = tls2_nm1_;
  tls2_nm1_ = tls2_;    
  
  // scalars
  tke_nm2_ = tke_nm1_;
  tke_nm1_ = tke_;

  tdr_nm2_ = tdr_nm1_;
  tdr_nm1_ = tdr_;

  zeta_nm2_ = zeta_nm1_;
  zeta_nm1_ = zeta_;

  v2_nm2_ = v2_nm1_;
  v2_nm1_ = v2_;  
  
  // copy previous {n+1} to {n}
  tke_next_gf_.GetTrueDofs(tke_next_);
  tke_.Set(1.0,tke_next_);
  tke_gf_.SetFromTrueDofs(tke_);

  tdr_next_gf_.GetTrueDofs(tdr_next_);
  tdr_.Set(1.0,tdr_next_);
  tdr_gf_.SetFromTrueDofs(tdr_);

  zeta_next_gf_.GetTrueDofs(zeta_next_);
  zeta_.Set(1.0,zeta_next_);
  zeta_gf_.SetFromTrueDofs(zeta_);

  v2_next_gf_.GetTrueDofs(v2_next_);
  v2_.Set(1.0,v2_next_);
  v2_gf_.SetFromTrueDofs(v2_);  
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
  } else if (scalar == "v2") {
    As_->Mult(v2_, Nv2_);
    tmpR0_.Set(time_coeff_.ab1, Nv2_);
    tmpR0_.Add(time_coeff_.ab2, Nv2_nm1_);
    tmpR0_.Add(time_coeff_.ab3, Nv2_nm2_);    
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
  Ms_->AddMult(prod_next_, res_, +1.0);

  // destruction => 1/2 included in lhs
  tmpR0_.Set((1.0 - des_wgt_), tdr_next_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // Update Helmholtz operator  
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  
  Hk_form_->Update();
  Hk_form_->Assemble();
  Hk_form_->FormSystemMatrix(tke_ess_tdof_, Hk_);
  HkInv_->SetOperator(*Hk_);
  if (partial_assembly_) {
    delete HkInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hk_form_->AssembleDiagonal(diag_pa);
    HkInvPC_ = new OperatorJacobiSmoother(diag_pa, tke_ess_tdof_);
    HkInv_->SetPreconditioner(*HkInvPC_);
  }

  // Prepare for the solve
  for (auto &tke_dbc : tke_dbcs_) {
    tke_next_gf_.ProjectBdrCoefficient(*tke_dbc.coeff, tke_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  Hk_form_->FormLinearSystem(tke_ess_tdof_, tke_next_gf_, res_gf_, Hk_, Xt2, Bt2, 1);    

  // solve helmholtz eq for temp
  HkInv_->Mult(Bt2, Xt2);
  assert(HkInv_->GetConverged());

  Hk_form_->RecoverFEMSolution(Xt2, res_gf_, tke_next_gf_);
  tke_next_gf_.GetTrueDofs(tke_next_);

  // hard-clip
  {
    double *dTKE = tke_next_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      dTKE[i] = std::max(dTKE[i], 0.0);
    }
  }
  tke_next_gf_.SetFromTrueDofs(tke_next_);
}

void ZetaModel::tdrStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection (->tmpR0_)
  convection("tdr");
  res_.Add(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, tdr_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, tdr_nm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, tdr_nm2_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // production
  tmpR0_.Set(1.0, prod_next_);
  tmpR0_ /= tts_next_;
  {
    double twoThirds = 2.0/3.0;
    double ceps1_min;
    ceps1_min = 1.4 * (1.0 + 0.05 * std::sqrt(1.5));
    //const double *dk = tke_next_.HostRead();    
    //const double *dv2 = v2_next_.HostRead();
    const double *dz = zeta_next_.HostRead();    
    double *data = tmpR0_.HostReadWrite();  
    for (int i = 0; i < SdofInt_; i++) {
      // zeta model
      //double ceps1 = 1.4 * (1.0 + 0.012 / dz[i]);

      // code-friendly
      //double vtmp, ktmp;
      //ktmp = std::max(dk[i], 0.0); // if using extrapolation must guard against very small negatives in sqrt
      //vtmp = std::max(dv2[i], v2_min_);
      //vtmp = std::min(vtmp, twoThirds*ktmp);
      //double ceps1 = 1.4 * (1.0 + 0.05 * std::sqrt(ktmp/vtmp));
      double ceps1 = 1.4 * (1.0 + 0.05 / std::sqrt(std::max(dz[i], zeta_min_)));
      ceps1 = std::min(ceps1, 1.55);
      ceps1 = std::max(ceps1, ceps1_min);
    
      data[i] *= ceps1;
    }
  }
  Ms_->AddMult(tmpR0_, res_, +1.0);

  // destruction => 1/2 included in lhs
  tmpR0_.Set((1.0 - des_wgt_) * Ce2_, tdr_next_);
  tmpR0_ /= tts_next_;
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // Update Helmholtz operator
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  diag_coeff_ = tdr_diag_coeff_;
  diff_total_coeff_ = tdr_diff_total_coeff_;

  // boundary condition
  Array<int> empty;
  Lk_form_->Update();
  Lk_form_->Assemble();
  //Lk_form_->FormSystemMatrix(tke_ess_tdof_, Lk_);
  Lk_form_->FormSystemMatrix(empty, Lk_);  
  Lk_->Mult(tke_next_, tmpR0_);
  MsRhoInv_->Mult(tmpR0_, tmpR0a_);
  tmpR0a_ *= -1.0;  
  tdr_wall_gf_.SetFromTrueDofs(tmpR0a_);

  He_form_->Update();
  He_form_->Assemble();
  He_form_->FormSystemMatrix(tdr_ess_tdof_, He_);

  HeInv_->SetOperator(*He_);
  if (partial_assembly_) {
    delete HeInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    He_form_->AssembleDiagonal(diag_pa);
    HeInvPC_ = new OperatorJacobiSmoother(diag_pa, tdr_ess_tdof_);
    HeInv_->SetPreconditioner(*HeInvPC_);
  }

  // project new tdr bc onto gf which transfers actual ess bc's to solver
  for (auto &tdr_dbc : tdr_dbcs_) {
    // tdr_next_gf_.ProjectBdrCoefficient(*tdr_dbc.coeff, tdr_dbc.attr);
    tdr_next_gf_.ProjectBdrCoefficient(*tdr_wall_eval_coeff_, tdr_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  He_form_->FormLinearSystem(tdr_ess_tdof_, tdr_next_gf_, res_gf_, He_, Xt2, Bt2, 1);
  
  // solve helmholtz eq for temp
  HeInv_->Mult(Bt2, Xt2);
  assert(HeInv_->GetConverged());

  He_form_->RecoverFEMSolution(Xt2, res_gf_, tdr_next_gf_);
  tdr_next_gf_.GetTrueDofs(tdr_next_);

  // hard-clip
  double *dTDR = tdr_next_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dTDR[i] = std::max(dTDR[i], 0.0);
  }
  tdr_next_gf_.SetFromTrueDofs(tdr_next_);
}

void ZetaModel::zetaStep() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection
  convection("zeta");  // ->tmpR0_
  res_.Add(-1.0, tmpR0_);

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
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);

  Hz_form_->Update();
  Hz_form_->Assemble();
  Hz_form_->FormSystemMatrix(zeta_ess_tdof_, Hz_);

  HzInv_->SetOperator(*Hz_);
  if (partial_assembly_) {
    delete HzInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hz_form_->AssembleDiagonal(diag_pa);
    HzInvPC_ = new OperatorJacobiSmoother(diag_pa, zeta_ess_tdof_);
    HzInv_->SetPreconditioner(*HzInvPC_);
  }

  // Prepare for the solve
  for (auto &zeta_dbc : zeta_dbcs_) {
    zeta_next_gf_.ProjectBdrCoefficient(*zeta_dbc.coeff, zeta_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  Hz_form_->FormLinearSystem(zeta_ess_tdof_, zeta_next_gf_, res_gf_, Hz_, Xt2, Bt2, 1);

  // solve helmholtz eq for temp
  HzInv_->Mult(Bt2, Xt2);
  assert(HzInv_->GetConverged());

  Hz_form_->RecoverFEMSolution(Xt2, res_gf_, zeta_next_gf_);
  zeta_next_gf_.GetTrueDofs(zeta_next_);

  // hard-clip
  double *dZ = zeta_next_.HostReadWrite();
  for (int i = 0; i < SdofInt_; i++) {
    dZ[i] = std::max(dZ[i], zeta_min_);
    dZ[i] = std::min(dZ[i], 2.0/3.0);    
  }
  zeta_next_gf_.SetFromTrueDofs(zeta_next_);
}

void ZetaModel::v2Step() {
  // Build the right-hand-side
  res_ = 0.0;

  // convection
  convection("v2");  // ->tmpR0_
  res_.Add(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, v2_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, v2_nm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, v2_nm2_);
  MsRho_->AddMult(tmpR0_, res_, -1.0);

  // production
  tmpR0_.Set(1.0, tke_next_);
  // tmpR0_ *= fRate_;
  res_gf_.ProjectGridFunction(fRate_gf_);
  res_gf_.GetTrueDofs(tmpR0a_);
  {
    const double *df = tmpR0a_.HostRead();
    const double *dtkol = tts_kol_.HostRead();
    double *data = tmpR0_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      // data[i] *= std::min(df[i], 12.0/dtkol[i]);
      data[i] *= df[i];
    }
  }
  MsRho_->AddMult(tmpR0_, res_, +1.0);
  
  // destruction => 1/2 included in lhs
  tmpR0_.Set(6.0*(1.0-des_wgt_), tdr_next_);
  tmpR0_ *= zeta_;
  MsRho_->AddMult(tmpR0_, res_, -1.0);  

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt_gf_ = *(thermoChem_interface_->density);
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
  
  Hv_form_->Update();
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(v2_ess_tdof_, Hv_);

  HvInv_->SetOperator(*Hv_);
  if (partial_assembly_) {
    delete HvInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hv_form_->AssembleDiagonal(diag_pa);
    HvInvPC_ = new OperatorJacobiSmoother(diag_pa, v2_ess_tdof_);
    HvInv_->SetPreconditioner(*HvInvPC_);
  }

  // Prepare for the solve
  for (auto &v2_dbc : v2_dbcs_) {
    v2_next_gf_.ProjectBdrCoefficient(*v2_dbc.coeff, v2_dbc.attr);    
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(res_, res_gf_);

  Vector Xt2, Bt2;
  Hv_form_->FormLinearSystem(v2_ess_tdof_, v2_next_gf_, res_gf_, Hv_, Xt2, Bt2, 1);

  // solve helmholtz eq for temp
  HvInv_->Mult(Bt2, Xt2);
  assert(HvInv_->GetConverged());

  Hv_form_->RecoverFEMSolution(Xt2, res_gf_, v2_next_gf_);
  v2_next_gf_.GetTrueDofs(v2_next_);

  // hard-clip
  {
    const double *dtke = tke_next_.HostReadWrite();  
    double *dv2 = v2_next_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      dv2[i] = std::max(dv2[i], 0.0);
      // clip when used in certain areas, allow to be above 2/3k for destruction in v2
      // dv2[i] = std::min(dv2[i], 2.0/3.0 * dtke[i]);
      if(dv2[i] != dv2[i]) std::cout << " v2 is actually NaN!" << endl;
    }
  }
  v2_next_gf_.SetFromTrueDofs(v2_next_);
}

void ZetaModel::fStep() {
  // Build the right-hand-side
  res_ = 0.0;
  resf_ = 0.0;  

  // assemble source
  /*
  tmpR0b_.Set(C2_, prod_);
  tmpR0b_ /= rho_;
  tmpR0b_ /= tdr_;
  tmpR0b_ += (C1_ - 1.0);

  tmpR0a_.Set(1.0, zeta_);
  tmpR0a_ -= 2.0 / 3.0;
  tmpR0a_ *= tmpR0b_;
  tmpR0a_ /= tts_;
  tmpR0a_ /= tls2_;
  */

  // code-friendly form
  // rhs: 1/T * [(C1-6)*v2/k - 2/3*(C1-1)] - C2*Pk/(rho*k)
  double Ctmp1, Ctmp2;
  Ctmp1 = 2.0/3.0 * (C1_ - 1.0);
  Ctmp2 = (C1_ - 6.0);

  // C2*Pk/(rho*k)  
  tmpR0b_.Set(C2_, prod_);
  tmpR0b_ /= rho_;
  {
    const double *dk = tke_next_.HostRead();
    double *data = tmpR0b_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {  
      data[i] /= std::max(dk[i], tke_min_);
    }
  }
  
  // v2/k
  {
    double twoThirds = 2.0/3.0;
    const double *dv2 = v2_next_.HostRead();    
    const double *dk = tke_next_.HostRead();
    double *data = tmpR0a_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {  
      data[i] = std::min(dv2[i], twoThirds * dk[i]);
    }    
    for (int i = 0; i < SdofInt_; i++) {  
      data[i] /= std::max(dk[i], tke_min_);
    }
  }
  tmpR0a_ *= Ctmp2;
  tmpR0a_ -= Ctmp1;
  tmpR0a_ /= tts_;
  tmpR0a_ -= tmpR0b_;

  // limiter-term
  /*
  {
    double twoThirds = 2.0/3.0;
    double tmp;
    const double *dz = zeta_next_.HostRead();
    double *data = tmpR0c_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      tmp = dz[i] - twoThirds;
      tmp = std::max(tmp, 0.0);
      tmp *= (C1_ - 1.0);
      data[i] = tmp;
    }    
  }
  tmpR0c_ /= tts_;
  tmpR0a_ += tmpR0c_;
  {
    double *data = tmpR0a_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      data[i] = std::min(data[i], 0.0);
    }    
  }
  */

  //HACK limit
  /*
  {
    double twoThirds = 2.0/3.0;
    double Pv2_max;
    const double *dP = prod_.HostRead();    
    const double *de = tdr_.HostRead();
    const double *dv2 = v2_.HostRead();
    const double *dk = tke_.HostRead();        
    const double *drho = rho_.HostRead();    
    double *data = tmpR0a_.HostReadWrite();
    for (int i = 0; i < SdofInt_; i++) {
      Pv2_max = twoThirds * (dP[i]/drho[i] - de[i]);
      // Pv2_max += 6.0 * (dv2[i]/std::max(dk[i], tke_min_)) * de[i];
      Pv2_max += 6.0 * twoThirds * de[i];      
      data[i] = - Pv2_max;
    }
  }
  */

  // minus is to (-) on operator
  //Ms_->AddMult(tmpR0a_, res_, -1.0);

  // if not including L2 in laplacian term
  tmpR0a_ /= tls2_; 
  
  // project to f-space
  res_gf_.SetFromTrueDofs(tmpR0a_);
  resf_gf_.ProjectGridFunction(res_gf_);
  resf_gf_.GetTrueDofs(ftmpR0_);
  
  Mf_->AddMult(ftmpR0_, resf_, -1.0);
  resf_gf_.SetFromTrueDofs(resf_);  
  
  // Update Helmholtz operator  
  Hf_form_->Update();
  Hf_form_->Assemble();
  Hf_form_->FormSystemMatrix(fRate_ess_tdof_, Hf_);

  HfInv_->SetOperator(*Hf_);
  if (partial_assembly_) {
    delete HfInvPC_;
    Vector diag_pa(ffes_->GetTrueVSize());
    Hf_form_->AssembleDiagonal(diag_pa);
    HfInvPC_ = new OperatorJacobiSmoother(diag_pa, fRate_ess_tdof_);
    HfInv_->SetPreconditioner(*HfInvPC_);
  }

  // Prepare for the solve
  for (auto &fRate_dbc : fRate_dbcs_) {
    fRate_gf_.ProjectBdrCoefficient(*fRate_dbc.coeff, fRate_dbc.attr);
  }
  ffes_->GetRestrictionMatrix()->MultTranspose(resf_, resf_gf_);

  Vector Xt2, Bt2;
  Hf_form_->FormLinearSystem(fRate_ess_tdof_, fRate_gf_, resf_gf_, Hf_, Xt2, Bt2, 1);

  // solve helmholtz eq for temp
  HfInv_->Mult(Bt2, Xt2);
  assert(HfInv_->GetConverged());

  Hf_form_->RecoverFEMSolution(Xt2, resf_gf_, fRate_gf_);
  fRate_gf_.GetTrueDofs(fRate_);

  // hard-clip
  {
    double *df = fRate_.HostReadWrite();
    for (int i = 0; i < ffes_->GetTrueVSize(); i++) {
      df[i] = std::max(df[i], 0.0);
      if(df[i] != df[i]) std::cout << " f is actually NaN!" << endl;      
    }
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

void ZetaModel::AddV2DirichletBC(const double &v2, Array<int> &attr) {
  v2_dbcs_.emplace_back(attr, new ConstantCoefficient(v2));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!v2_ess_attr_[i]);
      v2_ess_attr_[i] = 1;
    }
  }
}

void ZetaModel::AddV2DirichletBC(Coefficient *coeff, Array<int> &attr) {
  v2_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!v2_ess_attr_[i]);
      v2_ess_attr_[i] = 1;
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
