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

#include "algebraicSubgridModels.hpp"
#include "cases.hpp"
#include "externalData_base.hpp"
#include "io.hpp"
#include "loMach.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;

/// forward declarations
static double radius(const Vector &pos) { return pos[0]; }
FunctionCoefficient radius_coeff(radius);

static double negativeRadius(const Vector &pos) { return -pos[0]; }
FunctionCoefficient negative_radius_coeff(negativeRadius);

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
      tpsP_(tps),
      pmesh_(pmesh),
      vorder_(vorder),
      porder_(porder),
      dim_(pmesh->Dimension()),
      coeff_(coeff) {
  assert(pmesh_ != NULL);

  rank0_ = (pmesh_->GetMyRank() == 0);
  axisym_ = false;
  nvel_ = dim_;

  // make sure there is room for BC attributes
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    vel_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    vel_ess_attr_ = 0;

    pres_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    pres_ess_attr_ = 0;
  }

  if (tps != nullptr) {
    // if we have Tps object, use it to set other options...

    // Axisymmetric simulation?
    tps->getInput("loMach/axisymmetric", axisym_, false);

    if (axisym_) {
      assert(dim_ == 2);
      nvel_ = 3;
    }

    if (axisym_) {
      swirl_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
      swirl_ess_attr_ = 0;
    }

    // Use "numerical integration" (i.e., under-integrate so that mass matrix is diagonal)
    // NOTE: this should default to false as it is generally not safe, but much of the
    // test results rely on it being true
    tps->getInput("loMach/tomboulides/numerical-integ", numerical_integ_, true);

    // Can't use numerical integration with axisymmetric b/c it
    // locates quadrature points on the axis, which can lead to
    // singular mass matrices and evaluations of 1/r = 1/0.
    if (axisym_) assert(!numerical_integ_);

    // exposing solver tolerance options to user
    tps->getInput("loMach/tomboulides/psolve-atol", pressure_solve_atol_, default_atol_);
    tps->getInput("loMach/tomboulides/hsolve-atol", hsolve_atol_, default_atol_);
    tps->getInput("loMach/tomboulides/msolve-atol", mass_inverse_atol_, default_atol_);

    tps->getInput("loMach/tomboulides/psolve-rtol", pressure_solve_rtol_, default_rtol_);
    tps->getInput("loMach/tomboulides/hsolve-rtol", hsolve_rtol_, default_rtol_);
    tps->getInput("loMach/tomboulides/msolve-rtol", mass_inverse_rtol_, default_rtol_);

    tps->getInput("loMach/tomboulides/psolve-maxIters", pressure_solve_max_iter_, default_max_iter_);
    tps->getInput("loMach/tomboulides/hsolve-maxIters", hsolve_max_iter_, default_max_iter_);
    tps->getInput("loMach/tomboulides/msolve-maxIters", mass_inverse_max_iter_, default_max_iter_);
  }
}

Tomboulides::~Tomboulides() {
  // miscellaneous
  delete mass_lform_;

  // objects allocated by initializeOperators
  delete swirl_var_viscosity_form_;
  delete rho_ur_ut_form_;
  delete As_form_;
  delete Ms_rho_form_;
  delete Hs_form_;
  delete Hs_inv_;
  delete Hs_inv_pc_;
  delete ur_conv_axi_form_;
  delete Faxi_poisson_form_;
  delete S_mom_form_;
  delete S_poisson_form_;
  delete u_bdr_form_;
  delete pp_div_bdr_form_;
  delete Hv_inv_;
  delete Hv_inv_pc_;
  delete Hv_form_;
  delete Mv_rho_form_;
  delete G_form_;
  delete D_form_;
  delete Mv_rho_inv_;
  delete Mv_rho_inv_pc_;
  delete Mv_inv_;
  delete Mv_inv_pc_;
  delete Mv_form_;
  delete Ms_inv_;    
  delete Ms_form_;
  delete Nconv_form_;
  //delete Nconv_rho_form_;  
  delete forcing_form_;
  delete L_iorho_inv_;
  delete L_iorho_inv_ortho_pc_;
  delete L_iorho_inv_pc_;
  delete L_iorho_lor_;
  delete L_iorho_form_;

  delete swirl_var_viscosity_coeff_;
  delete utheta_vec_coeff_;
  delete rho_ur_ut_coeff_;
  delete ur_ut_coeff_;
  delete u_next_rad_coeff_;
  delete rad_rhou_coeff_;
  delete u_next_coeff_;
  delete ur_conv_forcing_coeff_;

  for (size_t i = 0; i < rad_vel_coeff_.size(); i++) delete rad_vel_coeff_[i];

  delete rad_pp_div_coeff_;
  delete visc_forcing_coeff_;
  delete rad_mu_coeff_;
  delete rad_rho_over_dt_coeff_;
  delete rad_rho_coeff_;
  delete rad_S_mom_coeff_;
  delete rad_S_poisson_coeff_;
  delete S_mom_coeff_;
  delete S_poisson_coeff_;
  delete gradmu_Qt_coeff_;
  delete Qt_coeff_;
  delete twoS_gradmu_coeff_;
  delete graduT_gradmu_coeff_;
  delete gradu_gradmu_coeff_;
  delete grad_u_next_transp_coeff_;
  delete grad_u_next_coeff_;
  delete grad_mu_coeff_;
  delete pp_div_rad_comp_coeff_;
  delete pp_div_coeff_;
  delete mu_coeff_;
  delete rho_over_dt_coeff_;
  delete iorho_coeff_;
  delete rho_coeff_;

  // objects allocated by initalizeSelf
  if (axisym_) delete gravity_vec_;
  delete utheta_next_gf_;
  delete utheta_gf_;
  delete u_next_rad_comp_gf_;
  delete pp_div_rad_comp_gf_;
  delete mu_total_gf_;
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
  delete gradU_gf_;
  delete gradV_gf_;
  delete gradW_gf_;
  delete epsi_gf_;
  delete vfes_;
  delete vfec_;
  delete sfes_;
  delete sfec_;
}

void Tomboulides::initializeSelf() {
  // Initialize minimal state and interface
  vfec_ = new H1_FECollection(vorder_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);
  u_curr_gf_ = new ParGridFunction(vfes_);
  u_next_gf_ = new ParGridFunction(vfes_);
  ustar_gf_ = new ParGridFunction(vfes_);
  rho_tmp_gf_ = new ParGridFunction(sfes_);    
  curl_gf_ = new ParGridFunction(vfes_);
  curlcurl_gf_ = new ParGridFunction(vfes_);
  resu_gf_ = new ParGridFunction(vfes_);
  pp_div_gf_ = new ParGridFunction(vfes_);
  gradU_gf_ = new ParGridFunction(vfes_);
  gradV_gf_ = new ParGridFunction(vfes_);
  gradW_gf_ = new ParGridFunction(vfes_);
  sfec_ = new H1_FECollection(vorder_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  pfec_ = new H1_FECollection(porder_);
  pfes_ = new ParFiniteElementSpace(pmesh_, pfec_);
  p_gf_ = new ParGridFunction(pfes_);
  resp_gf_ = new ParGridFunction(pfes_);

  epsi_gf_ = new ParGridFunction(sfes_);

  mu_total_gf_ = new ParGridFunction(pfes_);

  // pp_div_rad_comp_gf_ = new ParGridFunction(pfes_, *pp_div_gf_);
  // u_next_rad_comp_gf_ = new ParGridFunction(pfes_, *u_next_gf_);

  if (axisym_) {
    pp_div_rad_comp_gf_ = new ParGridFunction(pfes_);
    u_next_rad_comp_gf_ = new ParGridFunction(pfes_);

    utheta_gf_ = new ParGridFunction(pfes_);
    utheta_next_gf_ = new ParGridFunction(pfes_);
  }

  *u_curr_gf_ = 0.0;
  *u_next_gf_ = 0.0;
  *ustar_gf_ = 0.0;  
  *curl_gf_ = 0.0;
  *curlcurl_gf_ = 0.0;
  *resu_gf_ = 0.0;
  *rho_tmp_gf_ = 0.0;  

  *gradU_gf_ = 0.0;
  *gradV_gf_ = 0.0;
  *gradW_gf_ = 0.0;

  *p_gf_ = 0.0;
  *resp_gf_ = 0.0;
  *mu_total_gf_ = 0.0;

  *epsi_gf_ = 0.0;

  if (axisym_) {
    *utheta_gf_ = 0.0;
    *utheta_next_gf_ = 0.0;
  }

  // exports
  toThermoChem_interface_.velocity = u_next_gf_;
  toThermoChem_interface_.ustar = ustar_gf_;    
  if (axisym_) {
    toThermoChem_interface_.swirl_supported = true;
    toThermoChem_interface_.swirl = utheta_next_gf_;
  }

  toTurbModel_interface_.velocity = u_next_gf_;
  if (axisym_) {
    toTurbModel_interface_.swirl_supported = true;
    toTurbModel_interface_.swirl = utheta_next_gf_;
  }
  toTurbModel_interface_.gradU = gradU_gf_;
  toTurbModel_interface_.gradV = gradV_gf_;
  toTurbModel_interface_.gradW = gradW_gf_;

  // Allocate Vector storage
  const int vfes_truevsize = vfes_->GetTrueVSize();
  const int sfes_truevsize = sfes_->GetTrueVSize();
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
  grad_Qt_vec_.SetSize(vfes_truevsize);
  ress_vec_.SetSize(vfes_truevsize);
  S_poisson_vec_.SetSize(vfes_truevsize);
  if (axisym_) {
    ur_conv_forcing_vec_.SetSize(vfes_truevsize);
  }

  resp_vec_.SetSize(pfes_truevsize);
  p_vec_.SetSize(pfes_truevsize);
  pp_div_bdr_vec_.SetSize(pfes_truevsize);
  u_bdr_vec_.SetSize(pfes_truevsize);
  Qt_vec_.SetSize(pfes_truevsize);
  rho_vec_.SetSize(pfes_truevsize);
  rho_tmp_vec_.SetSize(pfes_truevsize);  
  mu_vec_.SetSize(pfes_truevsize);
  Faxi_poisson_vec_.SetSize(pfes_truevsize);
  if (axisym_) {
    utheta_vec_.SetSize(pfes_truevsize);
    utheta_m1_vec_.SetSize(pfes_truevsize);
    utheta_m2_vec_.SetSize(pfes_truevsize);
    utheta_next_vec_.SetSize(pfes_truevsize);
  }

  tmpR0_.SetSize(sfes_truevsize);
  tmpR1_.SetSize(vfes_truevsize);

  gradU_.SetSize(vfes_truevsize);
  gradV_.SetSize(vfes_truevsize);
  gradW_.SetSize(vfes_truevsize);

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
  grad_Qt_vec_ = 0.0;
  ress_vec_ = 0.0;
  S_poisson_vec_ = 0.0;

  resp_vec_ = 0.0;
  p_vec_ = 0.0;
  pp_div_bdr_vec_ = 0.0;
  u_bdr_vec_ = 0.0;
  Qt_vec_ = 0.0;
  rho_vec_ = 0.0;
  rho_tmp_vec_ = 0.0;  
  mu_vec_ = 0.0;

  if (axisym_) {
    utheta_vec_ = 0.0;
    utheta_m1_vec_ = 0.0;
    utheta_m2_vec_ = 0.0;
    utheta_next_vec_ = 0.0;
  }

  // Gravity
  assert(dim_ >= 2);
  Vector zerog(dim_);
  zerog = 0.0;
  gravity_.SetSize(dim_);
  tpsP_->getVec("loMach/gravity", gravity_, dim_, zerog);
  gravity_vec_ = new VectorConstantCoefficient(gravity_);
  Array<int> domain_attr(pmesh_->attributes.Max());
  domain_attr = 1;

  if (axisym_) {
    rad_gravity_vec_ = new ScalarVectorProductCoefficient(radius_coeff, *gravity_vec_);
    forcing_terms_.emplace_back(domain_attr, rad_gravity_vec_);
  } else {
    forcing_terms_.emplace_back(domain_attr, gravity_vec_);
  }

  // Initial condition function.  For options, see cases.cpp
  tpsP_->getInput("loMach/tomboulides/ic", ic_string_, std::string(""));

  // set IC if we have one at this point
  if (ic_string_ == "uniform") {
    Vector zero(dim_);
    zero = 0.0;
    velocity_ic_.SetSize(dim_);
    std::string basepath("loMach/tomboulides");
    tpsP_->getVec("loMach/tomboulides/velocity", velocity_ic_, dim_, zero);
    VectorConstantCoefficient u_excoeff(velocity_ic_);
    u_curr_gf_->ProjectCoefficient(u_excoeff);
  } else if (!ic_string_.empty()) {
    vfptr user_func = vel_ic(ic_string_);
    VectorFunctionCoefficient u_excoeff(nvel_, user_func);
    u_excoeff.SetTime(0.0);
    u_curr_gf_->ProjectCoefficient(u_excoeff);
  }

  // Boundary conditions
  // number of BC regions defined
  int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // Inlet BCs (Dirichlet on velocity)
  for (int i = 1; i <= numInlets; i++) {
    int patch;
    std::string type;
    std::string basepath("boundaryConditions/inlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    MPI_Barrier(vfes_->GetComm());
    if (type == "uniform") {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: Setting uniform Dirichlet velocity on patch = " << patch << std::endl;
      }
      Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
      inlet_attr = 0;
      inlet_attr[patch - 1] = 1;

      Vector zero(dim_);
      zero = 0.0;

      Vector velocity_value(dim_);
      tpsP_->getVec((basepath + "/velocity").c_str(), velocity_value, dim_, zero);

      addVelDirichletBC(velocity_value, inlet_attr);

      if (axisym_) {
        double swirl;
        tpsP_->getInput((basepath + "/swirl").c_str(), swirl, 0.0);
        addSwirlDirichletBC(swirl, inlet_attr);
      }

    } else if (type == "interpolate") {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: Setting interpolated Dirichlet velocity on patch = " << patch << std::endl;
      }
      Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
      inlet_attr = 0;
      inlet_attr[patch - 1] = 1;

      velocity_field_ = new VectorGridFunctionCoefficient(extData_interface_->Udata);
      addVelDirichletBC(velocity_field_, inlet_attr);

      // axisymmetric not testsed with interpolation BCs yet.  For now, just stop.
      assert(!axisym_);

    } else {
      Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
      inlet_attr = 0;
      inlet_attr[patch - 1] = 1;

      vfptr user_func = vel_bc(type);
      addVelDirichletBC(user_func, inlet_attr);

      if (axisym_) {
        addSwirlDirichletBC(0.0, inlet_attr);
      }
    }

    /*
    } else if (type == "fully-developed-pipe") {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: Setting uniform inlet velocity on patch = " << patch << std::endl;
      }
      Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
      inlet_attr = 0;
      inlet_attr[patch - 1] = 1;

      addVelDirichletBC(vel_exact_pipe, inlet_attr);
      if (axisym_) {
        addSwirlDirichletBC(0.0, inlet_attr);
      }
    } else {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: When reading " << basepath << ", encountered inlet type = " << type << std::endl;
        std::cout << "Tomboulides: Tomboulides flow solver does not support this type." << std::endl;
      }
      assert(false);
      exit(1);
    }
    */
  }

  // Wall Bcs
  for (int i = 1; i <= numWalls; i++) {
    int patch;
    std::string type;
    std::string basepath("boundaryConditions/wall" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    if (type == "viscous_isothermal" || type == "viscous_adiabatic" || type == "viscous" || type == "no-slip") {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: Setting Dirichlet velocity on patch = " << patch << std::endl;
      }

      Array<int> wall_attr(pmesh_->bdr_attributes.Max());
      wall_attr = 0;
      wall_attr[patch - 1] = 1;

      Vector zero(dim_);
      zero = 0.0;

      Vector velocity_value(dim_);
      tpsP_->getVec((basepath + "/velocity").c_str(), velocity_value, dim_, zero);

      addVelDirichletBC(velocity_value, wall_attr);

      if (axisym_) {
        double swirl;
        tpsP_->getInput((basepath + "/swirl").c_str(), swirl, 0.0);
        addSwirlDirichletBC(swirl, wall_attr);
      }
    } else {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: When reading " << basepath << ", encountered wall type = " << type << std::endl;
        std::cout << "Tomboulides: Tomboulides flow solver does not support this type." << std::endl;
      }
      assert(false);
      exit(1);
    }
  }

  // Outlet BCs
  for (int i = 1; i <= numOutlets; i++) {
    if (rank0_) std::cout << "Caught outlet" << std::endl;
    int patch;
    std::string type;
    std::string basepath("boundaryConditions/outlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    if (type == "uniform") {
      if (rank0_) std::cout << "attempting to apply uniform pressure outlet" << patch << std::endl;
      Array<int> outlet_attr(pmesh_->bdr_attributes.Max());
      outlet_attr = 0;
      outlet_attr[patch - 1] = 1;

      double pback;
      tpsP_->getInput((basepath + "/pressure").c_str(), pback, 0.0);

      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: Setting uniform outlet pressure on patch = " << patch << std::endl;
      }
      addPresDirichletBC(pback, outlet_attr);
    } else {
      if (pmesh_->GetMyRank() == 0) {
        std::cout << "Tomboulides: When reading " << basepath << ", encountered outlet type = " << type << std::endl;
        std::cout << "Tomboulides: Tomboulides flow solver does not support this type." << std::endl;
      }
      assert(false);
      exit(1);
    }
  }
}

void Tomboulides::initializeOperators() {
  assert(thermo_interface_ != NULL);

  // polynomial order for smoother of precons
  smoother_poly_order_ = vorder_ + 1;

  // AMG strength threshold
  if (dim_ == 2) {
    pressure_strength_thres_ = 0.25;
  } else {
    pressure_strength_thres_ = 0.6;
  }

  // Create all the Coefficient objects we need
  rho_coeff_ = new GridFunctionCoefficient(thermo_interface_->density);
  nlcoeff_ = new GridFunctionCoefficient(rho_tmp_gf_);
  nlcoeff_const_.constant = -1.0;

  if (axisym_) {
    iorho_coeff_ = new RatioCoefficient(radius_coeff, *rho_coeff_);
  } else {
    iorho_coeff_ = new RatioCoefficient(1.0, *rho_coeff_);
  }

  Hv_bdfcoeff_.constant = 1.0 / coeff_.dt;
  rho_over_dt_coeff_ = new ProductCoefficient(Hv_bdfcoeff_, *rho_coeff_);

  updateTotalViscosity();

  mu_coeff_ = new GridFunctionCoefficient(mu_total_gf_);
  pp_div_coeff_ = new VectorGridFunctionCoefficient(pp_div_gf_);
  pp_div_rad_comp_coeff_ = new GridFunctionCoefficient(pp_div_rad_comp_gf_);

  // coefficients used in the variable viscosity terms
  grad_mu_coeff_ = new GradientGridFunctionCoefficient(mu_total_gf_);
  grad_u_next_coeff_ = new GradientVectorGridFunctionCoefficient(u_next_gf_);
  grad_u_next_transp_coeff_ = new TransposeMatrixCoefficient(*grad_u_next_coeff_);

  gradu_gradmu_coeff_ = new MatrixVectorProductCoefficient(*grad_u_next_coeff_, *grad_mu_coeff_);
  graduT_gradmu_coeff_ = new MatrixVectorProductCoefficient(*grad_u_next_transp_coeff_, *grad_mu_coeff_);
  twoS_gradmu_coeff_ = new VectorSumCoefficient(*gradu_gradmu_coeff_, *graduT_gradmu_coeff_);

  Qt_coeff_ = new GridFunctionCoefficient(thermo_interface_->thermal_divergence);
  gradmu_Qt_coeff_ = new ScalarVectorProductCoefficient(*Qt_coeff_, *grad_mu_coeff_);

  S_poisson_coeff_ = new VectorSumCoefficient(*twoS_gradmu_coeff_, *gradmu_Qt_coeff_, 1.0, -2. / 3);
  S_mom_coeff_ = new VectorSumCoefficient(*graduT_gradmu_coeff_, *gradmu_Qt_coeff_, 1.0, -1.0);

  // Coefficients for axisymmetric
  if (axisym_) {
    rad_rho_coeff_ = new ProductCoefficient(radius_coeff, *rho_coeff_);
    rad_rho_over_dt_coeff_ = new ProductCoefficient(Hv_bdfcoeff_, *rad_rho_coeff_);
    rad_mu_coeff_ = new ProductCoefficient(radius_coeff, *mu_coeff_);
    mu_over_rad_coeff_ = new RatioCoefficient(*mu_coeff_, radius_coeff);
    rad_S_poisson_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *S_poisson_coeff_);
    rad_S_mom_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *S_mom_coeff_);

    // NB: Takes ownership of mu_over_rad_coeff_
    visc_forcing_coeff_ = new VectorArrayCoefficient(2);
    visc_forcing_coeff_->Set(0, mu_over_rad_coeff_);

    utheta_coeff_ = new GridFunctionCoefficient(utheta_next_gf_);
    utheta2_coeff_ = new ProductCoefficient(*utheta_coeff_, *utheta_coeff_);

    // NB: Takes ownership of utheta2_coeff_
    ur_conv_forcing_coeff_ = new VectorArrayCoefficient(2);
    ur_conv_forcing_coeff_->Set(0, utheta2_coeff_);

    u_next_coeff_ = new VectorGridFunctionCoefficient(u_next_gf_);
    rad_rhou_coeff_ = new ScalarVectorProductCoefficient(*rad_rho_coeff_, *u_next_coeff_);

    u_next_rad_coeff_ = new GridFunctionCoefficient(u_next_rad_comp_gf_);
    ur_ut_coeff_ = new ProductCoefficient(*u_next_rad_coeff_, *utheta_coeff_);
    rho_ur_ut_coeff_ = new ProductCoefficient(*rho_coeff_, *ur_ut_coeff_);

    // NB: This is a sort of hacky/sneaky way to form the variable
    // viscosity contribution to the swirl equation.  Should find a
    // better way.
    utheta_vec_coeff_ = new VectorArrayCoefficient(2);
    utheta_vec_coeff_->Set(0, utheta_coeff_);
    swirl_var_viscosity_coeff_ = new InnerProductCoefficient(*grad_mu_coeff_, *utheta_vec_coeff_);
  }

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
  pfes_->GetEssentialTrueDofs(pres_ess_attr_, pres_ess_tdof_);
  if (axisym_) {
    pfes_->GetEssentialTrueDofs(swirl_ess_attr_, swirl_ess_tdof_);
  }

  // Create the operators

  // Variable coefficient Laplacian: \nabla \cdot ( (1/\rho) \nabla )
  L_iorho_form_ = new ParBilinearForm(pfes_);
  auto *L_iorho_blfi = new DiffusionIntegrator(*iorho_coeff_);
  if (numerical_integ_) {
    L_iorho_blfi->SetIntRule(&ir_ni_p);
  }
  L_iorho_form_->AddDomainIntegrator(L_iorho_blfi);
  L_iorho_form_->Assemble();
  L_iorho_form_->FormSystemMatrix(pres_ess_tdof_, L_iorho_op_);

  // Variable coefficient Laplacian inverse (gets destoryed and rebuilt every step)
  L_iorho_lor_ = new ParLORDiscretization(*L_iorho_form_, pres_ess_tdof_);
  L_iorho_inv_pc_ = new HypreBoomerAMG(L_iorho_lor_->GetAssembledMatrix());
  L_iorho_inv_pc_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_ortho_pc_ = new OrthoSolver(pfes_->GetComm());
  L_iorho_inv_ortho_pc_->SetSolver(*L_iorho_inv_pc_);

  L_iorho_inv_ = new CGSolver(pfes_->GetComm());
  L_iorho_inv_->iterative_mode = true;
  L_iorho_inv_->SetOperator(*L_iorho_op_);
  if (pres_dbcs_.empty()) {
    L_iorho_inv_->SetPreconditioner(*L_iorho_inv_ortho_pc_);
  } else {
    L_iorho_inv_->SetPreconditioner(*L_iorho_inv_pc_);
  }
  L_iorho_inv_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_->SetRelTol(pressure_solve_rtol_);
  L_iorho_inv_->SetAbsTol(pressure_solve_atol_);
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

  Nconv_form_ = new ParNonlinearForm(vfes_);
  VectorConvectionNLFIntegrator *nlc_nlfi;
  if (axisym_) {
    nlc_nlfi = new VectorConvectionNLFIntegrator(negative_radius_coeff);
  } else {
    nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff_const_);
  }
  if (numerical_integ_) {
    nlc_nlfi->SetIntRule(&ir_ni_v);
  }
  Nconv_form_->AddDomainIntegrator(nlc_nlfi);
  if (partial_assembly_) {
    Nconv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    Nconv_form_->Setup();
  }
  
  // The nonlinear (i.e., convection) term
  // Coefficient is -1 so we can just add to rhs
  // ...cannot be constant with all-mach nlcoeff_.constant = -1.0;
  /*
  Nconv_rho_form_ = new ParNonlinearForm(vfes_);
  VectorConvectionNLFIntegrator *nlcr_nlfi;
  if (axisym_) {
    nlcr_nlfi = new VectorConvectionNLFIntegrator(negative_radius_coeff);
  } else {
    nlcr_nlfi = new VectorConvectionNLFIntegrator(nlcoeff_);
  }
  if (numerical_integ_) {
    nlcr_nlfi->SetIntRule(&ir_ni_v);
  }
  Nconv_rho_form_->AddDomainIntegrator(nlcr_nlfi);
  if (partial_assembly_) {
    Nconv_rho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    Nconv_rho_form_->Setup();
  }
  */

  // Mass matrix for the scalar space
  // NB: this is used to add the Q contribution to the RHS of the
  // pressure Poisson equation.  In doing so, we ASSUME that the
  // pressure space and the Q space are the same.  Need to assert this
  // somehow.
  Ms_form_ = new ParBilinearForm(pfes_);
  MassIntegrator *ms_blfi;
  if (axisym_) {
    ms_blfi = new MassIntegrator(radius_coeff);
  } else {
    ms_blfi = new MassIntegrator();
  }
  if (numerical_integ_) {
    ms_blfi->SetIntRule(&ir_ni_p);
  }
  Ms_form_->AddDomainIntegrator(ms_blfi);
  if (partial_assembly_) {
    Ms_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ms_form_->Assemble();
  Ms_form_->FormSystemMatrix(empty, Ms_op_);

  // Inverse (unweighted) mass operator (scalar space)
  if (partial_assembly_) {
    Vector diag_pa(vfes_->GetTrueVSize());
    Ms_form_->AssembleDiagonal(diag_pa);
    Ms_inv_pc_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    Ms_inv_pc_ = new HypreSmoother(*Ms_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Ms_inv_pc_)->SetType(smoother_type_, smoother_passes_);
    dynamic_cast<HypreSmoother *>(Ms_inv_pc_)->SetSOROptions(smoother_relax_weight_, smoother_relax_omega_);
    dynamic_cast<HypreSmoother *>(Ms_inv_pc_)
        ->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_, smoother_eig_est_);
  }
  Ms_inv_ = new CGSolver(vfes_->GetComm());
  Ms_inv_->iterative_mode = false;
  Ms_inv_->SetOperator(*Mv_op_);
  Ms_inv_->SetPreconditioner(*Mv_inv_pc_);
  Ms_inv_->SetPrintLevel(mass_inverse_pl_);
  Ms_inv_->SetAbsTol(mass_inverse_atol_);
  Ms_inv_->SetRelTol(mass_inverse_rtol_);
  Ms_inv_->SetMaxIter(mass_inverse_max_iter_);

  
  Ms_rho_form_ = new ParBilinearForm(pfes_);
  MassIntegrator *msr_blfi;
  if (axisym_) {
    msr_blfi = new MassIntegrator(*rad_rho_coeff_);
  } else {
    msr_blfi = new MassIntegrator(*rho_coeff_);
  }
  if (numerical_integ_) {
    msr_blfi->SetIntRule(&ir_ni_p);
  }
  Ms_rho_form_->AddDomainIntegrator(msr_blfi);
  Ms_rho_form_->Assemble();
  Ms_rho_form_->FormSystemMatrix(empty, Ms_rho_op_);

  // Mass matrix for the velocity
  Mv_form_ = new ParBilinearForm(vfes_);
  VectorMassIntegrator *mv_blfi;
  if (axisym_) {
    mv_blfi = new VectorMassIntegrator(radius_coeff);
  } else {
    mv_blfi = new VectorMassIntegrator();
  }
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
  VectorMassIntegrator *mvr_blfi;
  if (axisym_) {
    mvr_blfi = new VectorMassIntegrator(*rad_rho_coeff_);
  } else {
    //mvr_blfi = new VectorMassIntegrator(*rho_coeff_);
    mvr_blfi = new VectorMassIntegrator(*nlcoeff_); // use rho_tmp_gf_
  }
  if (numerical_integ_) {
    mvr_blfi->SetIntRule(&ir_ni_v);
  }
  Mv_rho_form_->AddDomainIntegrator(mvr_blfi);
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);

  // Inverse (unweighted) mass operator (velocity space)
  if (partial_assembly_) {
    Vector diag_pa(vfes_->GetTrueVSize());
    Mv_form_->AssembleDiagonal(diag_pa);
    Mv_inv_pc_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    Mv_inv_pc_ = new HypreSmoother(*Mv_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Mv_inv_pc_)->SetType(smoother_type_, smoother_passes_);
    dynamic_cast<HypreSmoother *>(Mv_inv_pc_)->SetSOROptions(smoother_relax_weight_, smoother_relax_omega_);
    dynamic_cast<HypreSmoother *>(Mv_inv_pc_)
        ->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_, smoother_eig_est_);
  }
  Mv_inv_ = new CGSolver(vfes_->GetComm());
  Mv_inv_->iterative_mode = false;
  Mv_inv_->SetOperator(*Mv_op_);
  Mv_inv_->SetPreconditioner(*Mv_inv_pc_);
  Mv_inv_->SetPrintLevel(mass_inverse_pl_);
  Mv_inv_->SetAbsTol(mass_inverse_atol_);
  Mv_inv_->SetRelTol(mass_inverse_rtol_);
  Mv_inv_->SetMaxIter(mass_inverse_max_iter_);

  Mv_rho_inv_pc_ = new HypreSmoother(*Mv_rho_op_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(Mv_rho_inv_pc_)->SetType(smoother_type_, smoother_passes_);
  dynamic_cast<HypreSmoother *>(Mv_rho_inv_pc_)->SetSOROptions(smoother_relax_weight_, smoother_relax_omega_);
  dynamic_cast<HypreSmoother *>(Mv_rho_inv_pc_)
      ->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_, smoother_eig_est_);

  Mv_rho_inv_ = new CGSolver(vfes_->GetComm());
  Mv_rho_inv_->iterative_mode = false;
  Mv_rho_inv_->SetOperator(*Mv_rho_op_);
  Mv_rho_inv_->SetPreconditioner(*Mv_rho_inv_pc_);
  Mv_rho_inv_->SetPrintLevel(mass_inverse_pl_);
  Mv_rho_inv_->SetAbsTol(mass_inverse_atol_);
  Mv_rho_inv_->SetRelTol(mass_inverse_rtol_);
  Mv_rho_inv_->SetMaxIter(mass_inverse_max_iter_);

  // Divergence operator
  D_form_ = new ParMixedBilinearForm(vfes_, pfes_);
  VectorDivergenceIntegrator *vd_mblfi;
  if (axisym_) {
    vd_mblfi = new VectorDivergenceIntegrator(radius_coeff);
  } else {
    vd_mblfi = new VectorDivergenceIntegrator();
  }
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
  // auto *g_mblfi = new GradientIntegrator();
  GradientIntegrator *g_mblfi;
  if (axisym_) {
    g_mblfi = new GradientIntegrator(radius_coeff);
  } else {
    g_mblfi = new GradientIntegrator();
  }
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
  VectorMassIntegrator *hmv_blfi;
  VectorDiffusionIntegrator *hdv_blfi;
  if (axisym_) {
    hmv_blfi = new VectorMassIntegrator(*rad_rho_over_dt_coeff_);
    hdv_blfi = new VectorDiffusionIntegrator(*rad_mu_coeff_);
  } else {
    hmv_blfi = new VectorMassIntegrator(*rho_over_dt_coeff_);
    hdv_blfi = new VectorDiffusionIntegrator(*mu_coeff_);
  }
  if (numerical_integ_) {
    hmv_blfi->SetIntRule(&ir_ni_v);
    hdv_blfi->SetIntRule(&ir_ni_v);
  }
  Hv_form_->AddDomainIntegrator(hmv_blfi);
  Hv_form_->AddDomainIntegrator(hdv_blfi);
  if (axisym_) {
    auto *hfv_blfi = new VectorMassIntegrator(*visc_forcing_coeff_);
    Hv_form_->AddDomainIntegrator(hfv_blfi);
  }
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(vel_ess_tdof_, Hv_op_);

  // Helmholtz solver
  Hv_inv_pc_ = new HypreSmoother(*Hv_op_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(Hv_inv_pc_)->SetType(smoother_type_, smoother_passes_);
  dynamic_cast<HypreSmoother *>(Hv_inv_pc_)->SetSOROptions(hsmoother_relax_weight_, hsmoother_relax_omega_);
  dynamic_cast<HypreSmoother *>(Hv_inv_pc_)
      ->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_, smoother_eig_est_);

  Hv_inv_ = new CGSolver(vfes_->GetComm());
  Hv_inv_->iterative_mode = true;
  Hv_inv_->SetOperator(*Hv_op_);
  Hv_inv_->SetPreconditioner(*Hv_inv_pc_);
  Hv_inv_->SetPrintLevel(hsolve_pl_);
  Hv_inv_->SetAbsTol(hsolve_atol_);
  Hv_inv_->SetRelTol(hsolve_rtol_);
  Hv_inv_->SetMaxIter(hsolve_max_iter_);
  
  //
  pp_div_bdr_form_ = new ParLinearForm(pfes_);
  BoundaryNormalLFIntegrator *ppd_bnlfi;
  if (axisym_) {
    rad_pp_div_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *pp_div_coeff_);
    ppd_bnlfi = new BoundaryNormalLFIntegrator(*rad_pp_div_coeff_);
  } else {
    ppd_bnlfi = new BoundaryNormalLFIntegrator(*pp_div_coeff_);
  }
  if (numerical_integ_) {
    ppd_bnlfi->SetIntRule(&ir_ni_p);
  }
  pp_div_bdr_form_->AddBoundaryIntegrator(ppd_bnlfi, vel_ess_attr_);

  u_bdr_form_ = new ParLinearForm(pfes_);
  for (auto &vel_dbc : vel_dbcs_) {
    BoundaryNormalLFIntegrator *ubdr_bnlfi;
    if (axisym_) {
      rad_vel_coeff_.push_back(new ScalarVectorProductCoefficient(radius_coeff, *vel_dbc.coeff));
      ubdr_bnlfi = new BoundaryNormalLFIntegrator(*rad_vel_coeff_[rad_vel_coeff_.size() - 1]);
    } else {
      ubdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
    }
    if (numerical_integ_) {
      ubdr_bnlfi->SetIntRule(&ir_ni_p);
    }
    u_bdr_form_->AddBoundaryIntegrator(ubdr_bnlfi, vel_dbc.attr);
  }

  S_poisson_form_ = new ParLinearForm(vfes_);
  VectorDomainLFIntegrator *s_rhs_dlfi;
  if (axisym_) {
    s_rhs_dlfi = new VectorDomainLFIntegrator(*rad_S_poisson_coeff_);
  } else {
    s_rhs_dlfi = new VectorDomainLFIntegrator(*S_poisson_coeff_);
  }
  if (numerical_integ_) {
    s_rhs_dlfi->SetIntRule(&ir_ni_v);
  }
  S_poisson_form_->AddDomainIntegrator(s_rhs_dlfi);

  S_mom_form_ = new ParLinearForm(vfes_);
  VectorDomainLFIntegrator *s_mom_dlfi;
  if (axisym_) {
    s_mom_dlfi = new VectorDomainLFIntegrator(*rad_S_mom_coeff_);
  } else {
    s_mom_dlfi = new VectorDomainLFIntegrator(*S_mom_coeff_);
  }
  if (numerical_integ_) {
    s_mom_dlfi->SetIntRule(&ir_ni_v);
  }
  S_mom_form_->AddDomainIntegrator(s_mom_dlfi);

  if (axisym_) {
    Faxi_poisson_form_ = new ParLinearForm(pfes_);
    auto *f_rhs_dlfi = new DomainLFIntegrator(*pp_div_rad_comp_coeff_);
    if (numerical_integ_) {
      f_rhs_dlfi->SetIntRule(&ir_ni_p);
    }
    Faxi_poisson_form_->AddDomainIntegrator(f_rhs_dlfi);

    ur_conv_axi_form_ = new ParLinearForm(vfes_);
    auto *urca_dlfi = new VectorDomainLFIntegrator(*ur_conv_forcing_coeff_);
    if (numerical_integ_) {
      urca_dlfi->SetIntRule(&ir_ni_v);
    }
    ur_conv_axi_form_->AddDomainIntegrator(urca_dlfi);

    // Helmholtz
    Hs_form_ = new ParBilinearForm(pfes_);
    auto *hms_blfi = new MassIntegrator(*rad_rho_over_dt_coeff_);
    auto *hds_blfi = new DiffusionIntegrator(*rad_mu_coeff_);
    auto *hfs_blfi = new MassIntegrator(*mu_over_rad_coeff_);

    Hs_form_->AddDomainIntegrator(hms_blfi);
    Hs_form_->AddDomainIntegrator(hds_blfi);
    Hs_form_->AddDomainIntegrator(hfs_blfi);

    Hs_form_->Assemble();
    Hs_form_->FormSystemMatrix(swirl_ess_tdof_, Hs_op_);

    Hs_inv_pc_ = new HypreSmoother(*Hs_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Hs_inv_pc_)->SetType(smoother_type_, smoother_passes_);
    dynamic_cast<HypreSmoother *>(Hs_inv_pc_)->SetSOROptions(hsmoother_relax_weight_, hsmoother_relax_omega_);
    dynamic_cast<HypreSmoother *>(Hs_inv_pc_)
        ->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_, smoother_eig_est_);

    Hs_inv_ = new CGSolver(pfes_->GetComm());
    Hs_inv_->iterative_mode = true;
    Hs_inv_->SetOperator(*Hs_op_);
    Hs_inv_->SetPreconditioner(*Hs_inv_pc_);
    Hs_inv_->SetPrintLevel(hsolve_pl_);
    Hs_inv_->SetAbsTol(hsolve_atol_);
    Hs_inv_->SetRelTol(hsolve_rtol_);
    Hs_inv_->SetMaxIter(hsolve_max_iter_);

    // Convection terms
    As_form_ = new ParBilinearForm(pfes_);

    // NB: -1 so that sign is correct on rhs
    auto *as_blfi = new ConvectionIntegrator(*rad_rhou_coeff_, -1.0);

    As_form_->AddDomainIntegrator(as_blfi);
    As_form_->Assemble();
    As_form_->FormSystemMatrix(empty, As_op_);

    rho_ur_ut_form_ = new ParLinearForm(pfes_);
    auto *rurut_dlfi = new DomainLFIntegrator(*rho_ur_ut_coeff_);
    rho_ur_ut_form_->AddDomainIntegrator(rurut_dlfi);

    swirl_var_viscosity_form_ = new ParLinearForm(pfes_);
    auto *svv_dlfi = new DomainLFIntegrator(*swirl_var_viscosity_coeff_);
    swirl_var_viscosity_form_->AddDomainIntegrator(svv_dlfi);
  }

  // Ensure u_vec_ consistent with u_curr_gf_
  u_curr_gf_->GetTrueDofs(u_vec_);
  if (axisym_) {
    utheta_gf_->GetTrueDofs(utheta_vec_);
  }

  *u_next_gf_ = *u_curr_gf_;
  u_next_gf_->GetTrueDofs(u_next_vec_);

  evaluateVelocityGradient();
}

void Tomboulides::initializeIO(IODataOrganizer &io) const {
  io.registerIOFamily("Velocity", "/velocity", u_curr_gf_, true, true, vfec_);
  io.registerIOVar("/velocity", "x-comp", 0);
  if (dim_ >= 2) io.registerIOVar("/velocity", "y-comp", 1);
  if (dim_ == 3) io.registerIOVar("/velocity", "z-comp", 2);

  if (axisym_) {
    io.registerIOFamily("Velocity azimuthal", "/swirl", utheta_gf_, true, true, pfec_);
    io.registerIOVar("/swirl", "swirl", 0);
  }
}

void Tomboulides::initializeViz(mfem::ParaViewDataCollection &pvdc) const {
  pvdc.RegisterField("velocity", u_curr_gf_);
  pvdc.RegisterField("pressure", p_gf_);
  if (axisym_) {
    pvdc.RegisterField("swirl", utheta_gf_);
  }
}

void Tomboulides::initializeStats(Averaging &average, IODataOrganizer &io, bool continuation) const {
  if (average.ComputeMean()) {
    // fields for averaging
    average.registerField(std::string("velocity"), u_curr_gf_, true, 0, nvel_);
    average.registerField(std::string("pressure"), p_gf_, false, 0, 1);
    average.registerField(std::string("dissipation"), epsi_gf_, false, 0, 1);

    // io init
    io.registerIOFamily("Time-averaged velocity", "/meanVel", average.GetMeanField(std::string("velocity")), false,
                        continuation, vfec_);
    io.registerIOVar("/meanVel", "<u>", 0, true);
    if (dim_ >= 2) io.registerIOVar("/meanVel", "<v>", 1, true);
    if (dim_ == 3) io.registerIOVar("/meanVel", "<w>", 2, true);

    io.registerIOFamily("Time-averaged pressure", "/meanPres", average.GetMeanField(std::string("pressure")), false,
                        continuation, pfec_);
    io.registerIOVar("/meanPres", "<P>", 0, true);

    io.registerIOFamily("Time-averaged dissipation", "/meanEpsi", average.GetMeanField(std::string("dissipation")),
                        false, continuation, pfec_);
    io.registerIOVar("/meanEpsi", "<e>", 0, true);

    // rms
    io.registerIOFamily("RMS velocity fluctuation", "/rmsData", average.GetVariField(std::string("velocity")), false,
                        continuation, vfec_);
    if (nvel_ == 3) {
      io.registerIOVar("/rmsData", "uu", 0, true);
      io.registerIOVar("/rmsData", "vv", 1, true);
      io.registerIOVar("/rmsData", "ww", 2, true);
      io.registerIOVar("/rmsData", "uv", 3, true);
      io.registerIOVar("/rmsData", "uw", 4, true);
      io.registerIOVar("/rmsData", "vw", 5, true);
    } else if (nvel_ == 2) {
      io.registerIOVar("/rmsData", "uu", 0, true);
      io.registerIOVar("/rmsData", "vv", 1, true);
      io.registerIOVar("/rmsData", "uv", 2, true);
    } else {
      // only nvel = 2 or 3 supported
      assert(false);
    }
  }
}

void Tomboulides::computeDissipation(Averaging &average, const int iter) {
  if (average.ComputeMean()) {
    int sample_interval = average.GetSamplesInterval();
    int sample_start = average.GetStartMean();
    if (iter % sample_interval == 0 && iter >= sample_start) {
      // fluctuating velocity
      u_curr_gf_->GetTrueDofs(tmpR1_);
      average.GetMeanField("velocity")->GetTrueDofs(u_vec_);
      tmpR1_.Add(-1.0, u_vec_);

      // gradient of u'
      setScalarFromVector(tmpR1_, 0, &tmpR0_);
      G_op_->Mult(tmpR0_, gradU_);
      setScalarFromVector(tmpR1_, 1, &tmpR0_);
      G_op_->Mult(tmpR0_, gradV_);
      if (dim_ == 3) {
        setScalarFromVector(tmpR1_, 2, &tmpR0_);
        G_op_->Mult(tmpR0_, gradW_);
      }

      Mv_inv_->Mult(gradU_, tmpR1_);
      gradU_ = tmpR1_;
      Mv_inv_->Mult(gradV_, tmpR1_);
      gradV_ = tmpR1_;
      if (dim_ == 3) {
        Mv_inv_->Mult(gradW_, tmpR1_);
        gradW_ = tmpR1_;
      }

      gradU_gf_->SetFromTrueDofs(gradU_);
      gradV_gf_->SetFromTrueDofs(gradV_);
      if (dim_ == 3) gradW_gf_->SetFromTrueDofs(gradW_);

      // const double *dmu = (*thermo_interface_->viscosity).HostRead();
      const double *dmu = mu_total_gf_->HostRead();
      const double *drho = (*thermo_interface_->density).HostRead();
      const double *dGradU = gradU_gf_->HostRead();
      const double *dGradV = gradV_gf_->HostRead();
      const double *dGradW = gradW_gf_->HostRead();
      double *depsi = epsi_gf_->HostReadWrite();

      int Sdof = epsi_gf_->Size();
      for (int dof = 0; dof < Sdof; dof++) {
        depsi[dof] = 0.0;

        DenseMatrix gradUp;
        gradUp.SetSize(nvel_, dim_);
        for (int dir = 0; dir < dim_; dir++) {
          gradUp(0, dir) = dGradU[dof + dir * Sdof];
        }
        for (int dir = 0; dir < dim_; dir++) {
          gradUp(1, dir) = dGradV[dof + dir * Sdof];
        }
        if (dim_ == 3) {
          for (int dir = 0; dir < dim_; dir++) {
            gradUp(2, dir) = dGradW[dof + dir * Sdof];
          }
        }

        for (int i = 0; i < dim_; i++) {
          for (int j = 0; j < dim_; j++) {
            depsi[dof] += gradUp(i, j) * gradUp(i, j);
          }
        }
        depsi[dof] *= 2.0 * dmu[dof] / drho[dof];
      }

      // probably not necessary but just to be extra safe
      u_curr_gf_->GetTrueDofs(u_vec_);
      gradU_gf_->GetTrueDofs(gradU_);
      gradV_gf_->GetTrueDofs(gradV_);
      gradW_gf_->GetTrueDofs(gradW_);
    }
  }
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

  // Ensure u_vec_ consistent with u_curr_gf_
  u_curr_gf_->GetTrueDofs(u_vec_);

  //------------------------------------------------------------------------
  // Step 1: Update any operators invalidated by changing data
  // external to this class
  // ------------------------------------------------------------------------

  // Update total viscosity field
  updateTotalViscosity();

  // Update the variable coefficient Laplacian to account for change
  // in density
  sw_press_.Start();
  L_iorho_form_->Update();
  L_iorho_form_->Assemble();
  L_iorho_form_->FormSystemMatrix(pres_ess_tdof_, L_iorho_op_);

  // Update inverse for the variable coefficient Laplacian
  // TODO(trevilo): Really need to delete and recreate???
  delete L_iorho_inv_;
  delete L_iorho_inv_ortho_pc_;
  delete L_iorho_inv_pc_;
  delete L_iorho_lor_;

  L_iorho_lor_ = new ParLORDiscretization(*L_iorho_form_, pres_ess_tdof_);
  L_iorho_inv_pc_ = new HypreBoomerAMG(L_iorho_lor_->GetAssembledMatrix());
  L_iorho_inv_pc_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_pc_->SetCoarsening(amg_coarsening_);
  L_iorho_inv_pc_->SetAggressiveCoarsening(amg_aggresive_);
  L_iorho_inv_pc_->SetInterpolation(amg_interpolation_);
  L_iorho_inv_pc_->SetStrengthThresh(pressure_strength_thres_);
  L_iorho_inv_pc_->SetRelaxType(amg_relax_);
  L_iorho_inv_pc_->SetMaxLevels(amg_max_levels_);
  L_iorho_inv_pc_->SetMaxIter(amg_max_iters_);
  L_iorho_inv_pc_->SetTol(0);
  // NOTE: other cycle types (i.e. not V-cycle) do not scale well in parallel, so option removed for now
  // L_iorho_inv_pc_->SetCycleType(1);
  L_iorho_inv_ortho_pc_ = new OrthoSolver(pfes_->GetComm());
  L_iorho_inv_ortho_pc_->SetSolver(*L_iorho_inv_pc_);

  L_iorho_inv_ = new CGSolver(pfes_->GetComm());
  L_iorho_inv_->iterative_mode = true;
  L_iorho_inv_->SetOperator(*L_iorho_op_);
  if (pres_dbcs_.empty()) {
    L_iorho_inv_->SetPreconditioner(*L_iorho_inv_ortho_pc_);
  } else {
    L_iorho_inv_->SetPreconditioner(*L_iorho_inv_pc_);
  }
  L_iorho_inv_->SetPrintLevel(pressure_solve_pl_);
  L_iorho_inv_->SetRelTol(pressure_solve_rtol_);
  L_iorho_inv_->SetAbsTol(pressure_solve_atol_);
  L_iorho_inv_->SetMaxIter(pressure_solve_max_iter_);
  sw_press_.Stop();

  // Update density weighted mass
  Array<int> empty;
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);

  Mv_rho_inv_->SetOperator(*Mv_rho_op_);

  // Update the Helmholtz operator and inverse
  sw_helm_.Start();
  Hv_bdfcoeff_.constant = coeff_.bd0 / coeff_.dt;
  Hv_form_->Update();
  Hv_form_->Assemble();
  Hv_form_->FormSystemMatrix(vel_ess_tdof_, Hv_op_);

  Hv_inv_->SetOperator(*Hv_op_);

  sw_helm_.Stop();

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

  if (axisym_) {
    // Extrapolate the swirl velocity
    {
      const auto d_u = utheta_vec_.Read();
      const auto d_um1 = utheta_m1_vec_.Read();
      const auto d_um2 = utheta_m2_vec_.Read();
      auto d_uext = utheta_next_vec_.Write();
      const auto ab1 = coeff_.ab1;
      const auto ab2 = coeff_.ab2;
      const auto ab3 = coeff_.ab3;
      MFEM_FORALL(i, utheta_next_vec_.Size(), { d_uext[i] = ab1 * d_u[i] + ab2 * d_um1[i] + ab3 * d_um2[i]; });
    }
    utheta_next_gf_->SetFromTrueDofs(utheta_next_vec_);

    // evaluate the rho u_{\theta}^2 term (from convection)
    ur_conv_axi_form_->Assemble();
    ur_conv_axi_form_->ParallelAssemble(ur_conv_forcing_vec_);
    forcing_vec_ += ur_conv_forcing_vec_;
  }

  // vstar / dt = M^{-1} (extrapolated nonlinear + forcing) --- eqn.
  Mv_inv_->Mult(forcing_vec_, ustar_vec_);
  if (!Mv_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Mv inverse did not converge." << std::endl;
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

  // NB: axisymmetric (no swirl!) implemented to here

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
  if (axisym_) {
    ComputeCurlAxi(*u_next_gf_, *curl_gf_, false);
    ComputeCurlAxi(*curl_gf_, *curlcurl_gf_, true);
  } else {
    if (dim_ == 2) {
      ComputeCurl2D(*u_next_gf_, *curl_gf_, false);
      ComputeCurl2D(*curl_gf_, *curlcurl_gf_, true);
    } else {
      ComputeCurl3D(*u_next_gf_, *curl_gf_);
      ComputeCurl3D(*curl_gf_, *curlcurl_gf_);
    }
  }
  curlcurl_gf_->GetTrueDofs(pp_div_vec_);

  // Negate b/c term that appears in residual is -curl curl u
  pp_div_vec_.Neg();

  // TODO(trevilo): This calculation of the pressure Poisson RHS
  // assumes that mu is constant (i.e., there are terms that are
  // proportional to grad(mu) that are not included below).  Relax
  // this assumption by adding the necessary terms!

  // Compute and add grad(Qt) contribution
  assert(thermo_interface_->thermal_divergence != nullptr);
  thermo_interface_->thermal_divergence->GetTrueDofs(Qt_vec_);
  G_op_->Mult(Qt_vec_, resu_vec_);
  Mv_inv_->Mult(resu_vec_, grad_Qt_vec_);
  if (!Mv_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Mv inverse did not converge (grad Qt calc)." << std::endl;
    exit(1);
  }
  grad_Qt_vec_ *= (4. / 3);
  pp_div_vec_ += grad_Qt_vec_;

  // Multiply pp_div_vec_ by nu
  // TODO(trevilo): This is ugly.  Find a better way.
  thermo_interface_->density->GetTrueDofs(rho_vec_);
  mu_total_gf_->GetTrueDofs(mu_vec_);
  if (dim_ == 2) {
    auto d_pp_div_vec = pp_div_vec_.ReadWrite();
    auto d_rho = rho_vec_.Read();
    auto d_mu = mu_vec_.Read();
    auto nnode = rho_vec_.Size();
    MFEM_FORALL(i, nnode, {
      d_pp_div_vec[i] *= (d_mu[i] / d_rho[i]);
      d_pp_div_vec[nnode + i] *= (d_mu[i] / d_rho[i]);
    });
  } else {
    auto d_pp_div_vec = pp_div_vec_.ReadWrite();
    auto d_rho = rho_vec_.Read();
    auto d_mu = mu_vec_.Read();
    auto nnode = rho_vec_.Size();
    MFEM_FORALL(i, nnode, {
      d_pp_div_vec[i] *= (d_mu[i] / d_rho[i]);
      d_pp_div_vec[nnode + i] *= (d_mu[i] / d_rho[i]);
      d_pp_div_vec[2 * nnode + i] *= (d_mu[i] / d_rho[i]);
    });
  }

  // Add ustar/dt contribution
  pp_div_vec_ += ustar_vec_;

  // Evaluate and add variable viscosity terms
  S_poisson_form_->Assemble();
  S_poisson_form_->ParallelAssemble(ress_vec_);
  Mv_rho_inv_->Mult(ress_vec_, S_poisson_vec_);
  pp_div_vec_ += S_poisson_vec_;

  pp_div_gf_->SetFromTrueDofs(pp_div_vec_);

  // rhs = div(pp_div_vec_)
  D_op_->Mult(pp_div_vec_, resp_vec_);

  // Add Qt term (rhs += -bd0 * Qt / dt)
  Ms_op_->AddMult(Qt_vec_, resp_vec_, -coeff_.bd0 / dt);

  // Add axisymmetric "forcing" term to rhs
  if (axisym_) {
    pp_div_gf_->HostRead();
    {
      // Copy radial components of pp_div_gf_.  We should be able to
      // just make pp_div_rad_comp_gf_ wrap pp_div_gf_, but that is
      // causing problems (that aren't understood) on some gpu
      // systems.  As a workaround, we copy instead.
      auto d_pp_div_rad = pp_div_rad_comp_gf_->Write();
      auto d_pp_div = pp_div_gf_->Read();
      MFEM_FORALL(i, pp_div_rad_comp_gf_->Size(), { d_pp_div_rad[i] = d_pp_div[i]; });
    }
    pp_div_rad_comp_gf_->HostRead();

    Faxi_poisson_form_->Update();
    Faxi_poisson_form_->Assemble();
    Faxi_poisson_form_->ParallelAssemble(Faxi_poisson_vec_);
    resp_vec_ += Faxi_poisson_vec_;
  }

  // Negate residual s.t. sign is consistent with -\nabla ( (1/\rho) \cdot \nabla p ) on LHS
  resp_vec_.Neg();

  // Add boundary terms to residual
  pp_div_bdr_form_->Assemble();
  pp_div_bdr_form_->ParallelAssemble(pp_div_bdr_vec_);
  resp_vec_.Add(1.0, pp_div_bdr_vec_);

  u_bdr_form_->Assemble();
  u_bdr_form_->ParallelAssemble(u_bdr_vec_);
  resp_vec_.Add(-coeff_.bd0 / dt, u_bdr_vec_);

  // Orthogonalize rhs if no Dirichlet BCs on pressure
  sw_press_.Start();
  if (pres_dbcs_.empty()) {
    Orthogonalize(resp_vec_, pfes_);
  }

  for (auto &pres_dbc : pres_dbcs_) {
    p_gf_->ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
  }

  // Isn't this the same as SetFromTrueDofs????
  pfes_->GetRestrictionMatrix()->MultTranspose(resp_vec_, *resp_gf_);

  Vector X1, B1;
  L_iorho_form_->FormLinearSystem(pres_ess_tdof_, *p_gf_, *resp_gf_, L_iorho_op_, X1, B1, 1);
  L_iorho_inv_->Mult(B1, X1);
  if (!L_iorho_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Poisson solve did not converge." << std::endl;
    exit(1);
  }
  sw_press_.Stop();

  // iter_spsolve = SpInv->GetNumIterations();
  // res_spsolve = SpInv->GetFinalNorm();
  L_iorho_form_->RecoverFEMSolution(X1, *resp_gf_, *p_gf_);

  // If the boundary conditions on the pressure are pure Neumann remove the
  // nullspace by removing the mean of the pressure solution. This is also
  // ensured by the OrthoSolver wrapper for the preconditioner which removes
  // the nullspace after every application.
  if (pres_dbcs_.empty()) {
    meanZero(*p_gf_);
  }
  p_gf_->GetTrueDofs(p_vec_);

  //------------------------------------------------------------------------
  // Step 4: Helmholtz solve for the velocity
  //------------------------------------------------------------------------
  resu_vec_ = 0.0;

  // Variable viscosity term
  S_mom_form_->Assemble();
  S_mom_form_->ParallelAssemble(resu_vec_);

  // -grad(p)
  G_op_->AddMult(p_vec_, resu_vec_, -1.0);

  // Add grad(mu * Qt) term
  Qt_vec_ *= mu_vec_;  // NB: pointwise multiply
  G_op_->AddMult(Qt_vec_, resu_vec_, 1.0 / 3.0);

  // rho * vstar / dt term
  Mv_rho_op_->AddMult(ustar_vec_, resu_vec_);

  for (auto &vel_dbc : vel_dbcs_) {
    u_next_gf_->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
  }

  vfes_->GetRestrictionMatrix()->MultTranspose(resu_vec_, *resu_gf_);

  Vector X2, B2;
  Hv_form_->FormLinearSystem(vel_ess_tdof_, *u_next_gf_, *resu_gf_, Hv_op_, X2, B2, 1);
  Hv_inv_->Mult(B2, X2);
  if (!Hv_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Helmholtz solve did not converge." << std::endl;
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

  // update gradients for turbulence model
  evaluateVelocityGradient();

  if (axisym_) {
    u_next_gf_->HostRead();
    {
      // Copy radial components of u_next_gf_.  Same comments here
      // about pp_div_rad_comp_gf_ above.
      auto d_u_next_rad = u_next_rad_comp_gf_->Write();
      auto d_u_next = u_next_gf_->Read();
      MFEM_FORALL(i, u_next_rad_comp_gf_->Size(), { d_u_next_rad[i] = d_u_next[i]; });
    }
    u_next_rad_comp_gf_->HostRead();

    // Update the Helmholtz operator and inverse
    Hv_bdfcoeff_.constant = coeff_.bd0 / dt;
    Hs_form_->Update();
    Hs_form_->Assemble();
    Hs_form_->FormSystemMatrix(swirl_ess_tdof_, Hs_op_);
    Hs_inv_->SetOperator(*Hs_op_);

    // Update scalar, density weighted mass matrix
    Ms_rho_form_->Update();
    Ms_rho_form_->Assemble();
    Ms_rho_form_->FormSystemMatrix(empty, Ms_rho_op_);

    // Update convection operator
    As_form_->Update();
    As_form_->Assemble();
    As_form_->FormSystemMatrix(empty, As_op_);

    rho_ur_ut_form_->Update();
    rho_ur_ut_form_->Assemble();

    swirl_var_viscosity_form_->Update();
    swirl_var_viscosity_form_->Assemble();

    // Form variable viscosity contribution (d(mu)/dr * utheta)
    swirl_var_viscosity_form_->ParallelAssemble(Faxi_poisson_vec_);

    // Form convection contribution, starting with -rho*ur*utheta
    rho_ur_ut_form_->ParallelAssemble(resp_vec_);
    resp_vec_ += Faxi_poisson_vec_;

    // res -> -d(mu)/dr * utheta - rho*ur*utheta
    resp_vec_.Neg();

    // Convection contribution to the rhs (NB: utheta_next_vec_ contains extrapolated utheta at this point)
    As_op_->AddMult(utheta_next_vec_, resp_vec_);

    // Unsteady contribution to RHS (overwrites extrapolated utheta b/c no longer necessary)
    {
      const double bd1idt = -coeff_.bd1 / dt;
      const double bd2idt = -coeff_.bd2 / dt;
      const double bd3idt = -coeff_.bd3 / dt;
      const auto d_u = utheta_vec_.Read();
      const auto d_um1 = utheta_m1_vec_.Read();
      const auto d_um2 = utheta_m2_vec_.Read();
      auto d_ut_next = utheta_next_vec_.Write();
      MFEM_FORALL(i, utheta_next_vec_.Size(),
                  { d_ut_next[i] = bd1idt * d_u[i] + bd2idt * d_um1[i] + bd3idt * d_um2[i]; });
    }

    Ms_rho_op_->AddMult(utheta_next_vec_, resp_vec_);

    // Apply swirl Dirichlet BC
    for (auto &swirl_dbc : swirl_dbcs_) {
      utheta_next_gf_->ProjectBdrCoefficient(*swirl_dbc.coeff, swirl_dbc.attr);
    }

    pfes_->GetRestrictionMatrix()->MultTranspose(resp_vec_, *resp_gf_);

    Vector Xs, Bs;
    Hs_form_->FormLinearSystem(swirl_ess_tdof_, *utheta_next_gf_, *resp_gf_, Hs_op_, Xs, Bs, 1);

    Hs_inv_->Mult(Bs, Xs);
    if (!Hs_inv_->GetConverged()) {
      if (rank0_) std::cout << "ERROR: Helmholtz solve did not converge." << std::endl;
      exit(1);
    }
    Hs_form_->RecoverFEMSolution(Xs, *resp_gf_, *utheta_next_gf_);
    utheta_next_gf_->GetTrueDofs(utheta_next_vec_);

    // Rotate values in solution history
    utheta_m2_vec_ = utheta_m1_vec_;
    utheta_m1_vec_ = utheta_vec_;

    // Update the current solution and corresponding GridFunction
    utheta_next_gf_->GetTrueDofs(utheta_next_vec_);
    utheta_vec_ = utheta_next_vec_;
    utheta_gf_->SetFromTrueDofs(utheta_vec_);
  }
}

//
//  - Solving for v (v* in f-p):
//    3(rho[m-1]v - 4(rho*u)[n] + (rho*u)[n-1])/(2dt) + C(rho[m-1]*v) = V(v) - G(p[m-1])
//  - if extrapolating:
//    v = 2dt/3 * { +V(u_ext[n+1]) - G(p[m-1]) - C(rho*u)_ext[n+1] } + 4/3*{ +(rho*u)[n] - (rho*u)[n-1] }
//  - even if C is implicit, convective vel must be extrapolated 
//  - f-p is set bdf2/ab2
//
void Tomboulides::predictionStep() {

  // For convenience, get time and dt
  const double time = coeff_.time;
  const double dt = coeff_.dt;

  // Ensure u_vec_ consistent with u_curr_gf_
  u_curr_gf_->GetTrueDofs(u_vec_);

  //------------------------------------------------------------------------
  // Step 1: Update any operators invalidated by changing data
  // external to this class
  // ------------------------------------------------------------------------

  // Update total viscosity field
  updateTotalViscosity();

  // Update density weighted mass
  Array<int> empty;
  //Mv_rho_form_->Update();
  //Mv_rho_form_->Assemble();
  //Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  //Mv_rho_inv_->SetOperator(*Mv_rho_op_);

  // Update the Helmholtz operator and inverse
  //sw_helm_.Start();
  //Hv_bdfcoeff_.constant = coeff_.bd0 / dt;
  //Hv_form_->Update();
  //Hv_form_->Assemble();
  //Hv_form_->FormSystemMatrix(vel_ess_tdof_, Hv_op_);
  //Hv_inv_->SetOperator(*Hv_op_);
  //sw_helm_.Stop();

  //------------------------------------------------------------------------
  // Step 2: Compute vstar / dt (as in eqn 2.3 from Tomboulides)
  // ------------------------------------------------------------------------

  // Evaluate the forcing at the end of the time step
  for (auto &force : forcing_terms_) {
    force.coeff->SetTime(time + dt);
  }
  forcing_form_->Assemble();
  forcing_form_->ParallelAssemble(forcing_vec_);

  // Evaluate convection term in the velocity eqn
  // need appropriate rho at each stage here as well
  // handled via coeff updates (appropriate because term
  // is computed in integrated weak-form)

  /*
  thermo_interface_->rn->GetTrueDofs(rho_vec_);
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);
  Nconv_rho_form_->Update();
  //Nconv_form_->Assemble();
  Nconv_rho_form_->Mult(u_vec_, N_vec_);  

  thermo_interface_->rnm1->GetTrueDofs(rho_vec_);
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);
  Nconv_rho_form_->Update();
  //Nconv_form_->Assemble();  
  Nconv_rho_form_->Mult(um1_vec_, Nm1_vec_);

  thermo_interface_->rnm2->GetTrueDofs(rho_vec_);
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);
  Nconv_rho_form_->Update();
  //Nconv_form_->Assemble();  
  Nconv_rho_form_->Mult(um2_vec_, Nm2_vec_);
  */

  // very hacky approach... not safe for gpu!
  // div(rho[n]*u[n]*u[n])  
  thermo_interface_->rn->GetTrueDofs(rho_vec_);
  {
    const auto d_r = rho_vec_.Read();
    auto d_sqrtr = tmpR0_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_sqrtr[i] = std::sqrt(d_r[i]); });    
  }
  tmpR1_.Set(1.0,u_vec_);
  {
    const auto d_sqrtr = tmpR0_.Read();
    auto d_rhu = tmpR1_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + tmpR0_.Size()] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + 2*tmpR0_.Size()] *= d_sqrtr[i]; });    
  }
  Nconv_form_->Mult(tmpR1_, N_vec_);  

  // div(rho[n-1]*u[n-1]*u[n-1])
  thermo_interface_->rnm1->GetTrueDofs(rho_vec_);
  {
    const auto d_r = rho_vec_.Read();
    auto d_sqrtr = tmpR0_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_sqrtr[i] = std::sqrt(d_r[i]); });    
  }
  tmpR1_.Set(1.0,um1_vec_);
  {
    const auto d_sqrtr = tmpR0_.Read();
    auto d_rhu = tmpR1_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + tmpR0_.Size()] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + 2*tmpR0_.Size()] *= d_sqrtr[i]; });    
  }
  Nconv_form_->Mult(tmpR1_, Nm1_vec_);  

  // div(rho[n-2]*u[n-2]*u[n-2])
  thermo_interface_->rnm2->GetTrueDofs(rho_vec_);
  {
    const auto d_r = rho_vec_.Read();
    auto d_sqrtr = tmpR0_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_sqrtr[i] = std::sqrt(d_r[i]); });    
  }
  tmpR1_.Set(1.0,um2_vec_);
  {
    const auto d_sqrtr = tmpR0_.Read();
    auto d_rhu = tmpR1_.ReadWrite();
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + tmpR0_.Size()] *= d_sqrtr[i]; });
    MFEM_FORALL(i, tmpR0_.Size(), { d_rhu[i + 2*tmpR0_.Size()] *= d_sqrtr[i]; });    
  }
  Nconv_form_->Mult(tmpR1_, Nm2_vec_);  


  // extrapolate to [n+1] state (minus here because the negative sign is no longer included in nlcoeff_)
  {
    const auto d_N = N_vec_.Read();
    const auto d_Nm1 = Nm1_vec_.Read();
    const auto d_Nm2 = Nm2_vec_.Read();
    auto d_force = forcing_vec_.ReadWrite();
    const auto ab1 = coeff_.ab1;
    const auto ab2 = coeff_.ab2;
    const auto ab3 = coeff_.ab3;
    // MFEM_FORALL(i, forcing_vec_.Size(), { d_force[i] += (ab1 * d_N[i] + ab2 * d_Nm1[i] + ab3 * d_Nm2[i]); });
    MFEM_FORALL(i, forcing_vec_.Size(), { d_force[i] -= (ab1*d_N[i] + ab2*d_Nm1[i] + ab3*d_Nm2[i]); });    
  }

  if (axisym_) {
    // Extrapolate the swirl velocity
    {
      const auto d_u = utheta_vec_.Read();
      const auto d_um1 = utheta_m1_vec_.Read();
      const auto d_um2 = utheta_m2_vec_.Read();
      auto d_uext = utheta_next_vec_.Write();
      const auto ab1 = coeff_.ab1;
      const auto ab2 = coeff_.ab2;
      const auto ab3 = coeff_.ab3;
      MFEM_FORALL(i, utheta_next_vec_.Size(), { d_uext[i] = ab1 * d_u[i] + ab2 * d_um1[i] + ab3 * d_um2[i]; });
    }
    utheta_next_gf_->SetFromTrueDofs(utheta_next_vec_);

    // evaluate the rho u_{\theta}^2 term (from convection)
    ur_conv_axi_form_->Assemble();
    ur_conv_axi_form_->ParallelAssemble(ur_conv_forcing_vec_);
    forcing_vec_ += ur_conv_forcing_vec_;
  }

  // vstar / dt += BDF contribution
  // not sure if pointwise rho*u is okay here...
  // if not, form Mv_rho_ operators for each rho term and follow with an Minv
  // (or just shift up above preceding Minv...)

  thermo_interface_->rn->GetTrueDofs(rho_vec_);
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);  
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  Mv_rho_form_->Mult(u_vec_, tmpR1_);
  forcing_vec_.Add(-coeff_.bd1 / dt, tmpR1_);

  thermo_interface_->rnm1->GetTrueDofs(rho_vec_);  
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);  
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  Mv_rho_form_->Mult(um1_vec_, tmpR1_);
  forcing_vec_.Add(-coeff_.bd1 / dt, tmpR1_);

  thermo_interface_->rnm2->GetTrueDofs(rho_vec_);    
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);  
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  Mv_rho_form_->Mult(um2_vec_, tmpR1_);
  forcing_vec_.Add(-coeff_.bd3 / dt, tmpR1_);
    
  /*{
    const double bd1idt = -coeff_.bd1 / dt;
    const double bd2idt = -coeff_.bd2 / dt;
    const double bd3idt = -coeff_.bd3 / dt;
    const auto d_u = u_vec_.Read();
    const auto d_um1 = um1_vec_.Read();
    const auto d_um2 = um2_vec_.Read();
    const auto d_rn = rn_vec_.Read();
    const auto d_rnm1 = rnm1_vec_.Read();
    const auto d_rnm2 = rnm2_vec_.Read();    
    auto d_ustar = ustar_vec_.ReadWrite();
    MFEM_FORALL(i, ustar_vec_.Size(), { d_ustar[i] += bd1idt*d_rn[i]*d_u[i] + bd2idt*d_rnm1[i]*d_um1[i] + bd3idt*d_rnm2[i]*d_um2[i]; });
    }*/
  
  // vstar / dt = M^{-1} (us + extrapolated nonlinear + forcing)
  //Mv_inv_->Mult(forcing_vec_, ustar_vec_);
  //if (!Mv_inv_->GetConverged()) {
  //  if (rank0_) std::cout << "ERROR: Mv inverse did not converge." << std::endl;
  //  exit(1);
  //}

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
  if (axisym_) {
    ComputeCurlAxi(*u_next_gf_, *curl_gf_, false);
    ComputeCurlAxi(*curl_gf_, *curlcurl_gf_, true);
  } else {
    if (dim_ == 2) {
      ComputeCurl2D(*u_next_gf_, *curl_gf_, false);
      ComputeCurl2D(*curl_gf_, *curlcurl_gf_, true);
    } else {
      ComputeCurl3D(*u_next_gf_, *curl_gf_);
      ComputeCurl3D(*curl_gf_, *curlcurl_gf_);
    }
  }
  curlcurl_gf_->GetTrueDofs(pp_div_vec_);

  // Negate b/c term that appears in residual is -curl curl u
  pp_div_vec_.Neg();
  Mv_form_->Mult(pp_div_vec_,tmpR1_);
  pp_div_vec_.Set(1.0,tmpR1_);

  // TODO(trevilo): This calculation of the viscous terms
  // assumes that mu is constant (i.e., there are terms that are
  // proportional to grad(mu) that are not included below).
  // This assumption has been corrected by adding the necessary terms
  // below.

  // Compute and add grad(Qt) contribution
  // assert(thermo_interface_->thermal_divergence != nullptr);
  // thermo_interface_->thermal_divergence->GetTrueDofs(Qt_vec_);  
  // setting Qt as extrapolated div(u)_[n+1] for all-mach...
  D_op_->Mult(uext_vec_, tmpR0_);
  Ms_inv_->Mult(tmpR0_,Qt_vec_);
  //G_op_->Mult(Qt_vec_, resu_vec_);
  // Mv_inv_->Mult(resu_vec_, grad_Qt_vec_);
  G_op_->Mult(Qt_vec_, grad_Qt_vec_);
  //if (!Mv_inv_->GetConverged()) {
  //  if (rank0_) std::cout << "ERROR: Mv inverse did not converge (grad Qt calc)." << std::endl;
  //  exit(1);
  //}
  grad_Qt_vec_ *= (4. / 3.);
  pp_div_vec_ += grad_Qt_vec_;

  // Multiply pp_div_vec_ by nu
  // TODO(trevilo): This is ugly.  Find a better way.
  thermo_interface_->density->GetTrueDofs(rho_vec_);
  mu_total_gf_->GetTrueDofs(mu_vec_);
  if (dim_ == 2) {
    auto d_pp_div_vec = pp_div_vec_.ReadWrite();
    auto d_rho = rho_vec_.Read();
    auto d_mu = mu_vec_.Read();
    auto nnode = rho_vec_.Size();
    MFEM_FORALL(i, nnode, {
	//d_pp_div_vec[i] *= (d_mu[i] / d_rho[i]);
	//d_pp_div_vec[nnode + i] *= (d_mu[i] / d_rho[i]);
      d_pp_div_vec[i] *= (d_mu[i]);
      d_pp_div_vec[nnode + i] *= (d_mu[i]);
    });
  } else {
    auto d_pp_div_vec = pp_div_vec_.ReadWrite();
    auto d_rho = rho_vec_.Read();
    auto d_mu = mu_vec_.Read();
    auto nnode = rho_vec_.Size();
    MFEM_FORALL(i, nnode, {
	//d_pp_div_vec[i] *= (d_mu[i] / d_rho[i]);
	//d_pp_div_vec[nnode + i] *= (d_mu[i] / d_rho[i]);
	//d_pp_div_vec[2 * nnode + i] *= (d_mu[i] / d_rho[i]);
      d_pp_div_vec[i] *= (d_mu[i]);
      d_pp_div_vec[nnode + i] *= (d_mu[i]);
      d_pp_div_vec[2 * nnode + i] *= (d_mu[i]);   
    });
  }

  // Add contribution to vstar (beta0/dt * rho[m+1] * vstar = ...)
  ustar_vec_ += pp_div_vec_;

  // Evaluate and add variable viscosity terms
  S_poisson_form_->Assemble();
  S_poisson_form_->ParallelAssemble(ress_vec_);
  // Note: hitting with Mrho^-1 in tomb because this was used for p-p and not ustar
  // Mv_rho_inv_->Mult(ress_vec_, S_poisson_vec_);
  // ustar_vec_ += S_poisson_vec_;
  ustar_vec_ += ress_vec_;

  // add -grad(p[n])
  G_op_->AddMult(p_vec_, ustar_vec_, -1.0);
  
  // now full explicit rhs gets hit with a (dt/beta0)
  ustar_vec_ *= dt / coeff_.bd0;

  // Extrapolate the rho field to rho[n+1]
  thermo_interface_->rn->GetTrueDofs(rho_vec_);
  rho_vec_ *= coeff_.ab1;
  thermo_interface_->rnm1->GetTrueDofs(tmpR0_);
  tmpR0_ *= coeff_.ab2;
  rho_vec_ += tmpR0_;
  thermo_interface_->rnm2->GetTrueDofs(tmpR0_);
  tmpR0_ *= coeff_.ab3;
  rho_vec_ += tmpR0_;

  // update rho-weighted mass matrix
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);  
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  Mv_rho_inv_->SetOperator(*Mv_rho_op_);

  // solve system with ustar = Mv_rho_inv_(rhs)
  resu_vec_.Set(1.0,ustar_vec_);
  Mv_rho_inv_->Mult(resu_vec_,ustar_vec_);
  if (!Mv_rho_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Predictor  Mv_inv inverse did not converge." << std::endl;
    exit(1);
  }

  // and complete
  ustar_gf_->SetFromTrueDofs(ustar_vec_);
  
} // end predictorStep


//
// Correct v* using the pressure-solve:
//  -- v[n+1] = v' + ustar
//  -- rho_ext[n+1] * v' = - dt/bdf0 * grad(p')
//  -- (bdf0/dt * C_rho) * p' + div(C_rho * ustar * p') = dt/dbf0 * nabla^2(p') - mass_imbalance
//  pressure should be updated (w/p') by this point
//
void Tomboulides::correctionStep() {

  // For convenience, get time and dt
  const double time = coeff_.time;
  const double dt = coeff_.dt;

  // Extrapolate the rho field to rho[n+1]
  thermo_interface_->rn->GetTrueDofs(rho_vec_);
  rho_vec_ *= coeff_.ab1;
  thermo_interface_->rnm1->GetTrueDofs(tmpR0_);
  tmpR0_ *= coeff_.ab2;
  rho_vec_ += tmpR0_;
  thermo_interface_->rnm2->GetTrueDofs(tmpR0_);
  tmpR0_ *= coeff_.ab3;
  rho_vec_ += tmpR0_;

  // update rho-weighted mass matrix
  Array<int> empty;    
  rho_tmp_gf_->SetFromTrueDofs(rho_vec_);  
  Mv_rho_form_->Update();
  Mv_rho_form_->Assemble();
  Mv_rho_form_->FormSystemMatrix(empty, Mv_rho_op_);
  Mv_rho_inv_->SetOperator(*Mv_rho_op_);

  // pressure gradient
  (thermo_interface_->pressure)->GetTrueDofs(tmpR0_);
  G_op_->Mult(tmpR0_, resu_vec_);
  resu_vec_ *= -dt/coeff_.bd0;

  // solve system for v'
  //Mv_rho_inv_->Mult(resu_vec_,tmpR1_);

  for (auto &vel_dbc : vel_dbcs_) {
    u_next_gf_->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
  }

  vfes_->GetRestrictionMatrix()->MultTranspose(resu_vec_, *resu_gf_);

  Vector Xu, Bu;
  Mv_rho_form_->FormLinearSystem(vel_ess_tdof_, *u_next_gf_, *resu_gf_, Mv_rho_op_, Xu, Bu, 1);
  Mv_rho_inv_->Mult(Bu, Xu);
  if (!Mv_rho_inv_->GetConverged()) {
    if (rank0_) std::cout << "ERROR: Mass solve for u-prime did not converge." << std::endl;
    exit(1);
  }
  Mv_rho_form_->RecoverFEMSolution(Xu, *resu_gf_, *u_next_gf_);
  
  // update full velocity
  u_next_gf_->GetTrueDofs(tmpR1_);    
  tmpR1_.Add(1.0,ustar_vec_);
  u_next_gf_->SetFromTrueDofs(tmpR1_);

  // May need a full momentum solve here instead of the update,
  // as-is, u[n+1] is consistent with grad(P),
  // however, if rho* and rho[n+1] are different (enough), mass will not be conserved

} //end correction step

double Tomboulides::computeL2Error() const {
  double err = -1.0;
  if (ic_string_ == "tgv2d") {
    std::cout << "Evaluating TGV2D error..." << std::endl;
    vfptr user_func = vel_ic(ic_string_);
    VectorFunctionCoefficient u_excoeff(nvel_, user_func);
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

void Tomboulides::updateTotalViscosity() {
  *mu_total_gf_ = (*thermo_interface_->viscosity);
  *mu_total_gf_ += (*turbModel_interface_->eddy_viscosity);
  *mu_total_gf_ *= (*sponge_interface_->visc_multiplier);
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

void Tomboulides::addVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr) {
  vel_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!vel_ess_attr_[i]);
      vel_ess_attr_[i] = 1;
    }
  }
}

void Tomboulides::addVelDirichletBC(void (*f)(const Vector &, double, Vector &), Array<int> &attr) {
  addVelDirichletBC(new VectorFunctionCoefficient(dim_, f), attr);
}

void Tomboulides::addVelDirichletBC(std::function<void(const Vector &, double, Vector &)> f, Array<int> &attr) {
  addVelDirichletBC(new VectorFunctionCoefficient(dim_, f), attr);
}

/// Add a Dirichlet boundary condition to the pressure field.
void Tomboulides::addPresDirichletBC(double p, Array<int> &attr) {
  pres_dbcs_.emplace_back(attr, new ConstantCoefficient(p));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!pres_ess_attr_[i]);  // if pres_ess_attr[i] already set, fail b/c duplicate
      pres_ess_attr_[i] = 1;
    }
  }
}

void Tomboulides::addSwirlDirichletBC(double ut, mfem::Array<int> &attr) {
  assert(axisym_);
  swirl_dbcs_.emplace_back(attr, new ConstantCoefficient(ut));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!swirl_ess_attr_[i]);  // if swirl_ess_attr[i] already set, fail b/c duplicate
      swirl_ess_attr_[i] = 1;
    }
  }
}

double Tomboulides::maxVelocityMagnitude() {
  double local_max_vel_magnitude = 0.0;
  double global_max_vel_magnitude = 0.0;

  u_curr_gf_->GetTrueDofs(u_vec_);

  int n_scalar_dof = u_vec_.Size() / dim_;

  const double *vel = u_vec_.HostRead();

  for (int i = 0; i < n_scalar_dof; i++) {
    const double ux = vel[i];
    const double uy = vel[n_scalar_dof + i];
    double uz = 0.0;
    if (dim_ == 3) {
      uz = vel[2 * n_scalar_dof + i];
    }
    const double vmag = sqrt(ux * ux + uy * uy + uz * uz);
    if (vmag > local_max_vel_magnitude) {
      local_max_vel_magnitude = vmag;
    }
  }

  MPI_Reduce(&local_max_vel_magnitude, &global_max_vel_magnitude, 1, MPI_DOUBLE, MPI_MAX, 0, pfes_->GetComm());

  return global_max_vel_magnitude;
}

void Tomboulides::evaluateVelocityGradient() {
  setScalarFromVector(u_next_vec_, 0, &tmpR0_);
  G_op_->Mult(tmpR0_, tmpR1_);
  Mv_inv_->Mult(tmpR1_, gradU_);
  setScalarFromVector(u_next_vec_, 1, &tmpR0_);
  G_op_->Mult(tmpR0_, tmpR1_);
  Mv_inv_->Mult(tmpR1_, gradV_);
  if (dim_ == 3) {
    setScalarFromVector(u_next_vec_, 2, &tmpR0_);
    G_op_->Mult(tmpR0_, tmpR1_);
    Mv_inv_->Mult(tmpR1_, gradW_);
  }
  gradU_gf_->SetFromTrueDofs(gradU_);
  gradV_gf_->SetFromTrueDofs(gradV_);
  gradW_gf_->SetFromTrueDofs(gradW_);
}
