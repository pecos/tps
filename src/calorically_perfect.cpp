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

#include "calorically_perfect.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>
#include <mfem/general/forall.hpp>

#include "cases.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

/// forward declarations (if any)

MFEM_HOST_DEVICE double Sutherland(const double T, const double mu_star, const double T_star, const double S_star) {
  const double T_rat = T / T_star;
  const double T_rat_32 = T_rat * sqrt(T_rat);
  const double S_rat = (T_star + S_star) / (T + S_star);
  return mu_star * T_rat_32 * S_rat;
}

CaloricallyPerfectThermoChem::CaloricallyPerfectThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts,
                                                           temporalSchemeCoefficients &time_coeff,
                                                           ParGridFunction *gridScale, TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;
  gridScale_gf_ = gridScale;

  std::string visc_model;
  tpsP_->getInput("loMach/calperfect/viscosity-model", visc_model, std::string("sutherland"));
  if (visc_model == "sutherland") {
    constant_viscosity_ = false;
    tpsP_->getInput("loMach/calperfect/sutherland/mu0", mu0_, 1.68e-5);
    tpsP_->getInput("loMach/calperfect/sutherland/T0", sutherland_T0_, 273.0);
    tpsP_->getInput("loMach/calperfect/sutherland/S0", sutherland_S0_, 110.4);
  } else if (visc_model == "constant") {
    constant_viscosity_ = true;
    tpsP_->getInput("loMach/calperfect/constant-visc/mu0", mu0_, 1.68e-5);
  } else {
    if (rank0_) {
      std::cout << "ERROR: loMach/calperfect/viscosity-model = " << visc_model << " not supported." << std::endl;
    }
    assert(false);
    exit(1);
  }

  std::string density_model;
  tpsP_->getInput("loMach/calperfect/density-model", density_model, std::string("ideal-gas"));
  if (density_model == "ideal-gas") {
    constant_density_ = false;
  } else if (density_model == "constant") {
    constant_density_ = true;
    tpsP_->getInput("loMach/calperfect/constant-density", static_rho_, 1.0);
  } else {
    if (rank0_) {
      std::cout << "ERROR: loMach/calperfect/density-model = " << density_model << " not supported." << std::endl;
    }
    assert(false);
    exit(1);
  }

  tpsP_->getInput("loMach/ambientPressure", ambient_pressure_, 101325.0);
  thermo_pressure_ = ambient_pressure_;
  dtP_ = 0.0;

  tpsP_->getInput("initialConditions/temperature", T_ic_, 300.0);

  tpsP_->getInput("loMach/calperfect/Rgas", Rgas_, 287.0);
  tpsP_->getInput("loMach/calperfect/Prandtl", Pr_, 0.71);
  tpsP_->getInput("loMach/calperfect/gamma", gamma_, 1.4);
  invPr_ = 1.0 / Pr_;

  Cp_ = gamma_ * Rgas_ / (gamma_ - 1);

  filter_temperature_ = loMach_opts->filterTemp;
  filter_alpha_ = loMach_opts->filterWeight;
  filter_cutoff_modes_ = loMach_opts->nFilter;

  tpsP_->getInput("loMach/openSystem", domain_is_open_, false);

  // NOTE: this should default to FALSE (but would break all related test cases...)
  tpsP_->getInput("loMach/calperfect/numerical-integ", numerical_integ_, true);

  tpsP_->getInput("loMach/calperfect/linear-solver-rtol", rtol_, default_rtol_);
  tpsP_->getInput("loMach/calperfect/linear-solver-max-iter", max_iter_, default_max_iter_);
  tpsP_->getInput("loMach/calperfect/linear-solver-verbosity", pl_solve_, 0);

  // not deleting above block to maintain backwards-compatability
  tpsP_->getInput("loMach/calperfect/hsolve-rtol", hsolve_rtol_, rtol_);
  tpsP_->getInput("loMach/calperfect/hsolve-atol", hsolve_atol_, default_atol_);
  tpsP_->getInput("loMach/calperfect/hsolve-max-iter", hsolve_max_iter_, max_iter_);
  tpsP_->getInput("loMach/calperfect/hsolve-verbosity", hsolve_pl_, pl_solve_);

  tpsP_->getInput("loMach/calperfect/msolve-rtol", mass_inverse_rtol_, rtol_);
  tpsP_->getInput("loMach/calperfect/msolve-atol", mass_inverse_atol_, default_atol_);
  tpsP_->getInput("loMach/calperfect/msolve-max-iter", mass_inverse_max_iter_, max_iter_);
  tpsP_->getInput("loMach/calperfect/msolve-verbosity", mass_inverse_pl_, pl_solve_);

  // artificial diffusion (SUPG)
  tpsP_->getInput("loMach/calperfect/streamwise-stabilization", sw_stab_, false);
  tpsP_->getInput("loMach/calperfect/Reh_factor", Reh_factor_, 0.5);
  tpsP_->getInput("loMach/calperfect/Reh_offset", Reh_offset_, 1.0);
}

CaloricallyPerfectThermoChem::~CaloricallyPerfectThermoChem() {
  // allocated in initializeOperators
  delete sfes_filter_;
  delete sfec_filter_;
  delete LQ_bdry_;
  delete LQ_form_;
  delete MqInv_;
  delete MqInvPC_;
  delete Mq_form_;
  delete HtInv_;
  delete HtInvPC_;
  delete MsInv_;
  delete MsInvPC_;
  delete Ht_form_;
  delete MsRho_form_;
  delete Ms_form_;
  delete At_form_;
  delete rhou_coeff_;
  delete rhon_next_coeff_;
  delete un_next_coeff_;
  delete kap_gradT_coeff_;
  delete gradT_coeff_;
  delete mut_coeff_;
  delete mult_coeff_;
  delete thermal_diff_coeff_;
  delete thermal_diff_sum_coeff_;
  delete thermal_diff_total_coeff_;
  delete rho_over_dt_coeff_;
  delete rho_coeff_;

  delete umag_coeff_;
  delete gscale_coeff_;
  delete visc_coeff_;
  delete visc_inv_coeff_;
  delete reh1_coeff_;
  delete reh2_coeff_;
  delete Reh_coeff_;
  delete csupg_coeff_;
  delete uw1_coeff_;
  delete uw2_coeff_;
  delete upwind_coeff_;
  delete swdiff_coeff_;
  delete supg_coeff_;

  // allocated in initializeSelf
  delete sfes_;
  delete sfec_;
}

void CaloricallyPerfectThermoChem::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing CaloricallyPerfectThermoChem solver.\n");

  //-----------------------------------------------------
  // 1) Prepare the required finite element objects
  //-----------------------------------------------------
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  // Check if fully periodic mesh
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    temp_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    temp_ess_attr_ = 0;

    Qt_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    Qt_ess_attr_ = 0;
  }
  if (rank0_) grvy_printf(ginfo, "CaloricallyPerfectThermoChem paces constructed...\n");

  int sfes_truevsize = sfes_->GetTrueVSize();

  Qt_.SetSize(sfes_truevsize);
  Qt_ = 0.0;
  Qt_gf_.SetSpace(sfes_);
  Qt_gf_ = 0.0;

  Tn_.SetSize(sfes_truevsize);
  Tn_next_.SetSize(sfes_truevsize);

  Tnm1_.SetSize(sfes_truevsize);
  Tnm2_.SetSize(sfes_truevsize);

  // temp convection terms
  NTn_.SetSize(sfes_truevsize);
  NTnm1_.SetSize(sfes_truevsize);
  NTnm2_.SetSize(sfes_truevsize);
  NTn_ = 0.0;
  NTnm1_ = 0.0;
  NTnm2_ = 0.0;

  Text_.SetSize(sfes_truevsize);
  Text_gf_.SetSpace(sfes_);

  resT_gf_.SetSpace(sfes_);
  resT_.SetSize(sfes_truevsize);
  resT_ = 0.0;

  Tn_next_gf_.SetSpace(sfes_);
  Tn_gf_.SetSpace(sfes_);
  Tnm1_gf_.SetSpace(sfes_);
  Tnm2_gf_.SetSpace(sfes_);

  rn_.SetSize(sfes_truevsize);
  rn_ = 1.0;
  rn_gf_.SetSpace(sfes_);
  rn_gf_ = 1.0;

  visc_.SetSize(sfes_truevsize);
  visc_ = 1.0e-12;

  kappa_.SetSize(sfes_truevsize);
  kappa_ = 1.0e-12;

  kappa_gf_.SetSpace(sfes_);
  kappa_gf_ = 0.0;

  visc_gf_.SetSpace(sfes_);
  visc_gf_ = 0.0;

  tmpR0_.SetSize(sfes_truevsize);
  tmpR0b_.SetSize(sfes_truevsize);

  R0PM0_gf_.SetSpace(sfes_);

  rhoDt_gf_.SetSpace(sfes_);

  if (rank0_) grvy_printf(ginfo, "CaloricallyPerfectThermoChem vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &visc_gf_;
  toFlow_interface_.thermal_divergence = &Qt_gf_;
  toTurbModel_interface_.density = &rn_gf_;
  toTurbModel_interface_.viscosity = &visc_gf_;
  if (rank0_) {
    std::cout << "exports set..." << endl;
  }

  tpsP_->getInput("loMach/calperfect/numerical-integ", numerical_integ_, true);

  //-----------------------------------------------------
  // 2) Set the initial condition
  //-----------------------------------------------------

  // Notes:
  //
  // 1) If need arises, can provide spatially varying IC as follows:
  //
  // FunctionCoefficient t_ic_coef(temp_ic);
  // Tn_gf_.ProjectCoefficient(t_ic_coef);
  //
  // where temp_ic is a function that returns the IC temperature at a
  // point in space.
  //
  // 2) For restarts, this IC is overwritten by the restart field,
  // which is read later.

  tpsP_->getInput("loMach/calperfect/ic", ic_string_, std::string(""));

  // set IC if we have one at this point (see cases.cpp for options)
  if (!ic_string_.empty()) {
    sfptr user_func = temp_ic(ic_string_);
    FunctionCoefficient t_excoeff(user_func);
    t_excoeff.SetTime(0.0);
    Tn_gf_.ProjectCoefficient(t_excoeff);
  } else {
    ConstantCoefficient t_ic_coef;
    t_ic_coef.constant = T_ic_;
    Tn_gf_.ProjectCoefficient(t_ic_coef);
  }

  Tn_gf_.GetTrueDofs(Tn_);
  Tnm1_gf_.SetFromTrueDofs(Tn_);
  Tnm2_gf_.SetFromTrueDofs(Tn_);
  Tnm1_gf_.GetTrueDofs(Tnm1_);
  Tnm2_gf_.GetTrueDofs(Tnm2_);

  //-----------------------------------------------------
  // 3) Set the boundary conditions
  //-----------------------------------------------------
  int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // inlet bc
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
        double temperature_value;
        tpsP_->getRequiredInput((basepath + "/temperature").c_str(), temperature_value);
        if (rank0_) {
          std::cout << "Calorically Perfect: Setting uniform Dirichlet temperature on patch = " << patch << std::endl;
        }
        AddTempDirichletBC(temperature_value, inlet_attr);

      } else if (type == "interpolate") {
        Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
        inlet_attr = 0;
        inlet_attr[patch - 1] = 1;
        temperature_bc_field_ = new GridFunctionCoefficient(extData_interface_->Tdata);
        if (rank0_) {
          std::cout << "Calorically Perfect: Setting interpolated Dirichlet temperature on patch = " << patch
                    << std::endl;
        }
        AddTempDirichletBC(temperature_bc_field_, inlet_attr);

        // Force the IC to agree with the interpolated inlet BC
        //
        // NB: It is still possible for Tn_gf_ on a restart to
        // disagree with this BC.  Specifically, since the restart
        // field is read after this projection, if it does not satisfy
        // this BC, there will be a discrepancy (which will be
        // eliminated after the first step).
        Tn_gf_.ProjectBdrCoefficient(*temperature_bc_field_, inlet_attr);
      } else {
        if (rank0_) {
          std::cout << "ERROR: Calorically Perfect inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      }
    }
  }
  if (rank0_) {
    std::cout << "inlet bc set..." << endl;
  }

  // outlet bc
  {
    Array<int> attr_outlet(pmesh_->bdr_attributes.Max());
    attr_outlet = 0;

    // No code for this yet, so die if detected
    // assert(numOutlets == 0);

    // But... outlets will just get homogeneous Neumann on T, so
    // basically need to do nothing.
  }

  // Wall BCs
  {
    if (rank0_) std::cout << "There are " << pmesh_->bdr_attributes.Max() << " boundary attributes" << std::endl;
    Array<int> attr_wall(pmesh_->bdr_attributes.Max());
    attr_wall = 0;

    for (int i = 1; i <= numWalls; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/wall" + std::to_string(i));

      tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
      tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

      if (type == "viscous_isothermal") {
        if (rank0_) std::cout << "Adding patch = " << patch << " to isothermal wall list" << std::endl;

        attr_wall = 0;
        attr_wall[patch - 1] = 1;

        double Twall;
        tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Twall);

        ConstantCoefficient *Twall_coeff = new ConstantCoefficient();
        Twall_coeff->constant = Twall;
        AddTempDirichletBC(Twall_coeff, attr_wall);

        ConstantCoefficient *Qt_bc_coeff = new ConstantCoefficient();
        Qt_bc_coeff->constant = 0.0;
        AddQtDirichletBC(Qt_bc_coeff, attr_wall);
      }
    }
    if (rank0_) std::cout << "Temp wall bc completed: " << numWalls << endl;
  }

  sfes_->GetEssentialTrueDofs(temp_ess_attr_, temp_ess_tdof_);
  sfes_->GetEssentialTrueDofs(Qt_ess_attr_, Qt_ess_tdof_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem Essential true dof step" << endl;
}

void CaloricallyPerfectThermoChem::initializeOperators() {
  dt_ = time_coeff_.dt;

  // polynomial order for smoother of precons
  smoother_poly_order_ = order_ + 1;

  Array<int> empty;

  // GLL integration rule (Numerical Integration)
  // unsteady: p+p [+p] = 2p [3p]
  // convection: p+p+(p-1) [+p] = 3p-1 [4p-1]
  // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
  const IntegrationRule &ir_i = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 2 * order_ + 1);
  const IntegrationRule &ir_nli = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 4 * order_);
  const IntegrationRule &ir_di = gll_rules_.Get(sfes_->GetFE(0)->GetGeomType(), 3 * order_ - 1);
  if (rank0_) std::cout << "Integration rules set" << endl;

  // coefficients for operators
  rho_coeff_ = new GridFunctionCoefficient(&rn_gf_);

  rhoDt_gf_ = rn_gf_;
  rhoDt_gf_ /= dt_;
  rho_over_dt_coeff_ = new GridFunctionCoefficient(&rhoDt_gf_);

  // thermal_diff_coeff.constant = thermal_diff;
  thermal_diff_coeff_ = new GridFunctionCoefficient(&kappa_gf_);
  mut_coeff_ = new GridFunctionCoefficient(turbModel_interface_->eddy_viscosity);
  thermal_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *thermal_diff_coeff_, invPr_, 1.0);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);
  thermal_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *thermal_diff_sum_coeff_);
  gradT_coeff_ = new GradientGridFunctionCoefficient(&Tn_next_gf_);
  kap_gradT_coeff_ = new ScalarVectorProductCoefficient(*thermal_diff_total_coeff_, *gradT_coeff_);

  // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho u \cdot \nabla \phi_j
  un_next_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhon_next_coeff_ = new GridFunctionCoefficient(&rn_gf_);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rhon_next_coeff_, *un_next_coeff_);

  // artifical diffusion coefficients
  if (sw_stab_) {
    umag_coeff_ = new VectorMagnitudeCoefficient(*un_next_coeff_);
    gscale_coeff_ = new GridFunctionCoefficient(gridScale_gf_);
    visc_coeff_ = new GridFunctionCoefficient(&visc_gf_);
    visc_inv_coeff_ = new PowerCoefficient(*visc_coeff_, -1.0);

    // compute Reh
    reh1_coeff_ = new ProductCoefficient(*rho_coeff_, *visc_inv_coeff_);
    reh2_coeff_ = new ProductCoefficient(*reh1_coeff_, *gscale_coeff_);
    Reh_coeff_ = new ProductCoefficient(*reh2_coeff_, *umag_coeff_);

    // Csupg
    // auto csupgLambda = std::bind(csupgFactor, std::placeholders::_1,  Reh_factor_, Reh_offset_);
    std::function<double(double)> csupgLambda = std::bind(csupgFactor, std::placeholders::_1,  Reh_factor_, Reh_offset_);
    csupg_coeff_ = new ExtTransformedCoefficient(Reh_coeff_, csupgLambda);

    // compute upwind magnitude
    uw1_coeff_ = new ProductCoefficient(*rho_coeff_, *csupg_coeff_);
    uw2_coeff_ = new ProductCoefficient(*uw1_coeff_, *gscale_coeff_);
    upwind_coeff_ = new ProductCoefficient(*uw2_coeff_, *umag_coeff_);

    // streamwise diffusion direction
    swdiff_coeff_ = new TransformedMatrixVectorCoefficient(un_next_coeff_, &streamwiseTensor);

    supg_coeff_ = new ScalarMatrixProductCoefficient(*upwind_coeff_, *swdiff_coeff_);
  }

  At_form_ = new ParBilinearForm(sfes_);
  auto *at_blfi = new ConvectionIntegrator(*rhou_coeff_);
  if (numerical_integ_) {
    at_blfi->SetIntRule(&ir_nli);
  }
  At_form_->AddDomainIntegrator(at_blfi);
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem At operator set" << endl;

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
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem MsRho operator set" << endl;

  // Helmholtz
  Ht_form_ = new ParBilinearForm(sfes_);
  auto *hmt_blfi = new MassIntegrator(*rho_over_dt_coeff_);
  auto *hdt_blfi = new DiffusionIntegrator(*thermal_diff_total_coeff_);
  if (numerical_integ_) {
    hmt_blfi->SetIntRule(&ir_di);
    hdt_blfi->SetIntRule(&ir_di);
  }
  // SUPG diffusion
  if (sw_stab_) {
    auto *sdt_blfi = new DiffusionIntegrator(*supg_coeff_);
    if (numerical_integ_) {
      sdt_blfi->SetIntRule(&ir_di);
    }
    Ht_form_->AddDomainIntegrator(sdt_blfi);
  }
  Ht_form_->AddDomainIntegrator(hmt_blfi);
  Ht_form_->AddDomainIntegrator(hdt_blfi);
  Ht_form_->Assemble();
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem Ht operator set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Ms_form_->AssembleDiagonal(diag_pa);
    MsInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsInvPC_ = new HypreSmoother(*Ms_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsInvPC_)->SetType(HypreSmoother::Jacobi, smoother_passes_);
    dynamic_cast<HypreSmoother *>(MsInvPC_)->SetSOROptions(smoother_relax_weight_, smoother_relax_omega_);
    dynamic_cast<HypreSmoother *>(MsInvPC_)->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_,
                                                            smoother_eig_est_);
  }
  MsInv_ = new CGSolver(sfes_->GetComm());
  MsInv_->iterative_mode = false;
  MsInv_->SetOperator(*Ms_);
  MsInv_->SetPreconditioner(*MsInvPC_);
  MsInv_->SetPrintLevel(mass_inverse_pl_);
  MsInv_->SetRelTol(mass_inverse_rtol_);
  MsInv_->SetAbsTol(mass_inverse_atol_);
  MsInv_->SetMaxIter(mass_inverse_max_iter_);

  HtInvPC_ = new HypreSmoother(*Ht_.As<HypreParMatrix>());
  dynamic_cast<HypreSmoother *>(HtInvPC_)->SetType(HypreSmoother::Jacobi, smoother_passes_);
  dynamic_cast<HypreSmoother *>(HtInvPC_)->SetSOROptions(hsmoother_relax_weight_, hsmoother_relax_omega_);
  dynamic_cast<HypreSmoother *>(HtInvPC_)->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_,
                                                          smoother_eig_est_);

  HtInv_ = new CGSolver(sfes_->GetComm());
  HtInv_->iterative_mode = true;
  HtInv_->SetOperator(*Ht_);
  HtInv_->SetPreconditioner(*HtInvPC_);
  HtInv_->SetPrintLevel(hsolve_pl_);
  HtInv_->SetRelTol(hsolve_rtol_);
  HtInv_->SetAbsTol(hsolve_atol_);
  HtInv_->SetMaxIter(hsolve_max_iter_);
  if (rank0_) std::cout << "Temperature operators set" << endl;

  // Qt .....................................
  Mq_form_ = new ParBilinearForm(sfes_);
  auto *mq_blfi = new MassIntegrator;
  if (numerical_integ_) {
    mq_blfi->SetIntRule(&ir_i);
  }
  Mq_form_->AddDomainIntegrator(mq_blfi);
  if (partial_assembly_) {
    Mq_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mq_form_->Assemble();
  Mq_form_->FormSystemMatrix(Qt_ess_tdof_, Mq_);
  // Mq_form_->FormSystemMatrix(empty, Mq);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem Mq operator set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Mq_form_->AssembleDiagonal(diag_pa);
    MqInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MqInvPC_ = new HypreSmoother(*Mq_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MqInvPC_)->SetType(HypreSmoother::Jacobi, smoother_passes_);
    dynamic_cast<HypreSmoother *>(MqInvPC_)->SetSOROptions(smoother_relax_weight_, smoother_relax_omega_);
    dynamic_cast<HypreSmoother *>(MqInvPC_)->SetPolyOptions(smoother_poly_order_, smoother_poly_fraction_,
                                                            smoother_eig_est_);
  }
  MqInv_ = new CGSolver(sfes_->GetComm());
  MqInv_->iterative_mode = false;
  MqInv_->SetOperator(*Mq_);
  MqInv_->SetPreconditioner(*MqInvPC_);
  MqInv_->SetPrintLevel(mass_inverse_pl_);
  MqInv_->SetRelTol(mass_inverse_rtol_);
  MqInv_->SetAbsTol(mass_inverse_atol_);
  MqInv_->SetMaxIter(mass_inverse_max_iter_);

  LQ_form_ = new ParBilinearForm(sfes_);
  auto *lqd_blfi = new DiffusionIntegrator(*thermal_diff_total_coeff_);
  if (numerical_integ_) {
    lqd_blfi->SetIntRule(&ir_di);
  }
  LQ_form_->AddDomainIntegrator(lqd_blfi);

  // SUPG diffusion
  if (sw_stab_) {
    auto *slqd_blfi = new DiffusionIntegrator(*supg_coeff_);
    if (numerical_integ_) {
      slqd_blfi->SetIntRule(&ir_di);
    }
    LQ_form_->AddDomainIntegrator(slqd_blfi);
  }

  if (partial_assembly_) {
    LQ_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  LQ_form_->Assemble();
  LQ_form_->FormSystemMatrix(empty, LQ_);

  LQ_bdry_ = new ParLinearForm(sfes_);
  auto *lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff_, 2, -1);
  if (numerical_integ_) {
    lq_bdry_lfi->SetIntRule(&ir_di);
  }
  LQ_bdry_->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr_);
  if (rank0_) std::cout << "CaloricallyPerfectThermoChem LQ operator set" << endl;

  // // remaining initialization
  // if (config_->resetTemp == true) {
  //   double *data = Tn_gf_.HostReadWrite();
  //   for (int i = 0; i < Sdof; i++) {
  //     data[i] = config_->initRhoRhoVp[4];
  //   }
  //   if (rank0_) std::cout << "Reset temperature to IC" << endl;
  // }

  // copy IC to other temp containers
  Tn_gf_.GetTrueDofs(Tn_);
  Tn_next_gf_.SetFromTrueDofs(Tn_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);

  // Temp filter
  if (filter_temperature_) {
    sfec_filter_ = new H1_FECollection(order_ - filter_cutoff_modes_);
    sfes_filter_ = new ParFiniteElementSpace(pmesh_, sfec_filter_);

    Tn_NM1_gf_.SetSpace(sfes_filter_);
    Tn_NM1_gf_ = 0.0;

    Tn_filtered_gf_.SetSpace(sfes_);
    Tn_filtered_gf_ = 0.0;
  }

  // and initialize system mass
  computeSystemMass();
}

/**
   Rotate temperature state in registers
 */
void CaloricallyPerfectThermoChem::UpdateTimestepHistory(double dt) {
  // temperature
  NTnm2_ = NTnm1_;
  NTnm1_ = NTn_;
  Tnm2_ = Tnm1_;
  Tnm1_ = Tn_;
  Tn_next_gf_.GetTrueDofs(Tn_next_);
  Tn_ = Tn_next_;
  Tn_gf_.SetFromTrueDofs(Tn_);
}

void CaloricallyPerfectThermoChem::step() {
  dt_ = time_coeff_.dt;
  time_ = time_coeff_.time;

  // Set current time for velocity Dirichlet boundary conditions.
  for (auto &temp_dbc : temp_dbcs_) {
    temp_dbc.coeff->SetTime(time_ + dt_);
  }

  // Prepare for residual calc
  extrapolateState();
  updateBC(0);  // NB: can't ramp right now
  updateThermoP();
  updateDensity(1.0);
  updateDiffusivity();

  // Build the right-hand-side
  resT_ = 0.0;

  // convection
  computeExplicitTempConvectionOP(true);  // ->tmpR0_
  resT_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, Tn_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, Tnm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, Tnm2_);

  MsRho_->AddMult(tmpR0_, resT_, -1.0);

  // dPo/dt
  tmpR0_ = (dtP_ / Cp_);
  Ms_->AddMult(tmpR0_, resT_);

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt_gf_ = rn_gf_;
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);

  Ht_form_->Update();
  Ht_form_->Assemble();
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht_);
  HtInv_->SetOperator(*Ht_);

  // Prepare for the solve
  for (auto &temp_dbc : temp_dbcs_) {
    Tn_next_gf_.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(resT_, resT_gf_);

  Vector Xt2, Bt2;
  Ht_form_->FormLinearSystem(temp_ess_tdof_, Tn_next_gf_, resT_gf_, Ht_, Xt2, Bt2, 1);

  // solve helmholtz eq for temp
  HtInv_->Mult(Bt2, Xt2);
  assert(HtInv_->GetConverged());

  Ht_form_->RecoverFEMSolution(Xt2, resT_gf_, Tn_next_gf_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);

  // explicit filter
  if (filter_temperature_) {
    const auto filter_alpha = filter_alpha_;
    Tn_NM1_gf_.ProjectGridFunction(Tn_next_gf_);
    Tn_filtered_gf_.ProjectGridFunction(Tn_NM1_gf_);
    const auto d_Tn_filtered_gf = Tn_filtered_gf_.Read();
    auto d_Tn_gf = Tn_next_gf_.ReadWrite();
    MFEM_FORALL(i, Tn_next_gf_.Size(),
                { d_Tn_gf[i] = (1.0 - filter_alpha) * d_Tn_gf[i] + filter_alpha * d_Tn_filtered_gf[i]; });
    Tn_next_gf_.GetTrueDofs(Tn_next_);
  }

  // prepare for external use
  updateDensity(1.0);
  // computeQt();
  computeQtTO();

  UpdateTimestepHistory(dt_);
}

void CaloricallyPerfectThermoChem::computeExplicitTempConvectionOP(bool extrap) {
  Array<int> empty;
  At_form_->Update();
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);
  if (extrap == true) {
    At_->Mult(Tn_, NTn_);
  } else {
    At_->Mult(Text_, NTn_);
  }

  // ab predictor
  if (extrap == true) {
    tmpR0_.Set(time_coeff_.ab1, NTn_);
    tmpR0_.Add(time_coeff_.ab2, NTnm1_);
    tmpR0_.Add(time_coeff_.ab3, NTnm2_);
  } else {
    tmpR0_.Set(1.0, NTn_);
  }
}

void CaloricallyPerfectThermoChem::initializeIO(IODataOrganizer &io) {
  io.registerIOFamily("Temperature", "/temperature", &Tn_gf_, true, true, sfec_);
  io.registerIOVar("/temperature", "temperature", 0);
}

void CaloricallyPerfectThermoChem::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", &Tn_gf_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
  pvdc.RegisterField("Qt", &Qt_gf_);
}

void CaloricallyPerfectThermoChem::initializeStats(Averaging &average, IODataOrganizer &io, bool continuation) {
  if (average.ComputeMean()) {
    // fields for averaging
    average.registerField(std::string("temperature"), &Tn_gf_, false, 0, 1);

    // io init
    io.registerIOFamily("Time-averaged temperature", "/meanTemp", average.GetMeanField(std::string("temperature")),
                        false, continuation, sfec_);
    io.registerIOVar("/meanTemp", "<T>", 0, true);
  }
}

/**
   Update boundary conditions for Temperature, useful for
   ramping a dirichlet condition over time
 */
void CaloricallyPerfectThermoChem::updateBC(int current_step) {
  // TODO(trevilo): By making the Dirichlet BC time dependent, we
  // already have a mechanism to handle this.  Is there a reason not
  // to do it that way?

  // if (numInlets > 0) {
  //   double *dTInf = buffer_tInletInf->HostReadWrite();
  //   double *dT = buffer_tInlet->HostReadWrite();
  //   //int nRamp = config_->rampStepsInlet;
  //   int nRamp = 1;
  //   double wt = std::pow(std::min((double)current_step / (double)nRamp, 1.0), 1.0);
  //   for (int i = 0; i < Sdof; i++) {
  //     dT[i] = wt * dTInf[i] + (1.0 - wt) * config_->initRhoRhoVp[nvel + 1];
  //   }
  // }
}

void CaloricallyPerfectThermoChem::extrapolateState() {
  // extrapolated temp at {n+1}
  Text_.Set(time_coeff_.ab1, Tn_);
  Text_.Add(time_coeff_.ab2, Tnm1_);
  Text_.Add(time_coeff_.ab3, Tnm2_);

  // store ext in _next containers, remove Text_, Uext completely later
  Tn_next_gf_.SetFromTrueDofs(Text_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);
}

// update thermodynamic pressure
void CaloricallyPerfectThermoChem::updateThermoP() {
  if (!domain_is_open_) {
    double allMass, PNM1;
    double myMass = 0.0;
    tmpR0_ = 1.0;
    tmpR0_ /= Tn_;
    tmpR0_ *= (thermo_pressure_ / Rgas_);
    Ms_->Mult(tmpR0_, tmpR0b_);
    myMass = tmpR0b_.Sum();
    MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PNM1 = thermo_pressure_;
    thermo_pressure_ = system_mass_ / allMass * PNM1;
    dtP_ = (thermo_pressure_ - PNM1) / dt_;
    // if( rank0_ == true ) std::cout << " Original closed system mass: " << system_mass_ << " [kg]" << endl;
    // if( rank0_ == true ) std::cout << " New (closed) system mass: " << allMass << " [kg]" << endl;
    // if( rank0_ == true ) std::cout << " ThermoP: " << thermo_pressure_ << " dtP_: " << dtP_ << endl;
  }
}

void CaloricallyPerfectThermoChem::updateDiffusivity() {
  // viscosity
  if (!constant_viscosity_) {
    double *d_visc = visc_.Write();
    const double *d_T = Tn_.Read();
    const double mu_star = mu0_;
    const double T_star = sutherland_T0_;
    const double S_star = sutherland_S0_;
    MFEM_FORALL(i, Tn_.Size(), { d_visc[i] = Sutherland(d_T[i], mu_star, T_star, S_star); });
  } else {
    visc_ = mu0_;
  }

  // if (config_->sgsModelType > 0) {
  //   const double *dataSubgrid = turbModel_interface_->eddy_viscosity->HostRead();
  //   double *dataVisc = visc_gf.HostReadWrite();
  //   double *data = viscTotal_gf.HostReadWrite();
  //   double *Rdata = rn_gf.HostReadWrite();
  //   for (int i = 0; i < Sdof; i++) {
  //     data[i] = dataVisc[i] + Rdata[i] * dataSubgrid[i];
  //   }
  // }

  // if (config_->linViscData.isEnabled) {
  //   double *vMult = viscMult.HostReadWrite();
  //   double *dataVisc = viscTotal_gf.HostReadWrite();
  //   for (int i = 0; i < Sdof; i++) {
  //     dataVisc[i] *= viscMult[i];
  //   }
  // }

  // thermal diffusivity: storing as kappa eventhough does NOT
  // include Cp for now
  // {
  //   double *data = kappa_gf.HostReadWrite();
  //   double *dataVisc = viscTotal_gf.HostReadWrite();
  //   // double Ctmp = Cp / Pr;
  //   double Ctmp = 1.0 / Pr;
  //   for (int i = 0; i < Sdof; i++) {
  //     data[i] = Ctmp * dataVisc[i];
  //   }
  // }

  visc_gf_.SetFromTrueDofs(visc_);

  // NB: Here "kappa" is kappa/Cp = mu/Pr
  kappa_ = visc_;
  kappa_ /= Pr_;
  kappa_gf_.SetFromTrueDofs(kappa_);
}

void CaloricallyPerfectThermoChem::updateDensity(double tStep) {
  Array<int> empty;

  // set rn
  if (constant_density_ != true) {
    if (tStep == 1.0) {
      rn_ = (thermo_pressure_ / Rgas_);
      rn_ /= Tn_next_;
    } else if (tStep == 0.5) {
      // multConstScalarInv((thermo_pressure_ / Rgas_), Tn_next_, &tmpR0a);
      // multConstScalarInv((thermo_pressure_ / Rgas_), Tn_, &tmpR0b);
      // rn_.Set(0.5, tmpR0a);
      // rn_.Add(0.5, tmpR0b);
      rn_ = (thermo_pressure_ / Rgas_);
      rn_ /= Tn_next_;
      rn_ *= 0.5;

      tmpR0_ = (thermo_pressure_ / Rgas_);
      tmpR0_ /= Tn_;
      rn_.Add(0.5, tmpR0_);
    } else {
      multConstScalarInv((thermo_pressure_ / Rgas_), Tn_, &rn_);
    }
  } else {
    rn_ = static_rho_;
  }
  rn_gf_.SetFromTrueDofs(rn_);

  MsRho_form_->Update();
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);

  // project to p-space in case not same as vel-temp
  R0PM0_gf_.SetFromTrueDofs(rn_);
  // R0PM1_gf.ProjectGridFunction(R0PM0_gf);
}

void CaloricallyPerfectThermoChem::computeSystemMass() {
  system_mass_ = 0.0;
  double myMass = 0.0;
  rn_ = (thermo_pressure_ / Rgas_);
  rn_ /= Tn_;
  Ms_->Mult(rn_, tmpR0_);
  myMass = tmpR0_.Sum();
  MPI_Allreduce(&myMass, &system_mass_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank0_ == true) std::cout << " Closed system mass: " << system_mass_ << " [kg]" << endl;
}

/// Add a Dirichlet boundary condition to the temperature field
void CaloricallyPerfectThermoChem::AddTempDirichletBC(const double &temp, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, new ConstantCoefficient(temp));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }
}

void CaloricallyPerfectThermoChem::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      // std::cout << "patch: " << i << " temp_ess_attr: " << temp_ess_attr_[i] << endl;
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }

  /*
  if (rank0_) {
    mfem::out << "Adding Temperature Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((temp_ess_attr_[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      temp_ess_attr_[i] = 1;
    }
  }
  */
}

void CaloricallyPerfectThermoChem::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void CaloricallyPerfectThermoChem::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr) {
  Qt_dbcs_.emplace_back(attr, coeff);

  if (rank0_ && pmesh_->GetMyRank() == 0) {
    mfem::out << "Adding Qt Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((Qt_ess_attr_[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      Qt_ess_attr_[i] = 1;
    }
  }
}

void CaloricallyPerfectThermoChem::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddQtDirichletBC(new FunctionCoefficient(f), attr);
}

void CaloricallyPerfectThermoChem::computeQtTO() {
  tmpR0_ = 0.0;
  LQ_bdry_->Update();
  LQ_bdry_->Assemble();
  LQ_bdry_->ParallelAssemble(tmpR0_);
  tmpR0_.Neg();

  Array<int> empty;
  LQ_form_->Update();
  LQ_form_->Assemble();
  LQ_form_->FormSystemMatrix(empty, LQ_);
  LQ_->AddMult(Tn_next_, tmpR0_);  // tmpR0_ += LQ{Tn_next}

  sfes_->GetRestrictionMatrix()->MultTranspose(tmpR0_, resT_gf_);

  Qt_ = 0.0;
  Qt_gf_.SetFromTrueDofs(tmpR0_);

  Vector Xqt, Bqt;
  Mq_form_->FormLinearSystem(Qt_ess_tdof_, Qt_gf_, resT_gf_, Mq_, Xqt, Bqt, 1);

  MqInv_->Mult(Bqt, Xqt);
  Mq_form_->RecoverFEMSolution(Xqt, resT_gf_, Qt_gf_);

  Qt_gf_.GetTrueDofs(Qt_);
  Qt_ *= -Rgas_ / thermo_pressure_;
  Qt_gf_.SetFromTrueDofs(Qt_);
}

void CaloricallyPerfectThermoChem::screenHeader(std::vector<std::string> &header) const {
  if (!domain_is_open_) {
    header.resize(1);
    header[0] = "P/P0";
  }
}

void CaloricallyPerfectThermoChem::screenValues(std::vector<double> &values) {
  if (!domain_is_open_) {
    values.resize(1);
    values[0] = thermo_pressure_ / ambient_pressure_;
  }
}
