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
#include <mfem/general/forall.hpp>

#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

MFEM_HOST_DEVICE double Sutherland(const double T, const double mu_star, const double T_star, const double S_star) {
  const double T_rat = T / T_star;
  const double T_rat_32 = T_rat * sqrt(T_rat);
  const double S_rat = (T_star + S_star) / (T + S_star);
  return mu_star * T_rat_32 * S_rat;
}

ThermoChem::ThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff,
                       TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

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

  Cp_ = gamma_ * Rgas_ / (gamma_ - 1);

  filterTemp = loMach_opts->filterTemp;
  filter_alpha = loMach_opts->filterWeight;
  filter_cutoff_modes = loMach_opts->nFilter;

  tpsP_->getInput("loMach/openSystem", domain_is_open_, false);

  tpsP_->getInput("loMach/calperfect/linear-solver-rtol", rtol_, 1e-12);
  tpsP_->getInput("loMach/calperfect/linear-solver-max-iter", max_iter_, 1000);
  tpsP_->getInput("loMach/calperfect/linear-solver-verbosity", pl_solve_, 0);
}

ThermoChem::~ThermoChem() {
  // allocated in initializeOperators
  delete LQ_bdry_;
  delete LQ_form_;
  delete MqInv;
  delete MqInvPC;
  delete Mq_form_;
  delete HtInv;
  delete HtInvPC;
  delete MsInv;
  delete MsInvPC;
  delete Ht_form_;
  delete MsRho_form_;
  delete Ms_form_;
  delete At_form_;
  delete rhou_coeff_;
  delete rhon_next_coeff_;
  delete un_next_coeff_;
  delete kap_gradT_coeff_;
  delete gradT_coeff_;
  delete thermal_diff_coeff_;
  delete rho_over_dt_coeff_;
  delete rho_coeff_;

  // allocated in initializeSelf
  delete sfes_;
  delete sfec_;
}

void ThermoChem::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing ThermoChem solver.\n");

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
  if (rank0_) grvy_printf(ginfo, "ThermoChem paces constructed...\n");

  int sfes_truevsize = sfes_->GetTrueVSize();

  Qt.SetSize(sfes_truevsize);
  Qt = 0.0;
  Qt_gf_.SetSpace(sfes_);

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
  Text_gf_.SetSpace(sfes_);

  resT_gf_.SetSpace(sfes_);
  resT.SetSize(sfes_truevsize);
  resT = 0.0;

  Tn_next_gf_.SetSpace(sfes_);
  Tn_gf_.SetSpace(sfes_);
  Tnm1_gf_.SetSpace(sfes_);
  Tnm2_gf_.SetSpace(sfes_);

  rn.SetSize(sfes_truevsize);
  rn = 1.0;
  rn_gf_.SetSpace(sfes_);
  rn_gf_ = 1.0;

  visc.SetSize(sfes_truevsize);
  visc = 1.0e-12;

  kappa.SetSize(sfes_truevsize);
  kappa = 1.0e-12;

  kappa_gf_.SetSpace(sfes_);
  kappa_gf_ = 0.0;

  viscTotal_gf_.SetSpace(sfes_);
  viscTotal_gf_ = 0.0;

  visc_gf_.SetSpace(sfes_);
  visc_gf_ = 0.0;

  tmpR0.SetSize(sfes_truevsize);
  tmpR0a.SetSize(sfes_truevsize);
  tmpR0b.SetSize(sfes_truevsize);
  tmpR0c.SetSize(sfes_truevsize);

  R0PM0_gf_.SetSpace(sfes_);

  rhoDt.SetSpace(sfes_);

  if (rank0_) grvy_printf(ginfo, "ThermoChem vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &visc_gf_;
  toFlow_interface_.thermal_divergence = &Qt_gf_;
  toTurbModel_interface_.density = &rn_gf_;

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

  ConstantCoefficient t_ic_coef;
  t_ic_coef.constant = T_ic_;
  Tn_gf_.ProjectCoefficient(t_ic_coef);

  Tn_gf_.GetTrueDofs(Tn);
  Tnm1_gf_.SetFromTrueDofs(Tn);
  Tnm2_gf_.SetFromTrueDofs(Tn);
  Tnm1_gf_.GetTrueDofs(Tnm1);
  Tnm2_gf_.GetTrueDofs(Tnm2);

  //-----------------------------------------------------
  // 3) Set the boundary conditions
  //-----------------------------------------------------
  int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  ConstantCoefficient Qt_bc_coef;
  Array<int> Qattr(pmesh_->bdr_attributes.Max());

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

      // TODO(trevilo): inlet types uniform and interpolate should be supported
      if (type == "uniform") {
        if (rank0_) {
          std::cout << "ERROR: Inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      } else if (type == "interpolate") {
        if (rank0_) {
          std::cout << "ERROR: Inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      } else {
        if (rank0_) {
          std::cout << "ERROR: Inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      }
    }
  }

  // outlet bc
  {
    Array<int> attr_outlet(pmesh_->bdr_attributes.Max());
    attr_outlet = 0;

    // No code for this yet, so die if detected
    assert(numOutlets == 0);

    // But... outlets will just get homogeneous Neumann on T, so
    // basically need to do nothing.
  }

  // Wall BCs
  {
    std::cout << "There are " << pmesh_->bdr_attributes.Max() << " boundary attributes!" << std::endl;
    Array<int> attr_wall(pmesh_->bdr_attributes.Max());
    attr_wall = 0;

    for (int i = 1; i <= numWalls; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/wall" + std::to_string(i));

      tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
      tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

      if (type == "viscous_isothermal") {
        std::cout << "Adding patch = " << patch << " to isothermal wall list!" << std::endl;

        attr_wall = 0;
        attr_wall[patch - 1] = 1;

        double Twall;
        tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Twall);

        ConstantCoefficient *Twall_coeff = new ConstantCoefficient();
        Twall_coeff->constant = Twall;

        AddTempDirichletBC(Twall_coeff, attr_wall);
      }
    }
    if (rank0_) std::cout << "Temp wall bc completed: " << numWalls << endl;
  }

  // TODO(trevilo): Add BCs for Q!!!
  //
  // // Qt wall bc dirichlet (divU = 0)
  // Qattr = 0;
  // for (int i = 0; i < numWalls; i++) {
  //   Qt_bc_coef.constant = 0.0;
  //   for (int iFace = 1; iFace < pmesh_->bdr_attributes.Max() + 1; iFace++) {
  //     if (config_->wallPatchType[i].second == WallType::VISC_ISOTH ||
  //         config_->wallPatchType[i].second == WallType::VISC_ADIAB) {
  //       if (iFace == config_->wallPatchType[i].first) {
  //         Qattr[iFace - 1] = 1;
  //         if (rank0_) std::cout << "Dirichlet wall Qt BC: " << iFace << endl;
  //       }
  //     }
  //   }
  // }
  // buffer_qbc = new ConstantCoefficient(Qt_bc_coef);
  // AddQtDirichletBC(buffer_qbc, Qattr);

  sfes_->GetEssentialTrueDofs(temp_ess_attr_, temp_ess_tdof_);
  sfes_->GetEssentialTrueDofs(Qt_ess_attr_, Qt_ess_tdof_);
  if (rank0_) std::cout << "ThermoChem Essential true dof step" << endl;
}

void ThermoChem::initializeOperators() {
  dt_ = time_coeff_.dt;

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

  rhoDt = rn_gf_;
  rhoDt /= dt_;
  rho_over_dt_coeff_ = new GridFunctionCoefficient(&rhoDt);

  // thermal_diff_coeff.constant = thermal_diff;
  thermal_diff_coeff_ = new GridFunctionCoefficient(&kappa_gf_);
  gradT_coeff_ = new GradientGridFunctionCoefficient(&Tn_next_gf_);
  kap_gradT_coeff_ = new ScalarVectorProductCoefficient(*thermal_diff_coeff_, *gradT_coeff_);

  // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho u \cdot \nabla \phi_j
  un_next_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhon_next_coeff_ = new GridFunctionCoefficient(&rn_gf_);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rhon_next_coeff_, *un_next_coeff_);

  At_form_ = new ParBilinearForm(sfes_);
  auto *at_blfi = new ConvectionIntegrator(*rhou_coeff_);
  if (numerical_integ_) {
    at_blfi->SetIntRule(&ir_nli);
  }
  At_form_->AddDomainIntegrator(at_blfi);
  if (partial_assembly_) {
    At_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At);
  if (rank0_) std::cout << "ThermoChem At operator set" << endl;

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
  Ms_form_->FormSystemMatrix(empty, Ms);

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
  MsRho_form_->FormSystemMatrix(empty, MsRho);
  if (rank0_) std::cout << "ThermoChem MsRho operator set" << endl;

  Ht_form_ = new ParBilinearForm(sfes_);
  auto *hmt_blfi = new MassIntegrator(*rho_over_dt_coeff_);
  auto *hdt_blfi = new DiffusionIntegrator(*thermal_diff_coeff_);

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
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht);
  if (rank0_) std::cout << "ThermoChem Ht operator set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Ms_form_->AssembleDiagonal(diag_pa);
    MsInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsInvPC = new HypreSmoother(*Ms.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MsInv = new CGSolver(sfes_->GetComm());
  MsInv->iterative_mode = false;
  MsInv->SetOperator(*Ms);
  MsInv->SetPreconditioner(*MsInvPC);
  MsInv->SetPrintLevel(pl_solve_);
  MsInv->SetRelTol(rtol_);
  MsInv->SetMaxIter(max_iter_);

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Ht_form_->AssembleDiagonal(diag_pa);
    HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof_);
  } else {
    HtInvPC = new HypreSmoother(*Ht.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HtInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }

  HtInv = new CGSolver(sfes_->GetComm());

  HtInv->iterative_mode = true;
  HtInv->SetOperator(*Ht);
  HtInv->SetPreconditioner(*HtInvPC);
  HtInv->SetPrintLevel(pl_solve_);
  HtInv->SetRelTol(rtol_);
  HtInv->SetMaxIter(max_iter_);
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
  Mq_form_->FormSystemMatrix(Qt_ess_tdof_, Mq);
  // Mq_form_->FormSystemMatrix(empty, Mq);
  if (rank0_) std::cout << "ThermoChem Mq operator set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Mq_form_->AssembleDiagonal(diag_pa);
    MqInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MqInvPC = new HypreSmoother(*Mq.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MqInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MqInv = new CGSolver(sfes_->GetComm());
  MqInv->iterative_mode = false;
  MqInv->SetOperator(*Mq);
  MqInv->SetPreconditioner(*MqInvPC);
  MqInv->SetPrintLevel(pl_solve_);
  MqInv->SetRelTol(rtol_);
  MqInv->SetMaxIter(max_iter_);

  LQ_form_ = new ParBilinearForm(sfes_);
  auto *lqd_blfi = new DiffusionIntegrator(*thermal_diff_coeff_);
  if (numerical_integ_) {
    lqd_blfi->SetIntRule(&ir_di);
  }
  LQ_form_->AddDomainIntegrator(lqd_blfi);
  if (partial_assembly_) {
    LQ_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  LQ_form_->Assemble();
  LQ_form_->FormSystemMatrix(empty, LQ);

  LQ_bdry_ = new ParLinearForm(sfes_);
  auto *lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff_, 2, -1);
  if (numerical_integ_) {
    lq_bdry_lfi->SetIntRule(&ir_di);
  }
  LQ_bdry_->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr_);
  if (rank0_) std::cout << "ThermoChem LQ operator set" << endl;

  // // remaining initialization
  // if (config_->resetTemp == true) {
  //   double *data = Tn_gf_.HostReadWrite();
  //   for (int i = 0; i < Sdof; i++) {
  //     data[i] = config_->initRhoRhoVp[4];
  //   }
  //   if (rank0_) std::cout << "Reset temperature to IC" << endl;
  // }

  // copy IC to other temp containers
  Tn_gf_.GetTrueDofs(Tn);
  Tn_next_gf_.SetFromTrueDofs(Tn);
  Tn_next_gf_.GetTrueDofs(Tn_next);

  // Temp filter
  if (filterTemp) {
    sfec_filter = new H1_FECollection(order_ - filter_cutoff_modes);
    sfes_filter = new ParFiniteElementSpace(pmesh_, sfec_filter);

    Tn_NM1_gf_.SetSpace(sfes_filter);
    Tn_NM1_gf_ = 0.0;

    Tn_filtered_gf_.SetSpace(sfes_);
    Tn_filtered_gf_ = 0.0;
  }

  // // viscous sponge
  // ParGridFunction coordsDof(vfes);
  // pmesh_->GetNodes(coordsDof);
  // if (config_->linViscData.isEnabled) {
  //   if (rank0_) std::cout << "Viscous sponge active" << endl;
  //   double *vMult = viscMult.HostReadWrite();
  //   double *hcoords = coordsDof.HostReadWrite();
  //   double wgt = 0.;
  //   for (int n = 0; n < sfes_->GetNDofs(); n++) {
  //     double coords[3];
  //     for (int d = 0; d < dim; d++) {
  //       coords[d] = hcoords[n + d * sfes_->GetNDofs()];
  //     }
  //     viscSpongePlanar(coords, wgt);
  //     vMult[n] = wgt + (config_->linViscData.uniformMult - 1.0);
  //   }
  // }

  // and initialize system mass
  computeSystemMass();
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
  Tn_next_gf_.GetTrueDofs(Tn_next);
  Tn = Tn_next;
  Tn_gf_.SetFromTrueDofs(Tn);
}

void ThermoChem::step() {
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
  resT = 0.0;

  // convection
  computeExplicitTempConvectionOP(true);  // ->tmpR0
  resT.Set(-1.0, tmpR0);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0.Set(time_coeff_.bd1 / dt_, Tn);
  tmpR0.Add(time_coeff_.bd2 / dt_, Tnm1);
  tmpR0.Add(time_coeff_.bd3 / dt_, Tnm2);

  MsRho->Mult(tmpR0, tmpR0b);
  resT.Add(-1.0, tmpR0b);

  // dPo/dt
  tmpR0 = (dtP_ / Cp_);
  Ms->Mult(tmpR0, tmpR0b);
  resT.Add(1.0, tmpR0b);

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt = rn_gf_;
  rhoDt *= (time_coeff_.bd0 / dt_);

  Ht_form_->Update();
  Ht_form_->Assemble();
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht);

  HtInv->SetOperator(*Ht);
  if (partial_assembly_) {
    delete HtInvPC;
    Vector diag_pa(sfes_->GetTrueVSize());
    Ht_form_->AssembleDiagonal(diag_pa);
    HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof_);
    HtInv->SetPreconditioner(*HtInvPC);
  }

  // Prepare for the solve
  for (auto &temp_dbc : temp_dbcs_) {
    Tn_next_gf_.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(resT, resT_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Ht.As<ConstrainedOperator>();
    EliminateRHS(*Ht_form_, *HC, temp_ess_tdof_, Tn_next_gf_, resT_gf_, Xt2, Bt2, 1);
  } else {
    Ht_form_->FormLinearSystem(temp_ess_tdof_, Tn_next_gf_, resT_gf_, Ht, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for temp
  HtInv->Mult(Bt2, Xt2);
  assert(HtInv->GetConverged());

  Ht_form_->RecoverFEMSolution(Xt2, resT_gf_, Tn_next_gf_);
  Tn_next_gf_.GetTrueDofs(Tn_next);

  // explicit filter
  if (filterTemp) {
    const auto filter_alpha_ = filter_alpha;
    Tn_NM1_gf_.ProjectGridFunction(Tn_next_gf_);
    Tn_filtered_gf_.ProjectGridFunction(Tn_NM1_gf_);
    const auto d_Tn_filtered_gf = Tn_filtered_gf_.Read();
    auto d_Tn_gf = Tn_next_gf_.ReadWrite();
    MFEM_FORALL(i, Tn_next_gf_.Size(),
                { d_Tn_gf[i] = (1.0 - filter_alpha_) * d_Tn_gf[i] + filter_alpha_ * d_Tn_filtered_gf[i]; });
    Tn_next_gf_.GetTrueDofs(Tn_next);
  }

  // prepare for external use
  updateDensity(1.0);
  // computeQt();
  computeQtTO();

  UpdateTimestepHistory(dt_);
}

void ThermoChem::computeExplicitTempConvectionOP(bool extrap) {
  Array<int> empty;
  At_form_->Update();
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At);
  if (extrap == true) {
    At->Mult(Tn, NTn);
  } else {
    At->Mult(Text, NTn);
  }

  // ab predictor
  if (extrap == true) {
    tmpR0.Set(time_coeff_.ab1, NTn);
    tmpR0.Add(time_coeff_.ab2, NTnm1);
    tmpR0.Add(time_coeff_.ab3, NTnm2);
  } else {
    tmpR0.Set(1.0, NTn);
  }
}

void ThermoChem::initializeIO(IODataOrganizer &io) {
  io.registerIOFamily("Temperature", "/temperature", &Tn_gf_, false);
  io.registerIOVar("/temperature", "temperature", 0);
}

void ThermoChem::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", &Tn_gf_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
}

/**
   Update boundary conditions for Temperature, useful for
   ramping a dirichlet condition over time
 */
void ThermoChem::updateBC(int current_step) {
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

void ThermoChem::extrapolateState() {
  // extrapolated temp at {n+1}
  Text.Set(time_coeff_.ab1, Tn);
  Text.Add(time_coeff_.ab2, Tnm1);
  Text.Add(time_coeff_.ab3, Tnm2);

  // store ext in _next containers, remove Text, Uext completely later
  Tn_next_gf_.SetFromTrueDofs(Text);
  Tn_next_gf_.GetTrueDofs(Tn_next);
}

// update thermodynamic pressure
void ThermoChem::updateThermoP() {
  if (!domain_is_open_) {
    double allMass, PNM1;
    double myMass = 0.0;
    tmpR0 = 1.0;
    tmpR0 /= Tn;
    tmpR0 *= (thermo_pressure_ / Rgas_);
    Ms->Mult(tmpR0, tmpR0b);
    myMass = tmpR0b.Sum();
    MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PNM1 = thermo_pressure_;
    thermo_pressure_ = system_mass_ / allMass * PNM1;
    dtP_ = (thermo_pressure_ - PNM1) / dt_;
    // if( rank0_ == true ) std::cout << " Original closed system mass: " << system_mass_ << " [kg]" << endl;
    // if( rank0_ == true ) std::cout << " New (closed) system mass: " << allMass << " [kg]" << endl;
    // if( rank0_ == true ) std::cout << " ThermoP: " << thermo_pressure_ << " dtP_: " << dtP_ << endl;
  }
}

void ThermoChem::updateDiffusivity() {
  // viscosity
  if (!constant_viscosity_) {
    double *d_visc = visc.Write();
    const double *d_T = Tn.Read();
    const double mu_star = mu0_;
    const double T_star = sutherland_T0_;
    const double S_star = sutherland_S0_;
    MFEM_FORALL(i, Tn.Size(), { d_visc[i] = Sutherland(d_T[i], mu_star, T_star, S_star); });
  } else {
    visc = mu0_;
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

  // viscTotal_gf.SetFromTrueDofs(visc);

  // viscTotal_gf.GetTrueDofs(visc);
  viscTotal_gf_.SetFromTrueDofs(visc);

  // NB: Here kappa is kappa / Cp = mu / Pr
  kappa = visc;
  kappa /= Pr_;
  kappa_gf_.SetFromTrueDofs(kappa);
}

void ThermoChem::updateDensity(double tStep) {
  Array<int> empty;

  // set rn
  if (constant_density_ != true) {
    if (tStep == 1.0) {
      rn = (thermo_pressure_ / Rgas_);
      rn /= Tn_next;
    } else if (tStep == 0.5) {
      multConstScalarInv((thermo_pressure_ / Rgas_), Tn_next, &tmpR0a);
      multConstScalarInv((thermo_pressure_ / Rgas_), Tn, &tmpR0b);
      rn.Set(0.5, tmpR0a);
      rn.Add(0.5, tmpR0b);
    } else {
      multConstScalarInv((thermo_pressure_ / Rgas_), Tn, &rn);
    }
  } else {
    rn = static_rho_;
  }
  rn_gf_.SetFromTrueDofs(rn);

  MsRho_form_->Update();
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho);

  // project to p-space in case not same as vel-temp
  R0PM0_gf_.SetFromTrueDofs(rn);
  // R0PM1_gf.ProjectGridFunction(R0PM0_gf);
}

void ThermoChem::computeSystemMass() {
  system_mass_ = 0.0;
  double myMass = 0.0;
  rn = (thermo_pressure_ / Rgas_);
  rn /= Tn;
  Ms->Mult(rn, tmpR0);
  myMass = tmpR0.Sum();
  MPI_Allreduce(&myMass, &system_mass_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank0_ == true) std::cout << " Closed system mass: " << system_mass_ << " [kg]" << endl;
}

void ThermoChem::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, coeff);

  if (rank0_ && pmesh_->GetMyRank() == 0) {
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
}

void ThermoChem::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void ThermoChem::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr) {
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

void ThermoChem::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddQtDirichletBC(new FunctionCoefficient(f), attr);
}

void ThermoChem::computeQtTO() {
  Array<int> empty;
  tmpR0 = 0.0;
  LQ_bdry_->Update();
  LQ_bdry_->Assemble();
  LQ_bdry_->ParallelAssemble(tmpR0);
  tmpR0.Neg();

  LQ_form_->Update();
  LQ_form_->Assemble();
  LQ_form_->FormSystemMatrix(empty, LQ);
  LQ->AddMult(Tn_next, tmpR0);  // tmpR0 += LQ{Tn_next}
  MqInv->Mult(tmpR0, Qt);

  Qt *= -Rgas_ / thermo_pressure_;

  /*
    for (int be = 0; be < pmesh->GetNBE(); be++) {
    int bAttr = pmesh.GetBdrElement(be)->GetAttribute();
    if (bAttr == WallType::VISC_ISOTH || bAttr = WallType::VISC_ADIAB) {
    Array<int> vdofs;
    sfes_->GetBdrElementVDofs(be, vdofs);
    for (int i = 0; i < vdofs.Size(); i++) {
    Qt[vdofs[i]] = 0.0;
    }
    }
    }
  */

  Qt_gf_.SetFromTrueDofs(Qt);
}

#if 0
// Temporarily remove viscous sponge capability.  This sould be housed
// in a separate class that we just instantiate and call, rather than
// directly in ThermoChem.

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

/**
   Interpolation of inlet bc from external file using
   Gaussian interpolation.  Not at all general and only for use
   with torch
 */
void ThermoChem::interpolateInlet() {
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  int Sdof = sfes_->GetNDofs();

  string fname;
  fname = "./inputs/inletPlane.csv";
  const int fname_length = fname.length();
  char *char_array = new char[fname_length + 1];
  strcpy(char_array, fname.c_str());
  // snprintf(char_array, fname.c_str());
  int nCount = 0;

  // open, find size
  if (rank0_) {
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
  if (rank0_) {
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
          inlet[nLines].temp = thermo_pressure_ / (Rgas_ * std::max(buffer, 1.0));
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
  if (rank0_) {
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
  for (int n = 0; n < sfes_->GetNDofs(); n++) {
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

    // if(rank0_) { std::cout << " attempting to record interpolated values in buffer" << endl; fflush(stdout); }
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
  int Sdof = sfes_->GetNDofs();

  ParGridFunction coordsDof(vfes);
  pmesh_->GetNodes(coordsDof);
  double *dataTemp = buffer_tInlet->HostWrite();
  double *dataTempInf = buffer_tInletInf->HostWrite();

  Vector inlet_vec(3);
  double inlet_temp, inlet_pres;
  inlet_temp = config_->amb_pres / (Rgas_ * config_->densInlet);
  inlet_pres = config_->amb_pres;

  for (int n = 0; n < sfes_->GetNDofs(); n++) {
    auto hcoords = coordsDof.HostRead();
    double xp[3];
    for (int d = 0; d < dim; d++) {
      xp[d] = hcoords[n + d * vfes->GetNDofs()];
    }
    dataTemp[n] = inlet_temp;
    dataTempInf[n] = inlet_temp;
  }
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
#endif
