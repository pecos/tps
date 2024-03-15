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

#include "lte_thermo_chem.hpp"

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

LteThermoChem::LteThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff,
                             TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  // Initialize thermo TableInput (data read below)
  std::vector<TableInput> thermo_tables(5);
  for (size_t i = 0; i < thermo_tables.size(); i++) {
    thermo_tables[i].order = 1;
    thermo_tables[i].xLogScale = false;
    thermo_tables[i].fLogScale = false;
  }

  std::string table_file;
  tps->getRequiredInput("loMach/ltethermo/table-file", table_file);

  // Read data from hdf5 file containing 6 columns: T, mu, kappa, sigma, Rgas, Cp
  DenseMatrix table_data;
  bool success;
  success = h5ReadBcastMultiColumnTable(table_file, std::string("T_mu_kap_sig_R_Cp"), pmesh_->GetComm(), table_data,
                                        thermo_tables);
  if (!success) exit(ERROR);

  mu_table_ = new LinearTable(thermo_tables[0]);
  kappa_table_ = new LinearTable(thermo_tables[1]);
  sigma_table_ = new LinearTable(thermo_tables[2]);
  Rgas_table_ = new LinearTable(thermo_tables[3]);
  Cp_table_ = new LinearTable(thermo_tables[4]);

  tpsP_->getInput("loMach/ambientPressure", ambient_pressure_, 101325.0);
  thermo_pressure_ = ambient_pressure_;
  dtP_ = 0.0;

  tpsP_->getInput("initialConditions/temperature", T_ic_, 300.0);

  filter_temperature_ = loMach_opts->filterTemp;
  filter_alpha_ = loMach_opts->filterWeight;
  filter_cutoff_modes_ = loMach_opts->nFilter;

  tps->getInput("loMach/openSystem", domain_is_open_, false);

  tps->getInput("loMach/ltethermo/turb-Prandtl", Prt_, 0.9);
  invPrt_ = 1.0 / Prt_;

  tps->getInput("loMach/ltethermo/linear-solver-rtol", rtol_, 1e-12);
  tps->getInput("loMach/ltethermo/linear-solver-max-iter", max_iter_, 1000);
  tps->getInput("loMach/ltethermo/linear-solver-verbosity", pl_solve_, 0);
}

LteThermoChem::~LteThermoChem() {
  // allocated in initializeOperators
  delete sfes_filter_;
  delete sfec_filter_;
  delete HtInv_;
  delete HtInvPC_;
  delete MsInv_;
  delete MsInvPC_;
  delete MrhoInv_;
  delete MrhoInvPC_;
  delete Ht_form_;
  delete M_rho_form_;
  delete M_rho_Cp_form_;
  delete Ms_form_;
  delete At_form_;
  delete rho_Cp_u_coeff_;
  delete un_next_coeff_;
  delete kap_gradT_coeff_;
  delete gradT_coeff_;
  delete mut_coeff_;
  delete mult_coeff_;
  delete thermal_diff_coeff_;
  delete thermal_diff_sum_coeff_;
  delete thermal_diff_total_coeff_;
  delete rho_Cp_over_dt_coeff_;
  delete rho_Cp_coeff_;
  delete rho_coeff_;

  // allocated in initializeSelf
  delete sfes_;
  delete sfec_;
}

void LteThermoChem::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing LteThermoChem solver.\n");

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
  if (rank0_) grvy_printf(ginfo, "LteThermoChem paces constructed...\n");

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

  rnm1_.SetSize(sfes_truevsize);
  rnm2_.SetSize(sfes_truevsize);
  rnm3_.SetSize(sfes_truevsize);

  rn_gf_.SetSpace(sfes_);
  rn_gf_ = 1.0;

  visc_.SetSize(sfes_truevsize);
  visc_ = 1.0e-12;

  kappa_.SetSize(sfes_truevsize);
  kappa_ = 1.0e-12;

  sigma_.SetSize(sfes_truevsize);
  sigma_ = 0.0;

  Rgas_.SetSize(sfes_truevsize);
  Rgas_ = 0.0;

  Cp_.SetSize(sfes_truevsize);
  Cp_ = 0.0;

  mu_gf_.SetSpace(sfes_);
  mu_gf_ = 0.0;

  kappa_gf_.SetSpace(sfes_);
  kappa_gf_ = 0.0;

  sigma_gf_.SetSpace(sfes_);
  sigma_gf_ = 0.0;

  jh_gf_.SetSpace(sfes_);
  jh_gf_ = 0.0;

  Rgas_gf_.SetSpace(sfes_);
  Rgas_gf_ = 0.0;

  Cp_gf_.SetSpace(sfes_);
  Cp_gf_ = 0.0;

  tmpR0_.SetSize(sfes_truevsize);
  tmpR0b_.SetSize(sfes_truevsize);

  R0PM0_gf_.SetSpace(sfes_);

  if (rank0_) grvy_printf(ginfo, "LteThermoChem vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &mu_gf_;
  toFlow_interface_.thermal_divergence = &Qt_gf_;
  toTurbModel_interface_.density = &rn_gf_;

  plasma_conductivity_gf_ = &sigma_gf_;
  joule_heating_gf_ = &jh_gf_;

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

  Tn_gf_.GetTrueDofs(Tn_);
  Tn_next_ = Tn_;

  Tnm1_gf_.SetFromTrueDofs(Tn_);
  Tnm2_gf_.SetFromTrueDofs(Tn_);
  Tnm1_gf_.GetTrueDofs(Tnm1_);
  Tnm2_gf_.GetTrueDofs(Tnm2_);

  // Initialize the gas constant and Cp
  {
    const double *d_T = Tn_next_.Read();

    double *d_Rgas = Rgas_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_Rgas[i] = Rgas_table_->eval(d_T[i]); });

    double *d_Cp = Cp_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_Cp[i] = Cp_table_->eval(d_T[i]); });
  }
  Rgas_gf_.SetFromTrueDofs(Rgas_);
  Cp_gf_.SetFromTrueDofs(Cp_);

  // Initialize the transport properties
  {
    const double *d_T = Tn_.Read();

    double *d_visc = visc_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_visc[i] = mu_table_->eval(d_T[i]); });

    double *d_kap = kappa_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_kap[i] = kappa_table_->eval(d_T[i]); });
  }
  mu_gf_.SetFromTrueDofs(visc_);
  kappa_gf_.SetFromTrueDofs(kappa_);

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
      // Assumed homogeneous Nuemann on T, so nothing to do
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

        ConstantCoefficient *Qt_bc_coeff = new ConstantCoefficient();
        Qt_bc_coeff->constant = 0.0;
        AddQtDirichletBC(Qt_bc_coeff, attr_wall);
      }
    }
    if (rank0_) std::cout << "Temp wall bc completed: " << numWalls << endl;
  }

  sfes_->GetEssentialTrueDofs(temp_ess_attr_, temp_ess_tdof_);
  sfes_->GetEssentialTrueDofs(Qt_ess_attr_, Qt_ess_tdof_);
  if (rank0_) std::cout << "LteThermoChem Essential true dof step" << endl;
}

void LteThermoChem::initializeOperators() {
  const double dt_ = time_coeff_.dt;

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
  Cp_coeff_ = new GridFunctionCoefficient(&Cp_gf_);
  rho_Cp_coeff_ = new ProductCoefficient(*rho_coeff_, *Cp_coeff_);

  bd0_over_dt.constant = 1.0 / dt_;
  rho_Cp_over_dt_coeff_ = new ProductCoefficient(bd0_over_dt, *rho_Cp_coeff_);

  thermal_diff_coeff_ = new GridFunctionCoefficient(&kappa_gf_);
  mut_coeff_ = new GridFunctionCoefficient(turbModel_interface_->eddy_viscosity);
  thermal_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *thermal_diff_coeff_, invPrt_, 1.0);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);
  thermal_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *thermal_diff_sum_coeff_);

  gradT_coeff_ = new GradientGridFunctionCoefficient(&Tn_next_gf_);
  kap_gradT_coeff_ = new ScalarVectorProductCoefficient(*thermal_diff_total_coeff_, *gradT_coeff_);

  un_next_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);
  rhon_next_coeff_ = new GridFunctionCoefficient(&rn_gf_);

  rho_Cp_u_coeff_ = new ScalarVectorProductCoefficient(*rho_Cp_coeff_, *un_next_coeff_);

  // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho Cp u \cdot \nabla \phi_j
  At_form_ = new ParBilinearForm(sfes_);
  auto *at_blfi = new ConvectionIntegrator(*rho_Cp_u_coeff_);
  if (numerical_integ_) {
    at_blfi->SetIntRule(&ir_nli);
  }
  At_form_->AddDomainIntegrator(at_blfi);
  if (partial_assembly_) {
    At_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);
  if (rank0_) std::cout << "LteThermoChem At operator set" << endl;

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

  // mass matrix with rho * Cp weight
  M_rho_Cp_form_ = new ParBilinearForm(sfes_);
  auto *mrc_blfi = new MassIntegrator(*rho_Cp_coeff_);
  if (numerical_integ_) {
    mrc_blfi->SetIntRule(&ir_i);
  }
  M_rho_Cp_form_->AddDomainIntegrator(mrc_blfi);
  if (partial_assembly_) {
    M_rho_Cp_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  M_rho_Cp_form_->Assemble();
  M_rho_Cp_form_->FormSystemMatrix(empty, M_rho_Cp_);

  // mass matrix with rho weight (used in Qt solve)
  M_rho_form_ = new ParBilinearForm(sfes_);
  auto *msrho_blfi = new MassIntegrator(*rho_Cp_coeff_);
  if (numerical_integ_) {
    msrho_blfi->SetIntRule(&ir_i);
  }
  M_rho_form_->AddDomainIntegrator(msrho_blfi);
  if (partial_assembly_) {
    M_rho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  M_rho_form_->Assemble();
  M_rho_form_->FormSystemMatrix(empty, M_rho_);

  // helmholtz
  Ht_form_ = new ParBilinearForm(sfes_);
  auto *hmt_blfi = new MassIntegrator(*rho_Cp_over_dt_coeff_);
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
  if (rank0_) std::cout << "LteThermoChem Ht operator set" << endl;

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
    M_rho_form_->AssembleDiagonal(diag_pa);
    MrhoInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MrhoInvPC_ = new HypreSmoother(*M_rho_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MrhoInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  MrhoInv_ = new CGSolver(sfes_->GetComm());
  MrhoInv_->iterative_mode = false;
  MrhoInv_->SetOperator(*M_rho_);
  MrhoInv_->SetPreconditioner(*MrhoInvPC_);
  MrhoInv_->SetPrintLevel(pl_solve_);
  MrhoInv_->SetRelTol(rtol_);
  MrhoInv_->SetMaxIter(max_iter_);

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
  if (rank0_) std::cout << "Temperature operators set" << endl;

  // Qt .....................................

  // Convection (for rho): Arho(i,j) = \int_{\Omega} \phi_i u \cdot \nabla \phi_j
  A_rho_form_ = new ParBilinearForm(sfes_);
  auto *ar_blfi = new ConvectionIntegrator(*un_next_coeff_);
  if (numerical_integ_) {
    ar_blfi->SetIntRule(&ir_nli);
  }
  A_rho_form_->AddDomainIntegrator(ar_blfi);
  if (partial_assembly_) {
    A_rho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  A_rho_form_->Assemble();
  A_rho_form_->FormSystemMatrix(empty, A_rho_);

  // copy IC to other temp containers
  Tn_gf_.GetTrueDofs(Tn_);
  Tn_next_gf_.SetFromTrueDofs(Tn_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);

  // After IC is filled, compute density
  updateDensity();

  // And prepare history
  rnm3_ = 0.0;
  rnm2_ = 0.0;
  rnm1_ = rn_;

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
void LteThermoChem::updateHistory() {
  // convection term
  NTnm2_ = NTnm1_;
  NTnm1_ = NTn_;

  // temperature
  Tnm2_ = Tnm1_;
  Tnm1_ = Tn_;
  Tn_next_gf_.GetTrueDofs(Tn_next_);
  Tn_ = Tn_next_;
  Tn_gf_.SetFromTrueDofs(Tn_);

  // density (used in Qt calc)
  rnm3_ = rnm2_;
  rnm2_ = rnm1_;
  rnm1_ = rn_;
}

void LteThermoChem::step() {
  const double dt_ = time_coeff_.dt;
  const double time_ = time_coeff_.time;

  // Set current time for velocity Dirichlet boundary conditions.
  for (auto &temp_dbc : temp_dbcs_) {
    temp_dbc.coeff->SetTime(time_ + dt_);
  }

  // Sequence:
  //   1) Extrapolate temperature
  //   2) Compute transport and thermo properties with extrapolated temperature
  //   3) Compute density with extrapolated temperature and updated Rgas
  //   4) Form operators and residual based on extrapolated state
  //   5) Solve for new temperature
  //   6) Compute provisional density (i.e., using old pressure)
  //   7) Update pressure and density to conserve mass (if closed)

  // Prepare for residual calc
  extrapolateTemperature();
  updateProperties();
  updateDensity();

  // Build the right-hand-side
  resT_ = 0.0;

  // convection
  computeExplicitTempConvectionOP();  // --> Writes to tmpR0_
  resT_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, Tn_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, Tnm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, Tnm2_);

  M_rho_Cp_->AddMult(tmpR0_, resT_, -1.0);

  // dPo/dt
  tmpR0_ = dtP_;
  Ms_->AddMult(tmpR0_, resT_);

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  bd0_over_dt.constant = (time_coeff_.bd0 / dt_);
  Ht_form_->Update();
  Ht_form_->Assemble();
  Ht_form_->FormSystemMatrix(temp_ess_tdof_, Ht_);

  HtInv_->SetOperator(*Ht_);
  if (partial_assembly_) {
    delete HtInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Ht_form_->AssembleDiagonal(diag_pa);
    HtInvPC_ = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof_);
    HtInv_->SetPreconditioner(*HtInvPC_);
  }

  // Prepare for the solve
  for (auto &temp_dbc : temp_dbcs_) {
    Tn_next_gf_.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(resT_, resT_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Ht_.As<ConstrainedOperator>();
    EliminateRHS(*Ht_form_, *HC, temp_ess_tdof_, Tn_next_gf_, resT_gf_, Xt2, Bt2, 1);
  } else {
    Ht_form_->FormLinearSystem(temp_ess_tdof_, Tn_next_gf_, resT_gf_, Ht_, Xt2, Bt2, 1);
  }

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

  // prepare for external use and next step
  updateProperties();
  updateDensity();
  updateThermoP();

  computeQt();

  updateHistory();
}

void LteThermoChem::computeExplicitTempConvectionOP() {
  Array<int> empty;
  At_form_->Update();
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);

  At_->Mult(Tn_, NTn_);

  // ab predictor
  tmpR0_.Set(time_coeff_.ab1, NTn_);
  tmpR0_.Add(time_coeff_.ab2, NTnm1_);
  tmpR0_.Add(time_coeff_.ab3, NTnm2_);
}

void LteThermoChem::initializeIO(IODataOrganizer &io) {
  io.registerIOFamily("Temperature", "/temperature", &Tn_gf_, false);
  io.registerIOVar("/temperature", "temperature", 0);
}

void LteThermoChem::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", &Tn_gf_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
  pvdc.RegisterField("Qt", &Qt_gf_);
  pvdc.RegisterField("Rgas", &Rgas_gf_);
  pvdc.RegisterField("Cp", &Cp_gf_);
}

void LteThermoChem::extrapolateTemperature() {
  // extrapolated temp at {n+1}
  Text_.Set(time_coeff_.ab1, Tn_);
  Text_.Add(time_coeff_.ab2, Tnm1_);
  Text_.Add(time_coeff_.ab3, Tnm2_);

  // store ext in _next containers
  Tn_next_gf_.SetFromTrueDofs(Text_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);
}

// update thermodynamic pressure
void LteThermoChem::updateThermoP() {
  if (domain_is_open_) return;

  const double dt_ = time_coeff_.dt;

  double allMass, PNM1;
  double myMass = 0.0;
  Ms_->Mult(rn_, tmpR0b_);
  myMass = tmpR0b_.Sum();
  MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  PNM1 = thermo_pressure_;

  // Scale pressure
  thermo_pressure_ = system_mass_ / allMass * PNM1;
  dtP_ = (thermo_pressure_ - PNM1) / dt_;

  // Scale density
  rn_ *= (system_mass_ / allMass);
  rn_gf_.SetFromTrueDofs(rn_);
}

void LteThermoChem::updateProperties() {
  // TODO(trevilo): Refactor for gpu support
  {
    const double *d_T = Tn_.Read();

    double *d_visc = visc_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_visc[i] = mu_table_->eval(d_T[i]); });

    double *d_kap = kappa_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_kap[i] = kappa_table_->eval(d_T[i]); });
  }
  mu_gf_.SetFromTrueDofs(visc_);
  kappa_gf_.SetFromTrueDofs(kappa_);

  // Update the gas constant and Cp
  {
    const double *d_T = Tn_next_.Read();

    double *d_Rgas = Rgas_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_Rgas[i] = Rgas_table_->eval(d_T[i]); });

    double *d_Cp = Cp_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_Cp[i] = Cp_table_->eval(d_T[i]); });
  }
  Rgas_gf_.SetFromTrueDofs(Rgas_);
  Cp_gf_.SetFromTrueDofs(Cp_);
}

void LteThermoChem::evaluatePlasmaConductivityGF() {
  {
    const double *d_T = Tn_next_.Read();

    double *d_sigma = sigma_.Write();
    MFEM_FORALL(i, Tn_.Size(), { d_sigma[i] = sigma_table_->eval(d_T[i]); });
  }
  sigma_gf_.SetFromTrueDofs(Rgas_);
}

void LteThermoChem::updateDensity() {
  // Update the density using the ideal gas law
  rn_ = thermo_pressure_;
  rn_ /= Rgas_;
  rn_ /= Tn_next_;
  rn_gf_.SetFromTrueDofs(rn_);

  // Update the density-weighted mass matrix
  Array<int> empty;
  M_rho_Cp_form_->Update();
  M_rho_Cp_form_->Assemble();
  M_rho_Cp_form_->FormSystemMatrix(empty, M_rho_Cp_);
}

void LteThermoChem::computeSystemMass() {
  // TODO(trevilo): Make sure rn_ is up to date when this is called
  double allMass = 0.0;
  double myMass = 0.0;
  Ms_->Mult(rn_, tmpR0b_);
  myMass = tmpR0b_.Sum();
  MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  system_mass_ = allMass;
}

/// Add a Dirichlet boundary condition to the temperature field
void LteThermoChem::AddTempDirichletBC(const double &temp, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, new ConstantCoefficient(temp));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }
}

void LteThermoChem::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, coeff);
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      // std::cout << "patch: " << i << " temp_ess_attr: " << temp_ess_attr_[i] << endl;
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }
}

void LteThermoChem::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void LteThermoChem::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr) {
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

void LteThermoChem::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddQtDirichletBC(new FunctionCoefficient(f), attr);
}

void LteThermoChem::computeQt() {
  Qt_ = 0.0;
  Qt_gf_.SetFromTrueDofs(Qt_);

  Array<int> empty;

  // Update rho weighted mass matrix and solver
  M_rho_form_->Update();
  M_rho_form_->Assemble();
  M_rho_form_->FormSystemMatrix(empty, M_rho_);
  MrhoInv_->SetOperator(*M_rho_);

  // Convection contribution
  A_rho_form_->Update();
  A_rho_form_->Assemble();
  A_rho_form_->FormSystemMatrix(empty, A_rho_);
  A_rho_->Mult(rn_, resT_);

  // Unsteady contribution
  const double dt = time_coeff_.dt;
  tmpR0_.Set(time_coeff_.bd0 / dt, rn_);
  tmpR0_.Add(time_coeff_.bd1 / dt, rnm1_);
  tmpR0_.Add(time_coeff_.bd2 / dt, rnm2_);
  tmpR0_.Add(time_coeff_.bd3 / dt, rnm3_);
  Ms_->AddMult(tmpR0_, resT_);

  // rho Qt = -D\rho/Dt (this .Neg() handles - on rhs)
  resT_.Neg();

  // Prep for solve
  sfes_->GetRestrictionMatrix()->MultTranspose(resT_, resT_gf_);

  Vector Xqt, Bqt;
  M_rho_form_->FormLinearSystem(Qt_ess_tdof_, Qt_gf_, resT_gf_, M_rho_, Xqt, Bqt, 1);

  // Solve
  MrhoInv_->Mult(Bqt, Xqt);
  M_rho_form_->RecoverFEMSolution(Xqt, resT_gf_, Qt_gf_);

  Qt_gf_.GetTrueDofs(Qt_);
}
