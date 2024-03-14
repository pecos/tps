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

#include "reactingFlow.hpp"

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

MFEM_HOST_DEVICE double Sutherland_lcl(const double T, const double mu_star, const double T_star, const double S_star) {
  const double T_rat = T / T_star;
  const double T_rat_32 = T_rat * sqrt(T_rat);
  const double S_rat = (T_star + S_star) / (T + S_star);
  return mu_star * T_rat_32 * S_rat;
}

ReactingFlow::ReactingFlow(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts,
                                                           temporalSchemeCoefficients &time_coeff, TPS::Tps *tps)
    : tpsP_(tps),
      pmesh_(pmesh),
      dim_(pmesh->Dimension()),      
      time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  /// Basic input information
  tpsP_->getInput("loMach/ambientPressure", ambient_pressure_, 101325.0);
  thermo_pressure_ = ambient_pressure_;
  dtP_ = 0.0;

  tpsP_->getInput("initialConditions/temperature", T_ic_, 300.0);

  filter_temperature_ = loMach_opts->filterTemp;
  filter_alpha_ = loMach_opts->filterWeight;
  filter_cutoff_modes_ = loMach_opts->nFilter;

  tpsP_->getInput("loMach/openSystem", domain_is_open_, false);

  tpsP_->getInput("loMach/reacting/linear-solver-rtol", rtol_, 1e-12);
  tpsP_->getInput("loMach/reacting/linear-solver-max-iter", max_iter_, 2000);
  tpsP_->getInput("loMach/reacting/linear-solver-verbosity", pl_solve_, 0);

  // plasma conditions. ???
  workFluid_ = USER_DEFINED;
  gasModel_ = PERFECT_MIXTURE;
  transportModel_ = ARGON_MINIMAL;
  // transportModel_ = ARGON_MIXTURE;  
  chemistryModel_ = NUM_CHEMISTRYMODEL;

  
  /// Minimal amount of info for mixture input struct
  Vector inputCV;
  mixtureInput_.f = workFluid_;  
  tpsP_->getInput("plasma_models/species_number", nSpecies_, 3);
  mixtureInput_.numSpecies = nSpecies_;
  tpsP_->getInput("plasma_models/ambipolar", mixtureInput_.ambipolar, false);  
  tpsP_->getInput("plasma_models/includeElectron", mixtureInput_.isElectronIncluded, true);
  tpsP_->getInput("plasma_models/two_temperature", mixtureInput_.twoTemperature, false);
  tpsP_->getInput("plasma_models/const_plasma_conductivity", const_plasma_conductivity_, 0.0);
  
  gasParams_.SetSize(nSpecies_, GasParams::NUM_GASPARAMS);  
  inputCV.SetSize(nSpecies_);

  for (int i = 1; i <= nSpecies_; i++) {
    double formEnergy, levelDegeneracy;
    std::string basepath("species/species" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/formation_energy").c_str(), formEnergy);
    tpsP_->getInput((basepath + "/level_degeneracy").c_str(), levelDegeneracy, 1.0);
    tpsP_->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(), inputCV(i - 1));

    gasParams_(i - 1, GasParams::FORMATION_ENERGY) = formEnergy;
    gasParams_(i - 1, GasParams::SPECIES_DEGENERACY) = levelDegeneracy;        
  }
      
  mixture_ = new PerfectMixture(mixtureInput_, dim_, dim_, const_plasma_conductivity_);

  
  /// Minimal amount of info for transport input struct
  std::vector<std::string> InputSpeciesNames;
  InputSpeciesNames.resize(nSpecies_);
  std::map<std::string, int> speciesMapping;
  // speciesMapping.resize(nSpecies_);  
  Vector speciesMass(nSpecies_);
  Vector speciesCharge(nSpecies_);
  Vector mixtureToInputMap(nSpecies_);
  Vector atomMW;
  int backgroundIndex;
  int paramIdx = 0;  
  int targetIdx;

  tpsP_->getRequiredInput("atoms/numAtoms", nAtoms_);
  if (nSpecies_ > 1) {
    tpsP_->getRequiredInput("species/background_index", backgroundIndex);
    atomMW.SetSize(nAtoms_);
    for (int a = 1; a <= nAtoms_; a++) {
      std::string basepath("atoms/atom" + std::to_string(a));
      std::string atomName;
      tpsP_->getRequiredInput((basepath + "/name").c_str(), atomName);
      tpsP_->getRequiredInput((basepath + "/mass").c_str(), atomMW(a - 1));
      atomMap_[atomName] = a - 1;
    }
  }  
  
  // not sure about this one...
  speciesNames_.resize(nSpecies_);
  speciesComposition_.SetSize(nSpecies_, nAtoms_);
  speciesComposition_ = 0.0;
  
  for (int i = 1; i <= nSpecies_; i++) {
    std::string speciesName;    
    std::string basepath("species/species" + std::to_string(i));
    tpsP_->getRequiredInput((basepath + "/name").c_str(), speciesName);    
    InputSpeciesNames[i - 1] = speciesName;    
  }
  
  tpsP_->getRequiredInput("species/background_index", backgroundIndex);
  tpsP_->getInput("plasma_models/transport_model/argon_minimal/third_order_thermal_conductivity", argonInput_.thirdOrderkElectron, true);
  tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/enabled", argonInput_.multiply, false);
  if (argonInput_.multiply) {
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/viscosity", argonInput_.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity", argonInput_.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",argonInput_.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity", argonInput_.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency", argonInput_.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/diffusivity", argonInput_.diffMult, 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/mobility", argonInput_.mobilMult, 1.0);
  }  
      
  // input file species index.    
  for (int sp = 0; sp < nSpecies_; sp++) { 
    if (sp == backgroundIndex - 1) {
      targetIdx = nSpecies_ - 1;
    } else if (InputSpeciesNames[sp] == "E") {
      targetIdx = nSpecies_ - 2;
      mixtureInput_.isElectronIncluded = true;
    } else {
      targetIdx = paramIdx;
      paramIdx++;
    }
    speciesMapping[InputSpeciesNames[sp]] = targetIdx;
    speciesNames_[targetIdx] = InputSpeciesNames[sp];
    mixtureToInputMap[targetIdx] = sp;    
  }

  DenseMatrix inputSpeciesComposition;
  inputSpeciesComposition.SetSize(nSpecies_, nAtoms_);
  inputSpeciesComposition = 0.0;
  for (int i = 1; i <= nSpecies_; i++) {
    std::string type, speciesName;
    std::vector<std::pair<std::string, std::string>> composition;
    std::string basepath("species/species" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/name").c_str(), speciesName);
    tpsP_->getRequiredPairs((basepath + "/composition").c_str(), composition);
    for (size_t c = 0; c < composition.size(); c++) {
      if (atomMap_.count(composition[c].first)) {
        int atomIdx = atomMap_[composition[c].first];
        inputSpeciesComposition(i - 1, atomIdx) = stoi(composition[c].second);
      } else {
        grvy_printf(GRVY_ERROR, "Requested atom %s for species %s is not available!\n", composition[c].first.c_str(), speciesName.c_str());
      }
    }
  }
  
  for (int sp = 0; sp < nSpecies_; sp++) {
    int inputIdx = mixtureToInputMap[sp];
    // initialMassFractions(sp) = inputInitialMassFraction(inputIdx);
    // constantMolarCV(sp) = inputCV(inputIdx);
    for (int a = 0; a < nAtoms_; a++) {
      speciesComposition_(sp, a) = inputSpeciesComposition(inputIdx, a);
    }
  }
  
  if (speciesMapping.count("Ar")) {
    argonInput_.neutralIndex = speciesMapping["Ar"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar' !\n");
    exit(ERROR);
  }
  if (speciesMapping.count("Ar.+1")) {
    argonInput_.ionIndex = speciesMapping["Ar.+1"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar.+1' !\n");
    exit(ERROR);
  }
  if (speciesMapping.count("E")) {
    argonInput_.electronIndex = speciesMapping["E"];
  } else {
    grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
    exit(ERROR);
  }

  Array<ArgonSpcs> speciesType(nSpecies_);
  identifySpeciesType(speciesType);  
  identifyCollisionType(speciesType, argonInput_.collisionIndex);  

  transport_ = new ArgonMinimalTransport(mixture_, argonInput_);
  
  
  /// Minimal amount of info for chemistry input struct  
  chemistryInput_.model = chemistryModel_;
  tpsP_->getInput("reactions/number_of_reactions", nReactions_, 0); 
  tpsP_->getInput("reactions/minimum_chemistry_temperature", chemistryInput_.minimumTemperature, 0.0);

  Vector reactionEnergies(nReactions_);  
  Vector detailedBalance(nReactions_);
  // Vector reactionModels(nReactions_);
  Array<ReactionModel> reactionModels;
  reactionModels.SetSize(nReactions_);
  Vector equilibriumConstantParams(nReactions_);  
  DenseMatrix reactantStoich(nSpecies_, nReactions_);
  DenseMatrix productStoich(nSpecies_, nReactions_);
  
  std::vector<mfem::Vector> rxnModelParamsHost;
  rxnModelParamsHost.clear();

  std::vector<std::string> reactionEquations;
  reactionEquations.resize(nReactions_);    
  
  for (int r = 1; r <= nReactions_; r++) {
    std::string basepath("reactions/reaction" + std::to_string(r));

    std::string equation, model;
    tpsP_->getRequiredInput((basepath + "/equation").c_str(), equation);
    reactionEquations[r - 1] = equation;
    tpsP_->getRequiredInput((basepath + "/model").c_str(), model);
    
    double energy;    
    tpsP_->getRequiredInput((basepath + "/reaction_energy").c_str(), energy);
    reactionEnergies[r - 1] = energy;
    
    if (model == "arrhenius") {
      reactionModels[r - 1] = ARRHENIUS;

      double A, b, E;
      tpsP_->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP_->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP_->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      rxnModelParamsHost.push_back(Vector({A, b, E}));

    } else if (model == "hoffert_lien") {
      reactionModels[r - 1] = HOFFERTLIEN;
      double A, b, E;
      tpsP_->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP_->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP_->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      rxnModelParamsHost.push_back(Vector({A, b, E}));

      // } else if (model == "tabulated") {
      // reactionModels[r - 1] = TABULATED_RXN;
      // std::string inputPath(basepath + "/tabulated");
      // readTable(inputPath, chemistryInput_.reactionInputs[r - 1].tableInput);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown reaction_model -> %s", model.c_str());
      exit(ERROR);
    }
    
    bool detailedBal;
    tpsP_->getInput((basepath + "/detailed_balance").c_str(), detailedBal, false);
    detailedBalance[r - 1] = detailedBal;
    
    if (detailedBal) {
      double A, b, E;
      tpsP_->getRequiredInput((basepath + "/equilibrium_constant/A").c_str(), A);
      tpsP_->getRequiredInput((basepath + "/equilibrium_constant/b").c_str(), b);
      tpsP_->getRequiredInput((basepath + "/equilibrium_constant/E").c_str(), E);

      // NOTE(kevin): array keeps max param in the indexing, as reactions can have diff no of params.
      equilibriumConstantParams[0 + (r - 1) * gpudata::MAXCHEMPARAMS] = A;
      equilibriumConstantParams[1 + (r - 1) * gpudata::MAXCHEMPARAMS] = b;
      equilibriumConstantParams[2 + (r - 1) * gpudata::MAXCHEMPARAMS] = E;
    }

    Array<double> stoich(nSpecies_);
    tpsP_->getRequiredVec((basepath + "/reactant_stoichiometry").c_str(), stoich, nSpecies_);
    
    for (int sp = 0; sp < nSpecies_; sp++) {
      int inputSp = mixtureToInputMap[sp];
      reactantStoich(sp, r - 1) = stoich[inputSp];
    }
    
    tpsP_->getRequiredVec((basepath + "/product_stoichiometry").c_str(), stoich, nSpecies_);
    for (int sp = 0; sp < nSpecies_; sp++) {
      int inputSp = mixtureToInputMap[sp];
      productStoich(sp, r - 1) = stoich[inputSp];
    }    
  }

  if (speciesMapping.count("E")) {
    chemistryInput_.electronIndex = speciesMapping["E"];
  } else {
    chemistryInput_.electronIndex = -1;
  }
  
  size_t rxn_param_idx = 0;
  chemistryInput_.numReactions = nReactions_;  
  for (int r = 0; r < nReactions_; r++) {
    chemistryInput_.reactionEnergies[r] = reactionEnergies[r];
    chemistryInput_.detailedBalance[r] = detailedBalance[r];
    for (int sp = 0; sp < nSpecies_; sp++) {
      chemistryInput_.reactantStoich[sp + r * nSpecies_] = reactantStoich(sp, r);
      chemistryInput_.productStoich[sp + r * nSpecies_] = productStoich(sp, r);
    }

    chemistryInput_.reactionModels[r] = reactionModels[r];
    for (int p = 0; p < gpudata::MAXCHEMPARAMS; p++) {
      chemistryInput_.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS] = equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS];
    }

    if (reactionModels[r] != TABULATED_RXN) {
      assert(rxn_param_idx < rxnModelParamsHost.size());
      chemistryInput_.reactionInputs[r].modelParams = rxnModelParamsHost[rxn_param_idx].Read();
      rxn_param_idx += 1;
    }
  }   
  
  chemistry_ = new Chemistry(mixture_, chemistryInput_);  
}

ReactingFlow::~ReactingFlow() {
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

  delete HyInv_;
  delete HyInvPC_;
  delete Hy_form_;  
  delete species_diff_coeff_;
  delete species_diff_sum_coeff_;
  delete species_diff_total_coeff_;

  delete mixture_;
  delete transport_;
  delete chemistry_;
  
  // allocated in initializeSelf
  delete sfes_;
  delete sfec_;
  delete yfes_;
  delete yfec_;  
}

void ReactingFlow::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing ReactingFlow solver.\n");

  //-----------------------------------------------------
  // 1) Prepare the required finite element objects
  //-----------------------------------------------------
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  yfec_ = new H1_FECollection(order_, nSpecies_);
  yfes_ = new ParFiniteElementSpace(pmesh_, yfec_, nSpecies_);
  
  // Check if fully periodic mesh
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    temp_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    temp_ess_attr_ = 0;

    Qt_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    Qt_ess_attr_ = 0;

    spec_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    spec_ess_attr_ = 0;    
  }
  if (rank0_) grvy_printf(ginfo, "ReactingFlow paces constructed...\n");

  int sDofInt_ = sfes_->GetTrueVSize();
  int yDofInt_ = yfes_->GetTrueVSize();  

  Qt_.SetSize(sDofInt_);
  Qt_ = 0.0;
  Qt_gf_.SetSpace(sfes_);
  Qt_gf_ = 0.0;

  Tn_.SetSize(sDofInt_);
  Tn_next_.SetSize(sDofInt_);

  Tnm1_.SetSize(sDofInt_);
  Tnm2_.SetSize(sDofInt_);

  // temp convection terms
  NTn_.SetSize(sDofInt_);
  NTnm1_.SetSize(sDofInt_);
  NTnm2_.SetSize(sDofInt_);
  NTn_ = 0.0;
  NTnm1_ = 0.0;
  NTnm2_ = 0.0;

  Text_.SetSize(sDofInt_);
  Text_gf_.SetSpace(sfes_);

  resT_gf_.SetSpace(sfes_);
  resT_.SetSize(sDofInt_);
  resT_ = 0.0;

  Tn_next_gf_.SetSpace(sfes_);
  Tn_gf_.SetSpace(sfes_);
  Tnm1_gf_.SetSpace(sfes_);
  Tnm2_gf_.SetSpace(sfes_);

  // species
  Yn_.SetSize(yDofInt_);
  Yn_next_.SetSize(yDofInt_);
  Ynm1_.SetSize(yDofInt_);
  Ynm2_.SetSize(yDofInt_);
  Yext_.SetSize(yDofInt_);
  Yn_ = 0.0;
  Ynm1_ = 0.0;
  Ynm2_ = 0.0;
  
  resY_.SetSize(sDofInt_);  
  resY_ = 0.0;

  hw_.SetSize(yDofInt_);
  hw_ = 0.0;

  // only YnFull for plotting
  YnFull_gf_.SetSpace(yfes_);  

  // rest can just be sfes
  Yn_gf_.SetSpace(sfes_);  
  Yext_gf_.SetSpace(sfes_);  
  resY_gf_.SetSpace(sfes_);  
  Yn_next_gf_.SetSpace(sfes_);
  Ynm1_gf_.SetSpace(sfes_);
  Ynm2_gf_.SetSpace(sfes_);  
  
  rn_.SetSize(sDofInt_);
  rn_ = 1.0;
  rn_gf_.SetSpace(sfes_);
  rn_gf_ = 1.0;

  visc_.SetSize(sDofInt_);
  visc_ = 1.0e-12;

  visc_gf_.SetSpace(sfes_);
  visc_gf_ = 0.0;  

  kappa_.SetSize(sDofInt_);
  kappa_ = 1.0e-12;

  kappa_gf_.SetSpace(sfes_);
  kappa_gf_ = 0.0;

  diffY_.SetSize(yDofInt_);
  diffY_ = 1.0e-12;

  diffY_gf_.SetSpace(sfes_);
  diffY_gf_ = 0.0;

  prodY_.SetSize(yDofInt_);
  prodY_ = 1.0e-12;

  hw_.SetSize(yDofInt_);
  hw_ = 0.0;
  
  prodY_gf_.SetSpace(sfes_);
  prodY_gf_ = 0.0;    

  tmpR0_.SetSize(sDofInt_);
  tmpR0a_.SetSize(sDofInt_);  
  tmpR0b_.SetSize(sDofInt_);
  tmpR0c_.SetSize(sDofInt_);  

  R0PM0_gf_.SetSpace(sfes_);

  rhoDt.SetSpace(sfes_);

  if (rank0_) grvy_printf(ginfo, "ReactingFlow vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &visc_gf_;
  toFlow_interface_.thermal_divergence = &Qt_gf_;
  toTurbModel_interface_.density = &rn_gf_;
  if (rank0_) {
    std::cout << "exports set..." << endl;
  }

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

        ConstantCoefficient *Qt_bc_coeff = new ConstantCoefficient();
        Qt_bc_coeff->constant = 0.0;
        AddQtDirichletBC(Qt_bc_coeff, attr_wall);
      }
    }
    if (rank0_) std::cout << "Temp wall bc completed: " << numWalls << endl;
  }

  sfes_->GetEssentialTrueDofs(temp_ess_attr_, temp_ess_tdof_);
  sfes_->GetEssentialTrueDofs(Qt_ess_attr_, Qt_ess_tdof_);
  if (rank0_) std::cout << "ReactingFlow Essential true dof step" << endl;
}

void ReactingFlow::initializeOperators() {
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
  mut_coeff_ = new GridFunctionCoefficient(turbModel_interface_->eddy_viscosity);
  thermal_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *thermal_diff_coeff_, invPr_, 1.0);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);
  thermal_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *thermal_diff_sum_coeff_);
  gradT_coeff_ = new GradientGridFunctionCoefficient(&Tn_next_gf_);
  kap_gradT_coeff_ = new ScalarVectorProductCoefficient(*thermal_diff_total_coeff_, *gradT_coeff_);

  // species diff coeff, diffY_gf gets filled with correct species diffusivity before its solve
  species_diff_coeff_ = new GridFunctionCoefficient(&diffY_gf_);
  species_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *species_diff_coeff_, invSc_, 1.0); // is this correct?
  species_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *species_diff_sum_coeff_);
  
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
  At_form_->FormSystemMatrix(empty, At_);
  if (rank0_) std::cout << "ReactingFlow At operator set" << endl;

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
  if (rank0_) std::cout << "ReactingFlow MsRho operator set" << endl;

  // temperature Helmholtz
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
  if (rank0_) std::cout << "ReactingFlow Ht operator set" << endl;

  // species Helmholtz
  Hy_form_ = new ParBilinearForm(sfes_);
  auto *hmy_blfi = new MassIntegrator(*rho_over_dt_coeff_);
  auto *hdy_blfi = new DiffusionIntegrator(*species_diff_total_coeff_);

  if (numerical_integ_) {
    hmy_blfi->SetIntRule(&ir_di);
    hdy_blfi->SetIntRule(&ir_di);
  }
  Hy_form_->AddDomainIntegrator(hmy_blfi);
  Hy_form_->AddDomainIntegrator(hdy_blfi);
  if (partial_assembly_) {
    Hy_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Hy_form_->Assemble();
  Hy_form_->FormSystemMatrix(spec_ess_tdof_, Hy_);
  if (rank0_) std::cout << "ReactingFlow Hy operator set" << endl;
  
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
  if (rank0_) std::cout << "Temperature operators set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Hy_form_->AssembleDiagonal(diag_pa);
    HyInvPC_ = new OperatorJacobiSmoother(diag_pa, spec_ess_tdof_);
  } else {
    HyInvPC_ = new HypreSmoother(*Hy_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HyInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }

  HyInv_ = new CGSolver(sfes_->GetComm());
  HyInv_->iterative_mode = true;
  HyInv_->SetOperator(*Hy_);
  HyInv_->SetPreconditioner(*HyInvPC_);
  HyInv_->SetPrintLevel(pl_solve_);
  HyInv_->SetRelTol(rtol_);
  HyInv_->SetMaxIter(max_iter_);
  if (rank0_) std::cout << "Species operators set" << endl;
  
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
  if (rank0_) std::cout << "ReactingFlow Mq operator set" << endl;

  if (partial_assembly_) {
    Vector diag_pa(sfes_->GetTrueVSize());
    Mq_form_->AssembleDiagonal(diag_pa);
    MqInvPC_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MqInvPC_ = new HypreSmoother(*Mq_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MqInvPC_)->SetType(HypreSmoother::Jacobi, 1);
  }
  MqInv_ = new CGSolver(sfes_->GetComm());
  MqInv_->iterative_mode = false;
  MqInv_->SetOperator(*Mq_);
  MqInv_->SetPreconditioner(*MqInvPC_);
  MqInv_->SetPrintLevel(pl_solve_);
  MqInv_->SetRelTol(rtol_);
  MqInv_->SetMaxIter(max_iter_);

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
  LQ_form_->FormSystemMatrix(empty, LQ_);

  LQ_bdry_ = new ParLinearForm(sfes_);
  auto *lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff_, 2, -1);
  if (numerical_integ_) {
    lq_bdry_lfi->SetIntRule(&ir_di);
  }
  LQ_bdry_->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr_);
  if (rank0_) std::cout << "ReactingFlow LQ operator set" << endl;

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
void ReactingFlow::UpdateTimestepHistory(double dt) {
  // temperature
  NTnm2_ = NTnm1_;
  NTnm1_ = NTn_;
  Tnm2_ = Tnm1_;
  Tnm1_ = Tn_;
  Tn_next_gf_.GetTrueDofs(Tn_next_);
  Tn_ = Tn_next_;
  Tn_gf_.SetFromTrueDofs(Tn_);
}

void ReactingFlow::step() {
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
  speciesProduction();

  // advance species, zero slot is from calculated sum of others
  for (int iSpecies = 1; iSpecies < nSpecies_; iSpecies++) {
    speciesStep(iSpecies);
  }
  speciesOneStep();
  YnFull_gf_.SetFromTrueDofs(Yn_next_);
  heatOfFormation();

  // advance temperature
  temperatureStep();

  // explicit filter temperature
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

  // explicit filter species (is this necessary?)
  
  // prepare for external use
  updateDensity(1.0);
  // computeQt();
  computeQtTO();

  UpdateTimestepHistory(dt_);
}

void ReactingFlow::temperatureStep() {
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

  // heat of formation
  Ms_->AddMult(hw_, resT_);  

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt = rn_gf_;
  rhoDt *= (time_coeff_.bd0 / dt_);

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

}

void ReactingFlow::speciesOneStep() {
  tmpR0_ = 0.0;
  double *dataYn = Yn_next_.HostReadWrite();
  double *dataY0 = tmpR0_.HostReadWrite();    
  for (int n = 1; n < nSpecies_; n++) {
    for (int i = 0; i < sDofInt_; i++) {
      dataY0[i] += dataYn[i + n * sDofInt_];
    }
  }
  for (int i = 0; i < sDofInt_; i++) {
    dataYn[i + 0 * sDofInt_] = 1.0 - dataY0[i];
  }  
}

void ReactingFlow::speciesStep(int iSpec) {

  // copy relevant species properties from full Vector to particular case
  setScalarFromVector(diffY_, iSpec, &tmpR0_);  
  diffY_gf_.SetFromTrueDofs(tmpR0_);  
  
  // Build the right-hand-side  
  resY_ = 0.0;

  // convection
  computeExplicitSpecConvectionOP(iSpec, true);  // ->tmpR0_
  resY_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  setScalarFromVector(Yn_,   iSpec, &tmpR0a_);
  setScalarFromVector(Ynm1_, iSpec, &tmpR0b_);
  setScalarFromVector(Ynm2_, iSpec, &tmpR0c_);     
  tmpR0_.Set(time_coeff_.bd1 / dt_, tmpR0a_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, tmpR0b_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, tmpR0c_);

  MsRho_->AddMult(tmpR0_, resY_, -1.0);

  // production of iSpec
  setScalarFromVector(prodY_, iSpec, &tmpR0_);
  Ms_->AddMult(tmpR0_, resT_);

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt = rn_gf_;
  rhoDt *= (time_coeff_.bd0 / dt_);

  Hy_form_->Update();
  Hy_form_->Assemble();
  Hy_form_->FormSystemMatrix(spec_ess_tdof_, Hy_);

  HyInv_->SetOperator(*Hy_);
  if (partial_assembly_) {
    delete HyInvPC_;
    Vector diag_pa(sfes_->GetTrueVSize());
    Hy_form_->AssembleDiagonal(diag_pa);
    HyInvPC_ = new OperatorJacobiSmoother(diag_pa, spec_ess_tdof_);
    HyInv_->SetPreconditioner(*HyInvPC_);
  }

  // Prepare for the solve
  for (auto &spec_dbc : spec_dbcs_) {
    Yn_next_gf_.ProjectBdrCoefficient(*spec_dbc.coeff, spec_dbc.attr);
  }
  sfes_->GetRestrictionMatrix()->MultTranspose(resY_, resY_gf_);

  Vector Xt2, Bt2;
  if (partial_assembly_) {
    auto *HC = Hy_.As<ConstrainedOperator>();
    EliminateRHS(*Hy_form_, *HC, spec_ess_tdof_, Yn_next_gf_, resY_gf_, Xt2, Bt2, 1);
  } else {
    Hy_form_->FormLinearSystem(spec_ess_tdof_, Yn_next_gf_, resY_gf_, Hy_, Xt2, Bt2, 1);
  }

  // solve helmholtz eq for iSpec
  HyInv_->Mult(Bt2, Xt2);
  assert(HyInv_->GetConverged());

  // copy into full species vector & gf
  Hy_form_->RecoverFEMSolution(Xt2, resT_gf_, Yn_next_gf_);
  Yn_next_gf_.GetTrueDofs(tmpR0_);
  setVectorFromScalar(tmpR0_, iSpec, &Yn_next_);
}

void ReactingFlow::speciesProduction() {
  for (int n = 0; n < nSpecies_; n++) {
    for (int i = 0; i < sDofInt_; i++) {
      prodY_[i + n * sDofInt_] = 0.0;
    }
  }
}

void ReactingFlow::heatOfFormation() {
  double *data = hw_.HostReadWrite();
  double *dw = prodY_.HostReadWrite();
  double ho;
  for (int n = 0; n < nSpecies_; n++) {
    ho = 0.0; // replace with each species enth of formation
    for (int i = 0; i < sDofInt_; i++) {
      hw_[i + n * sDofInt_] = ho * dw[i + n * sDofInt_];
    }
  }
}

void ReactingFlow::computeExplicitTempConvectionOP(bool extrap) {
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

void ReactingFlow::computeExplicitSpecConvectionOP(int iSpec, bool extrap) {

  Array<int> empty;
  At_form_->Update();
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);
  if (extrap == true) {
    setScalarFromVector(Yn_, iSpec, &tmpR0_);
  } else {
    setScalarFromVector(Yext_, iSpec, &tmpR0_);
  }
  At_->Mult(tmpR0_, tmpR0a_);

  // ab predictor
  if (extrap == true) {
    setScalarFromVector(NYnm1_, iSpec, &tmpR0b_);
    setScalarFromVector(NYnm2_, iSpec, &tmpR0c_);         
    tmpR0_.Set(time_coeff_.ab1, tmpR0a_);
    tmpR0_.Add(time_coeff_.ab2, tmpR0b_);
    tmpR0_.Add(time_coeff_.ab3, tmpR0c_);
  } else {
    tmpR0_.Set(1.0, tmpR0a_);
  }
  setVectorFromScalar(tmpR0a_, iSpec, &NYn_);   
}

void ReactingFlow::initializeIO(IODataOrganizer &io) {
  io.registerIOFamily("Temperature", "/temperature", &Tn_gf_, false);
  io.registerIOVar("/temperature", "temperature", 0);
  
  io.registerIOFamily("Species", "/species", &YnFull_gf_, false);
  io.registerIOVar("/species", "Spec1", 0);
  if (nSpecies_ >= 2) io.registerIOVar("/velocity", "Spec2", 1);
  if (nSpecies_ >= 3) io.registerIOVar("/velocity", "Spec3", 2);
  if (nSpecies_ >= 4) io.registerIOVar("/velocity", "Spec4", 3);
  if (nSpecies_ == 5) io.registerIOVar("/velocity", "Spec5", 4);
  
}

void ReactingFlow::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", &Tn_gf_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
  pvdc.RegisterField("Qt", &Qt_gf_);
  pvdc.RegisterField("Species", &YnFull_gf_);  
}

/**
   Update boundary conditions for Temperature, useful for
   ramping a dirichlet condition over time
 */
void ReactingFlow::updateBC(int current_step) {
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

void ReactingFlow::extrapolateState() {
  // extrapolated temp at {n+1}
  Text_.Set(time_coeff_.ab1, Tn_);
  Text_.Add(time_coeff_.ab2, Tnm1_);
  Text_.Add(time_coeff_.ab3, Tnm2_);

  // store ext in _next containers, remove Text_, Uext completely later
  Tn_next_gf_.SetFromTrueDofs(Text_);
  Tn_next_gf_.GetTrueDofs(Tn_next_);

  // extrapolated species at {n+1}
  Yext_.Set(time_coeff_.ab1, Yn_);
  Yext_.Add(time_coeff_.ab2, Ynm1_);
  Yext_.Add(time_coeff_.ab3, Ynm2_);  
}

// update thermodynamic pressure
void ReactingFlow::updateThermoP() {
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

void ReactingFlow::updateDiffusivity() {

  /*
  double vel[gpudata::MAXDIM];
  double vtmp[gpudata::MAXDIM];
  double stress[gpudata::MAXDIM * gpudata::MAXDIM];

  // TODO(kevin): update E-field with EM coupling.
  double Efield[gpudata::MAXDIM];
  for (int v = 0; v < nvel; v++) Efield[v] = 0.0;

  double speciesEnthalpies[gpudata::MAXSPECIES];
  mixture->computeSpeciesEnthalpies(state, speciesEnthalpies);

  double transportBuffer[FluxTrns::NUM_FLUX_TRANS];
  // NOTE(kevin): in flux, only dim-components of diffusionVelocity will be used.
  double diffusionVelocity[gpudata::MAXSPECIES * gpudata::MAXDIM];

  transport->ComputeFluxTransportProperties(state, gradUp, Efield, radius, distance, transportBuffer,
                                            diffusionVelocity);
  double visc = transportBuffer[FluxTrns::VISCOSITY];
  double bulkViscosity = transportBuffer[FluxTrns::BULK_VISCOSITY];
  bulkViscosity -= 2. / 3. * visc;
  double k = transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY];
  double ke = transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY];
  double Pr_Cp = visc / k;

  if (twoTemperature) {
    for (int d = 0; d < dim; d++) {
      double qeFlux = ke * gradUp[num_equation - 1 + d * num_equation];
      flux[1 + nvel + d * num_equation] += qeFlux;
      flux[num_equation - 1 + d * num_equation] += qeFlux;
      flux[num_equation - 1 + d * num_equation] -=
          speciesEnthalpies[numSpecies - 2] * diffusionVelocity[numSpecies - 2 + d * numSpecies];
    }
  } else {
    k += ke;
  }

  // make sure density visc. flux is 0
  for (int d = 0; d < dim; d++) flux[0 + d * num_equation] = 0.;

  for (int d = 0; d < dim; d++) {
    flux[(1 + nvel) + d * num_equation] += vtmp[d];
    flux[(1 + nvel) + d * num_equation] += k * gradUp[(1 + nvel) + d * num_equation];
    // compute diffusive enthalpy flux.
    for (int sp = 0; sp < numSpecies; sp++) {
      flux[(1 + nvel) + d * num_equation] -= speciesEnthalpies[sp] * diffusionVelocity[sp + d * numSpecies];
    }
  }

  // if (eqSystem == NS_PASSIVE) {
  //   double Sc = mixture->GetSchmidtNum();
  //   for (int d = 0; d < dim; d++) flux(num_equation - 1, d) = visc / Sc * gradUp(num_equation - 1, d);
  // }
  // NOTE: NS_PASSIVE will not be needed (automatically incorporated).
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // NOTE: diffusionVelocity is set to be (numSpecies,nvel)-matrix.
    // however only dim-components are used for flux.
    for (int d = 0; d < dim; d++)
      flux[(nvel + 2 + sp) + d * num_equation] = -state[nvel + 2 + sp] * diffusionVelocity[sp + d * numSpecies];
  }
  */


  // >>> OLD
  
  // viscosity
  if (!constant_viscosity_) {
    double *d_visc = visc_.Write();
    const double *d_T = Tn_.Read();
    const double mu_star = mu0_;
    const double T_star = sutherland_T0_;
    const double S_star = sutherland_S0_;
    MFEM_FORALL(i, Tn_.Size(), { d_visc[i] = Sutherland_lcl(d_T[i], mu_star, T_star, S_star); });
  } else {
    visc_ = mu0_;
  }

  visc_gf_.SetFromTrueDofs(visc_);

  // NB: Here "kappa" is kappa/Cp = mu/Pr
  kappa_ = visc_;
  kappa_ /= Pr_;
  kappa_gf_.SetFromTrueDofs(kappa_);

  // temporary
  for (int n = 0; n < nSpecies_; n++) {
    setVectorFromScalar(visc_, n, &diffY_);    
  }
  diffY_ /= Sc_;      
  
}

void ReactingFlow::updateDensity(double tStep) {
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

void ReactingFlow::computeSystemMass() {
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
void ReactingFlow::AddTempDirichletBC(const double &temp, Array<int> &attr) {
  temp_dbcs_.emplace_back(attr, new ConstantCoefficient(temp));
  for (int i = 0; i < attr.Size(); ++i) {
    if (attr[i] == 1) {
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }
}

void ReactingFlow::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr) {
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

void ReactingFlow::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void ReactingFlow::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr) {
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

void ReactingFlow::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddQtDirichletBC(new FunctionCoefficient(f), attr);
}

void ReactingFlow::computeQtTO() {
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

/// identifySpeciesType and identifyCollisionType copies from M2ulPhyS
void ReactingFlow::identifySpeciesType(Array<ArgonSpcs> &speciesType) {
  speciesType.SetSize(nSpecies_);
  
  for (int sp = 0; sp < nSpecies_; sp++) {
    speciesType[sp] = NONE_ARGSPCS;

    Vector spComp(nAtoms_);
    speciesComposition_.GetRow(sp, spComp);

    if (spComp(atomMap_["Ar"]) == 1.0) {
      if (spComp(atomMap_["E"]) == -1.0) {
        speciesType[sp] = AR1P;
      } else {
        bool argonMonomer = true;
        for (int a = 0; a < nAtoms_; a++) {
          if (a == atomMap_["Ar"]) continue;
          if (spComp(a) != 0.0) {
            argonMonomer = false;
            break;
          }
        }
        if (argonMonomer) {
          speciesType[sp] = AR;
        } else {
          // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
          std::string name = speciesNames_[sp];
          grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                      name.c_str());
          exit(-1);
        }
      }
    } else if (spComp(atomMap_["E"]) == 1.0) {
      bool electron = true;
      for (int a = 0; a < nAtoms_; a++) {
        if (a == atomMap_["E"]) continue;
        if (spComp(a) != 0.0) {
          electron = false;
          break;
        }
      }
      if (electron) {
        speciesType[sp] = ELECTRON;
      } else {
        // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
        std::string name = speciesNames_[sp];
        grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                    name.c_str());
        exit(-1);
      }
    } else {
      // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
      std::string name = speciesNames_[sp];
      grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                  name.c_str());
      exit(-1);
    }
  }

  // Check all species are identified.
  for (int sp = 0; sp < nSpecies_; sp++) {
    if (speciesType[sp] == NONE_ARGSPCS) {
      // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
      std::string name = speciesNames_[sp];
      grvy_printf(GRVY_ERROR, "The species %s is not identified in ArgonMixtureTransport! \n", name.c_str());
      exit(-1);
    }
  }

  return;
}

void ReactingFlow::identifyCollisionType(const Array<ArgonSpcs> &speciesType, ArgonColl *collisionIndex) {
  // collisionIndex_.resize(numSpecies);
  for (int spI = 0; spI < nSpecies_; spI++) {
    // collisionIndex_[spI].resize(numSpecies - spI);
    for (int spJ = spI; spJ < nSpecies_; spJ++) {
      // If not initialized, will raise an error.
      collisionIndex[spI + spJ * nSpecies_] = NONE_ARGCOLL;

      const double pairType =
          gasParams_(spI, GasParams::SPECIES_CHARGES) * gasParams_(spJ, GasParams::SPECIES_CHARGES);
      if (pairType > 0.0) {  // Repulsive screened Coulomb potential
        collisionIndex[spI + spJ * nSpecies_] = CLMB_REP;
      } else if (pairType < 0.0) {  // Attractive screened Coulomb potential
        collisionIndex[spI + spJ * nSpecies_] = CLMB_ATT;
      } else {  // determines collision type by species pairs.
        if ((speciesType[spI] == AR) && (speciesType[spJ] == AR)) {
          collisionIndex[spI + spJ * nSpecies_] = AR_AR;
        } else if (((speciesType[spI] == AR) && (speciesType[spJ] == AR1P)) ||
                   ((speciesType[spI] == AR1P) && (speciesType[spJ] == AR))) {
          collisionIndex[spI + spJ * nSpecies_] = AR_AR1P;
        } else if (((speciesType[spI] == AR) && (speciesType[spJ] == ELECTRON)) ||
                   ((speciesType[spI] == ELECTRON) && (speciesType[spJ] == AR))) {
          collisionIndex[spI + spJ * nSpecies_] = AR_E;
        } else {
          // std::string name1 = speciesNames_[(*mixtureToInputMap_)[spI]];
          // std::string name2 = speciesNames_[(*mixtureToInputMap_)[spJ]];
          std::string name1 = speciesNames_[spI];
          std::string name2 = speciesNames_[spJ];
          grvy_printf(GRVY_ERROR, "%s-%s is not supported in ArgonMixtureTransport! \n", name1.c_str(), name2.c_str());
          exit(-1);
        }
      }
    }
  }

  // Check all collision types are initialized.
  for (int spI = 0; spI < nSpecies_; spI++) {
    for (int spJ = spI; spJ < nSpecies_; spJ++) {
      if (collisionIndex[spI + spJ * nSpecies_] == NONE_ARGCOLL) {
        // std::string name1 = speciesNames_[(*mixtureToInputMap_)[spI]];
        // std::string name2 = speciesNames_[(*mixtureToInputMap_)[spJ]];
        std::string name1 = speciesNames_[spI];
        std::string name2 = speciesNames_[spJ];
        grvy_printf(GRVY_ERROR, "%s-%s is not initialized in ArgonMixtureTransport! \n", name1.c_str(), name2.c_str());
        exit(-1);
      }
    }
  }

  return;
}


#if 0
void ReactingFlow::uniformInlet() {
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

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
