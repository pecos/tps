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
#include "radiation.hpp"
#include "tps2Boltzmann.hpp"

using namespace mfem;
using namespace mfem::common;

/// forward declarations
double species_stepLeft(const Vector &coords, double t);
double species_stepRight(const Vector &coords, double t);
double species_uniform(const Vector &coords, double t);
double binaryTest(const Vector &coords, double t);

static double radius(const Vector &pos) { return pos[0]; }
static FunctionCoefficient radius_coeff(radius);

ReactingFlow::ReactingFlow(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff,
                           TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), dim_(pmesh->Dimension()), time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  /// Basic input information
  tpsP_->getInput("loMach/ambientPressure", ambient_pressure_, 101325.0);
  thermo_pressure_ = ambient_pressure_;
  Pnm1_ = thermo_pressure_;
  Pnm2_ = Pnm1_;
  Pnm3_ = Pnm1_;
  dtP_ = 0.0;

  tps->getInput("loMach/axisymmetric", axisym_, false);

  tpsP_->getInput("initialConditions/temperature", T_ic_, 300.0);

  filter_temperature_ = loMach_opts->filterTemp;
  filter_alpha_ = loMach_opts->filterWeight;
  filter_cutoff_modes_ = loMach_opts->nFilter;

  tpsP_->getInput("loMach/openSystem", domain_is_open_, false);

  tpsP_->getInput("loMach/reacting/linear-solver-rtol", rtol_, 1e-12);
  tpsP_->getInput("loMach/reacting/linear-solver-max-iter", max_iter_, 2000);
  tpsP_->getInput("loMach/reacting/linear-solver-verbosity", pl_solve_, 0);

  tpsP_->getInput("loMach/reacting/eddy-Pr", Pr_, 0.72);
  tpsP_->getInput("loMach/reacting/eddy-Sc", Sc_, 1.0);

  // plasma conditions. ???
  workFluid_ = USER_DEFINED;
  gasModel_ = PERFECT_MIXTURE;
  transportModel_ = ARGON_MIXTURE;
  chemistryModel_ = NUM_CHEMISTRYMODEL;

  /// Minimal amount of info for mixture input struct
  mixtureInput_.f = workFluid_;
  tpsP_->getInput("plasma_models/species_number", nSpecies_, 3);
  mixtureInput_.numSpecies = nSpecies_;
  tpsP_->getInput("plasma_models/ambipolar", mixtureInput_.ambipolar, false);
  // tpsP_->getInput("plasma_models/includeElectron", mixtureInput_.isElectronIncluded, true);
  tpsP_->getInput("plasma_models/two_temperature", mixtureInput_.twoTemperature, false);
  tpsP_->getInput("plasma_models/const_plasma_conductivity", const_plasma_conductivity_, 0.0);
  tpsP_->getInput("plasma_models/is_rad_decay_in_NEC", radiative_decay_NECincluded_, true);

  if (mixtureInput_.twoTemperature) {
    if (rank0_) std::cout << "Two temperature is not yet supported in low Mach reacting flow." << std::endl;
    exit(ERROR);
  }

  gasParams_.SetSize(nSpecies_, GasParams::NUM_GASPARAMS);
  gasParams_ = 0.0;
  speciesMolarCv_.SetSize(nSpecies_);
  speciesMolarCp_.SetSize(nSpecies_);
  specificHeatRatios_.SetSize(nSpecies_);
  initialMassFraction_.SetSize(nSpecies_);

  for (int i = 0; i < nSpecies_ * GasParams::NUM_GASPARAMS; i++) {
    mixtureInput_.gasParams[i] = 0.0;
  }
  for (int i = 0; i < nSpecies_; i++) {
    mixtureInput_.molarCV[i] = 0.0;
  }

  for (int i = 1; i <= nSpecies_; i++) {
    double formEnergy, levelDegeneracy;
    std::string basepath("species/species" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/formation_energy").c_str(), formEnergy);
    tpsP_->getInput((basepath + "/level_degeneracy").c_str(), levelDegeneracy, 1.0);
    tpsP_->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(), speciesMolarCv_[i - 1]);

    gasParams_((i - 1), GasParams::FORMATION_ENERGY) = formEnergy;
    gasParams_((i - 1), GasParams::SPECIES_DEGENERACY) = levelDegeneracy;

    mixtureInput_.gasParams[(i - 1) + nSpecies_ * GasParams::FORMATION_ENERGY] = formEnergy;
    mixtureInput_.gasParams[(i - 1) + nSpecies_ * GasParams::SPECIES_DEGENERACY] = levelDegeneracy;
    tpsP_->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(), mixtureInput_.molarCV[i - 1]);
    tpsP_->getRequiredInput((basepath + "/initialMassFraction").c_str(), initialMassFraction_(i - 1));
  }

  /// Minimal amount of info for transport input struct
  std::vector<std::string> InputSpeciesNames;
  InputSpeciesNames.resize(nSpecies_);
  std::map<std::string, int> speciesMapping;
  Vector mixtureToInputMap(nSpecies_);
  int backgroundIndex;
  int paramIdx = 0;
  int targetIdx;

  tpsP_->getRequiredInput("atoms/numAtoms", nAtoms_);
  if (nSpecies_ > 1) {
    tpsP_->getRequiredInput("species/background_index", backgroundIndex);
    atomMW_.SetSize(nAtoms_);
    for (int a = 1; a <= nAtoms_; a++) {
      std::string basepath("atoms/atom" + std::to_string(a));
      std::string atomName;
      tpsP_->getRequiredInput((basepath + "/name").c_str(), atomName);
      tpsP_->getRequiredInput((basepath + "/mass").c_str(), atomMW_(a - 1));
      atomMap_[atomName] = a - 1;
      if (rank0_) {
        std::cout << " Atom: " << a << " is named " << atomName << " and has mass " << atomMW_(a - 1) << endl;
      }
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
  tpsP_->getInput("plasma_models/transport_model/argon_minimal/third_order_thermal_conductivity",
                  argonInput_.thirdOrderkElectron, true);
  if (!(argonInput_.thirdOrderkElectron) && rank0_)
    std::cout << "Notice: Using 1st order electron thermal conductivity." << endl;
  tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/enabled", argonInput_.multiply, false);
  if (argonInput_.multiply) {
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/viscosity",
                    argonInput_.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity",
                    argonInput_.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",
                    argonInput_.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity",
                    argonInput_.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
    tpsP_->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency",
                    argonInput_.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
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

    // valgrind is mad at this line...
    tpsP_->getRequiredPairs((basepath + "/composition").c_str(), composition);

    for (size_t c = 0; c < composition.size(); c++) {
      if (atomMap_.count(composition[c].first)) {
        int atomIdx = atomMap_[composition[c].first];
        inputSpeciesComposition(i - 1, atomIdx) = stoi(composition[c].second);
      } else {
        grvy_printf(GRVY_ERROR, "Requested atom %s for species %s is not available!\n", composition[c].first.c_str(),
                    speciesName.c_str());
      }
    }
  }

  Vector speciesCharge(nSpecies_);
  for (int sp = 0; sp < nSpecies_; sp++) {
    if (speciesNames_[sp] == "Ar.+1") {
      speciesCharge[sp] = +1.0;
    } else if (speciesNames_[sp] == "E") {
      speciesCharge[sp] = -1.0;
    } else {
      speciesCharge[sp] = 0.0;
    }
  }
  gasParams_.SetCol(GasParams::SPECIES_CHARGES, speciesCharge);

  for (int sp = 0; sp < nSpecies_; sp++) {
    mixtureInput_.gasParams[sp + nSpecies_ * GasParams::SPECIES_CHARGES] = speciesCharge[sp];
  }

  for (int sp = 0; sp < nSpecies_; sp++) {
    int inputIdx = mixtureToInputMap[sp];
    for (int a = 0; a < nAtoms_; a++) {
      speciesComposition_(sp, a) = inputSpeciesComposition(inputIdx, a);
    }
  }

  Vector speciesMass(nSpecies_);
  speciesMass = 0.0;
  for (int sp = 0; sp < nSpecies_; sp++) {
    for (int a = 0; a < nAtoms_; a++) {
      speciesMass[sp] += speciesComposition_(sp, a) * atomMW_[a];
    }
  }
  gasParams_.SetCol(GasParams::SPECIES_MW, speciesMass);

  for (int sp = 0; sp < nSpecies_; sp++) {
    mixtureInput_.gasParams[sp + nSpecies_ * GasParams::SPECIES_MW] = speciesMass[sp];
  }

  // for convience
  Rgas_ = UNIVERSALGASCONSTANT;

  for (int sp = 0; sp < nSpecies_; sp++) {
    specificHeatRatios_[sp] = (speciesMolarCv_[sp] + 1.0) / speciesMolarCv_[sp];
    speciesMolarCp_[sp] = specificHeatRatios_[sp] * speciesMolarCv_[sp];
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

  mixture_ = new PerfectMixture(mixtureInput_, dim_, dim_, const_plasma_conductivity_);
  transport_ = new ArgonMixtureTransport(mixture_, argonInput_);

  /// Minimal amount of info for chemistry input struct
  chemistryInput_.model = chemistryModel_;
  tpsP_->getInput("reactions/number_of_reactions", nReactions_, 0);
  tpsP_->getInput("reactions/minimum_chemistry_temperature", chemistryInput_.minimumTemperature, 0.0);

  Vector reactionEnergies(nReactions_);
  Vector detailedBalance(nReactions_);
  Array<ReactionModel> reactionModels;
  reactionModels.SetSize(nReactions_);
  Vector equilibriumConstantParams(nSpecies_ * nReactions_);
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

    } else if (model == "tabulated") {
      reactionModels[r - 1] = TABULATED_RXN;
      std::string inputPath(basepath + "/tabulated");
      readTableWrapper(inputPath, chemistryInput_.reactionInputs[r - 1].tableInput);

    } else if (model == "radiative_decay") {
      reactionModels[r - 1] = RADIATIVE_DECAY;
      double R;
      tpsP_->getRequiredInput((basepath + "/radius").c_str(), R);
      rxnModelParamsHost.push_back(Vector({R}));

    } else if (model == "bte") {
      reactionModels[r - 1] = GRIDFUNCTION_RXN;
      int index;
      tpsP_->getRequiredInput((basepath + "/bte/index").c_str(), index);
      chemistryInput_.reactionInputs[r - 1].indexInput = index;
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
      chemistryInput_.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS] =
          equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS];
    }

    if (reactionModels[r] != TABULATED_RXN && reactionModels[r] != GRIDFUNCTION_RXN) {
      assert(rxn_param_idx < rxnModelParamsHost.size());
      chemistryInput_.reactionInputs[r].modelParams = rxnModelParamsHost[rxn_param_idx].Read();
      rxn_param_idx += 1;
    }
  }

  // radiative decay needs this information as well
  chemistryInput_.speciesNames.resize(nSpecies_);
  paramIdx = 0;
  for (int sp = 0; sp < nSpecies_; sp++) {
    if (sp == backgroundIndex - 1) {
      targetIdx = nSpecies_ - 1;
    } else if (InputSpeciesNames[sp] == "E") {
      targetIdx = nSpecies_ - 2;
    } else {
      targetIdx = paramIdx;
      paramIdx++;
    }
    chemistryInput_.speciesMapping[InputSpeciesNames[sp]] = targetIdx;
    chemistryInput_.speciesNames[targetIdx] = InputSpeciesNames[sp];
  }

  chemistry_ = new Chemistry(mixture_, chemistryInput_);

  // Initialize radiation (just NEC for now) model
  std::string type;
  std::string basepath("plasma_models/radiation_model");

  tpsP_->getInput(basepath.c_str(), type, std::string("none"));
  std::string model_input_path(basepath + "/" + type);

  RadiationInput rad_model_input;
  if (type == "net_emission") {
    rad_model_input.model = NET_EMISSION;

    std::string coefficient_type;
    tpsP_->getRequiredInput((model_input_path + "/coefficient").c_str(), coefficient_type);
    if (coefficient_type == "tabulated") {
      rad_model_input.necModel = TABULATED_NEC;
      std::string table_input_path(model_input_path + "/tabulated");
      std::string filename;
      tpsP_->getRequiredInput((table_input_path + "/filename").c_str(), filename);

      std::vector<TableInput> rad_tables(1);
      for (size_t i = 0; i < rad_tables.size(); i++) {
        rad_tables[i].order = 1;
        rad_tables[i].xLogScale = false;
        rad_tables[i].fLogScale = false;
      }

      // Read data from hdf5 file containing 4 columns: T, energy, gas constant, speed of sound
      DenseMatrix rad_data;
      bool success =
          h5ReadBcastMultiColumnTable(filename, std::string("table"), pmesh_->GetComm(), rad_data, rad_tables);
      if (!success) exit(ERROR);

      rad_model_input.necTableInput.order = 1;
      rad_model_input.necTableInput.xLogScale = false;
      rad_model_input.necTableInput.fLogScale = false;
      rad_model_input.necTableInput.Ndata = rad_tables[0].Ndata;
      rad_model_input.necTableInput.xdata = rad_tables[0].xdata;
      rad_model_input.necTableInput.fdata = rad_tables[0].fdata;

      radiation_ = new NetEmission(rad_model_input);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown net emission coefficient type -> %s\n", coefficient_type.c_str());
      exit(ERROR);
    }
  } else if (type == "none") {
    radiation_ = nullptr;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown radiation model -> %s\n", type.c_str());
    exit(ERROR);
  }

  tpsP_->getInput("loMach/reactingFlow/ic", ic_string_, std::string(""));

  // will be over-written if dynamic is active
  tpsP_->getInput("loMach/reactingFlow/sub-steps", nSub_, 1);
  tpsP_->getInput("loMach/reactingFlow/dynamic-substep", dynamic_substepping_, false);
  if (dynamic_substepping_) nSub_ = 2;

  // default value is purely empirical atm
  tpsP_->getInput("loMach/reactingFlow/dynamic-fraction", stabFrac_, 100);

  // Check time marching order.  Operator split (i.e., nSub_ > 1) not supported for order > 1.
  if ((nSub_ > 1) && (time_coeff_.order > 1)) {
    if (rank0_) {
      std::cout << "ERROR: BDF order > 1 not supported with operator split." << std::endl;
      std::cout << "       Either set loMach/reactingFlow/sub-steps = 1 or time/bdfOrder = 1" << std::endl;
    }
    assert(false);
    exit(1);
  }

  // By default, do not use operator split scheme.
  operator_split_ = false;
  if (nSub_ > 1) {
    operator_split_ = true;
  }
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
  delete MsRhoCp_form_;
  delete Ms_form_;
  delete At_form_;
  delete G_form_;
  delete jh_form_;

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
  delete rhoCp_over_dt_coeff_;
  // delete rho_coeff_;
  delete cpMix_coeff_;
  delete rhoCp_coeff_;

  delete jh_coeff_;
  delete radiation_sink_coeff_;

  delete rad_rho_coeff_;
  delete rad_rho_Cp_coeff_;
  delete rad_rho_u_coeff_;
  delete rad_rho_Cp_u_coeff_;
  delete rad_rho_over_dt_coeff_;
  delete rad_species_diff_total_coeff_;
  delete rad_rho_Cp_over_dt_coeff_;
  delete rad_thermal_diff_total_coeff_;
  delete rad_jh_coeff_;
  delete rad_kap_gradT_coeff_;

  delete Ay_form_;
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
  delete vfes_;
  delete vfec_;
  delete sfes_;
  delete sfec_;
  delete yfes_;
  delete yfec_;
}

void ReactingFlow::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing ReactingFlow solver.\n");

  nActiveSpecies_ = mixture_->GetNumActiveSpecies();

  // TODO(trevilo): These shouldn't be hardcoded.
  invPr_ = 1 / Pr_;
  invSc_ = 1 / Sc_;

  //-----------------------------------------------------
  // 1) Prepare the required finite element objects
  //-----------------------------------------------------
  sfec_ = new H1_FECollection(order_, dim_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  yfec_ = new H1_FECollection(order_, dim_);
  yfes_ = new ParFiniteElementSpace(pmesh_, yfec_, nSpecies_);

  vfec_ = new H1_FECollection(order_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  // Check if fully periodic mesh
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    temp_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    temp_ess_attr_ = 0;

    Qt_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    Qt_ess_attr_ = 0;

    spec_ess_attr_.SetSize(pmesh_->bdr_attributes.Max());
    spec_ess_attr_ = 0;
  }

  sDof_ = sfes_->GetVSize();
  yDof_ = yfes_->GetVSize();
  sDofInt_ = sfes_->GetTrueVSize();
  yDofInt_ = yfes_->GetTrueVSize();

  weff_gf_.SetSpace(vfes_);
  weff_gf_ = 0.0;

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

  NYn_.SetSize(yDofInt_);
  NYnm1_.SetSize(yDofInt_);
  NYnm2_.SetSize(yDofInt_);
  NYn_ = 0.0;
  NYnm1_ = 0.0;
  NYnm2_ = 0.0;

  // for time-splitting
  spec_buffer_.SetSize(yDofInt_);
  temp_buffer_.SetSize(sDofInt_);
  YnStar_.SetSize(yDofInt_);
  TnStar_.SetSize(sDofInt_);

  // number density
  Xn_.SetSize(yDofInt_);
  Xn_ = 0.0;

  resY_.SetSize(sDofInt_);
  resY_ = 0.0;

  hw_.SetSize(sDofInt_);
  hw_ = 0.0;

  // only YnFull for plotting
  YnFull_gf_.SetSpace(yfes_);
  YnFull_gf_ = 0.0;

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

  prodY_gf_.SetSpace(sfes_);
  prodY_gf_ = 0.0;

  prodE_.SetSize(yDofInt_);
  prodE_ = 1.0e-12;

  emission_gf_.SetSpace(sfes_);
  emission_gf_ = 0.0;

  sigma_gf_.SetSpace(sfes_);
  sigma_gf_ = 0.0;

  sigma_.SetSize(sDofInt_);
  sigma_ = 0.0;

  jh_gf_.SetSpace(sfes_);
  jh_gf_ = 0.0;

  jh_.SetSize(sDofInt_);
  jh_ = 0.0;

  radiation_sink_gf_.SetSpace(sfes_);
  radiation_sink_gf_ = 0.0;

  radiation_sink_.SetSize(sDofInt_);
  radiation_sink_ = 0.0;

  tmpR1_.SetSize(dim_ * sDofInt_);
  tmpR1a_.SetSize(dim_ * sDofInt_);
  tmpR1b_.SetSize(dim_ * sDofInt_);
  tmpR1c_.SetSize(dim_ * sDofInt_);

  tmpR0_.SetSize(sDofInt_);
  tmpR0a_.SetSize(sDofInt_);
  tmpR0b_.SetSize(sDofInt_);
  tmpR0c_.SetSize(sDofInt_);

  R0PM0_gf_.SetSpace(sfes_);

  rhoDt_gf_.SetSpace(sfes_);

  // this is only needed to form the op-coeff
  CpY_gf_.SetSpace(sfes_);
  CpY_gf_ = 1000.6;
  CpY_.SetSize(yDofInt_);
  CpY_ = 1000.6;

  CpMix_gf_.SetSpace(sfes_);
  CpMix_gf_ = 1000.6;

  crossDiff_.SetSize(sDofInt_);
  crossDiff_ = 0.0;

  Mmix_gf_.SetSpace(sfes_);
  Rmix_gf_.SetSpace(sfes_);

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &visc_gf_;
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

  // TODO(trevilo): Move into cases.{hpp,cpp}
  if (!ic_string_.empty()) {
    if (ic_string_ == "step") {
      if (rank0_) std::cout << "Setting stepwise initial Yn condition." << endl;

      FunctionCoefficient y0_excoeff(species_stepLeft);
      y0_excoeff.SetTime(0.0);
      Yn_gf_.ProjectCoefficient(y0_excoeff);
      Yn_gf_.GetTrueDofs(tmpR0_);
      setVectorFromScalar(tmpR0_, 0, &Yn_);

      FunctionCoefficient y1_excoeff(species_uniform);
      y1_excoeff.SetTime(0.0);
      Yn_gf_.ProjectCoefficient(y1_excoeff);
      Yn_gf_.GetTrueDofs(tmpR0_);
      setVectorFromScalar(tmpR0_, 1, &Yn_);

      FunctionCoefficient y2_excoeff(species_stepRight);
      y2_excoeff.SetTime(0.0);
      Yn_gf_.ProjectCoefficient(y2_excoeff);
      Yn_gf_.GetTrueDofs(tmpR0_);
      setVectorFromScalar(tmpR0_, 2, &Yn_);

    } else if (ic_string_ == "binaryTest") {
      if (rank0_) std::cout << "Setting binary diffusion test Yn condition." << endl;

      FunctionCoefficient y0_excoeff(binaryTest);
      y0_excoeff.SetTime(0.0);
      Yn_gf_.ProjectCoefficient(y0_excoeff);
      Yn_gf_.GetTrueDofs(tmpR0_);
      setVectorFromScalar(tmpR0_, 0, &Yn_);

      for (int i = 0; i < sDofInt_; i++) {
        tmpR0a_[i] = 0.0;
      }
      setVectorFromScalar(tmpR0a_, 1, &Yn_);

      for (int i = 0; i < sDofInt_; i++) {
        tmpR0a_[i] = 1.0 - tmpR0_[i];
      }
      setVectorFromScalar(tmpR0a_, 2, &Yn_);

    } else {
      if (rank0_) std::cout << "Unknown reactingFlow ic name: " << ic_string_ << ". Exiting." << endl;
      exit(1);
    }

  } else {
    if (rank0_) std::cout << "Setting uniform initial Yn condition from input file." << endl;

    ConstantCoefficient Yn_ic_coef;
    for (int sp = 0; sp < nSpecies_; sp++) {
      Yn_ic_coef.constant = initialMassFraction_(sp);
      Yn_gf_.ProjectCoefficient(Yn_ic_coef);
      Yn_gf_.GetTrueDofs(tmpR0_);
      setVectorFromScalar(tmpR0_, sp, &Yn_);
    }
  }
  Ynm1_ = Yn_;
  Ynm2_ = Yn_;
  Yn_next_gf_ = Yn_gf_;
  YnFull_gf_.SetFromTrueDofs(Yn_);

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
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
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
          std::cout << "Rx Flow: Setting uniform Dirichlet temperature on patch = " << patch << std::endl;
        }
        AddTempDirichletBC(temperature_value, inlet_attr);

      } else if (type == "interpolate") {
        Array<int> inlet_attr(pmesh_->bdr_attributes.Max());
        inlet_attr = 0;
        inlet_attr[patch - 1] = 1;
        temperature_bc_field_ = new GridFunctionCoefficient(extData_interface_->Tdata);
        if (rank0_) {
          std::cout << "Rx Flow: Setting interpolated Dirichlet temperature on patch = " << patch << std::endl;
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
          std::cout << "ERROR: Rx Flow inlet type = " << type << " not supported." << std::endl;
        }
        assert(false);
        exit(1);
      }
    }
    if (rank0_) {
      std::cout << "inlet bc set..." << endl;
    }
  }

  // outlet bc
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    Array<int> attr_outlet(pmesh_->bdr_attributes.Max());
    attr_outlet = 0;
    // Outlet is homogeneous Neumann on species and T, so nothing more to do here
  }

  // Wall BCs
  if (!(pmesh_->bdr_attributes.Size() == 0)) {
    if (rank0_) {
      std::cout << "There are " << pmesh_->bdr_attributes.Max() << " boundary attributes" << std::endl;
    }
    Array<int> attr_wall(pmesh_->bdr_attributes.Max());
    attr_wall = 0;

    for (int i = 1; i <= numWalls; i++) {
      int patch;
      std::string type;
      std::string basepath("boundaryConditions/wall" + std::to_string(i));

      tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
      tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

      if (type == "viscous_isothermal") {
        if (rank0_) {
          std::cout << "Adding patch = " << patch << " to isothermal wall list!" << std::endl;
        }

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

  // with ic set, update Rmix
  updateMixture();
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

  // coefficients for operators
  cpMix_coeff_ = new GridFunctionCoefficient(&CpMix_gf_);
  rhon_next_coeff_ = new GridFunctionCoefficient(&rn_gf_);
  species_Cp_coeff_ = new GridFunctionCoefficient(&CpY_gf_);
  mut_coeff_ = new GridFunctionCoefficient(turbModel_interface_->eddy_viscosity);
  mult_coeff_ = new GridFunctionCoefficient(sponge_interface_->diff_multiplier);
  un_next_coeff_ = new VectorGridFunctionCoefficient(flow_interface_->velocity);

  // for unsteady terms
  rhoDt_gf_ = rn_gf_;
  rhoDt_gf_ /= dt_;
  rho_over_dt_coeff_ = new GridFunctionCoefficient(&rhoDt_gf_);
  rhoCp_over_dt_coeff_ = new ProductCoefficient(*cpMix_coeff_, *rho_over_dt_coeff_);

  // thermal_diff_coeff.constant = thermal_diff;
  thermal_diff_coeff_ = new GridFunctionCoefficient(&kappa_gf_);
  thermal_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *thermal_diff_coeff_, invPr_, 1.0);
  thermal_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *thermal_diff_sum_coeff_);
  gradT_coeff_ = new GradientGridFunctionCoefficient(&Tn_next_gf_);
  kap_gradT_coeff_ = new ScalarVectorProductCoefficient(*thermal_diff_total_coeff_, *gradT_coeff_);

  // species diff coeff, diffY_gf gets filled with correct species diffusivity before its solve
  species_diff_coeff_ = new GridFunctionCoefficient(&diffY_gf_);
  species_diff_sum_coeff_ = new SumCoefficient(*mut_coeff_, *species_diff_coeff_, invSc_, 1.0);  // is this correct?
  species_diff_total_coeff_ = new ProductCoefficient(*mult_coeff_, *species_diff_sum_coeff_);
  species_diff_Cp_coeff_ = new ProductCoefficient(*species_Cp_coeff_, *species_diff_total_coeff_);

  // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho u \cdot \nabla \phi_j
  rhoCp_coeff_ = new ProductCoefficient(*cpMix_coeff_, *rhon_next_coeff_);
  rhouCp_coeff_ = new ScalarVectorProductCoefficient(*rhoCp_coeff_, *un_next_coeff_);
  rhou_coeff_ = new ScalarVectorProductCoefficient(*rhon_next_coeff_, *un_next_coeff_);

  // Joule heating
  jh_coeff_ = new GridFunctionCoefficient(&jh_gf_);

  // Radiation energy sink
  radiation_sink_coeff_ = new GridFunctionCoefficient(&radiation_sink_gf_);

  if (axisym_) {
    // for axisymmetric case, need to multply many coefficients by the radius
    rad_rho_coeff_ = new ProductCoefficient(radius_coeff, *rhon_next_coeff_);
    rad_rho_Cp_coeff_ = new ProductCoefficient(radius_coeff, *rhoCp_coeff_);
    rad_rho_u_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *rhou_coeff_);
    rad_rho_Cp_u_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *rhouCp_coeff_);
    rad_rho_over_dt_coeff_ = new ProductCoefficient(radius_coeff, *rho_over_dt_coeff_);
    rad_species_diff_total_coeff_ = new ProductCoefficient(radius_coeff, *species_diff_total_coeff_);
    rad_rho_Cp_over_dt_coeff_ = new ProductCoefficient(radius_coeff, *rhoCp_over_dt_coeff_);
    rad_thermal_diff_total_coeff_ = new ProductCoefficient(radius_coeff, *thermal_diff_total_coeff_);
    rad_jh_coeff_ = new ProductCoefficient(radius_coeff, *jh_coeff_);
    rad_radiation_sink_coeff_ = new ProductCoefficient(radius_coeff, *radiation_sink_coeff_);
    rad_kap_gradT_coeff_ = new ScalarVectorProductCoefficient(radius_coeff, *kap_gradT_coeff_);
  }

  At_form_ = new ParBilinearForm(sfes_);
  ConvectionIntegrator *at_blfi;
  if (axisym_) {
    at_blfi = new ConvectionIntegrator(*rad_rho_Cp_u_coeff_);
  } else {
    at_blfi = new ConvectionIntegrator(*rhouCp_coeff_);
  }
  if (numerical_integ_) {
    at_blfi->SetIntRule(&ir_nli);
  }
  At_form_->AddDomainIntegrator(at_blfi);
  if (partial_assembly_) {
    At_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  At_form_->Assemble();
  At_form_->FormSystemMatrix(empty, At_);

  Ay_form_ = new ParBilinearForm(sfes_);
  ConvectionIntegrator *ay_blfi;
  if (axisym_) {
    ay_blfi = new ConvectionIntegrator(*rad_rho_u_coeff_);
  } else {
    ay_blfi = new ConvectionIntegrator(*rhou_coeff_);
  }
  if (numerical_integ_) {
    ay_blfi->SetIntRule(&ir_nli);
  }
  Ay_form_->AddDomainIntegrator(ay_blfi);
  if (partial_assembly_) {
    Ay_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ay_form_->Assemble();
  Ay_form_->FormSystemMatrix(empty, Ay_);

  // mass matrix
  Ms_form_ = new ParBilinearForm(sfes_);
  MassIntegrator *ms_blfi;
  if (axisym_) {
    ms_blfi = new MassIntegrator(radius_coeff);
  } else {
    ms_blfi = new MassIntegrator;
  }
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
  MassIntegrator *msrho_blfi;
  if (axisym_) {
    msrho_blfi = new MassIntegrator(*rad_rho_coeff_);
  } else {
    msrho_blfi = new MassIntegrator(*rhon_next_coeff_);
  }
  if (numerical_integ_) {
    msrho_blfi->SetIntRule(&ir_i);
  }
  MsRho_form_->AddDomainIntegrator(msrho_blfi);
  if (partial_assembly_) {
    MsRho_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);

  // mass matrix with rho and Cp
  MsRhoCp_form_ = new ParBilinearForm(sfes_);
  MassIntegrator *msrhocp_blfi;
  if (axisym_) {
    msrhocp_blfi = new MassIntegrator(*rad_rho_Cp_coeff_);
  } else {
    msrhocp_blfi = new MassIntegrator(*rhoCp_coeff_);
  }
  if (numerical_integ_) {
    msrhocp_blfi->SetIntRule(&ir_i);
  }
  MsRhoCp_form_->AddDomainIntegrator(msrhocp_blfi);
  if (partial_assembly_) {
    MsRhoCp_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MsRhoCp_form_->Assemble();
  MsRhoCp_form_->FormSystemMatrix(empty, MsRhoCp_);

  // temperature Helmholtz
  Ht_form_ = new ParBilinearForm(sfes_);
  MassIntegrator *hmt_blfi;
  if (axisym_) {
    hmt_blfi = new MassIntegrator(*rad_rho_Cp_over_dt_coeff_);
  } else {
    hmt_blfi = new MassIntegrator(*rhoCp_over_dt_coeff_);
  }
  DiffusionIntegrator *hdt_blfi;
  if (axisym_) {
    hdt_blfi = new DiffusionIntegrator(*rad_thermal_diff_total_coeff_);
  } else {
    hdt_blfi = new DiffusionIntegrator(*thermal_diff_total_coeff_);
  }
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

  // species Helmholtz
  Hy_form_ = new ParBilinearForm(sfes_);
  MassIntegrator *hmy_blfi;
  if (axisym_) {
    hmy_blfi = new MassIntegrator(*rad_rho_over_dt_coeff_);
  } else {
    hmy_blfi = new MassIntegrator(*rho_over_dt_coeff_);
  }
  DiffusionIntegrator *hdy_blfi;
  if (axisym_) {
    hdy_blfi = new DiffusionIntegrator(*rad_species_diff_total_coeff_);
  } else {
    hdy_blfi = new DiffusionIntegrator(*species_diff_total_coeff_);
  }
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

  // Vector space mass matrix (used in gradient calcs)
  Mv_form_ = new ParBilinearForm(vfes_);
  VectorMassIntegrator *mv_blfi;
  if (axisym_) {
    mv_blfi = new VectorMassIntegrator(radius_coeff);
  } else {
    mv_blfi = new VectorMassIntegrator;
  }
  if (numerical_integ_) {
    mv_blfi->SetIntRule(&ir_i);
  }
  Mv_form_->AddDomainIntegrator(mv_blfi);
  if (partial_assembly_) {
    Mv_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mv_form_->Assemble();
  Mv_form_->FormSystemMatrix(empty, Mv_op_);

  // Inverse (unweighted) mass operator (velocity space)
  if (partial_assembly_) {
    Vector diag_pa(vfes_->GetTrueVSize());
    Mv_form_->AssembleDiagonal(diag_pa);
    Mv_inv_pc_ = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    Mv_inv_pc_ = new HypreSmoother(*Mv_op_.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(Mv_inv_pc_)->SetType(HypreSmoother::Jacobi, 1);
  }
  Mv_inv_ = new CGSolver(vfes_->GetComm());
  Mv_inv_->iterative_mode = false;
  Mv_inv_->SetOperator(*Mv_op_);
  Mv_inv_->SetPreconditioner(*Mv_inv_pc_);
  Mv_inv_->SetPrintLevel(pl_solve_);
  Mv_inv_->SetRelTol(rtol_);
  Mv_inv_->SetAbsTol(1e-18);
  Mv_inv_->SetMaxIter(max_iter_);

  // Energy sink/source terms: Joule heating and radiation sink
  jh_form_ = new ParLinearForm(sfes_);
  DomainLFIntegrator *jh_dlfi;
  DomainLFIntegrator *rad_dlfi;
  if (axisym_) {
    jh_dlfi = new DomainLFIntegrator(*rad_jh_coeff_);
    rad_dlfi = new DomainLFIntegrator(*rad_radiation_sink_coeff_);
  } else {
    jh_dlfi = new DomainLFIntegrator(*jh_coeff_);
    rad_dlfi = new DomainLFIntegrator(*radiation_sink_coeff_);
  }
  if (numerical_integ_) {
    jh_dlfi->SetIntRule(&ir_i);
    rad_dlfi->SetIntRule(&ir_i);
  }
  jh_form_->AddDomainIntegrator(jh_dlfi);
  jh_form_->AddDomainIntegrator(rad_dlfi);

  // Qt .....................................
  Mq_form_ = new ParBilinearForm(sfes_);
  MassIntegrator *mq_blfi;
  if (axisym_) {
    mq_blfi = new MassIntegrator(radius_coeff);
  } else {
    mq_blfi = new MassIntegrator;
  }
  if (numerical_integ_) {
    mq_blfi->SetIntRule(&ir_i);
  }
  Mq_form_->AddDomainIntegrator(mq_blfi);
  if (partial_assembly_) {
    Mq_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mq_form_->Assemble();
  Mq_form_->FormSystemMatrix(Qt_ess_tdof_, Mq_);

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
  DiffusionIntegrator *lqd_blfi;
  if (axisym_) {
    lqd_blfi = new DiffusionIntegrator(*rad_thermal_diff_total_coeff_);
  } else {
    lqd_blfi = new DiffusionIntegrator(*thermal_diff_total_coeff_);
  }
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
  BoundaryNormalLFIntegrator *lq_bdry_lfi;
  if (axisym_) {
    lq_bdry_lfi = new BoundaryNormalLFIntegrator(*rad_kap_gradT_coeff_, 2, -1);
  } else {
    lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff_, 2, -1);
  }
  if (numerical_integ_) {
    lq_bdry_lfi->SetIntRule(&ir_di);
  }
  LQ_bdry_->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr_);

  // for explicit species-diff source term in energy (inv not needed)
  // TODO(trevilo): This operator is not used.  Can we eliminate it?
  LY_form_ = new ParBilinearForm(sfes_);
  auto *lyd_blfi = new DiffusionIntegrator(*species_diff_Cp_coeff_);
  if (numerical_integ_) {
    lyd_blfi->SetIntRule(&ir_di);
  }
  LY_form_->AddDomainIntegrator(lyd_blfi);
  if (partial_assembly_) {
    LY_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  LY_form_->Assemble();
  LY_form_->FormSystemMatrix(empty, LY_);

  // gradient of scalar
  G_form_ = new ParMixedBilinearForm(sfes_, vfes_);
  GradientIntegrator *g_mblfi;
  if (axisym_) {
    g_mblfi = new GradientIntegrator(radius_coeff);
  } else {
    g_mblfi = new GradientIntegrator();
  }
  if (numerical_integ_) {
    g_mblfi->SetIntRule(&ir_i);
  }
  G_form_->AddDomainIntegrator(g_mblfi);
  if (partial_assembly_) {
    G_form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  G_form_->Assemble();
  G_form_->FormRectangularSystemMatrix(empty, empty, G_);

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

  // If we are restarting from an LTE field, compute the species mass
  // fractions from the temperature and pressure
  bool restart_from_lte;
  tpsP_->getInput("io/restartFromLTE", restart_from_lte, false);
  if (restart_from_lte) {
    Vector n_sp(nSpecies_);
    Vector rho_sp(nSpecies_);
    const double *h_T = Tn_.HostRead();
    double *h_Yn = Yn_.HostWrite();
    for (int i = 0; i < sDofInt_; i++) {
      const double Ti = h_T[i];

      // Evaluate the mole densities of each species at this point,
      // assuming LTE at given temperature and pressure
      mixture_->GetSpeciesFromLTE(Ti, thermo_pressure_, n_sp.HostWrite());

      // From the mole densities, evaluate the mass densities and mixture density
      double mixture_density = 0.0;
      for (int sp = 0; sp < nSpecies_; sp++) {
        rho_sp[sp] = n_sp[sp] * mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
        mixture_density += rho_sp[sp];
      }

      // Finally, evaluate mass fraction
      for (int sp = 0; sp < nSpecies_; sp++) {
        h_Yn[i + sp * sDofInt_] = rho_sp[sp] / mixture_density;
      }
    }
    Ynm1_ = Yn_;
    Ynm2_ = Yn_;
    YnFull_gf_.SetFromTrueDofs(Yn_);
  }

  // Ensure Yn_ is consistent with YnFull_gf_.  Specifically this is
  // necessary on standard restart, when the solution is read into
  // YnFull_gf_ after the Yn_ IC is set.
  YnFull_gf_.GetTrueDofs(Yn_);

  // and initialize system mass
  updateMixture();
  updateDensity(0.0);

  computeSystemMass();

  // for initial plot
  updateDiffusivity();
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

  // species
  NYnm2_ = NYnm1_;
  NYnm1_ = NYn_;

  Ynm2_ = Ynm1_;
  Ynm1_ = Yn_;
  Yn_ = Yn_next_;
}

void ReactingFlow::step() {
  dt_ = time_coeff_.dt;
  time_ = time_coeff_.time;

  // Set current time for velocity Dirichlet boundary conditions.
  for (auto &temp_dbc : temp_dbcs_) {
    temp_dbc.coeff->SetTime(time_ + dt_);
  }

  // Prepare for residual calc
  updateMixture();
  extrapolateState();
  updateBC(0);  // NB: can't ramp right now

  if (!operator_split_) {
    updateThermoP();
  }

  updateDensity(1.0);
  updateDiffusivity();

  // PART I: Form and solve implicit systems for species and temperature.
  //
  // If NOT using operator splitting, these systems include all terms,
  // with the reaction source terms treated explicitly in time.  When
  // using operator splitting, this solve includes the
  // advection-diffusion terms but the reaction source terms are
  // handled separately with substepping below.

  if (!operator_split_) {
    speciesProduction();
  }

  // advance species, last slot is from calculated sum of others
  for (int iSpecies = 0; iSpecies < nActiveSpecies_; iSpecies++) {
    speciesStep(iSpecies);
    // Yn_next_ = Yn_;
  }
  if (mixtureInput_.ambipolar) {
    // Evaluate electron mass fraction based on quasi-neutrality

    // temporary storage for electron mass fraction
    tmpR0_ = 0.0;

    // Y_electron = sum_{i \in active species} (m_electron / m_i) * q_i * Y_i
    for (int iSpecies = 0; iSpecies < nActiveSpecies_; iSpecies++) {
      setScalarFromVector(Yn_next_, iSpecies, &tmpR0a_);
      const double q_sp = mixture_->GetGasParams(iSpecies, GasParams::SPECIES_CHARGES);
      const double m_sp = mixture_->GetGasParams(iSpecies, GasParams::SPECIES_MW);
      const double fac = q_sp / m_sp;
      tmpR0a_ *= fac;
      tmpR0_ += tmpR0a_;
    }
    const int iElectron = nSpecies_ - 2;  // TODO(trevilo): check me!
    const double m_electron = mixture_->GetGasParams(iElectron, GasParams::SPECIES_MW);
    tmpR0_ *= m_electron;
    setVectorFromScalar(tmpR0_, iElectron, &Yn_next_);
  }
  speciesLastStep();
  YnFull_gf_.SetFromTrueDofs(Yn_next_);

  // ho_i*w_i
  if (!operator_split_) {
    heatOfFormation();
  }

  crossDiffusion();

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

  // explicit filter species
  // Add if found to be necessary

  if (operator_split_) {
    /// PART II: time-splitting of reaction

    // Save Yn_ and Tn_ because substep routines overwrite these as
    // with the {n}+iSub substates, which is convenient b/c helper
    // functions (like speciesProduction) use these Vectors
    spec_buffer_.Set(1.0, Yn_);
    temp_buffer_.Set(1.0, Tn_);

    // Evaluate (Yn_next_ - Yn_)/nSub and store in YnStar_, and analog
    // for TnStar_.
    substepState();

    // number of substeps
    if (dynamic_substepping_) evalSubstepNumber();

    for (int iSub = 0; iSub < nSub_; iSub++) {
      // update wdot quantities at full substep in Yn/Tn state
      updateMixture();
      updateThermoP();
      updateDensity(0.0, false);
      speciesProduction();
      heatOfFormation();

      // advance over substep
      for (int iSpecies = 0; iSpecies < nActiveSpecies_; iSpecies++) {
        speciesSubstep(iSpecies, iSub);
      }
      if (mixtureInput_.ambipolar) {
        // Evaluate electron mass fraction based on quasi-neutrality

        // temporary storage for electron mass fraction
        tmpR0_ = 0.0;

        // Y_electron = sum_{i \in active species} (m_electron / m_i) * q_i * Y_i
        for (int iSpecies = 0; iSpecies < nActiveSpecies_; iSpecies++) {
          setScalarFromVector(Yn_, iSpecies, &tmpR0a_);
          const double q_sp = mixture_->GetGasParams(iSpecies, GasParams::SPECIES_CHARGES);
          const double m_sp = mixture_->GetGasParams(iSpecies, GasParams::SPECIES_MW);
          const double fac = q_sp / m_sp;
          tmpR0a_ *= fac;
          tmpR0_ += tmpR0a_;
        }
        const int iElectron = nSpecies_ - 2;  // TODO(trevilo): check me!
        const double m_electron = mixture_->GetGasParams(iElectron, GasParams::SPECIES_MW);
        tmpR0_ *= m_electron;
        setVectorFromScalar(tmpR0_, iElectron, &Yn_);
      }
      speciesLastSubstep();
      Yn_gf_.SetFromTrueDofs(Yn_);

      temperatureSubstep(iSub);
      Tn_gf_.SetFromTrueDofs(Tn_);
    }

    // set register to correct full step fields
    Yn_next_ = Yn_;
    Tn_next_ = Tn_;

    YnFull_gf_.SetFromTrueDofs(Yn_next_);
    Tn_next_gf_.SetFromTrueDofs(Tn_next_);

    Yn_ = spec_buffer_;
    Tn_ = temp_buffer_;
    Tn_gf_.SetFromTrueDofs(Tn_);
  }

  /// PART III: prepare for external use
  updateDensity(1.0);
  computeQtTO();

  UpdateTimestepHistory(dt_);

  updateMixture();
  updateDiffusivity();
}

void ReactingFlow::evalSubstepNumber() {
  double myMaxProd = 0.0;
  double maxProd = 0.0;
  double tmp, deltaYn;

  updateMixture();
  updateThermoP();
  updateDensity(0.0);
  speciesProduction();

  {
    const double *dataProd = prodY_.HostRead();
    const double *dataYn = Yn_.HostRead();
    for (int i = 0; i < sDofInt_; i++) {
      for (int sp = 0; sp < nSpecies_; sp++) {
        // basic based on max production
        // myMaxProd = std::max(std::abs(dataProd[i + sp * sDofInt_]), myMaxProd);

        // modified to amplify production if species could go negative or exceed unity
        tmp = dataYn[i + sp * sDofInt_] + dataProd[i + sp * sDofInt_] * time_coeff_.dt;
        if (tmp >= 1.0) {
          tmp = tmp - 1.0;
        } else if (tmp > 0.0) {
          tmp = 0.0;
        } else {
          tmp = std::abs(tmp);
        }
        tmp /= 0.5 * time_coeff_.dt;
        myMaxProd = std::max(std::abs(dataProd[i + sp * sDofInt_]) + tmp, myMaxProd);
      }
    }
  }
  MPI_Reduce(&myMaxProd, &maxProd, 1, MPI_DOUBLE, MPI_MAX, 0, tpsP_->getTPSCommWorld());
  deltaYn = maxProd * time_coeff_.dt;

  // want: dYsub < 1/100
  // dYsub = deltaYn/nSub
  // deltaYn/nSub < 1/100
  // nSub > deltaYn * 100
  nSub_ = stabFrac_ * std::ceil(deltaYn);
  nSub_ = std::max(nSub_, 2);
}

void ReactingFlow::temperatureStep() {
  Array<int> empty;

  // Update radiation sink
  if (radiation_ != nullptr) {
    const double *d_T = Tn_next_.Read();
    double *d_rad = radiation_sink_.Write();
    Radiation *rmodel = radiation_;
    MFEM_FORALL(i, Tn_next_.Size(), { d_rad[i] = rmodel->computeEnergySink(d_T[i]); });
  }
  radiation_sink_gf_.SetFromTrueDofs(radiation_sink_);

  // Build the right-hand-side
  resT_ = 0.0;

  // convection
  computeExplicitTempConvectionOP(true);  // ->tmpR0_
  resT_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  tmpR0_.Set(time_coeff_.bd1 / dt_, Tn_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, Tnm1_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, Tnm2_);

  MsRhoCp_form_->Update();
  MsRhoCp_form_->Assemble();
  MsRhoCp_form_->FormSystemMatrix(empty, MsRhoCp_);
  MsRhoCp_->AddMult(tmpR0_, resT_, -1.0);

  // If NOT using operator splitting, include the unsteady pressure
  // term and the heat of formation contribution on the rhs.  If using
  // operator splitting, these are handled in temperatureSubstep.
  if (!operator_split_) {
    // dPo/dt
    tmpR0_ = dtP_;  // with rho*Cp on LHS, no Cp on this term
    Ms_->AddMult(tmpR0_, resT_);

    // heat of formation
    Ms_->AddMult(hw_, resT_);
  }

  // Joule heating (and radiation sink)
  jh_form_->Update();
  jh_form_->Assemble();
  jh_form_->ParallelAssemble(jh_);
  resT_ += jh_;

  // species-temp diffusion term, already in int-weak form
  resT_.Add(1.0, crossDiff_);

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, Cp, and kappa
  // NOTE:  rhoDt is not used directly but is used in rhoCp_over_dt_coeff_
  rhoDt_gf_ = rn_gf_;
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);

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

void ReactingFlow::temperatureSubstep(int iSub) {
  // substep dt
  double dtSub = dt_ / (double)nSub_;

  CpMix_gf_.GetTrueDofs(CpMix_);

  // heat of formation term
  tmpR0_.Set(1.0, hw_);

  // pressure
  tmpR0_ += dtP_;

  // contribution to temperature update is dt * rhs / (rho * Cp)
  tmpR0_ /= rn_;
  tmpR0_ /= CpMix_;
  tmpR0_ *= dtSub;

  // TnStar has star state at substep here
  tmpR0_.Add(1.0, TnStar_);
  tmpR0_.Add(1.0, Tn_);

  // Tn now has full state at substep
  Tn_.Set(1.0, tmpR0_);
}

void ReactingFlow::speciesLastStep() {
  tmpR0_ = 0.0;
  double *dataYn = Yn_next_.HostReadWrite();
  double *dataYsum = tmpR0_.HostReadWrite();
  for (int n = 0; n < nSpecies_ - 1; n++) {
    for (int i = 0; i < sDofInt_; i++) {
      dataYsum[i] += dataYn[i + n * sDofInt_];
    }
  }
  for (int i = 0; i < sDofInt_; i++) {
    dataYn[i + (nSpecies_ - 1) * sDofInt_] = 1.0 - dataYsum[i];
  }
}

void ReactingFlow::speciesLastSubstep() {
  tmpR0_ = 0.0;
  double *dataYn = Yn_.HostReadWrite();
  double *dataYsum = tmpR0_.HostReadWrite();
  for (int n = 0; n < nSpecies_ - 1; n++) {
    for (int i = 0; i < sDofInt_; i++) {
      dataYsum[i] += dataYn[i + n * sDofInt_];
    }
  }
  for (int i = 0; i < sDofInt_; i++) {
    dataYn[i + (nSpecies_ - 1) * sDofInt_] = 1.0 - dataYsum[i];
  }
}

void ReactingFlow::speciesStep(int iSpec) {
  Array<int> empty;

  // copy relevant species properties from full Vector to particular case
  setScalarFromVector(diffY_, iSpec, &tmpR0_);

  diffY_gf_.SetFromTrueDofs(tmpR0_);

  // Build the right-hand-side
  resY_ = 0.0;

  // convection
  computeExplicitSpecConvectionOP(iSpec, true);  // ->tmpR0_
  resY_.Set(-1.0, tmpR0_);

  // for unsteady term, compute and add known part of BDF unsteady term
  setScalarFromVector(Yn_, iSpec, &tmpR0a_);
  setScalarFromVector(Ynm1_, iSpec, &tmpR0b_);
  setScalarFromVector(Ynm2_, iSpec, &tmpR0c_);
  tmpR0_.Set(time_coeff_.bd1 / dt_, tmpR0a_);
  tmpR0_.Add(time_coeff_.bd2 / dt_, tmpR0b_);
  tmpR0_.Add(time_coeff_.bd3 / dt_, tmpR0c_);

  MsRho_form_->Update();
  MsRho_form_->Assemble();
  MsRho_form_->FormSystemMatrix(empty, MsRho_);
  MsRho_->AddMult(tmpR0_, resY_, -1.0);

  // If NOT using operator splitting, include the reaction source term
  // on the rhs.  If using operator splitting, this is handled by
  // speciesSubstep.
  if (!operator_split_) {
    setScalarFromVector(prodY_, iSpec, &tmpR0_);
    Ms_->AddMult(tmpR0_, resY_);
  }

  // Add natural boundary terms here later
  // NB: adiabatic natural BC is handled, but don't have ability to impose non-zero heat flux yet

  // Update Helmholtz operator to account for changing dt, rho, and kappa
  rhoDt_gf_ = rn_gf_;
  rhoDt_gf_ *= (time_coeff_.bd0 / dt_);
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
  setScalarFromVector(Yn_, iSpec, &tmpR0a_);
  Yn_next_gf_.SetFromTrueDofs(tmpR0a_);

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
  if (!(HyInv_->GetConverged())) {
    if (rank0_) std::cout << " Species " << iSpec << " is not converging..." << endl;
  }
  assert(HyInv_->GetConverged());

  // copy into full species vector & gf
  Hy_form_->RecoverFEMSolution(Xt2, resT_gf_, Yn_next_gf_);
  Yn_next_gf_.GetTrueDofs(tmpR0_);

  for (int i = 0; i < sDofInt_; i++) {
    if (tmpR0_[i] < 0.0) tmpR0_[i] = 0.0;
  }
  Yn_next_gf_.SetFromTrueDofs(tmpR0_);

  setVectorFromScalar(tmpR0_, iSpec, &Yn_next_);
}

// Y{n + (substep+1)} = dt * wdot(Y{n + (substep)} + Y{n + (substep+1)}*
void ReactingFlow::speciesSubstep(int iSpec, int iSub) {
  // substep dt
  double dtSub = dt_ / (double)nSub_;

  // production of iSpec
  setScalarFromVector(prodY_, iSpec, &tmpR0_);
  tmpR0_ /= rn_;
  tmpR0_ *= dtSub;

  // YnStar has star state at substep here
  setScalarFromVector(YnStar_, iSpec, &tmpR0a_);
  tmpR0_.Add(1.0, tmpR0a_);

  setScalarFromVector(Yn_, iSpec, &tmpR0a_);
  tmpR0_.Add(1.0, tmpR0a_);

  // clip any small negative values
  {
    double *data = tmpR0_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      data[i] = max(data[i], 0.0);
    }
  }

  // Yn now has full state at substep
  setVectorFromScalar(tmpR0_, iSpec, &Yn_);
}

void ReactingFlow::speciesProduction() {
  const double *dataT = Tn_.HostRead();
  const double *dataRho = rn_.HostRead();
  const double *dataY = Yn_.HostRead();
  double *dataProd = prodY_.HostWrite();
  double *dataEmit = prodE_.HostWrite();

  // const int nEq = dim_ + 2 + nActiveSpecies_;
  Vector state(gpudata::MAXEQUATIONS);
  state = 0.0;

  // Vectors used in computing chemical sources at each point
  Vector n_sp;          // set to size nSpecies_ in computeNumberDensities
  Vector kfwd;          // set to size nReactions_ in computeForwardRareCoeffs
  Vector keq;           // set to size nReactions_ in computeEquilibriumConstants
  Vector progressRate;  // set to size nReactions_ in computeProgressRate
  Vector creationRate;  // set to size nSpecies_ in computeCreationRate
  Vector emissionRate;  // set to size nSpecies_ in computeCreationRate

  kfwd.SetSize(chemistry_->getNumReactions());
  keq.SetSize(chemistry_->getNumReactions());

  for (int i = 0; i < sDofInt_; i++) {
    // Get temperature
    const double Th = dataT[i];
    const double Te = Th;  // single temperature model

    // Set the conserved state.  This is only used here to evaluate
    // the molar density, so only the mixture (mass) density and
    // species (mass) densities are used and need to be filled
    state[0] = dataRho[i];
    for (int sp = 0; sp < nActiveSpecies_; sp++) {
      state[dim_ + 1 + sp + 1] = dataRho[i] * dataY[i + sp * sDofInt_];
    }

    // Evaluate the mole densities (from mass densities)
    mixture_->computeNumberDensities(state, n_sp);

    // Evaluate the chemical source terms
    chemistry_->computeForwardRateCoeffs(n_sp.Read(), Th, Te, i, kfwd.HostWrite());
    chemistry_->computeEquilibriumConstants(Th, Te, keq.HostWrite());
    chemistry_->computeProgressRate(n_sp, kfwd, keq, progressRate);
    chemistry_->computeCreationRate(progressRate, creationRate, emissionRate);

    // Place sources into storage for use in speciesStep
    for (int sp = 0; sp < nSpecies_; sp++) {
      dataProd[i + sp * sDofInt_] = creationRate[sp];
      dataEmit[i + sp * sDofInt_] = emissionRate[sp];
    }
  }
}

void ReactingFlow::heatOfFormation() {
  hw_ = 0.0;
  tmpR0_ = 0.0;
  double *h_hw = hw_.HostReadWrite();
  double *h_em = tmpR0_.HostReadWrite();

  const double *dataT = Tn_.HostRead();
  const double *h_prodY = prodY_.HostRead();
  const double *h_prodE = prodE_.HostRead();

  double hspecies;

  // NOTE: not very elegant but dont want to nest an if
  if (radiative_decay_NECincluded_) {
    for (int i = 0; i < sDofInt_; i++) {
      const double T = dataT[i];

      // Sum over species to get enthalpy term
      for (int n = 0; n < nSpecies_; n++) {
        double molarCV = speciesMolarCv_[n];
        molarCV *= UNIVERSALGASCONSTANT;
        double molarCP = molarCV + UNIVERSALGASCONSTANT;

        hspecies = (molarCP * T + gasParams_(n, GasParams::FORMATION_ENERGY)) / gasParams_(n, GasParams::SPECIES_MW);
        h_hw[i] -= hspecies * (h_prodY[i + n * sDofInt_] - h_prodE[i + n * sDofInt_]);
        h_em[i] = hspecies * h_prodE[i + n * sDofInt_];  // not sure of sign
      }
    }

  } else {
    for (int i = 0; i < sDofInt_; i++) {
      const double T = dataT[i];

      // Sum over species to get enthalpy term
      for (int n = 0; n < nSpecies_; n++) {
        double molarCV = speciesMolarCv_[n];
        molarCV *= UNIVERSALGASCONSTANT;
        double molarCP = molarCV + UNIVERSALGASCONSTANT;

        hspecies = (molarCP * T + gasParams_(n, GasParams::FORMATION_ENERGY)) / gasParams_(n, GasParams::SPECIES_MW);
        h_hw[i] -= hspecies * h_prodY[i + n * sDofInt_];
        h_em[i] = hspecies * h_prodE[i + n * sDofInt_];  // not sure of sign
      }
    }
  }

  emission_gf_.SetFromTrueDofs(tmpR0_);
}

void ReactingFlow::crossDiffusion() {
  G_->Mult(Tn_, tmpR1_);
  Mv_inv_->Mult(tmpR1_, tmpR1a_);
  tmpR1c_ = 0.0;
  for (int i = 0; i < nSpecies_; i++) {
    setScalarFromVector(Yn_next_, i, &tmpR0a_);
    setScalarFromVector(diffY_, i, &tmpR0b_);
    setScalarFromVector(CpY_, i, &tmpR0c_);
    G_->Mult(tmpR0a_, tmpR1_);
    Mv_inv_->Mult(tmpR1_, tmpR1b_);
    tmpR0b_ *= tmpR0c_;
    multScalarVectorIP(tmpR0b_, &tmpR1b_, dim_);
    tmpR1c_ += tmpR1b_;
  }
  weff_gf_.SetFromTrueDofs(tmpR1c_);
  dotVector(tmpR1a_, tmpR1c_, &tmpR0_, dim_);
  Ms_->Mult(tmpR0_, crossDiff_);
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

  Ay_form_->Update();
  Ay_form_->Assemble();
  Ay_form_->FormSystemMatrix(empty, Ay_);

  if (extrap == true) {
    setScalarFromVector(Yn_, iSpec, &tmpR0_);
  } else {
    setScalarFromVector(Yext_, iSpec, &tmpR0_);
  }
  Ay_->Mult(tmpR0_, tmpR0a_);

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

  // TODO(trevilo): This hackery is necessary b/c we don't have access
  // to the IOOptions object we already read (as part of the low Mach
  // parsing in LoMachSolver::parseSolverOptions()).  That object
  // should be made available here.
  bool restart_from_lte;
  tpsP_->getInput("io/restartFromLTE", restart_from_lte, false);

  // If restarting from LTE, we don't expect to find species in the restart file
  const bool species_in_restart_file = !restart_from_lte;

  io.registerIOFamily("Species", "/species", &YnFull_gf_, false, species_in_restart_file);
  for (int sp = 0; sp < nSpecies_; sp++) {
    std::string speciesName = std::to_string(sp);
    io.registerIOVar("/species", "Y_" + speciesName, sp, species_in_restart_file);
  }
}

void ReactingFlow::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", &Tn_gf_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("sigma", &sigma_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
  pvdc.RegisterField("mu", &visc_gf_);
  pvdc.RegisterField("Qt", &Qt_gf_);
  pvdc.RegisterField("Rmix", &Rmix_gf_);
  pvdc.RegisterField("CpMix", &CpMix_gf_);
  pvdc.RegisterField("Sjoule", &jh_gf_);
  pvdc.RegisterField("epsilon_rad", &radiation_sink_gf_);
  pvdc.RegisterField("weff", &weff_gf_);
  pvdc.RegisterField("emission", &emission_gf_);

  vizSpecFields_.clear();
  vizSpecNames_.clear();
  for (int sp = 0; sp < nSpecies_; sp++) {
    vizSpecFields_.push_back(new ParGridFunction(sfes_, YnFull_gf_, (sp * sDof_)));
    vizSpecNames_.push_back(std::string("Yn_" + speciesNames_[sp]));
    pvdc.RegisterField(vizSpecNames_[sp], vizSpecFields_[sp]);
  }
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

// Interpolate star state to substeps. Linear for now, use higher order
// if time-splitting order can be improved
void ReactingFlow::substepState() {
  // delta over substep of star state
  double wt = 1.0 / (double)nSub_;

  TnStar_.Set(wt, Tn_next_);
  TnStar_.Add(-wt, temp_buffer_);

  YnStar_.Set(wt, Yn_next_);
  YnStar_.Add(-wt, spec_buffer_);
}

void ReactingFlow::updateMixture() {
  {
    double *dataY = Yn_gf_.HostReadWrite();
    double *dataR = Rmix_gf_.HostReadWrite();
    double *dataM = Mmix_gf_.HostReadWrite();
    Mmix_gf_ = 0.0;

    for (int sp = 0; sp < nSpecies_; sp++) {
      setScalarFromVector(Yn_, sp, &tmpR0_);
      Yn_gf_.SetFromTrueDofs(tmpR0_);
      for (int i = 0; i < sDof_; i++) {
        dataM[i] += dataY[i] / gasParams_(sp, GasParams::SPECIES_MW);
      }
    }

    for (int i = 0; i < sDof_; i++) {
      dataM[i] = 1.0 / dataM[i];
    }
    for (int i = 0; i < sDof_; i++) {
      dataR[i] = Rgas_ / dataM[i];
    }
  }

  // can use mixture calls directly for this
  for (int sp = 0; sp < nSpecies_; sp++) {
    setScalarFromVector(Yn_, sp, &tmpR0a_);
    setScalarFromVector(Xn_, sp, &tmpR0b_);
    Mmix_gf_.GetTrueDofs(tmpR0c_);
    double *d_Y = tmpR0a_.HostReadWrite();
    double *d_X = tmpR0b_.HostReadWrite();
    double *d_M = tmpR0c_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      d_X[i] = d_Y[i] * d_M[i] / gasParams_(sp, GasParams::SPECIES_MW);
    }
    setVectorFromScalar(tmpR0b_, sp, &Xn_);
  }

  // mixture Cp
  {
    double *d_CMix = tmpR0c_.HostReadWrite();
    double *d_Yn = Yn_.HostReadWrite();
    double *d_Rho = rn_.HostReadWrite();
    double *d_Cp = CpY_.HostReadWrite();

    // int nEq = dim_ + 2 + nActiveSpecies_;
    Vector state(gpudata::MAXEQUATIONS);
    state = 0.0;

    Vector n_sp;

    for (int i = 0; i < sDofInt_; i++) {
      double cpMix;

      // Set up conserved state (just the mass densities, which is all we need here)
      state[0] = d_Rho[i];
      for (int sp = 0; sp < nActiveSpecies_; sp++) {
        state[dim_ + 1 + sp + 1] = d_Rho[i] * d_Yn[i + sp * sDofInt_];
      }

      // Evaluate the mole densities (from mass densities)
      mixture_->computeNumberDensities(state, n_sp);

      // GetMixtureCp returns cpMix = sum_s X_s Cp_s, where X_s is
      // mole density of species s and Cp_s is molar specific heat
      // (constant pressure) of species s, so cpMix at this point
      // (units J/m^3), which is rho*Cp, where rho is the mixture
      // density (kg/m^3) and Cp is the is the mixture mass specific
      // heat (units J/(kg*K).
      mixture_->GetMixtureCp(n_sp, d_Rho[i], cpMix);

      // Everything else expects CpMix_gf_ to be the mixture mass Cp
      // (with units J/(kg*K)), so divide by mixture density
      d_CMix[i] = cpMix / d_Rho[i];

      for (int sp = 0; sp < nSpecies_; sp++) {
        double molarCV = speciesMolarCv_[sp];
        molarCV *= UNIVERSALGASCONSTANT;
        double molarCP = molarCV + UNIVERSALGASCONSTANT;
        const double cp_sp = molarCP / gasParams_(sp, GasParams::SPECIES_MW);
        d_Cp[i + sp * sDofInt_] = cp_sp;
      }
    }
  }
  CpMix_gf_.SetFromTrueDofs(tmpR0c_);
}

// update thermodynamic pressure
void ReactingFlow::updateThermoP() {
  Rmix_gf_.GetTrueDofs(tmpR0a_);
  if (!domain_is_open_) {
    double allMass;
    double myMass = 0.0;

    // rotate Po stores
    Pnm3_ = Pnm2_;
    Pnm2_ = Pnm1_;
    Pnm1_ = thermo_pressure_;

    // calculate total mass given previous Po
    tmpR0_ = thermo_pressure_;
    tmpR0_ /= Tn_;
    tmpR0_ /= tmpR0a_;
    Ms_->Mult(tmpR0_, tmpR0b_);
    myMass = tmpR0b_.Sum();
    MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, tpsP_->getTPSCommWorld());

    // update Po so mass is conserved given the new temperature
    thermo_pressure_ = system_mass_ / allMass * Pnm1_;
    dtP_ = time_coeff_.bd0 * thermo_pressure_ + time_coeff_.bd1 * Pnm1_ + time_coeff_.bd2 * Pnm2_ +
           time_coeff_.bd3 * Pnm3_;

    // For operator split, dtP term is in split substeps, so need to
    // divide dt by nSub.  For 'unified' approach (not operator
    // split), nSub = 1, so it makes no difference.
    dtP_ *= (nSub_ / dt_);
  }
}

void ReactingFlow::updateDiffusivity() {
  (flow_interface_->velocity)->GetTrueDofs(tmpR1_);
  const double *dataTemp = Tn_.HostRead();
  const double *dataRho = rn_.HostRead();
  const double *dataU = tmpR1_.HostRead();
  double diffY_min = 1.0e-8;  // make readable

  // species diffusivities
  {
    double *dataDiff = diffY_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      double Efield[gpudata::MAXDIM];
      for (int v = 0; v < dim_; v++) Efield[v] = 0.0;

      double state[gpudata::MAXEQUATIONS];
      double conservedState[gpudata::MAXEQUATIONS];
      double diffSp[gpudata::MAXSPECIES];

      // Populate *primitive* state vector = [rho, velocity, temperature, species mole densities]
      state[0] = dataRho[i];
      for (int eq = 0; eq < dim_; eq++) {
        state[eq + 1] = dataU[i + eq * sDofInt_];
      }
      state[dim_ + 1] = dataTemp[i];
      for (int sp = 0; sp < nActiveSpecies_; sp++) {
        state[dim_ + 2 + sp] = dataRho[i] * Yn_[i + sp * sDofInt_] / mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
      }

      mixture_->GetConservativesFromPrimitives(state, conservedState);
      transport_->computeMixtureAverageDiffusivity(conservedState, Efield, diffSp, false);
      for (int sp = 0; sp < nSpecies_; sp++) {
        diffSp[sp] = std::max(diffSp[sp], diffY_min);
        dataDiff[i + sp * sDofInt_] = dataRho[i] * diffSp[sp];
      }
    }
  }

  // viscosity
  {
    double *dataVisc = visc_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      double state[gpudata::MAXEQUATIONS];
      double conservedState[gpudata::MAXEQUATIONS];
      double visc[2];

      // Populate *primitive* state vector = [rho, velocity, temperature, species mole densities]
      state[0] = dataRho[i];
      for (int eq = 0; eq < dim_; eq++) {
        state[eq + 1] = dataU[i + eq * sDofInt_];
      }
      state[dim_ + 1] = dataTemp[i];
      for (int sp = 0; sp < nActiveSpecies_; sp++) {
        state[dim_ + 2 + sp] = dataRho[i] * Yn_[i + sp * sDofInt_] / mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
      }

      mixture_->GetConservativesFromPrimitives(state, conservedState);
      transport_->GetViscosities(conservedState, state, visc);
      dataVisc[i] = visc[0];
    }
  }
  visc_gf_.SetFromTrueDofs(visc_);

  // thermal conductivity (kappa = alpha*rho*Cp = (nu/Pr)*(rho*Cp))
  {
    double *dataKappa = kappa_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      // int nEq = dim_ + 2 + nActiveSpecies_;
      double state[gpudata::MAXEQUATIONS];
      double conservedState[gpudata::MAXEQUATIONS];
      double kappa[2];

      // Populate *primitive* state vector = [rho, velocity, temperature, species mole densities]
      state[0] = dataRho[i];
      for (int eq = 0; eq < dim_; eq++) {
        state[eq + 1] = dataU[i + eq * sDofInt_];
      }
      state[dim_ + 1] = dataTemp[i];
      for (int sp = 0; sp < nActiveSpecies_; sp++) {
        state[dim_ + 2 + sp] = dataRho[i] * Yn_[i + sp * sDofInt_] / mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
      }

      mixture_->GetConservativesFromPrimitives(state, conservedState);
      transport_->GetThermalConductivities(conservedState, state, kappa);
      dataKappa[i] = kappa[0] + kappa[1];  // for single temperature, transport includes both heavy and electron kappa
    }
  }
  kappa_gf_.SetFromTrueDofs(kappa_);

  // electrical conductivity
  {
    double *h_sig = sigma_.HostReadWrite();
    for (int i = 0; i < sDofInt_; i++) {
      // int nEq = dim_ + 2 + nActiveSpecies_;
      double state[gpudata::MAXEQUATIONS];
      double conservedState[gpudata::MAXEQUATIONS];

      // Populate *primitive* state vector = [rho, velocity, temperature, species mole densities]
      state[0] = dataRho[i];
      for (int eq = 0; eq < dim_; eq++) {
        state[eq + 1] = dataU[i + eq * sDofInt_];
      }
      state[dim_ + 1] = dataTemp[i];
      for (int sp = 0; sp < nActiveSpecies_; sp++) {
        state[dim_ + 2 + sp] = dataRho[i] * Yn_[i + sp * sDofInt_] / mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
      }

      mixture_->GetConservativesFromPrimitives(state, conservedState);

      double sig;
      transport_->ComputeElectricalConductivity(conservedState, sig);
      h_sig[i] = sig;
    }
  }
  sigma_gf_.SetFromTrueDofs(sigma_);
}

void ReactingFlow::evaluatePlasmaConductivityGF() {
  (flow_interface_->velocity)->GetTrueDofs(tmpR1_);
  const double *dataTemp = Tn_.HostRead();
  const double *dataRho = rn_.HostRead();
  const double *dataU = tmpR1_.HostRead();

  double *h_sig = sigma_.HostReadWrite();
  for (int i = 0; i < sDofInt_; i++) {
    // int nEq = dim_ + 2 + nActiveSpecies_;
    double state[gpudata::MAXEQUATIONS];
    double conservedState[gpudata::MAXEQUATIONS];

    // Populate *primitive* state vector = [rho, velocity, temperature, species mole densities]
    state[0] = dataRho[i];
    for (int eq = 0; eq < dim_; eq++) {
      state[eq + 1] = dataU[i + eq * sDofInt_];
    }
    state[dim_ + 1] = dataTemp[i];
    for (int sp = 0; sp < nActiveSpecies_; sp++) {
      state[dim_ + 2 + sp] = dataRho[i] * Yn_[i + sp * sDofInt_] / mixture_->GetGasParams(sp, GasParams::SPECIES_MW);
    }

    mixture_->GetConservativesFromPrimitives(state, conservedState);

    double sig;
    transport_->ComputeElectricalConductivity(conservedState, sig);
    h_sig[i] = sig;
  }
  sigma_gf_.SetFromTrueDofs(sigma_);
}

void ReactingFlow::updateDensity(double tStep, bool update_mass_matrix) {
  Array<int> empty;
  Rmix_gf_.GetTrueDofs(tmpR0a_);

  // set rn
  if (constant_density_ != true) {
    if (tStep == 1.0) {
      rn_ = thermo_pressure_;
      rn_ /= tmpR0a_;
      rn_ /= Tn_next_;
    } else if (tStep == 0.5) {
      rn_ = thermo_pressure_;
      rn_ /= tmpR0a_;
      rn_ /= Tn_next_;
      rn_ *= 0.5;
      tmpR0_ = thermo_pressure_;
      tmpR0_ /= tmpR0a_;
      tmpR0_ /= Tn_;
      rn_.Add(0.5, tmpR0_);
    } else {
      multConstScalarInv(thermo_pressure_, tmpR0a_, &tmpR0b_);
      multScalarScalarInv(tmpR0b_, Tn_, &rn_);
    }
  } else {
    rn_ = static_rho_;
  }

  const double min_rho = rn_.Min();
  if (min_rho < 0.0) {
    for (int i = 0; i < rn_.Size(); i++) {
      if (rn_[i] < 0.0) {
        printf("i = %d, Tn = %.6e, Tnext = %.6e, rho = %.6e\n", i, Tn_[i], Tn_next_[i], rn_[i]);
        fflush(stdout);
      }
    }
  }
  rn_gf_.SetFromTrueDofs(rn_);

  if (update_mass_matrix) {
    MsRho_form_->Update();
    MsRho_form_->Assemble();
    MsRho_form_->FormSystemMatrix(empty, MsRho_);
  }

  // project to p-space in case not same as vel-temp
  R0PM0_gf_.SetFromTrueDofs(rn_);
}

void ReactingFlow::computeSystemMass() {
  Rmix_gf_.GetTrueDofs(tmpR0_);
  system_mass_ = 0.0;
  double myMass = 0.0;
  rn_ = thermo_pressure_;
  rn_ /= tmpR0_;
  rn_ /= Tn_;
  Ms_->Mult(rn_, tmpR0_);
  myMass = tmpR0_.Sum();
  MPI_Allreduce(&myMass, &system_mass_, 1, MPI_DOUBLE, MPI_SUM, tpsP_->getTPSCommWorld());
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
      assert(!temp_ess_attr_[i]);
      temp_ess_attr_[i] = 1;
    }
  }
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

  // Joule heating (and radiation sink)
  jh_form_->Update();
  jh_form_->Assemble();
  jh_form_->ParallelAssemble(jh_);
  tmpR0_ -= jh_;

  // heat of formation
  Ms_->AddMult(hw_, tmpR0_, -1.0);

  // species-temp diffusion term, already in int-weak form
  tmpR0_.Add(-1.0, crossDiff_);

  sfes_->GetRestrictionMatrix()->MultTranspose(tmpR0_, resT_gf_);

  Qt_ = 0.0;
  Qt_gf_.SetFromTrueDofs(Qt_);

  Vector Xqt, Bqt;
  Mq_form_->FormLinearSystem(Qt_ess_tdof_, Qt_gf_, resT_gf_, Mq_, Xqt, Bqt, 1);

  MqInv_->Mult(Bqt, Xqt);
  Mq_form_->RecoverFEMSolution(Xqt, resT_gf_, Qt_gf_);

  Qt_gf_ *= Rmix_gf_;
  Qt_gf_ /= CpMix_gf_;
  Qt_gf_ /= thermo_pressure_;
  Qt_gf_.Neg();
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
        std::string name = speciesNames_[sp];
        grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                    name.c_str());
        exit(-1);
      }
    } else {
      std::string name = speciesNames_[sp];
      grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                  name.c_str());
      exit(-1);
    }
  }

  // Check all species are identified.
  for (int sp = 0; sp < nSpecies_; sp++) {
    if (speciesType[sp] == NONE_ARGSPCS) {
      std::string name = speciesNames_[sp];
      grvy_printf(GRVY_ERROR, "The species %s is not identified in ArgonMixtureTransport! \n", name.c_str());
      exit(-1);
    }
  }

  return;
}

void ReactingFlow::readTableWrapper(std::string inputPath, TableInput &result) {
  std::string filename;
  tpsP_->getInput((inputPath + "/x_log").c_str(), result.xLogScale, false);
  tpsP_->getInput((inputPath + "/f_log").c_str(), result.fLogScale, false);
  tpsP_->getInput((inputPath + "/order").c_str(), result.order, 1);
  tpsP_->getRequiredInput((inputPath + "/filename").c_str(), filename);
  readTable(tpsP_->getTPSCommWorld(), filename, result.xLogScale, result.fLogScale, result.order, tableHost_, result);
}

void ReactingFlow::identifyCollisionType(const Array<ArgonSpcs> &speciesType, ArgonColl *collisionIndex) {
  for (int spI = 0; spI < nSpecies_; spI++) {
    for (int spJ = spI; spJ < nSpecies_; spJ++) {
      // If not initialized, will raise an error.
      collisionIndex[spI + spJ * nSpecies_] = NONE_ARGCOLL;

      const double pairType = gasParams_(spI, GasParams::SPECIES_CHARGES) * gasParams_(spJ, GasParams::SPECIES_CHARGES);
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
          std::string name1 = speciesNames_[spI];
          std::string name2 = speciesNames_[spJ];
          grvy_printf(GRVY_ERROR, "1. %s - %s is not supported in ArgonMixtureTransport! \n", name1.c_str(),
                      name2.c_str());
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
        grvy_printf(GRVY_ERROR, "2. %s - %s is not initialized in ArgonMixtureTransport! \n", name1.c_str(),
                    name2.c_str());
        exit(-1);
      }
    }
  }

  return;
}

double binaryTest(const Vector &coords, double t) {
  double x = coords(0);
  double y = coords(1);
  // double z = coords(2);
  double pi = 3.14159265359;
  double kx, ky;
  double Lx, Ly;
  double yn;

  kx = 2.0;
  ky = 0.0;
  Lx = 5.0;
  Ly = 1.0;

  yn = 0.5 + 0.45 * cos(2.0 * pi * kx * x / Lx) * cos(2.0 * pi * ky * y / Ly);

  return yn;
}

double species_stepLeft(const Vector &coords, double t) {
  double x = coords(0);
  // douable y = coords(1);
  // double z = coords(2);
  double pi = 3.14159265359;
  double yn;
  yn = 1.0e-12;
  // if (x < 0.0) { yn = 1.0; }
  yn = 0.01 * 0.5 * (cos(2.0 * pi * x) + 1.0);
  return yn;
}

double species_stepRight(const Vector &coords, double t) {
  double x = coords(0);
  // double y = coords(1);
  // double z = coords(2);
  double pi = 3.14159265359;
  double yn;
  yn = 1.0e-12;
  // if (x > 0.0) { yn = 1.0; }
  yn = 0.01 * 0.5 * (sin(2.0 * pi * x) + 1.0) + 0.99;
  yn = 1.0;
  return yn;
}

double species_uniform(const Vector &coords, double t) {
  // double x = coords(0);
  // double y = coords(1);
  // double z = coords(2);
  double yn;
  yn = 1.0e-12;
  return yn;
}

void ReactingFlow::push(TPS::Tps2Boltzmann &interface) {
  assert(interface.IsInitialized());

  //const int nscalardofs(vfes_->GetNDofs());

  mfem::ParGridFunction *species =
      new mfem::ParGridFunction(&interface.NativeFes(TPS::Tps2Boltzmann::Index::SpeciesDensities));

  std::cout << sDofInt_ << " " << interface.Nspecies() << std::endl << std::flush;
  MPI_Barrier(pmesh_->GetComm());
  mfem::Vector speciesInt(sDofInt_*interface.Nspecies());
  double *species_data = speciesInt.HostWrite();

  const double *dataRho = rn_.HostRead();
  const double *dataY = Yn_.HostRead();

  double state_local[gpudata::MAXEQUATIONS];
  double species_local[gpudata::MAXSPECIES];

  for (int i = 0; i < gpudata::MAXEQUATIONS; ++i)
    state_local[i] = 0.;

  for (int i = 0; i < gpudata::MAXSPECIES; ++i)
    species_local[i] = 0.;

  for (int i = 0; i < sDofInt_; i++) {
    state_local[0] = dataRho[i];
    for (int asp = 0; asp < nActiveSpecies_; asp++)
      state_local[dim_ + 2 + asp] = dataRho[i]*dataY[i+asp*sDofInt_];
    mixture_->computeNumberDensities(state_local, species_local);

    for (int sp = 0; sp < interface.Nspecies(); sp++)
      species_data[i + sp * sDofInt_] = AVOGADRONUMBER * species_local[sp];
  }

  std::cout << "species->SetFromTrueDofs(speciesInt);" << std::endl << std::flush;
  MPI_Barrier(pmesh_->GetComm());
  species->SetFromTrueDofs(speciesInt);
  std::cout << "Tn_gf_.SetFromTrueDofs(Tn_);" << std::endl << std::flush;
  MPI_Barrier(pmesh_->GetComm());
  Tn_gf_.SetFromTrueDofs(Tn_);
  interface.interpolateFromNativeFES(*species, TPS::Tps2Boltzmann::Index::SpeciesDensities);
  interface.interpolateFromNativeFES(Tn_gf_, TPS::Tps2Boltzmann::Index::HeavyTemperature);
  interface.interpolateFromNativeFES(Tn_gf_, TPS::Tps2Boltzmann::Index::ElectronTemperature);

  interface.setTimeStep(this->dt_);
  interface.setCurrentTime(this->time_);

  delete species;
}

void ReactingFlow::fetch(TPS::Tps2Boltzmann &interface) {
  mfem::ParFiniteElementSpace *reaction_rates_fes(&(interface.NativeFes(TPS::Tps2Boltzmann::Index::ReactionRates)));
  externalReactionRates_.reset(new mfem::ParGridFunction(reaction_rates_fes));
  interface.interpolateToNativeFES(*externalReactionRates_, TPS::Tps2Boltzmann::Index::ReactionRates);
#if defined(_CUDA_) || defined(_HIP_)
  const double *data(externalReactionRates_->Read());
  int size(externalReactionRates_->FESpace()->GetNDofs());
  assert(externalReactionRates_->FESpace()->GetOrdering() == mfem::Ordering::byNODES);
  gpu::deviceSetChemistryReactionData<<<1, 1>>>(data, size, chemistry_);
#else
  chemistry_->setGridFunctionRates(*externalReactionRates_);
#endif
}


#if 0
void ReactingFlow::uniformInlet() {
  int myRank;
  MPI_Comm_rank(tpsP_->getTPSCommWorld(), &myRank);

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
  // double x = coords(0);
  double y = coords(1);
  // double z = coords(2);
  double temp;
  temp = Tlo + 0.5 * (y + 1.0) * (Thi - Tlo);
  // temp = 0.5 * (Thi + Tlo);

  return temp;
}

double temp_wall(const Vector &coords, double t) {
  double Thi = 400.0;
  double Tlo = 200.0;
  double Tmean, tRamp, wt;
  // double x = coords(0);
  // double y = coords(1);
  // double z = coords(2);
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
  // double y = coords(1);
  // double z = coords(2);
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
  // double x = coords(0);
  // double y = coords(1);
  // double z = coords(2);
  double temp;
  temp = Tlo + (y + 0.5) * (Thi - Tlo);
  return temp;
}
#endif
