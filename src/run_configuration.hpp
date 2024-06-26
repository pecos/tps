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

#ifndef RUN_CONFIGURATION_HPP_
#define RUN_CONFIGURATION_HPP_

#include <tps_config.h>

#include <list>
#include <string>
#include <utility>
#include <vector>

#include "averaging.hpp"
#include "dataStructures.hpp"
#include "io.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

// Class to manage the run options as
// specified in the input file
// Lines begining with # are ignored
class RunConfiguration {
 public:
  // IO options
  IOOptions io_opts_;

  // Averaging options
  AveragingOptions avg_opts_;

  // mesh file file
  string meshFile;
  int ref_levels;

  // partition file
  string partFile;

  // time integrator. Possible values
  //  1: ForwardEulerSolver
  //  2: RK2Solver(1.0)
  //  3: RK3SSPSolver
  //  4: RK4Solver
  //  6: RK6Solver
  int timeIntegratorType;

  // subgrid scale model. Possible values
  //  0: None
  //  1: Smagorinsky
  //  2: Sigma
  int sgsModelType;

  // only include fluctuating vel in sgs model
  bool sgsExcludeMean;

  // order of the solution. Defaults to 4
  int solOrder;

  // Integration rule (quadrature)
  // 0 -> Gauss-Legendre
  // 1 -> Gauss-Lobatto
  int integrationRule;

  // Basis type
  // 0 -> Gauss-Legendre
  // 1 -> Gauss-Lobatto
  int basisType;

  double cflNum;

  // variable/constant time-step
  bool constantTimeStep;
  double dt_fixed;
  double dt_initial;
  double dt_factor;
  int solver_iter;
  double solver_tol;
  int bdfOrder;
  int abOrder;

  // num iterations. Defaults to 0
  int numIters;

  // iteration frequency to provide iteration timing info
  int timingFreq;

  // output interval in num. of iters.
  int itersOut;

  // Use Roe-type Riemann solver
  bool useRoe;

  // restart controls
  bool restart_hdf5_conversion;  // read in older ascii format
  int restart_cycle;

  // Restart from different order solution on same mesh.  New order
  // set by POL_ORDER, old order determined from restart file.
  bool singleRestartFile;

  // working fluid. Options thus far
  // DRY_AIR
  // Defaults to DRAY_AIR
  WorkingFluid workFluid;

  // Factor by which the viscosity is multiplied.
  double visc_mult;

  // Bulk viscosity factor. Bulk viscosity is calculated as
  // dynamic_viscosity (accounting for visc_mult) times bulk_visc
  // i.e., visc(sutherland)*visc_mult*bulk_visc
  double bulk_visc;

  // The following four keywords define two planes in which
  // a linearly varying viscosity can be defined between these two.
  // The planes are defined by the normal and one point being the
  // normal equal for both planes. The visc ratio will vary linearly
  // from 0 to viscRatio from the plane defined by pointInit and
  // the plane defined by point0.
  linearlyVaryingVisc linViscData;

  planeDumpData planeDump;

  SutherlandData sutherland_;

  SpongeZoneData* spongeData_;
  int numSpongeRegions_;
  void initSpongeData();

  int numHeatSources;
  heatSourceData* heatSource;

  // Reference length
  double refLength;

  // minimum grid size for SGS
  double sgsFloor;

  // SGS model constant
  double sgs_model_const;

  // equations to be solved. Options thus far
  Equations eqSystem;
  bool axisymmetric_;

  // initial constant field
  double initRhoRhoVp[5];

  // ic flag to use internal ic function definition
  bool useICFunction;
  bool useICBoxFunction;

  // reset Temp field to IC
  bool resetTemp;

  // wall bc flag to use internal function definition
  bool useWallFunction;
  bool useWallBox;  // NEED A BETTER WAY TO DO THIS!

  // Imposed pressure gradient
  bool isForcing;
  double gradPress[3];

  // Boussinesq term
  bool isGravity;
  double gravity[3];

  // closed or open system
  bool isOpen;

  // loMach specific additions
  double amb_pres, const_visc, const_dens;

  // Inlet BC data
  Array<double> inletBC;
  std::vector<pair<int, InletType>> inletPatchType;
  double velInlet[3];
  double tempInlet;
  double densInlet;
  int rampStepsInlet;

  // Outlet BC data
  Array<double> outletBC;
  std::vector<pair<int, OutletType>> outletPatchType;
  double outletPressure;

  // Wall BC data
  // Array<double> wallBC;
  std::vector<WallData> wallBC;
  std::vector<pair<int, WallType>> wallPatchType;

  // Apply BCs in gradient calculation
  bool useBCinGrad;

  // Passive scalar data
  Array<passiveScalarData*> arrayPassiveScalar;

  // Resource management system - job monitoring
  bool rm_enableMonitor_;  // flag to trigger RMS monitoring and job resubmissions
  int rm_threshold_;       // remaining secs for current job to trigger re-submission
  int rm_checkFrequency_;  // iteration frequency to check RMS status

  int exit_checkFrequency_;  // iteration frequency to check for early exit

  // number of species;
  int numSpecies;
  int backgroundIndex;
  // int electronIndex;
  Vector initialMassFractions;
  double initialElectronTemperature;

  std::vector<std::string> speciesNames;

  // TODO(kevin): make tps input parser accessible to all classes.
  int numReactions;
  Vector reactionEnergies;
  std::vector<std::string> reactionEquations;
  Array<ReactionModel> reactionModels;
  // NOTE(kevin): this array of Vectors is to provide a proper pointer for both gpu and cpu.
  // Indexes of this array do not correspond exactly to reaction index.
  std::vector<mfem::Vector> rxnModelParamsHost;
  DenseMatrix reactantStoich, productStoich;
  Array<bool> detailedBalance;
  // std::vector<std::vector<double>> equilibriumConstantParams;
  double equilibriumConstantParams[gpudata::MAXCHEMPARAMS * gpudata::MAXREACTIONS];

  // NOTE(kevin): this vector of DenseMatrix stores all the tables and provides a proper pointer for both gpu and cpu.
  // note that this will store all types of tabulated data- rxn rate, collision integrals, etc.
  // Indexes of this vector therefore do not correspond exactly to rxn index or other quantities.
  std::list<mfem::DenseMatrix> tableHost;

  std::map<int, int> mixtureToInputMap;
  std::map<std::string, int> speciesMapping;
  DenseMatrix gasParams;
  Vector constantMolarCV;
  Vector constantMolarCP;

  int numAtoms;
  DenseMatrix speciesComposition;
  Vector atomMW;  // used 'atom' instead of 'element', to avoid confusion with finite 'element'.
  std::map<std::string, int> atomMap;

  // ambipolar flag
  bool ambipolar;

  // two temperature flag
  bool twoTemperature;

  // For USER_DEFINED gas mixture, set models from input file.
  GasModel gasModel;
  TransportModel transportModel;
  ChemistryModel chemistryModel_;

  // flag for using third-order electron thermal conductivity
  bool thirdOrderkElectron;

  // flag to use mixing length model
  bool use_mixing_length;

  // data structure for constant transport property class
  constantTransportData constantTransport;
  mixingLengthTransportData mix_length_trans_input_;

  // Use MMS
  bool use_mms_;
  std::string mms_name_;
  bool mmsCompareRhs_ = false;   // used for utils/compare_rhs
  bool mmsSaveDetails_ = false;  // used for utils/compare_rhs

  DryAirInput dryAirInput;
  PerfectMixtureInput perfectMixtureInput;
  LteMixtureInput lteMixtureInput;
  ArgonTransportInput argonTransportInput;
  ChemistryInput chemistryInput;
  RadiationInput radiationInput;

  double const_plasma_conductivity_;

  // mesh periodicitity
  bool periodic;
  double xTrans, yTrans, zTrans;

  // the wall BC values are a mess and buried
  double wallTemperature;

  PostProcessInput postprocessInput;

  bool compute_distance;
  bool read_distance;

  RunConfiguration();
  ~RunConfiguration();

  bool GetPeriodic() { return periodic; }
  double GetXTrans() { return xTrans; }
  double GetYTrans() { return yTrans; }
  double GetZTrans() { return zTrans; }
  double GetWallTemp() { return wallTemperature; }

  string GetMeshFileName() { return meshFile; }
  string GetOutputName() { return io_opts_.output_dir_; }
  string GetPartitionBaseName() const { return partFile; }
  int GetUniformRefLevels() { return ref_levels; }

  int GetTimeIntegratorType() { return timeIntegratorType; }
  int GetSgsModelType() { return sgsModelType; }

  int GetSolutionOrder() { return solOrder; }
  int GetIntegrationRule() { return integrationRule; }
  int GetBasisType() { return basisType; }

  double GetCFLNumber() { return cflNum; }
  bool isTimeStepConstant() const { return constantTimeStep; }
  double GetFixedDT() const { return dt_fixed; }
  int GetNumIters() { return numIters; }
  int GetNumItersOutput() { return itersOut; }
  bool RoeRiemannSolver() const { return useRoe; }

  int GetMeanStartIter() { return avg_opts_.step_start_mean_; }
  int GetMeanSampleInterval() { return avg_opts_.sample_interval_; }
  bool GetRestartMean() { return avg_opts_.enable_mean_continuation_; }
  bool isMeanHistEnabled() { return avg_opts_.save_mean_history_; }

  WorkingFluid GetWorkingFluid() { return workFluid; }
  double GetViscMult() { return visc_mult; }
  double GetBulkViscMult() { return bulk_visc; }
  double GetReferenceLength() { return refLength; }
  double GetSgsFloor() { return sgsFloor; }
  double GetSgsConstant() { return sgs_model_const; }
  Equations GetEquationSystem() const { return eqSystem; }
  bool isAxisymmetric() const { return axisymmetric_; }
  double* GetConstantInitialCondition() { return &initRhoRhoVp[0]; }

  linearlyVaryingVisc& GetLinearVaryingData() { return linViscData; }
  SpongeZoneData& GetSpongeZoneData(int sz) { return spongeData_[sz]; }

  // resource manager controls
  bool isAutoRestart() { return rm_enableMonitor_; }
  int rm_threshold() { return rm_threshold_; }
  int rm_checkFreq() { return rm_checkFrequency_; }

  int exit_checkFreq() { return io_opts_.exit_check_frequency_; }

  bool thereIsForcing() { return isForcing; }
  double* GetImposedPressureGradient() { return &gradPress[0]; }

  int GetRestartCycle() { return io_opts_.enable_restart_; }
  void SetRestartCycle(int iter) {
    restart_cycle = iter;
    return;
  }
  bool RestartFromAux() { return io_opts_.restart_variable_order_; }
  bool RestartHDFConversion() { return restart_hdf5_conversion; }
  bool isRestartSerialized(string mode);
  bool isRestartPartitioned(string mode);

  const std::vector<pair<int, InletType>>* GetInletPatchType() const { return &inletPatchType; }
  Array<double> GetInletData(int i);

  const std::vector<pair<int, OutletType>>* GetOutletPatchType() const { return &outletPatchType; }
  Array<double> GetOutletData(int out);

  std::vector<pair<int, WallType>>* GetWallPatchType() { return &wallPatchType; }
  WallData GetWallData(int w);

  Array<passiveScalarData*>& GetPassiveScalarData() { return arrayPassiveScalar; }
  passiveScalarData* GetPassiveScalarData(int i) { return arrayPassiveScalar[i]; }

  int GetNumSpecies() { return numSpecies; }
  double GetGasParams(int species, GasParams param) { return gasParams(species, param); }
  bool IsAmbipolar() { return ambipolar; }
  bool IsTwoTemperature() { return twoTemperature; }
  GasModel GetGasModel() { return gasModel; }
  TransportModel GetTranportModel() { return transportModel; }
  ChemistryModel GetChemistryModel() { return chemistryModel_; }
  double getConstantMolarCV(int species) { return constantMolarCV(species); }
  double getConstantMolarCP(int species) { return constantMolarCP(species); }
};

#endif  // RUN_CONFIGURATION_HPP_
