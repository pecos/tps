// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

#include <mfem.hpp>
#include <string>
#include <utility>
#include <vector>

#include "dataStructures.hpp"

using namespace mfem;
using namespace std;

// Class to manage the run options as
// specified in the input file
// Lines begining with # are ignored
class RunConfiguration {
 public:
  // mesh file file
  string meshFile;
  int ref_levels;

  // partition file
  string partFile;

  // output file name
  string outputFile;

  // time integrator. Possible values
  //  1: ForwardEulerSolver
  //  2: RK2Solver(1.0)
  //  3: RK3SSPSolver
  //  4: RK4Solver
  //  6: RK6Solver
  int timeIntegratorType;

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

  // num iterations. Defaults to 0
  int numIters;

  // iteration frequency to provide iteration timing info
  int timingFreq;

  // output interval in num. of iters.
  int itersOut;

  // Use Roe-type Riemann solver
  bool useRoe;

  // restart controls
  bool restart;
  bool restart_hdf5_conversion;  // read in older ascii format
  std::string restart_serial;    // mode for serial restarts
  int restart_cycle;

  // Restart from different order solution on same mesh.  New order
  // set by POL_ORDER, old order determined from restart file.
  bool restartFromAux;
  bool singleRestartFile;

  // mean and RMS
  int sampleInterval;
  int startIter;
  bool restartMean;
  bool meanHistEnable;

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

  SpongeZoneData* spongeData_;
  int numSpongeRegions_;
  void initSpongeData();

  int numHeatSources;
  heatSourceData* heatSource;

  // Reference length
  double refLength;

  // equations to be solved. Options thus far
  Equations eqSystem;
  bool axisymmetric_;

  bool SBP;

  // initial constant field
  double initRhoRhoVp[5];

  bool isForcing;
  // Imposed pressure gradient
  double gradPress[3];

  // Inlet BC data
  Array<double> inletBC;
  std::vector<pair<int, InletType>> inletPatchType;

  // Outlet BC data
  Array<double> outletBC;
  std::vector<pair<int, OutletType>> outletPatchType;

  // Wall BC data
  // Array<double> wallBC;
  std::vector<wallData> wallBC;
  std::vector<pair<int, WallType>> wallPatchType;

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
  std::vector<std::vector<double>> reactionModelParams;
  DenseMatrix reactantStoich, productStoich;
  Array<bool> detailedBalance;
  std::vector<std::vector<double>> equilibriumConstantParams;

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

  // data structure for constant transport property class
  constantTransportData constantTransport;

  // Use MMS
  bool use_mms_;
  std::string mms_name_;
  bool mmsCompareRhs_ = false;  // used for utils/compare_rhs
  bool mmsSaveDetails_ = false;  // used for utils/compare_rhs

  RunConfiguration();
  ~RunConfiguration();

  void readInputFile(std::string inpuFileName);

  string GetMeshFileName() { return meshFile; }
  string GetOutputName() { return outputFile; }
  string GetPartitionBaseName() { return partFile; }
  int GetUniformRefLevels() { return ref_levels; }

  int GetTimeIntegratorType() { return timeIntegratorType; }

  int GetSolutionOrder() { return solOrder; }
  int GetIntegrationRule() { return integrationRule; }
  int GetBasisType() { return basisType; }

  double GetCFLNumber() { return cflNum; }
  bool isTimeStepConstant() const { return constantTimeStep; }
  double GetFixedDT() const { return dt_fixed; }
  int GetNumIters() { return numIters; }
  int GetNumItersOutput() { return itersOut; }
  bool RoeRiemannSolver() const { return useRoe; }

  int GetMeanStartIter() { return startIter; }
  int GetMeanSampleInterval() { return sampleInterval; }
  bool GetRestartMean() { return restartMean; }
  bool isMeanHistEnabled() { return meanHistEnable; }

  WorkingFluid GetWorkingFluid() { return workFluid; }
  double GetViscMult() { return visc_mult; }
  double GetBulkViscMult() { return bulk_visc; }
  double GetReferenceLength() { return refLength; }
  Equations GetEquationSystem() const { return eqSystem; }
  bool isAxisymmetric() const { return axisymmetric_; }
  bool isSBP() { return SBP; }
  double* GetConstantInitialCondition() { return &initRhoRhoVp[0]; }

  linearlyVaryingVisc& GetLinearVaryingData() { return linViscData; }
  SpongeZoneData& GetSpongeZoneData(int sz) { return spongeData_[sz]; }

  // resource manager controls
  bool isAutoRestart() { return rm_enableMonitor_; }
  int rm_threshold() { return rm_threshold_; }
  int rm_checkFreq() { return rm_checkFrequency_; }

  int exit_checkFreq() { return exit_checkFrequency_; }

  bool thereIsForcing() { return isForcing; }
  double* GetImposedPressureGradient() { return &gradPress[0]; }

  int GetRestartCycle() { return restart; }
  void SetRestartCycle(int iter) {
    restart_cycle = iter;
    return;
  }
  bool RestartFromAux() { return restartFromAux; }
  bool RestartHDFConversion() { return restart_hdf5_conversion; }
  std::string RestartSerial() { return restart_serial; }

  const std::vector<pair<int, InletType>>* GetInletPatchType() const { return &inletPatchType; }
  Array<double> GetInletData(int i);

  const std::vector<pair<int, OutletType>>* GetOutletPatchType() const { return &outletPatchType; }
  Array<double> GetOutletData(int out);

  std::vector<pair<int, WallType>>* GetWallPatchType() { return &wallPatchType; }
  wallData GetWallData(int w);

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
