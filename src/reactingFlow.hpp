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

#ifndef REACTINGFLOW_HPP_
#define REACTINGFLOW_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "thermo_chem_base.hpp"
#include "tps_mfem_wrap.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "argon_transport.hpp"
#include "chemistry.hpp"


using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;

/**
   Energy/species class specific for temperature solves with loMach flows
 */
class ReactingFlow : public ThermoChemModelBase {
 private:
  // Options-related structures
  TPS::Tps *tpsP_ = nullptr;

  // Mesh and discretization scheme info
  ParMesh *pmesh_ = nullptr;
  int dim_;
  int order_;
  IntegrationRules gll_rules_;
  const temporalSchemeCoefficients &time_coeff_;
  double dt_;
  double time_;

  // number of species and dofs
  int nSpecies_, nActiveSpecies_, nReactions_, nAtoms_;  
  int sDof_, sDofInt_;
  int yDof_, yDofInt_;  

  WorkingFluid workFluid_;
  GasModel gasModel_;
  TransportModel transportModel_;
  ChemistryModel chemistryModel_;
  
  // packaged inputs
  PerfectMixtureInput mixtureInput_;
  ArgonTransportInput argonInput_;
  ChemistryInput chemistryInput_;
  
  GasMixture *mixture_ = NULL;
  // TransportProperties *transport_ = NULL;
  ArgonMixtureTransport *transport_ = NULL;
  Chemistry *chemistry_ = NULL;

  std::vector<std::string> speciesNames_;
  std::map<std::string, int> atomMap_;
  DenseMatrix speciesComposition_;
  DenseMatrix gasParams_;
  //double gasParams_[gpudata::MAXSPECIES * GasParams::NUM_GASPARAMS];  
  double const_plasma_conductivity_;
  
  // Flags
  bool rank0_;                      /**< true if this is rank 0 */
  bool partial_assembly_ = false;   /**< Enable/disable partial assembly of forms. */
  bool numerical_integ_ = true;     /**< Enable/disable numerical integration rules of forms. */
  bool constant_viscosity_ = false; /**< Enable/disable constant viscosity */
  bool constant_density_ = false;   /**< Enable/disable constant density */
  bool domain_is_open_ = false;     /**< true if domain is open */

  // Linear-solver-related options
  int pl_solve_ = 0;    /**< Verbosity level passed to mfem solvers */
  int max_iter_;        /**< Maximum number of linear solver iterations */
  double rtol_ = 1e-12; /**< Linear solver relative tolerance */

  // Boundary condition info
  Array<int> temp_ess_attr_; /**< List of patches with Dirichlet BC on temperature */
  Array<int> Qt_ess_attr_;   /**< List of patches with Dirichlet BC on Q (thermal divergence) */
  Array<int> spec_ess_attr_; /**< List of patches with Dirichlet BC on species */  

  Array<int> temp_ess_tdof_; /**< List of true dofs with Dirichlet BC on temperature */
  Array<int> Qt_ess_tdof_;   /**< List of true dofs with Dirichlet BC on Q */
  Array<int> spec_ess_tdof_; /**< List of true dofs with Dirichlet BC on species */  

  std::vector<DirichletBC_T<Coefficient>> temp_dbcs_; /**< vector of Dirichlet BC coefficients for T*/
  std::vector<DirichletBC_T<Coefficient>> Qt_dbcs_;   /**< vector of Dirichlet BC coefficients for Q*/
  std::vector<DirichletBC_T<Coefficient>> spec_dbcs_; /**< vector of Dirichlet BC coefficients for Yn*/  

  // Scalar modeling parameters
  double mu0_;           /**< Dynamic viscosity, either multiplier for Sutherland or constant */
  double sutherland_T0_; /**< Temperature constant for Sutherland's law */
  double sutherland_S0_; /**< S constant for Sutherland's law */

  double Pr_, invPr_, Cp_, gamma_, Rgas_;
  double Sc_, invSc_;
  double static_rho_;

  /// pressure-related, closed-system thermo pressure changes
  double ambient_pressure_, thermo_pressure_, system_mass_;
  double dtP_;

  // Initial temperature value (if constant IC)
  double T_ic_;

  // FEM related fields and objects

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Species \f$H^1\f$ finite element collection.
  FiniteElementCollection *yfec_ = nullptr;

  // Species \f$H^1\f$ finite element space.
  ParFiniteElementSpace *yfes_ = nullptr;  

  // Fields
  ParGridFunction Tnm1_gf_, Tnm2_gf_;
  ParGridFunction Tn_gf_, Tn_next_gf_, Text_gf_, resT_gf_;
  ParGridFunction rn_gf_;
  ParGridFunction rhoDt;

  // additions for species
  ParGridFunction Ynm1_gf_, Ynm2_gf_;
  ParGridFunction Yn_gf_, Yn_next_gf_, Yext_gf_, resY_gf_;
  ParGridFunction prodY_gf_;  
  ParGridFunction YnFull_gf_;
  ParGridFunction CpY_gf_;
  ParGridFunction Rmix_gf_;
  ParGridFunction Mmix_gf_;  

  ParGridFunction visc_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction diffY_gf_;  
  ParGridFunction R0PM0_gf_;
  ParGridFunction Qt_gf_;

  // ParGridFunction *buffer_tInlet_ = nullptr;
  GridFunctionCoefficient *temperature_bc_field_ = nullptr;

  VectorGridFunctionCoefficient *un_next_coeff_ = nullptr;
  GridFunctionCoefficient *rhon_next_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rhou_coeff_ = nullptr;
  GridFunctionCoefficient *thermal_diff_coeff_ = nullptr;
  GridFunctionCoefficient *mut_coeff_ = nullptr;
  GridFunctionCoefficient *mult_coeff_ = nullptr;
  SumCoefficient *thermal_diff_sum_coeff_ = nullptr;
  ProductCoefficient *thermal_diff_total_coeff_ = nullptr;
  GradientGridFunctionCoefficient *gradT_coeff_ = nullptr;
  ScalarVectorProductCoefficient *kap_gradT_coeff_ = nullptr;
  GridFunctionCoefficient *rho_over_dt_coeff_ = nullptr;
  GridFunctionCoefficient *rho_coeff_ = nullptr;

  GridFunctionCoefficient *species_diff_coeff_ = nullptr;
  SumCoefficient *species_diff_sum_coeff_ = nullptr;
  ProductCoefficient *species_diff_total_coeff_ = nullptr;

  // ConstantCoefficient *species_Cp_coeff_ = nullptr;
  GridFunctionCoefficient *species_Cp_coeff_ = nullptr;      
  ProductCoefficient *species_diff_Cp_coeff_ = nullptr;    

  // operators and solvers
  ParBilinearForm *At_form_ = nullptr;
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *MsRho_form_ = nullptr;
  ParBilinearForm *Ht_form_ = nullptr;
  ParBilinearForm *Hy_form_ = nullptr;  
  ParBilinearForm *Mq_form_ = nullptr;
  ParBilinearForm *LQ_form_ = nullptr;
  ParLinearForm *LQ_bdry_ = nullptr;
  ParBilinearForm *LY_form_ = nullptr;  

  OperatorHandle LQ_;
  OperatorHandle LY_;  
  OperatorHandle At_;
  OperatorHandle Ht_;
  OperatorHandle Hy_;  
  OperatorHandle Ms_;
  OperatorHandle MsRho_;
  OperatorHandle Mq_;

  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *MqInvPC_ = nullptr;
  mfem::CGSolver *MqInv_ = nullptr;
  mfem::Solver *HtInvPC_ = nullptr;
  mfem::CGSolver *HtInv_ = nullptr;
  mfem::Solver *HyInvPC_ = nullptr;
  mfem::CGSolver *HyInv_ = nullptr;
  
  // Vectors
  Vector Tn_, Tn_next_, Tnm1_, Tnm2_;
  Vector NTn_, NTnm1_, NTnm2_;
  Vector Text_;
  Vector resT_;
  Vector tmpR0_;
  Vector tmpR0a_, tmpR0b_, tmpR0c_;

  // additions for species
  Vector Yn_, Yn_next_, Ynm1_, Ynm2_;
  Vector NYn_, NYnm1_, NYnm2_;
  Vector Yext_;
  Vector Xn_;  
  Vector resY_;
  Vector prodY_;
  Vector hw_;
  // Vector CpY_;
  Vector SDFT_;  
  Vector inputCV_;
  Vector initialMassFraction_;
  Vector atomMW_;   

  Vector Qt_;
  Vector rn_;
  Vector diffY_;
  Vector kappa_;
  Vector visc_;

  // Parameters and objects used in filter-based stabilization
  bool filter_temperature_ = false;
  int filter_cutoff_modes_ = 0;
  double filter_alpha_ = 0.0;

  FiniteElementCollection *sfec_filter_ = nullptr;
  ParFiniteElementSpace *sfes_filter_ = nullptr;
  ParGridFunction Tn_NM1_gf_;
  ParGridFunction Tn_filtered_gf_;

 public:
  ReactingFlow(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff,
                               TPS::Tps *tps);
  virtual ~ReactingFlow();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;

  // Functions added here
  void speciesLastStep();  
  void speciesStep(int iSpec);
  void temperatureStep();
  void updateMixture();  
  void updateThermoP();
  void extrapolateState();
  void updateDensity(double tStep);
  void updateBC(int current_step);
  void updateDiffusivity();
  void computeSystemMass();
  void computeExplicitTempConvectionOP(bool extrap);
  void computeExplicitSpecConvectionOP(int iSpec, bool extrap);  
  void computeQt();
  void computeQtTO();
  void speciesProduction();
  void heatOfFormation();
  void diffusionForTemperature();

  /// for creation of structs to interface with old plasma/chem stuff
  void identifySpeciesType(Array<ArgonSpcs> &speciesType);
  void identifyCollisionType(const Array<ArgonSpcs> &speciesType, ArgonColl *collisionIndex);
  
  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentTemperature() { return &Tn_gf_; }

  /// Return a pointer to the current density ParGridFunction.
  ParGridFunction *GetCurrentDensity() { return &rn_gf_; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  ParGridFunction *GetCurrentViscosity() { return &visc_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiffusivity() { return &kappa_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiv() { return &Qt_gf_; }

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTempDirichletBC(const double &temp, Array<int> &attr);
  void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
  void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);
};
#endif  // REACTINGFLOW_HPP_
