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
class Tps2Boltzmann;
}

#include <hdf5.h>
#include <tps_config.h>

#include <fstream>
#include <iostream>

#include "argon_transport.hpp"
#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "dirichlet_bc_helper.hpp"
#include "equation_of_state.hpp"
#include "gpu_constructor.hpp"
#include "io.hpp"
#include "mpi_groups.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "utils.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;
class Radiation;

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

  std::string ic_string_;

  // number of species and dofs
  int nSpecies_, nActiveSpecies_, nReactions_, nAtoms_;
  int sDof_, sDofInt_;
  int yDof_, yDofInt_;

  #ifdef HAVE_PYTHON
  int nBTEReactions_;
  #endif

  // Number of reactions and dofs
  int rDof_, rDofInt_;

  WorkingFluid workFluid_;
  GasModel gasModel_;
  TransportModel transportModel_;
  ChemistryModel chemistryModel_;
  Radiation *radiation_ = nullptr;

  // packaged inputs
  PerfectMixtureInput mixtureInput_;
  ArgonTransportInput argonInput_;
  ChemistryInput chemistryInput_;

  PerfectMixture *mixture_ = NULL;
  ArgonMixtureTransport *transport_ = NULL;
  Chemistry *chemistry_ = NULL;
  // External reaction rates when chemistry is implemented using the BTE option
  std::unique_ptr<ParGridFunction> externalReactionRates_gf_;  // Has repeated interface dofs.
  Vector externalReactionRates_;  // Only true data

  std::vector<std::string> speciesNames_;
  std::map<std::string, int> atomMap_;
  DenseMatrix speciesComposition_;
  DenseMatrix gasParams_;
  double const_plasma_conductivity_;
  bool radiative_decay_NECincluded_;

  // Flags
  bool rank0_;                      /**< true if this is rank 0 */
  bool partial_assembly_ = false;   /**< Enable/disable partial assembly of forms. */
  bool numerical_integ_ = false;    // true;     /**< Enable/disable numerical integration rules of forms. */
  bool constant_viscosity_ = false; /**< Enable/disable constant viscosity */
  bool constant_density_ = false;   /**< Enable/disable constant density */
  bool domain_is_open_ = false;     /**< true if domain is open */
  bool axisym_ = false;             /**< true if simulation is axisymmetric */

  #ifdef HAVE_PYTHON
  bool bte_from_tps_ = false;       /**< true if the BTE solver is called from within TPS (C++ call Python) */
  std::string collisionsFile, solver_type;       /**< string to store the path of the collisions cross-sections file, BTE solver type (will be passed to BTE) */
  int ee_collisions = 0;            /**< flag to enable electron-electron collisions in BTE solver */
  #endif

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

  // Vector \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec_ = nullptr;

  // Vector \f$H^1\f$ finite element space.
  ParFiniteElementSpace *vfes_ = nullptr;

  // Reactions \f$H^1\f$ finite element collection.
  FiniteElementCollection *rfec_ = nullptr;

  // Reactions \f$H^1\f$ finite element space.
  ParFiniteElementSpace *rfes_ = nullptr;

  // Fields
  ParGridFunction Tnm1_gf_, Tnm2_gf_;
  ParGridFunction Tn_gf_, Tn_next_gf_, Text_gf_, resT_gf_;
  ParGridFunction rn_gf_;
  ParGridFunction rhoDt_gf_;
  ParGridFunction weff_gf_;

  // additions for species
  ParGridFunction Ynm1_gf_, Ynm2_gf_;
  ParGridFunction Yn_gf_, Yn_next_gf_, Yext_gf_, resY_gf_;
  ParGridFunction prodY_gf_;
  ParGridFunction YnFull_gf_;
  ParGridFunction CpY_gf_;
  ParGridFunction CpMix_gf_;
  ParGridFunction Rmix_gf_;
  ParGridFunction Mmix_gf_;
  ParGridFunction emission_gf_;


  // additions for reaction progress rates
  ParGridFunction reacR_gf_;

  ParGridFunction visc_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction diffY_gf_;
  ParGridFunction R0PM0_gf_;
  ParGridFunction Qt_gf_;

  ParGridFunction radiation_sink_gf_;

  // Interface with EM
  ParGridFunction sigma_gf_;
  ParGridFunction jh_gf_;

#ifdef HAVE_PYTHON
  // ParGridFunctions for the real and imaginary parts of the electric field
  // We only store the magnitude (works only for axisymmetric case)
  ParGridFunction er_gf_;
  ParGridFunction ei_gf_;

  // additions for rate coefficients
  ParGridFunction kReac_gf_;
#endif

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
  GridFunctionCoefficient *cpMix_coeff_ = nullptr;
  ProductCoefficient *rhoCp_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rhouCp_coeff_ = nullptr;
  ProductCoefficient *rhoCp_over_dt_coeff_ = nullptr;

  GridFunctionCoefficient *species_diff_coeff_ = nullptr;
  SumCoefficient *species_diff_sum_coeff_ = nullptr;
  ProductCoefficient *species_diff_total_coeff_ = nullptr;

  GridFunctionCoefficient *species_Cp_coeff_ = nullptr;
  ProductCoefficient *species_diff_Cp_coeff_ = nullptr;

  GridFunctionCoefficient *jh_coeff_ = nullptr;
  GridFunctionCoefficient *radiation_sink_coeff_ = nullptr;

  ProductCoefficient *rad_rho_coeff_ = nullptr;
  ProductCoefficient *rad_rho_Cp_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_rho_u_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_rho_Cp_u_coeff_ = nullptr;
  ProductCoefficient *rad_rho_over_dt_coeff_ = nullptr;
  ProductCoefficient *rad_species_diff_total_coeff_ = nullptr;
  ProductCoefficient *rad_rho_Cp_over_dt_coeff_ = nullptr;
  ProductCoefficient *rad_thermal_diff_total_coeff_ = nullptr;
  ProductCoefficient *rad_jh_coeff_ = nullptr;
  ProductCoefficient *rad_radiation_sink_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_kap_gradT_coeff_ = nullptr;

  // operators and solvers
  ParBilinearForm *At_form_ = nullptr;
  ParBilinearForm *Ay_form_ = nullptr;
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *MsRho_form_ = nullptr;
  ParBilinearForm *MsRhoCp_form_ = nullptr;
  ParBilinearForm *Ht_form_ = nullptr;
  ParBilinearForm *Hy_form_ = nullptr;
  ParBilinearForm *Mq_form_ = nullptr;
  ParBilinearForm *LQ_form_ = nullptr;
  ParLinearForm *LQ_bdry_ = nullptr;
  ParBilinearForm *LY_form_ = nullptr;
  ParMixedBilinearForm *G_form_ = nullptr;
  ParBilinearForm *Mv_form_ = nullptr;

  ParLinearForm *jh_form_ = nullptr;

  OperatorHandle LQ_;
  OperatorHandle LY_;
  OperatorHandle At_;
  OperatorHandle Ay_;
  OperatorHandle Ht_;
  OperatorHandle Hy_;
  OperatorHandle Ms_;
  OperatorHandle MsRho_;
  OperatorHandle MsRhoCp_;
  OperatorHandle Mq_;
  OperatorHandle G_;
  OperatorHandle Mv_op_;

  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *MqInvPC_ = nullptr;
  mfem::CGSolver *MqInv_ = nullptr;
  mfem::Solver *HtInvPC_ = nullptr;
  mfem::CGSolver *HtInv_ = nullptr;
  mfem::Solver *HyInvPC_ = nullptr;
  mfem::CGSolver *HyInv_ = nullptr;
  mfem::Solver *Mv_inv_pc_ = nullptr;
  mfem::CGSolver *Mv_inv_ = nullptr;

  // Vectors
  Vector Tn_, Tn_next_, Tnm1_, Tnm2_;
  Vector NTn_, NTnm1_, NTnm2_;
  Vector Text_;
  Vector resT_;
  Vector tmpR0_;
  Vector tmpR0a_, tmpR0b_, tmpR0c_;
  Vector tmpR1_;
  Vector tmpR1a_, tmpR1b_, tmpR1c_;
  Vector jh_;
  Vector radiation_sink_;

#ifdef HAVE_PYTHON
  // Vectors for real and imaginary parts of electric field magnitude
  Vector er_, ei_;
  Vector bterates_;
  Vector bte_rr_mapping_;
  // additions for reaction rate coefficients
  Vector kReac_;
#endif

  // additions for species
  Vector Yn_, Yn_next_, Ynm1_, Ynm2_;
  Vector NYn_, NYnm1_, NYnm2_;
  Vector Yext_;
  Vector Xn_;
  Vector resY_;
  Vector prodY_;
  Vector prodE_;
  Vector hw_;
  Vector CpY_;
  Vector crossDiff_;
  Vector speciesMolarCv_;
  Vector speciesMolarCp_;
  Vector specificHeatRatios_;
  Vector initialMassFraction_;
  Vector atomMW_;
  Vector CpMix_;

  // additions for reaction progress rates
  Vector reacR_;

  Vector Qt_;
  Vector rn_;
  Vector diffY_;
  Vector kappa_;
  Vector visc_;
  Vector sigma_;

  // time-splitting
  Vector YnStar_, spec_buffer_;
  Vector TnStar_, temp_buffer_;
  bool operator_split_ = false;
  bool implicit_chemistry_ = false;
  int nSub_;
  bool dynamic_substepping_ = true;
  int stabFrac_;

  bool implicit_chemistry_verbose_ = false;
  int implicit_chemistry_maxiter_ = 200;
  double implicit_chemistry_fd_eps_ = 1e-7;
  double implicit_chemistry_rtol_ = 1e-8;
  double implicit_chemistry_atol_ = 1e-12;
  double implicit_chemistry_smin_ = 1e-12;

  // Parameters and objects used in filter-based stabilization
  bool filter_temperature_ = false;
  int filter_cutoff_modes_ = 0;
  double filter_alpha_ = 0.0;

  FiniteElementCollection *sfec_filter_ = nullptr;
  ParFiniteElementSpace *sfes_filter_ = nullptr;
  ParGridFunction Tn_NM1_gf_;
  ParGridFunction Tn_filtered_gf_;

  double Pnm1_, Pnm2_, Pnm3_;

  std::list<mfem::DenseMatrix> tableHost_;
  std::vector<ParGridFunction *> vizSpecFields_;
  std::vector<std::string> vizSpecNames_;

  std::vector<ParGridFunction *> vizProdFields_;
  std::vector<std::string> vizProdNames_;

  // PARGRID FUNCTION AND STRING FOR REACTION PROGRESS RATES
  std::vector<ParGridFunction *> vizReacFields_;
  std::vector<std::string> vizReacNames_;

#ifdef HAVE_PYTHON
  // PARGRID FUNCTION AND STRING FOR REACTION PROGRESS RATES
  std::vector<ParGridFunction *> vizkReacFields_;
  std::vector<std::string> vizkReacNames_;

  // k_blend = bl_frac * k_BTE + (1 - bl_frac)*k_LTE
  // bl_frac is the blending coefficient which is initialized as bl_frac_init
  // and incremented by bl_frac_increment after every bl_frac_change_freq steps of the main TPS solver
  // bl_frac_init -> specify in inputs file to say what is the blending fraction at start of simulation
  // This needs to be updated if you are starting a BTE blended run from a check point
  double bl_frac_init_, bl_frac_increment_, bl_frac_;
  int bl_frac_change_freq_ = 1;
  int solve_bte_every_n = 1;

  // Number of v-space grids per MPI rank for the BTE solver
  int n_vspace_grids = 1;
  int regrid_bte_every_n = 1; // call the BTE grid_setup function to setup v-space grids every nth step

  // grid_idx_to_npts : Vector of length n_vspace_grids where each element contains the number of points in that v-space grid
  // grid_idx_to_spatial_idx_map: Vector of length sDofInt_ which contains the indices of the points in each v-space grid
  // For example, if grid_idx = i contains ng_i points, grid_idx_to_spatial_idx_map[ilo:ilo+ng] contains the indices of the 
  // points in the ith v-space grid. Here, ilo_{i} = (\sum_{m=0}^{m=i} ng_m) - ng_i
  std::vector<std::int32_t> grid_idx_to_npts;
  std::vector<std::int64_t> grid_idx_to_spatial_idx_map;
#endif

 public:
  ReactingFlow(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff, TPS::Tps *tps);
  virtual ~ReactingFlow();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void evaluatePlasmaConductivityGF() final;

  // Functions added here
  void speciesLastStep();
  void speciesStep(int iSpec);
  void temperatureStep();
  void updateMixture();
  void updateThermoP();
  void extrapolateState();
  void updateDensity(double tStep, bool update_mass_matrix = true);
  void updateBC(int current_step);
  void updateDiffusivity();
  void computeSystemMass();
  void computeExplicitTempConvectionOP(bool extrap);
  void computeExplicitSpecConvectionOP(int iSpec, bool extrap);
  void computeQt();
  void computeQtTO();
  void speciesProduction();
  void heatOfFormation();
  void crossDiffusion();

  /**
   * @brief Evaluate reaction source terms at single point
   *
   * Given the mass fractions and temperature at a point, evaluate the
   * reaction source terms.  This functionality is used in building
   * the residual needed for nonlinear, implicit thermochemistry
   * within the operator split paradigm where the local-in-space terms
   * are split from the convection-diffusion terms.
   *
   * The incoming YT must contain the nActiveSpecies_ mass fractions
   * for the active species and temperature.
   */
  void evaluateReactingSource(const double *YT, const int dofindex, double *omega
  #ifdef HAVE_PYTHON
    , double *BTErr
  #endif
  );

  /**
   * @brief Solve the thermochemistry update
   *
   * Given the mass fractions and temperature at a point, the backward
   * Euler update for the local thermochemistry solve.
   *
   * The incoming YT must contain the nActiveSpecies_ mass fractions
   * for the active species and temperature.  This state is
   * overwritten with the new local state at the end of the time step.
   */
  void solveChemistryStep(double *YT, const int dofindex, const double dt
  #ifdef HAVE_PYTHON
    , double *BTErr
  #endif
  );

  // time-splitting
  void substepState();
  void speciesLastSubstep();
  void speciesSubstep(int iSpec, int iSub);
  void temperatureSubstep(int iSub);

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

  void AddSpecDirichletBC(const double &Y, Array<int> &attr);

  void evalSubstepNumber();
  void readTableWrapper(std::string inputPath, TableInput &result);

  void push(TPS::Tps2Boltzmann &interface) override;
  void fetch(TPS::Tps2Boltzmann &interface) override;
};
#endif  // REACTINGFLOW_HPP_
