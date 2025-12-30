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

#ifndef LTE_THERMO_CHEM_HPP_
#define LTE_THERMO_CHEM_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "dataStructures.hpp"
#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "table.hpp"
#include "thermo_chem_base.hpp"
#include "tps_mfem_wrap.hpp"
#include "utils.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;
class Radiation;

/**
 * @brief Implementation of "local thermodynamic equilibrium"
 * thermochemistry model using table lookups
 *
 * This class supports LTE simulations coupled with the low Mach flow
 * solver.  In particular, the low Mach form of the energy equation is
 * solved, with the transport properties---specifically, mu (dynamic
 * viscosity), kappa (thermal conductivity), and sigma (electrical
 * conductivity)---and thermodynamic properties---specifically, Rgas
 * (the specific gas constant) and Cp (the specific heat at constant
 * pressure)---determine by interpolating tables provided by the user.
 * It is envisioned that these tables are generated from external
 * equilibrium calculations (for variable temperature, fixed pressure)
 * for some given mixture, but this is entirely up to the user.
 */
class LteThermoChem final : public ThermoChemModelBase {
 private:
  // Options-related structures
  TPS::Tps *tpsP_ = nullptr;

  // Mesh and discretization scheme info
  ParMesh *pmesh_ = nullptr;
  int order_;
  IntegrationRules gll_rules_;
  const temporalSchemeCoefficients &time_coeff_;

  // Flags
  bool rank0_;                    /**< true if this is rank 0 */
  bool partial_assembly_ = false; /**< Enable/disable partial assembly of forms. */
  bool numerical_integ_ = false;  /**< Enable/disable numerical integration rules of forms. */
  bool domain_is_open_ = false;   /**< true if domain is open */
  bool axisym_ = false;
  bool sw_stab_ = false; /**< Enable/disable supg stabilization. */

  // Linear-solver-related options
  int pl_solve_ = 0;    /**< Verbosity level passed to mfem solvers */
  int max_iter_;        /**< Maximum number of linear solver iterations */
  double rtol_ = 1e-12; /**< Linear solver relative tolerance */

  // Boundary condition info
  Array<int> temp_ess_attr_; /**< List of patches with Dirichlet BC on temperature */
  Array<int> Qt_ess_attr_;   /**< List of patches with Dirichlet BC on Q (thermal divergence) */

  Array<int> temp_ess_tdof_; /**< List of true dofs with Dirichlet BC on temperature */
  Array<int> Qt_ess_tdof_;   /**< List of true dofs with Dirichlet BC on Q */

  std::vector<DirichletBC_T<Coefficient>> temp_dbcs_; /**< vector of Dirichlet BC coefficients for T*/
  std::vector<DirichletBC_T<Coefficient>> Qt_dbcs_;   /**< vector of Dirichlet BC coefficients for Q*/

  LinearTable *mu_table_;     // dynamic viscosity
  LinearTable *kappa_table_;  // thermal conductivity
  LinearTable *sigma_table_;  // electrical conductivity
  LinearTable *Rgas_table_;   // specific gas constant
  LinearTable *Cp_table_;     // specific heat at constant pressure

  Radiation *radiation_ = nullptr;

  /// pressure-related, closed-system thermo pressure changes
  double ambient_pressure_, thermo_pressure_, system_mass_;
  double dtP_;

  // Initial temperature value (if constant IC)
  double T_ic_;

  double Prt_;
  double invPrt_;

  // FEM related fields and objects

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Fields
  ParGridFunction Tnm1_gf_, Tnm2_gf_;
  ParGridFunction Tn_gf_, Tn_next_gf_, Text_gf_, resT_gf_;
  ParGridFunction rn_gf_;

  ParGridFunction mu_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction sigma_gf_;
  ParGridFunction jh_gf_;
  ParGridFunction Rgas_gf_;
  ParGridFunction Cp_gf_;
  ParGridFunction radiation_sink_gf_;
  ParGridFunction thermal_diff_total_gf_;

  ParGridFunction R0PM0_gf_;
  ParGridFunction Qt_gf_;

  ParGridFunction *gridScale_gf_ = nullptr;

  // ParGridFunction *buffer_tInlet_ = nullptr;
  GridFunctionCoefficient *temperature_bc_field_ = nullptr;

  ConstantCoefficient bd0_over_dt;
  VectorGridFunctionCoefficient *un_next_coeff_ = nullptr;
  GridFunctionCoefficient *rhon_next_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rho_Cp_u_coeff_ = nullptr;
  GridFunctionCoefficient *thermal_diff_coeff_ = nullptr;
  GridFunctionCoefficient *mut_coeff_ = nullptr;
  ProductCoefficient *kapt_coeff_ = nullptr;
  GridFunctionCoefficient *mult_coeff_ = nullptr;
  SumCoefficient *thermal_diff_sum_coeff_ = nullptr;
  ProductCoefficient *thermal_diff_total_coeff_ = nullptr;
  GradientGridFunctionCoefficient *gradT_coeff_ = nullptr;
  ScalarVectorProductCoefficient *kap_gradT_coeff_ = nullptr;
  GridFunctionCoefficient *rho_coeff_ = nullptr;
  GridFunctionCoefficient *Cp_coeff_ = nullptr;
  ProductCoefficient *rho_Cp_over_dt_coeff_ = nullptr;
  ProductCoefficient *rho_Cp_coeff_ = nullptr;
  GridFunctionCoefficient *jh_coeff_ = nullptr;
  GridFunctionCoefficient *radiation_sink_coeff_ = nullptr;

  ProductCoefficient *rad_rho_coeff_ = nullptr;
  ProductCoefficient *rad_rho_Cp_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_rho_Cp_u_coeff_ = nullptr;
  ProductCoefficient *rad_rho_Cp_over_dt_coeff_ = nullptr;
  ProductCoefficient *rad_thermal_diff_total_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_un_next_coeff_ = nullptr;
  ProductCoefficient *rad_jh_coeff_ = nullptr;
  ProductCoefficient *rad_radiation_sink_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_kap_gradT_coeff_ = nullptr;

  VectorMagnitudeCoefficient *umag_coeff_ = nullptr;
  GridFunctionCoefficient *gscale_coeff_ = nullptr;
  GridFunctionCoefficient *visc_coeff_ = nullptr;
  PowerCoefficient *visc_inv_coeff_ = nullptr;
  ProductCoefficient *reh1_coeff_ = nullptr;
  ProductCoefficient *reh2_coeff_ = nullptr;
  ProductCoefficient *Reh_coeff_ = nullptr;
  TransformedCoefficient *csupg_coeff_ = nullptr;
  ProductCoefficient *uw1_coeff_ = nullptr;
  ProductCoefficient *uw2_coeff_ = nullptr;
  ProductCoefficient *upwind_coeff_ = nullptr;
  TransformedMatrixVectorCoefficient *swdiff_coeff_ = nullptr;
  ScalarMatrixProductCoefficient *supg_coeff_ = nullptr;

  // operators and solvers
  ParBilinearForm *At_form_ = nullptr;
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *M_rho_Cp_form_ = nullptr;
  ParBilinearForm *Ht_form_ = nullptr;

  ParBilinearForm *M_rho_form_ = nullptr;
  ParBilinearForm *A_rho_form_ = nullptr;

  ParLinearForm *jh_form_ = nullptr;

  ParBilinearForm *Mq_form_ = nullptr;
  ParBilinearForm *LQ_form_ = nullptr;
  ParLinearForm *LQ_bdry_ = nullptr;

  OperatorHandle At_;
  OperatorHandle Ht_;
  OperatorHandle Ms_;
  OperatorHandle Mq_;
  OperatorHandle LQ_;
  OperatorHandle M_rho_Cp_;
  OperatorHandle M_rho_;
  OperatorHandle A_rho_;

  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *MqInvPC_ = nullptr;
  mfem::CGSolver *MqInv_ = nullptr;
  mfem::Solver *MrhoInvPC_ = nullptr;
  mfem::CGSolver *MrhoInv_ = nullptr;
  mfem::Solver *HtInvPC_ = nullptr;
  mfem::CGSolver *HtInv_ = nullptr;

  // Vectors
  Vector Tn_, Tn_next_, Tnm1_, Tnm2_;
  Vector NTn_, NTnm1_, NTnm2_;
  Vector Text_;
  Vector resT_;
  Vector tmpR0_, tmpR0b_;

  Vector Qt_;
  Vector rn_, rnm1_, rnm2_, rnm3_;
  Vector kappa_;
  Vector visc_;
  Vector sigma_;
  Vector Rgas_;
  Vector Cp_;
  Vector jh_;
  Vector radiation_sink_;

  // Parameters and objects used in filter-based stabilization
  bool filter_temperature_ = false;
  int filter_cutoff_modes_ = 0;
  double filter_alpha_ = 0.0;

  FiniteElementCollection *sfec_filter_ = nullptr;
  ParFiniteElementSpace *sfes_filter_ = nullptr;
  ParGridFunction Tn_NM1_gf_;
  ParGridFunction Tn_filtered_gf_;

 public:
  LteThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff,
                ParGridFunction *gridScale, TPS::Tps *tps);
  virtual ~LteThermoChem();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;  
  
  void step() final;
  void massImbalanceStep() final {};
  void pressureStep() final {};
  void densityPredictionStep() final {};
  void temperatureStep() final {};
  void densityStep() final {};
  
  void evaluatePlasmaConductivityGF() final;

  // Functions added here
  void updateThermoP();
  void extrapolateTemperature();
  void updateDensity();
  void updateProperties();
  void computeSystemMass();
  void computeExplicitTempConvectionOP();
  void computeQt();
  void updateHistory();

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentTemperature() { return &Tn_gf_; }

  /// Return a pointer to the current density ParGridFunction.
  ParGridFunction *GetCurrentDensity() { return &rn_gf_; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  ParGridFunction *GetCurrentViscosity() { return &mu_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiffusivity() { return &kappa_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiv() { return &Qt_gf_; }

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTempDirichletBC(const double &temp, Array<int> &attr);
  void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
  void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);
};
#endif  // LTE_THERMO_CHEM_HPP_
