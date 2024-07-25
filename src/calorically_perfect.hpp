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

#ifndef CALORICALLY_PERFECT_HPP_
#define CALORICALLY_PERFECT_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "averaging.hpp"
#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "thermo_chem_base.hpp"
#include "tps_mfem_wrap.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;

/**
   Energy/species class specific for temperature solves with loMach flows
 */
class CaloricallyPerfectThermoChem : public ThermoChemModelBase {
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

  // Flags
  bool rank0_;                      /**< true if this is rank 0 */
  bool partial_assembly_ = false;   /**< Enable/disable partial assembly of forms. */
  bool numerical_integ_ = false;    /**< Enable/disable numerical integration rules of forms. */
  bool constant_viscosity_ = false; /**< Enable/disable constant viscosity */
  bool constant_density_ = false;   /**< Enable/disable constant density */
  bool domain_is_open_ = false;     /**< true if domain is open */

  // Linear-solver-related options
  int smoother_poly_order_;
  double smoother_poly_fraction_ = 0.75;
  int smoother_eig_est_ = 10;
  int smoother_passes_ = 1;
  double smoother_relax_weight_ = 0.4;
  double smoother_relax_omega_ = 1.0;
  double hsmoother_relax_weight_ = 0.8;
  double hsmoother_relax_omega_ = 0.1;

  // solver tolerance options
  int pl_solve_; /**< Verbosity level passed to mfem solvers */
  int max_iter_; /**< Maximum number of linear solver iterations */
  double rtol_;  /**< Linear solver relative tolerance */

  int default_max_iter_ = 1000;
  double default_rtol_ = 1.0e-10;
  double default_atol_ = 1.0e-12;

  int mass_inverse_pl_ = 0;
  int mass_inverse_max_iter_;
  double mass_inverse_rtol_;
  double mass_inverse_atol_;

  int hsolve_pl_ = 0;
  int hsolve_max_iter_;
  double hsolve_rtol_;
  double hsolve_atol_;

  // streamwise-stabilization
  bool sw_stab_;
  double re_offset_;
  double re_factor_;

  // Boundary condition info
  Array<int> temp_ess_attr_; /**< List of patches with Dirichlet BC on temperature */
  Array<int> Qt_ess_attr_;   /**< List of patches with Dirichlet BC on Q (thermal divergence) */

  Array<int> temp_ess_tdof_; /**< List of true dofs with Dirichlet BC on temperature */
  Array<int> Qt_ess_tdof_;   /**< List of true dofs with Dirichlet BC on Q */

  std::vector<DirichletBC_T<Coefficient>> temp_dbcs_; /**< vector of Dirichlet BC coefficients for T*/
  std::vector<DirichletBC_T<Coefficient>> Qt_dbcs_;   /**< vector of Dirichlet BC coefficients for Q*/

  // Scalar modeling parameters
  double mu0_;           /**< Dynamic viscosity, either multiplier for Sutherland or constant */
  double sutherland_T0_; /**< Temperature constant for Sutherland's law */
  double sutherland_S0_; /**< S constant for Sutherland's law */

  double Pr_, invPr_, Cp_, gamma_, Rgas_;
  double static_rho_;

  /// pressure-related, closed-system thermo pressure changes
  double ambient_pressure_, system_mass_;
  // double thermo_pressure_;
  double dtP_;

  // Initial temperature value (if constant IC)
  double T_ic_;

  // FEM related fields and objects

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Vector fe collection and space
  FiniteElementCollection *vfec_ = nullptr;
  ParFiniteElementSpace *vfes_ = nullptr;

  // Fields
  ParGridFunction Tnm1_gf_, Tnm2_gf_;
  ParGridFunction Tn_gf_, Tn_next_gf_, Text_gf_, resT_gf_;
  ParGridFunction rn_gf_;
  ParGridFunction rhoDt;

  ParGridFunction visc_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction R0PM0_gf_;
  ParGridFunction Qt_gf_;

  ParGridFunction tmpR0_gf_;
  ParGridFunction tmpR1_gf_;
  ParGridFunction vel_gf_;
  ParGridFunction *gridScale_gf_ = nullptr;

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

  // operators and solvers
  ParBilinearForm *At_form_ = nullptr;
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *Mv_form_ = nullptr;
  ParBilinearForm *MsRho_form_ = nullptr;
  ParBilinearForm *Ht_form_ = nullptr;
  ParBilinearForm *Mq_form_ = nullptr;
  ParBilinearForm *LQ_form_ = nullptr;
  ParMixedBilinearForm *D_form_ = nullptr;
  ParMixedBilinearForm *G_form_ = nullptr;
  ParLinearForm *LQ_bdry_ = nullptr;

  OperatorHandle LQ_;
  OperatorHandle At_;
  OperatorHandle Ht_;
  OperatorHandle Ms_;
  OperatorHandle MsRho_;
  OperatorHandle Mq_;
  OperatorHandle Mv_;
  OperatorHandle D_op_;
  OperatorHandle G_op_;

  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *MqInvPC_ = nullptr;
  mfem::CGSolver *MqInv_ = nullptr;
  mfem::Solver *HtInvPC_ = nullptr;
  mfem::CGSolver *HtInv_ = nullptr;
  mfem::Solver *Mv_inv_pc_ = nullptr;
  mfem::CGSolver *Mv_inv_ = nullptr;

  // Vectors
  Vector Tn_, Tn_next_, Tnm1_, Tnm2_;
  Vector NTn_, NTnm1_, NTnm2_;
  Vector Text_;
  Vector resT_;
  Vector tmpR0_, tmpR0a_, tmpR0b_, tmpR0c_;
  Vector tmpR1_;
  Vector swDiff_;
  Vector gradT_;

  Vector Qt_;
  Vector rn_;
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

#if 0
  // TODO(trevilo): Re-enable variable inlet BC
  ParGridFunction *buffer_tInlet = nullptr;
  ParGridFunction *buffer_tInletInf = nullptr;
  GridFunctionCoefficient *tInletField = nullptr;

  // TODO(trevilo): Re-enable the viscosity multiplier functionality
  ParGridFunction viscMult_gf;
  viscositySpongeData vsd_;

  void interpolateInlet();
  void uniformInlet();
  void viscSpongePlanar(double *x, double &wgt);
#endif

 public:
  CaloricallyPerfectThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff,
                               ParGridFunction *gridScale, TPS::Tps *tps);
  virtual ~CaloricallyPerfectThermoChem();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void initializeStats(Averaging &average, IODataOrganizer &io, bool continuation) final;

  void screenHeader(std::vector<std::string> &header) const final;
  void screenValues(std::vector<double> &values) final;

  // Functions added here
  void updateThermoP();
  void extrapolateState();
  void updateDensity(double tStep);
  void updateBC(int current_step);
  void updateDiffusivity();
  void computeSystemMass();
  void computeExplicitTempConvectionOP(bool extrap);
  void computeQt();
  void computeQtTO();
  //  void streamwiseDiffusion(Vector &phi, Vector &swDiff);
  void streamwiseDiffusion(Vector &gradPhi, Vector &swDiff);

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

  /// Return thermodynamic pressure for restarts
  // double GetCurrentThermoPressure() { return thermo_pressure_; }
  // double SetCurrentThermoPressure(double Po) { thermo_pressure_ = Po;}

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTempDirichletBC(const double &temp, Array<int> &attr);
  void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
  void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);
};
#endif  // CALORICALLY_PERFECT_HPP_
