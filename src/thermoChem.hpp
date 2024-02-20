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

#ifndef THERMOCHEM_HPP_
#define THERMOCHEM_HPP_

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

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;

/**
   Energy/species class specific for temperature solves with loMach flows
 */
class ThermoChem : public ThermoChemModelBase {
 private:
  // Options-related structures
  TPS::Tps *tpsP_ = nullptr;

  // Mesh and discretization scheme info
  ParMesh *pmesh_ = nullptr;
  int order_;
  IntegrationRules gll_rules_;
  const temporalSchemeCoefficients &timeCoeff_;
  double dt_;
  double time_;

  // Flags
  bool rank0_;  // flag to indicate rank 0

  /// Enable/disable partial assembly of forms.
  bool partial_assembly_ = false;

  /// Enable/disable numerical integration rules of forms.
  bool numerical_integ_ = true;

  // loMach options to run as incompressible
  bool constant_viscosity_ = false;
  bool constant_density_ = false;

  bool domain_is_open_ = false;

  // Linear-solver-related options

  // Print levels.
  int pl_solve_ = 0;

  // max iterations
  int max_iter_;

  // Relative tolerances.
  double rtol_ = 1e-12;

  // Boundary condition info

  // All essential attributes.
  Array<int> temp_ess_attr_;
  Array<int> Qt_ess_attr_;

  // All essential true dofs.
  Array<int> temp_ess_tdof_;
  Array<int> Qt_ess_tdof_;

  // Bookkeeping for temperature dirichlet bcs.
  std::vector<DirichletBC_T<Coefficient>> temp_dbcs_;

  // Bookkeeping for Qt dirichlet bcs.
  std::vector<DirichletBC_T<Coefficient>> Qt_dbcs_;

  // Scalar modeling parameters

  // transport parameters
  double mu0_;
  double sutherland_T0_;
  double sutherland_S0_;

  double Pr_, Cp_, gamma_, Rgas_;
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

  // Fields
  ParGridFunction Tnm1_gf_, Tnm2_gf_;
  ParGridFunction Tn_gf_, Tn_next_gf_, Text_gf_, resT_gf_;
  ParGridFunction rn_gf_;
  ParGridFunction rhoDt;

  ParGridFunction viscTotal_gf_;
  ParGridFunction visc_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction R0PM0_gf_;
  ParGridFunction Qt_gf_;

  VectorGridFunctionCoefficient *un_next_coeff = nullptr;
  GridFunctionCoefficient *rhon_next_coeff = nullptr;
  ScalarVectorProductCoefficient *rhou_coeff = nullptr;
  GridFunctionCoefficient *thermal_diff_coeff = nullptr;
  GradientGridFunctionCoefficient *gradT_coeff = nullptr;
  ScalarVectorProductCoefficient *kap_gradT_coeff = nullptr;

  ConstantCoefficient *buffer_qbc = nullptr;
  GridFunctionCoefficient *rhoDtField = nullptr;
  GridFunctionCoefficient *Rho = nullptr;

  // operators and solvers
  ParBilinearForm *At_form = nullptr;
  ParBilinearForm *Ms_form = nullptr;
  ParBilinearForm *MsRho_form = nullptr;
  ParBilinearForm *Ht_form = nullptr;
  ParBilinearForm *Mq_form = nullptr;
  ParBilinearForm *LQ_form = nullptr;
  ParLinearForm *LQ_bdry = nullptr;

  OperatorHandle LQ;
  OperatorHandle At;
  OperatorHandle Ht;
  OperatorHandle Ms;
  OperatorHandle MsRho;
  OperatorHandle Mq;

  mfem::Solver *MsInvPC = nullptr;
  mfem::CGSolver *MsInv = nullptr;
  mfem::Solver *MqInvPC = nullptr;
  mfem::CGSolver *MqInv = nullptr;
  mfem::Solver *HtInvPC = nullptr;
  mfem::CGSolver *HtInv = nullptr;

  // Vectors
  Vector fTn, Tn, Tn_next, Tnm1, Tnm2, NTn, NTnm1, NTnm2;
  Vector Text;
  Vector resT, tmpR0a, tmpR0b, tmpR0c;

  Vector tmpR0;

  Vector Qt;
  Vector rn;
  Vector kappa;
  Vector visc;

  // Parameters and objects used in filter-based stabilization
  bool filterTemp = false;
  int filter_cutoff_modes = 0;
  double filter_alpha = 0.0;

  FiniteElementCollection *sfec_filter = nullptr;
  ParFiniteElementSpace *sfes_filter = nullptr;
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
  ThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff, TPS::Tps *tps);
  virtual ~ThermoChem();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final;
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;

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

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentTemperature() { return &Tn_gf_; }

  /// Return a pointer to the current density ParGridFunction.
  ParGridFunction *GetCurrentDensity() { return &rn_gf_; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  ParGridFunction *GetCurrentTotalViscosity() { return &viscTotal_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentTotalThermalDiffusivity() { return &kappa_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiv() { return &Qt_gf_; }

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
  void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);
};
#endif  // THERMOCHEM_HPP_
