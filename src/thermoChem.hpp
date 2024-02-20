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

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "argon_transport.hpp"
#include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"
#include "turb_model_base.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachSolver;
class LoMachOptions;
struct temporalSchemeCoefficients;

/**
   Energy/species class specific for temperature solves with loMach flows
 */
class ThermoChem : public ThermoChemModelBase {
  friend class LoMachSolver;

 private:
  TPS::Tps *tpsP_;
  LoMachOptions *loMach_opts_ = nullptr;

  // MPI_Groups *groupsMPI;
  int nprocs_;  // total number of MPI procs
  int rank;     // local MPI rank
  bool rank0;   // flag to indicate rank 0

  // Run options
  RunConfiguration *config_ = nullptr;

  // All essential attributes.
  Array<int> temp_ess_attr;
  Array<int> Qt_ess_attr;

  // All essential true dofs.
  Array<int> temp_ess_tdof;
  Array<int> Qt_ess_tdof;

  // Bookkeeping for temperature dirichlet bcs.
  std::vector<DirichletBC_T<Coefficient>> temp_dbcs;

  // Bookkeeping for Qt dirichlet bcs.
  std::vector<DirichletBC_T<Coefficient>> Qt_dbcs;

  // space sizes
  int Sdof, SdofInt, Pdof, PdofInt, NdofInt, NdofR0Int;

  /// Enable/disable verbose output.
  bool verbose = true;

  /// Enable/disable partial assembly of forms.
  bool partial_assembly = false;

  /// Enable/disable numerical integration rules of forms.
  bool numerical_integ = true;

  ParMesh *pmesh_ = nullptr;

  // The order of the scalar spaces
  int order, porder, norder;
  IntegrationRules gll_rules;

  // local copies of time integration information
  double bd0, bd1, bd2, bd3;
  double ab1, ab2, ab3;
  int dim;

  // just keep these saved for ease
  int numWalls, numInlets, numOutlets;

  // loMach options to run as incompressible
  bool constantViscosity = false;
  bool constantDensity = false;
  bool incompressibleSolve = false;
  bool pFilter = false;

  // transport parameters
  double mu0_;
  double sutherland_T0_;
  double sutherland_S0_;

  double dt;
  double time;
  int iter;

  // Coefficients necessary to take a time step (including dt).
  // Assumed to be externally managed and determined, so just get a
  // reference here.
  const temporalSchemeCoefficients &timeCoeff_;

  /// constant prop solves
  double Pr, Cp, gamma;
  double kin_vis, dyn_vis;
  double static_rho, Rgas;

  /// pressure-related, closed-system thermo pressure changes
  double ambientPressure, thermoPressure, systemMass;
  double tPm1, tPm2, dtP;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes = nullptr;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes = nullptr;

  /// Pressure \f$H^1\f$ finite element collection.
  FiniteElementCollection *pfec = nullptr;

  /// Pressure \f$H^1\f$ finite element space.
  ParFiniteElementSpace *pfes = nullptr;

  // operators
  DiffusionIntegrator *hdt_blfi = nullptr;
  MassIntegrator *hmt_blfi = nullptr;
  ParBilinearForm *Lt_form = nullptr;
  ParBilinearForm *At_form = nullptr;
  ParBilinearForm *Ms_form = nullptr;
  ParBilinearForm *MsRho_form = nullptr;
  ParMixedBilinearForm *Ds_form = nullptr;
  ParBilinearForm *Ht_form = nullptr;
  ParBilinearForm *Mq_form = nullptr;
  ParBilinearForm *LQ_form = nullptr;
  ParLinearForm *LQ_bdry = nullptr;
  GridFunctionCoefficient *Text_gfcoeff = nullptr;
  ParLinearForm *Text_bdr_form = nullptr;
  ParLinearForm *ft_form = nullptr;
  ParLinearForm *t_bdr_form = nullptr;

  VectorGridFunctionCoefficient *un_next_coeff = nullptr;
  GridFunctionCoefficient *rhon_next_coeff = nullptr;
  ScalarVectorProductCoefficient *rhou_coeff = nullptr;
  GridFunctionCoefficient *thermal_diff_coeff = nullptr;
  GradientGridFunctionCoefficient *gradT_coeff = nullptr;
  ScalarVectorProductCoefficient *kap_gradT_coeff = nullptr;

  ConstantCoefficient Lt_coeff;
  ConstantCoefficient Ht_lincoeff;
  ConstantCoefficient Ht_bdfcoeff;

  ConstantCoefficient *t_bc_coef0;
  ConstantCoefficient *t_bc_coef1;
  ConstantCoefficient *t_bc_coef2;
  ConstantCoefficient *t_bc_coef3;

  ConstantCoefficient *buffer_tbc = nullptr;
  ConstantCoefficient *buffer_qbc = nullptr;

  ParGridFunction *bufferInvRho = nullptr;
  ParGridFunction *bufferVisc = nullptr;
  ParGridFunction *bufferBulkVisc = nullptr;
  ParGridFunction *bufferRho = nullptr;
  ParGridFunction rhoDt;
  ParGridFunction *bufferTemp = nullptr;

  GridFunctionCoefficient *viscField = nullptr;
  GridFunctionCoefficient *bulkViscField = nullptr;
  GridFunctionCoefficient *rhoDtField = nullptr;
  GridFunctionCoefficient *invRho = nullptr;
  GridFunctionCoefficient *Rho = nullptr;
  GridFunctionCoefficient *kappaField = nullptr;

  ParGridFunction *buffer_tInlet = nullptr;
  ParGridFunction *buffer_tInletInf = nullptr;
  GridFunctionCoefficient *tInletField = nullptr;

  // space varying viscosity multiplier
  ParGridFunction viscMult_gf;
  viscositySpongeData vsd_;
  ParGridFunction viscTotal_gf;
  ParGridFunction visc_gf;
  ParGridFunction eddyVisc_gf;
  ParGridFunction kappa_gf;

  OperatorHandle Lt;
  OperatorHandle LQ;
  OperatorHandle At;
  OperatorHandle Ht;
  OperatorHandle Ms;
  OperatorHandle MsRho;
  OperatorHandle Mq;
  OperatorHandle Ds;

  mfem::Solver *MsInvPC = nullptr;
  mfem::CGSolver *MsInv = nullptr;
  mfem::Solver *MqInvPC = nullptr;
  mfem::CGSolver *MqInv = nullptr;
  mfem::Solver *HtInvPC = nullptr;
  mfem::CGSolver *HtInv = nullptr;
  mfem::Solver *MvInvPC = nullptr;

  ParGridFunction Tnm1_gf, Tnm2_gf;
  ParGridFunction Tn_gf, Tn_next_gf, Text_gf, resT_gf;
  ParGridFunction rn_gf;

  Vector fTn, Tn, Tn_next, Tnm1, Tnm2, NTn, NTnm1, NTnm2;
  Vector Text, Text_bdr, t_bdr;
  Vector resT, tmpR0a, tmpR0b, tmpR0c;

  ParGridFunction *un_next_gf = nullptr;
  ParGridFunction *subgridVisc_gf = nullptr;

  Vector tmpR0;
  Vector tmpR1;
  ParGridFunction R0PM0_gf;
  ParGridFunction R0PM1_gf;

  ParGridFunction Qt_gf;

  Vector gradMu, gradRho;
  Vector Qt;
  Vector rn;
  Vector kappa;
  Vector visc;
  Vector viscMult;
  Vector subgridVisc;

  // Print levels.
  int pl_mvsolve = 0;
  int pl_htsolve = 0;
  int pl_mtsolve = 0;

  // Relative tolerances.
  double rtol_hsolve = 1e-12;
  double rtol_htsolve = 1e-12;

  // Iteration counts.
  int iter_mtsolve = 0, iter_htsolve = 0;

  // Residuals.
  double res_mtsolve = 0.0, res_htsolve = 0.0;

  // Filter-based stabilization => move to "input" options
  int filter_cutoff_modes = 0;
  double filter_alpha = 0.0;

  // Initial temperature value (if constant IC)
  double T_ic_;

  FiniteElementCollection *sfec_filter = nullptr;
  ParFiniteElementSpace *sfes_filter = nullptr;
  ParGridFunction Tn_NM1_gf;
  ParGridFunction Tn_filtered_gf;

 public:
  ThermoChem(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, RunConfiguration *config,
             temporalSchemeCoefficients &timeCoeff, TPS::Tps *tps);

  virtual ~ThermoChem() {}

  void initializeSelf();
  void initializeFromFlow(flowToThermoChem *flow);
  void initializeFromTurbModel(turbModelToThermoChem *turbModel);
  void step();

  /**
   * @brief Hook to let derived classes register restart fields with the IODataOrganizer.
   */
  void initializeIO(IODataOrganizer &io) final;
  void initializeViz(ParaViewDataCollection &pvdc) final;

  void updateThermoP();
  void extrapolateState();
  void updateDensity(double tStep);
  void updateBC(int current_step);
  void updateDiffusivity();
  void computeSystemMass();
  void computeExplicitTempConvectionOP(bool extrap);
  void computeQt();
  void computeQtTO();
  void interpolateInlet();
  void uniformInlet();

  /// Initialize forms, solvers and preconditioners.
  void setup(double dt);

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentTemperature() { return &Tn_gf; }

  /// Return a pointer to the current density ParGridFunction.
  ParGridFunction *GetCurrentDensity() { return &rn_gf; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  ParGridFunction *GetCurrentTotalViscosity() { return &viscTotal_gf; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentTotalThermalDiffusivity() { return &kappa_gf; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiv() { return &Qt_gf; }

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
  void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);

  // MFEM_HOST_DEVICE
  void viscSpongePlanar(double *x, double &wgt);
};
#endif  // THERMOCHEM_HPP_
