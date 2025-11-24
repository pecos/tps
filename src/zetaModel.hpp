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

#ifndef ZETAMODEL_HPP_
#define ZETAMODEL_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
// #include <hdf5.h>
#include <tps_config.h>

// #include "loMach_options.hpp"
#include "../utils/mfem_extras/pfem_extras.hpp"
#include "averaging.hpp"
#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "run_configuration.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "turb_model_base.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachSolver;
class LoMachOptions;
struct temporalSchemeCoefficients;

/// Add some description here...
class ZetaModel : public TurbModelBase {
  friend class LoMachSolver;

 private:
  TPS::Tps *tpsP_;
  LoMachOptions *loMach_opts_ = nullptr;

  // Mesh and discretization scheme info
  ParMesh *pmesh_ = nullptr;
  int order_;
  IntegrationRules gll_rules_;
  const temporalSchemeCoefficients &time_coeff_;
  double dt_;
  double time_;
  int nvel_, dim_;
  double dt_nm2_, dt_nm1_;
  int forder_ = 1;

  std::string ic_string_;

  // MPI_Groups *groupsMPI;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // space sizes
  int Sdof_, SdofInt_;

  // axisymmetry
  bool axisym_;

  /// Enable/disable verbose output.
  bool verbose = true;

  // Flags
  bool partial_assembly_ = false;
  bool numerical_integ_ = false;
  //bool numerical_integ_ = true;

  // Linear-solver-related options
  int pl_solve_ = 0;
  int max_iter_ = 2000;
  double rtol_ = 1e-10;
  
  int f_max_iter_ = 4000;
  double f_rtol_ = 1e-8;  

  double tke_ic_, tdr_ic_;
  double tke_min_, tdr_min_, zeta_min_, v2_min_;
  double fRate_min_, tts_min_, tls_min_;
  double fRate_max_, tts_max_, tls_max_;
  double mut_min_;
  double des_wgt_;
  double zfp_max_;

  // just keep these saved for ease
  int numWalls_, numInlets_, numOutlets_;

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec_ = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes_ = nullptr;

  FiniteElementCollection *ffec_ = nullptr;
  ParFiniteElementSpace *ffes_ = nullptr;  

  /// velocity
  ParGridFunction *vel_gf_ = nullptr;
  Vector vel_;

  /// swirl
  ParGridFunction *swirl_gf_ = nullptr;
  Vector swirl_;

  /// velocity gradients
  ParGridFunction *gradU_gf_ = nullptr;
  ParGridFunction *gradV_gf_ = nullptr;
  ParGridFunction *gradW_gf_ = nullptr;
  Vector gradU_;
  Vector gradV_;
  Vector gradW_;

  // Fields
  ParGridFunction rhoDt_gf_;

  /// fluid density
  // ParGridFunction *rho_gf_ = nullptr;
  Vector rho_;

  /// molecular viscosity
  // ParGridFunction *mu_gf_ = nullptr;
  Vector mu_;
  Vector mult_;  

  /// grid information
  ParGridFunction *gridScale_gf_ = nullptr;

  /// eddy viscosity
  ParGridFunction eddyVisc_gf_;
  Vector eddyVisc_;

  /// turbulent kinetic energy
  ParGridFunction tke_gf_;
  Vector tke_;
  ParGridFunction tke_next_gf_;
  Vector tke_next_;
  Vector tke_nm1_, tke_nm2_;
  Vector Ntke_, Ntke_nm1_, Ntke_nm2_;
  ParGridFunction tke_lapl_gf_;
  Vector tke_lapl_;  

  /// turbulent dissipation rate
  ParGridFunction tdr_gf_;
  Vector tdr_;
  ParGridFunction tdr_next_gf_;
  Vector tdr_next_;
  Vector tdr_nm1_, tdr_nm2_;
  Vector Ntdr_, Ntdr_nm1_, Ntdr_nm2_;
  ParGridFunction tdr_wall_gf_;
  GridFunctionCoefficient *tdr_bc_ = nullptr;  

  /// ratio of wall normal stress component to k
  ParGridFunction zeta_gf_;
  Vector zeta_;
  ParGridFunction zeta_next_gf_;
  Vector zeta_next_;
  Vector zeta_nm1_, zeta_nm2_;
  Vector Nzeta_, Nzeta_nm1_, Nzeta_nm2_;

  ParGridFunction v2_gf_;
  Vector v2_;
  ParGridFunction v2_next_gf_;
  Vector v2_next_;
  Vector v2_nm1_, v2_nm2_;
  Vector Nv2_, Nv2_nm1_, Nv2_nm2_;
  
  /// elliptic relaxation rate
  ParGridFunction fRate_gf_;
  Vector fRate_;

  /// turbulent length scale
  ParGridFunction tls_gf_;
  Vector tls_;
  ParGridFunction tls2_gf_;
  ParGridFunction tls2_next_gf_;  
  Vector tls2_, tls2_next_, tls2_nm1_, tls2_nm2_;  

  /// turbulent time scale
  ParGridFunction tts_gf_;
  ParGridFunction tts_next_gf_;  
  Vector tts_, tts_strain_, tts_kol_;
  Vector tts_next_, tts_nm1_, tts_nm2_;

  /// turbulent production
  ParGridFunction prod_gf_;
  ParGridFunction prod_next_gf_;  
  Vector prod_, prod_next_, prod_nm1_, prod_nm2_;

  /// model coefficients
  double Cmu_ = 0.22;
  double sigmaK_ = 1.0;
  double sigmaE_ = 1.3;
  double sigmaZ_ = 1.2;
  double Ce2_ = 1.9;
  double C1_ = 1.4;
  double C2prime_ = 0.65;
  double C2_ = 0.3;  
  double Ct_ = 6.0;
  //double Cl_ = 0.36;
  //double Cn_ = 85.0;  
  double Cl_ = 0.23;
  double Cn_ = 70.0;  
  double Ce1_;  // function of local zeta

  ParGridFunction res_gf_;
  Vector res_;

  ParGridFunction resf_gf_;
  Vector resf_;  

  ParGridFunction vfres_gf_;
  
  // ParGridFunction diag_gf_;

  ParGridFunction sMag_gf_;  
  Vector tmpR0_, tmpR0a_, tmpR0b_, tmpR0c_;
  Vector strain_, sMag_;
  Vector ftmpR0_;
  
  ParGridFunction radius_gf_;
  Vector radius_v_;

  /// coefficient fields for operators
  GridFunctionCoefficient *delta_coeff_ = nullptr;
  GradientGridFunctionCoefficient *gradTKE_coeff_ = nullptr;
  ScalarVectorProductCoefficient *nu_gradTKE_coeff_ = nullptr;
  GridFunctionCoefficient *mu_coeff_ = nullptr;
  RatioCoefficient *nu_coeff_ = nullptr;
  RatioCoefficient *nu_delta_coeff_ = nullptr;
  ProductCoefficient *two_nu_delta_coeff_ = nullptr;
  ProductCoefficient *two_nuNeg_delta_coeff_ = nullptr;
  GradientGridFunctionCoefficient *gradZeta_coeff_ = nullptr;
  //ScalarVectorProductCoefficient *tdr_wall_coeff_ = nullptr;
  //DivergenceGridFunctionCoefficient *tdr_wall_coeff_ = nullptr;
  GridFunctionCoefficient *tdr_wall_coeff_ = nullptr;
  ScalarVectorProductCoefficient *fRate_wall_coeff_ = nullptr;
  ScalarVectorProductCoefficient *wall_coeff_ = nullptr;
  GridFunctionCoefficient *tdr_wall_eval_coeff_ = nullptr;

  GridFunctionCoefficient *swirl_coeff_ = nullptr;
  ProductCoefficient *rad_rho_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_rhou_coeff_ = nullptr;
  ProductCoefficient *rad_tke_diag_coeff_ = nullptr;
  ProductCoefficient *rad_tke_diff_total_coeff_ = nullptr;
  ProductCoefficient *rad_tdr_diag_coeff_ = nullptr;
  ProductCoefficient *rad_tdr_diff_total_coeff_ = nullptr;
  ProductCoefficient *rad_v2_diag_coeff_ = nullptr;
  ProductCoefficient *rad_f_diag_coeff_ = nullptr;
  ProductCoefficient *rad_unity_coeff_ = nullptr;
  ProductCoefficient *rad_zeta_diag_coeff_ = nullptr;
  ProductCoefficient *rad_zeta_diff_total_coeff_ = nullptr;
  ProductCoefficient *rad_scalar_diff_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rad_nu_gradTKE_coeff_ = nullptr;

  GridFunctionCoefficient *tts_coeff_ = nullptr;
  GridFunctionCoefficient *tls2_coeff_ = nullptr;
  GridFunctionCoefficient *prod_coeff_ = nullptr;
  GridFunctionCoefficient *tke_coeff_ = nullptr;
  GridFunctionCoefficient *tdr_coeff_ = nullptr;  
  GridFunctionCoefficient *rho_coeff_ = nullptr;
  VectorGridFunctionCoefficient *vel_coeff_ = nullptr;
  ScalarVectorProductCoefficient *rhou_coeff_ = nullptr;
  GridFunctionCoefficient *scalar_diff_coeff_ = nullptr;
  GridFunctionCoefficient *mut_coeff_ = nullptr;
  GridFunctionCoefficient *mult_coeff_ = nullptr;
  SumCoefficient *tke_diff_sum_coeff_ = nullptr;
  SumCoefficient *tdr_diff_sum_coeff_ = nullptr;
  SumCoefficient *zeta_diff_sum_coeff_ = nullptr;
  ProductCoefficient *tke_diff_total_coeff_ = nullptr;
  ProductCoefficient *tdr_diff_total_coeff_ = nullptr;
  ProductCoefficient *zeta_diff_total_coeff_ = nullptr;
  ConstantCoefficient *unity_diff_coeff_ = nullptr;
  ProductCoefficient *unity_diff_total_coeff_ = nullptr;
  ConstantCoefficient *unity_coeff_ = nullptr;
  ConstantCoefficient *zero_coeff_ = nullptr;
  ConstantCoefficient *posTwo_coeff_ = nullptr;
  ConstantCoefficient *negTwo_coeff_ = nullptr;

  GridFunctionCoefficient *rhoDt_coeff_ = nullptr;
  RatioCoefficient *rhoTTS_coeff_ = nullptr;
  ConstantCoefficient *Ce2_coeff_ = nullptr;
  ProductCoefficient *Ce2rhoTTS_coeff_ = nullptr;
  RatioCoefficient *Pk_coeff_ = nullptr;
  RatioCoefficient *ek_coeff_ = nullptr;
  ProductCoefficient *ek_rho_coeff_ = nullptr;    
  SumCoefficient *tke_diag_coeff_ = nullptr;
  SumCoefficient *tdr_diag_coeff_ = nullptr;
  SumCoefficient *zeta_diag_coeff_ = nullptr;
  SumCoefficient *v2_diag_coeff_ = nullptr;  
  RatioCoefficient *f_diag_coeff_ = nullptr;
  SumCoefficient *f_diag_total_coeff_ = nullptr;

  ProductCoefficient *diff_total_coeff_ = nullptr;
  SumCoefficient *diag_coeff_ = nullptr;

  GridFunctionCoefficient *tke_field_ = nullptr;

  /// operators and solvers
  ParBilinearForm *As_form_ = nullptr;
  ParBilinearForm *Ms_form_ = nullptr;
  ParBilinearForm *MsRho_form_ = nullptr;
  ParBilinearForm *Mf_form_ = nullptr;  
  ParBilinearForm *Hk_form_ = nullptr;
  ParBilinearForm *He_form_ = nullptr;
  ParBilinearForm *Hv_form_ = nullptr;
  ParBilinearForm *Hf_form_ = nullptr;
  ParBilinearForm *Hz_form_ = nullptr;
  ParBilinearForm *Lk_form_ = nullptr;
  ParBilinearForm *Lf_form_ = nullptr;
  ParLinearForm *Lk_bdry_ = nullptr;  
  ParLinearForm *He_bdry_ = nullptr;

  OperatorHandle As_;
  OperatorHandle Ms_;
  OperatorHandle MsRho_;
  OperatorHandle Mf_;  
  OperatorHandle Hk_;
  OperatorHandle He_;
  OperatorHandle Hv_;
  OperatorHandle Hf_;
  OperatorHandle Hz_;
  OperatorHandle Lk_;
  OperatorHandle Lf_;     

  mfem::Solver *MsInvPC_ = nullptr;
  mfem::CGSolver *MsInv_ = nullptr;
  mfem::Solver *MsRhoInvPC_ = nullptr;
  mfem::CGSolver *MsRhoInv_ = nullptr;  
  mfem::Solver *HkInvPC_ = nullptr;
  mfem::Solver *HeInvPC_ = nullptr;
  mfem::Solver *HvInvPC_ = nullptr;
  mfem::Solver *HfInvPC_ = nullptr;
  mfem::Solver *HzInvPC_ = nullptr;    
  mfem::CGSolver *HkInv_ = nullptr;
  mfem::CGSolver *HeInv_ = nullptr;
  mfem::CGSolver *HvInv_ = nullptr;
  mfem::CGSolver *HfInv_ = nullptr;
  // mfem::GMRESSolver *HfInv_ = nullptr;  
  mfem::CGSolver *HzInv_ = nullptr;    

  // Boundary condition info
  Array<int> tke_ess_attr_;
  Array<int> tdr_ess_attr_;
  Array<int> zeta_ess_attr_;
  Array<int> v2_ess_attr_;  
  Array<int> fRate_ess_attr_;
  Array<int> tke_ess_tdof_;
  Array<int> tdr_ess_tdof_;
  Array<int> zeta_ess_tdof_;
  Array<int> v2_ess_tdof_;  
  Array<int> fRate_ess_tdof_;
  Array<int> *ess_attr_ = nullptr;
  Array<int> *ess_tdof_ = nullptr;

  std::vector<DirichletBC_T<Coefficient>> tke_dbcs_;
  std::vector<DirichletBC_T<Coefficient>> tdr_dbcs_;
  std::vector<DirichletBC_T<Coefficient>> zeta_dbcs_;
  std::vector<DirichletBC_T<Coefficient>> v2_dbcs_;  
  std::vector<DirichletBC_T<Coefficient>> fRate_dbcs_;

 public:
  ZetaModel(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff, TPS::Tps *tps,
            ParGridFunction *gridScale);
  virtual ~ZetaModel();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void updateBC(int current_step);
  void step() final;
  void setup() final;
  void initializeIO(IODataOrganizer &io) final;  
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void tkeStep();
  void tdrStep();
  void zetaStep();
  void v2Step();  
  void fStep();
  void convection(string scalar);
  void updateTimestepHistory();
  void updateZeta();
  void extrapolateState(); 
  void extrapolateRHS();  
  void updateProd();
  void updateMsRho();  
  void updateTLS();
  void updateTTS();
  void computeStrain();
  void updateMuT();
  void computeTDRwall();

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *getCurrentEddyViscosity() { return &eddyVisc_gf_; }

  /// Add a Dirichlet boundary condition to the temperature and Qt field.
  void AddTKEDirichletBC(const double &tke, Array<int> &attr);
  void AddTKEDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTKEDirichletBC(ScalarFuncT *f, Array<int> &attr);

  void AddV2DirichletBC(const double &tke, Array<int> &attr);
  void AddV2DirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddV2DirichletBC(ScalarFuncT *f, Array<int> &attr);
  
  void AddTDRDirichletBC(const double &tdr, Array<int> &attr);
  void AddTDRDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddTDRDirichletBC(ScalarFuncT *f, Array<int> &attr);

  void AddZETADirichletBC(const double &zeta, Array<int> &attr);
  void AddZETADirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddZETADirichletBC(ScalarFuncT *f, Array<int> &attr);

  void AddFRATEDirichletBC(const double &fRate, Array<int> &attr);
  void AddFRATEDirichletBC(Coefficient *coeff, Array<int> &attr);
  void AddFRATEDirichletBC(ScalarFuncT *f, Array<int> &attr);
};
#endif  // ZETAMODEL_HPP_
