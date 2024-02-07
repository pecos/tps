
#ifndef THERMOCHEM_HPP
#define THERMOCHEM_HPP

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
#include <hdf5.h>
#include <tps_config.h>

//#include "loMach_options.hpp"
#include "io.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "run_configuration.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "transport_properties.hpp"
#include "argon_transport.hpp"
#include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "radiation.hpp"
//#include "loMach.hpp"
#include "../utils/mfem_extras/pfem_extras.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachSolver;
class LoMachOptions;

/// Container for a Dirichlet boundary condition of the temperature field.
class TempDirichletBC_T
{
public:
   TempDirichletBC_T(Array<int> attr, Coefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   TempDirichletBC_T(TempDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~TempDirichletBC_T() { delete coeff; }

   Array<int> attr;
   Coefficient *coeff;
};

class QtDirichletBC_T
{
public:
   QtDirichletBC_T(Array<int> attr, Coefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   QtDirichletBC_T(QtDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~QtDirichletBC_T() { delete coeff; }

   Array<int> attr;
   Coefficient *coeff;
};


/// Add some description here...
class ThermoChem
{
  friend class LoMachSolver;
  //friend class Flow;
  //friend class TurbModel;  
private:

   TPS::Tps *tpsP_;
   LoMachSolver *loMach_;
   LoMachOptions *loMach_opts_;

   MPI_Groups *groupsMPI;
   int nprocs_;  // total number of MPI procs
   int rank_;    // local MPI rank
   bool rank0_;  // flag to indicate rank 0

   // Run options
   RunConfiguration config;
  
   // All essential attributes.
   Array<int> temp_ess_attr;
   Array<int> Qt_ess_attr;  
  
   // All essential true dofs.
   Array<int> temp_ess_tdof;
   Array<int> Qt_ess_tdof;  
  
   // Bookkeeping for temperature dirichlet bcs.
   std::vector<TempDirichletBC_T> temp_dbcs;

   // Bookkeeping for Qt dirichlet bcs.
   std::vector<QtDirichletBC_T> Qt_dbcs;  
  
   // space sizes
   int Sdof, SdofInt, NdofInt, NdofR0Int;

   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;
   bool partial_assembly_pressure = true;  

   /// Enable/disable numerical integration rules of forms.
   //bool numerical_integ = false;
   bool numerical_integ = true;
  
   //ParMesh *pmesh = nullptr;
   ParMesh &pmesh;  
  
   // The order of the scalar spaces
   int order;
   IntegrationRules gll_rules;    

   double bd0, bd1, bd2, bd3;
   double ab1, ab2, ab3;
   int nvel, dim;
   int num_equation;    
   int MaxIters;

   // just keep these saved for ease
   int numWalls, numInlets, numOutlets;
  
   // loMach options to run as incompressible
   bool constantViscosity = false;
   bool constantDensity = false;
   bool incompressibleSolve = false;  
   bool pFilter = false;

   double dt;
   double time;
   int iter;
  
   // temporary
   double Re_tau, Pr, Cp, gamma;
  
   /// Kinematic viscosity (dimensionless).
   double kin_vis, dyn_vis;

   // make everything dimensionless after generalizing form channel
   double ambientPressure, thermoPressure, static_rho, Rgas, systemMass;
   double tPm1, tPm2, dtP;

   // Scalar \f$H^1\f$ finite element collection.
   FiniteElementCollection *sfec = nullptr;

   // Scalar \f$H^1\f$ finite element space.
   ParFiniteElementSpace *sfes = nullptr;  

   /// Velocity \f$H^1\f$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   /// Velocity \f$(H^1)^d\f$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;
  
   // operators
   DiffusionIntegrator *hdt_blfi = nullptr;
   MassIntegrator *hmt_blfi = nullptr;
   ParBilinearForm *Mv_form = nullptr;  
   ParBilinearForm *Lt_form = nullptr;    
   ParBilinearForm *At_form = nullptr;  
   ParBilinearForm *Ms_form = nullptr;
   ParBilinearForm *MsRho_form = nullptr;
   ParMixedBilinearForm *G_form = nullptr;  
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
  
   ConstantCoefficient t_bc_coef0;
   ConstantCoefficient t_bc_coef1;
   ConstantCoefficient t_bc_coef2;
   ConstantCoefficient t_bc_coef3;    
  
   ConstantCoefficient *buffer_tbc;
   ConstantCoefficient *buffer_qbc;    

   ParGridFunction *bufferInvRho;  
   ParGridFunction *bufferVisc;
   ParGridFunction *bufferBulkVisc;
   ParGridFunction *bufferRho;
   ParGridFunction *bufferRhoDt;  
   ParGridFunction *bufferAlpha;
   ParGridFunction *bufferTemp;

   GridFunctionCoefficient *viscField;
   GridFunctionCoefficient *bulkViscField;  
   GridFunctionCoefficient *rhoDtField;
   GridFunctionCoefficient *invRho;
   GridFunctionCoefficient *Rho;  
   GridFunctionCoefficient *alphaField;  

   ParGridFunction *buffer_tInlet;
   ParGridFunction *buffer_tInletInf;
   GridFunctionCoefficient *tInletField;

   // space varying viscosity multiplier  
   ParGridFunction *bufferViscMult;  
   viscositySpongeData vsd_;
   ParGridFunction viscTotal_gf;
   ParGridFunction alphaTotal_gf;      

   OperatorHandle Lt;
   OperatorHandle LQ;  
   OperatorHandle At;
   OperatorHandle Ht;
   OperatorHandle Mv;
   OperatorHandle G;
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
   mfem::CGSolver *MvInv = nullptr;  
  
   ParGridFunction Tnm1_gf, Tnm2_gf;
   ParGridFunction Tn_gf, Tn_next_gf, Text_gf, resT_gf;
   ParGridFunction rn_gf, gradT_gf;
  
   Vector fTn, Tn, Tn_next, Tnm1, Tnm2, NTn, NTnm1, NTnm2;
   Vector Text, Text_bdr, t_bdr;
   Vector resT, tmpR0a, tmpR0b, tmpR0c;
  
   Vector &tmpR0;
   Vector &tmpR1;
   ParGridFunction &R0PM0_gf;
   ParGridFunction &R0PM1_gf;
   ParGridFunction &un_gf;
   ParGridFunction &un_next_gf;

   Vector gradT;
   Vector gradMu, gradRho;  
   Vector Qt;      
   Vector rn;
   Vector alphaSml;
   Vector viscSml;
   Vector viscMultSml;
   Vector subgridViscSml;  

   // Print levels.
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;
   int pl_mtsolve = 0;  

   // Relative tolerances.
   double rtol_spsolve = 1e-12;
   double rtol_hsolve = 1e-12;
   double rtol_htsolve = 1e-12;    
  
   // Iteration counts.
   int iter_mtsolve = 0, iter_htsolve = 0;

   // Residuals.
   double res_mtsolve = 0.0, res_htsolve = 0.0;  

   // Filter-based stabilization => move to "input" options
   int filter_cutoff_modes = 0;
   double filter_alpha = 0.0;
  
   FiniteElementCollection *sfec_filter = nullptr;
   ParFiniteElementSpace *sfes_filter = nullptr;
   ParGridFunction Tn_NM1_gf;
   ParGridFunction Tn_filtered_gf;

   // Pointers to the different classes
   GasMixture *mixture;    // valid on host
   GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture  
   TransportProperties *transportPtr = NULL;  // valid on both host and device
   // TransportProperties *d_transport = NULL;  // valid on device, when available; otherwise = transportPtr

   // subgrid scale => move to turb model class
   ParGridFunction *bufferSubgridVisc;
  
  
public:
  ThermoChem(TPS::Tps *tps, LoMachSolver *loMach);
  virtual ~ThermoChem() {}

   void initialize();
  
   void thermoChemStep(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);

   void updateThermoP();
   void extrapolateState(int current_step);
   void updateDensity(double tStep);
   //void updateGradients(double tStep);
   void updateGradientsOP(double tStep);  
   void updateBC(int current_step);
   void updateDiffusivity();
   void computeSystemMass();
   
   void computeExplicitTempConvectionOP(bool extrap);      
   void computeQt();
   void computeQtTO();
   void interpolateInlet();
   void uniformInlet();  
  
   /// Initialize forms, solvers and preconditioners.
   void Setup(double dt);  

   /// Return a pointer to the current temperature ParGridFunction.
   ParGridFunction *GetCurrentTemperature() { return &Tn_gf; }

   /// Return a pointer to the current density ParGridFunction.  
   ParGridFunction *GetCurrentDensity() { return &rn_gf; }  

   /// Return a pointer to the current total viscosity ParGridFunction.  
   ParGridFunction *GetCurrentTotalViscosity() { return &viscTotal_gf; }    
  
   /// Return a pointer to the current total thermal diffusivity ParGridFunction.    
   ParGridFunction *GetCurrentTotalThermalDiffusivity() { return &alphaTotal_gf; }

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);  

   /// Add a Dirichlet boundary condition to the temperature and Qt field.
   void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);    
   void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);      

   /// Eliminate essential BCs in an Operator and apply to RHS.
   // rename this to something sensible "ApplyEssentialBC" or something
   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);

   /// Remove mean from a Vector.
   /**
    * Modify the Vector @a v by subtracting its mean using
    * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
    */
   void Orthogonalize(Vector &v);

   // subgrid scale models => move to turb model class
   void sgsSmag(const DenseMatrix &gradUp, double delta, double &nu_sgs);
   void sgsSigma(const DenseMatrix &gradUp, double delta, double &nu_sgs);  
  
  
};
#endif
