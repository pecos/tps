
#ifndef THERMOCHEM_HPP
#define THERMOCHEM_HPP

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
#include <hdf5.h>
#include <tps_config.h>

#include "loMach_options.hpp"
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
#include "loMach.hpp"
#include "../utils/mfem_extras/pfem_extras.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

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
  //friend class LoMachSolver;
  //friend class Flow;  
private:

   TPS::Tps *tpsP_;
   LoMachSolver *loMach_;
  
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
  
   // The order of the scalar spaces
   int sorder;
   IntegrationRules gll_rules;    

   // temporary
   double Re_tau, Pr, Cp, gamma;
  
   /// Kinematic viscosity (dimensionless).
   double kin_vis;

   // make everything dimensionless after generalizing form channel
   double ambientPressure, thermoPressure, static_rho, Rgas, systemMass;
   double tPm1, tPm2, dtP;

   // Scalar \f$H^1\f$ finite element collection.
   FiniteElementCollection *sfec = nullptr;

   // Scalar \f$H^1\f$ finite element space.
   ParFiniteElementSpace *sfes = nullptr;  

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

   mfem::Solver *MsInvPC = nullptr;
   mfem::CGSolver *MsInv = nullptr;
   mfem::Solver *MqInvPC = nullptr;
   mfem::CGSolver *MqInv = nullptr;    
   mfem::Solver *HtInvPC = nullptr;
   mfem::CGSolver *HtInv = nullptr;

   ParGridFunction Tnm1_gf, Tnm2_gf;
   ParGridFunction Tn_gf, Tn_next_gf, Text_gf, resT_gf;  
   Vector fTn, Tn, Tn_next, Tnm1, Tnm2, NTn, NTnm1, NTnm2;
   Vector Text, Text_bdr, t_bdr;
   Vector resT, tmpR0, tmpR0a, tmpR0b, tmpR0c;

   Vector gradT;
   Vector Qt;      
   Vector rn;
   Vector alphaSml;
   ParGridFunction rn_gf;    
  
   // Iteration counts.
   int iter_mtsolve = 0, iter_htsolve = 0;

   // Residuals.
   double res_mtsolve = 0.0, res_htsolve = 0.0;  

   FiniteElementCollection *sfec_filter = nullptr;
   ParFiniteElementSpace *sfes_filter = nullptr;
   ParGridFunction Tn_NM1_gf;
   ParGridFunction Tn_filtered_gf;

   // Pointers to the different classes
   GasMixture *mixture;    // valid on host
   GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture  
   TransportProperties *transportPtr = NULL;  // valid on both host and device
   // TransportProperties *d_transport = NULL;  // valid on device, when available; otherwise = transportPtr
  
  
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
   
   void computeExplicitTempConvectionOP(bool extrap);      
   void computeQt();
   void computeQtTO();    

   /// Initialize forms, solvers and preconditioners.
   void Setup(double dt);  

   /// Return a pointer to the current temperature ParGridFunction.
   ParGridFunction *GetCurrentTemperature() { return &Tn_gf; }

   /// Return a pointer to the current density ParGridFunction.  
   ParGridFunction *GetCurrentDensity() { return &rn_gf; }  

   /// Return a pointer to the current total thermal diffusivity ParGridFunction.    
   ParGridFunction *GetCurrentTotalThermalDiffusivity() { return &alphaTotal_gf; }

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);  

   /// Add a Dirichlet boundary condition to the temperature and Qt field.
   void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);    
   void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);      
  
};
#endif
