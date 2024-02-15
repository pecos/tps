
#ifndef FLOW_HPP
#define FLOW_HPP

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


/// Container for a Dirichlet boundary condition of the velocity field.
class VelDirichletBC_T
{
public:
   VelDirichletBC_T(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   VelDirichletBC_T(VelDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~VelDirichletBC_T() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};

  
/// Container for a Dirichlet boundary condition of the pressure field.
class PresDirichletBC_T
{
public:
   PresDirichletBC_T(Array<int> attr, Coefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   PresDirichletBC_T(PresDirichletBC_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~PresDirichletBC_T() { delete coeff; }

   Array<int> attr;
   Coefficient *coeff;
};

/// Container for an acceleration term.
class AccelTerm_T
{
public:
   AccelTerm_T(Array<int> attr, VectorCoefficient *coeff)
      : attr(attr), coeff(coeff)
   {}

   AccelTerm_T(AccelTerm_T &&obj)
   {
      // Deep copy the attribute array
      this->attr = obj.attr;

      // Move the coefficient pointer
      this->coeff = obj.coeff;
      obj.coeff = nullptr;
   }

   ~AccelTerm_T() { delete coeff; }

   Array<int> attr;
   VectorCoefficient *coeff;
};


/// Add some description here...
class Flow
{
  friend class LoMachSolver;
  //friend class Flow;
  //friend class TurbModel;  
private:

  //TPS::Tps *tpsP_;
  //LoMachSolver *loMach_;
   LoMachOptions *loMach_opts = nullptr;

  //MPI_Groups *groupsMPI;
   int nprocs_;  // total number of MPI procs
   int rank;    // local MPI rank
   bool rank0;  // flag to indicate rank 0

   // Run options
   RunConfiguration *config = nullptr;
  
   // All essential attributes.
   Array<int> vel_ess_attr;
   Array<int> pres_ess_attr;  
  
   // All essential true dofs.
   Array<int> vel_ess_tdof;
   Array<int> pres_ess_tdof;  

   // Bookkeeping for velocity dirichlet bcs.
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for pressure dirichlet bcs.
   std::vector<PresDirichletBC_T> pres_dbcs;
  
   // Bookkeeping for acceleration (forcing) terms.
   std::vector<AccelTerm_T> accel_terms;

   // Bookkeeping for gravity (forcing) terms.
   std::vector<AccelTerm_T> gravity_terms;
  
   // space sizes;
   int Sdof, Pdof, Vdof, Ndof, NdofR0;
   int SdofInt, PdofInt, VdofInt, NdofInt, NdofR0Int;
  
   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;
   bool partial_assembly_pressure = false;  

   /// Enable/disable numerical integration rules of forms.
   //bool numerical_integ = false;
   bool numerical_integ = true;
  
   ParMesh *pmesh = nullptr;
   //ParMesh &pmesh;  
  
   // The order of the scalar spaces
   int order;
   int porder;
   int norder;  
   IntegrationRules gll_rules;    

   double bd0, bd1, bd2, bd3;
   double ab1, ab2, ab3;
   int nvel, dim;
   int num_equation;    
   int MaxIters;

   // Timers.
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;
  
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

   // nonlinear term
   FiniteElementCollection *nfec = nullptr;
   ParFiniteElementSpace *nfes = nullptr;
   FiniteElementCollection *nfecR0 = nullptr;
   ParFiniteElementSpace *nfesR0 = nullptr;  
  
   VectorMassIntegrator *hmv_blfi = nullptr;  
   VectorDiffusionIntegrator *hdv_blfi = nullptr;
   ElasticityIntegrator *hev_blfi = nullptr;

   // operators  
   ParNonlinearForm *N = nullptr;
   ParBilinearForm *Mv_form = nullptr;
   ParBilinearForm *MvRho_form = nullptr;
   ParBilinearForm *Ms_form = nullptr;
   ParBilinearForm *MsRho_form = nullptr;  
   ParBilinearForm *Sp_form = nullptr;
   ParMixedBilinearForm *D_form = nullptr;
   ParMixedBilinearForm *DRho_form = nullptr;  
   ParMixedBilinearForm *G_form = nullptr;
   ParMixedBilinearForm *Gp_form = nullptr;  
   ParBilinearForm *H_form = nullptr;
   VectorGridFunctionCoefficient *FText_gfcoeff = nullptr;
   ParLinearForm *FText_bdr_form = nullptr;
   ParLinearForm *f_form = nullptr;
   ParLinearForm *g_bdr_form = nullptr;
   ParLinearForm *gravity_form = nullptr;  

   // pressure
   GridFunctionCoefficient *Pext_gfcoeff = nullptr;
   ParLinearForm *Pext_bdr_form = nullptr;
   ParLinearForm *p_bdr_form = nullptr;  
  
   /// Linear form to compute the mass matrix in various subroutines.
   ParLinearForm *mass_lf = nullptr;
   ConstantCoefficient onecoeff;
   double volume = 0.0;

   ConstantCoefficient nlcoeff;
   ConstantCoefficient Sp_coeff;
   ConstantCoefficient H_lincoeff;
   ConstantCoefficient H_bdfcoeff;

   VectorConstantCoefficient *buffer_ubc = nullptr;
   VectorConstantCoefficient *buffer_accel = nullptr;
   VectorConstantCoefficient *wall_ubc = nullptr;
   VectorConstantCoefficient *bufferInlet_ubc = nullptr;
   ConstantCoefficient *bufferInlet_tbc = nullptr;
   ConstantCoefficient *bufferInlet_pbc = nullptr;
   ConstantCoefficient *bufferOutlet_pbc = nullptr;      

   ParGridFunction *bufferPM0 = nullptr;
   ParGridFunction *bufferPM1 = nullptr;
   GridFunctionCoefficient *viscField = nullptr;
   GridFunctionCoefficient *bulkViscField = nullptr;
   GridFunctionCoefficient *rhoDtField = nullptr;
   VectorGridFunctionCoefficient *rhoDtFieldR1 = nullptr;        
   GridFunctionCoefficient *invRho = nullptr;
   GridFunctionCoefficient *Rho = nullptr;
   //VectorGridFunctionCoefficient *FText_gfcoeff;    

   ParGridFunction *buffer_uInlet = nullptr;
   ParGridFunction *buffer_uInletInf = nullptr;
   ParGridFunction *buffer_pInlet = nullptr;
   VectorGridFunctionCoefficient *uInletField = nullptr;
   GridFunctionCoefficient *pInletField = nullptr;

   OperatorHandle Mv;
   OperatorHandle MvRho;  
   OperatorHandle Sp;
   OperatorHandle D;
   OperatorHandle DRho;  
   OperatorHandle G; 
   OperatorHandle Gp; 
   OperatorHandle H;
   OperatorHandle Ms;
   OperatorHandle MsRho;
   OperatorHandle Mq;  
   OperatorHandle Ds;  

   mfem::Solver *MsInvPC = nullptr;
   mfem::CGSolver *MsInv = nullptr;  
   mfem::Solver *MvInvPC = nullptr;
   mfem::CGSolver *MvInv = nullptr;  
   mfem::Solver *HInvPC = nullptr;
   mfem::CGSolver *HInv = nullptr;
   //mfem::GMRESSolver *HInv = nullptr;
   //mfem::BiCGSTABSolver *HInv = nullptr;  

   mfem::CGSolver *SpInv = nullptr;
   //mfem::GMRESSolver *SpInv = nullptr;
   //mfem::BiCGSTABSolver *SpInv = nullptr;
   //mfem::HypreBoomerAMG *SpInvPC = nullptr;
   mfem::OrthoSolver *SpInvOrthoPC = nullptr;

   // if not using amg to solve pressure
   mfem::Solver *SpInvPC = nullptr; 
  
   Vector fn, un, un_next, unm1, unm2;
   Vector u_star, u_half;  
   Vector uBn; 
   Vector Nun, Nunm1, Nunm2;
   Vector Fext, FText, Lext, Uext, Ldiv, LdivImp;  
   Vector resu, tmpR1, tmpR1a, tmpR1b, tmpR1c;
   Vector gradDivU;
   Vector gradU, gradV, gradW;
   Vector gradX, gradY, gradZ;
   Vector FBext;
   Vector divU, dtRho;
   Vector gravity;
   Vector boussinesqField;
   Vector uns;
   Vector curlSpace;
   Vector bufferR0sml, bufferR1sml;
   Vector pn, pnm1, deltaP, resp, FText_bdr, g_bdr;
   Vector tmpR0PM1;
   //Vector pnBig;
   Vector gridScale;
   Vector tmpR0, tmpR0a, tmpR0b, tmpR0c;
   Vector Pext_bdr, p_bdr;
   Vector gradMu, gradRho;
  
   ParGridFunction un_gf, un_next_gf, curlu_gf, curlcurlu_gf;
   ParGridFunction Lext_gf, FText_gf, resu_gf, Pext_gf, gradT_gf;
   ParGridFunction curl_gf;
   ParGridFunction unm1_gf, unm2_gf;
   ParGridFunction u_star_gf;    
   ParGridFunction pn_gf, resp_gf;
   ParGridFunction sml_gf, big_gf;

   ParGridFunction gradU_gf, gradV_gf, gradW_gf;  
    
  //ParGridFunction *bufferInvRho = nullptr;
   ParGridFunction bufferInvRho;
   ParGridFunction *bufferVisc = nullptr;
  //ParGridFunction bufferVisc;
   ParGridFunction *bufferBulkVisc = nullptr;
   ParGridFunction *bufferRho = nullptr;
  //ParGridFunction *bufferRhoDt = nullptr;
   ParGridFunction bufferRhoDt;
   ParGridFunction *bufferRhoDtR1 = nullptr;  
   ParGridFunction *bufferFText = nullptr;

   ParGridFunction *buffer_vInlet = nullptr;
   ParGridFunction *buffer_vInletInf = nullptr;
   GridFunctionCoefficient *vInletField = nullptr;
  
   // space varying viscosity multiplier  
   //ParGridFunction *bufferViscMult;  

   // pointers to externals
   ParGridFunction *visc_gf = nullptr;
   ParGridFunction *rn_gf = nullptr;
   ParGridFunction *Qt_gf = nullptr;

   // internals of external gf
   Vector rn, visc, Qt;  
  
   // Print levels.
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;
   int pl_mtsolve = 0;    

   // Relative tolerances.
   double rtol_spsolve = 1e-12;
   double rtol_hsolve = 1e-12;

   // Iteration counts.
   int iter_mvsolve = 0, iter_spsolve = 0, iter_hsolve = 0;

   // Residuals.
   double res_mvsolve = 0.0, res_spsolve = 0.0, res_hsolve = 0.0;
  
   // LOR related.
   ParLORDiscretization *lor = nullptr;
  
   // Filter-based stabilization => move to "input" options
   int filter_cutoff_modes = 0;
   double filter_alpha = 0.0;

   FiniteElementCollection *vfec_filter = nullptr;
   ParFiniteElementSpace *vfes_filter = nullptr;
   ParGridFunction un_NM1_gf;
   ParGridFunction un_filtered_gf;

   // swap spaces
   ParGridFunction R0PM0_gf;
   ParGridFunction R0PM1_gf;
   ParGridFunction R1PM0_gf;
   ParGridFunction R1PM0a_gf;
   ParGridFunction R1PM0b_gf;
   ParGridFunction R1PM0c_gf;
   ParGridFunction R1PX2_gf;     

   Vector r0pm0;
   Vector r1pm0; 
   Vector r0px2a, r0px2b, r0px2c;  
   Vector r1px2a, r1px2b, r1px2c;
  
  
public:
  Flow(mfem::ParMesh *pmesh_, RunConfiguration *config_, LoMachOptions *loMach_opts_);
  virtual ~Flow() {}

   void initialize();

   void initializeExternal(ParGridFunction *visc_gf_, ParGridFunction *rn_gf_, ParGridFunction *Qt_gf_, int nWalls, int nInlets, int nOutlets);
  
   void flowStep(double &time, double dt, const int cur_step, const int start_step, std::vector<double> bdf); //, bool provisional = false);

   void extrapolateState(int current_step, std::vector<double> ab);
   void updateGradients(double tStep);
   void updateGradientsOP(double tStep);  
   void updateBC(int current_step);   
   void computeExplicitForcing();
   void computeExplicitUnsteady(double dt);
   void computeExplicitUnsteadyBDF(double dt);    
   void computeExplicitConvection(double uStep);
   void computeExplicitConvectionOP(double uStep, bool extrap);
   void computeExplicitDiffusion();
   void computeImplicitDiffusion();  
   void interpolateInlet();
   void uniformInlet();  
  
   /// Initialize forms, solvers and preconditioners.
   void setup(double dt);  

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }
  
   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelGradU() { return &gradU_gf; }
   ParGridFunction *GetCurrentVelGradV() { return &gradV_gf; }
   ParGridFunction *GetCurrentVelGradW() { return &gradW_gf; }  
  
   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);  

   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);
  
   /// Add an acceleration term to the RHS of the equation.
   /**
    * The VecFuncT @a f is evaluated at the current time t and extrapolated
    * together with the nonlinear parts of the Navier Stokes equation.
    */
   void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);
   void AddAccelTerm(VecFuncT *f, Array<int> &attr);

   /// Add an Boussinesq term to the RHS of the equation.
   void AddGravityTerm(VectorCoefficient *coeff, Array<int> &attr);
   void AddGravityTerm(VecFuncT *g, Array<int> &attr);
  
   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^2\f$.
   //void ComputeCurl2D(ParGridFunction &u,
   //                   ParGridFunction &cu,
   //                   bool assume_scalar = false);

   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^3\f$.
   //void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);
  
   /// Eliminate essential BCs in an Operator and apply to RHS.
   // rename this to something sensible "ApplyEssentialBC" or something
  /*
   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);
  */
   /// Remove mean from a Vector.
   /**
    * Modify the Vector @a v by subtracting its mean using
    * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
    */
  /*
   void Orthogonalize(Vector &v);
  */
  
   /// Remove the mean from a ParGridFunction.
   /**
    * Modify the ParGridFunction @a v by subtracting its mean using
    * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
    */
   void MeanZero(ParGridFunction &v);  
  
};
#endif
