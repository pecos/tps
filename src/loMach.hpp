

#ifndef LOMACH_HPP
#define LOMACH_HPP

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
#include <hdf5.h>
#include <tps_config.h>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach_options.hpp"
#include "io.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "run_configuration.hpp"
#include "mfem.hpp"
//#include "solvers.hpp"
#include "mfem/linalg/solvers.hpp"

#include "transport_properties.hpp"
#include "argon_transport.hpp"
#include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "radiation.hpp"


using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

// helper stuff
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



/** Base class for lo-Mach solvers
 *
 */
class LoMachSolver : public TPS::Solver {
protected:

   mfem::MPI_Session &mpi_;
  //mfem::MPI_Session &mpi;  
   LoMachOptions loMach_opts_;

   MPI_Groups *groupsMPI;
   //MPI_Session &mpi;
   int nprocs_;  // total number of MPI procs
   int rank_;    // local MPI rank
   bool rank0_;  // flag to indicate rank 0

   // Run options
   RunConfiguration config;

   // History file
   std::ofstream histFile;

   // Number of dimensions
   int dim;
   int nvel;

   // Number of equations
   int num_equation;  

   // Is it ambipolar?
   bool ambipolar = false;

   // Is it two-temperature?
   bool twoTemperature_ = false;  
  
   // Average handler
   Averaging *average;
  
   double dt;
   double time;
   int iter;
  
   // pointer to parent Tps class
   TPS::Tps *tpsP_;

   mfem::ParMesh *pmesh_;
   int dim_;
   int true_size_;
   Array<int> offsets_;

   // min/max element size
   double hmin, hmax;

  int MaxIters;
  double max_speed;
  double CFL;
  int numActiveSpecies, numSpecies;

   // exit status code;
   int exit_status_;

   // mapping from local to global element index
   int *locToGlobElem;
  
   // just keep these saved for ease
   int numWalls, numInlets, numOutlets;
  
   /// Print information about the Navier version.
   void PrintInfo();

   /// Update the EXTk/BDF time integration coefficient.
   /**
    * Depending on which time step the computation is in, the EXTk/BDF time
    * integration coefficients have to be set accordingly. This allows
    * bootstrapping with a BDF scheme of order 1 and increasing the order each
    * following time step, up to order 3 (or whichever order is set in
    * SetMaxBDFOrder).
    */
   void SetTimeIntegrationCoefficients(int step);

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

   /// Enable/disable debug output.
   bool debug = false;

   /// Enable/disable verbose output.
   bool verbose = true;

   /// Enable/disable partial assembly of forms.
   bool partial_assembly = false;

   /// Enable/disable numerical integration rules of forms.
   bool numerical_integ = false;
  
   /// The parallel mesh.
   ParMesh *pmesh = nullptr;
   //Mesh *serial_mesh;  

   /// The order of the velocity and pressure space.
   int order;
   int porder;
   int norder;  

   // total number of mesh elements (serial)
   int nelemGlobal_;

   // original mesh partition info (stored on rank 0)
   Array<int> partitioning_;
   const int defaultPartMethod = 1;  
  
   // temporary
  double Re_tau, Pr;
  
   /// Kinematic viscosity (dimensionless).
   double kin_vis;

   // make everything dimensionless after generalizing form channel
   double ambientPressure, Rgas;  

   IntegrationRules gll_rules;

   /// Velocity \f$H^1\f$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   /// Velocity \f$(H^1)^d\f$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;

   //// total space for compatibility
   ParFiniteElementSpace *fvfes = nullptr;  
  
   /// Pressure \f$H^1\f$ finite element collection.
   FiniteElementCollection *pfec = nullptr;

   /// Pressure \f$H^1\f$ finite element space.
   ParFiniteElementSpace *pfes = nullptr;

   /// Temperature \f$H^1\f$ finite element collection.
   FiniteElementCollection *tfec = nullptr;

   /// Temperature \f$H^1\f$ finite element space.
   ParFiniteElementSpace *tfes = nullptr;  

   /// density \f$H^1\f$ finite element collection.
   FiniteElementCollection *rfec = nullptr;

   /// density \f$H^1\f$ finite element space.
   ParFiniteElementSpace *rfes = nullptr;  
  
   // nonlinear term
   FiniteElementCollection *nfec = nullptr;
   ParFiniteElementSpace *nfes = nullptr;
  
   ParNonlinearForm *N = nullptr;
   ParBilinearForm *Mv_form = nullptr;
   ParBilinearForm *Sp_form = nullptr;
   ParMixedBilinearForm *D_form = nullptr;
   ParMixedBilinearForm *G_form = nullptr;
   ParBilinearForm *H_form = nullptr;
   VectorGridFunctionCoefficient *FText_gfcoeff = nullptr;
   ParLinearForm *FText_bdr_form = nullptr;
   ParLinearForm *f_form = nullptr;
   ParLinearForm *g_bdr_form = nullptr;

   // temperature
   ParBilinearForm *Mt_form = nullptr;
   ParMixedBilinearForm *Dt_form = nullptr;  
   ParBilinearForm *Ht_form = nullptr;
   GridFunctionCoefficient *Text_gfcoeff = nullptr;
   ParLinearForm *Text_bdr_form = nullptr;
   ParLinearForm *ft_form = nullptr;
   ParLinearForm *t_bdr_form = nullptr;    
  
   /// Linear form to compute the mass matrix in various subroutines.
   ParLinearForm *mass_lf = nullptr;
   ConstantCoefficient onecoeff;
   double volume = 0.0;

   ConstantCoefficient nlcoeff;
   ConstantCoefficient Sp_coeff;
   ConstantCoefficient H_lincoeff;
   ConstantCoefficient H_bdfcoeff;
   ConstantCoefficient Ht_lincoeff;
   ConstantCoefficient Ht_bdfcoeff;  

  //VectorConstantCoefficient u_bc_coef;
  //ConstantCoefficient t_bc_coef;

   VectorConstantCoefficient *buffer_ubc;
   ConstantCoefficient *buffer_tbc;
   VectorConstantCoefficient *wall_ubc;  
  
   ParGridFunction *bufferInvRho;  
   ParGridFunction *bufferVisc;
   ParGridFunction *bufferAlpha;
   ParGridFunction *bufferTemp;
   ParGridFunction *bufferPM0;      
   ParGridFunction *bufferPM1;    
   GridFunctionCoefficient *viscField;
   GridFunctionCoefficient *invRho;
   GridFunctionCoefficient *alphaField;    
  
   OperatorHandle Mv;
   OperatorHandle Sp;
   OperatorHandle D;
   OperatorHandle G;
   OperatorHandle H;

   OperatorHandle Mt;
   OperatorHandle Dt;  
   OperatorHandle Ht;

   mfem::Solver *MvInvPC = nullptr;
   mfem::CGSolver *MvInv = nullptr;  
   mfem::Solver *HInvPC = nullptr;
   mfem::CGSolver *HInv = nullptr;

   mfem::CGSolver *SpInv = nullptr;  
   mfem::HypreBoomerAMG *SpInvPC = nullptr;
   mfem::OrthoSolver *SpInvOrthoPC = nullptr;

   // if not using amg to solve pressure
   //mfem::Solver *SpInvPC = nullptr; 

   mfem::Solver *MtInvPC = nullptr;
   mfem::CGSolver *MtInv = nullptr;  
   mfem::Solver *HtInvPC = nullptr;
   mfem::CGSolver *HtInv = nullptr;  

   Vector fn, un, un_next, unm1, unm2;
   Vector uBn, uBnm1, uBnm2;  
   Vector Nun, Nunm1, Nunm2;
   Vector Fext, FText, Lext;          
   Vector resu, tmpR1;
   Vector FBext;

   Vector pn, resp, FText_bdr, g_bdr;
   Vector tmpR0PM1;

   Vector pnBig;
  
   ParGridFunction un_gf, un_next_gf, curlu_gf, curlcurlu_gf,
                   Lext_gf, FText_gf, resu_gf;

   //ParGridFunction rn_gf, resr_gf;  
   ParGridFunction pn_gf, resp_gf;
   ParGridFunction sml_gf, big_gf;

   // temperature additions
   Vector fTn, Tn, Tn_next, Tnm1, Tnm2, NTn, NTnm1, NTnm2;
   Vector Text, Text_bdr, t_bdr;
   Vector resT, tmpR0;
   ParGridFunction Tn_gf, Tn_next_gf, Text_gf, resT_gf;

   // density, not actually solved for
   Vector rn;
   ParGridFunction rn_gf;

   // swap spaces
   ParGridFunction R0PM0_gf;
   ParGridFunction R0PM1_gf;
   ParGridFunction R1PM0_gf;  
   ParGridFunction R1PX2_gf;    
  
   // All essential attributes.
   Array<int> vel_ess_attr;
   Array<int> pres_ess_attr;
   Array<int> temp_ess_attr;  

   // All essential true dofs.
   Array<int> vel_ess_tdof;
   Array<int> pres_ess_tdof;
   Array<int> temp_ess_tdof;

   // Bookkeeping for velocity dirichlet bcs.
   std::vector<VelDirichletBC_T> vel_dbcs;

   // Bookkeeping for pressure dirichlet bcs.
   std::vector<PresDirichletBC_T> pres_dbcs;

   // Bookkeeping for temperature dirichlet bcs.
   std::vector<TempDirichletBC_T> temp_dbcs;
  
   // Bookkeeping for acceleration (forcing) terms.
   std::vector<AccelTerm_T> accel_terms;

   int max_bdf_order = 3;
   //int max_bdf_order = 2;
   //int max_bdf_order = 1;
   int cur_step = 0;
   std::vector<double> dthist = {0.0, 0.0, 0.0};

   // BDFk/EXTk coefficients.
   double bd0 = 0.0;
   double bd1 = 0.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 0.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   // Timers.
   StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

   // Print levels.
   int pl_mvsolve = 0;
   int pl_spsolve = 0;
   int pl_hsolve = 0;
   int pl_amg = 0;
   int pl_mtsolve = 0;  

   // Relative tolerances.
   double rtol_spsolve = 1e-6;
   double rtol_hsolve = 1e-8;
   double rtol_htsolve = 1e-6;  

   // Iteration counts.
   int iter_mvsolve = 0, iter_spsolve = 0, iter_hsolve = 0;
   int iter_mtsolve = 0, iter_htsolve = 0;  

   // Residuals.
   double res_mvsolve = 0.0, res_spsolve = 0.0, res_hsolve = 0.0;
   double res_mtsolve = 0.0, res_htsolve = 0.0;  

   // LOR related.
   ParLORDiscretization *lor = nullptr;

   // Filter-based stabilization => move to "input" options
   int filter_cutoff_modes = 1;
   double filter_alpha = 0.0;
   //double filter_alpha = 1.0;
  
   FiniteElementCollection *vfec_filter = nullptr;
   ParFiniteElementSpace *vfes_filter = nullptr;
   ParGridFunction un_NM1_gf;
   ParGridFunction un_filtered_gf;
  
   FiniteElementCollection *tfec_filter = nullptr;
   ParFiniteElementSpace *tfes_filter = nullptr;
   ParGridFunction Tn_NM1_gf;
   ParGridFunction Tn_filtered_gf;

   // stuff that really shouldnt be part of the solver but M2ulPhys is a disaster
   /*
   RunConfiguration config;
   std::ofstream histFile;
   int dim;
   int nvel;
   int num_equation;  
   ParaViewDataCollection *paraviewColl = NULL;
   double hmin;
   double hmax;
   int exit_status_;  
   IODataOrganizer ioData;
   */

   // Equations solved
   Equations eqSystem;

   // order of polynomials for auxiliary solution
   bool loadFromAuxSol;
  
   // Pointers to the different classes
   GasMixture *mixture;    // valid on host
   GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture  

   TransportProperties *transportPtr = NULL;  // valid on both host and device
   // TransportProperties *d_transport = NULL;  // valid on device, when available; otherwise = transportPtr

   Chemistry *chemistry_ = NULL;
   Radiation *radiation_ = NULL;
  
   // to interface with existing code
   ParGridFunction *U;
   ParGridFunction *Up;  

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> *offsets;
   BlockVector *u_block;
   BlockVector *up_block;

   // paraview collection pointer
   ParaViewDataCollection *paraviewColl = NULL;
   // DataCollection *visitColl = NULL;

   // Visualization functions (these are pointers to Up)
   ParGridFunction *temperature, *dens, *vel, *vtheta, *passiveScalar;
   ParGridFunction *electron_temp_field;
   ParGridFunction *press;
   std::vector<ParGridFunction *> visualizationVariables_;
   std::vector<std::string> visualizationNames_;
   AuxiliaryVisualizationIndexes visualizationIndexes_;
   ParGridFunction *plasma_conductivity_;
   ParGridFunction *joule_heating_;

   // I/O organizer
   IODataOrganizer ioData;
  
  
public:
   /// Initialize data structures, set FE space order and kinematic viscosity.
   /**
    * The ParMesh @a mesh can be a linear or curved parallel mesh. The @a order
    * of the finite element spaces is this algorithm is of equal order
    * \f$(P_N)^d P_N\f$ for velocity and pressure respectively. This means the
    * pressure is in discretized in the same space (just scalar instead of a
    * vector space) as the velocity.
    *
    * Kinematic viscosity (dimensionless) is set using @a kin_vis and
    * automatically converted to the Reynolds number. If you want to set the
    * Reynolds number directly, you can provide the inverse.
    */
  LoMachSolver(mfem::MPI_Session &mpi, LoMachOptions loMach_opts, TPS::Tps *tps);
  virtual ~LoMachSolver() {}
  //(ParMesh *mesh, int order, int porder, int norder, double kin_vis, double Po, double Rgas);

   void initialize();
   void parseSolverOptions() override;
   void parseSolverOptions2();
   void parsePeriodicInputs();
   void parseFlowOptions();
   void parseTimeIntegrationOptions();
   void parseStatOptions();
   void parseIOSettings();
   void parseRMSJobOptions();  
   void parseBCInputs();
   void parseICOptions();
   void parsePostProcessVisualizationInputs();
   void parseFluidPreset();  
   void initSolutionAndVisualizationVectors();
   void initialTimeStep();  
   void solve();
   void updateU();
   void copyU();  

   // i/o routines
   void read_partitioned_soln_data(hid_t file, string varName, size_t index, double *data);
   void read_serialized_soln_data(hid_t file, string varName, int numDof, int varOffset, double *data, IOFamily &fam);
   void restart_files_hdf5(string mode, string inputFileName = std::string());
   void partitioning_file_hdf5(string mode);
   // void serialize_soln_for_write(IOFamily &fam);
   void write_soln_data(hid_t group, string varName, hid_t dataspace, double *data);

   void projectInitialSolution();  
   void writeHDF5(string inputFileName = std::string()) { restart_files_hdf5("write", inputFileName); }
   void readHDF5(string inputFileName = std::string()) { restart_files_hdf5("read", inputFileName); }
   void writeParaview(int iter, double time) {
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
  }
 
  
   /// Initialize forms, solvers and preconditioners.
   void Setup(double dt);

   /// Compute solution at the next time step t+dt.
   /**
    * This method can be called with the default value @a provisional which
    * always accepts the computed time step by automatically calling
    * UpdateTimestepHistory.
    *
    * If @a provisional is set to true, the solution at t+dt is not accepted
    * automatically and the application code has to call UpdateTimestepHistory
    * and update the @a time variable accordingly.
    *
    * The application code can check the provisional step by retrieving the
    * GridFunction with the method GetProvisionalVelocity. If the check fails,
    * it is possible to retry the step with a different time step by not
    * calling UpdateTimestepHistory and calling this method with the previous
    * @a time and @a cur_step.
    *
    * The method and parameter choices are based on [1].
    *
    * [1] D. Wang, S.J. Ruuth (2008) Variable step-size implicit-explicit
    * linear multistep methods for time-dependent partial differential
    * equations
    */
   void Step(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);

   /// Return a pointer to the provisional velocity ParGridFunction.
   ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

   /// Return a pointer to the current velocity ParGridFunction.
   ParGridFunction *GetCurrentVelocity() { return &un_gf; }

   /// Return a pointer to the current pressure ParGridFunction.
   ParGridFunction *GetCurrentPressure() { return &pn_gf; }

   /// Return a pointer to the current temperature ParGridFunction.
   ParGridFunction *GetCurrentTemperature() { return &Tn_gf; }

   /// Return a pointer to the current density ParGridFunction.  
   ParGridFunction *GetCurrentDensity() { return &rn_gf; }  

   /// Return a pointer to the current temperature ParGridFunction.
   //ParGridFunction *GetCurrentDensity() { return &rn_gf; }    

   LoMachOptions GetOptions() { return loMach_opts_; }
  
   /// Add a Dirichlet boundary condition to the velocity field.
   void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
   void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the pressure field.
   void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

   /// Add a Dirichlet boundary condition to the temperature field.
   void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);  

   /// Add an acceleration term to the RHS of the equation.
   /**
    * The VecFuncT @a f is evaluated at the current time t and extrapolated
    * together with the nonlinear parts of the Navier Stokes equation.
    */
   void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);
   void AddAccelTerm(VecFuncT *f, Array<int> &attr);

   /// Enable partial assembly for every operator.
   void EnablePA(bool pa) { partial_assembly = pa; }

   /// Enable numerical integration rules. This means collocated quadrature at
   /// the nodal points.
   void EnableNI(bool ni) { numerical_integ = ni; }

   /// Print timing summary of the solving routine.
   /**
    * The summary shows the timing in seconds in the first row of
    *
    * 1. SETUP: Time spent for the setup of all forms, solvers and
    *    preconditioners.
    * 2. STEP: Time spent computing a full time step. It includes all three
    *    solves.
    * 3. EXTRAP: Time spent for extrapolation of all forcing and nonlinear
    *    terms.
    * 4. CURLCURL: Time spent for computing the curl curl term in the pressure
    *    Poisson equation (see references for detailed explanation).
    * 5. PSOLVE: Time spent in the pressure Poisson solve.
    * 6. HSOLVE: Time spent in the Helmholtz solve.
    *
    * The second row shows a proportion of a column relative to the whole
    * time step.
    */
   void PrintTimingData();

  //~LoMachSolver();

   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^2\f$.
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);

   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^3\f$.
   void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);

   /// Remove mean from a Vector.
   /**
    * Modify the Vector @a v by subtracting its mean using
    * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
    */
   void Orthogonalize(Vector &v);

   /// Remove the mean from a ParGridFunction.
   /**
    * Modify the ParGridFunction @a v by subtracting its mean using
    * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
    */
   void MeanZero(ParGridFunction &v);

   /// Rotate entries in the time step and solution history arrays.
   void UpdateTimestepHistory(double dt);

   /// Set the maximum order to use for the BDF method.
   void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; };

   /// Compute CFL
   double ComputeCFL(ParGridFunction &u, double dt);

   /// Enforce CFL
   void EnforceCFL(double maxCFL, ParGridFunction &u, double &dt);  

   /// Set the number of modes to cut off in the interpolation filter
   void SetCutoffModes(int c) { filter_cutoff_modes = c; }

   /// Set the interpolation filter parameter @a a
   /**
    * If @a a is > 0, the filtering algorithm for the velocity field after every
    * time step from [1] is used. The parameter should be 0 > @a >= 1.
    *
    * [1] Paul Fischer, Julia Mullen (2001) Filter-based stabilization of
    * spectral element methods
    */
   void SetFilterAlpha(double a) { filter_alpha = a; }
  
};  

#endif
  
/// Transient incompressible Navier Stokes solver in a split scheme formulation.
/**
 * This implementation of a transient incompressible Navier Stokes solver uses
 * the non-dimensionalized formulation. The coupled momentum and
 * incompressibility equations are decoupled using the split scheme described in
 * [1]. This leads to three solving steps.
 *
 * 1. An extrapolation step for all nonlinear terms which are treated
 *    explicitly. This step avoids a fully coupled nonlinear solve and only
 *    requires a solve of the mass matrix in velocity space \f$M_v^{-1}\f$. On
 *    the other hand this introduces a CFL stability condition on the maximum
 *    timestep.
 *
 * 2. A Poisson solve \f$S_p^{-1}\f$.
 *
 * 3. A Helmholtz like solve \f$(M_v - \partial t K_v)^{-1}\f$.
 *
 * The numerical solver setup for each step are as follows.
 *
 * \f$M_v^{-1}\f$ is solved using CG with Jacobi as preconditioner.
 *
 * \f$S_p^{-1}\f$ is solved using CG with AMG applied to the low order refined
 * (LOR) assembled pressure Poisson matrix. To avoid assembling a matrix for
 * preconditioning, one can use p-MG as an alternative (NYI).
 *
 * \f$(M_v - \partial t K_v)^{-1}\f$ due to the CFL condition we expect the time
 * step to be small. Therefore this is solved using CG with Jacobi as
 * preconditioner. For large time steps a preconditioner like AMG or p-MG should
 * be used (NYI).
 *
 * Statements marked with NYI mean this feature is planned but Not Yet
 * Implemented.
 *
 * A detailed description is available in [1] section 4.2. The algorithm is
 * originated from [2].
 *
 * [1] Michael Franco, Jean-Sylvain Camier, Julian Andrej, Will Pazner (2020)
 * High-order matrix-free incompressible flow solvers with GPU acceleration and
 * low-order refined preconditioners (https://arxiv.org/abs/1910.03032)
 *
 * [2] A. G. Tomboulides, J. C. Y. Lee & S. A. Orszag (1997) Numerical
 * Simulation of Low Mach Number Reactive Flows
 */
