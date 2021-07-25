#ifndef M2ulPhyS_CLASS
#define M2ulPhyS_CLASS
/*
 *  MFEM MULTY PHYSICS SOLVER (M2ulPhyS)
 * This class contains all necessary functions and classes
 * reuired to run example 18 without the need of having global
 * variables and functions.
 */

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>
#include "mpi_groups.hpp"
#include "run_configuration.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "rhs_operator.hpp"
#include "face_integrator.hpp"
#include "riemann_solver.hpp"
#include "domain_integrator.hpp"
#include "sbp_integrators.hpp"
#include "BCintegrator.hpp"
#include "faceGradientIntegration.hpp"
#include "averaging_and_rms.hpp"

#include "dgNonlinearForm.hpp"
#include "gradNonLinearForm.hpp"

#ifdef _MASA_
#include "masa_handler.hpp"
#endif

#ifdef HAVE_GRVY
#include "grvy.h"
#endif

using namespace mfem;
using namespace std;

// application exit codes
enum ExitCodes {NORMAL = 0, ERROR = 1, JOB_RESTART = 10 };

class M2ulPhyS
{
private:
  MPI_Groups *groupsMPI;
  MPI_Session &mpi;
  int  nprocs_;			// total number of MPI procs
  int  rank_;			// local MPI rank
  bool rank0_;			// flag to indicate rank 0
  
  // Run options
  RunConfiguration config;
  
  // Number of simensions
  int dim;
  
  // Number of equations
  int num_equation;
  
  // order of polynomials
  int order;
  // total number of mesh elements (serial)
  int nelemGlobal_;
  
  // order of polynomials for auxiliary solution
  bool loadFromAuxSol;

  // Equations solved
  Equations eqSystem;

  // Maximum characteristic speed (updated by integrators)
  double max_char_speed;
  
  // reference to mesh
  ParMesh *mesh;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;
  
  // time integrator
  ODESolver *timeIntegrator;
  
  // Pointers to the different classes
  EquationOfState *eqState;
  
  ParGridFunction *spaceVaryViscMult; // space varying viscosity multiplier
  
  Fluxes *fluxClass;
  
  RHSoperator *rhsOperator;

  // Integration rule
  int intRuleType; // 0: GaussLegendre; 1: GaussLobatto
  IntegrationRules *intRules;
  
  // Interpolant function
  // 0: GaussLegendre; 1: GaussLobatto
  int basisType;
  
  // Finite element collection
  //DG_FECollection *fec;
  //H1_FECollection *fec;
  FiniteElementCollection *fec;
  
  // Finite element space for a scalar (thermodynamic quantity)
  ParFiniteElementSpace *fes;
  
  // Finite element space for a mesh-dim vector quantity (momentum)
  ParFiniteElementSpace *dfes;
  
  // Finite element space for all variables together (total thermodynamic state)
  ParFiniteElementSpace *vfes;
  
  // nodes IDs and indirection array
  Array<int> nodesIDs;
  Array<int> posDofIds;
  // count of number of elements of each type
  Array<int> numElems;
  //Array<int> posDofQshape1; // position, num. dof and integration points for each face
  Vector shapeWnor1; // shape functions, weight and normal for each face at ach integration point
  //Array<int> posDofshape2;
  Vector shape2;
  const int maxIntPoints = 49; // corresponding to QUAD face with p=5
  //const int maxDofs = 64;      // corresponding to HEX with p=5
  const int maxDofs = 216;      // corresponding to HEX with p=5
  
  Array<int> elemFaces; // number and faces IDs of each element
  Array<int> elems12Q; // elements connecting a face and integration points
  
  // BC integration
  Vector shapesBC;
  Vector normalsWBC;
  Array<int> intPointsElIDBC; // integration points and element ID
  
  
  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets;
  BlockVector *u_block;
  BlockVector *up_block;
  
  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;
  //DataCollection *visitColl = NULL;
  
  // Riemann Solver
  RiemannSolver *rsolver;
  
  // RHS operators
  //ParNonlinearForm *A;
  DGNonLinearForm *A;
  
  FaceIntegrator *faceIntegrator;
  SBPintegrator  *SBPoperator;
  
  MixedBilinearForm *Aflux;
  DomainIntegrator *domainIntegrator;
  
  // Boundary condition non-linear integrator
  BCintegrator *bcIntegrator;
  
  // Coefficient for SBP operators and 
  bool isSBP;
  double alpha;

  // Conservative variables
  ParGridFunction *U;
  
  // Primitive variables
  ParGridFunction *Up;
  
  // Visualization functions (these are pointers to Up)
  ParGridFunction *press, *dens, *vel;
  
  // gradient of primitive variables
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  //ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;
  
  // Average handler
  Averaging *average;
  
  // time variable
  double time;
  
  // time step
  double dt;
  
  // iteration
  int iter;
  
  // time of end of simulation
  double t_final;
  int MaxIters;
  
  // Courant-Friedrich-Levy condition
  double CFL;
  
  // Reference length
  double refLength;
  
  // minimum element size
  double hmin;

  // exit status code;
  int exit_status_;

  // mapping from local to global element index
  int* locToGlobElem;

  // a serial mesh, finite element space, and grid function
  // for use if we want to write a serial file
  Mesh *serial_mesh;
  FiniteElementSpace *serial_fes;
  GridFunction *serial_soln;


  void getAttributesInPartition(Array<int> &local_attr);
  
  void initIndirectionArrays();
  
  void initVariables();
  void initSolutionAndVisualizationVectors();
  void initialTimeStep();

#ifdef _MASA_
  static void MASA_exactSoln(const Vector& x, double tin, Vector& y);
#endif
  static void InitialConditionEulerVortex(const Vector &x, Vector &y);
  static void testInitialCondition(const Vector &x, Vector &y);
  void uniformInitialConditions();
  void initGradUp();

  // i/o routines
  void write_restart_files();
  void read_restart_files();
  void restart_files_hdf5(string mode);
  void partitioning_file_hdf5(string mode);
  void serialize_soln_for_write();
  
  void Check_NAN();
  bool Check_JobResubmit();
  void Cache_Paraview_Timesteps();


public:
  M2ulPhyS(MPI_Session &_mpi,
           string &inputFileName );
  ~M2ulPhyS();
  
  void projectInitialSolution();
  
  void Iterate();

  void Header();
  
  // Accessors
  RHSoperator getRHSoperator(){ return *rhsOperator; }

  static int Check_NaN_GPU(ParGridFunction *U, int lengthU);

  // Exit code access
  void SetStatus(int code){exit_status_=code; return;}
  int  GetStatus(){return exit_status_;}
  
};

#endif // M2ulPhyS_CLASS
