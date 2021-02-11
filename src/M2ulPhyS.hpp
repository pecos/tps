#ifndef M2ulPhyS_CLASS
#define M2ulPhyS_CLASS
/*
 *  MFEM MULTY PHYSICS SOLVER (M2ulPhyS)
 * This class contains all necessary functions and classes
 * reuired to run example 18 without the need of having global
 * variables and functions.
 */

#include "mfem.hpp"
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

using namespace mfem;
using namespace std;

class M2ulPhyS
{
private:
  MPI_Groups *groupsMPI;
  MPI_Session &mpi;
  
  // Run options
  RunConfiguration config;
  
  // Number of simensions
  int dim;
  
  // Number of equations
  int num_equation;
  
  // order of solution polynomials
  int order;
  
  // Equations solved
  Equations eqSystem;

  // Maximum characteristic speed (updated by integrators)
  double max_char_speed;
  
  // reference to mesh
  ParMesh *mesh;
  
  // time integrator
  ODESolver *timeIntegrator;
  
  // Pointers to the different classes
  EquationOfState *eqState;
  
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
  ParNonlinearForm *A;
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
  
  // Visualization functions
  ParGridFunction *press, *dens, *vel;
  
  // gradient of primitive variables
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  ParNonlinearForm *gradUp_A;
  
//   // Gradients Up
//   Array<double> gradUp;
  
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
  
  // minimum element size
  double hmin;
  
  void getAttributesInPartition(Array<int> &local_attr);
  
  void initVariables();
  void initSolutionAndVisualizationVectors();
  
  static void InitialConditionEulerVortex(const Vector &x, Vector &y);
  static void testInitialCondition(const Vector &x, Vector &y);
  void uniformInitialConditions();
  void initGradUp();
  
  void write_restart_files();
  void read_restart_files();
  

public:
  M2ulPhyS(MPI_Session &_mpi,
           string &inputFileName );
  ~M2ulPhyS();
  
  void projectInitialSolution();
  
  void Iterate();
  
  // Accessors
  RHSoperator getRHSoperator(){ return *rhsOperator; }
  
  
};

#endif // M2ulPhyS_CLASS
