#ifndef M2ulPhyS_CLASS
#define M2ulPhyS_CLASS
/*
 *  MFEM MULTY PHYSICS SOLVER (M2ulPhyS)
 * This class contains all necessary functions and classes
 * reuired to run example 18 without the need of having global
 * variables and functions.
 */

#include "mfem.hpp"
#include "run_configuration.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "rhs_operator.hpp"
#include "face_integrator.hpp"
#include "riemann_solver.hpp"
#include "domain_integrator.hpp"
#include "sbp_integrators.hpp"

using namespace mfem;
using namespace std;

class M2ulPhyS
{
private:
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
  Mesh *mesh;
  
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
  FiniteElementSpace *fes;
  
  // Finite element space for a mesh-dim vector quantity (momentum)
  FiniteElementSpace *dfes;
  
  // Finite element space for all variables together (total thermodynamic state)
  FiniteElementSpace *vfes;
  
  
  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets;
  BlockVector *u_block;
  
  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;

  // Momentum grid function on dfes for visualization.
  //GridFunction mom;
  
  // Riemann Solver
  RiemannSolver *rsolver;
  
  // RHS operators
  NonlinearForm *A;
  FaceIntegrator *faceIntegrator;
  SBPintegrator  *SBPoperator;
  
  MixedBilinearForm *Aflux;
  DomainIntegrator *domainIntegrator;
  
  // Coefficient for SBP operators and 
  bool isSBP;
  double alpha;

  // State/solution vector
  GridFunction *sol;
  
  // time variable
  double time;
  
  // time step
  double dt;
  
  // time of end of simulation
  double t_final;
  int MaxIters;
  
  // Courant-Friedrich-Levy condition
  double CFL;
  
  // minimum element size
  double hmin;
  
  void initVariables();
  void initSolutionAndVisualizationVectors();
  
  static void InitialConditionEulerVortex(const Vector &x, Vector &y);
  static void testInitialCondition(const Vector &x, Vector &y);
  static void uniformInitialConditions();
  

public:
  M2ulPhyS(string &inputFileName );
  ~M2ulPhyS();
  
  void projectInitialSolution();
  
  void Iterate();
  
  // Accessors
  RHSoperator getRHSoperator(){ return *rhsOperator; }
  
  
};

#endif // M2ulPhyS_CLASS
