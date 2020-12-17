#ifndef M2ulPhyS_CLASS
#define M2ulPhyS_CLASS
/*
 *  MFEM MULTY PHYSICS SOLVER (M2ulPhyS)
 * This class contains all necessary functions and classes
 * reuired to run example 18 without the need of having global
 * variables and functions.
 */

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "rhs_operator.hpp"
#include "face_integrator.hpp"
#include "riemann_solver.hpp"
#include "domain_integrator.hpp"

using namespace mfem;
using namespace std;

class M2ulPhyS
{
private:
  // Number of simensions
  const int dim;
  
  // Number of equations
  int num_equation;
  
  // order of solution polynomials
  const int order;
  
  // Equations solved
  const Equations eqSystem;

  // Maximum characteristic speed (updated by integrators)
  double max_char_speed;
  
  // reference to mesh
  Mesh &mesh;
  
  // time integrator
  ODESolver *timeIntegrator;
  
  // Pointers to the different classes
  EquationOfState *eqState;
  
  Fluxes *fluxClass;
  
  RHSoperator *rhsOperator;
  
  // Finite element collection
  DG_FECollection *fec;
  
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

  // Momentum grid function on dfes for visualization.
  //GridFunction mom;
  
  // Riemann Solver
  RiemannSolver *rsolver;
  
  // RHS operators
  NonlinearForm *A;
  FaceIntegrator *faceIntegrator;
  
  MixedBilinearForm *Aflux;
  DomainIntegrator *domainIntegrator;

  // State/solution vector
  GridFunction *sol;
  
  // time variable
  double time;
  
  // time step
  double dt;
  
  // time of end of simulation
  const double t_final;
  
  // Courant-Friedrich-Levy condition
  double CFL;
  
  // minimum element size
  double hmin;
  
  void initSolutionAndVisualizationVectors();
  static void InitialConditionTest(const Vector &x, Vector &y);
  static void uniformInitialConditions();
  

public:
  M2ulPhyS(Mesh &_mesh,
          int order,
          int solverType,
          double _t_final,
          Equations _eqSystem,
          WorkingFluid _fluid = DRY_AIR
         );
  ~M2ulPhyS();
  
  void projectInitialSolution();
  
  void Iterate();
  
  // Accessors
  RHSoperator getRHSoperator(){ return *rhsOperator; }
  
  
};

#endif // M2ulPhyS_CLASS
