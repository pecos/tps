#ifndef BC_INTEGRATOR
#define BC_INTEGRATOR

#include "mfem.hpp"
#include "riemann_solver.hpp"
#include "equation_of_state.hpp"
#include "unordered_map"
#include "BoundaryCondition.hpp"
#include "run_configuration.hpp"

using namespace mfem;

// Boundary face term: <F.n(u),[w]>
class BCintegrator : public NonlinearFormIntegrator
{
protected:
  RunConfiguration &config;
  
  RiemannSolver *rsolver;
  EquationOfState *eqState;
   
  const int dim;
  const int num_equation;

  double &max_char_speed;
  IntegrationRules *intRules;

  Mesh *mesh;

  std::unordered_map<int,BoundaryCondition*> BCmap;

  //void calcMeanState();
  void computeState(const int attr, Vector &stateIn, Vector &stateOut);

public:
   BCintegrator(Mesh *_mesh,
                IntegrationRules *_intRules,
                RiemannSolver *rsolver_, 
                EquationOfState *_eqState,
                const int _dim,
                const int _num_equation,
                double &_max_char_speed,
                RunConfiguration &_runFile );
   ~BCintegrator();

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

#endif // BC_INTEGRATOR
