#ifndef BOUNDARY_CONDITION
#define BOUNDARY_CONDITION

#include "mfem.hpp"
#include "riemann_solver.hpp"
#include "equation_of_state.hpp"

using namespace mfem;

class BoundaryCondition
{
protected:
  RiemannSolver *rsolver;
  EquationOfState *eqState;
  FiniteElementSpace *vfes;
  IntegrationRules *intRules;
  double &dt;
  const int dim;
  const int num_equation;
  const int patchNumber;
  
public:
  BoundaryCondition(RiemannSolver *_rsolver, 
                    EquationOfState *_eqState,
                    FiniteElementSpace *_vfes,
                    IntegrationRules *_intRules,
                    double &dt,
                    const int _dim,
                    const int _num_equation,
                    const int _patchNumber);
  ~BoundaryCondition();
  
  virtual void computeBdrFlux(Vector &normal,
                              Vector &stateIn, 
                              DenseMatrix &gradState,
                              Vector &bdrFlux) = 0;
                              
  virtual void updateMean(IntegrationRules *intRules,
                          GridFunction *Up) = 0;
};

#endif // BOUNDARY_CONDITION
