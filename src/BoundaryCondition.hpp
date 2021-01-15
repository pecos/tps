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
  double &dt;
  const int dim;
  const int num_equation;
  const int patchNumber;
  
public:
  BoundaryCondition(RiemannSolver *_rsolver, 
                    EquationOfState *_eqState,
                    double &dt,
                    const int _dim,
                    const int _num_equation,
                    const int _patchNumber);
  ~BoundaryCondition();
  
  virtual void computeState(Vector &nor,
                            Vector &stateIn, 
                            Vector &stateOut) = 0;
};

#endif // BOUNDARY_CONDITION
