#ifndef RIEMANN_SOLVER
#define RIEMANN_SOLVER

#include "mfem.hpp"
#include "equation_of_state.hpp"

using namespace mfem;

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
  const int &num_equation;
  
  Vector flux1;
  Vector flux2;

  EquationOfState *eqState;

public:
   RiemannSolver(int &_num_equation,
                 EquationOfState *_eqState);
   
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
   
   void ComputeFluxDotN(const Vector &state, 
                        const Vector &nor,
                        Vector &fluxN);
};

#endif // RIEMANN_SOLVER
