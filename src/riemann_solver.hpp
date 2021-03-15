#ifndef RIEMANN_SOLVER
#define RIEMANN_SOLVER

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"

using namespace mfem;

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
  const int &num_equation;
  
  Vector flux1;
  Vector flux2;

  EquationOfState *eqState;
  Fluxes *fluxClass;
  
  void Eval_LF(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux); 
  void Eval_Roe(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux); 

public:
   RiemannSolver(int &_num_equation,
                 EquationOfState *_eqState,
                 Fluxes *_fluxClass );
   
   void Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux, bool LF=false);
   
   void ComputeFluxDotN(const Vector &state, 
                        const Vector &nor,
                        Vector &fluxN);
};

#endif // RIEMANN_SOLVER
