#ifndef FLUXES  
#define FLUXES  

/*
 * Class that deals with flux operations
 */

#include "mfem.hpp"
#include "equation_of_state.hpp"

using namespace mfem;

class Fluxes
{
private:
  EquationOfState *eqState;
  
public:
  Fluxes(EquationOfState *_eqState);
  void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux);
};

#endif // FLUXES  
