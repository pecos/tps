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
  
  // Compute the split fersion of the flux for SBP operations
  // Output matrices a_mat, c_mat need not have the right size
  void ComputeSplitFlux(const Vector &state, 
                        DenseMatrix &a_mat, 
                        DenseMatrix &c_mat);
};

#endif // FLUXES  
