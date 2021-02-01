#ifndef FLUXES  
#define FLUXES  

/*
 * Class that deals with flux operations
 */

#include "mfem.hpp"
#include "equation_of_state.hpp"

enum Equations { EULER, NS, MHD};

using namespace mfem;

class Fluxes
{
private:
  EquationOfState *eqState;
  
  Equations &eqSystem;
  
  const int &dim;
  
  const int &num_equations;
  
public:
  Fluxes(EquationOfState *_eqState,
         Equations &_eqSystem,
         const int &_num_equations,
         const int &_dim
        );
  
  void ComputeTotalFlux(const Vector &state,
                        const DenseMatrix &gradUp, 
                        DenseMatrix &flux);
  
  void ComputeConvectiveFluxes(const Vector &state,
                               DenseMatrix &flux );
  
  void ComputeViscousFluxes(const Vector &state,
                            const DenseMatrix &gradUp, 
                            DenseMatrix &flux);
  
  // Compute the split fersion of the flux for SBP operations
  // Output matrices a_mat, c_mat need not have the right size
  void ComputeSplitFlux(const Vector &state, 
                        DenseMatrix &a_mat, 
                        DenseMatrix &c_mat);
};

#endif // FLUXES  
