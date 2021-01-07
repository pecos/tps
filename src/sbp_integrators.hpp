#ifndef SBP_INTEGRATOR
#define SBP_INTEGRATOR

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"

using namespace mfem;

// Volume integrals...DESCRIBE THE OPERATIONS
class SBPintegrator : public NonlinearFormIntegrator
{
private:
   
   const int dim;
   const int num_equation;
   
   const double &alpha;
   
   EquationOfState *eqState;
   Fluxes *fluxClass;
   
   IntegrationRules *intRules;

public:
   SBPintegrator( EquationOfState *_eqState,
                  Fluxes *_fluxClass,
                  IntegrationRules *_intRules,
                  const int _dim,
                  const int _num_equation,
                  double &_alpha
                );

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);
};

#endif // SBP_INTEGRATOR
