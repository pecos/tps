#ifndef DOMAIN_INTEGRATOR
#define DOMAIN_INTEGRATOR

/*
 * Class that includes domain integrators
 * MFEM library requires this to be its own class
 */

#include <mfem.hpp>
#include <tps_config.h>
#include "fluxes.hpp"

using namespace mfem;
using namespace std;

// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
  
  Fluxes *fluxClass;
  
   
   const int dim;
   const int num_equation;
   IntegrationRules *intRules;
   const int intRuleType;

public:
   DomainIntegrator(Fluxes *_fluxClass,
                    IntegrationRules *_intRules,
                    int _intRuleType,
                    const int _dim, 
                    const int _num_equation);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

#endif // DOMAIN_INTEGRATOR
