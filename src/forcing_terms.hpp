#ifndef FORCING_TERMS
#define FORCING_TERMS

#include "mfem.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

class ForcingTerms
{
protected:
  const int &dim;
  const int &num_equation;
  const int &order;
  const int &intRuleType;
  IntegrationRules *intRules;
  ParFiniteElementSpace *vfes;
  ParGridFunction *Up;
  ParGridFunction *gradUp;
  
  
  // added term
  ParGridFunction *b;
  
public:
  ForcingTerms( const int &_dim,
                const int &_num_equation,
                const int &_order,
                const int &_intRuleType,
                IntegrationRules *_intRules,
                ParFiniteElementSpace *_vfes,
                ParGridFunction *_Up,
                ParGridFunction *_gradUp );
  ~ForcingTerms();
  
  virtual void updateTerms() = 0;
  virtual void addForcingIntegrals(Vector &in) = 0;
};

// Constant pressure gradient term 
class ConstantPressureGradient: public ForcingTerms
{
private:
  //RunConfiguration &config;
  double pressGrad[3];
  
public:
  ConstantPressureGradient( const int &_dim,
                            const int &_num_equation,
                            const int &_order,
                            const int &_intRuleType,
                            IntegrationRules *_intRules,
                            ParFiniteElementSpace *_vfes,
                            ParGridFunction *_Up,
                            ParGridFunction *_gradUp,
                            RunConfiguration &_config );
  
  // Terms do not need updating
  virtual void updateTerms();
  
  virtual void addForcingIntegrals(Vector &in);
};

#endif // FORCING_TERMS
