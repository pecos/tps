#ifndef FORCING_TERMS
#define FORCING_TERMS

#include "mfem.hpp"
#include "run_configuration.hpp"

#ifdef _MASA_
#include "masa.h"
#endif

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
  virtual void addForcingIntegrals(Vector &in);
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
  
  //virtual void addForcingIntegrals(Vector &in);
};

#ifdef _MASA_
// Manufactured Solution using MASA
class MASA_forcings: public ForcingTerms
{
private:
  double &time;
  
public:
  MASA_forcings(const int &_dim,
                const int &_num_equation,
                const int &_order,
                const int &_intRuleType,
                IntegrationRules *_intRules,
                ParFiniteElementSpace *_vfes,
                ParGridFunction *_Up,
                ParGridFunction *_gradUp,
                double &_time );

  // Terms do not need updating
  virtual void updateTerms();
};
#endif // _MASA_

#endif // FORCING_TERMS
