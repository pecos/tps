#ifndef RHS_OPERATOR
#define RHS_OPERATOR

#include "mfem.hpp"

#include "fluxes.hpp"
#include "equation_of_state.hpp"
#include "BCintegrator.hpp"
#include "forcing_terms.hpp"
#include "run_configuration.hpp"

using namespace mfem;


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class RHSoperator : public TimeDependentOperator
{
private:
  const int dim;
  
  const Equations &eqSystem;
  
  double &max_char_speed;
  const int &num_equation;
  
  IntegrationRules *intRules;
  const int intRuleType;
  
  Fluxes *fluxClass;
  EquationOfState *eqState;

  ParFiniteElementSpace *vfes;

  ParNonlinearForm *A;
  
  MixedBilinearForm *Aflux;
  
  ParMesh *mesh;
  
  //DenseTensor *Me_inv;
  DenseMatrix *Me_inv;
  
  const bool &isSBP;
  const double &alpha;

  mutable Vector *state;
  
  // reference to primitive varibales
  ParGridFunction *Up;
  
  // gradients of primitives and associated forms&FE space
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  ParNonlinearForm *gradUp_A;
  
  BCintegrator *bcIntegrator;
  
  ForcingTerms *forcing;

  void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;
  
  void updatePrimitives(const Vector &x) const;
  void calcGradientsPrimitives() const;

public:
   RHSoperator(const int _dim,
               const int &_num_equations,
               const int &_order,
               const Equations &_eqSystem,
               double &_max_char_speed,
               IntegrationRules *_intRules,
               int _intRuleType,
               Fluxes *_fluxClass,
               EquationOfState *_eqState,
               ParFiniteElementSpace *_vfes,
               ParNonlinearForm *_A, 
               MixedBilinearForm *_Aflux,
               ParMesh *_mesh,
               ParGridFunction *_Up,
               ParGridFunction *_gradUp,
               ParFiniteElementSpace *_gradUpfes,
               ParNonlinearForm *_gradUp_A,
               BCintegrator *_bcIntegrator,
               bool &_isSBP,
               double &_alpha,
               RunConfiguration &_config
              );

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~RHSoperator();
};

#endif // RHS_OPERATOR
