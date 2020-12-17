#ifndef RHS_OPERATOR
#define RHS_OPERATOR

#include "mfem.hpp"

#include "fluxes.hpp"
#include "equation_of_state.hpp"

using namespace mfem;

enum Equations { EULER, NS, MHD};


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class RHSoperator : public TimeDependentOperator
{
private:
   const int dim;
   
   const Equations &eqSystem;
   
   double &max_char_speed;
   int num_equation;
   
   Fluxes *fluxClass;
   EquationOfState *eqState;

   FiniteElementSpace *vfes;

   NonlinearForm *A;
   
   MixedBilinearForm *Aflux;
   
   DenseTensor *Me_inv;

   mutable Vector *state;
//    mutable DenseMatrix f;
//    mutable DenseTensor flux;
//    mutable Vector z;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   RHSoperator(const int _dim,
               const Equations &_eqSystem,
               double &_max_char_speed,
               Fluxes *_fluxClass,
               EquationOfState *_eqState,
               FiniteElementSpace *_vfes,
               NonlinearForm *_A, 
               MixedBilinearForm *_Aflux);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~RHSoperator();
};

#endif // RHS_OPERATOR
