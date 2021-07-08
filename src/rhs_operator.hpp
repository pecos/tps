#ifndef RHS_OPERATOR
#define RHS_OPERATOR

#include <mfem.hpp>
#include <tps_config.h>
#include "fluxes.hpp"
#include "equation_of_state.hpp"
#include "BCintegrator.hpp"
#include "forcing_terms.hpp"
#include "run_configuration.hpp"
#include "gradients.hpp"
#include "dgNonlinearForm.hpp"
#include "gradNonLinearForm.hpp"

using namespace mfem;


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class RHSoperator : public TimeDependentOperator
{
private:
  Gradients *gradients;
  
  double &time;
  
  const int dim;
  
  const Equations &eqSystem;
  
  double &max_char_speed;
  const int &num_equation;
  
  IntegrationRules *intRules;
  const int intRuleType;
  
  Fluxes *fluxClass;
  EquationOfState *eqState;

  ParFiniteElementSpace *vfes;
  
  // nodes IDs and indirection array
  Array<int> &nodesIDs;
  Array<int> &posDofIds;
  // count of number of elements of each type
  Array<int> &numElems;
  
  int *h_numElems;
  int *h_posDofIds;
  
  Vector &shapeWnor1;
  Vector &shape2;
  Array<int> &elemFaces;
  Array<int> &elems12Q;
  const int &maxIntPoints;
  const int &maxDofs;

  DGNonLinearForm *A;
  
  MixedBilinearForm *Aflux;
  
  ParMesh *mesh;
  
  Array< DenseMatrix* > Me_inv;
  Vector invMArray;
  Array<int> posDofInvM;
  
  const bool &isSBP;
  const double &alpha;
  
  // reference to primitive varibales
  ParGridFunction *Up;
  
  // gradients of primitives and associated forms&FE space
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  //ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;
  
  BCintegrator *bcIntegrator;
  
  Array<ForcingTerms*> forcing;
  
  mutable DenseTensor flux;
  mutable Vector z;

  //void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;
  void GetFlux(const Vector &state, DenseTensor &flux) const;

public:
   RHSoperator(double &_time,
               const int _dim,
               const int &_num_equations,
               const int &_order,
               const Equations &_eqSystem,
               double &_max_char_speed,
               IntegrationRules *_intRules,
               int _intRuleType,
               Fluxes *_fluxClass,
               EquationOfState *_eqState,
               ParFiniteElementSpace *_vfes,
               Array<int> &_nodesIDs,
               Array<int> &_posDofIds,
               Array<int> &_numElems,
               Vector &_shapeWnor1,
               Vector &_shape2,
               Array<int> &_elemFaces,
               Array<int> &_elems12Q,
               const int &_maxIntPoints,
               const int &_maxDofs,
               //ParNonlinearForm *_A,
               DGNonLinearForm *_A,
               MixedBilinearForm *_Aflux,
               ParMesh *_mesh,
               ParGridFunction *_Up,
               ParGridFunction *_gradUp,
               ParFiniteElementSpace *_gradUpfes,
               //ParNonlinearForm *_gradUp_A,
               GradNonLinearForm *_gradUp_A,
               BCintegrator *_bcIntegrator,
               bool &_isSBP,
               double &_alpha,
               RunConfiguration &_config
              );

   virtual void Mult(const Vector &x, Vector &y) const;
   void updatePrimitives(const Vector &x) const;

   virtual ~RHSoperator();


  // GPU functions
  static void copyZk2Z_gpu(Vector &z,Vector &zk,const int eq,const int dof);
  static void copyDataForFluxIntegration_gpu(const Vector &z,
					 DenseTensor &flux,
					 Vector &fk,
					 Vector &zk,
					 const int eq,
					 const int dof,
					 const int dim);
  static void updatePrimitives_gpu(Vector *Up, 
				   const Vector *x_in,
				   const double gamma,
				   const int ndofs,
				   const int dim,
				   const int num_equations );
  static void multiPlyInvers_gpu(Vector &y,
				 Vector &z,
				 const Array<int> &nodesIDs,
				 const Array<int> &posDofIds,
				 const Vector &invMArray,
				 const Array<int> &posDofInvM,
				 const int num_equation,
				 const int totNumDof,
				 const int NE,
				 const int elemOffset,
				 const int dof);
};

#endif // RHS_OPERATOR
