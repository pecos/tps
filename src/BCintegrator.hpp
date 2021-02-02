#ifndef BC_INTEGRATOR
#define BC_INTEGRATOR

#include "mfem.hpp"
#include "riemann_solver.hpp"
#include "equation_of_state.hpp"
#include "unordered_map"
#include "BoundaryCondition.hpp"
#include "run_configuration.hpp"
#include "fluxes.hpp"

using namespace mfem;

// Boundary face term: <F.n(u),[w]>
class BCintegrator : public NonlinearFormIntegrator
{
protected:
  RunConfiguration &config;
  
  RiemannSolver *rsolver;
  EquationOfState *eqState;
  Fluxes *fluxClass;
   
  const int dim;
  const int num_equation;

  double &max_char_speed;
  IntegrationRules *intRules;

  ParMesh *mesh;
  
  // pointer to finite element space
  ParFiniteElementSpace *vfes;
  
  // pointer to primitive varibales
  ParGridFunction *Up;
  
  Array<double> &gradUp;

  std::unordered_map<int,BoundaryCondition*> BCmap;

  //void calcMeanState();
  void computeBdrFlux(const int attr, 
                      Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);
  
  void getAttributesInPartition(Array<int> &local_attr);

public:
   BCintegrator(ParMesh *_mesh,
                ParFiniteElementSpace *_vfes,
                IntegrationRules *_intRules,
                RiemannSolver *rsolver_, 
                double &_dt,
                EquationOfState *_eqState,
                Fluxes *_fluxClass,
                ParGridFunction *_Up,
                Array<double> &_gradUp,
                const int _dim,
                const int _num_equation,
                double &_max_char_speed,
                RunConfiguration &_runFile );
   ~BCintegrator();

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
   
   void updateBCMean( ParGridFunction *Up);
};

#endif // BC_INTEGRATOR
