#ifndef IN_OUT_BC
#define IN_OUT_BC

#include "mfem.hpp"
#include "riemann_solver.hpp"
#include "equation_of_state.hpp"

using namespace mfem;
using namespace std;

// Boundary face term: <F.n(u),[w]>
class InOutBC : public NonlinearFormIntegrator
{
protected:
  RiemannSolver *rsolver;
  EquationOfState *eqState;
   
   const int dim;
   const int num_equation;
   
   double &max_char_speed;
   IntegrationRules *intRules;
   
   Mesh *mesh;
   
   // Mean boundary state
   Vector *meanState;
   
   // In/out conditions specified in the configuration file
   Array<double> inputState;
   
   // Number of the BC as specified in the mesh file
   const int patchNumber;
   
   //void calcMeanState();
   virtual void computeState(Vector &stateIn, Vector &stateOut) = 0;

public:
   InOutBC( Mesh *_mesh,
            IntegrationRules *_intRules,
            RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            const int _dim,
            const int _num_equation,
            double &_max_char_speed,
            int _patchNumber );
   ~InOutBC();

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

#endif // IN_OUT_BC
