#ifndef FACE_INTEGRATOR
#define FACE_INTEGRATOR

#include "mfem.hpp"
#include "riemann_solver.hpp"

using namespace mfem;

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver *rsolver;
//    Vector shape1;
//    Vector shape2;
//    Vector funval1;
//    Vector funval2;
//    Vector nor;
//    Vector fluxN;
   
   const int dim;
   const int num_equation;
   
   double &max_char_speed;
   IntegrationRules *intRules;

public:
   FaceIntegrator(IntegrationRules *_intRules,
                  RiemannSolver *rsolver_, 
                  const int _dim,
                  const int _num_equation,
                  double &_max_char_speed );

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

#endif // FACE_INTEGRATOR
