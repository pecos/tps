#ifndef GRAD_FACE_INTEGRATOR
#define GRAD_FACE_INTEGRATOR

#include "mfem.hpp"

using namespace mfem;

// Interior face term: <F.n(u),[w]>
class GradFaceIntegrator : public NonlinearFormIntegrator
{
private:
   
   const int dim;
   const int num_equation;
   IntegrationRules *intRules;

public:
   GradFaceIntegrator(IntegrationRules *_intRules,
                      const int _dim,
                      const int _num_equation);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

#endif // GRAD_FACE_INTEGRATOR
