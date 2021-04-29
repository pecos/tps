#ifndef FACE_INTEGRATOR
#define FACE_INTEGRATOR

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>
#include "riemann_solver.hpp"
#include "fluxes.hpp"

using namespace mfem;

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver *rsolver;
   Fluxes *fluxClass;
   ParFiniteElementSpace *vfes;
   
   const int dim;
   const int num_equation;
   
   double &max_char_speed;
   
   const ParGridFunction *gradUp;
   const ParFiniteElementSpace *gradUpfes;
   
   IntegrationRules *intRules;
   
   DenseMatrix *faceMassMatrix1, *faceMassMatrix2;
   int faceNum;
   bool faceMassMatrixComputed;
   bool useLinear;

  int totDofs;
  Array<int> vdofs1;
  Array<int> vdofs2;
  Vector shape1;
  Vector shape2;
  Vector funval1;
  Vector funval2;
  Vector nor;
  Vector fluxN;
  //DenseMatrix gradUp1;
  //DenseMatrix gradUp2;
  DenseTensor gradUp1;
  DenseTensor gradUp2;  
  DenseMatrix gradUp1i;
  DenseMatrix gradUp2i;
  DenseMatrix viscF1;
  DenseMatrix viscF2;
  DenseMatrix elfun1_mat;
  DenseMatrix elfun2_mat;
  DenseMatrix elvect1_mat;
  DenseMatrix elvect2_mat;
   
   void getElementsGrads_cpu( FaceElementTransformations &Tr,
                              const FiniteElement &el1, 
                              const FiniteElement &el2, 
                              DenseTensor &gradUp1, 
                              DenseTensor &gradUp2);
   
   void NonLinearFaceIntegration( const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
   void MassMatrixFaceIntegral( const FiniteElement &el1,
                                const FiniteElement &el2,
                                FaceElementTransformations &Tr,
                                const Vector &elfun, Vector &elvect);

public:
   FaceIntegrator(IntegrationRules *_intRules,
                  RiemannSolver *rsolver_, 
                  Fluxes *_fluxClass,
                  ParFiniteElementSpace *_vfes,
                  bool _useLinear,
                  const int _dim,
                  const int _num_equation,
                  ParGridFunction *_gradUp,
                  ParFiniteElementSpace *_gradUpfes,
                  double &_max_char_speed );
   ~FaceIntegrator();

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
   
   static void getElementsGrads_gpu(const ParGridFunction *gradUp,
                                    ParFiniteElementSpace *vfes,
                                    const ParFiniteElementSpace *gradUpfes,
                                    FaceElementTransformations &Tr,
                                    const FiniteElement &el1, 
                                    const FiniteElement &el2, 
                                    DenseTensor &gradUp1, 
                                    DenseTensor &gradUp2,
                                    const int &num_equation,
                                    const int &totalDofs,
                                    const int &dim );
};

#endif // FACE_INTEGRATOR
