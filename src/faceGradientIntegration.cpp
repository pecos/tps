
#include "faceGradientIntegration.hpp"

// Implementation of class FaceIntegrator
GradFaceIntegrator::GradFaceIntegrator(IntegrationRules *_intRules,
                               const int _dim,
                               const int _num_equation):
dim(_dim),
num_equation(_num_equation),
intRules(_intRules)
{
  
}

void GradFaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
  // Compute the term <nU,[w]> on the interior faces.
  Vector shape1; shape1.UseDevice(false);
  Vector shape2; shape2.UseDevice(false);
  Vector nor(dim); nor.UseDevice(false);
  Vector mean; mean.UseDevice(false);
  mean.SetSize(num_equation);

  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elfun.Read();
  elvect.UseDevice(true);
  elvect.SetSize((dof1+dof2)*num_equation*dim);
  elvect = 0.0;

// #ifdef _GPU_
//   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
//   DenseMatrix elfun2_mat(elfun.GetData()+dof1*num_equation, 
//                          dof2, num_equation);
#ifndef _GPU_
  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation*dim);
  DenseMatrix elfun2_mat(elfun.GetData()+dof1*num_equation*dim, 
                         dof2, num_equation*dim);
  DenseMatrix elvect1_mat(elvect.GetData(),dof1, num_equation*dim);
  DenseMatrix elvect2_mat(elvect.GetData()+dof1*num_equation*dim, 
                          dof2, num_equation*dim);
  elvect1_mat = 0.;
  elvect2_mat = 0.;
#endif

  // Integration order calculation from DGTraceIntegrator
  int intorder;
  if (Tr.Elem2No >= 0)
    intorder = (std::min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                2*std::max(el1.GetOrder(), el2.GetOrder()));
  else
  {
    intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
  }
  if (el1.Space() == FunctionSpace::Pk)
  {
    intorder++;
  }
  //IntegrationRules IntRules2(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip); // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate U values at the point
    Vector iUp1,iUp2; iUp1.UseDevice(false); iUp2.UseDevice(false);
    iUp1.SetSize(num_equation); iUp2.SetSize(num_equation);
    for(int eq=0;eq<num_equation;eq++)
    {
      double sum = 0.;
      for(int k=0;k<dof1;k++)
      {
#ifdef _GPU_
        sum += shape1(k)*elfun(k+eq*dof1);
#else
        sum += shape1[k]*elfun1_mat(k,eq);
#endif
      }
      iUp1(eq) = sum;
      sum = 0.;
      for(int k=0;k<dof2;k++)
      {
#ifdef _GPU_
        sum += shape2(k)*elfun(k+eq*dof2 + dof1*num_equation);
#else
        sum += shape2[k]*elfun2_mat(k,eq);
#endif
      }
      iUp2(eq) = sum;
      mean(eq) = (iUp1(eq)+iUp2(eq))/2.;
    }

    // Get the normal vector and the flux on the face
    CalcOrtho(Tr.Jacobian(), nor);
    
    nor *= ip.weight;
    
    for(int d=0;d<dim;d++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int k=0;k<dof1;k++)
        {
#ifdef _GPU_
          elvect(k+eq+d*num_equation) += (mean(eq)-iUp1(eq))*nor(d)*shape1(k);
#else
          elvect1_mat(k,eq+d*num_equation) += (mean(eq)-iUp1(eq))*nor(d)*shape1(k);
#endif
        }
        for(int k=0;k<dof2;k++)
        {
#ifdef _GPU_
          elvect(k+eq+d*num_equation + dof1*num_equation*dim) -= (mean(eq)-iUp2(eq))*nor(d)*shape2(k);
#else
          elvect2_mat(k,eq+d*num_equation) -= (mean(eq)-iUp2(eq))*nor(d)*shape2(k);
#endif
        }
      }
    }
      
  }
}
