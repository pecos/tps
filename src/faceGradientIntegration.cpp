
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
  Vector shape1;
  Vector shape2;
  Vector nor(dim);
  Vector mean(num_equation);

  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elvect.SetSize((dof1+dof2)*num_equation*dim);
  elvect = 0.0;

  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation*dim);
  DenseMatrix elfun2_mat(elfun.GetData()+dof1*num_equation*dim, 
                         dof2, num_equation*dim);
// std::cout<<"up in"<<std::endl;
// for(int i=0;i<dof1;i++)
// {
//   for(int j=0;j<num_equation*dim;j++)
//   {
//     std::cout<<elfun1_mat(i,j)<<" ";
//   }
//   std::cout<<std::endl;
// }

  DenseMatrix elvect1_mat(elvect.GetData(),dof1, num_equation);
  DenseMatrix elvect2_mat(elvect.GetData()+dof1*num_equation, 
                          dof2, num_equation*dim);

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
    for(int eq=0;eq<num_equation;eq++)
    {
      double sum = 0.;
      for(int k=0;k<dof1;k++)
      {
        sum += shape1[k]*elfun1_mat(k,eq);
      }
      for(int k=0;k<dof2;k++)
      {
        sum += shape2[k]*elfun2_mat(k,eq);
      }
      mean[eq] = sum/2.;
    }

    // Get the normal vector and the flux on the face
    CalcOrtho(Tr.Jacobian(), nor);
    
    mean *= ip.weight;
    
    for(int d=0;d<dim;d++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int k=0;k<dof1;k++)
        {
          elvect1_mat(k,eq+d*num_equation) += mean(eq)*nor[d]*shape1(k);
        }
        for(int k=0;k<dof2;k++)
        {
          elvect2_mat(k,eq+d*num_equation) -= mean(eq)*nor[d]*shape2(k);
        }
      }
    }
      
  }
// std::cout<<"func out"<<std::endl;
// for(int i=0;i<dof1;i++)
// {
//   for(int j=0;j<num_equation*dim;j++)
//   {
//     std::cout<<elvect1_mat(i,j)<<" ";
//   }
//   std::cout<<std::endl;
// }
}
