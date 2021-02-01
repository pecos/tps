
#include "face_integrator.hpp"

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(IntegrationRules *_intRules,
                               RiemannSolver *rsolver_, 
                               Fluxes *_fluxClass,
                               FiniteElementSpace *_vfes,
                               const int _dim,
                               const int _num_equation,
                               Array<double> &_gradUp,
                               double &_max_char_speed ):
rsolver(rsolver_),
fluxClass(_fluxClass),
vfes(_vfes),
dim(_dim),
num_equation(_num_equation),
max_char_speed(_max_char_speed),
gradUp(_gradUp),
intRules(_intRules)
{
  
}

void FaceIntegrator::getElementsGrads(FaceElementTransformations &Tr, 
                                      const FiniteElement &el1, 
                                      const FiniteElement &el2, 
                                      DenseMatrix &gradUp1, 
                                      DenseMatrix &gradUp2)
{
  const int totDofs = vfes->GetNDofs();
  Array<int> vdofs1;
  vfes->GetElementVDofs(Tr.Elem1->ElementNo, vdofs1);
  int eldDof = el1.GetDof();
  for(int n=0; n<eldDof; n++)
  {
    int index = vdofs1[n];
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++) gradUp1(n,eq +d*num_equation) = 
        gradUp[index + eq*totDofs + d*num_equation*totDofs];
    } 
  }
  
  Array<int> vdofs2;
  vfes->GetElementVDofs(Tr.Elem2->ElementNo, vdofs2);
  eldDof = el2.GetDof();
  for(int n=0; n<eldDof; n++)
  {
    int index = vdofs2[n];
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++) gradUp2(n,eq +d*num_equation) = 
        gradUp[index + eq*totDofs + d*num_equation*totDofs];
    } 
  }
}


void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
  
   Vector shape1;
   Vector shape2;
   Vector funval1(num_equation);
   Vector funval2(num_equation);
   Vector nor(dim);
   Vector fluxN(num_equation);
  
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2,
                          num_equation);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2,
                           num_equation);
   
   // Get gradients of elements
   DenseMatrix gradUp1(dof1,num_equation*dim);
   DenseMatrix gradUp2(dof2,num_equation*dim);
   getElementsGrads(Tr, el1, el2, gradUp1, gradUp2);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
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

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, funval1);
      elfun2_mat.MultTranspose(shape2, funval2);
      
      // Interpolate gradients at int. point
      DenseMatrix gradUp1i(num_equation,dim);
      DenseMatrix gradUp2i(num_equation,dim);
      gradUp1i = 0.; gradUp2i = 0.;
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          for(int k=0;k<dof1;k++) gradUp1i(eq,d) += 
                                  gradUp1(k,eq+d*num_equation)*shape1(k);
          for(int k=0;k<dof2;k++) gradUp2i(eq,d) += 
                                  gradUp2(k,eq+d*num_equation)*shape2(k);
        }
      }

      // Get the normal vector and the convective flux on the face
      CalcOrtho(Tr.Jacobian(), nor);
      const double mcs = rsolver->Eval(funval1, funval2, nor, fluxN);
      
      // compute viscous fluxes
      DenseMatrix viscF1(num_equation,dim);
      DenseMatrix viscF2(num_equation,dim);
      fluxClass->ComputeViscousFluxes(funval1,gradUp1i,viscF1);
      fluxClass->ComputeViscousFluxes(funval2,gradUp2i,viscF2);
      // compute mean flux
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) viscF1(eq,d) += viscF2(eq,d);
      }
      viscF1 *= -0.5;
      
      // add to convective fluxes
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) fluxN[eq] += viscF1(eq,d)*nor[d];
      }

      // Update max char speed
      if (mcs > max_char_speed) { max_char_speed = mcs; }

      fluxN *= ip.weight;
      for (int k = 0; k < num_equation; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}
