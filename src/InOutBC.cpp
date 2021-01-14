#include "InOutBC.hpp"

InOutBC::InOutBC(Mesh* _mesh, 
                 IntegrationRules* _intRules, 
                 RiemannSolver* rsolver_, 
                 EquationOfState *_eqState,
                 const int _dim, 
                 const int _num_equation, 
                 double& _max_char_speed,
                 int _patchNumber ):
rsolver(rsolver_),
eqState(_eqState),
dim(_dim),
num_equation(_num_equation),
max_char_speed(_max_char_speed),
intRules(_intRules),
mesh(_mesh),
patchNumber(_patchNumber)
{
  meanState = new Vector(num_equation);
}

InOutBC::~InOutBC()
{
  delete meanState;
}

void InOutBC::AssembleFaceVector(const FiniteElement& el1, 
                                 const FiniteElement& el2, 
                                 FaceElementTransformations& Tr, 
                                 const Vector& elfun, 
                                 Vector& elvect)
{
  cout<<"attr: " << Tr.Attribute << "elem x: "<< el1.GetNodes().IntPoint(0).x <<endl; 
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
//    DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2,
//                           num_equation);
   DenseMatrix elfun2_mat(dof1,num_equation);
   for(int n=0; n<dof1; n++)
   {
     Vector nodeState(num_equation), bcState(num_equation);
     elfun1_mat.GetRow(n, nodeState);
     computeState(nodeState, bcState);
     elfun2_mat.SetRow(n,bcState);
   }

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
//    DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2,
//                            num_equation);

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

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Jacobian(), nor);
      const double mcs = rsolver->Eval(funval1, funval2, nor, fluxN);

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
            //elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}
