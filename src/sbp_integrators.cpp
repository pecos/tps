

#include "sbp_integrators.hpp"

SBPintegrator::SBPintegrator(EquationOfState* _eqState, 
                             Fluxes *_fluxClass,
                             IntegrationRules* _intRules, 
                             const int _dim, 
                             const int _num_equation,
                             double &_alpha
                            ):
dim(_dim),
num_equation(_num_equation),
alpha(_alpha),
eqState(_eqState),
fluxClass(_fluxClass),
intRules(_intRules)
{
}

void SBPintegrator::AssembleElementVector(const FiniteElement& el, 
                                          ElementTransformation& Tr, 
                                          const Vector& elfun, 
                                          Vector& elvect)
{
  // Compute the non-linear terms of the SBP operations.
  
   Vector shape;
   DenseMatrix dshape, dshape_xi;
   Vector funval1(num_equation);
   Vector funval2(num_equation);
  
   const int dof = el.GetDof();

   shape.SetSize(dof);
   dshape.SetSize(dof, dim);
   dshape_xi.SetSize(dof, dim);

   elvect.SetSize(dof*num_equation);
   elvect = 0.0;

   // write data in matrix form
   DenseMatrix state_mat(elfun.GetData(), dof, num_equation);
   
   // Matrix where to write output data
   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equation);
   
   // if the fluxes are expressed as f=ac, compute values of a and c at each DoF
   DenseTensor a_tensor(dof,num_equation,dim);
   DenseTensor c_tensor(dof,num_equation,dim);
   for(int node=0; node<dof; node++)
   {
      Vector iState(num_equation);
      state_mat.GetRow(node, iState);
      DenseMatrix a_vec, c_vec;
      fluxClass->ComputeSplitFlux(iState, a_vec, c_vec);
      for(int d=0; d<dim; d++)
      {
         for(int eq=0; eq<num_equation; eq++)
         {
           //cout<<a_vec(eq,d)<<" ";
           a_tensor(node,eq,d) = a_vec(eq,d);
           c_tensor(node,eq,d) = c_vec(eq,d);
         }
         //cout<<endl;
      }
   }
   
   
   // Integration order calculation from DGTraceIntegrator
   int intorder = 2*el.GetOrder();
   intorder--; // SBP can only be used with collocated Gauss-Lobatto integration rules
   
   const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
   
   Vector aidli(num_equation);
   aidli = 0.;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      const double weight = ip.weight*(1.-alpha);
      
      Tr.SetIntPoint(&ip);
      
      DenseMatrix invJ = Tr.InverseJacobian();
      double detJac = Tr.Jacobian().Det();

      // Calculate basis functions and their derivatives
      el.CalcShape(Tr.GetIntPoint(), shape);
      el.CalcDShape(Tr.GetIntPoint(), dshape_xi);
      
      // compute a(xk), da/dx(xk), c(xk)
      DenseMatrix a(num_equation,dim), c(num_equation,dim);
      a  = 0.;
      //ap = 0.;
      c  = 0.;
      for(int eq=0; eq<num_equation; eq++)
      {
        for(int i=0; i<dof; i++) 
        {
          for(int d=0; d<dim; d++)
          {
            a(eq,d)  += a_tensor(i,eq,d)*shape(i);
            //ap(eq,d) += a_tensor(i,eq,d)*dshape_xi(i,d);
            c(eq,d)  += c_tensor(i,eq,d)*shape(i);
          }
        }
      }
      
      // Write output matrix
      for(int node=0; node<dof; node++)
      {
        for(int eq=0; eq<num_equation; eq++)
        {
          //double apxc = 0.;
          double aclp = 0.;
          for(int di=0; di<dim; di++) for(int dj=0; dj<dim; dj++)
          {
            //apxc+= ap(eq,di)*invJ(di,dj)*c(eq,dj);
            aclp += a(eq,dj)*c(eq,dj)*dshape_xi(node,di)*invJ(di,dj);
          } 
          elvect_mat(node,eq) += aclp*detJac*weight;
        }
      }
  }
}
