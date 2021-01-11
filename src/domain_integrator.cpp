
#include "domain_integrator.hpp"


// Implementation of class DomainIntegrator
DomainIntegrator::DomainIntegrator(Fluxes *_fluxClass,
                                   IntegrationRules *_intRules,
                                   int _intRuleType,
                                   const int _dim, 
                                   const int _num_equation):
  fluxClass(_fluxClass),
  dim(_num_equation),
  num_equation(_num_equation),
  intRules(_intRules),
  intRuleType(_intRuleType)
{
  
}

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Tr,
                                              DenseMatrix &elmat)
{
   // Assemble the form (vec(v), grad(w))
  
   Vector shape;
   //DenseMatrix flux(num_equation,dim);
   DenseMatrix dshapedr;
   DenseMatrix dshapedx;

   // Trial space = vector L2 space (mesh dim)
   // Test space  = scalar L2 space

   const int dof_trial = trial_fe.GetDof();
   const int dof_test = test_fe.GetDof();
   const int dim = trial_fe.GetDim();

   shape.SetSize(dof_trial);
   dshapedr.SetSize(dof_test, dim);
   dshapedx.SetSize(dof_test, dim);

   elmat.SetSize(dof_test, dof_trial * dim);
   elmat = 0.0;

   const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
   int intorder = 2 * maxorder;
   //if( intRuleType==1 ) intorder--; // when using Gauss-Lobatto
   const IntegrationRule *ir = &intRules->Get(trial_fe.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Calculate the shape functions
      trial_fe.CalcShape(ip, shape);
      shape *= ip.weight;

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      test_fe.CalcDShape(ip, dshapedr);
      Mult(dshapedr, Tr.AdjugateJacobian(), dshapedx);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dof_test; j++)
         {
            for (int k = 0; k < dof_trial; k++)
            {
               elmat(j, k + d * dof_trial) += shape(k) * dshapedx(j, d);
            }
         }
      }
   }
}
