

#include "rhs_operator.hpp"

// Implementation of class RHSoperator
RHSoperator::RHSoperator( const int _dim,
                          const Equations &_eqSystem,
                          double &_max_char_speed,
                          IntegrationRules *_intRules,
                          Fluxes *_fluxClass,
                          EquationOfState *_eqState,
                          FiniteElementSpace *_vfes,
                          NonlinearForm *_A, 
                          MixedBilinearForm *_Aflux )
   : TimeDependentOperator(_A->Height()),
     dim(_dim ),
     eqSystem(_eqSystem),
     max_char_speed(_max_char_speed),
     intRules(_intRules),
     fluxClass(_fluxClass),
     eqState(_eqState),
     vfes(_vfes),
     A(_A),
     Aflux(_Aflux)
//      state(num_equation)
//      f(num_equation, dim),
//      flux(vfes->GetNDofs(), dim, num_equation),
//      z(A->Height())
{
  switch(eqSystem)
  {
    case EULER:
      num_equation = dim + 2;
      break;
    case NS:
      num_equation = dim + 2;
      break;
    default:
      std::cout << "System of equations not specified" << std::endl;
      break;
  }
  
  state = new Vector(num_equation);
  
  Me_inv = new DenseTensor(vfes->GetFE(0)->GetDof(), vfes->GetFE(0)->GetDof(), vfes->GetNE());
  
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes->GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   
   for (int i = 0; i < vfes->GetNE(); i++)
   {
     int integrationOrder = 2*vfes->GetFE(i)->GetOrder() ;
     const IntegrationRule intRule = intRules->Get(vfes->GetFE(i)->GetGeomType(), integrationOrder);
     mi.SetIntRule(&intRule);
      mi.AssembleElementMatrix(*(vfes->GetFE(i) ), *(vfes->GetElementTransformation(i) ), Me);
      inv.Factor();
      inv.GetInverseMatrix( (*Me_inv)(i));
   }
   
   
   // This is for DEBUG ONLY!!!
   // Inverse of transformation jacobian
   DenseMatrix kk = vfes->GetElementTransformation(0)->Jacobian();
   
   // Derivation matrix
   DenseMatrix Dx;
   Dx.SetSize( Me.Size() );
   
   for(int node=0; node<Me.Size(); node++)
   {
     DenseMatrix column(Me.Size(),2);
     
     ElementTransformation *eltrans = vfes->GetElementTransformation(0);
     IntegrationPoint intP = eltrans->GetIntPoint();
     IntegrationRule intRule = vfes->GetFE(0)->GetNodes();
     intP.Set(intRule.IntPoint(node).x, 
              intRule.IntPoint(node).y, 0., 
              intRule.IntPoint(node).weight);
     eltrans->SetIntPoint(&intP);
     vfes->GetFE(0)->CalcPhysDShape( *eltrans, column);
     
     cout << "" <<endl;
     for(int i=0; i<Me.Size(); i++) cout<<column(i,0)<<" ";
     cout<<endl;
     for(int i=0; i<Me.Size(); i++) cout<<column(i,1)<<" ";
     cout<<endl;
     
      for(int i=0; i<Me.Size(); i++)
      {
        Dx(node, i) = column(i,0);
        //cout<< Dx(i, node) << " ";
      }
      //cout<<endl;
   }
   
   // Mass matrix
   int integrationOrder = 2*vfes->GetFE(0)->GetOrder() ;
   const IntegrationRule intRule = intRules->Get(vfes->GetFE(0)->GetGeomType(), integrationOrder);
   mi.SetIntRule(&intRule);
   mi.AssembleElementMatrix(*(vfes->GetFE(0) ), *(vfes->GetElementTransformation(0) ), Me);
   //Me *= 1./detJac;
   cout<<"Mass matrix"<<endl;
   for(int i=0; i<Me.Size(); i++)
   {
      for(int j=0; j<Me.Size(); j++)
      {
        if( fabs(Me(i,j))<1e-15 )
        {
          cout<<"* ";
        }else
        {
          cout<<Me(i,j)<<" ";
        }
      }
      cout<<endl;
   }
   
   // Matrix Q=MD
   DenseMatrix Q( Me.Size() );
   mfem::Mult(Me,Dx,Q);
   
   cout.precision(3);
   for(int i=0; i<Me.Size();i++)
   {
     for(int j=0; j<Me.Size();j++)
     {
       if( fabs(Q(i,j) + Q(j,i))<1e-5 )
       {
         cout<<"* ";
      }else
      {
        cout << Q(i,j) + Q(j,i) << " ";
      }
     }
    cout << endl;
  }
  
}

RHSoperator::~RHSoperator()
{
  delete state;
  delete Me_inv;
}


void RHSoperator::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   DenseTensor flux(vfes->GetNDofs(), dim, num_equation);
   Vector z(A->Height());

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A->Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes->GetNDofs(), num_equation);
   GetFlux(xmat, flux);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux(k).GetData(), dim * vfes->GetNDofs());
      Vector zk(z.GetData() + k * vfes->GetNDofs(), vfes->GetNDofs());
      Aflux->AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes->GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes->GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes->GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult((*Me_inv)(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Compute the flux at solution nodes.
void RHSoperator::GetFlux(const DenseMatrix &x, DenseTensor &flux) const
{
  
   //DenseTensor flux(vfes->GetNDofs(), dim, num_equation);
   DenseMatrix f(num_equation, dim);
   
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation; k++) { (*state)(k) = x(i, k); }
      fluxClass->ComputeFlux(*state, dim, f);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux(i, d, k) = f(k, d);
         }
      }

      // Update max char speed
      const double mcs = eqState->ComputeMaxCharSpeed(*state, dim);
      if (mcs > max_char_speed) { max_char_speed = mcs; }
   }
}
