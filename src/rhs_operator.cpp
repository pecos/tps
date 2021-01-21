

#include "rhs_operator.hpp"

// Implementation of class RHSoperator
RHSoperator::RHSoperator( const int _dim,
                          const Equations &_eqSystem,
                          double &_max_char_speed,
                          IntegrationRules *_intRules,
                          int _intRuleType,
                          Fluxes *_fluxClass,
                          EquationOfState *_eqState,
                          FiniteElementSpace *_vfes,
                          NonlinearForm *_A, 
                          MixedBilinearForm *_Aflux,
                          GridFunction *_Up,
                          Array<double> &_gradUp,
                          BCintegrator *_bcIntegrator,
                          bool &_isSBP,
                          double &_alpha
                        ):
TimeDependentOperator(_A->Height()),
dim(_dim ),
eqSystem(_eqSystem),
max_char_speed(_max_char_speed),
intRules(_intRules),
intRuleType(_intRuleType),
fluxClass(_fluxClass),
eqState(_eqState),
vfes(_vfes),
A(_A),
Aflux(_Aflux),
isSBP(_isSBP),
alpha(_alpha),
Up(_Up),
gradUp(_gradUp),
bcIntegrator(_bcIntegrator)
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
  
  // Finite element space associated with the gradients of primitives 
  qfes = new FiniteElementSpace(vfes->GetMesh(),vfes->FEColl(),
                                num_equation*dim, Ordering::byNODES);
  Aq = new NonlinearForm(qfes);
  
  //Me_inv = new DenseTensor(vfes->GetFE(0)->GetDof(), vfes->GetFE(0)->GetDof(), vfes->GetNE());
  Me_inv = new DenseMatrix[vfes->GetNE()];
   
   for (int i = 0; i < vfes->GetNE(); i++)
   {
      // Standard local assembly and inversion for energy mass matrices.
      const int dof = vfes->GetFE(i)->GetDof();
      DenseMatrix Me(dof);
      DenseMatrixInverse inv(&Me);
      MassIntegrator mi;
    
      int integrationOrder = 2*vfes->GetFE(i)->GetOrder();
      if(intRuleType==1 && vfes->GetFE(i)->GetGeomType()==Geometry::SQUARE) integrationOrder--; // when Gauss-Lobatto
      const IntegrationRule intRule = intRules->Get(vfes->GetFE(i)->GetGeomType(), integrationOrder);
      mi.SetIntRule(&intRule);
      mi.AssembleElementMatrix(*(vfes->GetFE(i) ), *(vfes->GetElementTransformation(i) ), Me);
      inv.Factor();
      //inv.GetInverseMatrix( (*Me_inv)(i));
      inv.GetInverseMatrix( Me_inv[i]);
   }
   
   
   // This is for DEBUG ONLY!!!
   int elem = 10;
    const int dof = vfes->GetFE(elem)->GetDof();
    DenseMatrix Me(dof);
   
   // Derivation matrix
   DenseMatrix Dx(dof), Kx(dof);
   Dx = 0.;
   Kx = 0.;
   ElementTransformation *Tr = vfes->GetElementTransformation(elem) ;
   int integrationOrder = 2*vfes->GetFE(elem)->GetOrder();
   if(intRuleType==1 && vfes->GetFE(elem)->GetGeomType()==Geometry::SQUARE) integrationOrder--; // when Gauss-Lobatto
   const IntegrationRule intRule = intRules->Get(vfes->GetFE(elem)->GetGeomType(), integrationOrder);
   for(int k=0; k<intRule.GetNPoints(); k++)
   {
     const IntegrationPoint &ip = intRule.IntPoint(k);
      double wk = ip.weight;
      Tr->SetIntPoint(&ip);
      
     DenseMatrix invJ = Tr->InverseJacobian();
     double detJac = Tr->Jacobian().Det();
     DenseMatrix dshape(dof,2);
     Vector shape( dof );
     vfes->GetFE(elem)->CalcShape( Tr->GetIntPoint(), shape);
     vfes->GetFE(elem)->CalcDShape(Tr->GetIntPoint(), dshape);
     
      for(int i=0; i<dof; i++)
      {
        for(int j=0; j<dof; j++)
        {
          Kx(j, i) += shape(i)*detJac*(dshape(j,0)*invJ(0,0) +
                                       dshape(j,1)*invJ(1,0) )*wk;
          Kx(j, i) += shape(i)*detJac*(dshape(j,0)*invJ(0,1) +
                                       dshape(j,1)*invJ(1,1) )*wk;
        }
      }
   }
   mfem::Mult(Kx,Me_inv[elem], Dx);
   Dx.Transpose();
   
   // Mass matrix
   MassIntegrator mi;
   mi.SetIntRule(&intRule);
   mi.AssembleElementMatrix(*(vfes->GetFE(elem) ), *(vfes->GetElementTransformation(elem) ), Me);
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
  delete qfes;
  delete[] Me_inv;
}


void RHSoperator::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
  max_char_speed = 0.;

  DenseTensor flux(vfes->GetNDofs(), dim, num_equation);
  Vector z(A->Height());

  // Update primite varibales
  updatePrimitives(x);
  
  // update boundary conditions
  bcIntegrator->updateBCMean( Up );
  
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
  for (int i = 0; i < vfes->GetNE(); i++)
  {
    Vector zval;
    Array<int> vdofs;
    const int dof = vfes->GetFE(i)->GetDof();
    DenseMatrix zmat, ymat(dof, num_equation);
    
    // Return the vdofs ordered byNODES
    vfes->GetElementVDofs(i, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), dof, num_equation);
    
    mfem::Mult(Me_inv[i], zmat, ymat);
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
     
      if( isSBP )
      {
        f *= alpha; // *= alpha
        double p = eqState->ComputePressure(*state, dim);
        p *= 1. - alpha;
        for(int d=0; d<dim; d++)
        {
          f(d+1,d) += p;
          f(num_equation-1,d) += (1.-0.)*p*(*state)(1+d)/(*state)(0);
        }
      }
        

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

void RHSoperator::updatePrimitives(const Vector &x) const
{
  double *dataUp = Up->GetData();
  for(int i=0;i<vfes->GetNDofs();i++)
  {
    Vector iState(num_equation);
    for(int eq=0;eq<num_equation;eq++) iState[eq] = x[i+eq*vfes->GetNDofs()];
    double p = eqState->ComputePressure(iState,dim);
    dataUp[i                   ] = iState[0];
    dataUp[i+  vfes->GetNDofs()] = iState[1]/iState[0];
    dataUp[i+2*vfes->GetNDofs()] = iState[2]/iState[0];
    dataUp[i+3*vfes->GetNDofs()] = p;
  }
  
  //compute gradients
  for(int el=0; el<vfes->GetNE(); el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    
    // get local primitive variables
    Array<int> vdofs;
    vfes->GetElementVDofs(el, vdofs);
    const int eldDof = elem->GetDof();
    DenseMatrix elUp(eldDof,num_equation);
    for(int d=0; d<eldDof; d++)
    {
      int index = vdofs[d];
      int nDofs = vfes->GetNDofs();
      for(int eq=0;eq<num_equation;eq++) elUp(d,eq) = dataUp[index +eq*nDofs];
    }

    
    const IntegrationRule ir = elem->GetNodes();
    for(int i=0; i<ir.GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr->SetIntPoint(&ip);
      
      DenseMatrix invJ = Tr->InverseJacobian();

      // Calculate basis functions and their derivatives
      Vector shape( eldDof );
      DenseMatrix dshape(eldDof,dim);
      
      elem->CalcShape(Tr->GetIntPoint(), shape);
      elem->CalcPhysDShape(*Tr, dshape);
      
      for(int eq=0; eq<num_equation; eq++)
      {
        for(int d=0; d<dim; d++)
        {
          double sum =0.;
          for(int k=0; k<eldDof; k++)
          {
            sum += elUp(k,eq)*dshape(k,d);
          }
          int nDofs = vfes->GetNDofs();
          gradUp[vdofs[i] +eq*nDofs + d*num_equation*nDofs] = sum;
        }
      }
    }
  }
}

