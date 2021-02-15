

#include "rhs_operator.hpp"

// Implementation of class RHSoperator
RHSoperator::RHSoperator( const int _dim,
                          const int &_num_equations,
                          const Equations &_eqSystem,
                          double &_max_char_speed,
                          IntegrationRules *_intRules,
                          int _intRuleType,
                          Fluxes *_fluxClass,
                          EquationOfState *_eqState,
                          ParFiniteElementSpace *_vfes,
                          ParNonlinearForm *_A, 
                          MixedBilinearForm *_Aflux,
                          ParMesh *_mesh,
                          ParGridFunction *_Up,
                          ParGridFunction *_gradUp,
                          ParFiniteElementSpace *_gradUpfes,
                          ParNonlinearForm *_gradUp_A,
                          BCintegrator *_bcIntegrator,
                          bool &_isSBP,
                          double &_alpha
                        ):
TimeDependentOperator(_A->Height()),
dim(_dim ),
eqSystem(_eqSystem),
max_char_speed(_max_char_speed),
num_equation(_num_equations),
intRules(_intRules),
intRuleType(_intRuleType),
fluxClass(_fluxClass),
eqState(_eqState),
vfes(_vfes),
A(_A),
Aflux(_Aflux),
mesh(_mesh),
isSBP(_isSBP),
alpha(_alpha),
Up(_Up),
gradUp(_gradUp),
gradUpfes(_gradUpfes),
gradUp_A(_gradUp_A),
bcIntegrator(_bcIntegrator)
{
  state = new Vector(num_equation);
  
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
   
   
#ifdef DEBUG 
{
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
#endif
  
}

RHSoperator::~RHSoperator()
{
  delete state;
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
  calcGradientsPrimitives();
  
  // update boundary conditions
  if(bcIntegrator!=NULL) bcIntegrator->updateBCMean( Up );
  
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
  
   DenseMatrix f(num_equation, dim);
   
   double *dataGradUp = gradUp->GetData();
   
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation; k++) (*state)(k) = x(i, k);
      DenseMatrix gradUpi(num_equation,dim);
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) gradUpi(eq,d) = 
                               dataGradUp[i+eq*dof+d*num_equation*dof];
      }
      
      fluxClass->ComputeTotalFlux(*state,gradUpi,f);
     
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
   
   double partition_max_char = max_char_speed;
   MPI_Allreduce(&partition_max_char, &max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
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
    if(dim==3) dataUp[i+3*vfes->GetNDofs()] = iState[3]/iState[0];
    dataUp[i+(num_equation-1)*vfes->GetNDofs()] = p;
  }
}

void RHSoperator::calcGradientsPrimitives() const
{
  const int totalDofs = vfes->GetNDofs();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();
  
  // Vars for face contributions
  Vector faceContrib(dim*num_equation*totalDofs);
  Vector xUp(dim*num_equation*totalDofs);
  xUp = 0.;
  faceContrib = 0.;
  
  // compute volume integral and fill out above vectors
  for(int el=0;el<vfes->GetNE();el++)
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
      for(int eq=0;eq<num_equation;eq++)
      {
        elUp(d,eq) = dataUp[index +eq*totalDofs];
        xUp[index+eq*totalDofs] = elUp(d,eq);
      } 
    }
    
    DenseMatrix elGradUp(eldDof,num_equation*dim);
    elGradUp = 0.;
    
    // element volume integral
    int intorder = 2*elem->GetOrder();
    if(intRuleType==1 && elem->GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);

    for(int i=0;i<ir->GetNPoints();i++)
    {
      IntegrationPoint ip = ir->IntPoint(i);
      Tr->SetIntPoint( &ip );
      
      // Calculate the shape functions
      Vector shape(eldDof);
      DenseMatrix dshape(eldDof,dim);
      elem->CalcShape(ip, shape);
      elem->CalcPhysDShape(*Tr,dshape);
      
      // calc Up at int. point
      Vector iUp(num_equation);
      DenseMatrix iGradUp(num_equation,dim);
      iGradUp = 0.;
      for(int eq=0;eq<num_equation;eq++)
      {
        double sum = 0.;
        for(int k=0;k<eldDof;k++)
        {
          for(int d=0;d<dim;d++) iGradUp(eq,d) += elUp(k,eq)*dshape(k,d);
          sum += elUp(k,eq)*shape(k);
        }
        iUp[eq] = sum;
      }
      
      double detJac = Tr->Jacobian().Det()*ip.weight;
      
      // Add volume contrubutions to gradient
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          for(int j=0;j<eldDof;j++)
          {
            //elGradUp(j,eq+d*num_equation) += iUp[eq]*dshape(j,d)*detJac;
            elGradUp(j,eq+d*num_equation) += shape(j)*iGradUp(eq,d)*detJac;
          }
        }
      }
    }
    
    // transfer result into gradUp
    for(int k=0; k<eldDof; k++)
    {
      int index = vdofs[k];
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          dataGradUp[index+eq*totalDofs+d*num_equation*totalDofs] = 
                                        -elGradUp(k,eq+d*num_equation);
        }
      } 
    }
  }
  
  // Calc boundary contribution
  gradUp_A->Mult(xUp,faceContrib);
  
  // Add contributions and multiply by invers mass matrix
  for(int el=0;el<vfes->GetNE();el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    const int eldDof = elem->GetDof();
    
    Array<int> vdofs;
    vfes->GetElementVDofs(el, vdofs);
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        Vector rhs(eldDof);
        for(int k=0;k<eldDof;k++)
        {
          int index = vdofs[k];
//           rhs[k] = gradUp[index+eq*totalDofs+d*num_equation*totalDofs]+
//                   faceContrib[index+eq*totalDofs+d*num_equation*totalDofs];
          rhs[k] = -dataGradUp[index+eq*totalDofs+d*num_equation*totalDofs]+
                   faceContrib[index+eq*totalDofs+d*num_equation*totalDofs];
        }
        
        // mult by inv mass matrix
        Vector aux(eldDof);
        aux = 0.;
        for(int i=0;i<eldDof;i++)
        {
          for(int j=0;j<eldDof;j++) aux[i] += Me_inv[el](i,j)*rhs[j];
        }   
        
        // save this in gradUp
        for(int k=0;k<eldDof;k++)
        {
          int index = vdofs[k];
          dataGradUp[index+eq*totalDofs+d*num_equation*totalDofs] = aux[k];
        }
      }
    }
  }
  
  // NOTE: not sure I need to this here. CHECK!!
  gradUp->ExchangeFaceNbrData();
}

