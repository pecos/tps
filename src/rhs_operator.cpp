

#include "rhs_operator.hpp"

#ifdef _GPU_
#include "general/forall.hpp"
#endif // _GPU_

// Implementation of class RHSoperator
RHSoperator::RHSoperator( double &_time,
                          const int _dim,
                          const int &_num_equations,
                          const int &_order,
                          const Equations &_eqSystem,
                          double &_max_char_speed,
                          IntegrationRules *_intRules,
                          int _intRuleType,
                          Fluxes *_fluxClass,
                          EquationOfState *_eqState,
                          ParFiniteElementSpace *_vfes,
                          Array<int> &_nodesIDs,
                          Array<int> &_posDofIds,
                          Array<int> &_numElems,
                          Vector &_shapeWnor1,
                          Vector &_shape2,
                          Array<int> &_elemFaces,
                          Array<int> &_elems12Q,
                          const int &_maxIntPoints,
                          const int &_maxDofs,
                          //ParNonlinearForm *_A, 
                          DGNonLinearForm *_A,
                          MixedBilinearForm *_Aflux,
                          ParMesh *_mesh,
                          ParGridFunction *_Up,
                          ParGridFunction *_gradUp,
                          ParFiniteElementSpace *_gradUpfes,
                          //ParNonlinearForm *_gradUp_A,
                          GradNonLinearForm *_gradUp_A,
                          BCintegrator *_bcIntegrator,
                          bool &_isSBP,
                          double &_alpha,
                          RunConfiguration &_config
                        ):
TimeDependentOperator(_A->Height()),
time(_time),
dim(_dim ),
eqSystem(_eqSystem),
max_char_speed(_max_char_speed),
num_equation(_num_equations),
intRules(_intRules),
intRuleType(_intRuleType),
fluxClass(_fluxClass),
eqState(_eqState),
vfes(_vfes),
nodesIDs(_nodesIDs),
posDofIds(_posDofIds),
numElems(_numElems),
shapeWnor1(_shapeWnor1),
shape2(_shape2),
elemFaces(_elemFaces),
elems12Q(_elems12Q),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs),
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
  flux.SetSize(vfes->GetNDofs(), dim, num_equation);
  z.UseDevice(true);
  z.SetSize(A->Height());
  z = 0.;
  
  h_numElems = numElems.HostReadWrite();
  h_posDofIds = posDofIds.HostReadWrite();
  
  //Me_inv = new DenseMatrix[vfes->GetNE()];
  Me_inv.SetSize( vfes->GetNE() );
  posDofInvM.SetSize(2*vfes->GetNE());
  auto hposDofInvM = posDofInvM.HostWrite();
  
  if( _config.thereIsForcing() )
  {
    forcing.Append( new ConstantPressureGradient( dim,
                                            num_equation,
                                            _order,
                                            intRuleType,
                                            intRules,
                                            vfes,
                                            Up,
                                            gradUp,
                                            _config) );
  }
#ifdef _MASA_
  forcing.Append(new MASA_forcings( dim,
                                    num_equation,
                                    _order,
                                    intRuleType,
                                    intRules,
                                    vfes,
                                    Up,
                                    gradUp,
                                    _config ) );
#endif
  std::vector<double> temp;
  temp.clear();
  
  for(int i=0;i<vfes->GetNE();i++)
  {
    // Standard local assembly and inversion for energy mass matrices.
    const int dof = vfes->GetFE(i)->GetDof();
    DenseMatrix Me(dof);
    //DenseMatrixInverse inv(&Me);
    Me_inv[i] = new DenseMatrix(dof,dof);
    
    MassIntegrator mi;

    int integrationOrder = 2*vfes->GetFE(i)->GetOrder();
    if(intRuleType==1 && vfes->GetFE(i)->GetGeomType()==Geometry::SQUARE) integrationOrder--; // when Gauss-Lobatto
    const IntegrationRule intRule = intRules->Get(vfes->GetFE(i)->GetGeomType(), integrationOrder);
    
    mi.SetIntRule(&intRule);
    mi.AssembleElementMatrix(*(vfes->GetFE(i) ), *(vfes->GetElementTransformation(i) ), Me);
    //inv.Factor();
    //inv.GetInverseMatrix( (*Me_inv)(i));
    //inv.GetInverseMatrix( Me_inv[i]);
    
    Me.Invert();
    for(int n=0;n<dof;n++) for(int j=0;j<dof;j++) (*Me_inv[i])(n,j) = Me(n,j);
    
    hposDofInvM[2*i   ] = temp.size();
    hposDofInvM[2*i +1] = dof;
    for(int j=0;j<dof;j++)
    {
      for(int k=0;k<dof;k++)
      {
        temp.push_back( (*Me_inv[i])(j,k) );
      }
    }
  }
  
  invMArray.UseDevice(true);
  invMArray.SetSize(temp.size());
  invMArray = 0.;
  auto hinvMArray = invMArray.HostWrite();
  for(int i=0;i<(int)temp.size();i++) hinvMArray[i] = temp[i];
  
#ifdef _GPU_
  auto dposDogInvM = posDofInvM.ReadWrite();
  auto dinvMarray = invMArray.ReadWrite();
#endif
  
  // create gradients object
  gradients = new Gradients(vfes,
                            gradUpfes,
                            dim,
                            num_equation,
                            Up,
                            gradUp,
                            gradUp_A,
                            intRules,
                            intRuleType,
                            nodesIDs,
                            posDofIds,
                            numElems,
                            Me_inv,
                            invMArray,
                            posDofInvM,
                            shapeWnor1,
                            shape2,
                            _elemFaces,
                            _elems12Q,
                            maxIntPoints,
                            maxDofs );
   
#ifdef DEBUG 
{
   int elem = 0;
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
   mfem::Mult(Kx, *Me_inv[elem], Dx);
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
  delete gradients;
  //delete[] Me_inv;
  for(int n=0;n<Me_inv.Size();n++) delete Me_inv[n];
  for( int i=0; i<forcing.Size();i++) delete forcing[i];
}


void RHSoperator::Mult(const Vector &x, Vector &y) const
{
  max_char_speed = 0.;
  
  // Update primite varibales
  updatePrimitives(x);
  gradients->computeGradients();
  
  // update boundary conditions
  if(bcIntegrator!=NULL) bcIntegrator->updateBCMean( Up );
  
  A->Mult(x, z);

  GetFlux(x, flux);
  
#ifdef _GPU_
  Vector fk,zk; fk.UseDevice(true);zk.UseDevice(true);
  fk.SetSize(dim*vfes->GetNDofs());
  zk.SetSize(vfes->GetNDofs());
#endif
  
  for(int eq=0;eq<num_equation;eq++)
  {
#ifdef _GPU_
    RHSoperator::copyDataForFluxIntegration_gpu(z, 
                                                flux,
                                                fk,
                                                zk,
                                                eq,
                                                vfes->GetNDofs(), 
                                                dim);
#else
    Vector fk(flux(eq).GetData(), dim * vfes->GetNDofs());
    Vector zk(z.HostReadWrite() + eq*vfes->GetNDofs(), vfes->GetNDofs());
#endif
    
    Aflux->AddMult(fk, zk);
#ifdef _GPU_
    RHSoperator::copyZk2Z_gpu(z,zk,eq,vfes->GetNDofs());
#endif
  }
  
  
  // add forcing terms
  for(int i=0; i<forcing.Size();i++)
  {
    // NOTE: Do not use RHSoperator::time here b/c it is not correctly
    // updated for each RK substep.  Instead, get time from parent
    // class using TimeDependentOperator::GetTime().
    forcing[i]->setTime(this->GetTime());
    forcing[i]->updateTerms();
    forcing[i]->addForcingIntegrals(z);
  }

  // 3. Multiply element-wise by the inverse mass matrices.
#ifdef _GPU_
  for(int eltype=0;eltype<numElems.Size();eltype++)
  {
    int elemOffset = 0;
    if( eltype!=0 )
    {
      for(int i=0;i<eltype;i++) elemOffset += h_numElems[i];
    }
    int dof = h_posDofIds[2*elemOffset +1];
    const int totDofs = vfes->GetNDofs();
    
    RHSoperator::multiPlyInvers_gpu(y,
                                    z,
                                    nodesIDs,
                                    posDofIds,
                                    invMArray,
                                    posDofInvM,
                                    num_equation,
                                    totDofs,
                                    h_numElems[eltype],
                                    elemOffset,
                                    dof);
  }
  
#else
  for(int el=0; el<vfes->GetNE();el++)
  {
    Vector zval;
    Array<int> vdofs;
    const int dof = vfes->GetFE(el)->GetDof();
    DenseMatrix zmat, ymat(dof, num_equation);
    
    // Return the vdofs ordered byNODES
    vfes->GetElementVDofs(el, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), dof, num_equation);
    
    mfem::Mult(*Me_inv[el], zmat, ymat);
    y.SetSubVector(vdofs, ymat.GetData());
  }  
#endif
}

void RHSoperator::copyZk2Z_gpu(Vector &z,Vector &zk,const int eq,const int dof)
{
#ifdef _GPU_
    const double *d_zk = zk.Read();
    double *d_z = z.ReadWrite();
    
    MFEM_FORALL(n,dof,
    {
      d_z[n + eq*dof] = d_zk[n];
    });
#endif
}

void RHSoperator::copyDataForFluxIntegration_gpu( const Vector &z, 
                                                  DenseTensor &flux,
                                                  Vector &fk,
                                                  Vector &zk,
                                                  const int eq,
                                                  const int dof, 
                                                  const int dim)
{
#ifdef _GPU_
  const double *d_flux = flux.Read();
  const double *d_z = z.Read();
  double *d_fk = fk.ReadWrite();
  double *d_zk = zk.ReadWrite();
  
  MFEM_FORALL(n,dof,
  {
    d_zk[n] = d_z[n + eq*dof ];
    for(int d=0;d<dim;d++)
    {
      d_fk[n + d*dof] = d_flux[n + d*dof + eq*dof*dim];
    }
  });
#endif
}

// Compute the flux at solution nodes.
void RHSoperator::GetFlux(const Vector &x, DenseTensor &flux) const
{
#ifdef _GPU_
  
  // ComputeConvectiveFluxes
  Fluxes::convectiveFluxes_gpu( x,
                                flux,
                                eqState->GetSpecificHeatRatio(),
                                vfes->GetNDofs(), 
                                dim,
                                num_equation);
  if( eqSystem==NS )
  {
    Fluxes::viscousFluxes_gpu(x,
                              gradUp,
                              flux,
                              eqState->GetSpecificHeatRatio(),
                              eqState->GetGasConstant(),
                              eqState->GetPrandtlNum(),
                              eqState->GetViscMultiplyer(),
                              vfes->GetNDofs(),
                              dim,
                              num_equation);
  }
#else
   DenseMatrix xmat(x.GetData(), vfes->GetNDofs(), num_equation);
   DenseMatrix f(num_equation, dim);
   
   double *dataGradUp = gradUp->GetData();
   
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();
   
   Vector state(num_equation);

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation; k++) (state)(k) = xmat(i, k);
      DenseMatrix gradUpi(num_equation,dim);
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) gradUpi(eq,d) = 
                               dataGradUp[i+eq*dof+d*num_equation*dof];
      }
      
      fluxClass->ComputeConvectiveFluxes(state,f);
     
      if( isSBP )
      {
        f *= alpha; // *= alpha
        double p = eqState->ComputePressure(state, dim);
        p *= 1. - alpha;
        for(int d=0; d<dim; d++)
        {
          f(d+1,d) += p;
          f(num_equation-1,d) += p*(state)(1+d)/(state)(0);
        }
      }
        
      if( eqSystem==NS )
      {
        DenseMatrix fvisc(num_equation,dim);
        fluxClass->ComputeViscousFluxes(state,gradUpi,fvisc);
        f -= fvisc;
      }

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux(i, d, k) = f(k, d);
         }
      }

      // Update max char speed
      const double mcs = eqState->ComputeMaxCharSpeed(state, dim);
      if (mcs > max_char_speed) { max_char_speed = mcs; }
   }
#endif // _GPU_

   double partition_max_char = max_char_speed;
   MPI_Allreduce(&partition_max_char, &max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
}

void RHSoperator::updatePrimitives(const Vector &x_in) const
{
#ifdef _GPU_
  
  RHSoperator::updatePrimitives_gpu(Up, 
                                    &x_in, 
                                    eqState->GetSpecificHeatRatio(), 
                                    vfes->GetNDofs(), 
                                    dim,
                                    num_equation );
#else
  double *dataUp = Up->GetData();
  for(int i=0;i<vfes->GetNDofs();i++)
  {
    Vector iState(num_equation);
    for(int eq=0;eq<num_equation;eq++) iState[eq] = x_in[i+eq*vfes->GetNDofs()];
    double p = eqState->ComputePressure(iState,dim);
    dataUp[i                   ] = iState[0];
    dataUp[i+  vfes->GetNDofs()] = iState[1]/iState[0];
    dataUp[i+2*vfes->GetNDofs()] = iState[2]/iState[0];
    if(dim==3) dataUp[i+3*vfes->GetNDofs()] = iState[3]/iState[0];
    dataUp[i+(num_equation-1)*vfes->GetNDofs()] = p;
  }
#endif // _GPU_
}

void RHSoperator::updatePrimitives_gpu(Vector *Up, 
                                      const Vector *x_in, 
                                      const double gamma, 
                                      const int ndofs, 
                                      const int dim,
                                      const int num_equations )
{
#ifdef _GPU_
  auto dataUp = Up->Write(); // make sure data is available in GPU
  auto dataIn = x_in->Read(); // make sure data is available in GPU
  
  MFEM_FORALL_2D(n,ndofs,num_equations,1,1,
  {
    MFEM_SHARED double state[5]; // assuming 5 equations
    //MFEM_SHARED double p;
    MFEM_SHARED double KE[3];
    
    MFEM_FOREACH_THREAD(eq,x,num_equations)
    {
      state[eq] = dataIn[n+eq*ndofs]; // loads data into shared memory
      MFEM_SYNC_THREAD;
      
      // compute pressure
      if( eq<dim ) KE[eq] = 0.5*state[eq]*state[eq]/state[0];
      MFEM_SYNC_THREAD;
      
      // each thread writes to global memory
      if( eq==0 ) dataUp[n        ] = state[0];
      if( eq==1 ) dataUp[n+  ndofs] = state[1]/state[0];
      if( eq==2 ) dataUp[n+2*ndofs] = state[2]/state[0];
      if( eq==3 && dim==3 )dataUp[n+3*ndofs] = state[3]/state[0];
      if( eq==num_equations-1 ) dataUp[n+(num_equations-1)*ndofs] = 
                                    EquationOfState::pressure(&state[0],&KE[0],gamma,dim,num_equations);
    }
  });
#endif
}

void RHSoperator::multiPlyInvers_gpu( Vector &y,
                                      Vector &z,
                                      const Array<int> &nodesIDs,
                                      const Array<int> &posDofIds,
                                      const Vector &invMArray,
                                      const Array<int> &posDofInvM,
                                      const int num_equation,
                                      const int totNumDof,
                                      const int NE,
                                      const int elemOffset,
                                      const int dof)
{
#ifdef _GPU_
  double *d_y = y.ReadWrite();
  const double *d_z = z.Read();
  auto d_nodesIDs = nodesIDs.Read();
  auto d_posDofIds = posDofIds.Read();
  auto d_posDofInvM = posDofInvM.Read();
  const double *d_invM = invMArray.Read();
  
  MFEM_FORALL_2D(el,NE,dof,1,1,
  {
    int eli = el + elemOffset;
    int offsetInv = d_posDofInvM[2*eli];
    int offsetIds = d_posDofIds[2*eli];
    // find out how to dinamically allocate shared memory to
    // store invMatrix values
    MFEM_FOREACH_THREAD(eq,y,num_equation)
    {
      MFEM_FOREACH_THREAD(i,x,dof)
      {
        int index = d_nodesIDs[offsetIds+i];
        double temp = 0;
        for(int k=0;k<dof;k++)
        {
          int indexk = d_nodesIDs[offsetIds +k];
          temp += d_invM[offsetInv +i*dof +k]*d_z[indexk + eq*totNumDof];
        }
        d_y[index+eq*totNumDof] = temp;
      }
    }
  });
#endif
}

