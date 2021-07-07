#include "BCintegrator.hpp"
#include "inletBC.hpp"
#include "outletBC.hpp"
#include "wallBC.hpp"
#include "general/forall.hpp"

BCintegrator::BCintegrator( MPI_Groups *_groupsMPI,
                            ParMesh *_mesh,
                            ParFiniteElementSpace *_vfes,
                            IntegrationRules *_intRules,
                            RiemannSolver *rsolver_, 
                            double &_dt,
                            EquationOfState *_eqState,
                            Fluxes *_fluxClass,
                            ParGridFunction *_Up,
                            ParGridFunction *_gradUp,
                            Vector &_shapesBC,
                            Vector &_normalsWBC,
                            Array<int> &_intPointsElIDBC,
                            const int _dim,
                            const int _num_equation,
                            double &_max_char_speed,
                            RunConfiguration &_runFile,
                            Array<int> &local_attr,
                            const int &_maxIntPoints,
                            const int &_maxDofs ):
groupsMPI(_groupsMPI),
config(_runFile),
rsolver(rsolver_),
eqState(_eqState),
fluxClass(_fluxClass),
max_char_speed(_max_char_speed),
intRules(_intRules),
mesh(_mesh),
vfes(_vfes),
Up(_Up),
gradUp(_gradUp),
shapesBC(_shapesBC),
normalsWBC(_normalsWBC),
intPointsElIDBC(_intPointsElIDBC),
dim(_dim),
num_equation(_num_equation),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
  BCmap.clear();
  
  // Init inlet BCs
  for(int in=0; in<config.GetInletPatchType()->size(); in++)
  {
    std::pair<int,InletType> patchANDtype = (*config.GetInletPatchType())[in];
    // check if attribute is in mesh
    bool attrInMesh = false;
    for(int i=0; i<local_attr.Size(); i++) 
    {
      if(patchANDtype.first==local_attr[i]) attrInMesh = true;
    }
    
    if( attrInMesh )
    {
      Array<double> data = config.GetInletData(in);
      BCmap[patchANDtype.first] = new InletBC(groupsMPI,
                                              rsolver, 
                                              eqState,
                                              vfes,
                                              intRules,
                                              _dt,
                                              dim,
                                              num_equation,
                                              patchANDtype.first,
                                              config.GetReferenceLength(),
                                              patchANDtype.second,
                                              data,
                                              _maxIntPoints,
                                              _maxDofs );
    }
  }
  
  // Init outlet BCs
  for(int o=0; o<config.GetOutletPatchType()->size(); o++)
  {
    std::pair<int,OutletType> patchANDtype = (*config.GetOutletPatchType())[o];
    // check if attribute is in mesh
    bool attrInMesh = false;
    for(int i=0; i<local_attr.Size(); i++) 
      if(patchANDtype.first==local_attr[i]) attrInMesh = true;
    
    if( attrInMesh )
    {
      Array<double> data = config.GetOutletData(o);
      BCmap[patchANDtype.first] = new OutletBC( groupsMPI,
                                                rsolver, 
                                                eqState,
                                                vfes,
                                                intRules,
                                                _dt,
                                                dim,
                                                num_equation,
                                                patchANDtype.first,
                                                config.GetReferenceLength(),
                                                patchANDtype.second,
                                                data,
                                                _maxIntPoints,
                                                _maxDofs );
    }
  }
  
  // Wall BCs
  for(int w=0; w<config.GetWallPatchType()->size(); w++)
  {
    std::pair<int,WallType> patchType = (*config.GetWallPatchType())[w];
    
    // check that patch is in mesh
    bool patchInMesh = false;
    for(int i=0; i<local_attr.Size(); i++)
    {
      if( patchType.first==local_attr[i] ) patchInMesh = true;
    }
    
    if( patchInMesh )
    {
      Array<double> wallData;
      if(patchType.second==VISC_ISOTH) wallData = config.GetWallData(w);
      
      BCmap[patchType.first] = new WallBC(rsolver, 
                                          eqState,
                                          fluxClass,
                                          vfes,
                                          intRules,
                                          _dt,
                                          dim,
                                          num_equation,
                                          patchType.first,
                                          patchType.second,
                                          wallData,
                                          intPointsElIDBC );
    }
  }
  
  // assign list of elements to each BC
  const int NumBCelems = vfes->GetNBE();
  if( NumBCelems>0 )
  {
    Mesh *mesh_bc = vfes->GetMesh();
    FaceElementTransformations *tr;
    
    for(int i=0;i<local_attr.Size();i++)
    {
      int attr = local_attr[i];
      Array<int> list;
      list.LoseData();
      
      for(int el=0;el<NumBCelems;el++)
      {
        tr = mesh_bc->GetBdrFaceTransformations(el);
        if (tr != NULL)
        {
          //if( tr->Attribute==attr ) list.Append( tr->Elem1No );
          if( tr->Attribute==attr ) list.Append( el );
        }
      }
      
      BCmap[attr]->setElementList( list );
    }
  }
}

BCintegrator::~BCintegrator()
{
  for(auto bc=BCmap.begin();bc!=BCmap.end(); bc++)
  {
    delete bc->second;
  }
}

void BCintegrator::initBCs()
{
  for(auto bc=BCmap.begin();bc!=BCmap.end(); bc++)
  {
    bc->second->initBCs();
  }
}



void BCintegrator::computeBdrFlux(const int attr, 
                                  Vector &normal,
                                  Vector &stateIn, 
                                  DenseMatrix &gradState,
                                  Vector &bdrFlux)
{
  BCmap[attr]->computeBdrFlux(normal,
                              stateIn, 
                              gradState,
                              bdrFlux);
}


void BCintegrator::updateBCMean(ParGridFunction *Up)
{
  for(auto bc=BCmap.begin();bc!=BCmap.end(); bc++)
  {
    bc->second->updateMean(intRules, Up);
  }
}

void BCintegrator::initState()
{
  for(auto bc=BCmap.begin();bc!=BCmap.end(); bc++)
  {
    bc->second->initState();
  }

}

void BCintegrator::integrateBCs( Vector &y, 
                                 const Vector &x,
                                 const Array<int> &nodesIDs,
                                 const Array<int> &posDofIds )
{
  for(auto bc=BCmap.begin();bc!=BCmap.end(); bc++)
  {
    bc->second->integrationBC(y, // output
                              x,
                              nodesIDs,
                              posDofIds,
                              Up,
                              gradUp,
                              shapesBC,
                              normalsWBC,
                              intPointsElIDBC,
                              maxIntPoints,
                              maxDofs);
  }
}



void BCintegrator::retrieveGradientsData_gpu( ParGridFunction* gradUp,
                                              DenseTensor &elGradUp,
                                              Array<int> &vdofs, 
                                              const int &num_equation, 
                                              const int &dim,
                                              const int &totalDofs,
                                              const int &elDofs )
{
#ifdef _GPU_
  const double *d_GradUp = gradUp->Read();
  double *d_elGradUp = elGradUp.ReadWrite();
  auto d_vdofs = vdofs.Read();
  
  MFEM_FORALL(i,elDofs,
  {
    const int index = d_vdofs[i];
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        d_elGradUp[i + eq*elDofs + d*elDofs*num_equation] = 
                      d_GradUp[index + eq*totalDofs + d*totalDofs*num_equation];
      }
    }
  });
#endif
}




void BCintegrator::AssembleFaceVector(const FiniteElement& el1, 
                                 const FiniteElement& el2, 
                                 FaceElementTransformations& Tr, 
                                 const Vector& elfun, 
                                 Vector& elvect)
{
  Vector shape1;
  Vector funval1(num_equation);
  Vector nor(dim);
  Vector fluxN(num_equation);
  //shape1.UseDevice(true);
  //funval1.UseDevice(true);
  //nor.UseDevice(true);
  //fluxN.UseDevice(true);
  
#ifndef _GPU_
  const double *dataGradUp = gradUp->HostRead();
#endif

  const int dof1 = el1.GetDof();
  //const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  //shape2.SetSize(dof2);

  //elvect.UseDevice(true);
  elvect.SetSize(dof1*num_equation);
  elvect = 0.0;

  //DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
  //DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
  
  // Retreive gradients of primitive variables for element
  Array<int> vdofs;
  vfes->GetElementVDofs(Tr.Elem1No, vdofs);
  int eldDof = vfes->GetFE(Tr.Elem1No)->GetDof();
  DenseTensor elGradUp(eldDof,num_equation,dim);
#ifdef _GPU_
  retrieveGradientsData_gpu(gradUp,
                            elGradUp,
                            vdofs, 
                            num_equation, 
                            dim,
                            vfes->GetNDofs(),
                            eldDof );
  elGradUp.HostRead();
#else
  for(int k=0; k<eldDof; k++)
  {
    int index = vdofs[k];
    int nDofs = vfes->GetNDofs();
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        elGradUp(k,eq,d) = dataGradUp[index + eq*nDofs + d*nDofs*num_equation];
      }
    }
  }
#endif

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
  
  const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);

  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip); // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

    // Interpolate elfun at the point
    //elfun1_mat.MultTranspose(shape1, funval1);

    // interpolated gradients
    DenseMatrix iGradUp(num_equation,dim);
    for(int eq=0; eq<num_equation; eq++)
    {

      // interpolation state
      double sum = 0.;
      for(int k=0; k<eldDof; k++)
	{
	  sum += elfun(k+eq*eldDof)*shape1(k);
	}
      funval1(eq) = sum;

      // interpolation gradients
      for(int d=0; d<dim; d++)
      { 
	sum = 0.;
        for(int k=0; k<eldDof; k++)
        {
          sum += elGradUp(k,eq,d)*shape1(k);
        }
        iGradUp(eq,d) = sum;
	//iGradUp(eq+d*num_equation) = sum;
      }
    }

    // Get the normal vector and the flux on the face
    CalcOrtho(Tr.Jacobian(), nor);
    computeBdrFlux(Tr.Attribute, nor,funval1,iGradUp,fluxN);

    fluxN *= ip.weight;
    for (int eq=0;eq<num_equation;eq++)
    {
      for (int s = 0; s < dof1; s++)
      {
        elvect(s+eq*dof1) -= fluxN(eq)*shape1(s);
      }
//          for (int s = 0; s < dof2; s++)
//          {
//             elvect2_mat(s, k) += fluxN(k) * shape2(s);
//          }
    }
  } // end loop over integration points
}
