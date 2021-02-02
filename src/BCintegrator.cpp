#include "BCintegrator.hpp"
#include "inletBC.hpp"
#include "outletBC.hpp"
#include "wallBC.hpp"

BCintegrator::BCintegrator( ParMesh *_mesh,
                            ParFiniteElementSpace *_vfes,
                            IntegrationRules *_intRules,
                            RiemannSolver *rsolver_, 
                            double &_dt,
                            EquationOfState *_eqState,
                            Fluxes *_fluxClass,
                            ParGridFunction *_Up,
                            Array<double> &_gradUp,
                            const int _dim,
                            const int _num_equation,
                            double &_max_char_speed,
                            RunConfiguration &_runFile ):
config(_runFile),
rsolver(rsolver_),
eqState(_eqState),
fluxClass(_fluxClass),
dim(_dim),
num_equation(_num_equation),
max_char_speed(_max_char_speed),
intRules(_intRules),
mesh(_mesh),
vfes(_vfes),
Up(_Up),
gradUp(_gradUp)
{
  // Check what attributes are in the local partition
  Array<int> local_attr;
  getAttributesInPartition(local_attr);
  
  // Init inlet BCs
  for(int in=0; in<config.GetInletPatchType()->Size(); in++)
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
      BCmap[patchANDtype.first] = new InletBC(rsolver, 
                                              eqState,
                                              vfes,
                                              intRules,
                                              _dt,
                                              dim,
                                              num_equation,
                                              patchANDtype.first,
                                              patchANDtype.second,
                                              data );
    }
  }
  
  // Init outlet BCs
  for(int o=0; o<config.GetOutletPatchType()->Size(); o++)
  {
    std::pair<int,OutletType> patchANDtype = (*config.GetOutletPatchType())[o];
    // check if attribute is in mesh
    bool attrInMesh = false;
    for(int i=0; i<local_attr.Size(); i++) 
      if(patchANDtype.first==local_attr[i]) attrInMesh = true;
    
    if( attrInMesh )
    {
      Array<double> data = config.GetOutletData(o);
      BCmap[patchANDtype.first] = new OutletBC( rsolver, 
                                                eqState,
                                                vfes,
                                                intRules,
                                                _dt,
                                                dim,
                                                num_equation,
                                                patchANDtype.first,
                                                patchANDtype.second,
                                                data );
    }
  }
  
  // Wall BCs
  for(int w=0; w<config.GetWallPatchType()->Size(); w++)
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
      BCmap[patchType.first] = new WallBC(rsolver, 
                                          eqState,
                                          fluxClass,
                                          vfes,
                                          intRules,
                                          _dt,
                                          dim,
                                          num_equation,
                                          patchType.first,
                                          patchType.second);
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

void BCintegrator::getAttributesInPartition(Array<int>& local_attr)
{
  local_attr.DeleteAll();
  for(int bel=0;bel<vfes->GetNBE(); bel++)
  {
    int attr = vfes->GetBdrAttribute(bel);
    bool attrInArray = false;
    for(int i=0;i<local_attr.Size();i++)
    {
      if( local_attr[i]==attr ) attrInArray = true;
    }
    if( !attrInArray ) local_attr.Append( attr );
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

  const int dof1 = el1.GetDof();
  //const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  //shape2.SetSize(dof2);

  elvect.SetSize(dof1*num_equation);
  elvect = 0.0;

  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
  
  // Retreive gradients of primitive variables for element
  Array<int> vdofs;
  vfes->GetElementVDofs(Tr.Elem1No, vdofs);
  int eldDof = vfes->GetFE(Tr.Elem1No)->GetDof();
  DenseTensor elGradUp(eldDof,num_equation,dim);
  for(int k=0; k<eldDof; k++)
  {
    int index = vdofs[k];
    int nDofs = vfes->GetNDofs();
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        elGradUp(k,eq,d) = gradUp[index + eq*nDofs + d*nDofs*num_equation];
      }
    }
  }
  
//   cout<<"Elem# "<< Tr.Elem1No <<endl;
//   for(int k=0; k<eldDof; k++)
//   {
//     int index = vdofs[k];
//     int nDofs = vfes->GetNDofs();
//     for(int d=0;d<dim;d++)
//     {
//       for(int eq=0;eq<num_equation;eq++)
//       {
//         cout<<gradUp[index + eq*nDofs + d*nDofs*num_equation]<<" ";
//       }
//     }
//     cout<<endl;
//   }

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
    elfun1_mat.MultTranspose(shape1, funval1);
    
    // interpolated gradients
    DenseMatrix iGradUp(num_equation,dim);
    for(int eq=0; eq<num_equation; eq++)
    {
      for(int d=0; d<dim; d++)
      {
        double sum = 0.;
        for(int k=0; k<eldDof; k++)
        {
          sum += elGradUp(k,eq,d)*shape1(k);
        }
        iGradUp(eq,d) = sum;
      }
    }

    // Get the normal vector and the flux on the face
    CalcOrtho(Tr.Jacobian(), nor);
    computeBdrFlux(Tr.Attribute, nor,funval1,iGradUp,fluxN);

    fluxN *= ip.weight;
    for (int k = 0; k < num_equation; k++)
    {
      for (int s = 0; s < dof1; s++)
      {
        elvect1_mat(s, k) -= fluxN(k) * shape1(s);
      }
//          for (int s = 0; s < dof2; s++)
//          {
//             elvect2_mat(s, k) += fluxN(k) * shape2(s);
//          }
    }
  }
}
