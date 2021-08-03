#include "gradients.hpp"
#include "dgNonlinearForm.hpp"

Gradients::Gradients(ParFiniteElementSpace* _vfes, 
                     ParFiniteElementSpace* _gradUpfes, 
                     int _dim,
                     int _num_equation,
                     ParGridFunction* _Up, 
                     ParGridFunction* _gradUp, 
                     //ParNonlinearForm *_gradUp_A,
                     GradNonLinearForm *_gradUp_A,
                     IntegrationRules *_intRules,
                     int _intRuleType,
                     Array<int>& _nodesIDs, 
                     Array<int>& _posDofIds, 
                     Array<int>& _numElems,
                     //DenseMatrix *_Me_inv,
                     Array<DenseMatrix*> &_Me_inv,
                     Vector &_invMArray,
                     Array<int> &_posDofInvM,
                     Vector &_shapeWnor1,
                     Vector &_shape2,
                     Array<int> &_elemFaces,
                     Array<int> &_elems12Q,
                     const int &_maxIntPoints,
                     const int &_maxDofs ):
ParNonlinearForm(_vfes),
vfes(_vfes),
gradUpfes(_gradUpfes),
dim(_dim),
num_equation(_num_equation),
Up(_Up),
gradUp(_gradUp),
gradUp_A(_gradUp_A),
intRules(_intRules),
intRuleType(_intRuleType),
nodesIDs(_nodesIDs),
posDofIds(_posDofIds),
numElems(_numElems),
Me_inv(_Me_inv),
invMArray(_invMArray),
posDofInvM(_posDofInvM),
shapeWnor1(_shapeWnor1),
shape2(_shape2),
elemFaces(_elemFaces),
elems12Q(_elems12Q),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
  h_numElems = numElems.HostReadWrite();
  h_posDofIds = posDofIds.HostReadWrite();
  
  // fill out gradient shape function arrays and their indirections
  std::vector<double> temp;
  std::vector<int> positions;
  temp.clear(); positions.clear();
  
  for(int el=0;el<vfes->GetNE();el++)
  {
    positions.push_back( (int)temp.size() );
    
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int elDof = elem->GetDof();
    
    // element volume integral
    int intorder = 2*elem->GetOrder();
    if(intRuleType==1 && elem->GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
    
    positions.push_back( ir->GetNPoints() );
    
    for(int i=0;i<ir->GetNPoints();i++)
    {
      IntegrationPoint ip = ir->IntPoint(i);
      Tr->SetIntPoint( &ip );
      
      // Calculate the shape functions
      Vector shape; shape.UseDevice(false);
      shape.SetSize(elDof);
      Vector dshapeVec; dshapeVec.UseDevice(false);
      dshapeVec.SetSize(elDof*dim);
      dshapeVec = 0.;
      DenseMatrix dshape(dshapeVec.HostReadWrite(),elDof,dim);
      elem->CalcShape(ip, shape);
      elem->CalcPhysDShape(*Tr,dshape);
      double detJac = Tr->Jacobian().Det()*ip.weight;
      
      for(int n=0;n<elDof;n++) temp.push_back( shape[n] );
      for(int d=0;d<dim;d++) for(int n=0;n<elDof;n++) temp.push_back( dshape(n,d) );
      temp.push_back( detJac );
    }
  }
  
  elemShapeDshapeWJ.UseDevice(true);
  elemShapeDshapeWJ.SetSize( (int)temp.size() );
  elemShapeDshapeWJ = 0.;
  auto helemShapeDshapeWJ = elemShapeDshapeWJ.HostWrite();
  for(int i=0;i<elemShapeDshapeWJ.Size();i++) helemShapeDshapeWJ[i] = temp[i];
  elemShapeDshapeWJ.Read();
  
  elemPosQ_shapeDshapeWJ.SetSize( (int)positions.size() );
  for(int i=0;i<elemPosQ_shapeDshapeWJ.Size();i++) elemPosQ_shapeDshapeWJ[i] = positions[i];
  elemPosQ_shapeDshapeWJ.Read();

#ifndef _GPU_
  // element derivative stiffness matrix
  Ke.SetSize( vfes->GetNE() );
  for(int el=0;el<vfes->GetNE();el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int eldDof = elem->GetDof();

    Ke[el] = new DenseMatrix(eldDof, dim*eldDof);

    // element volume integral
    int intorder = 2*elem->GetOrder();
    if(intRuleType==1 && elem->GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);

    Vector shape(eldDof);
    DenseMatrix dshape(eldDof,dim);
    DenseMatrix iGradUp(num_equation,dim);

    for(int i=0;i<ir->GetNPoints();i++)
    {
      IntegrationPoint ip = ir->IntPoint(i);
      Tr->SetIntPoint( &ip );

      // Calculate the shape functions
      elem->CalcShape(ip, shape);
      elem->CalcPhysDShape(*Tr,dshape);

      double detJac = Tr->Jacobian().Det()*ip.weight;

      for(int d=0;d<dim;d++)
      {
        for(int k=0;k<eldDof;k++)
        {
          for(int j=0;j<eldDof;j++)
          {
            (*Ke[el])(j, k+d*eldDof) += shape(j)*dshape(k,d)*detJac;
          }
        }
      }
    }
  }
#endif
}

Gradients::~Gradients()
{
#ifndef _GPU_
  for(int n=0;n<Ke.Size();n++) delete Ke[n];
#endif
}

void Gradients::computeGradients()
{
#ifdef _GPU_
  const Vector &px = Prolongate(*Up);
  
  // INTEGRATION DOMAIN SHARED FACES
  // WARNING! NOTE implement face integration to control loop
  Vector faceContrib;
  faceContrib.UseDevice(true);
  faceContrib.NewMemoryAndSize(gradUp->GetMemory(),
                               dim*num_equation*vfes->GetNDofs(),
                               false );
  DGNonLinearForm::setToZero_gpu(faceContrib,faceContrib.Size());
  
  gradUp_A->Mult(Up,faceContrib);
   
  for(int elType=0;elType<numElems.Size();elType++)
  {
    int elemOffset = 0;
    if( elType!=0 )
    {
      for(int i=0;i<elType;i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2*elemOffset +1];
    
    computeGradients_gpu(h_numElems[elType],
                        elemOffset,
                        dof_el,
                        vfes->GetNDofs(),
                        px,
                        *gradUp,
                        num_equation,
                        dim,
                        posDofIds,
                        nodesIDs,
                        elemShapeDshapeWJ,
                        elemPosQ_shapeDshapeWJ,
                        invMArray,
                        posDofInvM,
                        elemFaces,
                        shapeWnor1,
                        shape2,
                        maxDofs,
                        maxIntPoints,
                        elems12Q );
  }
  
  //gradUp->ExchangeFaceNbrData();
#else
  computeGradients_cpu();
#endif
}


void Gradients::computeGradients_cpu()
{
  const int totalDofs = vfes->GetNDofs();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();

  // Vars for face contributions
  Vector faceContrib(dim*num_equation*totalDofs);
  Vector xUp(dim*num_equation*totalDofs);
  xUp = 0.;
  faceContrib = 0.;

  // // compute volume integral and fill out above vectors
  // for(int el=0;el<vfes->GetNE();el++)
  // {
  //   const FiniteElement *elem = vfes->GetFE(el);
  //   ElementTransformation *Tr = vfes->GetElementTransformation(el);

  //   // get local primitive variables
  //   Array<int> vdofs;
  //   vfes->GetElementVDofs(el, vdofs);
  //   const int eldDof = elem->GetDof();
  //   DenseMatrix elUp(eldDof,num_equation);
  //   for(int d=0; d<eldDof; d++)
  //   {
  //     int index = vdofs[d];
  //     for(int eq=0;eq<num_equation;eq++)
  //     {
  //       elUp(d,eq) = dataUp[index +eq*totalDofs];
  //       xUp[index+eq*totalDofs] = elUp(d,eq);
  //     }
  //   }

  //   DenseMatrix elGradUp(eldDof,num_equation*dim);
  //   elGradUp = 0.;


  //   // element volume integral
  //   int intorder = 2*elem->GetOrder();
  //   if(intRuleType==1 && elem->GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
  //   const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);

  //   Vector shape(eldDof);
  //   DenseMatrix dshape(eldDof,dim);
  //   DenseMatrix iGradUp(num_equation,dim);

  //   for(int i=0;i<ir->GetNPoints();i++)
  //   {
  //     IntegrationPoint ip = ir->IntPoint(i);
  //     Tr->SetIntPoint( &ip );

  //     // Calculate the shape functions
  //     elem->CalcShape(ip, shape);
  //     elem->CalcPhysDShape(*Tr,dshape);

  //     // calc grad Up at int. point
  //     MultAtB(elUp, dshape, iGradUp);

  //     double detJac = Tr->Jacobian().Det()*ip.weight;

  //     // Add volume contrubutions to gradient
  //     for(int eq=0;eq<num_equation;eq++)
  //     {
  //       for(int d=0;d<dim;d++)
  //       {
  //         for(int j=0;j<eldDof;j++)
  //         {
  //           //elGradUp(j,eq+d*num_equation) += iUp[eq]*dshape(j,d)*detJac;
  //           elGradUp(j,eq+d*num_equation) += shape(j)*iGradUp(eq,d)*detJac;
  //         }
  //       }
  //     }
  //   }
    
  //   // transfer result into gradUp
  //   for(int k=0; k<eldDof; k++)
  //   {
  //     int index = vdofs[k];
  //     for(int eq=0;eq<num_equation;eq++)
  //     {
  //       for(int d=0;d<dim;d++)
  //       {
  //         dataGradUp[index+eq*totalDofs+d*num_equation*totalDofs] = 
  //                                       -elGradUp(k,eq+d*num_equation);
  //       }
  //     } 
  //   }
  // }

  // compute volume integral and fill out above vectors
  DenseMatrix elGradUp;
  for(int el=0;el<vfes->GetNE();el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);

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
      }
    }

    elGradUp.SetSize(eldDof,num_equation*dim);
    elGradUp = 0.;

    // Add volume contrubutions to gradient
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        for(int j=0;j<eldDof;j++)
        {
          for(int k=0;k<eldDof;k++)
          {
            elGradUp(j,eq+d*num_equation) += (*Ke[el])(j,k+d*eldDof)*elUp(k,eq);
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
  gradUp_A->Mult(Up,faceContrib);
  
  // Add contributions and multiply by invers mass matrix
  for(int el=0;el<vfes->GetNE();el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    const int eldDof = elem->GetDof();

    Array<int> vdofs;
    vfes->GetElementVDofs(el, vdofs);

    Vector aux(eldDof);
    Vector rhs(eldDof);

    for(int d=0;d<dim;d++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {

        for(int k=0;k<eldDof;k++)
        {
          int index = vdofs[k];
//           rhs[k] = gradUp[index+eq*totalDofs+d*num_equation*totalDofs]+
//                   faceContrib[index+eq*totalDofs+d*num_equation*totalDofs];
          rhs[k] = -dataGradUp[index+eq*totalDofs+d*num_equation*totalDofs]+
                   faceContrib[index+eq*totalDofs+d*num_equation*totalDofs];
        }

        // mult by inv mass matrix
        Me_inv[el]->Mult(rhs, aux);

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

#ifdef _GPU_
void Gradients::computeGradients_gpu(const int numElems,
                                     const int offsetElems,
                                     const int elDof,
                                     const int totalDofs,
                                     const Vector &Up,
                                     Vector &gradUp,
                                     const int num_equation,
                                     const int dim,
                                     const Array<int> &posDofIds,
                                     const Array<int> &nodesIDs,
                                     const Vector elemShapeDshapeWJ,
                                     const Array<int> elemPosQ_shapeDshapeWJ,
                                     const Vector &invMArray,
                                     const Array<int> &posDofInvM,
                                     const Array<int> &elemFaces,
                                     const Vector &shapeWnor1,
                                     const Vector &shape2,
                                     const int &maxDofs,
                                     const int &maxIntPoints,
                                     const Array<int> elems12Q )
{
  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = posDofIds.Read();
  auto d_nodesIDs = nodesIDs.Read();
  const double *d_elemShapeDshapeWJ = elemShapeDshapeWJ.Read();
  auto d_elemPosQ_shapeDshapeWJ = elemPosQ_shapeDshapeWJ.Read();
  const double *d_invMArray =invMArray.Read();
  auto d_posDofInvM = posDofInvM.Read();
  
  // pointers for face integration
  auto d_elemFaces = elemFaces.Read();
  auto d_shapeWnor1 = shapeWnor1.Read();
  const double *d_shape2 = shape2.Read();
  auto d_elems12Q = elems12Q.Read();
  
  MFEM_FORALL_2D(el,numElems,elDof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,elDof)
    {
      // allocate shared memory
      MFEM_SHARED double Ui[216*5],/*Uj[216*5],*/ gradUpi[216*5*3]/*,tmpGrad[216*5*3]*/;
      MFEM_SHARED double l1[216],dl1[216*3];
      MFEM_SHARED double meanU[5], u1[5], nor[3];
      
      const int eli = el + offsetElems;
      const int offsetIDs    = d_posDofIds[2*eli];
      const int offsetDShape = d_elemPosQ_shapeDshapeWJ[2*eli   ];
      const int Q            = d_elemPosQ_shapeDshapeWJ[2*eli +1];
      
      const int indexi = d_nodesIDs[offsetIDs + i];
      
      // fill out Ui and init gradUpi
      for(int eq=0;eq<num_equation;eq++)
      {
        Ui[i + eq*elDof] = d_Up[indexi + eq*totalDofs];
        for(int d=0;d<dim;d++)
        {
          gradUpi[i +eq*elDof +d*num_equation*elDof] = 0.;
          
        }
      }
      MFEM_SYNC_THREAD;
      
      // volume integral contribution to gradient
      double gradUpk[5*3];
      for(int k=0;k<Q;k++)
      {
        const double weightDetJac = d_elemShapeDshapeWJ[offsetDShape+elDof+dim*elDof+k*((dim+1)*elDof+1)];
        l1[i]                     = d_elemShapeDshapeWJ[offsetDShape+i+              k*((dim+1)*elDof+1)];
        for(int d=0;d<dim;d++) dl1[i+d*elDof] = 
                                    d_elemShapeDshapeWJ[offsetDShape+elDof+i+d*elDof+k*((dim+1)*elDof+1)];
        for(int j=0;j<num_equation*dim;j++) gradUpk[j] = 0.;
        MFEM_SYNC_THREAD;
        
        for(int eq=0;eq<num_equation;eq++)
        {
          for(int d=0;d<dim;d++)
          {
            for(int j=0;j<elDof;j++) gradUpk[eq+d*num_equation] += dl1[j+d*elDof]*Ui[j+eq*elDof];
          }
        }
        
        // add integration point contribution
        for(int eq=0;eq<num_equation;eq++)
        {
          for(int d=0;d<dim;d++)
          {
            gradUpi[i+eq*elDof+d*num_equation*elDof] += gradUpk[eq+d*num_equation]*l1[i]*weightDetJac;
          }
        }
      }
      MFEM_SYNC_THREAD;
      // because we don't have enough shared memory, save the previous result in
      // global memory
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] = 
                                      gradUpi[i+eq*elDof+d*num_equation*elDof];
        }
      }
      
      // ================  FACE CONTRIBUTION  ================
      const int elFaces = d_elemFaces[7*eli];
      for(int face=0;face<elFaces;face++)
      {
        const int gFace = d_elemFaces[7*eli+face+1];
        const int Q     = d_elems12Q[3*gFace+2];
        const int offsetShape1 = gFace*maxIntPoints*(maxDofs+1+dim);
        const int offsetShape2 = gFace*maxIntPoints*maxDofs;
        bool swapElems = false;
        
        // get neighbor
        int elj = d_elems12Q[3*gFace];
        if( elj==eli )
        {
          elj = d_elems12Q[3*gFace+1];
        }else
        {
          swapElems = true;
        }
        
        const int offsetElj = d_posDofIds[2*elj];
        int dofj = d_posDofIds[2*elj+1];
        
        int dof1 = elDof;
        int dof2 = dofj;
        if( swapElems )
        {
          dof1 = dofj;
          dof2 = elDof;
        }
        
        // get data from neightbor
        for(int eq=0;eq<num_equation;eq++)
        {
          for(int j=i;j<dofj;j+=elDof)
          {
            int index = d_nodesIDs[offsetElj+j];
            //Uj[j + eq*dofj] = d_Up[index + eq*totalDofs];
            // since gradUpi is no longer needed, use it to store Uj
            gradUpi[j + eq*dofj] = d_Up[index + eq*totalDofs];
          }
        }
        MFEM_SYNC_THREAD;
        
        // loop over integration points
        for(int k=0;k<Q;k++)
        {
          const double weight = d_shapeWnor1[offsetShape1+maxDofs +k*(maxDofs+1+dim)];
          
          for(int eq=i;eq<num_equation;eq+=elDof)
          {
            meanU[eq] = 0.;
            u1[eq] = 0.;
            if( eq<dim ) nor[eq] = d_shapeWnor1[offsetShape1+maxDofs+1+eq+k*(maxDofs+1+dim)];
          }
          
          for(int j=i;j<dof1;j+=elDof) l1[j] = d_shapeWnor1[offsetShape1+j+k*(maxDofs+1+dim)];
          // use dl1 to store shape2 values
          for(int j=i;j<dof2;j+=elDof) dl1[j] = d_shape2[offsetShape2+j+k*maxDofs];
          MFEM_SYNC_THREAD;
            
          if( i==elDof-1 ) // NOTE: only one thread does this - rethink this
          {
            for(int eq=0;eq<num_equation;eq++)
            {
              if( swapElems )
              {
                //for(int j=0;j<dof1;j++) u1[eq] += l1[j]*Uj[j+eq*dof1];
                for(int j=0;j<dof1;j++) u1[eq] += l1[j]*gradUpi[j+eq*dof1];
                for(int j=0;j<dof2;j++) meanU[eq] += dl1[j]*Ui[j+eq*dof2];
              }else
              {
                for(int j=0;j<dof1;j++) u1[eq] += l1[j]*Ui[j+eq*dof1];
                //for(int j=0;j<dof2;j++) meanU[eq] += dl1[j]*Uj[j+eq*dof2];
                 for(int j=0;j<dof2;j++) meanU[eq] += dl1[j]*gradUpi[j+eq*dof2];
              }
            }
          }
          MFEM_SYNC_THREAD;
          
          for(int eq=i;eq<num_equation;eq+=elDof) meanU[eq] = 0.5*(meanU[eq]-u1[eq]);
          MFEM_SYNC_THREAD;
          
          // add integration point contribution
          for(int eq=0;eq<num_equation;eq++)
          {
            for(int d=0;d<dim;d++)
            {
              double contrib;
              if( swapElems ) contrib = weight*dl1[i]*meanU[eq]*nor[d];
              else contrib = weight*l1[i]*meanU[eq]*nor[d];
                 
              d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] += contrib;
            }
          }
        }
      }
      
      /*// MULTIPLY BY INVERSE OF MASS MATRIX
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          //tmpGrad[i +eq*elDof + d*num_equation*elDof] = 0.;
          gradUpi[i +eq*elDof + d*num_equation*elDof] = 0.;
          for(int n=0;n<elDof;n++)
          {
            //tmpGrad[i +eq*elDof + d*num_equation*elDof] += 
            //                d_invMArray[offsetInv +i*elDof +n]*gradUpi[n + eq*elDof];
            // NOTE: this is VERY EXPENSIVE! Rethink this to see if it can be done
            //       differently with the given amount of shared memory (64KB)
            const int indexn = d_nodesIDs[offsetIDs + n];
            gradUpi[i +eq*elDof + d*num_equation*elDof] += 
                                d_gradUp[indexn+eq*totalDofs+d*num_equation*totalDofs]*
                                d_invMArray[offsetInv +i*elDof +n];
          }
        }
      }
      
      // add integral contrubution to global memory
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          //d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] = 
          //                        tmpGrad[i+eq*num_equation+d*num_equation*elDof];
          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] = 
                                    gradUpi[i+eq*elDof+d*num_equation*elDof];
        }
      }*/
      
    }
  });
  
  
  // MULTIPLY BY INVERSE OF MASS MATRIX
  //NOTE: I'm sure this can be done more efficiently. Have a think
  MFEM_FORALL_2D(el,numElems,elDof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,elDof)
    {
      MFEM_SHARED double gradUpi[216*5*3];
      
      const int eli = el + offsetElems;
      const int offsetIDs    = d_posDofIds[2*eli];
      const int offsetInv    = d_posDofInvM[2*eli];
      const int indexi = d_nodesIDs[offsetIDs + i];
      
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          //tmpGrad[i +eq*elDof + d*num_equation*elDof] = 0.;
          gradUpi[i +eq*elDof + d*num_equation*elDof] = 
                          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs];
        }
      }
      MFEM_SYNC_THREAD;
      
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          double temp = 0;
          for(int n=0;n<elDof;n++)
          {
            temp += gradUpi[n+eq*elDof+d*num_equation*elDof]*
                                d_invMArray[offsetInv +i*elDof +n];
          }
          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] = temp;
        }
      }
      /*MFEM_SYNC_THREAD;
      
      // copy to global memory
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] = 
                                    temp[i +eq*elDof + d*num_equation*elDof];
        }
      }*/
    }
  });
}
#endif

