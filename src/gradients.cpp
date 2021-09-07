#include "gradients.hpp"
#include "dgNonlinearForm.hpp"

Gradients::Gradients(ParFiniteElementSpace* _vfes, 
                     ParFiniteElementSpace* _gradUpfes, 
                     int _dim,
                     int _num_equation,
                     ParGridFunction* _Up, 
                     ParGridFunction* _gradUp, 
                     EquationOfState *_eqState,
                     GradNonLinearForm *_gradUp_A,
                     IntegrationRules *_intRules,
                     int _intRuleType,
                     const volumeFaceIntegrationArrays &_gpuArrays,
                     Array<DenseMatrix*> &_Me_inv,
                     Vector &_invMArray,
                     Array<int> &_posDofInvM,
                     const int &_maxIntPoints,
                     const int &_maxDofs ):
ParNonlinearForm(_vfes),
vfes(_vfes),
gradUpfes(_gradUpfes),
dim(_dim),
num_equation(_num_equation),
Up(_Up),
gradUp(_gradUp),
eqState(_eqState),
gradUp_A(_gradUp_A),
intRules(_intRules),
intRuleType(_intRuleType),
gpuArrays(_gpuArrays),
Me_inv(_Me_inv),
invMArray(_invMArray),
posDofInvM(_posDofInvM),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
  h_numElems  = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();

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
  const int totalDofs = vfes->GetNDofs();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();

  // Vars for face contributions
  Vector faceContrib(dim*num_equation*totalDofs);
  faceContrib = 0.;

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
  
  gradUp->ExchangeFaceNbrData();
}

#ifdef _GPU_
void Gradients::computeGradients_domain()
{
  DGNonLinearForm::setToZero_gpu(*gradUp,gradUp->Size());
  
  for(int elType=0;elType<gpuArrays.numElems.Size();elType++)
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
                        *Up, //px,
                        *gradUp,
                        num_equation,
                        dim,
                        gpuArrays,
                        maxDofs,
                        maxIntPoints );
  }
}

void Gradients::computeGradients_bdr()
{
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if( Nshared>0 )
  {
    integrationGradSharedFace_gpu(Up,
                                  transferUp->face_nbr_data,
                                  gradUp,
                                  vfes->GetNDofs(),
                                  dim,
                                  num_equation,
                                  eqState->GetSpecificHeatRatio(),
                                  eqState->GetGasConstant(), 
                                  eqState->GetViscMultiplyer(),
                                  eqState->GetBulkViscMultiplyer(),
                                  eqState->GetPrandtlNum(),
                                  gpuArrays,
                                  parallelData,
                                  maxIntPoints,
                                  maxDofs);
  }
  
  // Multiply by inverse mass matrix
  for(int elType=0;elType<gpuArrays.numElems.Size();elType++)
  {
    int elemOffset = 0;
    if( elType!=0 )
    {
      for(int i=0;i<elType;i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2*elemOffset +1];
    
    multInverse_gpu(h_numElems[elType],
                    elemOffset,
                    dof_el,
                    vfes->GetNDofs(),
                    *gradUp,
                    num_equation,
                    dim,
                    gpuArrays,
                    invMArray,
                    posDofInvM );
  }
}

void Gradients::computeGradients_gpu(const int numElems,
                                     const int offsetElems,
                                     const int elDof,
                                     const int totalDofs,
                                     const Vector &Up,
                                     Vector &gradUp,
                                     const int num_equation,
                                     const int dim,
                                     const volumeFaceIntegrationArrays &gpuArrays,
                                     const int &maxDofs,
                                     const int &maxIntPoints )
{
  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  const double *d_elemShapeDshapeWJ = gpuArrays.elemShapeDshapeWJ.Read();
  auto d_elemPosQ_shapeDshapeWJ = gpuArrays.elemPosQ_shapeDshapeWJ.Read();
  
  // pointers for face integration
  auto d_elemFaces = gpuArrays.elemFaces.Read();
  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();
  
  MFEM_FORALL_2D(el,numElems,elDof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,elDof)
    {
      // allocate shared memory
      MFEM_SHARED double Ui[216*5],/*Uj[216*5],*/ gradUpi[216*5*3]/*,tmpGrad[216*5*3]*/;
      MFEM_SHARED double l1[216],dl1[216*3];
      MFEM_SHARED double meanU[5], u1[5], nor[3];
      MFEM_SHARED int index_j[216];
      
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
          d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs] += 
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
        
        for(int j=i;j<dofj;j+=elDof) index_j[j] = d_nodesIDs[offsetElj+j];
        MFEM_SYNC_THREAD;
        
        // get data from neightbor
        for(int eq=0;eq<num_equation;eq++)
        {
          for(int j=i;j<dofj;j+=elDof)
          {
            int index = index_j[j];
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
          
          for(int eq=i;eq<num_equation;eq+=elDof)
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
    }
  });
  
}

void Gradients::integrationGradSharedFace_gpu(const Vector *Up,
                                              const Vector &faceUp,
                                              ParGridFunction *gradUp,
                                              const int &Ndofs,
                                              const int &dim,
                                              const int &num_equation,
                                              const double &gamma,
                                              const double &Rg,
                                              const double &viscMult,
                                              const double &bulkViscMult,
                                              const double &Pr,
                                              const volumeFaceIntegrationArrays &gpuArrays,
                                              const parallelFacesIntegrationArrays *parallelData,
                                              const int &maxIntPoints,
                                              const int &maxDofs)
{
  const double *d_up = Up->Read();
  double *d_gradUp = gradUp->ReadWrite();
  const double *d_faceData = faceUp.Read();
  
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();
  
  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const double *d_sharedShape2 = parallelData->sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = parallelData->sharedVdofs.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();
  
  MFEM_FORALL_2D(el,parallelData->sharedElemsFaces.Size()/7,maxDofs,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,maxDofs)
    {
      MFEM_SHARED double Upi[216*5], Upj[216*5], Fcontrib[216*5*3];
      MFEM_SHARED double l1[216],l2[216];
      MFEM_SHARED double meanUp[5], up1[5], up2[5], nor[3];
      
      const int el1      = d_sharedElemsFaces[0+el*7];
      const int numFaces = d_sharedElemsFaces[1+el*7];
      const int dof1 = d_sharedElem1Dof12Q[1+d_sharedElemsFaces[2+el*7]*4];
      
      bool elemDataRecovered = false;
      int indexi;
      
      for(int elFace=0;elFace<numFaces;elFace++)
      {
        const int f = d_sharedElemsFaces[1+elFace+1+el*7];

        const int dof2 = d_sharedElem1Dof12Q[2+f*4];
        const int Q    = d_sharedElem1Dof12Q[3+f*4];
        
        const int offsetEl1 = d_posDofIds[2*el1];
        
        if( i<dof1 && !elemDataRecovered )
        {
          indexi = d_nodesIDs[offsetEl1+i];
          for(int eq=0;eq<num_equation;eq++)
          {
            for(int d=0;d<dim;d++) Fcontrib[i+eq*dof1+d*num_equation*dof1] = 0.;
            Upi[i+eq*dof1] = d_up[indexi+eq*Ndofs];
          }
          elemDataRecovered = true;
        }
        if( i<dof2 ) // recover data from neighbor
        {
          for(int eq=0;eq<num_equation;eq++)
          {
            int index = d_sharedVdofs[i+eq*maxDofs+f*num_equation*maxDofs];
            Upj[i+eq*dof2] = d_faceData[index];
          }
        }
        MFEM_SYNC_THREAD;
        
        for(int k=0;k<Q;k++)
        {
          const double weight = 
            d_sharedShapeWnor1[maxDofs+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(i<dof1) l1[i] = d_sharedShapeWnor1[i+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(i<dim ) nor[i]= 
            d_sharedShapeWnor1[maxDofs+1+i+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(dim==2 && i==maxDofs-1) nor[2] = 0.; 
          if(i<dof2) l2[i] = d_sharedShape2[i+k*maxDofs+f*maxIntPoints*maxDofs];
          if(i<num_equation)
          {
            up1[i] = 0.;
            up2[i] = 0.;
            meanUp[i] = 0.;
          }
          MFEM_SYNC_THREAD;
          
          // interpolate
          if(i<num_equation)
          {
            for(int n=0;n<dof1;n++) up1[i] += Upi[n+i*dof1]*l1[n];
            for(int n=0;n<dof2;n++) up2[i] += Upj[n+i*dof2]*l2[n];
          }
          MFEM_SYNC_THREAD;
          
          if( i<num_equation ) meanUp[i] = 0.5*(up2[i]-up1[i]);
          MFEM_SYNC_THREAD;
          
          // add integration point contribution
          for(int d=0;d<dim;d++)
          {
            for(int eq=0;eq<num_equation;eq++)
            {
              if(i<dof1) Fcontrib[i+eq*dof1+d*num_equation*dof1] += meanUp[eq]*weight*l1[i]*nor[d];
            }
          }
        }
        MFEM_SYNC_THREAD;
      } 
      
      // write to global memory
      if(i<dof1)
      {
        for(int d=0;d<dim;d++)
        {
          for(int eq=0;eq<num_equation;eq++)
          {
            d_gradUp[indexi+eq*Ndofs+d*num_equation*Ndofs] += Fcontrib[i+eq*dof1+d*num_equation*dof1];
          }
        }
      }
    }
  });
}

void Gradients::multInverse_gpu(const int numElems,
                                const int offsetElems,
                                const int elDof,
                                const int totalDofs,
                                Vector &gradUp,
                                const int num_equation,
                                const int dim,
                                const volumeFaceIntegrationArrays &gpuArrays,
                                const Vector &invMArray,
                                const Array<int> &posDofInvM )
{
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  const double *d_invMArray =invMArray.Read();
  auto d_posDofInvM = posDofInvM.Read();
  
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
    }
  });
}
#endif

