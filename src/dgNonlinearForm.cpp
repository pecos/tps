#include <vector>
#include "dgNonlinearForm.hpp"
#include "riemann_solver.hpp"

DGNonLinearForm::DGNonLinearForm( ParFiniteElementSpace *_vfes,
                                  ParFiniteElementSpace *_gradFes,
                                  ParGridFunction *_gradUp,
                                  BCintegrator *_bcIntegrator,
                                  IntegrationRules *_intRules,
                                  const int _dim,
                                  const int _num_equation,
                                  EquationOfState *_eqState,
                                  const volumeFaceIntegrationArrays &_gpuArrays,
                                  const int &_maxIntPoints,
                                  const int &_maxDofs ):
ParNonlinearForm(_vfes),
vfes(_vfes),
gradFes(_gradFes),
gradUp(_gradUp),
bcIntegrator(_bcIntegrator),
intRules(_intRules),
dim(_dim),
num_equation(_num_equation),
eqState(_eqState),
gpuArrays(_gpuArrays),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
  h_numElems  = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();
}


void DGNonLinearForm::Mult(const Vector& x, Vector& y )
{
  ParNonlinearForm::Mult(x,y);
  // the following is equivalent to the line above
//   NonlinearForm::Mult(x,y);
//   
//   if (fnfi.Size())
//   {
//     // Terms over shared interior faces in parallel.
//     ParFiniteElementSpace *pfes = ParFESpace();
//     ParMesh *pmesh = pfes->GetParMesh();
//     FaceElementTransformations *tr;
//     const FiniteElement *fe1, *fe2;
//     Array<int> vdofs1, vdofs2;
//     Vector el_x, el_y;
//     aux1.HostReadWrite();
//     X.MakeRef(aux1, 0); // aux1 contains P.x
//     X.ExchangeFaceNbrData();
//     const int n_shared_faces = pmesh->GetNSharedFaces();
//     for (int i = 0; i < n_shared_faces; i++)
//     {
//         tr = pmesh->GetSharedFaceTransformations(i, true);
//         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();
// 
//         fe1 = pfes->GetFE(tr->Elem1No);
//         fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);
// 
//         pfes->GetElementVDofs(tr->Elem1No, vdofs1);
//         pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
// 
//         el_x.SetSize(vdofs1.Size() + vdofs2.Size());
//         X.GetSubVector(vdofs1, el_x.GetData());
//         X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());
// 
//         for (int k = 0; k < fnfi.Size(); k++)
//         {
//           fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
//           aux2.AddElementVector(vdofs1, el_y.GetData());
//         }
//     }
//   }
// 
//   P->MultTranspose(aux2, y);
// 
//   y.HostReadWrite();
//   for (int i = 0; i < ess_tdof_list.Size(); i++)
//   {
//     y(ess_tdof_list[i]) = 0.0;
//   }
}

#ifdef _GPU_
void DGNonLinearForm::Mult_domain(const Vector& x, Vector& y)
{
  setToZero_gpu(y,y.Size());
  
  Mesh *mesh = fes->GetMesh();

  // internal face integration
  if (fnfi.Size())
  {
    // This loop is for internal faces only
    for(int elType=0;elType<gpuArrays.numElems.Size();elType++)
    {
      int elemOffset = 0;
      if( elType!=0 )
      {
        for(int i=0;i<elType;i++) elemOffset += h_numElems[i];
      }
      int dof_el = h_posDofIds[2*elemOffset +1];
      
      faceIntegration_gpu(x, //px,
                          y, //py,
                          gradUp,
                          vfes->GetNDofs(),
                          mesh->GetNumFaces(),
                          h_numElems[elType],
                          elemOffset,
                          dof_el,
                          dim,
                          num_equation,
                          eqState->GetSpecificHeatRatio(),
                          eqState->GetGasConstant(),
                          eqState->GetViscMultiplyer(),
                          eqState->GetBulkViscMultiplyer(),
                          eqState->GetPrandtlNum(),
                          gpuArrays,
                          maxIntPoints,
                          maxDofs );
    }
  }
  
  // INTEGRATION BOUNDARIES
  if(bfnfi.Size()) bcIntegrator->integrateBCs(y,x,gpuArrays.nodesIDs,gpuArrays.posDofIds);
}

void DGNonLinearForm::Mult_bdr(const Vector& x, Vector& y)
{
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if( Nshared>0 )
  {
    sharedFaceIntegration_gpu(x,
                              transferU->face_nbr_data,
                              gradUp,
                              transferGradUp->face_nbr_data,
                              y, 
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
}



void DGNonLinearForm::setToZero_gpu(Vector &x, const int size)
{
  double *d_x = x.Write();
  
  MFEM_FORALL(i,size,
  {
    d_x[i] = 0.;
  });
}



void DGNonLinearForm::faceIntegration_gpu(const Vector &x,
                                          Vector &y,
                                          const ParGridFunction *gradUp,
                                          const int &Ndofs,
                                          const int &Nf,
                                          const int &NumElemType,
                                          const int &elemOffset,
                                          const int &elDof,
                                          const int &dim,
                                          const int &num_equation,
                                          const double &gamma,
                                          const double &Rg,
                                          const double &viscMult,
                                          const double &bulkViscMult,
                                          const double &Pr,
                                          const volumeFaceIntegrationArrays &gpuArrays,
                                          const int &maxIntPoints,
                                          const int &maxDofs)
{
  const double *d_x = x.Read();
  double *d_y = y.Write();
  const double *d_gradUp = gradUp->Read();
  auto d_elemFaces       = gpuArrays.elemFaces.Read();
  auto d_nodesIDs        = gpuArrays.nodesIDs.Read();
  auto d_posDofIds       = gpuArrays.posDofIds.Read();
  auto d_shapeWnor1      = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q        = gpuArrays.elems12Q.Read();
  
  MFEM_FORALL_2D(el,NumElemType,elDof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,elDof)
    {
      MFEM_SHARED double Ui[216*5], /*Uj[216*5],*/ Fcontrib[216*5];
      MFEM_SHARED double gradUpi[216*5*3]/*,gradUpj[216*5*3]*/;
      MFEM_SHARED double l1[216],l2[216];
      MFEM_SHARED double Rflux[5], u1[5], u2[5], nor[3];
      MFEM_SHARED double vFlux1[5*3],vFlux2[5*3];
      MFEM_SHARED double gradUp1[5*3],gradUp2[5*3];
      MFEM_SHARED double weight;
      MFEM_SHARED int offsetEl1,elFaces,elj;
      MFEM_SHARED int gFace,Q,offsetShape1,offsetShape2;
      MFEM_SHARED int offsetElj,dofj,dof1,dof2;
      MFEM_SHARED bool swapElems;
      
      const int eli = elemOffset + el;
      if(i==0) offsetEl1 = d_posDofIds[2*eli];
      if(i==1) elFaces = d_elemFaces[7*eli];
      MFEM_SYNC_THREAD;
      
      const int indexi = d_nodesIDs[offsetEl1+i];
      int index; // variable for auxiliary index
      
      for(int eq=0;eq<num_equation;eq++)
      {
        Ui[i + eq*elDof] = d_x[indexi + eq*Ndofs];
        Fcontrib[i + eq*elDof] = 0.;
        for(int d=0;d<dim;d++) gradUpi[i+eq*elDof+d*num_equation*elDof] = 
                    d_gradUp[indexi + eq*Ndofs+d*num_equation*Ndofs ];
      }
      MFEM_SYNC_THREAD;
      
      // loop over faces
      for(int face=0;face<elFaces;face++)
      {
        if(i==0) gFace = d_elemFaces[7*eli+face+1];
        MFEM_SYNC_THREAD;
        if(i==0) Q     = d_elems12Q[3*gFace+2];
        if(i==1) offsetShape1 = gFace*maxIntPoints*(maxDofs+1+dim);
        if(i==1) offsetShape2 = gFace*maxIntPoints*maxDofs;
        if(i==1) swapElems = false;
        
        // get neighbor
        if(i==2) elj = d_elems12Q[3*gFace];
        MFEM_SYNC_THREAD;
        
        if( elj==eli )
        {
          if(i==0) elj = d_elems12Q[3*gFace+1];
        }else
        {
          swapElems = true;
        }
        MFEM_SYNC_THREAD;
        
        if(i==0) offsetElj = d_posDofIds[2*elj];
        if(i==1) dofj = d_posDofIds[2*elj+1];
        MFEM_SYNC_THREAD;
        
        dof1 = elDof;
        dof2 = dofj;
        if( swapElems )
        {
          dof1 = dofj;
          dof2 = elDof;
        }
        
        // get data from neightbor
//         for(int eq=0;eq<num_equation;eq++)
//         {
//           for(int j=i;j<dofj;j+=elDof)
//           {
//             int index = d_nodesIDs[offsetElj+j];
//             Uj[j + eq*dofj] = d_x[index + eq*Ndofs];
//             for(int d=0;d<dim;d++) gradUpj[j+eq*dofj+d*num_equation*dofj] =
//                            d_gradUp[index+eq*Ndofs+d*num_equation*Ndofs];
//           }
//         }
        MFEM_SYNC_THREAD;
        
        // loop over integration points
        for(int k=0;k<Q;k++)
        {
          if(i==0) weight = d_shapeWnor1[offsetShape1+maxDofs +k*(maxDofs+1+dim)];
          
          for(int eq=i;eq<num_equation;eq+=elDof)
          {
            u1[eq] = 0.; u2[eq] = 0.; Rflux[eq] = 0.;
            for(int d=0;d<dim;d++) gradUp1[eq +d*num_equation] = 0.;
            for(int d=0;d<dim;d++) gradUp2[eq +d*num_equation] = 0.;
            if( eq<dim ) nor[eq] = d_shapeWnor1[offsetShape1+maxDofs+1+eq+k*(maxDofs+1+dim)];
          }
          
          for(int j=i;j<dof1;j+=elDof) l1[j] = d_shapeWnor1[offsetShape1+j+k*(maxDofs+1+dim)];
          for(int j=i;j<dof2;j+=elDof) l2[j] = d_shape2[offsetShape2+j+k*maxDofs];
          MFEM_SYNC_THREAD;
            
          for(int eq=i;eq<num_equation;eq+=elDof)
          {
            if( swapElems )
            {
              for(int j=0;j<dof1;j++)
              {
//                 u1[eq] += l1[j]*Uj[j+eq*dof1];
//                 for(int d=0;d<dim;d++) gradUp1[eq+d*num_equation] += 
//                        l1[j]*gradUpj[j+eq*dof1+d*num_equation*dof1];
                index = d_nodesIDs[offsetElj+j];
                u1[eq] += l1[j]*d_x[index + eq*Ndofs];
                for(int d=0;d<dim;d++) gradUp1[eq+d*num_equation] += 
                       l1[j]*d_gradUp[index+eq*Ndofs+d*num_equation*Ndofs];
              }
              for(int j=0;j<dof2;j++)
              {
                u2[eq] += l2[j]*Ui[j+eq*dof2];
                for(int d=0;d<dim;d++) gradUp2[eq+d*num_equation] += 
                      l2[j]*gradUpi[j+eq*dof2+d*num_equation*dof2];
              }
            }else
            {
              for(int j=0;j<dof1;j++)
              {
                u1[eq] += l1[j]*Ui[j+eq*dof1];
                for(int d=0;d<dim;d++) gradUp1[eq+d*num_equation] += 
                       l1[j]*gradUpi[j+eq*dof1+d*num_equation*dof1];
              }
              for(int j=0;j<dof2;j++)
              {
//                 u2[eq] += l2[j]*Uj[j+eq*dof2];
//                 for(int d=0;d<dim;d++) gradUp2[eq+d*num_equation] += 
//                       l2[j]*gradUpj[j+eq*dof2+d*num_equation*dof2];
                index = d_nodesIDs[offsetElj+j];
                u2[eq] += l2[j]*d_x[index + eq*Ndofs];
                for(int d=0;d<dim;d++) gradUp2[eq+d*num_equation] += 
                      l2[j]*d_gradUp[index+eq*Ndofs+d*num_equation*Ndofs];
              }
            }
          }
          MFEM_SYNC_THREAD;
          
          if( i==elDof-1 ) // NOTE: only one thread does this
          {
            // compute Riemann flux
            RiemannSolver::riemannLF_gpu( &u1[0], 
                                          &u2[0], 
                                          &Rflux[0],
                                          &nor[0],
                                          gamma,
                                          Rg,
                                          dim, 
                                          num_equation );
          }
          Fluxes::viscousFlux_gpu(&vFlux1[0],
                                  &u1[0],
                                  &gradUp1[0],
                                  gamma,
                                  Rg,
                                  viscMult,
                                  bulkViscMult,
                                  Pr,
                                  i,
                                  elDof,
                                  dim,
                                  num_equation );
          Fluxes::viscousFlux_gpu(&vFlux2[0],
                                  &u2[0],
                                  &gradUp2[0],
                                  gamma,
                                  Rg,
                                  viscMult,
                                  bulkViscMult,
                                  Pr,
                                  i,
                                  elDof,
                                  dim,
                                  num_equation );
          MFEM_SYNC_THREAD;
          //if(i<num_equation)
          for(int eq=i;eq<num_equation;eq+=elDof)
          {
            for(int d=0;d<dim;d++) vFlux1[eq+d*num_equation] = 
              0.5*(vFlux1[eq+d*num_equation]+vFlux2[eq+d*num_equation]);
          }
          MFEM_SYNC_THREAD;
          for(int eq=i;eq<num_equation;eq+=elDof)
          {
            for(int d=0;d<dim;d++) Rflux[eq] -= 
                                  vFlux1[eq+d*num_equation]*nor[d];
          }
          MFEM_SYNC_THREAD;
          
          // add integration point contribution
          for(int eq=0;eq<num_equation;eq++)
          {
            if( swapElems ) Fcontrib[i+eq*elDof] += weight*l2[i]*Rflux[eq];
            else Fcontrib[i+eq*elDof] -= weight*l1[i]*Rflux[eq];
          }
        }
      }
      
      // write to global memory
      for(int eq=0;eq<num_equation;eq++)
      {
        d_y[indexi + eq*Ndofs] += Fcontrib[i + eq*elDof];
      }
    }
  });
}


void DGNonLinearForm::sharedFaceIntegration_gpu(const Vector& x,
                                                const Vector &faceU,
                                                const ParGridFunction *gradUp,
                                                const Vector &faceGradUp,
                                                Vector& y, 
                                                const int &Ndofs, 
                                                const int &dim, 
                                                const int &num_equation, 
                                                const double &gamma, 
                                                const double &Rg,
                                                const double &viscMult,
                                                const double &bulkViscMult,
                                                const double &Pr,
                                                const volumeFaceIntegrationArrays &gpuArrays,
//                                                 const Array<int>& nodesIDs, 
//                                                 const Array<int>& posDofIds, 
                                                const parallelFacesIntegrationArrays *parallelData,
                                                const int& maxIntPoints, 
                                                const int& maxDofs)
{
  const double *d_x = x.Read();
  const double *d_gradUp = gradUp->Read();
  const double *d_faceGradUp = faceGradUp.Read();
  const double *d_faceData = faceU.Read();
  double *d_y = y.ReadWrite();
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();
  
  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const double *d_sharedShape2 = parallelData->sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = parallelData->sharedVdofs.Read();
  const int *d_sharedVdofsGrads = parallelData->sharedVdofsGradUp.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();
  
  MFEM_FORALL_2D(el,parallelData->sharedElemsFaces.Size()/7,maxDofs,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,maxDofs)
    {
      MFEM_SHARED double Ui[216*5], /*Uj[64*5],*/ Fcontrib[216*5];
      MFEM_SHARED double gradUpi[216*5*3]/*, gradUpj[64*5*3]*/;
      MFEM_SHARED double l1[216],l2[216];
      MFEM_SHARED double Rflux[5], u1[5], u2[5], nor[3];
      MFEM_SHARED double gradUp1[5*3],gradUp2[5*3];
      MFEM_SHARED double vFlux1[5*3],vFlux2[5*3];
      MFEM_SHARED double weight;
      MFEM_SHARED int el1,numFaces,dof1;
      MFEM_SHARED int f,dof2,Q,offsetEl1;
      MFEM_SHARED bool elemDataRecovered;
      
      if(i==0) el1      = d_sharedElemsFaces[0+el*7];
      if(i==1) numFaces = d_sharedElemsFaces[1+el*7];
      if(i==2) dof1 = d_sharedElem1Dof12Q[1+d_sharedElemsFaces[2+el*7]*4];
      MFEM_SYNC_THREAD;
      
      elemDataRecovered = false;
      int indexi;
      
      for(int elFace=0;elFace<numFaces;elFace++)
      {
        if(i==0) f = d_sharedElemsFaces[1+elFace+1+el*7];
        if(i==1) offsetEl1 = d_posDofIds[2*el1];
        MFEM_SYNC_THREAD;

        if(i==0) dof2 = d_sharedElem1Dof12Q[2+f*4];
        if(i==1) Q    = d_sharedElem1Dof12Q[3+f*4];
        MFEM_SYNC_THREAD;
        
        if( i<dof1 && !elemDataRecovered )
        {
          indexi = d_nodesIDs[offsetEl1+i];
          for(int eq=0;eq<num_equation;eq++)
          {
            Fcontrib[i+eq*dof1] = 0.;
            Ui[i+eq*dof1] = d_x[indexi+eq*Ndofs];
            for(int d=0;d<dim;d++) gradUpi[i+eq*dof1+d*num_equation*dof1] =
                      d_gradUp[indexi + eq*Ndofs+d*num_equation*Ndofs ];
          }
          elemDataRecovered = true;
        }
        MFEM_SYNC_THREAD;
        
        for(int k=0;k<Q;k++)
        {
          if(i==maxDofs-1) weight = 
            d_sharedShapeWnor1[maxDofs+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(i<dof1) l1[i] = d_sharedShapeWnor1[i+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(i<dim ) nor[i]= 
            d_sharedShapeWnor1[maxDofs+1+i+k*(maxDofs+1+dim)+f*maxIntPoints*(maxDofs+1+dim)];
          if(dim==2 && i==maxDofs-1) nor[2] = 0.; 
          if(i<dof2) l2[i] = d_sharedShape2[i+k*maxDofs+f*maxIntPoints*maxDofs];
          if(i<num_equation)
          {
            u1[i] = 0.;
            u2[i] = 0.;
            Rflux[i] = 0.;
            for(int d=0;d<dim;d++)
            {
              gradUp1[i+d*num_equation] = 0.;
              gradUp2[i+d*num_equation] = 0.;
            }
          }
          MFEM_SYNC_THREAD;
          
          // interpolate
          if(i<num_equation)
          {
            for(int n=0;n<dof1;n++)
            {
              u1[i] += Ui[n+i*dof1]*l1[n];
              for(int d=0;d<dim;d++) gradUp1[i+d*num_equation] +=
                            gradUpi[n+i*dof1+d*num_equation*dof1]*l1[n];
            }
            for(int n=0;n<dof2;n++)
            {
  //             u2[i] += Uj[n+i*dof2]*l2[n];
  //             for(int d=0;d<dim;d++) gradUp2[i+d*num_equation] +=
  //                           gradUpj[n+i*dof2+d*num_equation*dof2]*l2[n];
              int index = d_sharedVdofs[n+i*maxDofs+f*num_equation*maxDofs];
              u2[i] += l2[n]* d_faceData[index];
              for(int d=0;d<dim;d++) 
              {
                index = d_sharedVdofsGrads[n+i*maxDofs+d*num_equation*maxDofs+
                                          f*dim*num_equation*maxDofs];
                gradUp2[i+d*num_equation] +=
                            l2[n]*d_faceGradUp[index];
              }
            }
          }
          MFEM_SYNC_THREAD;
          // compute Riemann flux
          if(i==0) RiemannSolver::riemannLF_gpu(&u1[0], 
                                                &u2[0], 
                                                &Rflux[0],
                                                &nor[0],
                                                gamma,
                                                Rg,
                                                dim, 
                                                num_equation );
          Fluxes::viscousFlux_gpu(&vFlux1[0],
                                    &u1[0],
                                    &gradUp1[0],
                                    gamma,
                                    Rg,
                                    viscMult,
                                    bulkViscMult,
                                    Pr,
                                    i,
                                    maxDofs,
                                    dim,
                                    num_equation );
            Fluxes::viscousFlux_gpu(&vFlux2[0],
                                    &u2[0],
                                    &gradUp2[0],
                                    gamma,
                                    Rg,
                                    viscMult,
                                    bulkViscMult,
                                    Pr,
                                    i,
                                    maxDofs,
                                    dim,
                                    num_equation );
          MFEM_SYNC_THREAD;
          if(i<num_equation)
          {
            for(int d=0;d<dim;d++) vFlux1[i+d*num_equation] = 
              0.5*(vFlux1[i+d*num_equation]+vFlux2[i+d*num_equation]);
          }
          MFEM_SYNC_THREAD;
          if(i<num_equation) for(int d=0;d<dim;d++) Rflux[i] -= 
                                        vFlux1[i+d*num_equation]*nor[d];
          MFEM_SYNC_THREAD;
          
          // add integration point contribution
          for(int eq=0;eq<num_equation;eq++)
          {
            if(i<dof1) Fcontrib[i+eq*dof1] -= weight*l1[i]*Rflux[eq];
          }
        }
        MFEM_SYNC_THREAD;
        
      } 
      
      // write to global memory
      for(int eq=0;eq<num_equation;eq++)
      {
        if(i<dof1) d_y[indexi + eq*Ndofs] += Fcontrib[i + eq*dof1];
      }
    }
  });
}

#endif // _GPU_ 
