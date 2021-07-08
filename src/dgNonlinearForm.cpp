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
                                  Array<int> &_numElems,
                                  Array<int> &_nodesIDs,
                                  Array<int> &_posDofIds,
                                  Vector &_shapeWnor1,
                                  Vector &_shape2,
                                  Array<int> &_elemFaces,
                                  Array<int> &_elems12Q,
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
numElems(_numElems),
nodesIDs(_nodesIDs),
posDofIds(_posDofIds),
shapeWnor1(_shapeWnor1),
shape2(_shape2),
elemFaces(_elemFaces),
elems12Q(_elems12Q),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
  h_numElems = numElems.HostRead();
  h_posDofIds = posDofIds.HostRead();
  
  sharedDataFilled = false;
  
  sharedShapeWnor1.UseDevice(true);
  sharedShape2.UseDevice(true);
  //sharedElem1Dof12Q.SetSize(1);
  //sharedVdofs.SetSize(1);
  
//   sharedShapeWnor1 = 0.;
//   sharedShape2 = 0.;
//   sharedElem1Dof12Q = 0;
//   sharedVdofs = 0;
  
  fillSharedData();
}

void DGNonLinearForm::fillSharedData()
{
  ParFiniteElementSpace *pfes = ParFESpace();
  ParMesh *mesh = pfes->GetParMesh();
  mesh->ExchangeFaceNbrNodes();
  mesh->ExchangeFaceNbrData();
  vfes->ExchangeFaceNbrData();
  gradFes->ExchangeFaceNbrData();
  
  
  const int Nshared = mesh->GetNSharedFaces();
  if( Nshared>0 )
  {
    face_nbr_data.UseDevice(true);
    send_data.UseDevice(true);
    face_nbr_data.SetSize(vfes->GetFaceNbrVSize());
    send_data.SetSize(vfes->send_face_nbr_ldof.Size_of_connections());
    face_nbr_data = 0.;
    send_data = 0.;
    face_nbr_dataGrad.UseDevice(true);
    send_dataGrad.UseDevice(true);
    face_nbr_dataGrad.SetSize(gradFes->GetFaceNbrVSize());
    send_dataGrad.SetSize(gradFes->send_face_nbr_ldof.Size_of_connections());
    face_nbr_dataGrad = 0.;
    send_dataGrad = 0.;
    
    sharedShapeWnor1.SetSize(Nshared*maxIntPoints*(maxDofs+1+dim));
    sharedShape2.SetSize(Nshared*maxIntPoints*maxDofs);
    sharedElem1Dof12Q.SetSize(Nshared*4);
    sharedVdofs.SetSize(Nshared*num_equation*maxDofs);
    sharedVdofsGradUp.SetSize(Nshared*num_equation*maxDofs*dim);
    
    sharedShapeWnor1 = 0.;
    sharedShape2 = 0.;
    sharedElem1Dof12Q = 0;
    sharedVdofs = 0;
    sharedVdofsGradUp = 0;
    
    auto hsharedShapeWnor1 = sharedShapeWnor1.HostReadWrite();
    auto hsharedShape2 = sharedShape2.HostReadWrite();
    auto hsharedElem1Dof12Q = sharedElem1Dof12Q.HostReadWrite();
    auto hsharedVdofs = sharedVdofs.HostReadWrite();
    auto hsharedVdofsGrads = sharedVdofsGradUp.HostReadWrite();
    
    std::vector<int> unicElems; unicElems.clear();
    
    Array<int> vdofs2, vdofsGrad;
    FaceElementTransformations *tr;
    for (int i=0;i<Nshared;i++)
    {
        tr = mesh->GetSharedFaceTransformations(i, true);
        int Elem2NbrNo = tr->Elem2No - mesh->GetNE();

        const FiniteElement *fe1 = vfes->GetFE(tr->Elem1No);
        const FiniteElement *fe2 = vfes->GetFaceNbrFE(Elem2NbrNo);
        const int dof1 = fe1->GetDof();
        const int dof2 = fe2->GetDof();

        //vfes->GetElementVDofs(tr->Elem1No, vdofs1); // get these from nodesIDs
        vfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
        gradFes->GetFaceNbrElementVDofs(Elem2NbrNo,vdofsGrad);
        
        for(int n=0;n<dof2;n++)
        {
          for(int eq=0;eq<num_equation;eq++) 
          {
            hsharedVdofs[n+eq*maxDofs+i*num_equation*maxDofs] = 
                                                    vdofs2[n+eq*dof2];
            for(int d=0;d<dim;d++)
            {
              int index = n+eq*maxDofs+d*num_equation*maxDofs+
                          i*dim*num_equation*maxDofs;
              hsharedVdofsGrads[index] = 
                    vdofsGrad[n+eq*dof2+d*num_equation*dof2];
            }
          }
        }
        
        int intorder;
        if (tr->Elem2No >= 0)
            intorder = (min(tr->Elem1->OrderW(), tr->Elem2->OrderW()) +
                        2*max(fe1->GetOrder(), fe2->GetOrder()));
        else
        {
            intorder = tr->Elem1->OrderW() + 2*fe1->GetOrder();
        }
        if (fe1->Space() == FunctionSpace::Pk)
        {
            intorder++;
        }
        //IntegrationRules IntRules2(0, Quadrature1D::GaussLobatto);
        const IntegrationRule *ir = &intRules->Get(tr->GetGeometryType(), intorder);

        hsharedElem1Dof12Q[0 +i*4] = tr->Elem1No;
        hsharedElem1Dof12Q[1 +i*4] = dof1;
        hsharedElem1Dof12Q[2 +i*4] = dof2;
        hsharedElem1Dof12Q[3 +i*4] = ir->GetNPoints();
        
        bool inList = false;
        for(int n=0;n<unicElems.size();n++)
        {
          if( unicElems[n] == tr->Elem1No ) inList = true;
        }
        if(!inList) unicElems.push_back( tr->Elem1No );
        
        Vector shape1, shape2, nor;
        shape1.UseDevice(false); shape2.UseDevice(false); nor.UseDevice(false);
        shape1.SetSize(dof1);
        shape2.SetSize(dof2);
        nor.SetSize(dim);
        
        for(int q=0;q<ir->GetNPoints();q++)
        {
          const IntegrationPoint &ip = ir->IntPoint(q);
          tr->SetAllIntPoints(&ip);
          
          fe1->CalcShape(tr->GetElement1IntPoint(), shape1);
          fe2->CalcShape(tr->GetElement2IntPoint(), shape2);
          CalcOrtho(tr->Jacobian(), nor);
          
          for(int n=0;n<dof1;n++)
          {
            hsharedShapeWnor1[n+q*(maxDofs+1+dim)+i*maxIntPoints*(maxDofs+1+dim)] = shape1[n];
          }
          hsharedShapeWnor1[maxDofs+q*(maxDofs+1+dim)+i*maxIntPoints*(maxDofs+1+dim)] = ip.weight;
          
          for(int d=0;d<dim;d++) hsharedShapeWnor1[maxDofs+1+d+
                                        q*(maxDofs+1+dim)+i*maxIntPoints*(maxDofs+1+dim)] = nor[d];
          for(int n=0;n<dof2;n++)
          {
            hsharedShape2[n+q*maxDofs+i*maxIntPoints*maxDofs] = shape2[n];
          }
        }
    }
    
    sharedElemsFaces.SetSize(7*unicElems.size());
    sharedElemsFaces = -1;
    auto hsharedElemsFaces = sharedElemsFaces.HostWrite();
    for(int el=0;el<unicElems.size();el++)
    {
      const int eli = unicElems[el];
      for(int f=0;f<sharedElem1Dof12Q.Size()/4;f++)
      {
        if( eli==hsharedElem1Dof12Q[0 +f*4] )
        {
          hsharedElemsFaces[0+7*el] = hsharedElem1Dof12Q[0 +f*4];
          int numFace = hsharedElemsFaces[1+7*el];
          if( numFace==-1 ) numFace = 0;
          numFace++;
          hsharedElemsFaces[1+numFace+ 7*el] = f;
          hsharedElemsFaces[1+7*el] = numFace;
        }
      }
    }
  }else
  {
    sharedShapeWnor1.SetSize(1);
    sharedShape2.SetSize(1);
    sharedElem1Dof12Q.SetSize(1);
    sharedVdofs.SetSize(1);
    sharedVdofsGradUp.SetSize(1);
    sharedElemsFaces.SetSize(1);
  }
  
#ifdef _GPU_
  auto dsharedShapeWnor1 = sharedShapeWnor1.ReadWrite();
  auto dsharedShape2 = sharedShape2.ReadWrite();
  auto dsharedElemDof12Q = sharedElem1Dof12Q.ReadWrite();
  auto dsharedVdofs = sharedVdofs.ReadWrite();
  auto dsharedVdofsGradUp = sharedVdofsGradUp.ReadWrite();
  auto dsharedElemsFaces = sharedElemsFaces.ReadWrite();
#endif
  
  sharedDataFilled = true;
}


void DGNonLinearForm::exchangeBdrData(const Vector &x,
                                      ParFiniteElementSpace *pfes,
                                      Vector &face_nbr_data,
                                      Vector &send_data )
{
  if (pfes->GetFaceNbrVSize() <= 0)
   {
      return;
   }

   ParMesh *pmesh = pfes->GetParMesh();

//    face_nbr_data.SetSize(pfes->GetFaceNbrVSize());
//    send_data.SetSize(pfes->send_face_nbr_ldof.Size_of_connections());

   int *send_offset = pfes->send_face_nbr_ldof.GetI();
   const int *d_send_ldof = mfem::Read(pfes->send_face_nbr_ldof.GetJMemory(),
                                       send_data.Size());
   int *recv_offset = pfes->face_nbr_ldof.GetI();
   MPI_Comm MyComm = pfes->GetComm();

   int num_face_nbrs = pmesh->GetNFaceNeighbors();
   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

//    auto d_data = this->Read();
   auto d_data = x.Read();
   auto d_send_data = send_data.Write();
   MFEM_FORALL(i, send_data.Size(),
   {
      const int ldof = d_send_ldof[i];
      d_send_data[i] = d_data[ldof >= 0 ? ldof : -1-ldof];
   });

   bool mpi_gpu_aware = Device::GetGPUAwareMPI();
//    auto send_data_ptr = mpi_gpu_aware ? send_data.Read() : send_data.HostRead();
//    auto face_nbr_data_ptr = mpi_gpu_aware ? face_nbr_data.Write() :
//                             face_nbr_data.HostWrite();
   auto send_data_ptr = send_data.HostRead();
   auto face_nbr_data_ptr = face_nbr_data.HostWrite();
   
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&send_data_ptr[send_offset[fn]],
                send_offset[fn+1] - send_offset[fn],
                MPI_DOUBLE, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(&face_nbr_data_ptr[recv_offset[fn]],
                recv_offset[fn+1] - recv_offset[fn],
                MPI_DOUBLE, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   
   //auto kk = face_nbr_data.Read(); // offload to GPU

   delete [] statuses;
   delete [] requests;
}



void DGNonLinearForm::Mult(const Vector& x, Vector& y )
{
#ifdef _GPU_
  Mult_gpu( x, y);
#else
  // ParNonlinearForm::Mult(x,y);
  // the following is equivalent to the line above
  NonlinearForm::Mult(x,y);
  
  if (fnfi.Size())
  {
    // Terms over shared interior faces in parallel.
    ParFiniteElementSpace *pfes = ParFESpace();
    ParMesh *pmesh = pfes->GetParMesh();
    FaceElementTransformations *tr;
    const FiniteElement *fe1, *fe2;
    Array<int> vdofs1, vdofs2;
    Vector el_x, el_y;
    aux1.HostReadWrite();
    X.MakeRef(aux1, 0); // aux1 contains P.x
    X.ExchangeFaceNbrData();
    const int n_shared_faces = pmesh->GetNSharedFaces();
    for (int i = 0; i < n_shared_faces; i++)
    {
        tr = pmesh->GetSharedFaceTransformations(i, true);
        int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

        fe1 = pfes->GetFE(tr->Elem1No);
        fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

        pfes->GetElementVDofs(tr->Elem1No, vdofs1);
        pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

        el_x.SetSize(vdofs1.Size() + vdofs2.Size());
        X.GetSubVector(vdofs1, el_x.GetData());
        X.FaceNbrData().GetSubVector(vdofs2, el_x.GetData() + vdofs1.Size());

        for (int k = 0; k < fnfi.Size(); k++)
        {
          fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
          aux2.AddElementVector(vdofs1, el_y.GetData());
        }
    }
  }

  P->MultTranspose(aux2, y);

  y.HostReadWrite();
  for (int i = 0; i < ess_tdof_list.Size(); i++)
  {
    y(ess_tdof_list[i]) = 0.0;
  }
#endif
}

#ifdef _GPU_
void DGNonLinearForm::Mult_gpu(const Vector& x, Vector& y )
{
  setToZero_gpu(y,y.Size());
  
  Mesh *mesh = fes->GetMesh();

  // internal face integration
  if (fnfi.Size())
  {
    // This loop is for internal faces only
    for(int elType=0;elType<numElems.Size();elType++)
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
                          eqState->GetPrandtlNum(),
                          elemFaces,
                          nodesIDs,
                          posDofIds,
                          shapeWnor1,
                          shape2,
                          maxIntPoints,
                          maxDofs,
                          elems12Q );
    }
  }
  
  // INTEGRATION BOUNDARIES
  if(bfnfi.Size()) bcIntegrator->integrateBCs(y,x,nodesIDs,posDofIds);
   
  // INTEGRATION SHARED FACES
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if( Nshared>0 )
  {
    //if( !sharedDataFilled ) fillSharedData();
//     const Vector &px = Prolongate(x);
//     X.MakeRef(aux1, 0); // aux1 contains P.x
//     X.ExchangeFaceNbrData();
//     Vector &bdrData = X.FaceNbrData();
    exchangeBdrData(x,vfes,face_nbr_data,send_data);
    
    // get shared element gradients
//     gradUp->ExchangeFaceNbrData();
Vector &bdrGradUp = gradUp->FaceNbrData(); // data already exchanged when computing gradients
//     exchangeBdrData(*gradUp,gradFes,face_nbr_dataGrad,send_dataGrad);
    
    sharedFaceIntegration_gpu(x,
                              gradUp,
                              //bdrData,
                              face_nbr_data,
                              bdrGradUp,
//                               face_nbr_dataGrad,
                              y, 
                              vfes->GetNDofs(), 
                              dim, 
                              num_equation, 
                              eqState->GetSpecificHeatRatio(),
                              eqState->GetGasConstant(), 
                              eqState->GetViscMultiplyer(),
                              eqState->GetPrandtlNum(),
                              nodesIDs, 
                              posDofIds, 
                              sharedShapeWnor1, 
                              sharedShape2, 
                              sharedElem1Dof12Q, 
                              sharedVdofs,
                              sharedVdofsGradUp,
                              sharedElemsFaces,
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
                                          const double &Pr,
                                          const Array<int> &elemFaces,
                                          const Array<int> &nodesIDs,
                                          const Array<int> &posDofIds,
                                          const Vector &shapeWnor1,
                                          const Vector &shape2,
                                          const int &maxIntPoints,
                                          const int &maxDofs,
                                          const Array<int> &elems12Q )
{
  const double *d_x = x.Read();
  double *d_y = y.Write();
  const double *d_gradUp = gradUp->Read();
  auto d_elemFaces = elemFaces.Read();
  auto d_nodesIDs = nodesIDs.Read();
  auto d_posDofIds = posDofIds.Read();
  auto d_shapeWnor1 = shapeWnor1.Read();
  const double *d_shape2 = shape2.Read();
  auto d_elems12Q = elems12Q.Read();
  
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
      
      const int eli = elemOffset + el;
      const int offsetEl1 = d_posDofIds[2*eli];
      const int indexi = d_nodesIDs[offsetEl1+i];
      const int elFaces = d_elemFaces[7*eli];
      
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
          const double weight = d_shapeWnor1[offsetShape1+maxDofs +k*(maxDofs+1+dim)];
          
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
                int index = d_nodesIDs[offsetElj+j];
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
                int index = d_nodesIDs[offsetElj+j];
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
                                                const ParGridFunction *gradUp,
                                                const Vector &faceData,
                                                const Vector &faceGradUp,
                                                Vector& y, 
                                                const int &Ndofs, 
                                                const int &dim, 
                                                const int &num_equation, 
                                                const double &gamma, 
                                                const double &Rg,
                                                const double &viscMult,
                                                const double &Pr,
                                                const Array<int>& nodesIDs, 
                                                const Array<int>& posDofIds, 
                                                const Vector& sharedShapeWnor1, 
                                                const Vector& sharedShape2, 
                                                const Array<int>& sharedElem1Dof12Q, 
                                                const Array<int>& sharedVdofs, 
                                                const Array<int> &sharedVdofsGradUp,
                                                const Array<int> &sharedElemsFaces,
                                                const int& maxIntPoints, 
                                                const int& maxDofs)
{
  const double *d_x = x.Read();
  const double *d_gradUp = gradUp->Read();
  const double *d_faceGradUp = faceGradUp.Read();
  const double *d_faceData = faceData.Read();
  double *d_y = y.ReadWrite();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  
  const double *d_sharedShapeWnor1 = sharedShapeWnor1.Read();
  const double *d_sharedShape2 = sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = sharedVdofs.Read();
  const int *d_sharedVdofsGrads = sharedVdofsGradUp.Read();
  const int *d_sharedElemsFaces = sharedElemsFaces.Read();
  
  MFEM_FORALL_2D(el,sharedElemsFaces.Size()/7,maxDofs,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,maxDofs)
    {
      MFEM_SHARED double Ui[216*5], /*Uj[64*5],*/ Fcontrib[216*5];
      MFEM_SHARED double gradUpi[216*5*3]/*, gradUpj[64*5*3]*/;
      MFEM_SHARED double l1[216],l2[216];
      MFEM_SHARED double Rflux[5], u1[5], u2[5], nor[3];
      MFEM_SHARED double gradUp1[5*3],gradUp2[5*3];
      MFEM_SHARED double vFlux1[5*3],vFlux2[5*3];
      
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
          const double weight = 
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
