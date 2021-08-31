#include "gradNonLinearForm.hpp"
#include "riemann_solver.hpp"

GradNonLinearForm::GradNonLinearForm(ParFiniteElementSpace *_vfes,
                                     IntegrationRules *_intRules,
                                     const int _dim,
                                     const int _num_equation,
                                     const volumeFaceIntegrationArrays &_gpuArrays,
                                     const int &_maxIntPoints,
                                     const int &_maxDofs ):
ParNonlinearForm(_vfes),
vfes(_vfes),
intRules(_intRules),
dim(_dim),
num_equation(_num_equation),
gpuArrays(_gpuArrays),
maxIntPoints(_maxIntPoints),
maxDofs(_maxDofs)
{
}

void GradNonLinearForm::Mult(const ParGridFunction *Up, Vector& y )
{
  Vector x;
  
#ifdef _GPU_
//   TestNonLinearForm::setToZero_gpu(y,y.Size()); already done in gradient.cpd
  x.UseDevice(true);
  if(fnfi.Size())
  {
    x.NewMemoryAndSize(Up->GetMemory(), 
                      vfes->GetNDofs()*num_equation, 
                      false);
    Mult_gpu( x, y);
  }
#else
  const double *data = Up->GetData();
  x.SetSize(vfes->GetNDofs()*num_equation*dim);
  x = 0.;
  for(int i=0;i<vfes->GetNDofs()*num_equation;i++) x(i) = data[i];
  ParNonlinearForm::Mult(x,y);
#endif
}

#ifdef _GPU_
void GradNonLinearForm::Mult_gpu(const Vector& x, Vector& y )
{
   
  // INTEGRATION SHARED FACES
  const Vector &px = Prolongate(x);

  const int dofs = vfes->GetNDofs();
  
  // Terms over shared interior faces in parallel.
  ParFiniteElementSpace *pfes = ParFESpace();
  ParMesh *pmesh = pfes->GetParMesh();
  FaceElementTransformations *tr;
  const FiniteElement *fe1, *fe2;
  Array<int> vdofs1, vdofs2;
  Vector el_x, el_y;
  el_x.UseDevice(true);
  el_y.UseDevice(true);
  
  //if (bfnfi.Size()==0) y.HostReadWrite();
  X.MakeRef(aux1, 0); // aux1 contains P.x
  X.ExchangeFaceNbrData();
  const int n_shared_faces = pmesh->GetNSharedFaces();
  for (int i=0;i<n_shared_faces;i++)
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

      const int elDof1 = fe1->GetDof();
      const int elDof2 = fe2->GetDof();
      for (int k = 0; k < fnfi.Size(); k++)
      {
        fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
        //y.AddElementVector(vdofs1, el_y.GetData());
        GradNonLinearForm::IndexedAddToGlobalMemory(vdofs1,
                                                    y,
                                                    el_y,
                                                    dofs,
                                                    elDof1,
                                                    num_equation,
                                                    dim );
        
      }
  }
}

void GradNonLinearForm::IndexedAddToGlobalMemory( const Array<int> &vdofs,
                                                  Vector &y,
                                                  const Vector &el_y,
                                                  const int &dof,
                                                  const int &elDof,
                                                  const int &num_equation,
                                                  const int &dim )
{
  double *dy = y.ReadWrite();
  const double* d_ely = el_y.Read();
  auto d_vdofs = vdofs.Read();
  
  MFEM_FORALL(i,elDof,
  {
    const int index = d_vdofs[i];
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++)
      {
        dy[index+eq*dof+d*num_equation*dof] += d_ely[i+eq*elDof+d*num_equation*elDof];
      }
    }
  });
}

#endif // _GPU_ 
