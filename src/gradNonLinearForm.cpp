// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-
#include "gradNonLinearForm.hpp"

#include "riemann_solver.hpp"
//#include "doftrans.hpp"

GradNonLinearForm::GradNonLinearForm(ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, const int _dim,
                                     const int _num_equation)
    : ParNonlinearForm(_vfes), vfes(_vfes), intRules(_intRules), dim(_dim), num_equation(_num_equation) {}

void GradNonLinearForm::Mult(const ParGridFunction *Up, Vector &y) {
  Vector x;
  const double *data = Up->GetData();

  // NB: Setting x here accounts for the fact that the space Up lives
  // in is different from the space of the gradient.
  x.SetSize(vfes->GetNDofs() * num_equation * dim);
  x = 0.;
  for (int i = 0; i < vfes->GetNDofs() * num_equation; i++) x(i) = data[i];

  // initialize boundaryU counter for each bounary type
  GradFaceIntegrator *gfi = dynamic_cast<GradFaceIntegrator *>(bfnfi[0]);
  gfi->initNbrInlet();
  gfi->initNbrOutlet();
  gfi->initNbrWall();
  
  // After setting x as above, base class Mult does the right thing
  ParNonlinearForm::Mult(x, y);

  /*
  // need to take over base class Mult for correct calcualtion
  // of DG gradients. Modified from fem/pnonlinearform.cpp:Mult
  // ParNonlinearForm::Mult(x, y) calls NonlinearForm::Mult(x, y),
  // so both are here

   // NonlinearForm::Mult...................................................
   const Vector &px = Prolongate(x);
   if (P) { aux2.SetSize(P->Height()); }

   // If we are in parallel, ParNonLinearForm::Mult uses the aux2 vector. In
   // serial, place the result directly in y (when there is no P).
   Vector &py = P ? aux2 : y;

   if (ext)
   {
      ext->Mult(px, py);
      if (Serial())
      {
         if (cP) { cP->MultTranspose(py, y); }
         const int N = ess_tdof_list.Size();
         const auto tdof = ess_tdof_list.Read();
         auto Y = y.ReadWrite();
         MFEM_FORALL(i, N, Y[tdof[i]] = 0.0; );
      }
      // In parallel, the result is in 'py' which is an alias for 'aux2'.
      return;
   }

   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   //DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();

   py = 0.0;

   if (dnfi.Size())
   {
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         //doftrans = fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         //if (doftrans) {doftrans->InvTransformPrimal(el_x); }
         for (int k = 0; k < dnfi.Size(); k++)
         {
            dnfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
            //if (doftrans) {doftrans->TransformDual(el_y); }
            py.AddElementVector(vdofs, el_y);
         }
      }
   }

   if (fnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < fnfi.Size(); k++)
            {
               fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (bfnfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      // incrementers for each boundary type
      //Nbdr_inlet = 0;
      //Nbdr_outlet = 0;
      //Nbdr_wall = 0;
      //setNumInlet(0);
      //setNumOutlet(0);
      //setNumWall(0);
      //int Nbdr_inlet = 0;
      //int Nbdr_outlet = 0;
      //int Nbdr_wall = 0;

      //bfnfi.initNbrInlet();
      //bfnfi.initNbrOutlet();
      //bfnfi.initNbrWall();
      

      // physical boundary loop <warp>	 
      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }
	 
         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < bfnfi.Size(); k++)  // remove loop, add assert that size is 1
            {
               if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

	       // IntegrationRule loop is in AsembleFaceVector
               bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (Serial())
   {
      if (cP) { cP->MultTranspose(py, y); }

      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         y(ess_tdof_list[i]) = 0.0;
      }
      // y(ess_tdof_list[i]) = x(ess_tdof_list[i]);
   }
   // In parallel, the result is in 'py' which is an alias for 'aux2'.

  
   // ParNonlinearForm::Mult................................................... 
   if (fnfi.Size())
   {
      MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
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

   const int N = ess_tdof_list.Size();
   const auto idx = ess_tdof_list.Read();
   auto Y_RW = y.ReadWrite();
   MFEM_FORALL(i, N, Y_RW[idx[i]] = 0.0; );  
  */
  
}
