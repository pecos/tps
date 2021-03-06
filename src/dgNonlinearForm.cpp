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
#include "dgNonlinearForm.hpp"

#include <vector>

#include "riemann_solver.hpp"

DGNonLinearForm::DGNonLinearForm(RiemannSolver *rsolver, Fluxes *_flux, ParFiniteElementSpace *_vfes,
                                 ParFiniteElementSpace *_gradFes, ParGridFunction *_gradUp, BCintegrator *_bcIntegrator,
                                 IntegrationRules *_intRules, const int _dim, const int _num_equation,
                                 GasMixture *_mixture, const volumeFaceIntegrationArrays &_gpuArrays,
                                 const int &_maxIntPoints, const int &_maxDofs)
    : ParNonlinearForm(_vfes),
      rsolver_(rsolver),
      fluxes(_flux),
      vfes(_vfes),
      gradFes(_gradFes),
      gradUp_(_gradUp),
      bcIntegrator(_bcIntegrator),
      intRules(_intRules),
      dim_(_dim),
      num_equation_(_num_equation),
      mixture(_mixture),
      gpuArrays(_gpuArrays),
      maxIntPoints_(_maxIntPoints),
      maxDofs_(_maxDofs) {
  h_numElems = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();

  uk_el1.UseDevice(true);
  uk_el2.UseDevice(true);
  grad_upk_el1.UseDevice(true);
  grad_upk_el2.UseDevice(true);
  face_flux_.UseDevice(true);

  int nfaces = vfes->GetMesh()->GetNumFaces();
  uk_el1.SetSize(nfaces * maxIntPoints_ * num_equation_);
  uk_el2.SetSize(nfaces * maxIntPoints_ * num_equation_);
  grad_upk_el1.SetSize(nfaces * maxIntPoints_ * num_equation_ * dim_);
  grad_upk_el2.SetSize(nfaces * maxIntPoints_ * num_equation_ * dim_);
  uk_el1 = 0.;
  uk_el2 = 0.;
  grad_upk_el1 = 0.;
  grad_upk_el2 = 0.;

  face_flux_.SetSize(nfaces * maxIntPoints_ * num_equation_);
  face_flux_ = 0.;
}

void DGNonLinearForm::setParallelData(parallelFacesIntegrationArrays *_parallelData, dataTransferArrays *_transferU,
                                      dataTransferArrays *_transferGradUp) {
  parallelData = _parallelData;
  transferU = _transferU;
  transferGradUp = _transferGradUp;

  shared_flux.UseDevice(true);

  int maxNumElems = parallelData->sharedElemsFaces.Size() / 7;  // elements with shared faces
  // TODO(kevin): MAXNUMFACE
  shared_flux.SetSize(maxNumElems * 5 * maxIntPoints_ * num_equation_);
}

void DGNonLinearForm::Mult(const Vector &x, Vector &y) {
  ParNonlinearForm::Mult(x, y);
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
void DGNonLinearForm::Mult_domain(const Vector &x, Vector &y) {
  // setToZero_gpu(y, y.Size());  // assume y comes in set to zero (or that we are updating)

  // Internal face integration
  if (fnfi.Size()) {
    // Interpolate state info the faces (loops over elements)
    for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
      int elemOffset = 0;
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
      int dof_el = h_posDofIds[2 * elemOffset + 1];
      interpFaceData_gpu(x, elType, elemOffset, dof_el);
    }

    // Evaluate fluxes at quadrature points (loops over faces)
    evalFaceFlux_gpu();

    // Compute flux contributions to residual (loops over elements)
    for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
      int elemOffset = 0;
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
      int dof_el = h_posDofIds[2 * elemOffset + 1];
      faceIntegration_gpu(y, elType, elemOffset, dof_el);
    }
  }

  // INTEGRATION BOUNDARIES
  if (bfnfi.Size()) bcIntegrator->integrateBCs(y, x, gpuArrays.nodesIDs, gpuArrays.posDofIds);
}

void DGNonLinearForm::Mult_bdr(const Vector &x, Vector &y) {
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if (Nshared > 0) {
    sharedFaceInterpolation_gpu(x);
    sharedFaceIntegration_gpu(y);
  }
}

void DGNonLinearForm::setToZero_gpu(Vector &x, const int size) {
  double *d_x = x.Write();

  MFEM_FORALL(i, size, { d_x[i] = 0.; });
}

void DGNonLinearForm::faceIntegration_gpu(Vector &y, int elType, int elemOffset, int elDof) {
  // double *d_y = y.Write();
  double *d_y = y.ReadWrite();
  const double *d_f = face_flux_.Read();

  auto d_elemFaces = gpuArrays.elemFaces.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  const int Ndofs = vfes->GetNDofs();
  const int NumElemType = h_numElems[elType];
  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  // clang-format off
  MFEM_FORALL_2D(el, NumElemType, elDof, 1, 1,
  {
    MFEM_SHARED double shape[gpudata::MAXDOFS];  // shape[216];
    MFEM_SHARED double Fcontrib[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // Fcontrib[216 * 5];
    double Rflux[gpudata::MAXEQUATIONS];  // double Rflux[5];
    int indexes_i[gpudata::MAXDOFS];  // int indexes_i[216];

    const int eli = elemOffset + el;
    const int offsetEl1 = d_posDofIds[2 * eli];
    const int elFaces = d_elemFaces[7 * eli];

    // elem1 data
    for (int i = 0; i < elDof; i++) {
      int index = d_nodesIDs[offsetEl1 + i];
      indexes_i[i] = index;
      for (int eq = 0; eq < num_equation; eq++) {
        Fcontrib[i + eq * elDof] = 0.;
      }
    }

    for (int face = 0; face < elFaces; face++) {
      const int gFace = d_elemFaces[7 * eli + face + 1];
      const int Q = d_elems12Q[3 * gFace + 2];
      const int offsetShape1 = gFace * maxIntPoints * (maxDofs + 1 + dim);
      const int offsetShape2 = gFace * maxIntPoints * maxDofs;
      bool swapElems = false;

      // get neighbor
      int elj = d_elems12Q[3 * gFace];
      if (elj == eli) {
        elj = d_elems12Q[3 * gFace + 1];
      } else {
        swapElems = true;
      }

      int dof1 = elDof;
      int dof2 = d_posDofIds[2 * elj + 1];
      if (swapElems) {
        dof1 = d_posDofIds[2 * elj + 1];
        dof2 = elDof;
      }

      for (int k = 0; k < Q; k++) {
        // get shapes and normal
        const double weight = d_shapeWnor1[offsetShape1 + maxDofs + k * (maxDofs + 1 + dim)];

        if (swapElems) {
          MFEM_FOREACH_THREAD(j, x, elDof) { shape[j] = d_shape2[offsetShape2 + j + k * maxDofs]; }
        } else {
          // NB: Negaive sign correct b/c we *add* to flux below regardless of swapElems
          MFEM_FOREACH_THREAD(j, x, elDof) { shape[j] = -d_shapeWnor1[offsetShape1 + j + k * (maxDofs + 1 + dim)]; }
        }
        MFEM_SYNC_THREAD;

        for (int eq = 0; eq < num_equation; eq++) {
          Rflux[eq] = d_f[eq + k * num_equation + gFace * maxIntPoints * num_equation];
        }

        MFEM_FOREACH_THREAD(i, x, elDof) {
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * elDof] += weight * shape[i] * Rflux[eq];
          }
        }
        MFEM_SYNC_THREAD;
      }  // end loop over quad pts
    }  // end loop over faces

    MFEM_FOREACH_THREAD(i, x, elDof) {
      const int index = indexes_i[i];
      for (int eq = 0; eq < num_equation; eq++) {
        d_y[index + eq * Ndofs] += Fcontrib[i + eq * elDof];
      }
    }
    MFEM_SYNC_THREAD;
  });
  // clang-format on
}

void DGNonLinearForm::evalFaceFlux_gpu() {
  double *d_f = face_flux_.Write();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();
  const double *d_grad_uk_el1 = grad_upk_el1.Read();
  const double *d_grad_uk_el2 = grad_upk_el2.Read();

  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  // const double Sc = mixture->GetSchmidtNum();

  Mesh *mesh = fes->GetMesh();
  const int Nf = mesh->GetNumFaces();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  const RiemannSolver *d_rsolver = rsolver_;
  Fluxes *d_flux = fluxes;

  // clang-format off
  MFEM_FORALL(iface, Nf,
  {
    double u1[gpudata::MAXEQUATIONS], gradUp1[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double u2[gpudata::MAXEQUATIONS], gradUp2[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double vFlux1[gpudata::MAXEQUATIONS * gpudata::MAXDIM], vFlux2[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double Rflux[gpudata::MAXEQUATIONS], nor[gpudata::MAXDIM];

    const int Q = d_elems12Q[3 * iface + 2];
    const int offsetShape1 = iface * maxIntPoints * (maxDofs + 1 + dim);

    // loop over quad points on this face
    for (int k = 0; k < Q; k++) {
      // get normal vector
      for (int d = 0; d < dim; d++)
        nor[d] = d_shapeWnor1[offsetShape1 + maxDofs + 1 + d + k * (maxDofs + 1 + dim)];

      // get state and gradient
      for (int eq = 0; eq < num_equation; eq++) {
        u1[eq] = d_uk_el1[eq + k * num_equation + iface * maxIntPoints * num_equation];
        u2[eq] = d_uk_el2[eq + k * num_equation + iface * maxIntPoints * num_equation];

        for (int d = 0; d < dim; d++) {
//          gradUp1[eq + d * num_equation] = d_grad_uk_el1[eq + k * num_equation + d * maxIntPoints * num_equation +
//                                                         iface * dim * maxIntPoints * num_equation];
//          gradUp2[eq + d * num_equation] = d_grad_uk_el2[eq + k * num_equation + d * maxIntPoints * num_equation +
//                                                         iface * dim * maxIntPoints * num_equation];
          gradUp1[eq + d * num_equation] = d_grad_uk_el1[eq + d * num_equation + k * dim * num_equation +
                                                         iface * dim * maxIntPoints * num_equation];
          gradUp2[eq + d * num_equation] = d_grad_uk_el2[eq + d * num_equation + k * dim * num_equation +
                                                         iface * dim * maxIntPoints * num_equation];
        }
      }

      // evaluate flux
      // d_rsolver->Eval_LF(u1, u2, nor, Rflux);

      // For some unknown reason, the hip build is very sensitive to
      // how Eval_LF is called.  If we copy the state to u1, u2 and
      // normal to nor and then call as above (which behaves correctly
      // for CUDA), the resulting flux is nonsense.  But, if we just
      // point into d_uk_el1 at the right spots, all is well.  This
      // problem seems to have something to do with virtual functions
      // and pointer arguments, but I don't understand it.  However,
      // this approach is working.
      d_rsolver->Eval_LF(d_uk_el1 + k * num_equation + iface * maxIntPoints * num_equation,
                         d_uk_el2 + k * num_equation + iface * maxIntPoints * num_equation,
                         d_shapeWnor1 + offsetShape1 + maxDofs + 1 + k * (maxDofs + 1 + dim),
                         Rflux);

#if defined(_CUDA_)
      // TODO(kevin): implement radius.
      d_flux->ComputeViscousFluxes(u1, gradUp1, 0.0, vFlux1);
      d_flux->ComputeViscousFluxes(u2, gradUp2, 0.0, vFlux2);
      // d_flux->ComputeViscousFluxes(d_uk_el1 + k * num_equation + iface * maxIntPoints * num_equation,
      //             d_grad_uk_el1 + k * dim * num_equation + iface * maxIntPoints * dim * num_equation,
      //             0.0, vFlux1);
#elif defined(_HIP_)
      Fluxes::viscousFlux_serial_gpu(&vFlux1[0], &u1[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                     num_equation);
      Fluxes::viscousFlux_serial_gpu(&vFlux2[0], &u2[0], &gradUp2[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                     num_equation);
#endif
      for (int d = 0; d < dim; d++) {
        for (int eq = 0; eq < num_equation; eq++) {
          vFlux1[eq + d * num_equation] = 0.5 * (vFlux1[eq + d * num_equation] + vFlux2[eq + d * num_equation]);
        }
      }

      for (int d = 0; d < dim; d++) {
        for (int eq = 0; eq < num_equation; eq++) {
          Rflux[eq] -= vFlux1[eq + d * num_equation] * nor[d];
        }
      }

      // store
      for (int eq = 0; eq < num_equation; eq++) {
        d_f[eq + k * num_equation + iface * maxIntPoints * num_equation] = Rflux[eq];
      }
    }  // end loop over quad pts
  });
  // clang-format on
}

void DGNonLinearForm::interpFaceData_gpu(const Vector &x, int elType, int elemOffset, int elDof) {
  auto d_x = x.Read();
  double *d_uk_el1 = uk_el1.Write();
  double *d_uk_el2 = uk_el2.Write();
  double *d_grad_uk_el1 = grad_upk_el1.Write();
  double *d_grad_uk_el2 = grad_upk_el2.Write();

  const ParGridFunction *gradUp = gradUp_;

  const double *d_gradUp = gradUp->Read();
  auto d_elemFaces = gpuArrays.elemFaces.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  const int Ndofs = vfes->GetNDofs();
  const int NumElemsType = h_numElems[elType];
  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  // clang-format off
  MFEM_FORALL_2D(el, NumElemsType, maxIntPoints, 1, 1,
  {
    // assuming max. num of equations = 20
    // and max elem dof is 216 (a p=5 hex)
    double uk1[gpudata::MAXEQUATIONS], gradUpk1[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double shape[gpudata::MAXDOFS];
    int indexes_i[gpudata::MAXDOFS];

    const int eli = elemOffset + el;
    const int offsetEl1 = d_posDofIds[2 * eli];
    const int elFaces = d_elemFaces[7 * eli];
    const int dof1 = elDof;

    for (int i = 0; i < elDof; i++) {
      int index = d_nodesIDs[offsetEl1 + i];
      indexes_i[i] = index;
    }

    // loop over faces
    for (int face = 0; face < elFaces; face++) {
      const int gFace = d_elemFaces[7 * eli + face + 1];
      const int Q = d_elems12Q[3 * gFace + 2];
      int offsetShape1 = gFace * maxIntPoints * (maxDofs + 1 + dim);
      int offsetShape2 = gFace * maxIntPoints * maxDofs;

      // swapElems = false indicates that el is "element 1" for this face
      // swapElems = true  indicates that el is "element 2" for this face
      bool swapElems = false;
      if (eli != d_elems12Q[3 * gFace]) {
        swapElems = true;
      }

      // loop over quadrature points on this face
      MFEM_FOREACH_THREAD(k, x, Q) {
        // set interpolation data to 0
        for (int n = 0; n < num_equation; n++) {
          uk1[n] = 0.;
          for (int d = 0; d < dim; d++) {
            gradUpk1[n + d * num_equation] = 0.;
          }
        }

        // load shape functions
        if (swapElems) {
          for (int j = 0; j < dof1; j++) shape[j] = d_shape2[offsetShape2 + j + k * maxDofs];
        } else {
          for (int j = 0; j < dof1; j++) shape[j] = d_shapeWnor1[offsetShape1 + j + k * (maxDofs + 1 + dim)];
        }

        for (int eq = 0; eq < num_equation; eq++) {
          int index;

          // interpolate state
          for (int j = 0; j < dof1; j++) {
            index = indexes_i[j];
            uk1[eq] += d_x[index + eq * Ndofs] * shape[j];
          }

          // interpolate gradient
          for (int d = 0; d < dim; d++) {
            for (int j = 0; j < dof1; j++) {
              index = indexes_i[j];
              gradUpk1[eq + d * num_equation] += d_gradUp[index + eq * Ndofs + d * num_equation * Ndofs] * shape[j];
            }
          }
        }

        // save quad pt data to global memory
        if (swapElems) {
          for (int eq = 0; eq < num_equation; eq++) {
            d_uk_el2[eq + k * num_equation + gFace * maxIntPoints * num_equation] = uk1[eq];
            for (int d = 0; d < dim; d++) {
//              d_grad_uk_el2[eq + k * num_equation + d * maxIntPoints * num_equation +
//                            gFace * dim * maxIntPoints * num_equation] = gradUpk1[eq + d * num_equation];
              d_grad_uk_el2[eq + d * num_equation + k * dim * num_equation +
                            gFace * maxIntPoints * dim * num_equation] = gradUpk1[eq + d * num_equation];
            }
          }
        } else {
          for (int eq = 0; eq < num_equation; eq++) {
            d_uk_el1[eq + k * num_equation + gFace * maxIntPoints * num_equation] = uk1[eq];
            for (int d = 0; d < dim; d++) {
//              d_grad_uk_el1[eq + k * num_equation + d * maxIntPoints * num_equation +
//                            gFace * dim * maxIntPoints * num_equation] = gradUpk1[eq + d * num_equation];
              d_grad_uk_el1[eq + d * num_equation + k * dim * num_equation +
                            gFace * dim * maxIntPoints * num_equation] = gradUpk1[eq + d * num_equation];
            }
          }
        }
      }   // end loop over integration points (MFEM_FOREACH_THREAD)
    }  // end loop over faces
  });
  // clang-format on
}

// clang-format off
void DGNonLinearForm::sharedFaceIntegration_gpu(Vector &y) {
  double *d_y = y.ReadWrite();
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg    = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  // const double Sc = mixture->GetSchmidtNum();

  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();

  const double *d_shared_flux = shared_flux.Read();

  const int maxNumElems = parallelData->sharedElemsFaces.Size()/7;  // elements with shared faces
  const int Ndofs = vfes->GetNDofs();
  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  MFEM_FORALL_2D(el, parallelData->sharedElemsFaces.Size() / 7, maxDofs, 1, 1, {
    //
    double Fcontrib[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // double Fcontrib[216 * 5];
    double Rflux[gpudata::MAXEQUATIONS];  // double Rflux[5];
    int index_i[gpudata::MAXDOFS];  // int index_i[216];

    const int el1      = d_sharedElemsFaces[0 + el * 7];
    const int offsetEl1 = d_posDofIds[2 * el1];
    const int numFaces = d_sharedElemsFaces[1 + el * 7];
    const int dof1     = d_sharedElem1Dof12Q[1 + d_sharedElemsFaces[2 + el * 7] * 4];

    for (int i = 0; i <  dof1; i++) {
      index_i[i] = d_nodesIDs[offsetEl1 + i];
      for (int eq = 0; eq < num_equation; eq++) {
        Fcontrib[i + eq * dof1] = 0.;
      }
    }

    for (int elFace = 0; elFace < numFaces; elFace++) {
      const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
      const int Q = d_sharedElem1Dof12Q[3 + f * 4];

      for (int k = 0; k < Q; k++) {
        const double weight =
          d_sharedShapeWnor1[maxDofs + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];

        for (int eq = 0; eq < num_equation; eq++) {
          // TODO(kevin): MAXNUMFACE
          const int idxu = eq + k*num_equation + elFace*maxIntPoints*num_equation + el*5*maxIntPoints*num_equation;
          Rflux[eq] = d_shared_flux[idxu];
        }

        // add integration point contribution
        MFEM_FOREACH_THREAD(i, x, dof1) {
          const double shape = d_sharedShapeWnor1[i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * dof1] -= weight * shape * Rflux[eq];
          }
        }
        MFEM_SYNC_THREAD;
      }
    }
    // write to global memory
    MFEM_FOREACH_THREAD(i, x, dof1) {
      for (int eq = 0; eq < num_equation; eq++) {
        d_y[index_i[i] + eq * Ndofs] += Fcontrib[i + eq * dof1];
      }
    }
    MFEM_SYNC_THREAD;
  });
}
// clang-format on

void DGNonLinearForm::sharedFaceInterpolation_gpu(const Vector &x) {
  const double *d_x = x.Read();
  const double *d_gradUp = gradUp_->Read();
  const double *d_faceGradUp = transferGradUp->face_nbr_data.Read();
  const double *d_faceData = transferU->face_nbr_data.Read();
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();

  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const double *d_sharedShape2 = parallelData->sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = parallelData->sharedVdofs.Read();
  const int *d_sharedVdofsGrads = parallelData->sharedVdofsGradUp.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();

  const int maxNumElems = parallelData->sharedElemsFaces.Size() / 7;  // elements with shared faces
  const int Ndofs = vfes->GetNDofs();
  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();

  double *d_shared_flux = shared_flux.Write();

  const RiemannSolver *d_rsolver = rsolver_;
  Fluxes *d_flux = fluxes;

  MFEM_FORALL_2D(el, parallelData->sharedElemsFaces.Size() / 7, maxIntPoints, 1, 1, {
    double l1[gpudata::MAXDOFS], l2[gpudata::MAXDOFS];            // double l1[216], l2[216];
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS];  // double u1[5], u2[5];
    double gradUp1[gpudata::MAXDIM * gpudata::MAXEQUATIONS],      // double gradUp1[3 * 5];
        gradUp2[gpudata::MAXDIM * gpudata::MAXEQUATIONS],         // gradUp2[3 * 5];
        nor[gpudata::MAXDIM];                                     // nor[3];
    double Rflux[gpudata::MAXEQUATIONS],                          // double Rflux[5];
        vFlux1[gpudata::MAXDIM * gpudata::MAXEQUATIONS],          // vFlux1[3 * 5];
        vFlux2[gpudata::MAXDIM * gpudata::MAXEQUATIONS];          // vFlux2[3 * 5];
    int index_i[gpudata::MAXDOFS];                                // int index_i[216];

    const int el1 = d_sharedElemsFaces[0 + el * 7];
    const int numFaces = d_sharedElemsFaces[1 + el * 7];
    const int dof1 = d_sharedElem1Dof12Q[1 + d_sharedElemsFaces[2 + el * 7] * 4];

    const int offsetEl1 = d_posDofIds[2 * el1];

    for (int i = 0; i < dof1; i++) {
      index_i[i] = d_nodesIDs[offsetEl1 + i];
    }

    for (int elFace = 0; elFace < numFaces; elFace++) {
      const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
      const int dof2 = d_sharedElem1Dof12Q[2 + f * 4];
      const int Q = d_sharedElem1Dof12Q[3 + f * 4];

      // begin loop through integration points
      MFEM_FOREACH_THREAD(k, x, Q) {
        // load interpolating functions
        for (int i = 0; i < dof1; i++) {
          l1[i] = d_sharedShapeWnor1[i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
        }
        for (int i = 0; i < dof2; i++) {
          l2[i] = d_sharedShape2[i + k * maxDofs + f * maxIntPoints * maxDofs];
        }

        for (int d = 0; d < dim; d++) {
          nor[d] =
              d_sharedShapeWnor1[maxDofs + 1 + d + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
        }

        // set array for interpolated data to 0
        for (int eq = 0; eq < num_equation; eq++) {
          u1[eq] = u2[eq] = 0.;
          for (int d = 0; d < dim; d++) {
            gradUp1[eq + d * num_equation] = gradUp2[eq + d * num_equation] = 0.;
          }

          // load data for elem1
          for (int j = 0; j < dof1; j++) {
            u1[eq] += d_x[index_i[j] + eq * Ndofs] * l1[j];
          }

          // load data elem2
          for (int j = 0; j < dof2; j++) {
            int index = d_sharedVdofs[j + eq * maxDofs + f * num_equation * maxDofs];
            u2[eq] += d_faceData[index] * l2[j];
          }

          // interpolate gradients
          for (int d = 0; d < dim; d++) {
            // el1
            for (int j = 0; j < dof1; j++) {
              const double G = d_gradUp[index_i[j] + eq * Ndofs + d * num_equation * Ndofs];
              gradUp1[eq + d * num_equation] += l1[j] * G;
            }

            // el2
            for (int j = 0; j < dof2; j++) {
              int index =
                  d_sharedVdofsGrads[j + eq * maxDofs + d * num_equation * maxDofs + f * dim * num_equation * maxDofs];
              const double G = d_faceGradUp[index];
              gradUp2[eq + d * num_equation] += l2[j] * G;
            }
          }
        }

        // evaluate flux
#if defined(_CUDA_)
        // TODO(kevin): implement radius.
        d_rsolver->Eval_LF(u1, u2, nor, Rflux);
        d_flux->ComputeViscousFluxes(u1, gradUp1, 0.0, vFlux1);
        d_flux->ComputeViscousFluxes(u2, gradUp2, 0.0, vFlux2);
#elif defined(_HIP_)
        RiemannSolver::riemannLF_serial_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
        Fluxes::viscousFlux_serial_gpu(&vFlux1[0], &u1[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);
        Fluxes::viscousFlux_serial_gpu(&vFlux2[0], &u2[0], &gradUp2[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                       num_equation);
#endif

        for (int eq = 0; eq < num_equation; eq++) {
          for (int d = 0; d < dim; d++)
            vFlux1[eq + d * num_equation] = 0.5 * (vFlux1[eq + d * num_equation] + vFlux2[eq + d * num_equation]);
        }
        for (int eq = 0; eq < num_equation; eq++) {
          for (int d = 0; d < dim; d++) Rflux[eq] -= vFlux1[eq + d * num_equation] * nor[d];
        }

        for (int eq = 0; eq < num_equation; eq++) {
          // TODO(kevin): MAXNUMFACE
          const int idx =
              eq + k * num_equation + elFace * maxIntPoints * num_equation + el * 5 * maxIntPoints * num_equation;
          d_shared_flux[idx] = Rflux[eq];
        }
      }  // end loop through integration points
    }
  });
}

// clang-format on
#endif  // _GPU_
