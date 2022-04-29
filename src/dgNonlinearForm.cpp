// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

DGNonLinearForm::DGNonLinearForm(Fluxes *_flux, ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_gradFes,
                                 ParGridFunction *_gradUp, BCintegrator *_bcIntegrator, IntegrationRules *_intRules,
                                 const int _dim, const int _num_equation, GasMixture *_mixture,
                                 const volumeFaceIntegrationArrays &_gpuArrays, const int &_maxIntPoints,
                                 const int &_maxDofs)
    : ParNonlinearForm(_vfes),
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

  shared_uk_el1.UseDevice(true);
  shared_uk_el2.UseDevice(true);
  shared_grad_upk_el1.UseDevice(true);
  shared_grad_upk_el2.UseDevice(true);

  int maxNumElems = parallelData->sharedElemsFaces.Size() / 7;  // elements with shared faces
  shared_uk_el1.SetSize(maxNumElems * 5 * maxIntPoints_ * num_equation_);
  shared_uk_el2.SetSize(maxNumElems * 5 * maxIntPoints_ * num_equation_);
  shared_grad_upk_el1.SetSize(maxNumElems * 5 * maxIntPoints_ * num_equation_ * dim_);
  shared_grad_upk_el2.SetSize(maxNumElems * 5 * maxIntPoints_ * num_equation_ * dim_);
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
  setToZero_gpu(y, y.Size());

  Mesh *mesh = fes->GetMesh();

  // internal face integration
  if (fnfi.Size()) {

    // Interpolate state info the faces
    for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
      int elemOffset = 0;
      if (elType != 0) {
        for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
      }
      int dof_el = h_posDofIds[2 * elemOffset + 1];
      interpFaceData_gpu(x, elType, elemOffset, dof_el);
    }
    MFEM_DEVICE_SYNC;

    // Evaluate fluxes at quadrature points
    evalFaceFlux_gpu();
    MFEM_DEVICE_SYNC;

    // Compute flux contributions to residual
    // NB: This loop *must* be separate from above b/c we have to hit
    // every element prior to working on any faces in order to ensure
    // all necessary data is available
    for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
      int elemOffset = 0;
      if (elType != 0) {
        for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
      }
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
    sharedFaceInterpolation_gpu(x, transferU->face_nbr_data, gradUp_, transferGradUp->face_nbr_data, shared_uk_el1,
                                shared_uk_el2, shared_grad_upk_el1, shared_grad_upk_el2, vfes->GetNDofs(), dim_,
                                num_equation_, mixture, gpuArrays, parallelData, maxIntPoints_, maxDofs_);
    sharedFaceIntegration_gpu(x, transferU->face_nbr_data, gradUp_, transferGradUp->face_nbr_data, y, shared_uk_el1,
                              shared_uk_el2, shared_grad_upk_el1, shared_grad_upk_el2, vfes->GetNDofs(), dim_,
                              num_equation_, mixture, fluxes, gpuArrays, parallelData, maxIntPoints_, maxDofs_);
  }
}

void DGNonLinearForm::setToZero_gpu(Vector &x, const int size) {
  double *d_x = x.Write();

  MFEM_FORALL(i, size, { d_x[i] = 0.; });
}

// clang-format off
void DGNonLinearForm::faceIntegration_gpu(Vector &y, int elType, int elemOffset, int elDof) {
  double *d_y = y.Write();
  const double *d_f = face_flux_.Read();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();
  const double *d_grad_uk_el1 = grad_upk_el1.Read();
  const double *d_grad_uk_el2 = grad_upk_el2.Read();

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
  MFEM_FORALL(el, NumElemType,
  {
    double shape1[216], shape2[216];
    double u1[5], gradUp1[5 * 3];
    double u2[5], gradUp2[5 * 3];
    double vFlux1[5 * 3], vFlux2[5 * 3];
    double Rflux[5], nor[3], Fcontrib[216 * 5];
    int indexes_i[216];

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
        // for (int eq = 0; eq < dim; eq++)
        //   nor[eq] = d_shapeWnor1[offsetShape1 + maxDofs + 1 + eq + k * (maxDofs + 1 + dim)];

	for (int j = 0; j < dof1; j++) shape1[j] = d_shapeWnor1[offsetShape1 + j + k * (maxDofs + 1 + dim)];
        for (int j = 0; j < dof2; j++) shape2[j] = d_shape2[offsetShape2 + j + k * maxDofs];

	// for (int eq = 0; eq < num_equation; eq++) {
        //   u1[eq] = 0.;
        //   u2[eq] = 0.;
        //   for (int d = 0; d < dim; d++) {
        //     gradUp1[eq + d * num_equation] = 0.;
        //     gradUp2[eq + d * num_equation] = 0.;
        //   }
        // }

        // // interpolate to integration point k
        // for (int eq = 0; eq < num_equation; eq++) {
        //   u1[eq] = d_uk_el1[eq + k * num_equation + gFace * maxIntPoints * num_equation];
        //   u2[eq] = d_uk_el2[eq + k * num_equation + gFace * maxIntPoints * num_equation];

        //   for (int d = 0; d < dim; d++) {
        //     gradUp1[eq + d * num_equation] = d_grad_uk_el1[eq + k * num_equation + d * maxIntPoints * num_equation +
        //                                                    gFace * dim * maxIntPoints * num_equation];
        //     gradUp2[eq + d * num_equation] = d_grad_uk_el2[eq + k * num_equation + d * maxIntPoints * num_equation +
        //                                                    gFace * dim * maxIntPoints * num_equation];
        //   }
        // }

        // RiemannSolver::riemannLF_serial_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
        // Fluxes::viscousFlux_serial_gpu(&vFlux1[0], &u1[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
        //                                num_equation);
        // Fluxes::viscousFlux_serial_gpu(&vFlux2[0], &u2[0], &gradUp2[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
        //                                num_equation);
	// for (int d = 0; d < dim; d++) {
	//   for (int eq = 0; eq < num_equation; eq++) {
        //     vFlux1[eq + d * num_equation] = 0.5 * (vFlux1[eq + d * num_equation] + vFlux2[eq + d * num_equation]);
	//   }
        // }

	// for (int d = 0; d < dim; d++) {
	//   for (int eq = 0; eq < num_equation; eq++) {
	//     Rflux[eq] -= vFlux1[eq + d * num_equation] * nor[d];
	//   }
	// }

        for (int eq = 0; eq < num_equation; eq++) {
          Rflux[eq] = d_f[eq + k * num_equation + gFace * maxIntPoints * num_equation];
        }


	if (swapElems) {
	  for (int eq = 0; eq < num_equation; eq++) {
	    for (int i = 0; i < elDof; i++) {
              Fcontrib[i + eq * elDof] += weight * shape2[i] * Rflux[eq];
	    }
	  }
	} else {
	  for (int eq = 0; eq < num_equation; eq++) {
	    for (int i = 0; i < elDof; i++) {
	      Fcontrib[i + eq * elDof] -= weight * shape1[i] * Rflux[eq];
	    }
	  }
	}
      } // end loop over quad pts
    } // end loop over faces
  
    for (int eq = 0; eq < num_equation; eq++ ) {
      for (int i = 0; i < elDof; i++) {
	int index = indexes_i[i];
	d_y[index + eq * Ndofs] += Fcontrib[i + eq * elDof];
      }
    }
  });
}

// clang-format off
void DGNonLinearForm::evalFaceFlux_gpu() {
  double *d_f = face_flux_.Write();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();
  const double *d_grad_uk_el1 = grad_upk_el1.Read();
  const double *d_grad_uk_el2 = grad_upk_el2.Read();

  auto d_elemFaces = gpuArrays.elemFaces.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();

  Mesh *mesh = fes->GetMesh();
  const int Nf = mesh->GetNumFaces();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  // clang-format off
  MFEM_FORALL(iface, Nf,
  {
    double u1[5], gradUp1[5 * 3];
    double u2[5], gradUp2[5 * 3];
    double vFlux1[5 * 3], vFlux2[5 * 3];
    double Rflux[5], nor[3], Fcontrib[216 * 5];

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
          gradUp1[eq + d * num_equation] = d_grad_uk_el1[eq + k * num_equation + d * maxIntPoints * num_equation +
                                                         iface * dim * maxIntPoints * num_equation];
          gradUp2[eq + d * num_equation] = d_grad_uk_el2[eq + k * num_equation + d * maxIntPoints * num_equation +
                                                         iface * dim * maxIntPoints * num_equation];
        }
      }

      // evaluate flux
      RiemannSolver::riemannLF_serial_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
      Fluxes::viscousFlux_serial_gpu(&vFlux1[0], &u1[0], &gradUp1[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                     num_equation);
      Fluxes::viscousFlux_serial_gpu(&vFlux2[0], &u2[0], &gradUp2[0], gamma, Rg, viscMult, bulkViscMult, Pr, dim,
                                     num_equation);
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
    } // end loop over quad pts
  });
}

// clang-format off
void DGNonLinearForm::interpFaceData_gpu(const Vector &x, int elType, int elemOffset, int elDof) {
#ifdef _GPU_
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

  //MFEM_FORALL(el, NumElemsType, {
  MFEM_FORALL_2D(el, NumElemsType, maxIntPoints, 1, 1, {
      // assuming max. num of equations = 20
      // and max elem dof is 216 (a p=5 hex)
      double uk1[20], gradUpk1[20 * 3];
      double shape[216];
      int indexes_i[216];

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
        //for (int k = 0; k < Q; k++) {
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
		d_grad_uk_el2[eq + k * num_equation + d * maxIntPoints * num_equation +
			      gFace * dim * maxIntPoints * num_equation] = gradUpk1[eq + d * num_equation];
	      }
	    }
	  } else {
	    for (int eq = 0; eq < num_equation; eq++) {
	      d_uk_el1[eq + k * num_equation + gFace * maxIntPoints * num_equation] = uk1[eq];
	      for (int d = 0; d < dim; d++) {
		d_grad_uk_el1[eq + k * num_equation + d * maxIntPoints * num_equation +
			      gFace * dim * maxIntPoints * num_equation] = gradUpk1[eq + d * num_equation];
	      }
	    }
	  }

        }   // end loop over integrtion points
      }  // end loop over faces
    });
#endif
}

// clang-format off
void DGNonLinearForm::sharedFaceIntegration_gpu(
    const Vector &x, const Vector &faceU, const ParGridFunction *gradUp, const Vector &faceGradUp, Vector &y,
    Vector &shared_uk_el1,
    Vector &shared_uk_el2,
    Vector &shared_grad_upk_el1,
    Vector &shared_grad_upk_el2,
    const int &Ndofs, const int &dim, const int &num_equation, GasMixture *mixture, Fluxes *flux,
    const volumeFaceIntegrationArrays &gpuArrays,
    const parallelFacesIntegrationArrays *parallelData, const int &maxIntPoints, const int &maxDofs) {
  const double *d_x = x.Read();
  const double *d_gradUp = gradUp->Read();
  const double *d_faceGradUp = faceGradUp.Read();
  const double *d_faceData = faceU.Read();
  double *d_y = y.ReadWrite();
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();

  const double gamma = mixture->GetSpecificHeatRatio();
  const double Rg    = mixture->GetGasConstant();
  const double viscMult = mixture->GetViscMultiplyer();
  const double bulkViscMult = mixture->GetBulkViscMultiplyer();
  const double Pr = mixture->GetPrandtlNum();
  const double Sc = mixture->GetSchmidtNum();
  const Equations eqSystem = flux->GetEquationSystem();

  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const double *d_sharedShape2 = parallelData->sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = parallelData->sharedVdofs.Read();
  const int *d_sharedVdofsGrads = parallelData->sharedVdofsGradUp.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();

  int maxNumElems = parallelData->sharedElemsFaces.Size()/7;  // elements with shared faces
  const double *d_shared_uk1 = shared_uk_el1.Read();
  const double *d_shared_uk2 = shared_uk_el2.Read();
  const double *d_shared_gradUp1 = shared_grad_upk_el1.Read();
  const double *d_shared_gradUp2 = shared_grad_upk_el2.Read();

  MFEM_FORALL_2D(el, parallelData->sharedElemsFaces.Size() / 7, maxDofs, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, maxDofs) {
      //
      MFEM_SHARED double Fcontrib[216 * 20];
      MFEM_SHARED double l1[216], l2[216];
      MFEM_SHARED double Rflux[20], u1[20], u2[20], nor[3];
      MFEM_SHARED double gradUp1[20 * 3], gradUp2[20 * 3];
      MFEM_SHARED double vFlux1[20 * 3], vFlux2[20 * 3];

      const int el1      = d_sharedElemsFaces[0 + el * 7];
      const int numFaces = d_sharedElemsFaces[1 + el * 7];
      const int dof1     = d_sharedElem1Dof12Q[1 + d_sharedElemsFaces[2 + el * 7] * 4];

      bool elemDataRecovered = false;
      int indexi;

      for (int elFace = 0; elFace < numFaces; elFace++) {
        const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
        const int dof2 = d_sharedElem1Dof12Q[2 + f * 4];
        const int Q = d_sharedElem1Dof12Q[3 + f * 4];
        const int offsetEl1 = d_posDofIds[2 * el1];

        if (i < dof1 && !elemDataRecovered) {
          indexi = d_nodesIDs[offsetEl1 + i];
          for (int eq = 0; eq < num_equation; eq++) {
            Fcontrib[i + eq * dof1] = 0.;
          }
          elemDataRecovered = true;
        }
        MFEM_SYNC_THREAD;

        for (int k = 0; k < Q; k++) {
          const double weight =
            d_sharedShapeWnor1[maxDofs + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
          if (i < dof1)
            l1[i] = d_sharedShapeWnor1[i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
          if (i < dim)
            nor[i] = d_sharedShapeWnor1[maxDofs + 1 + i + k * (maxDofs + 1 + dim) +
                                        f * maxIntPoints * (maxDofs + 1 + dim)];
          if (dim == 2 && i == maxDofs - 1) nor[2] = 0.;
          if (i < dof2) l2[i] = d_sharedShape2[i + k * maxDofs + f * maxIntPoints * maxDofs];
          if (i < num_equation) {
            u1[i] = d_shared_uk1[i + k*num_equation + elFace*maxIntPoints*num_equation +
                                  el*5*maxIntPoints*num_equation];
            u2[i] = d_shared_uk2[i + k*num_equation + elFace*maxIntPoints*num_equation +
                                  el*5*maxIntPoints*num_equation];
            Rflux[i] = 0.;
            for (int d = 0; d < dim; d++) {
              gradUp1[i + d * num_equation] = d_shared_gradUp1[i + k*num_equation + elFace*maxIntPoints*num_equation +
                el*5*maxIntPoints*num_equation + d*maxNumElems*5*maxIntPoints*num_equation];
              gradUp2[i + d * num_equation] = d_shared_gradUp2[i + k*num_equation + elFace*maxIntPoints*num_equation +
                el*5*maxIntPoints*num_equation + d*maxNumElems*5*maxIntPoints*num_equation];
            }
          }
          MFEM_SYNC_THREAD;

          // interpolate
//           if (i < num_equation) {
//             for (int n = 0; n < dof1; n++) {
//               u1[i] += Ui[n + i * dof1] * l1[n];
//               for (int d = 0; d < dim; d++)
//                 gradUp1[i + d * num_equation] += gradUpi[n + i * dof1 + d * num_equation * dof1] * l1[n];
//             }
//             for (int n = 0; n < dof2; n++) {
//               //             u2[i] += Uj[n+i*dof2]*l2[n];
//               //             for(int d=0;d<dim;d++) gradUp2[i+d*num_equation] +=
//               //                           gradUpj[n+i*dof2+d*num_equation*dof2]*l2[n];
//               int index = d_sharedVdofs[n + i * maxDofs + f * num_equation * maxDofs];
//               u2[i] += l2[n] * d_faceData[index];
//               for (int d = 0; d < dim; d++) {
//                 index = d_sharedVdofsGrads[n + i * maxDofs + d * num_equation * maxDofs +
//                                              f * dim * num_equation * maxDofs];
//                 gradUp2[i + d * num_equation] += l2[n] * d_faceGradUp[index];
//               }
//             }
//           }
//           MFEM_SYNC_THREAD;
          // compute Riemann flux
          RiemannSolver::riemannLF_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg,
                                       dim, eqSystem, num_equation, i, maxDofs);
          Fluxes::viscousFlux_gpu(&vFlux1[0], &u1[0], &gradUp1[0], eqSystem, gamma, Rg, viscMult,
                                  bulkViscMult, Pr, Sc, i, maxDofs, dim, num_equation);
          Fluxes::viscousFlux_gpu(&vFlux2[0], &u2[0], &gradUp2[0], eqSystem, gamma, Rg, viscMult, bulkViscMult,
                                  Pr, Sc, i, maxDofs, dim, num_equation);

          MFEM_SYNC_THREAD;
          if (i < num_equation) {
            for (int d = 0; d < dim; d++)
              vFlux1[i + d * num_equation] = 0.5 * (vFlux1[i + d * num_equation] + vFlux2[i + d * num_equation]);
          }
          MFEM_SYNC_THREAD;
          if (i < num_equation) {
            for (int d = 0; d < dim; d++) Rflux[i] -= vFlux1[i + d * num_equation] * nor[d];
          }
          MFEM_SYNC_THREAD;

          // add integration point contribution
          for (int eq = 0; eq < num_equation; eq++) {
            if (i < dof1) Fcontrib[i + eq * dof1] -= weight * l1[i] * Rflux[eq];
          }
          MFEM_SYNC_THREAD;
        }
        MFEM_SYNC_THREAD;
      }

      // write to global memory
      for (int eq = 0; eq < num_equation; eq++) {
        if (i < dof1)
          d_y[indexi + eq * Ndofs] += Fcontrib[i + eq * dof1];
      }
    }
  });
}
// clang-format on

void DGNonLinearForm::sharedFaceInterpolation_gpu(const Vector &x, const Vector &faceU, const ParGridFunction *gradUp,
                                                  const Vector &faceGradUp, Vector &shared_uk_el1,
                                                  Vector &shared_uk_el2, Vector &shared_grad_upk_el1,
                                                  Vector &shared_grad_upk_el2, const int &Ndofs, const int &dim,
                                                  const int &num_equation, GasMixture *mixture,
                                                  const volumeFaceIntegrationArrays &gpuArrays,
                                                  const parallelFacesIntegrationArrays *parallelData,
                                                  const int &maxIntPoints, const int &maxDofs) {
  const double *d_x = x.Read();
  const double *d_gradUp = gradUp->Read();
  const double *d_faceGradUp = faceGradUp.Read();
  const double *d_faceData = faceU.Read();
  const int *d_nodesIDs = gpuArrays.nodesIDs.Read();
  const int *d_posDofIds = gpuArrays.posDofIds.Read();

  const double *d_sharedShapeWnor1 = parallelData->sharedShapeWnor1.Read();
  const double *d_sharedShape2 = parallelData->sharedShape2.Read();
  const int *d_sharedElem1Dof12Q = parallelData->sharedElem1Dof12Q.Read();
  const int *d_sharedVdofs = parallelData->sharedVdofs.Read();
  const int *d_sharedVdofsGrads = parallelData->sharedVdofsGradUp.Read();
  const int *d_sharedElemsFaces = parallelData->sharedElemsFaces.Read();

  int maxNumElems = parallelData->sharedElemsFaces.Size() / 7;  // elements with shared faces

  double *d_shared_uk1 = shared_uk_el1.Write();
  double *d_shared_uk2 = shared_uk_el2.Write();
  double *d_shared_gradUp1 = shared_grad_upk_el1.Write();
  double *d_shared_gradUp2 = shared_grad_upk_el2.Write();

  MFEM_FORALL_2D(el, parallelData->sharedElemsFaces.Size() / 7, maxDofs, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, maxDofs) {
      MFEM_SHARED double Ui[216];
      MFEM_SHARED double l1[216], l2[216];
      MFEM_SHARED double u1[20], u2[20];
      MFEM_SHARED double gradUp1[20 * 3], gradUp2[20 * 3];

      const int el1      = d_sharedElemsFaces[0 + el * 7];
      const int numFaces = d_sharedElemsFaces[1 + el * 7];
      const int dof1     = d_sharedElem1Dof12Q[1 + d_sharedElemsFaces[2 + el * 7] * 4];

      const int offsetEl1 = d_posDofIds[2 * el1];
      int  indexi;
      if ( i < dof1 ) indexi = d_nodesIDs[offsetEl1 + i];

      for (int elFace = 0; elFace < numFaces; elFace++) {
    const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
    const int dof2 = d_sharedElem1Dof12Q[2 + f * 4];
    const int Q = d_sharedElem1Dof12Q[3 + f * 4];

    // begin loop through integration points
    for (int k = 0; k < Q; k++) {
      // load interpolating functions
      if (i < dof1) l1[i] = d_sharedShapeWnor1[i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
      if (i < dof2) l2[i] = d_sharedShape2[i + k * maxDofs + f * maxIntPoints * maxDofs];
      MFEM_SYNC_THREAD;

      // set array for interpolated data to 0
      for (int eq = i; eq < num_equation; eq += maxDofs) {
        u1[eq] = 0.;
        u2[eq] = 0.;
        for (int d = 0; d < dim; d++) {
          gradUp1[eq + d * num_equation] = 0.;
          gradUp2[eq + d * num_equation] = 0.;
        }
      }
      MFEM_SYNC_THREAD;

      // loop through the equations
      for (int eq = 0; eq < num_equation; eq++) {
        // load data for elem1
        for (int n = i; n < dof1; n += maxDofs) {
          Ui[n] = d_x[indexi + eq * Ndofs];
        }
        MFEM_SYNC_THREAD;

        // interpolate el1
        // NOTE: make this parallel
        if (i == 0) {
          for (int j = 0; j < dof1; j++) u1[eq] += Ui[j] * l1[j];
        }
        MFEM_SYNC_THREAD;

        // load data elem2
        for (int j = i; j < dof2; j += maxDofs) {
          int index = d_sharedVdofs[j + eq * maxDofs + f * num_equation * maxDofs];
          Ui[j] = d_faceData[index];
        }
        MFEM_SYNC_THREAD;

        // interpolate el2
        // NOTE: make this parallel
        if (i == 0) {
          for (int j = 0; j < dof2; j++) u2[eq] += l2[j] * Ui[j];
        }
        MFEM_SYNC_THREAD;

        // interpolate gradients
        for (int d = 0; d < dim; d++) {
          // el1
          for (int j = i; j < dof1; j += maxDofs) Ui[j] = d_gradUp[indexi + eq * Ndofs + d * num_equation * Ndofs];
          MFEM_SYNC_THREAD;

          // NOTE: make interpolation parallel
          if (i == 0) {
            for (int j = 0; j < dof1; j++) gradUp1[eq + d * num_equation] += l1[j] * Ui[j];
          }
          MFEM_SYNC_THREAD;

          // el2
          for (int j = i; j < dof2; j += maxDofs) {
            int index =
                d_sharedVdofsGrads[j + eq * maxDofs + d * num_equation * maxDofs + f * dim * num_equation * maxDofs];
            Ui[j] = d_faceGradUp[index];
          }
          MFEM_SYNC_THREAD;

          // NOTE: make interpolation parallel
          if (i == 0) {
            for (int j = 0; j < dof2; j++) gradUp2[eq + d * num_equation] += l2[j] * Ui[j];
          }
          MFEM_SYNC_THREAD;
        }
      }
      // save interpolated data to global arrays
      for (int eq = i; eq < num_equation; eq += maxDofs) {
        d_shared_uk1[eq + k * num_equation + elFace * maxIntPoints * num_equation +
                     el * 5 * maxIntPoints * num_equation] = u1[eq];
        d_shared_uk2[eq + k * num_equation + elFace * maxIntPoints * num_equation +
                     el * 5 * maxIntPoints * num_equation] = u2[eq];
        for (int d = 0; d < dim; d++) {
          d_shared_gradUp1[eq + k * num_equation + elFace * maxIntPoints * num_equation +
                           el * 5 * maxIntPoints * num_equation + d * maxNumElems * 5 * maxIntPoints * num_equation] =
              gradUp1[eq + d * num_equation];
          d_shared_gradUp2[eq + k * num_equation + elFace * maxIntPoints * num_equation +
                           el * 5 * maxIntPoints * num_equation + d * maxNumElems * 5 * maxIntPoints * num_equation] =
              gradUp2[eq + d * num_equation];
        }
      }
    }    // end loop through integration points
      }  // end loop through faces
}
});
}

// clang-format on
#endif  // _GPU_
