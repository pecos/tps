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
#include "gradients.hpp"

#include "dgNonlinearForm.hpp"

Gradients::Gradients(ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_gradUpfes, int _dim, int _num_equation,
                     ParGridFunction *_Up, ParGridFunction *_gradUp, GasMixture *_mixture, GradNonLinearForm *_gradUp_A,
                     IntegrationRules *_intRules, int _intRuleType, const volumeFaceIntegrationArrays &_gpuArrays,
                     Array<DenseMatrix *> &_Me_inv, Vector &_invMArray, Array<int> &_posDofInvM,
                     const int &_maxIntPoints, const int &_maxDofs)
    : ParNonlinearForm(_vfes),
      vfes(_vfes),
      gradUpfes(_gradUpfes),
      dim_(_dim),
      num_equation_(_num_equation),
      Up(_Up),
      gradUp(_gradUp),
      mixture(_mixture),
      gradUp_A(_gradUp_A),
      intRules(_intRules),
      intRuleType(_intRuleType),
      gpuArrays(_gpuArrays),
      Me_inv(_Me_inv),
      invMArray(_invMArray),
      posDofInvM(_posDofInvM),
      maxIntPoints_(_maxIntPoints),
      maxDofs_(_maxDofs) {
  h_numElems = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();

  uk_el1.UseDevice(true);
  uk_el2.UseDevice(true);
  dun_face.UseDevice(true);

  int nfaces = vfes->GetMesh()->GetNumFaces();
  uk_el1.SetSize(nfaces * maxIntPoints_ * num_equation_);
  uk_el2.SetSize(nfaces * maxIntPoints_ * num_equation_);
  dun_face.SetSize(nfaces * maxIntPoints_ * num_equation_ * dim_);

  uk_el1 = 0.;
  uk_el2 = 0.;
  dun_face = 0.;

  Ke_positions_.SetSize(vfes->GetNE());
  auto h_Ke_positions = Ke_positions_.HostWrite();

  std::vector<double> temp;
  temp.clear();

  // element derivative stiffness matrix
  Ke.SetSize(vfes->GetNE());
  for (int el = 0; el < vfes->GetNE(); el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int eldDof = elem->GetDof();

    Ke[el] = new DenseMatrix(eldDof, dim_ * eldDof);

    // element volume integral
    int intorder = 2 * elem->GetOrder();
    if (intRuleType == 1 && elem->GetGeomType() == Geometry::SQUARE) intorder--;  // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);

    Vector shape(eldDof);
    DenseMatrix dshape(eldDof, dim_);
    DenseMatrix iGradUp(num_equation_, dim_);

    for (int i = 0; i < ir->GetNPoints(); i++) {
      IntegrationPoint ip = ir->IntPoint(i);
      Tr->SetIntPoint(&ip);

      // Calculate the shape functions
      elem->CalcShape(ip, shape);
      elem->CalcPhysDShape(*Tr, dshape);

      double detJac = Tr->Jacobian().Det() * ip.weight;

      for (int d = 0; d < dim_; d++) {
        for (int k = 0; k < eldDof; k++) {
          for (int j = 0; j < eldDof; j++) {
            (*Ke[el])(j, k + d * eldDof) += shape(j) * dshape(k, d) * detJac;
          }
        }
      }
    }

    h_Ke_positions[el] = temp.size();
    for (int j = 0; j < eldDof; j++) {
      for (int k = 0; k < dim_ * eldDof; k++) {
        temp.push_back((*Ke[el])(j, k));
      }
    }
  }

  Ke_array_.UseDevice(true);
  Ke_array_.SetSize(temp.size());
  auto h_Ke_array = Ke_array_.HostWrite();
  for (int i = 0; i < static_cast<int>(temp.size()); i++) h_Ke_array[i] = temp[i];

  auto d_Ke_array = Ke_array_.ReadWrite();
  auto d_Ke_positions = Ke_positions_.ReadWrite();
}

Gradients::~Gradients() {
  for (int n = 0; n < Ke.Size(); n++) delete Ke[n];
}

void Gradients::computeGradients() {
  const int totalDofs = vfes->GetNDofs();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();

  // Vars for face contributions
  Vector faceContrib(dim_ * num_equation_ * totalDofs);
  faceContrib = 0.;

  // compute volume integral and fill out above vectors
  DenseMatrix elGradUp;
  for (int el = 0; el < vfes->GetNE(); el++) {
    const FiniteElement *elem = vfes->GetFE(el);

    // get local primitive variables
    Array<int> vdofs;
    vfes->GetElementVDofs(el, vdofs);
    const int eldDof = elem->GetDof();
    DenseMatrix elUp(eldDof, num_equation_);
    for (int d = 0; d < eldDof; d++) {
      int index = vdofs[d];
      for (int eq = 0; eq < num_equation_; eq++) {
        elUp(d, eq) = dataUp[index + eq * totalDofs];
      }
    }

    elGradUp.SetSize(eldDof, num_equation_ * dim_);
    elGradUp = 0.;

    // Add volume contrubutions to gradient
    for (int eq = 0; eq < num_equation_; eq++) {
      for (int d = 0; d < dim_; d++) {
        for (int j = 0; j < eldDof; j++) {
          for (int k = 0; k < eldDof; k++) {
            elGradUp(j, eq + d * num_equation_) += (*Ke[el])(j, k + d * eldDof) * elUp(k, eq);
          }
        }
      }
    }

    // transfer result into gradUp
    for (int k = 0; k < eldDof; k++) {
      int index = vdofs[k];
      for (int eq = 0; eq < num_equation_; eq++) {
        for (int d = 0; d < dim_; d++) {
          dataGradUp[index + eq * totalDofs + d * num_equation_ * totalDofs] = -elGradUp(k, eq + d * num_equation_);
        }
      }
    }
  }

  // Calc boundary contribution
  gradUp_A->Mult(Up, faceContrib);

  // Add contributions and multiply by invers mass matrix
  for (int el = 0; el < vfes->GetNE(); el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    const int eldDof = elem->GetDof();

    Array<int> vdofs;
    vfes->GetElementVDofs(el, vdofs);

    Vector aux(eldDof);
    Vector rhs(eldDof);

    for (int d = 0; d < dim_; d++) {
      for (int eq = 0; eq < num_equation_; eq++) {
        for (int k = 0; k < eldDof; k++) {
          int index = vdofs[k];
          //           rhs[k] = gradUp[index+eq*totalDofs+d*num_equation_*totalDofs]+
          //                   faceContrib[index+eq*totalDofs+d*num_equation_*totalDofs];
          rhs[k] = -dataGradUp[index + eq * totalDofs + d * num_equation_ * totalDofs] +
                   faceContrib[index + eq * totalDofs + d * num_equation_ * totalDofs];
        }

        // mult by inv mass matrix
        Me_inv[el]->Mult(rhs, aux);

        // save this in gradUp
        for (int k = 0; k < eldDof; k++) {
          int index = vdofs[k];
          dataGradUp[index + eq * totalDofs + d * num_equation_ * totalDofs] = aux[k];
        }
      }
    }
  }

  gradUp->ExchangeFaceNbrData();
}

#ifdef _GPU_
void Gradients::computeGradients_domain() {
  DGNonLinearForm::setToZero_gpu(*gradUp, gradUp->Size());

  // Interpolate state info the faces (loops over elements)
  for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
    int elemOffset = 0;
    for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    int dof_el = h_posDofIds[2 * elemOffset + 1];
    interpFaceData_gpu(*Up, elType, elemOffset, dof_el);
  }

  evalFaceIntegrand_gpu();

  for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
    int elemOffset = 0;
    if (elType != 0) {
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2 * elemOffset + 1];

    faceContrib_gpu(h_numElems[elType], elemOffset, dof_el, vfes->GetNDofs(), *Up, *gradUp, num_equation_, dim_,
                    gpuArrays, maxDofs_, maxIntPoints_);
  }

  for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
    int elemOffset = 0;
    if (elType != 0) {
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2 * elemOffset + 1];

    computeGradients_gpu(h_numElems[elType], elemOffset, dof_el, vfes->GetNDofs(),
                         *Up,  // px,
                         *gradUp, num_equation_, dim_, gpuArrays, maxDofs_, maxIntPoints_);
  }
}

void Gradients::computeGradients_bdr() {
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if (Nshared > 0) {
    integrationGradSharedFace_gpu(Up, transferUp->face_nbr_data, gradUp, vfes->GetNDofs(), dim_, num_equation_,
                                  mixture->GetSpecificHeatRatio(), mixture->GetGasConstant(),
                                  mixture->GetViscMultiplyer(), mixture->GetBulkViscMultiplyer(),
                                  mixture->GetPrandtlNum(), gpuArrays, parallelData, maxIntPoints_, maxDofs_);
  }

  // Multiply by inverse mass matrix
  for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
    int elemOffset = 0;
    if (elType != 0) {
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2 * elemOffset + 1];
    multInverse_gpu(h_numElems[elType], elemOffset, dof_el, vfes->GetNDofs(), *gradUp, num_equation_, dim_, gpuArrays,
                    invMArray, posDofInvM);
  }
}

void Gradients::interpFaceData_gpu(const Vector &Up, int elType, int elemOffset, int elDof) {
  const double *d_x = Up.Read();  // Primitives!
  double *d_uk_el1 = uk_el1.Write();
  double *d_uk_el2 = uk_el2.Write();

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
    double uk1[20];
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
      MFEM_FOREACH_THREAD(k, x, Q) {
        // set interpolation data to 0
        for (int n = 0; n < num_equation; n++) {
          uk1[n] = 0.;
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
        }

        // save quad pt data to global memory
        if (swapElems) {
          for (int eq = 0; eq < num_equation; eq++) {
            d_uk_el2[eq + k * num_equation + gFace * maxIntPoints * num_equation] = uk1[eq];
          }
        } else {
          for (int eq = 0; eq < num_equation; eq++) {
            d_uk_el1[eq + k * num_equation + gFace * maxIntPoints * num_equation] = uk1[eq];
          }
        }
      }   // end loop over integration points (MFEM_FOREACH_THREAD)
    }  // end loop over faces
  });
  // clang-format on
}


void Gradients::computeGradients_gpu(const int numElems, const int offsetElems, const int elDof, const int totalDofs,
                                     const Vector &Up, Vector &gradUp, const int num_equation, const int dim,
                                     const volumeFaceIntegrationArrays &gpuArrays, const int &maxDofs,
                                     const int &maxIntPoints) {
  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  const double *d_elemShapeDshapeWJ = gpuArrays.elemShapeDshapeWJ.Read();
  auto d_elemPosQ_shapeDshapeWJ = gpuArrays.elemPosQ_shapeDshapeWJ.Read();

  auto d_Ke = Ke_array_.Read();
  auto d_Ke_pos = Ke_positions_.Read();

  MFEM_FORALL(el, numElems,
  {
    const int eli = el + offsetElems;
    const int offsetIDs    = d_posDofIds[2 * eli];
    const int offsetDShape = d_elemPosQ_shapeDshapeWJ[2 * eli   ];
    const int Q            = d_elemPosQ_shapeDshapeWJ[2 * eli + 1];

    const int keoffset = d_Ke_pos[eli];

    int index_i[216];
    double Ui[216], gradUpi[216 * 3];

    for (int i = 0; i < elDof; i++) {
      index_i[i] = d_nodesIDs[offsetIDs + i];
    }

    for ( int eq = 0; eq < num_equation; eq++ ) {
      for (int i = 0; i < elDof; i++) {
        Ui[i] = d_Up[index_i[i] + eq * totalDofs];
        for (int d = 0; d < dim; d++) {
          // this loads face contribution
          gradUpi[i + d * elDof] = d_gradUp[index_i[i] + eq * totalDofs + d * num_equation * totalDofs];
        }
      }

      // add the element interior contribution
      for (int d = 0; d < dim; d++) {
        for (int j = 0; j < elDof; j++) {
          for (int k = 0; k < elDof; k++) {
            gradUpi[j + d * elDof] += d_Ke[keoffset + dim * elDof * j + d * elDof + k] * Ui[k];
          }
        }
      }

      // write to global memory
      for (int i = 0; i < elDof; i++) {
        for (int d = 0; d < dim; d++) {
          d_gradUp[index_i[i] + eq * totalDofs + d * num_equation * totalDofs] = gradUpi[i + d * elDof];
        }
      }
    }  // end equation loop
  });
}

// clang-format on
void Gradients::evalFaceIntegrand_gpu() {
  auto d_dun = dun_face.Write();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();

  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  Mesh *mesh = fes->GetMesh();
  const int Nf = mesh->GetNumFaces();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  MFEM_FORALL(iface, Nf,
  {
    double u1[20], u2[20], nor[3];

    const int Q = d_elems12Q[3 * iface + 2];
    const int offsetShape1 = iface * maxIntPoints * (maxDofs + 1 + dim);
    const int offsetShape2 = iface * maxIntPoints * maxDofs;

    for (int k = 0; k < Q; k++) {
      const double weight = d_shapeWnor1[offsetShape1 + maxDofs + k * (maxDofs + 1 + dim)];

      double du[20];
      for (int eq = 0; eq < num_equation; eq++) {
        u1[eq] = d_uk_el1[eq + k * num_equation + iface * maxIntPoints * num_equation];
        u2[eq] = d_uk_el2[eq + k * num_equation + iface * maxIntPoints * num_equation];
        du[eq] = u2[eq] - u1[eq];
      }

      const int idx0 = iface * dim * maxIntPoints * num_equation + k * num_equation;
      for (int d = 0; d < dim; d++) {
        const int idx1 = idx0 + d * maxIntPoints * num_equation;
        nor[d] = weight * 0.5 * d_shapeWnor1[offsetShape1 + maxDofs + 1 + d + k * (maxDofs + 1 + dim)];
        for (int eq = 0; eq < num_equation; eq++) {
          d_dun[eq + idx1] = du[eq] * nor[d];
        }
      }
    }  // end loop over integration points
  });  // end loop over faces
}


// clang-format on
void Gradients::faceContrib_gpu(const int numElems, const int offsetElems, const int elDof, const int totalDofs,
                                const Vector &Up, Vector &gradUp, const int num_equation, const int dim,
                                const volumeFaceIntegrationArrays &gpuArrays, const int &maxDofs,
                                const int &maxIntPoints) {
  const double *d_Up = Up.Read();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();
  const double *d_dun = dun_face.Read();

  double *d_gradUp = gradUp.Write();  // NB: I assume this comes in set to zero!
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();

  // pointers for face integration
  auto d_elemFaces = gpuArrays.elemFaces.Read();
  auto d_shapeWnor1 = gpuArrays.shapeWnor1.Read();
  const double *d_shape2 = gpuArrays.shape2.Read();
  auto d_elems12Q = gpuArrays.elems12Q.Read();

  MFEM_FORALL_2D(el, numElems, elDof, 1, 1,
  {
    const int eli = el + offsetElems;
    const int offsetIDs    = d_posDofIds[2 * eli];

    MFEM_FOREACH_THREAD(i, x, elDof) {
      const int idx = d_nodesIDs[offsetIDs + i];
      double const* shape;

      // ================  FACE CONTRIBUTION  ================
      const int elFaces = d_elemFaces[7 * eli];
      for (int face = 0; face < elFaces; face++) {
        const int gFace = d_elemFaces[7 * eli + face + 1];
        const int Q = d_elems12Q[3 * gFace + 2];
        const int offsetShape1 = gFace * maxIntPoints * (maxDofs + 1 + dim);
        const int offsetShape2 = gFace * maxIntPoints * maxDofs;
        bool swapElems = false;

        // get neighbor
        int elj = d_elems12Q[3 * gFace];
        if (elj != eli) {
          swapElems = true;
        }

        for (int k = 0; k < Q; k++) {
          // load interpolators
          if (swapElems) {
            shape = d_shape2 + offsetShape2 + k * maxDofs;
          } else {
            shape = d_shapeWnor1 + offsetShape1 + k * (maxDofs + 1 + dim);
          }

          const int idx0 = k * num_equation + gFace * dim * maxIntPoints * num_equation;
          for (int d = 0; d < dim; d++) {
            const int idx1 = d * maxIntPoints * num_equation + idx0;
            for (int eq = 0; eq < num_equation; eq++) {
              const int idx2 = eq + idx1;
              const int ioff1 = eq * totalDofs + d * num_equation * totalDofs;
              d_gradUp[idx + ioff1] += d_dun[idx2] * shape[i];
            }
          }
        }
      }
    }
  });
}

void Gradients::integrationGradSharedFace_gpu(const Vector *Up, const Vector &faceUp, ParGridFunction *gradUp,
                                              const int &Ndofs, const int &dim, const int &num_equation,
                                              const double &gamma, const double &Rg, const double &viscMult,
                                              const double &bulkViscMult, const double &Pr,
                                              const volumeFaceIntegrationArrays &gpuArrays,
                                              const parallelFacesIntegrationArrays *parallelData,
                                              const int &maxIntPoints, const int &maxDofs) {
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

  MFEM_FORALL_2D(el, parallelData->sharedElemsFaces.Size() / 7, maxDofs, 1, 1, { // NOLINT
    MFEM_FOREACH_THREAD(i, x, maxDofs) { // NOLINT
      //
      MFEM_SHARED double Upi[216], Upj[216], Fcontrib[216 * 3];
      MFEM_SHARED double l1[216], l2[216];
      MFEM_SHARED double nor[3];

      const int el1      = d_sharedElemsFaces[0 + el * 7];
      const int numFaces = d_sharedElemsFaces[1 + el * 7];
      const int dof1     = d_sharedElem1Dof12Q[1 + d_sharedElemsFaces[2 + el * 7] * 4];
      const int offsetEl1 = d_posDofIds[2 * el1];

      int indexi;
      if ( i < dof1 ) indexi = d_nodesIDs[offsetEl1 + i];

      MFEM_SHARED double up1, up2;

      for (int elFace = 0; elFace < numFaces; elFace++) {
    const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
    const int dof2 = d_sharedElem1Dof12Q[2 + f * 4];
    const int Q = d_sharedElem1Dof12Q[3 + f * 4];

    for (int eq = 0; eq < num_equation; eq++) {
      if (i < dof1) Upi[i] = d_up[indexi + eq * Ndofs];
      if (i < dof2) {  // recover data from neighbor
        int index = d_sharedVdofs[i + eq * maxDofs + f * num_equation * maxDofs];
        Upj[i] = d_faceData[index];
      }
      MFEM_SYNC_THREAD;

      for (int n = i; n < dof1 * dim; n += maxDofs) Fcontrib[n] = 0.;

      for (int k = 0; k < Q; k++) {
        const double weight =
            d_sharedShapeWnor1[maxDofs + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
        if (i < dof1) l1[i] = d_sharedShapeWnor1[i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
        if (i < dim)
          nor[i] =
              d_sharedShapeWnor1[maxDofs + 1 + i + k * (maxDofs + 1 + dim) + f * maxIntPoints * (maxDofs + 1 + dim)];
        if (dim == 2 && i == maxDofs - 1) nor[2] = 0.;
        if (i < dof2) l2[i] = d_sharedShape2[i + k * maxDofs + f * maxIntPoints * maxDofs];
        MFEM_SYNC_THREAD;

        // interpolation
        // NOTE: make parallel!
        if (i == 0) {
          up1 = 0.;
          up2 = 0.;
          for (int n = 0; n < dof1; n++) up1 += Upi[n] * l1[n];
          for (int n = 0; n < dof2; n++) up2 += Upj[n] * l2[n];
        }
        MFEM_SYNC_THREAD;

        // add contribution
        if (i < dof1) {
          for (int d = 0; d < dim; d++) Fcontrib[i + d * dof1] += 0.5 * (up2 - up1) * weight * l1[i] * nor[d];
        }
        MFEM_SYNC_THREAD;
      }  // end integration loop

      // save contribution to global memory
      if (i < dof1) {
        for (int d = 0; d < dim; d++) {
          d_gradUp[indexi + eq * Ndofs + d * num_equation * Ndofs] += Fcontrib[i + d * dof1];
        }
      }
      MFEM_SYNC_THREAD;
    }    // end equation loop
      }  // end face loop
}
});
}

void Gradients::multInverse_gpu(const int numElems, const int offsetElems, const int elDof, const int totalDofs,
                                Vector &gradUp, const int num_equation, const int dim,
                                const volumeFaceIntegrationArrays &gpuArrays, const Vector &invMArray,
                                const Array<int> &posDofInvM) {
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  const double *d_invMArray = invMArray.Read();
  auto d_posDofInvM = posDofInvM.Read();

  MFEM_FORALL(el, numElems,
  {
    double gradUpi[216 * 3];

    const int eli       = el + offsetElems;
    const int offsetIDs = d_posDofIds[2 * eli];
    const int offsetInv = d_posDofInvM[2 * eli];

    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) {

        for (int i = 0; i < elDof; i++) {
          const int indexi    = d_nodesIDs[offsetIDs + i];
          gradUpi[i + d * elDof] = d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs];
        }


        for (int i = 0; i < elDof; i++) {
          const int indexi    = d_nodesIDs[offsetIDs + i];
          double temp = 0;
          for (int n = 0; n < elDof; n++) {
            temp += gradUpi[n + d * elDof] * d_invMArray[offsetInv + i * elDof + n];
          }
          d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs] = temp;
        }
      }
    }
  });


//   // NOTE: I'm sure this can be done more efficiently. Have a think
//   MFEM_FORALL_2D(el, numElems, elDof, 1, 1, { // NOLINT
//     MFEM_FOREACH_THREAD(i, x, elDof) { // NOLINT
//       //
//       MFEM_SHARED double gradUpi[216 * 3];

//       const int eli       = el + offsetElems;
//       const int offsetIDs = d_posDofIds[2 * eli];
//       const int offsetInv = d_posDofInvM[2 * eli];
//       const int indexi    = d_nodesIDs[offsetIDs + i];

//       for (int eq = 0; eq < num_equation; eq++) {
//     for (int d = 0; d < dim; d++) {
//       gradUpi[i + d * elDof] = d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs];
//     }

//     MFEM_SYNC_THREAD;

//     for (int d = 0; d < dim; d++) {
//       double temp = 0;
//       for (int n = 0; n < elDof; n++) {
//         temp += gradUpi[n + d * elDof] * d_invMArray[offsetInv + i * elDof + n];
//       }
//       d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs] = temp;
//     }
//     MFEM_SYNC_THREAD;
//       }
// }
// });
}
#endif
