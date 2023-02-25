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
#include "gradients.hpp"

#include "dgNonlinearForm.hpp"

Gradients::Gradients(ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_gradUpfes, int _dim, int _num_equation,
                     ParGridFunction *_Up, ParGridFunction *_gradUp, GasMixture *_mixture, GradNonLinearForm *_gradUp_A,
                     IntegrationRules *_intRules, int _intRuleType, const precomputedIntegrationData &_gpuArrays,
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
  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  h_numElems = elem_data.numElems.HostRead();

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

  Ke_array_.ReadWrite();
  Ke_positions_.ReadWrite();
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

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  auto h_elem_dof_num = elem_data.element_dof_number.HostRead();

  // Interpolate state info the faces (loops over elements)
  for (int elType = 0; elType < elem_data.numElems.Size(); elType++) {
    int elemOffset = 0;
    for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    int dof_el = h_elem_dof_num[elemOffset];
    interpFaceData_gpu(*Up, elType, elemOffset, dof_el);
  }

  evalFaceIntegrand_gpu();

  for (int elType = 0; elType < elem_data.numElems.Size(); elType++) {
    int elemOffset = 0;
    for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    int dof_el = h_elem_dof_num[elemOffset];
    faceContrib_gpu(elType, elemOffset, dof_el);
  }

  for (int elType = 0; elType < elem_data.numElems.Size(); elType++) {
    int elemOffset = 0;
    for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    int dof_el = h_elem_dof_num[elemOffset];
    computeGradients_gpu(elType, elemOffset, dof_el);
  }
}

void Gradients::computeGradients_bdr() {
  ParMesh *pmesh = vfes->GetParMesh();
  const int Nshared = pmesh->GetNSharedFaces();
  if (Nshared > 0) {
    interpGradSharedFace_gpu();
    integrationGradSharedFace_gpu();
  }

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  auto h_elem_dof_num = elem_data.element_dof_number.HostRead();

  // Multiply by inverse mass matrix
  for (int elType = 0; elType < elem_data.numElems.Size(); elType++) {
    int elemOffset = 0;
    for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    int dof_el = h_elem_dof_num[elemOffset];
    multInverse_gpu(h_numElems[elType], elemOffset, dof_el);
    // multInverse_gpu(h_numElems[elType], elemOffset, dof_el, vfes->GetNDofs(), *gradUp, num_equation_, dim_,
    // gpuArrays,
    //                 invMArray, posDofInvM);
  }
}

void Gradients::interpFaceData_gpu(const Vector &Up, int elType, int elemOffset, int elDof) {
  const double *d_x = Up.Read();  // Primitives!
  double *d_uk_el1 = uk_el1.Write();
  double *d_uk_el2 = uk_el2.Write();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  auto d_elemFaces = face_data.elemFaces.Read();
  auto d_elem_dofs_list = elem_data.element_dofs_list.Read();
  auto d_elem_dof_off = elem_data.element_dof_offset.Read();
  auto d_shape1 = face_data.face_el1_shape.Read();
  const double *d_shape2 = face_data.face_el2_shape.Read();
  auto d_face_el1 = face_data.face_el1.Read();
  auto d_face_nqp = face_data.face_num_quad.Read();

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
    double uk1[gpudata::MAXEQUATIONS];  // double uk1[20];
    double shape[gpudata::MAXDOFS];  // double shape[216];
    int indexes_i[gpudata::MAXDOFS];  // int indexes_i[216];

    const int eli = elemOffset + el;
    const int offsetEl1 = d_elem_dof_off[eli];
    const int elFaces = d_elemFaces[7 * eli];
    const int dof1 = elDof;

    for (int i = 0; i < elDof; i++) {
      int index = d_elem_dofs_list[offsetEl1 + i];
      indexes_i[i] = index;
    }

    // loop over faces
    for (int face = 0; face < elFaces; face++) {
      const int gFace = d_elemFaces[7 * eli + face + 1];
      const int Q = d_face_nqp[gFace];
      //int offsetShape1 = gFace * maxIntPoints * (maxDofs + 1 + dim);
      int offset_shape = gFace * maxIntPoints * maxDofs;

      // swapElems = false indicates that el is "element 1" for this face
      // swapElems = true  indicates that el is "element 2" for this face
      bool swapElems = false;
      if (eli != d_face_el1[gFace]) {
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
          for (int j = 0; j < dof1; j++) shape[j] = d_shape2[offset_shape + k * maxDofs + j];
        } else {
          for (int j = 0; j < dof1; j++) shape[j] = d_shape1[offset_shape + k * maxDofs + j];
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

void Gradients::computeGradients_gpu(const int elType, const int offsetElems, const int elDof) {
  const double *d_Up = Up->Read();
  double *d_gradUp = gradUp->ReadWrite();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  auto d_elem_dof_off = elem_data.element_dof_offset.Read();
  auto d_elem_dofs_list = elem_data.element_dofs_list.Read();

  auto d_Ke = Ke_array_.Read();
  auto d_Ke_pos = Ke_positions_.Read();

  const int numElems = h_numElems[elType];
  const int totalDofs = vfes->GetNDofs();

  const int num_equation = num_equation_;
  const int dim = dim_;

  // MFEM_FORALL(el, numElems, {
  MFEM_FORALL_2D(el, numElems, elDof, 1, 1, {
    const int eli = el + offsetElems;
    const int offsetIDs = d_elem_dof_off[eli];

    const int keoffset = d_Ke_pos[eli];

    int index_i[gpudata::MAXDOFS];                    // int index_i[216];
    MFEM_SHARED double Ui[gpudata::MAXDOFS],          // MFEM_SHARED double Ui[216],
        gradUpi[gpudata::MAXDOFS * gpudata::MAXDIM];  //  gradUpi[216 * 3];

    for (int i = 0; i < elDof; i++) {
      index_i[i] = d_elem_dofs_list[offsetIDs + i];
    }

    for (int eq = 0; eq < num_equation; eq++) {
      MFEM_FOREACH_THREAD(i, x, elDof) {
        Ui[i] = d_Up[index_i[i] + eq * totalDofs];
        for (int d = 0; d < dim; d++) {
          // this loads face contribution
          gradUpi[i + d * elDof] = d_gradUp[index_i[i] + eq * totalDofs + d * num_equation * totalDofs];
        }
      }
      MFEM_SYNC_THREAD;

      // add the element interior contribution
      MFEM_FOREACH_THREAD(j, x, elDof) {
        for (int d = 0; d < dim; d++) {
          for (int k = 0; k < elDof; k++) {
            gradUpi[j + d * elDof] += d_Ke[keoffset + dim * elDof * j + d * elDof + k] * Ui[k];
          }
        }
      }
      MFEM_SYNC_THREAD;

      // write to global memory
      MFEM_FOREACH_THREAD(i, x, elDof) {
        for (int d = 0; d < dim; d++) {
          d_gradUp[index_i[i] + eq * totalDofs + d * num_equation * totalDofs] = gradUpi[i + d * elDof];
        }
      }
      MFEM_SYNC_THREAD;
    }  // end equation loop
  });
}

// clang-format on
void Gradients::evalFaceIntegrand_gpu() {
  auto d_dun = dun_face.Write();
  const double *d_uk_el1 = uk_el1.Read();
  const double *d_uk_el2 = uk_el2.Read();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  auto d_weight = face_data.face_quad_weight.Read();
  auto d_normal = face_data.face_normal.Read();
  auto d_face_nqp = face_data.face_num_quad.Read();

  Mesh *mesh = fes->GetMesh();
  const int Nf = mesh->GetNumFaces();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  MFEM_FORALL(iface, Nf, {
    double u1[gpudata::MAXEQUATIONS], u2[gpudata::MAXEQUATIONS],
        nor[gpudata::MAXDIM];  // double u1[20], u2[20], nor[3];

    const int Q = d_face_nqp[iface];
    const int weight_offset = iface * maxIntPoints;
    const int normal_offset = iface * maxIntPoints * dim;

    for (int k = 0; k < Q; k++) {
      const double weight = d_weight[weight_offset + k];

      double du[gpudata::MAXEQUATIONS];  // double du[20];
      for (int eq = 0; eq < num_equation; eq++) {
        u1[eq] = d_uk_el1[eq + k * num_equation + iface * maxIntPoints * num_equation];
        u2[eq] = d_uk_el2[eq + k * num_equation + iface * maxIntPoints * num_equation];
        du[eq] = u2[eq] - u1[eq];
      }

      const int idx0 = iface * dim * maxIntPoints * num_equation + k * num_equation;
      for (int d = 0; d < dim; d++) {
        const int idx1 = idx0 + d * maxIntPoints * num_equation;
        nor[d] = weight * 0.5 * d_normal[normal_offset + k * dim + d];
        for (int eq = 0; eq < num_equation; eq++) {
          d_dun[eq + idx1] = du[eq] * nor[d];
        }
      }
    }  // end loop over integration points
  });  // end loop over faces
}

// clang-format on
void Gradients::faceContrib_gpu(const int elType, const int offsetElems, const int elDof) {
  const double *d_dun = dun_face.Read();

  double *d_gradUp = gradUp->Write();  // NB: I assume this comes in set to zero!

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  auto d_elem_dof_off = elem_data.element_dof_offset.Read();
  auto d_elem_dofs_list = elem_data.element_dofs_list.Read();

  // pointers for face integration
  auto d_elemFaces = face_data.elemFaces.Read();
  auto d_shape1 = face_data.face_el1_shape.Read();
  const double *d_shape2 = face_data.face_el2_shape.Read();
  auto d_face_el1 = face_data.face_el1.Read();
  auto d_face_nqp = face_data.face_num_quad.Read();

  const int numElems = h_numElems[elType];
  const int totalDofs = vfes->GetNDofs();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;

  MFEM_FORALL_2D(el, numElems, elDof, 1, 1, {
    const int eli = el + offsetElems;
    const int offsetIDs = d_elem_dof_off[eli];

    MFEM_FOREACH_THREAD(i, x, elDof) {
      const int idx = d_elem_dofs_list[offsetIDs + i];
      double const *shape;

      // ================  FACE CONTRIBUTION  ================
      const int elFaces = d_elemFaces[7 * eli];
      for (int face = 0; face < elFaces; face++) {
        const int gFace = d_elemFaces[7 * eli + face + 1];
        const int Q = d_face_nqp[gFace];
        const int offset_shape = gFace * maxIntPoints * maxDofs;
        bool swapElems = false;

        // get neighbor
        int elj = d_face_el1[gFace];
        if (elj != eli) {
          swapElems = true;
        }

        for (int k = 0; k < Q; k++) {
          // load interpolators
          if (swapElems) {
            shape = d_shape2 + offset_shape + k * maxDofs;
          } else {
            shape = d_shape1 + offset_shape + k * maxDofs;
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

void Gradients::interpGradSharedFace_gpu() {
  const double *d_up = Up->Read();
  const double *d_faceData = transferUp->face_nbr_data.Read();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  const int *d_elem_dofs_list = elem_data.element_dofs_list.Read();
  const int *d_elem_dof_off = elem_data.element_dof_offset.Read();
  const int *d_elem_dof_num = elem_data.element_dof_number.Read();

  const sharedFaceIntegrationData &parallelData = gpuArrays.shared_face_data;
  const double *d_weight = parallelData.face_quad_weight.Read();
  const double *d_normal = parallelData.face_normal.Read();
  const double *d_shape1 = parallelData.face_el1_shape.Read();
  const double *d_shape2 = parallelData.face_el2_shape.Read();
  const int *d_face_num_quad = parallelData.face_num_quad.Read();
  const int *d_face_num_dof2 = parallelData.face_num_dof2.Read();
  const int *d_elem2_dofs = parallelData.elem2_dofs.Read();
  const int *d_sharedElemsFaces = parallelData.sharedElemsFaces.Read();

  double *d_dun = dun_shared_face.Write();

  const int maxNumElems = parallelData.sharedElemsFaces.Size() / 7;  // elements with shared faces
  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;
  const int Ndofs = vfes->GetNDofs();

  MFEM_FORALL_2D(el, maxNumElems, maxIntPoints, 1, 1, {
    double l1[gpudata::MAXDOFS], l2[gpudata::MAXDOFS],
        nor[gpudata::MAXDIM];  // double l1[216], l2[216], nor[3];
    double u1, u2;
    int index_i[gpudata::MAXDOFS];  // int index_i[216];

    const int el1 = d_sharedElemsFaces[0 + el * 7];
    const int numFaces = d_sharedElemsFaces[1 + el * 7];
    const int dof1 = d_elem_dof_num[el1];

    const int offsetEl1 = d_elem_dof_off[el1];

    for (int i = 0; i < dof1; i++) {
      index_i[i] = d_elem_dofs_list[offsetEl1 + i];
    }

    for (int elFace = 0; elFace < numFaces; elFace++) {
      const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
      const int dof2 = d_face_num_dof2[f];
      const int Q = d_face_num_quad[f];

      // begin loop through integration points
      // for (int k = 0; k < Q; k++) {
      MFEM_FOREACH_THREAD(k, x, Q) {
        // load interpolating functions
        for (int i = 0; i < dof1; i++) {
          l1[i] = d_shape1[f * maxIntPoints * maxDofs + k * maxDofs + i];
        }
        for (int i = 0; i < dof2; i++) {
          l2[i] = d_shape2[f * maxIntPoints * maxDofs + k * maxDofs + i];
        }

        const double weight = d_weight[f * maxIntPoints + k];

        for (int d = 0; d < dim; d++) {
          nor[d] = d_normal[f * maxIntPoints * dim + k * dim + d];
        }

        // set array for interpolated data to 0
        for (int eq = 0; eq < num_equation; eq++) {
          u1 = u2 = 0.;

          // load data for elem1
          for (int j = 0; j < dof1; j++) {
            u1 += d_up[index_i[j] + eq * Ndofs] * l1[j];
          }

          // load data elem2
          for (int j = 0; j < dof2; j++) {
            int index = d_elem2_dofs[j + eq * maxDofs + f * num_equation * maxDofs];
            u2 += d_faceData[index] * l2[j];
          }

          // TODO(kevin): MAXNUMFACE
          const int idx = dim * (eq + k * num_equation + elFace * maxIntPoints * num_equation +
                                 el * 5 * maxIntPoints * num_equation);
          for (int d = 0; d < dim; d++) {
            d_dun[idx + d] = 0.5 * (u2 - u1) * nor[d] * weight;
          }
        }
      }  // end loop through integration points
    }
  });
}

void Gradients::integrationGradSharedFace_gpu() {
  double *d_gradUp = gradUp->ReadWrite();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  const int *d_elem_dofs_list = elem_data.element_dofs_list.Read();
  const int *d_elem_dof_off = elem_data.element_dof_offset.Read();
  const int *d_elem_dof_num = elem_data.element_dof_number.Read();

  const sharedFaceIntegrationData &parallelData = gpuArrays.shared_face_data;
  const double *d_shape1 = parallelData.face_el1_shape.Read();
  const int *d_face_num_quad = parallelData.face_num_quad.Read();
  const int *d_sharedElemsFaces = parallelData.sharedElemsFaces.Read();

  const double *d_dun = dun_shared_face.Read();

  const int dim = dim_;
  const int num_equation = num_equation_;
  const int maxIntPoints = maxIntPoints_;
  const int maxDofs = maxDofs_;
  const int Ndofs = vfes->GetNDofs();

  MFEM_FORALL_2D(el, parallelData.sharedElemsFaces.Size() / 7, maxDofs, 1, 1, {  // NOLINT
    // double l1[216];

    const int el1 = d_sharedElemsFaces[0 + el * 7];
    const int numFaces = d_sharedElemsFaces[1 + el * 7];
    const int dof1 = d_elem_dof_num[el1];
    const int offsetEl1 = d_elem_dof_off[el1];

    MFEM_FOREACH_THREAD(i, x, dof1) {
      const int indexi = d_elem_dofs_list[offsetEl1 + i];

      for (int elFace = 0; elFace < numFaces; elFace++) {
        const int f = d_sharedElemsFaces[1 + elFace + 1 + el * 7];
        const int Q = d_face_num_quad[f];

        for (int k = 0; k < Q; k++) {
          const double l1 = d_shape1[f * maxIntPoints * maxDofs + k * maxDofs + i];

          // add contribution
          for (int eq = 0; eq < num_equation; eq++) {
            // TODO(kevin): MAXNUMFACE
            const int idxR = dim * (eq + k * num_equation + elFace * maxIntPoints * num_equation +
                                    el * 5 * maxIntPoints * num_equation);
            for (int d = 0; d < dim; d++) {
              const int idxL = eq * Ndofs + d * num_equation * Ndofs;
              d_gradUp[indexi + idxL] += d_dun[idxR + d] * l1;
            }
          }  // end integration loop
        }
      }
    }
  });
}

void Gradients::multInverse_gpu(const int numElems, const int offsetElems, const int elDof) {
  double *d_gradUp = gradUp->ReadWrite();

  const elementIndexingData &elem_data = gpuArrays.element_indexing_data;
  const interiorFaceIntegrationData &face_data = gpuArrays.interior_face_data;

  auto d_elem_dof_off = elem_data.element_dof_offset.Read();
  auto d_elem_dofs_list = elem_data.element_dofs_list.Read();
  const double *d_invMArray = invMArray.Read();
  auto d_posDofInvM = posDofInvM.Read();

  const int totalDofs = vfes->GetNDofs();
  const int dim = dim_;
  const int num_equation = num_equation_;

  MFEM_FORALL_2D(el, numElems, elDof, 1, 1, {
    MFEM_SHARED double gradUpi[gpudata::MAXDOFS * gpudata::MAXDIM];  // MFEM_SHARED double gradUpi[216 * 3];

    const int eli = el + offsetElems;
    const int offsetIDs = d_elem_dof_off[eli];
    const int offsetInv = d_posDofInvM[2 * eli];

    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) {
        MFEM_FOREACH_THREAD(i, x, elDof) {
          const int indexi = d_elem_dofs_list[offsetIDs + i];
          gradUpi[i + d * elDof] = d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs];
        }
        MFEM_SYNC_THREAD;

        MFEM_FOREACH_THREAD(i, x, elDof) {
          const int indexi = d_elem_dofs_list[offsetIDs + i];
          double temp = 0;
          for (int n = 0; n < elDof; n++) {
            temp += gradUpi[n + d * elDof] * d_invMArray[offsetInv + i * elDof + n];
          }
          d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs] = temp;
        }
        MFEM_SYNC_THREAD;
      }
    }
  });
}
#endif
