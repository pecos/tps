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
#ifndef GRADIENTS_HPP_
#define GRADIENTS_HPP_

// Class to manage gradients of primitives

#include <tps_config.h>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "gradNonLinearForm.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

class Gradients : public ParNonlinearForm {
 private:
  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *gradUpfes;
  const int dim_;
  const int num_equation_;

  ParGridFunction *Up;
  ParGridFunction *gradUp;

  GasMixture *mixture;

  // ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;

  IntegrationRules *intRules;
  const int intRuleType;

  const precomputedIntegrationData &gpuArrays;
  Vector uk_el1;
  Vector uk_el2;
  Vector dun_face;

  const int *h_numElems;
  const int *h_posDofIds;

  // DenseMatrix *Me_inv;
  Array<DenseMatrix *> &Me_inv;
  Array<DenseMatrix *> Ke;
  Vector Ke_array_;
  Array<int> Ke_positions_;

  Vector &invMArray;
  Array<int> &posDofInvM;

  const int &maxIntPoints_;
  const int &maxDofs_;

  // gradients of shape functions for all nodes and weight multiplied by det(Jac)
  // at each integration point
  //   Vector elemShapeDshapeWJ; // [...l_0(i),...,l_dof(i),l_0_x(i),...,l_dof_d(i), w_i*detJac_i ...]
  //   Array<int> elemPosQ_shapeDshapeWJ; // position and num. of integration points for each element

  parallelFacesIntegrationArrays *parallelData;
  dataTransferArrays *transferUp;

  Vector dun_shared_face;

 public:
  Gradients(ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_gradUpfes, int _dim, int _num_equation,
            ParGridFunction *_Up, ParGridFunction *_gradUp, GasMixture *_mixture, GradNonLinearForm *_gradUp_A,
            IntegrationRules *_intRules, int _intRuleType, const precomputedIntegrationData &gpuArrays,
            Array<DenseMatrix *> &Me_inv, Vector &_invMArray, Array<int> &_posDofInvM, const int &_maxIntPoints,
            const int &_maxDofs);

  ~Gradients();

  void setParallelData(parallelFacesIntegrationArrays *_parData, dataTransferArrays *_transferUp) {
    parallelData = _parData;
    transferUp = _transferUp;

    dun_shared_face.UseDevice(true);

    int maxNumElems = parallelData->sharedElemsFaces.Size() / 7;  // elements with shared faces
    dun_shared_face.SetSize(dim_ * maxNumElems * 5 * maxIntPoints_ * num_equation_);
  }

  void computeGradients();

#ifdef _GPU_
  void computeGradients_domain();
  void computeGradients_bdr();

  void computeGradients_gpu(const int elType, const int offsetElems, const int elDof);

  void faceContrib_gpu(const int elType, const int offsetElems, const int elDof);

  void interpGradSharedFace_gpu();

  void integrationGradSharedFace_gpu();

  void multInverse_gpu(const int numElems, const int offsetElems, const int elDof);

  void interpFaceData_gpu(const Vector &x, int elType, int elemOffset, int elDof);
  void evalFaceIntegrand_gpu();
#endif
};

#endif  // GRADIENTS_HPP_
