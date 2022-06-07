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
#ifndef FACE_INTEGRATOR_HPP_
#define FACE_INTEGRATOR_HPP_

#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "fluxes.hpp"
#include "riemann_solver.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;

// TODO(kevin): In order to avoid repeated primitive variable evaluation,
// Fluxes and RiemannSolver should take Vector Up (on the evaulation point) as input argument,
// and FaceIntegrator should have a pointer to ParGridFunction *Up.
// Also should be able to have Up more than number of equations,
// while gradUp is evaluated only for the first num_equation variables.
// Need to discuss further.

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator {
 private:
  RiemannSolver *rsolver;
  Fluxes *fluxClass;
  ParFiniteElementSpace *vfes;

  const int dim;
  const int num_equation;

  double &max_char_speed;

  const ParGridFunction *gradUp;
  const ParFiniteElementSpace *gradUpfes;

  IntegrationRules *intRules;

  DenseMatrix *faceMassMatrix1, *faceMassMatrix2;
  int faceNum;
  bool faceMassMatrixComputed;
  bool useLinear;
  const bool axisymmetric_;

  int totDofs;
  Array<int> vdofs1;
  Array<int> vdofs2;
  Vector shape1;
  Vector shape2;
  Vector funval1;
  Vector funval2;
  Vector nor;
  Vector fluxN;
  // DenseMatrix gradUp1;
  // DenseMatrix gradUp2;
  DenseTensor gradUp1;
  DenseTensor gradUp2;
  DenseMatrix gradUp1i;
  DenseMatrix gradUp2i;
  DenseMatrix viscF1;
  DenseMatrix viscF2;
  DenseMatrix elfun1_mat;
  DenseMatrix elfun2_mat;
  DenseMatrix elvect1_mat;
  DenseMatrix elvect2_mat;

  void getElementsGrads_cpu(FaceElementTransformations &Tr, const FiniteElement &el1, const FiniteElement &el2,
                            DenseTensor &gradUp1, DenseTensor &gradUp2);

  void NonLinearFaceIntegration(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                const Vector &elfun, Vector &elvect);

  void MassMatrixFaceIntegral(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                              const Vector &elfun, Vector &elvect);

 public:
  FaceIntegrator(IntegrationRules *_intRules, RiemannSolver *rsolver_, Fluxes *_fluxClass, ParFiniteElementSpace *_vfes,
                 bool _useLinear, const int _dim, const int _num_equation, ParGridFunction *_gradUp,
                 ParFiniteElementSpace *_gradUpfes, double &_max_char_speed, bool axisym);
  ~FaceIntegrator();

  virtual void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);

  static void getElementsGrads_gpu(const ParGridFunction *gradUp, ParFiniteElementSpace *vfes,
                                   const ParFiniteElementSpace *gradUpfes, FaceElementTransformations &Tr,
                                   const FiniteElement &el1, const FiniteElement &el2, DenseTensor &gradUp1,
                                   DenseTensor &gradUp2, const int &num_equation, const int &totalDofs, const int &dim);
};

#endif  // FACE_INTEGRATOR_HPP_
