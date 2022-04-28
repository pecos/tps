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
#ifndef DGNONLINEARFORM_HPP_
#define DGNONLINEARFORM_HPP_

#include <tps_config.h>

#include <mfem.hpp>
#include <mfem/fem/nonlinearform.hpp>

#include "BCintegrator.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"

using namespace mfem;
using namespace std;

// Derived non linear form that re-implements the Mult method
// so that we can use GPU
class DGNonLinearForm : public ParNonlinearForm {
 private:
  Fluxes *fluxes;
  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *gradFes;

  ParGridFunction *gradUp;

  BCintegrator *bcIntegrator;

  IntegrationRules *intRules;
  const int dim;
  const int num_equation;
  GasMixture *mixture;

  const volumeFaceIntegrationArrays &gpuArrays;

  const int *h_numElems;
  const int *h_posDofIds;

  // Parallel shared faces integration
  parallelFacesIntegrationArrays *parallelData;
  mutable dataTransferArrays *transferU;
  mutable dataTransferArrays *transferGradUp;

  const int &maxIntPoints;
  const int &maxDofs;

  Vector uk_el1, grad_upk_el1;
  Vector uk_el2, grad_upk_el2;
  Vector face_flux_;

  Vector shared_uk_el1, shared_grad_upk_el1;
  Vector shared_uk_el2, shared_grad_upk_el2;

 public:
  DGNonLinearForm(Fluxes *_flux, ParFiniteElementSpace *f, ParFiniteElementSpace *gradFes, ParGridFunction *_gradUp,
                  BCintegrator *_bcIntegrator, IntegrationRules *intRules, const int dim, const int num_equation,
                  GasMixture *mixture, const volumeFaceIntegrationArrays &_gpuArrays, const int &maxIntPoints,
                  const int &maxDofs);

  void Mult(const Vector &x, Vector &y);

  void setParallelData(parallelFacesIntegrationArrays *_parallelData, dataTransferArrays *_transferU,
                       dataTransferArrays *_transferGradUp);

  static void faceIntegration_gpu(Vector &y, Vector &face_flux,
                                  Vector &uk_el1, Vector &uk_el2, Vector &grad_uk_el1, Vector &grad_uk_el2,
                                  const ParGridFunction *gradUp, Fluxes *flux, const int &Ndofs, const int &Nf,
                                  const int &NumElemsType, const int &elemOffset, const int &elDof, const int &dim,
                                  const int &num_equation, GasMixture *mixture,
                                  const volumeFaceIntegrationArrays &gpuArrays, const int &maxIntPoints,
                                  const int &maxDofs);

  static void sharedFaceIntegration_gpu(const Vector &x, const Vector &faceU, const ParGridFunction *gradUp,
                                        const Vector &faceGradUp, Vector &y, Vector &shared_uk_el1,
                                        Vector &shared_uk_el2, Vector &shared_grad_upk_el1, Vector &shared_grad_upk_el2,
                                        const int &Ndofs, const int &dim, const int &num_equation, GasMixture *mixture,
                                        Fluxes *flux, const volumeFaceIntegrationArrays &gpuArrays,
                                        const parallelFacesIntegrationArrays *parallelData, const int &maxIntPoints,
                                        const int &maxDofs);

  static void interpFaceData_gpu(const Vector &x, Vector &uk_el1, Vector &uk_el2, Vector &grad_uk_el1,
                                 Vector &grad_uk_el2, const ParGridFunction *gradUp, const int &Ndofs, const int &Nf,
                                 const int &NumElemsType, const int &elemOffset, const int &elDof, const int &dim,
                                 const int &num_equation, const double &gamma, const double &Rg, const double &viscMult,
                                 const double &bulkViscMult, const double &Pr,
                                 const volumeFaceIntegrationArrays &gpuArrays, const int &maxIntPoints,
                                 const int &maxDofs);
  static void sharedFaceInterpolation_gpu(const Vector &x, const Vector &faceU, const ParGridFunction *gradUp,
                                          const Vector &faceGradUp, Vector &shared_uk_el1, Vector &shared_uk_el2,
                                          Vector &shared_grad_upk_el1, Vector &shared_grad_upk_el2, const int &Ndofs,
                                          const int &dim, const int &num_equation, GasMixture *mixture,
                                          const volumeFaceIntegrationArrays &gpuArrays,
                                          const parallelFacesIntegrationArrays *parallelData, const int &maxIntPoints,
                                          const int &maxDofs);
  void evalFaceFlux_gpu(Vector &uk_el1, Vector &uk_el2, Vector &grad_uk_el1, Vector &grad_uk_el2,
                        Fluxes *flux,
                        const int &Ndofs, const int &Nf, const int &NumElemType,
                        const int &elemOffset, const int &elDof, const int &dim,
                        const int &num_equation, GasMixture *mixture,
                        const volumeFaceIntegrationArrays &gpuArrays, const int &maxIntPoints,
                        const int &maxDofs);


#ifdef _GPU_
  void Mult_domain(const Vector &x, Vector &y);
  void Mult_bdr(const Vector &x, Vector &y);
  static void setToZero_gpu(Vector &x, const int size);
#endif
};

#endif  // DGNONLINEARFORM_HPP_
