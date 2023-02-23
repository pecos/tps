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
#ifndef DGNONLINEARFORM_HPP_
#define DGNONLINEARFORM_HPP_

#include <tps_config.h>

#include <mfem/fem/nonlinearform.hpp>

#include "BCintegrator.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

// Derived non linear form that re-implements the Mult method
// so that we can use GPU
class DGNonLinearForm : public ParNonlinearForm {
 private:
  RiemannSolver *rsolver_;
  Fluxes *fluxes;
  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *gradFes;

  ParGridFunction *gradUp_;

  BCintegrator *bcIntegrator;

  IntegrationRules *intRules;
  const int dim_;
  const int num_equation_;
  GasMixture *mixture;

  const volumeFaceIntegrationArrays &gpuArrays;

  // Parallel shared faces integration
  parallelFacesIntegrationArrays *parallelData;
  mutable dataTransferArrays *transferU;
  mutable dataTransferArrays *transferGradUp;

  const int &maxIntPoints_;
  const int &maxDofs_;

  Vector uk_el1, grad_upk_el1;
  Vector uk_el2, grad_upk_el2;
  Vector face_flux_;

  Vector shared_flux;

 public:
  DGNonLinearForm(RiemannSolver *rsolver, Fluxes *_flux, ParFiniteElementSpace *f, ParFiniteElementSpace *gradFes,
                  ParGridFunction *_gradUp, BCintegrator *_bcIntegrator, IntegrationRules *intRules, const int dim,
                  const int num_equation, GasMixture *mixture, const volumeFaceIntegrationArrays &_gpuArrays,
                  const int &maxIntPoints, const int &maxDofs);

  void Mult(const Vector &x, Vector &y);

  void setParallelData(parallelFacesIntegrationArrays *_parallelData, dataTransferArrays *_transferU,
                       dataTransferArrays *_transferGradUp);

  void faceIntegration_gpu(Vector &y, int elType, int elemOffset, int elDof);

  void sharedFaceIntegration_gpu(Vector &y);

  void interpFaceData_gpu(const Vector &x, int elType, int elemOffset, int elDof);

  void sharedFaceInterpolation_gpu(const Vector &x);
  void evalFaceFlux_gpu();

#ifdef _GPU_
  void Mult_domain(const Vector &x, Vector &y);
  void Mult_bdr(const Vector &x, Vector &y);
  static void setToZero_gpu(Vector &x, const int size);
#endif
};

#endif  // DGNONLINEARFORM_HPP_
