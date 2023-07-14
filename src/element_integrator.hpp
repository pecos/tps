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
#ifndef ELEMENT_INTEGRATOR_HPP_
#define ELEMENT_INTEGRATOR_HPP_

#include <tps_config.h>

#include <mfem.hpp>

#include "fluxes.hpp"
#include "source_fcn.hpp"

// Element interior term: <grad w, F>
class ElementIntegrator : public mfem::NonlinearFormIntegrator {
 private:
  const Equations eqSys_;
  const int dim_;
  const int num_eqn_;
  const bool axisym_;

  Fluxes *flux_;
  SourceFunction *srcFcn_;
  mfem::IntegrationRules *int_rules_;

  // NB: Only necessary b/c of viscous scheme.  Can eliminate if we switch to SIPG.
  mfem::ParFiniteElementSpace *vfes_;
  mfem::ParGridFunction *gradUp_;

  // Distance function
  mfem::ParGridFunction *distance_;

  Array<int> vdofs_;

  void getElementGrad(const int elemNo, const FiniteElement &el, DenseTensor &gradUpElem);

 public:
  ElementIntegrator(int dim, int num_eqn, bool axisym, Equations eqSys, Fluxes *flux, SourceFunction *srcFcn,
                    mfem::IntegrationRules *int_rules, mfem::ParFiniteElementSpace *vfes,
                    mfem::ParGridFunction *gradUp, mfem::ParGridFunction *distance);

  virtual void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                                     const mfem::Vector &elfun, mfem::Vector &elvec);

  virtual void AssembleElementGrad(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                                   const mfem::Vector &elfun, mfem::DenseMatrix &elmat);
};

#endif  // ELEMENT_INTEGRATOR_HPP_
