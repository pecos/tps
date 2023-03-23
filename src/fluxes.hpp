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
#ifndef FLUXES_HPP_
#define FLUXES_HPP_

/*
 * Class that deals with flux operations
 */

#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace mfem;

// TODO(kevin): In order to avoid repeated primitive variable evaluation,
// Fluxes and RiemannSolver should take Vector Up (on the evaulation point) as input argument,
// and FaceIntegrator should have a pointer to ParGridFunction *Up.
// Also should be able to have Up more than number of equations,
// while gradUp is evaluated only for the first num_equation variables.
// Need to discuss further.
class Fluxes {
 private:
  GasMixture *mixture;
  Equations eqSystem;
  RunConfiguration *config_;
  TransportProperties *transport;

  int nvel;
  const int dim;
  const bool axisymmetric_;
  const int num_equation;

  void viscSpongePlanar(Vector x, double &wgt);

 public:
  MFEM_HOST_DEVICE Fluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                          //                          const int _num_equation, const int _dim, bool axisym);
                          const int _num_equation, const int _dim, bool axisym, RunConfiguration *_config = NULL);

  Equations GetEquationSystem() { return eqSystem; }

#ifdef _BUILD_DEPRECATED_
  void ComputeTotalFlux(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux);
#endif

  void ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux);
  MFEM_HOST_DEVICE void ComputeConvectiveFluxes(const double *state, double *flux) const;

  void ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, double radius, Vector transip, double delta,
                            DenseMatrix &flux);

  MFEM_HOST_DEVICE void ComputeViscousFluxes(const double *state, const double *gradUp, double radius, Vector transip,
                                             double delta, double *flux);

  void sgsSmag(const Vector &state, const DenseMatrix &gradUp, double delta, double &mu_sgs);
  void sgsSigma(const Vector &state, const DenseMatrix &gradUp, double delta, double &mu_sgs);

  // Compute viscous flux with prescribed boundary flux.
  void ComputeBdrViscousFluxes(const Vector &state, const DenseMatrix &gradUp, double radius, Vector transip,
                               double delta, const BoundaryViscousFluxData &bcFlux, Vector &normalFlux);

  MFEM_HOST_DEVICE void ComputeBdrViscousFluxes(const double *state, const double *gradUp, double radius,
                                                Vector transip, double delta, const BoundaryViscousFluxData &bcFlux,
                                                double *normalFlux);

  // Compute the split fersion of the flux for SBP operations
  // Output matrices a_mat, c_mat need not have the right size
  void ComputeSplitFlux(const Vector &state, DenseMatrix &a_mat, DenseMatrix &c_mat);

  MFEM_HOST_DEVICE bool isAxisymmetric() const { return axisymmetric_; }
};

#endif  // FLUXES_HPP_
