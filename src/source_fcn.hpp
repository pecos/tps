// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2023, The PECOS Development Team, University of Texas at Austin
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
#ifndef SOURCE_FCN_HPP_
#define SOURCE_FCN_HPP_

/** @file
 * Pointwise evaluation of source terms.
 */

#include <tps_config.h>

#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

/** \brief Base class for source functions.
 *
 * Provides for pointwise evaluation of source functions that depend
 * on position, conserved state, and primitive state gradients.  These
 * are called from class ElementIntegrator to build the contribution
 * to the residual from the source terms.
 */
class SourceFunction {
 protected:
  const int dim_;
  const int nvel_;
  const int neqn_;
  const Equations eqSys_;

 public:
  /// Constructor
  SourceFunction(const int dim, const int nvel, const int num_equation, const Equations eqSys);

  /// Destructor
  ~SourceFunction();

  /** \brief Evaluate the source term
   *
   * @param[in]  x      Spatial coordinates of point
   * @param[in]  U      Conserved state vector at point
   * @param[in]  gradUp Gradient of primitive variables at point
   * @param[out] S      Source term vector evaluated at point
   */
  virtual void evaluate(const mfem::Vector& x, const mfem::Vector& U, const mfem::DenseMatrix& gradUp,
                        mfem::Vector& S) = 0;
};

class AxisymmetricSourceFunction : public SourceFunction {
 protected:
  GasMixture* mixture_;
  TransportProperties* transport_;

 public:
  AxisymmetricSourceFunction(const int dim, const int nvel, const int num_equation, const Equations eqSys,
                             GasMixture* mix, TransportProperties* trans);
  ~AxisymmetricSourceFunction();

  void evaluate(const mfem::Vector& x, const mfem::Vector& U, const mfem::DenseMatrix& gradUp,
                mfem::Vector& S) override;
};

#endif  // SOURCE_FCN_HPP_
