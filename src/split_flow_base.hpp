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
#ifndef SPLIT_FLOW_BASE_HPP_
#define SPLIT_FLOW_BASE_HPP_
/** @file
 * @brief Contains base class and simplest possible variant of flow model
 */

#include "tps_mfem_wrap.hpp"

struct thermoChemToFlow;

struct timeCoefficients {
  // Current time
  double time;

  // Time step
  double dt;

  // "Adams-Bashforth" coefficients (explicit)
  double ab1;
  double ab2;
  double ab3;

  // "Backward difference" coefficients (implicit)
  double bd0;
  double bd1;
  double bd2;
  double bd3;
};

struct flowToThermoChem {
  const mfem::ParGridFunction *velocity = nullptr;

  bool swirl_supported = false;
  const mfem::ParGridFunction *swirl = nullptr;
};

class FlowBase {
 protected:
  const thermoChemToFlow *thermo_interface_;

 public:
  /// Destructor
  virtual ~FlowBase() {}

  virtual void initializeSelf() = 0;

  virtual void step() = 0;

  void initializeFromThermoChem(thermoChemToFlow *thermo) { thermo_interface_ = thermo; }

  virtual void initializeOperators() {}

  /// Interface object, provides fields necessary for the thermochemistry model
  flowToThermoChem interface;

  /// Get interface provided by thermo model
  const thermoChemToFlow *getThermoInterface() const { return thermo_interface_; }
};

class ZeroFlow : public FlowBase {
 protected:
  mfem::ParMesh *pmesh_;
  const int vorder_;
  const int dim_;

  mfem::FiniteElementCollection *fec_ = nullptr;
  mfem::ParFiniteElementSpace *fes_ = nullptr;
  mfem::ParGridFunction *velocity_ = nullptr;

 public:
  /// Constructor
  ZeroFlow(mfem::ParMesh *pmesh, int vorder);

  /// Destructor
  ~ZeroFlow() final;

  void initializeSelf() final;

  /// Velocity is always zero, so nothing to do
  void step(){};
};

#endif  // SPLIT_FLOW_BASE_HPP_
