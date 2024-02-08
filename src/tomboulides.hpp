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
#ifndef TOMBOULIDES_HPP_
#define TOMBOULIDES_HPP_
/** @file
 *
 * @brief Implements the time marching scheme from Tomboulides et al,
 * Numerical simulation of low Mach number reactive flows,
 * J. Sci. Comput, 12(2), 1997.
 *
 */

#include "split_flow_base.hpp"

class Tomboulides : public FlowBase {
 protected:
  mfem::ParMesh *pmesh_;
  const int vorder_;
  const int porder_;
  const int dim_;

  /// Velocity FEM objects and fields
  mfem::FiniteElementCollection *vfec_ = nullptr;
  mfem::ParFiniteElementSpace *vfes_ = nullptr;
  mfem::ParGridFunction *u_curr_gf_ = nullptr;
  mfem::ParGridFunction *u_next_gf_ = nullptr;

  /// Pressure \f$H^1\f$ finite element collection.
  mfem::FiniteElementCollection *pfec_ = nullptr;
  mfem::ParFiniteElementSpace *pfes_ = nullptr;
  mfem::ParGridFunction *p_gf_ = nullptr;

  /// Allocate state ParGridFunction objects and initialize interface
  void allocateState();

 public:
  /// Constructor
  Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder);

  /// Destructor
  ~Tomboulides() final;

  /// Initialize operators and fields
  void initializeSelf() final;

  /// Advance
  void step() final {}
};

#endif  // TOMBOULIDES_HPP_
