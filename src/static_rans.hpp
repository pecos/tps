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
#ifndef STATIC_RANS_HPP_
#define STATIC_RANS_HPP_
/** @file
 * @brief Contains class implementing a static constant eddy viscosity as a RANS model
 */

#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "turb_model_base.hpp"
#include "externalData_base.hpp"

/**
 * Provides RANS models to compute eddy viscosity
 * to be used for time split schemes, such as those available through
 * LoMachSolver
 */
class AlgebraicRans : public TurbModelBase {
 protected:
  mfem::ParMesh *pmesh_ = nullptr;
  int order_;
  int dim_;
  bool axisym_;

  mfem::FiniteElementCollection *sfec_ = nullptr;
  mfem::ParFiniteElementSpace *sfes_ = nullptr;

  mfem::ParGridFunction *mut_ = nullptr;
  mfem::ParGridFunction *nut_field_ = nullptr;

  ExternalDataBase *extData_ = nullptr;

 public:
  /// Constructor
  //  StaticRans(mfem::Mesh *smesh, mfem::ParMesh *pmesh, const mfem::Array<int> &partitioning, int order, TPS::Tps
  //  *tps);
  StaticRans(mfem::ParMesh *pmesh, const mfem::Array<int> &partitioning, int order, TPS::Tps *tps);

  /// Destructor
  virtual ~StaticRans();

  /**
   * @brief Allocate the eddy viscosity
   *
   * Initialized from constant file using gaussian interpolation
   */
  void initializeSelf() override;

  /**
   * @brief Add eddy viscosity and distance function to the visualization output
   */
  void initializeViz(mfem::ParaViewDataCollection &pvdc) override;

  /**
   * @brief Initialize the eddy viscosity.
   *
   * Compute the eddy viscosity by calling "step".
   */
  void initializeOperators() override { this->step(); }

  /**
   * @brief Compute the current eddy viscosity using the current velocity field.
   */
  void step() override;

  /**
   * @brief No-op for this class
   */
  void setup() override {}

  mfem::ParGridFunction *getCurrentEddyViscosity() override { return mut_; }
};

#endif  // ALGEBRAIC_RANS_HPP_
