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
#ifndef ALGEBRAIC_RANS_HPP_
#define ALGEBRAIC_RANS_HPP_
/** @file
 * @brief Contains class implementing algebraic RANS model capabilities
 */

#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "turb_model_base.hpp"

/**
 * Provides algebraic RANS models to compute eddy viscosity
 * to be used for time split schemes, such as those available through
 * LoMachSolver
 */
class AlgebraicRans : public TurbModelBase {
 protected:
  mfem::ParMesh *pmesh_ = nullptr;
  int order_;
  int dim_;
  bool axisym_;

  double max_mixing_length_;
  double kappa_von_karman_;

  mfem::FiniteElementCollection *sfec_ = nullptr;
  mfem::ParFiniteElementSpace *sfes_ = nullptr;

  mfem::FiniteElementCollection *vfec_ = nullptr;
  mfem::ParFiniteElementSpace *vfes_ = nullptr;

  mfem::ParGridFunction *vorticity_gf_ = nullptr;
  mfem::ParGridFunction *swirl_vorticity_gf_ = nullptr;

  mfem::ParGridFunction *mut_ = nullptr;
  mfem::ParGridFunction *distance_ = nullptr;
  mfem::ParGridFunction *ell_mix_gf_ = nullptr;

  // Only used by filter
  bool filter_mut_ = false;
  int filter_p_ = 1;
  mfem::FiniteElementCollection *sfec_filter_ = nullptr;
  mfem::ParFiniteElementSpace *sfes_filter_ = nullptr;
  mfem::ParGridFunction *low_order_mut_ = nullptr;

 public:
  /// Constructor
  //  AlgebraicRans(mfem::Mesh *smesh, mfem::ParMesh *pmesh, const mfem::Array<int> &partitioning, int order, TPS::Tps
  //  *tps);
  AlgebraicRans(mfem::ParMesh *pmesh, const mfem::Array<int> &partitioning, int order, TPS::Tps *tps,
                mfem::ParGridFunction *distance);

  /// Destructor
  virtual ~AlgebraicRans();

  /**
   * @brief Allocate the eddy viscosity
   *
   * Initialized to zero b/c the required interface fields are cannot be assumed valid yet.
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
