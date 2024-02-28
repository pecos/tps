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
#ifndef EXTERNAL_DATA_BASE_HPP_
#define EXTERNAL_DATA_BASE_HPP_
/** @file
 * @brief Contains base class and simplest possible variant of external data
 */

#include "tps_mfem_wrap.hpp"

/**
 * Provides wrapper for fields that need to be provided by the
 * ext data to the flow.
 */
struct extDataToFlow {
  const mfem::ParGridFunction *Udata = nullptr;
};

/**
 * Provides wrapper for fields that need to be provided by the
 * ext data to the thermo chem model.
 */
struct extDataToThermoChem {
  const mfem::ParGridFunction *Tdata = nullptr;
};

/**
 * Provides interface for ext data implementation
 */
class ExternalDataBase {
 protected:
 public:
  /// Destructor
  virtual ~ExternalDataBase() {}

  /**
   * @brief Initialize the model-owned data
   *
   * Initialize any fields that this class owns.
   */
  virtual void initializeSelf() = 0;

  /**
   * @brief Hook to let derived classes register visualization fields with ParaViewDataCollection
   */
  virtual void initializeViz(mfem::ParaViewDataCollection &pvdc) {}
  
  /**
   * @brief Take a single time step
   */
  virtual void step() = 0;

  /// Initialize sponge field, all thats necessary for static fields
  virtual void setup() = 0;

  /// Interface object, provides fields necessary for the flow
  extDataToFlow toFlow_interface_;

  /// Interface object, provides fields necessary for the turbModel
  extDataToThermoChem toThermoChem_interface_;
};

/**
 * Provides trivial field
 */
class ZeroExternalData final : public ExternalDataBase {
 protected:
  mfem::ParMesh *pmesh_;
  const int sorder_;
  int dim_;

  mfem::FiniteElementCollection *fec_ = nullptr;
  mfem::ParFiniteElementSpace *fes_ = nullptr;
  mfem::FiniteElementCollection *vfec_ = nullptr;
  mfem::ParFiniteElementSpace *vfes_ = nullptr;  
  mfem::ParGridFunction *Tdata_ = nullptr;
  mfem::ParGridFunction *Udata_ = nullptr;

 public:
  /**
   * @brief Constructor
   *
   * Capture information necessary to set up the class, which includes
   *
   * @param pmesh A pointer to the mesh
   * @param sorder The polynomial order for scalar fields
   * @param data The field data
   */
  ZeroExternalData(mfem::ParMesh *pmesh, int sorder);

  /// Free the interface fields and support objects
  ~ZeroExternalData() final;

  /// Allocate and initialize the interface fields to constant values
  void initializeSelf() final;

  /// Since everything is constant, step is a no-op
  void step() final {}
  void setup() final {}
};

#endif  // EXTERNAL_DATA_BASE_HPP_
