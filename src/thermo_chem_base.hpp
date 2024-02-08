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
#ifndef THERMO_CHEM_BASE_HPP_
#define THERMO_CHEM_BASE_HPP_
/** @file
 * @brief Contains base class and simplest possible variant of thermochemistry model.
 */

#include "tps_mfem_wrap.hpp"

struct flowToThermoChem;

/**
 * Provides wrapper for fields that need to be provided by the
 * thermochemistry model to the flow.
 */
struct thermoChemToFlow {
  const mfem::ParGridFunction *density = nullptr;
  const mfem::ParGridFunction *viscosity = nullptr;
  const mfem::ParGridFunction *thermal_divergence = nullptr;
};

/**
 * Provides interface for thermochemical model implementation
 * to be used for time split schemes, such as those available through
 * LoMachSolver
 */
class ThermoChemModelBase {
 protected:
  const flowToThermoChem *flow_interface_;

 public:
  /// Destructor
  virtual ~ThermoChemModelBase() {}

  /**
   * @brief Initialize the model-owned data
   *
   * Initialize any fields that this model owns.  This *must* include
   * the interface fields in thermoChemToFlow, but it may include
   * other fields as well (i.e.., internal state variables such as
   * temperature and/or species densities).  No parameters are passed
   * b/c it is assumed that any necessary information is stored as
   * part of the class, so you must collect this at construction time.
   */
  virtual void initializeSelf() = 0;

  /**
   * @brief Take a single time step
   */
  virtual void step() = 0;

  /**
   * @brief Initialize data from the flow class
   *
   * Initialize fields that the thermochemistry model needs from the
   * flow.
   */
  void initializeFromFlow(flowToThermoChem *flow) { flow_interface_ = flow; }

  /// Interface object, provides fields necessary for the flow
  thermoChemToFlow interface;

  /// Get interface provided by flow model
  const flowToThermoChem *getFlowInterface() const { return flow_interface_; }
};

/**
 * Provides simplest possible thermochemistry model---i.e., constant
 * density, constant viscosity, zero thermal divergence.
 */
class ConstantPropertyThermoChem : public ThermoChemModelBase {
 protected:
  mfem::ParMesh *pmesh_;
  const int sorder_;
  const double rho_;
  const double mu_;

  mfem::FiniteElementCollection *fec_ = nullptr;
  mfem::ParFiniteElementSpace *fes_ = nullptr;

  mfem::ParGridFunction *density_ = nullptr;
  mfem::ParGridFunction *viscosity_ = nullptr;
  mfem::ParGridFunction *thermal_divergence_ = nullptr;

 public:
  /**
   * @brief Constructor
   *
   * Capture information necessary to set up the class, which includes
   *
   * @param pmesh A pointer to the mesh
   * @param sorder The polynomial order for scalar fields
   * @param rho The (constant) value to use for the density
   * @param mu The (constant) value to use for the viscosity
   */
  ConstantPropertyThermoChem(mfem::ParMesh *pmesh, int sorder, double rho, double mu);

  /// Free the interface fields and support objects
  ~ConstantPropertyThermoChem() final;

  /// Allocate and initialize the interface fields to constant values
  void initializeSelf() final;

  /// Since everything is constant, step is a no-op
  void step() final {}
};

#endif  // THERMO_CHEM_BASE_HPP_