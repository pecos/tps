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
#ifndef TURB_MODEL_BASE_HPP_
#define TURB_MODEL_BASE_HPP_
/** @file
 * @brief Contains base class and simplest possible variant of thermochemistry model.
 */

#include "tps_mfem_wrap.hpp"

struct flowToTurbModel;

/**
 * Provides wrapper for fields that need to be provided by the
 * turbulence model to the flow.
 */
struct turbModelToFlow {
  const mfem::ParGridFunction *eddy_viscosity = nullptr;
};

struct thermoChemToTurbModel;

/**
 * Provides wrapper for fields that need to be provided by the
 * turbulence model to the thermo chem model.
 */
struct turbModelToThermoChem {
  const mfem::ParGridFunction *eddy_viscosity = nullptr;
};

struct extDataToTurbModel;

/**
 * Provides wrapper for fields that need to be provided by the
 * turbulence model to the thermo chem model.
 */
struct turbModelToExtData { };

/**
 * Provides interface for turbulence model implementation
 * to be used for time split schemes, such as those available through
 * LoMachSolver
 */
class TurbModelBase {
 protected:
  const flowToTurbModel *flow_interface_;
  const thermoChemToTurbModel *thermoChem_interface_;
  const extDataToTurbModel *extData_interface_;

 public:
  /// Destructor
  virtual ~TurbModelBase() {}

  /**
   * @brief Initialize the model-owned data
   *
   * Initialize any fields that this model owns.  This *must* include
   * the interface fields in turbModelToFlow, but it may include
   * other fields as well (i.e.., internal state variables such as
   * temperature and/or species densities).  No parameters are passed
   * b/c it is assumed that any necessary information is stored as
   * part of the class, so you must collect this at construction time.
   */
  virtual void initializeSelf() = 0;

  /**
   * @brief Hook to let derived classes register visualization fields with ParaViewDataCollection
   */
  virtual void initializeViz(mfem::ParaViewDataCollection &pvdc) {}

  /**
   * @brief Initialize model operators
   *
   * Default implementation here does nothing.  This hook exists to
   * allow the model to initialize operators that depend on the
   * interface intformation.  Thus, it must be called *after* the
   * interfaces are initialized (e.g., after initializeFromFlow).
   */
  virtual void initializeOperators() {}

  /**
   * @brief Take a single time step
   */
  virtual void step() = 0;

  /**
   * @brief Grid-related
   */
  virtual void setup() = 0;

  /**
   * @brief Initialize data from the flow class
   *
   * Initialize fields that the turbulence model needs from the
   * flow.
   */
  void initializeFromFlow(flowToTurbModel *flow) { flow_interface_ = flow; }

  /// Get interface provided by flow model
  const flowToTurbModel *getFlowInterface() const { return flow_interface_; }

  /// Interface object, provides fields necessary for the flow
  turbModelToFlow toFlow_interface_;

  /**
   * @brief Initialize data from the thermoChem class
   *
   * Initialize fields that the turbulence model needs from the
   * thermochem model.
   */
  void initializeFromThermoChem(thermoChemToTurbModel *thermoChem) { thermoChem_interface_ = thermoChem; }

  /// Get interface provided by thermoChem model
  const thermoChemToTurbModel *getThermoChemInterface() const { return thermoChem_interface_; }

  /// Interface object, provides fields necessary for the turbModel
  turbModelToThermoChem toThermoChem_interface_;
  
  /**
   * @brief Initialize data from the external data class
   *
   * Initialize fields that the turbulence model needs from the
   * external data.
   */
  void initializeFromExtData(extDataToTurbModel *extData) { extData_interface_ = extData; }

  /// Get interface provided by external data
  const extDataToTurbModel *getExtDataInterface() const { return extData_interface_; }

  /// Interface object, provides fields necessary for the turbModel
  turbModelToExtData toExtData_interface_;

  virtual mfem::ParGridFunction *getCurrentEddyViscosity() { return nullptr; }
  virtual mfem::ParGridFunction *getGridScale() { return nullptr; }
};

/**
 * Provides simplest possible turbulence model---i.e.,
 * zero eddy viscosity.
 */
class ZeroTurbModel final : public TurbModelBase {
 protected:
  mfem::ParMesh *pmesh_;
  const int sorder_;

  mfem::FiniteElementCollection *fec_ = nullptr;
  mfem::ParFiniteElementSpace *fes_ = nullptr;

  mfem::ParGridFunction *eddy_viscosity_ = nullptr;

 public:
  /**
   * @brief Constructor
   *
   * Capture information necessary to set up the class, which includes
   *
   * @param pmesh A pointer to the mesh
   * @param sorder The polynomial order for scalar fields
   * @param nuT The (zero) value to use for the eddy viscosity
   */
  ZeroTurbModel(mfem::ParMesh *pmesh, int sorder);

  /// Free the interface fields and support objects
  ~ZeroTurbModel() final;

  /// Allocate and initialize the interface fields to constant values
  void initializeSelf() final;

  /// Since everything is constant, step and setup are a no-op
  void setup() final {}
  void step() final {}
};

#endif  // TURB_MODEL_BASE_HPP_
