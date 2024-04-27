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

class IODataOrganizer;
class Averaging;
struct thermoChemToFlow;
struct turbModelToFlow;
struct spongeToFlow;
struct extDataToFlow;

struct flowToThermoChem {
  const mfem::ParGridFunction *velocity = nullptr;

  bool swirl_supported = false;
  const mfem::ParGridFunction *swirl = nullptr;
};

struct flowToTurbModel {
  const mfem::ParGridFunction *velocity = nullptr;

  bool swirl_supported = false;
  const mfem::ParGridFunction *swirl = nullptr;

  const mfem::ParGridFunction *gradU = nullptr;
  const mfem::ParGridFunction *gradV = nullptr;
  const mfem::ParGridFunction *gradW = nullptr;
};

class FlowBase {
 protected:
  const thermoChemToFlow *thermo_interface_;
  const turbModelToFlow *turbModel_interface_;
  const spongeToFlow *sponge_interface_;
  const extDataToFlow *extData_interface_;

 public:
  /// Destructor
  virtual ~FlowBase() {}

  virtual void initializeSelf() = 0;

  virtual void step() = 0;

  virtual mfem::ParGridFunction *getCurrentVelocity() = 0;
  virtual mfem::ParGridFunction *getCurrentVelocityGradientU() { return nullptr; }
  virtual mfem::ParGridFunction *getCurrentVelocityGradientV() { return nullptr; }
  virtual mfem::ParGridFunction *getCurrentVelocityGradientW() { return nullptr; }

  void initializeFromThermoChem(thermoChemToFlow *thermo) { thermo_interface_ = thermo; }
  void initializeFromTurbModel(turbModelToFlow *turbModel) { turbModel_interface_ = turbModel; }
  void initializeFromSponge(spongeToFlow *sponge) { sponge_interface_ = sponge; }
  void initializeFromExtData(extDataToFlow *extData) { extData_interface_ = extData; }

  virtual void initializeOperators() {}

  virtual void initializeIO(IODataOrganizer &io) const {}
  virtual void initializeViz(mfem::ParaViewDataCollection &pvdc) const {}
  virtual void initializeStats(Averaging &average, IODataOrganizer &io) const {}
  
  /**
   * @brief Header strings for screen dump
   *
   * Provides a hook for derived classes to pass a set of header
   * strings that will be printed to the screen
   */
  virtual void screenHeader(std::vector<std::string> &header) const { header.resize(0); }

  /**
   * @brief Values for screen dump
   *
   * Provides values that will be printed to the screen at user requested
   * frequency (as often as each iteration).
   */
  virtual void screenValues(std::vector<double> &values) { values.resize(0); }

  /// Interface object, provides fields necessary for the thermochemistry model
  flowToThermoChem toThermoChem_interface_;

  /// Get interface provided by thermo model
  const thermoChemToFlow *getThermoInterface() const { return thermo_interface_; }

  /// Interface object, provides fields necessary for the turbulence model
  flowToTurbModel toTurbModel_interface_;

  /// Get interface provided by thermo model
  const turbModelToFlow *getTurbModelInterface() const { return turbModel_interface_; }

  /// Get interface provided by sponge
  const spongeToFlow *getSpongeInterface() const { return sponge_interface_; }

  /// Get interface provided by external data
  const extDataToFlow *getExtDataInterface() const { return extData_interface_; }

  /**
   * @brief A hook to evaluate L2 norm of error
   *
   * Usually this will be a no-op, but it is useful for testing on
   * cases where we have an exact solution.
   */
  virtual double computeL2Error() const { return -1.0; }
};

class ZeroFlow final : public FlowBase {
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
  void step() {}

  mfem::ParGridFunction *getCurrentVelocity() final { return velocity_; }
};

#endif  // SPLIT_FLOW_BASE_HPP_
