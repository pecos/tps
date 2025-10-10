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

#ifndef STATIC_THERMO_HPP_
#define STATIC_THERMO_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "dataStructures.hpp"
#include "dirichlet_bc_helper.hpp"
#include "io.hpp"
#include "table.hpp"
#include "thermo_chem_base.hpp"
#include "tps_mfem_wrap.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachOptions;
struct temporalSchemeCoefficients;

/**
 * @brief Enforce constant temperature and zero Qt, for debugging
 * flow-only issues
 */
class StaticThermo final : public ThermoChemModelBase {
 private:
  // Options-related structures
  TPS::Tps *tpsP_ = nullptr;

  // Mesh and discretization scheme info
  ParMesh *pmesh_ = nullptr;
  int order_;
  const temporalSchemeCoefficients &time_coeff_;

  // Flags
  bool rank0_;                    /**< true if this is rank 0 */
  bool axisym_ = false;

  LinearTable *mu_table_;     // dynamic viscosity
  LinearTable *kappa_table_;  // thermal conductivity
  LinearTable *sigma_table_;  // electrical conductivity
  LinearTable *Rgas_table_;   // specific gas constant
  LinearTable *Cp_table_;     // specific heat at constant pressure

  /// pressure-related, closed-system thermo pressure changes
  double ambient_pressure_, thermo_pressure_;

  // Initial temperature value (if constant IC)
  double T_ic_;

  // FEM related fields and objects

  // Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec_ = nullptr;

  // Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes_ = nullptr;

  // Fields
  ParGridFunction *temp_;
  ParGridFunction rn_gf_;
  ParGridFunction mu_gf_;
  ParGridFunction kappa_gf_;
  ParGridFunction sigma_gf_;
  ParGridFunction Rgas_gf_;
  ParGridFunction Cp_gf_;
  ParGridFunction Qt_gf_;

  Vector Qt_;
  Vector rn_;
  Vector kappa_;
  Vector visc_;
  Vector sigma_;
  Vector Rgas_;
  Vector Cp_;

 public:
  StaticThermo(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff, TPS::Tps *tps);
  virtual ~StaticThermo();

  // Functions overriden from base class
  void initializeSelf() final;
  void initializeOperators() final;
  void step() final {};
  void initializeIO(IODataOrganizer &io) final {};
  void initializeViz(ParaViewDataCollection &pvdc) final;
  void evaluatePlasmaConductivityGF() final {};

  // Functions added here
  void updateDensity();

  /// Return a pointer to the current temperature ParGridFunction.
  ParGridFunction *GetCurrentTemperature() { return temp_; }

  /// Return a pointer to the current density ParGridFunction.
  ParGridFunction *GetCurrentDensity() { return &rn_gf_; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  ParGridFunction *GetCurrentViscosity() { return &mu_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiffusivity() { return &kappa_gf_; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  ParGridFunction *GetCurrentThermalDiv() { return &Qt_gf_; }
};
#endif  // STATIC_THERMO_HPP_
