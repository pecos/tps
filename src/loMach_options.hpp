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

#ifndef LOMACH_OPTIONS_HPP_
#define LOMACH_OPTIONS_HPP_

#include <iostream>
#include <map>
#include <string>

#include "io.hpp"

namespace TPS {
class Tps;
}

class TurbulenceModelOptions {
 public:
  TurbulenceModelOptions();
  void read(TPS::Tps *tps, std::string prefix = std::string(""));

  enum TurbulenceModelType { NONE, SMAGORINSKY, SIGMA, ALGEBRAIC_RANS };

  std::string turb_model_string_;
  std::map<std::string, TurbulenceModelType> turb_model_map_;
  TurbulenceModelType turb_model_type_;

  double turb_model_constant_;
  bool exclude_mean_;
};

class LoMachTemporalOptions {
 public:
  LoMachTemporalOptions();
  void read(TPS::Tps *tps, std::string prefix = std::string(""));

  enum IntegratorType { CURL_CURL, STAGGERED_TIME, DELTA_P };

  std::string integrator_string_;
  std::map<std::string, IntegratorType> integrator_map_;
  IntegratorType integrator_type_;

  bool enable_constant_dt_;

  double cfl_;
  double initial_dt_;
  double minimum_dt_;
  double maximum_dt_;
  double factor_dt_;
  double constant_dt_;

  int bdf_order_;
};

/** LoMach options
 *  A class for storing/passing options to Low Mach solvers.
 */
class LoMachOptions {
 public:
  // IO-related options
  int output_frequency_;
  int timing_frequency_;
  IOOptions io_opts_;

  // Temporal scheme related options
  int max_steps_; /**< Maximum number of time steps */
  LoMachTemporalOptions ts_opts_;

  // SGS-related options
  TurbulenceModelOptions turb_opts_;

  // Mesh-related options
  std::string mesh_file; /**< Mesh filename */
  double scale_mesh;
  int ref_levels; /**< Number of uniform mesh refinements */

  // Periodic domain
  bool periodic;
  double x_trans;
  double y_trans;
  double z_trans;
  bool periodicX;
  bool periodicY;
  bool periodicZ;

  // FEM
  int order; /**< Element order */
  int uOrder;
  int pOrder;
  int nOrder;

  // Model choices
  std::string flow_solver;   /**< Flow solver name */
  std::string thermo_solver; /**< Themo-chemical solver name */

  // TODO(trevilo): Do we want/need to keep any of these here?
  // They aren't currently being used.
  int max_iter; /**< Maximum number of linear solver iterations */
  double rtol;  /**< Linear solver relative tolerance */
  double atol;  /**< Linear solver absolute tolerance */

  bool thermalDiv;  // turn on/off Qt
  bool realDiv;     // use actual divergence

  // compute wall distance function or not
  bool compute_wallDistance;

  int nFilter;
  double filterWeight;
  bool filterTemp, filterVel;
  bool solveTemp;
  int nSpec;

  LoMachOptions() {
    order = 1;
    scale_mesh = 1.0;
    ref_levels = 0;
    max_iter = 100;
    rtol = 1.0e-8;
    atol = 1.0e-12;
    compute_wallDistance = false;
    nSpec = 0;
  }

  void print(std::ostream &out) {
    out << std::endl;
    out << "LoMach options:" << std::endl;
    out << "  mesh_file   = " << mesh_file << std::endl;
    out << "  scale_mesh  = " << scale_mesh << std::endl;
    out << "  flow_solver = " << flow_solver << std::endl;
    out << "  thermo_solver = " << thermo_solver << std::endl;
    out << "  order       = " << order << std::endl;
    out << "  ref_levels  = " << ref_levels << std::endl;
    out << "  ref_levels  = " << ref_levels << std::endl;
    out << "  max_iter    = " << max_iter << std::endl;
    out << "  rtol        = " << rtol << std::endl;
    out << "  atol        = " << atol << std::endl;
    out << std::endl;
  }
};

#endif  // LOMACH_OPTIONS_HPP_
