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

#include "tps_mfem_wrap.hpp"

/** LoMach options
 *  A class for storing/passing options to Low Mach solvers.
 */
class LoMachOptions {
 public:
  std::string mesh_file; /**< Mesh filename */
  double scale_mesh;

  std::string flow_solver;   /**< Flow solver name */
  std::string thermo_solver; /**< Themo-chemical solver name */

  int order; /**< Element order */
  int uOrder;
  int pOrder;
  int nOrder;
  int ref_levels; /**< Number of uniform mesh refinements */

  int max_iter; /**< Maximum number of linear solver iterations */
  double rtol;  /**< Linear solver relative tolerance */
  double atol;  /**< Linear solver absolute tolerance */

  bool thermalDiv; // turn on/off Qt
  bool realDiv;    // use actual divergence

  int nFilter;
  double filterWeight;
  bool filterTemp, filterVel;
  bool solveTemp;

  bool channelTest;

  LoMachOptions() {
    order = 1;
    ref_levels = 0;
    max_iter = 100;
    rtol = 1.0e-8;
    atol = 1.0e-12;
    /*
    preconditioner_background_sigma = -1;
    evaluate_magnetic_field = true;
    nBy = 0;
    yinterp_min = 0.0;
    yinterp_max = 1.0;
    top_only = false;
    bot_only = false;
    current_amplitude = 1.0;
    current_frequency = 1.0;
    mu0 = 1.0;
    */
  }

  void print(std::ostream &out) {
    out << std::endl;
    out << "LoMach options:" << std::endl;
    out << "  mesh_file   = " << mesh_file << std::endl;
    out << "  scale_mesh  = " << scale_mesh << std::endl;
    out << "  flow_solver = " << flow_solver << std::endl;
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
