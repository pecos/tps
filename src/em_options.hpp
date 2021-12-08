// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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

#ifndef EM_OPTIONS_HPP_
#define EM_OPTIONS_HPP_

#include <iostream>
#include <mfem.hpp>

/** Electromagnetics options
 *  A class for storing/passing options to EM solvers.
 */
class ElectromagneticOptions {
 public:
  const char *mesh_file; /**< Mesh filename */

  int order;      /**< Element order */
  int ref_levels; /**< Number of uniform mesh refinements */

  int max_iter; /**< Maximum number of linear solver iterations */
  double rtol;  /**< Linear solver relative tolerance */
  double atol;  /**< Linear solver absolute tolerance */

  bool top_only; /**< Flag to specify current in top rings only */
  bool bot_only; /**< Flag to specify current in bottom rings only */

  const char *By_file; /**< Filename for vertical component of magnetic field on y-axis */
  int nBy;             /**< Number of uniformly spaced points for By output */
  double yinterp_min;  /**< Begin value for uniformly spaced points */
  double yinterp_max;  /**< End value for uniformly spaced points */

  ElectromagneticOptions()
      : mesh_file("hello.msh"),
        order(1),
        ref_levels(0),
        max_iter(100),
        rtol(1e-6),
        atol(1e-10),
        top_only(false),
        bot_only(false),
        By_file("By.h5"),
        nBy(0),
        yinterp_min(0.0),
        yinterp_max(1.0) {}

  void AddElectromagneticOptions(mfem::OptionsParser &args) {
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file (for EM-only simulation)");
    args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree) (for EM-only).");
    args.AddOption(&ref_levels, "-r", "--ref", "Number of uniform refinements (for EM-only).");
    args.AddOption(&max_iter, "-i", "--maxiter", "Maximum number of iterations (for EM-only).");
    args.AddOption(&rtol, "-t", "--rtol", "Solver relative tolerance (for EM-only).");
    args.AddOption(&atol, "-a", "--atol", "Solver absolute tolerance (for EM-only).");
    args.AddOption(&top_only, "-top", "--top-only", "-ntop", "--no-top-only", "Run current through top branch only");
    args.AddOption(&bot_only, "-bot", "--bot-only", "-nbot", "--no-bot-only", "Run current through bottom branch only");
    args.AddOption(&By_file, "-by", "--byfile", "File for By interpolant output.");
    args.AddOption(&nBy, "-ny", "--nyinterp", "Number of interpolation points.");
    args.AddOption(&yinterp_min, "-y0", "--yinterpMin", "Minimum y interpolation value");
    args.AddOption(&yinterp_max, "-y1", "--yinterpMax", "Maximum y interpolation value");
  }

  void print(std::ostream &out) {
    out << std::endl;
    out << "Electromagnetics options:" << std::endl;
    out << "  mesh_file   = " << mesh_file << std::endl;
    out << "  order       = " << order << std::endl;
    out << "  ref_levels  = " << ref_levels << std::endl;
    out << "  ref_levels  = " << ref_levels << std::endl;
    out << "  max_iter    = " << max_iter << std::endl;
    out << "  rtol        = " << rtol << std::endl;
    out << "  atol        = " << atol << std::endl;
    out << "  top_only    = " << top_only << std::endl;
    out << "  bot_only    = " << top_only << std::endl;
    out << "  By_file     = " << By_file << std::endl;
    out << "  nBy         = " << nBy << std::endl;
    out << "  yinterp_min = " << yinterp_min << std::endl;
    out << "  yinterp_max = " << yinterp_max << std::endl;
    out << std::endl;
  }
};

#endif  // EM_OPTIONS_HPP_
