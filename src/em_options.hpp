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
  // Universal options
  bool qms;  /**< Use quasi-magnetostatic solver */
  bool seqs; /**< Use stabilized, electro-quasistatic gauge solver */

  const char *mesh_file; /**< Mesh filename */

  int order;      /**< Element order */
  int ref_levels; /**< Number of uniform mesh refinements */

  int max_iter; /**< Maximum number of linear solver iterations */
  double rtol;  /**< Linear solver relative tolerance */
  double atol;  /**< Linear solver absolute tolerance */

  const char *By_file; /**< Filename for vertical component of magnetic field on y-axis */
  int nBy;             /**< Number of uniformly spaced points for By output */
  double yinterp_min;  /**< Begin value for uniformly spaced points */
  double yinterp_max;  /**< End value for uniformly spaced points */

  // QMS only options
  bool top_only; /**< Flag to specify current in top rings only */
  bool bot_only; /**< Flag to specify current in bottom rings only */

  // SEQS only options
  mfem::Array<int> conductor_domains;  /**< List of volume attributes corresponding to conductors */
  mfem::Array<int> neumann_bc_attr;    /**< List of boundary attributes corresponding to Neumann boundary */
  double nd_conductivity;  /**< Non-dimensional conductivity: \frac{\sigma_0}{\omega \epsilon_0} */
  double nd_frequency;     /**< Non-dimensional frequency: \frac{\omega \ell}{c} */

  ElectromagneticOptions()
    :
    qms(true), seqs(false),
    mesh_file("hello.msh"), order(1), ref_levels(0),
    max_iter(100), rtol(1e-6), atol(1e-10),
    By_file("By.h5"), nBy(0),
    yinterp_min(0.0), yinterp_max(1.0),
    top_only(false), bot_only(false),
    conductor_domains(0), neumann_bc_attr(0),
    nd_conductivity(1e6), nd_frequency(0.001)
  { }

  void AddElectromagneticOptions(mfem::OptionsParser &args) {
    args.AddOption(&qms, "-qms", "--qms-solver", "-nqms", "--no-qms-solver",
                   "Use the quasi-magnetostatic solver (for EM-only)");
    args.AddOption(&seqs, "-seqs", "--seqs-solver", "-nseqs", "--no-seqs-solver",
                   "Use the stabilized, electro-quasistatic gauge solver (for EM-only)");
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file (for EM-only simulation)");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) (for EM-only).");
    args.AddOption(&ref_levels, "-r", "--ref",
                   "Number of uniform refinements (for EM-only).");
    args.AddOption(&max_iter, "-i", "--maxiter",
                   "Maximum number of iterations (for EM-only).");
    args.AddOption(&rtol, "-t", "--rtol",
                   "Solver relative tolerance (for EM-only).");
    args.AddOption(&atol, "-a", "--atol",
                   "Solver absolute tolerance (for EM-only).");
    args.AddOption(&By_file, "-by", "--byfile",
                   "File for By interpolant output.");
    args.AddOption(&nBy, "-ny", "--nyinterp",
                   "Number of interpolation points.");
    args.AddOption(&yinterp_min, "-y0", "--yinterpMin",
                   "Minimum y interpolation value");
    args.AddOption(&yinterp_max, "-y1", "--yinterpMax",
                   "Maximum y interpolation value");
    args.AddOption(&top_only, "-top", "--top-only", "-ntop", "--no-top-only",
                   "Run current through top branch only (QMS solver only)");
    args.AddOption(&bot_only, "-bot", "--bot-only", "-nbot", "--no-bot-only",
                   "Run current through bottom branch only (QMS solver only)");
    args.AddOption(&conductor_domains, "-cd", "--conductor-doms",
                   "List of conductor subdomain volume attributes (SEQS solver only)");
    args.AddOption(&neumann_bc_attr, "-nbc", "--neumann_bc",
                   "List of Neumann BC boundary attributes (SEQS solver only)");
    args.AddOption(&nd_conductivity, "-s", "--sigma",
                   "Non-dimensional conductivity, sigma_0/(omega*epsilon_0) (SEQS solver only)");
    args.AddOption(&nd_frequency, "-f", "--eta",
                   "Non-dimensional (angular) frequency, (omega*ell)/c (SEQS solver only)");
  }

  void print(std::ostream &out) {
    out << std::endl;
    out << "Electromagnetics options:" << std::endl;
    out << "  qms         = " << qms         << std::endl;
    out << "  seqs        = " << seqs        << std::endl;
    out << "  mesh_file   = " << mesh_file   << std::endl;
    out << "  order       = " << order       << std::endl;
    out << "  ref_levels  = " << ref_levels  << std::endl;
    out << "  ref_levels  = " << ref_levels  << std::endl;
    out << "  max_iter    = " << max_iter    << std::endl;
    out << "  rtol        = " << rtol        << std::endl;
    out << "  atol        = " << atol        << std::endl;
    out << "  By_file     = " << By_file     << std::endl;
    out << "  nBy         = " << nBy         << std::endl;
    out << "  yinterp_min = " << yinterp_min << std::endl;
    out << "  yinterp_max = " << yinterp_max << std::endl;
    out << std::endl;
    if (qms) {
      out << "  Quasimagnetostatic options:" << std::endl;
      out << "    top_only    = " << top_only    << std::endl;
      out << "    bot_only    = " << top_only    << std::endl;
    } else if (seqs) {
      out << "  SEQS solver options:" << std::endl;
      out << "    conductor_domains = ( ";
      for (int i = 0; i < conductor_domains.Size(); i++) {
        out << conductor_domains[i] << " ";
      }
      out << ") " << std::endl;
      out << "    neumann_bcs = ( ";
      for (int i = 0; i < neumann_bc_attr.Size(); i++) {
        out << neumann_bc_attr[i] << " ";
      }
      out << ") " << std::endl;
      out << "    nd_conductivity = " << nd_conductivity << std::endl;
      out << "    nd_frequency    = " << nd_frequency << std::endl;
    }
    out << std::endl;
  }
};

#endif  // EM_OPTIONS_HPP_
