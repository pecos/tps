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
#ifndef CYCLE_AVG_JOULE_COUPLING_HPP_
#define CYCLE_AVG_JOULE_COUPLING_HPP_

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <tps_config.h>

#include <iostream>
#include <mfem.hpp>

#include "M2ulPhyS.hpp"
#include "em_options.hpp"
#include "quasimagnetostatic.hpp"
#include "solver.hpp"

class CycleAvgJouleCoupling : public TPS::Solver {
 private:
  MPI_Session &mpi_;
  ElectromagneticOptions em_opt_;
  QuasiMagnetostaticSolverAxiSym *qmsa_solver_;
  M2ulPhyS *flow_solver_;

  int max_outer_iters_;

#ifdef HAVE_GSLIB
  FindPointsGSLIB *interp_flow_to_em_;
  FindPointsGSLIB *interp_em_to_flow_;
#endif

  int n_em_interp_nodes_;
  int n_flow_interp_nodes_;

 public:
  CycleAvgJouleCoupling(MPI_Session &mpi, string &inputFileName, TPS::Tps *tps, int max_out);
  ~CycleAvgJouleCoupling();

  void initializeInterpolationData();
  void interpConductivityFromFlowToEM();
  void interpJouleHeatingFromEMToFlow();

  void parseSolverOptions() override;
  void initialize() override;
  void solve() override;

  M2ulPhyS *getFlowSolver() { return flow_solver_; }
  QuasiMagnetostaticSolverAxiSym *getEMSolver() { return qmsa_solver_; }
};
#endif  // CYCLE_AVG_JOULE_COUPLING_HPP_