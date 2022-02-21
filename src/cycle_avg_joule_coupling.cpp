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
#include "cycle_avg_joule_coupling.hpp"

#include "M2ulPhyS.hpp"
#include "em_options.hpp"
#include "quasimagnetostatic.hpp"

CycleAvgJouleCoupling::CycleAvgJouleCoupling(MPI_Session &mpi, string &inputFileName, TPS::Tps *tps) : em_opt_() {
  qmsa_solver_ = new QuasiMagnetostaticSolverAxiSym(mpi, em_opt_, tps);
  flow_solver_ = new M2ulPhyS(mpi, inputFileName, tps);
  interp_flow_to_em_ = new FindPointsGSLIB(MPI_COMM_WORLD);
  interp_em_to_flow_ = new FindPointsGSLIB(MPI_COMM_WORLD);
}

CycleAvgJouleCoupling::~CycleAvgJouleCoupling() {
  delete flow_solver_;
  delete qmsa_solver_;
  delete interp_flow_to_em_;
  delete interp_em_to_flow_;
}

void CycleAvgJouleCoupling::initializeInterpolationData() {
  interp_flow_to_em_->Setup(*(flow_solver_->GetMesh()));
  interp_flow_to_em_->SetDefaultInterpolationValue(0);

  interp_em_to_flow_->Setup(*(qmsa_solver_->getMesh()));
  interp_em_to_flow_->SetDefaultInterpolationValue(0);
}

void CycleAvgJouleCoupling::interpConductivityFromFlowToEM() {
  const Vector vxyz_em = *(qmsa_solver_->getMesh()->GetNodes());
  Vector conductivity_em(vxyz_em.Size());
  const ParGridFunction *conductivity_flow_gf = flow_solver_->GetPlasmaConductivityGF();
  interp_flow_to_em_->Interpolate(vxyz_em, *conductivity_flow_gf, conductivity_em);

  ParGridFunction *conductivity_em_gf = qmsa_solver_->getPlasmaConductivityGF();
  *conductivity_em_gf = conductivity_em;
}

void CycleAvgJouleCoupling::interpJouleHeatingFromEMToFlow() {
  // Not implemented yet
  cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
  exit(1);
}

void CycleAvgJouleCoupling::parseSolverOptions() {
  qmsa_solver_->parseSolverOptions();
  flow_solver_->parseSolverOptions();
}

void CycleAvgJouleCoupling::initialize() {
  qmsa_solver_->initialize();
  flow_solver_->initialize();
  initializeInterpolationData();
}

void CycleAvgJouleCoupling::solve() {
  flow_solver_->solve();
  qmsa_solver_->solve();
}
