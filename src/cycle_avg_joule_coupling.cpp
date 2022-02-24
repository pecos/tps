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

CycleAvgJouleCoupling::CycleAvgJouleCoupling(MPI_Session &mpi, string &inputFileName, TPS::Tps *tps)
    : mpi_(mpi), em_opt_() {
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
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing interpolation data.\n");

  ParMesh *flow_mesh = flow_solver_->GetMesh();
  ParMesh *em_mesh = qmsa_solver_->getMesh();
  assert(flow_mesh != NULL);
  assert(em_mesh != NULL);

  assert(flow_mesh->GetNodes() != NULL);
  if (em_mesh->GetNodes() == NULL) {
    em_mesh->SetCurvature(1, false, -1, 0);
  }

  interp_flow_to_em_->Setup(*flow_mesh);
  interp_flow_to_em_->SetDefaultInterpolationValue(0.0);

  // TODO(trevilo): Add em to flow interpolation
  // interp_em_to_flow_->Setup(*(qmsa_solver_->getMesh()));
  // interp_em_to_flow_->SetDefaultInterpolationValue(0);
}

void CycleAvgJouleCoupling::interpConductivityFromFlowToEM() {
  const ParMesh *em_mesh = qmsa_solver_->getMesh();

  // NB: Code below only valid if 1) conductivity treated as H1
  // function, 2) order of conductivity field is same as the order of
  // the mesh and 3) nodes for em mesh are ordered byNODES
  //
  // TODO(trevilo): Generalize beyond the constraints above
  const Vector vxyz_em = *(em_mesh->GetNodes());
  const int dim = em_mesh->Dimension();
  const int n_mesh_nodes = vxyz_em.Size() / dim;

  Vector conductivity_em(n_mesh_nodes);
  const ParGridFunction *conductivity_flow_gf = flow_solver_->GetPlasmaConductivityGF();
  interp_flow_to_em_->Interpolate(vxyz_em, *conductivity_flow_gf, conductivity_em);

  ParGridFunction *conductivity_em_gf = qmsa_solver_->getPlasmaConductivityGF();
  *conductivity_em_gf = conductivity_em;
}

void CycleAvgJouleCoupling::interpJouleHeatingFromEMToFlow() {
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
  interpConductivityFromFlowToEM();
  qmsa_solver_->solve();
}
