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

CycleAvgJouleCoupling::CycleAvgJouleCoupling(MPI_Session &mpi, string &inputFileName, TPS::Tps *tps, int max_out)
    : mpi_(mpi), em_opt_(), max_outer_iters_(max_out) {
  qmsa_solver_ = new QuasiMagnetostaticSolverAxiSym(mpi, em_opt_, tps);
  flow_solver_ = new M2ulPhyS(mpi, inputFileName, tps);
#ifdef HAVE_GSLIB
  interp_flow_to_em_ = new FindPointsGSLIB(MPI_COMM_WORLD);
  interp_em_to_flow_ = new FindPointsGSLIB(MPI_COMM_WORLD);
#endif
}

CycleAvgJouleCoupling::~CycleAvgJouleCoupling() {
  delete flow_solver_;
  delete qmsa_solver_;
#ifdef HAVE_GSLIB
  delete interp_flow_to_em_;
  delete interp_em_to_flow_;
#endif
}

void CycleAvgJouleCoupling::initializeInterpolationData() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing interpolation data.\n");

#ifdef HAVE_GSLIB
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
  interp_em_to_flow_->Setup(*(qmsa_solver_->getMesh()));
  interp_em_to_flow_->SetDefaultInterpolationValue(0);
#else
  mfem_error("Cannot initialize interpolation without GSLIB support.");
#endif
}

void CycleAvgJouleCoupling::interpConductivityFromFlowToEM() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Interpolating conductivity to EM mesh.\n");
  const ParMesh *em_mesh = qmsa_solver_->getMesh();
  const ParFiniteElementSpace *em_fespace = qmsa_solver_->getFESpace();

  const int NE = em_mesh->GetNE();
  const int nsp = em_fespace->GetFE(0)->GetNodes().GetNPoints();
  const int dim = em_mesh->Dimension();

#ifdef HAVE_GSLIB
  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;

  vxyz.SetSize(nsp * NE * dim);
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = em_fespace->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = em_fespace->GetElementTransformation(i);

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + i * nsp, nsp);
    Vector rowy(vxyz.GetData() + i * nsp + NE * nsp, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + i * nsp + 2 * NE * nsp, nsp);
    }
    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  const int nodes_cnt = vxyz.Size() / dim;

  // Interpolate
  Vector conductivity_em(nodes_cnt);
  const ParGridFunction *conductivity_flow_gf = flow_solver_->GetPlasmaConductivityGF();
  interp_flow_to_em_->Interpolate(vxyz, *conductivity_flow_gf, conductivity_em);

  // Set grid function
  ParGridFunction *conductivity_em_gf = qmsa_solver_->getPlasmaConductivityGF();

  Array<int> vdofs;
  Vector vals;
  Vector elem_dof_vals(nsp);

  for (int i = 0; i < NE; i++) {
    em_fespace->GetElementVDofs(i, vdofs);
    vals.SetSize(vdofs.Size());
    for (int j = 0; j < nsp; j++) {
      // Arrange values byNodes
      elem_dof_vals(j) = conductivity_em(i*nsp + j);
    }
    conductivity_em_gf->SetSubVector(vdofs, elem_dof_vals);
  }

  conductivity_em_gf->SetFromTrueVector();
#else
  mfem_error("Cannot interpolate without GSLIB support.");
#endif
}

void CycleAvgJouleCoupling::interpJouleHeatingFromEMToFlow() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Interpolating Joule heating to flow mesh.\n");
  const ParMesh *flow_mesh = flow_solver_->GetMesh();
  const ParFiniteElementSpace *flow_fespace = flow_solver_->GetFESpace();

  const int NE = flow_mesh->GetNE();
  const int nsp = flow_fespace->GetFE(0)->GetNodes().GetNPoints();
  const int dim = flow_mesh->Dimension();

#ifdef HAVE_GSLIB
  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;

  vxyz.SetSize(nsp * NE * dim);
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = flow_fespace->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = flow_fespace->GetElementTransformation(i);

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + i * nsp, nsp);
    Vector rowy(vxyz.GetData() + i * nsp + NE * nsp, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + i * nsp + 2 * NE * nsp, nsp);
    }
    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  const int nodes_cnt = vxyz.Size() / dim;

  // Evaluate source grid function.
  Vector interp_vals(nodes_cnt);

  const ParGridFunction *joule_heating_gf = qmsa_solver_->getJouleHeatingGF();
  interp_em_to_flow_->Interpolate(vxyz, *joule_heating_gf, interp_vals);

  ParGridFunction *joule_heating_flow = flow_solver_->GetJouleHeatingGF();
  joule_heating_flow->SetFromTrueDofs(interp_vals);
#else
  mfem_error("Cannot interpolate without GSLIB support.");
#endif
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

  const int increment = flow_solver_->getMaximumIterations();
  int curr_iter = flow_solver_->getCurrentIterations();
  flow_solver_->setMaximumIterations(curr_iter + increment);

  for (int outer_iters = 0; outer_iters < max_outer_iters_; outer_iters++) {
    // EM
    interpConductivityFromFlowToEM();
    qmsa_solver_->solve();

    // flow
    interpJouleHeatingFromEMToFlow();
    flow_solver_->solve();

    // update max solver iters so that on the next time through the flow solver does something
    curr_iter = flow_solver_->getCurrentIterations();
    flow_solver_->setMaximumIterations(curr_iter + increment);
  }
}
