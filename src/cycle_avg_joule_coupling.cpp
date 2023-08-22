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

CycleAvgJouleCoupling::CycleAvgJouleCoupling(string &inputFileName, TPS::Tps *tps) : em_opt_(), current_iter_(0.) {
  MPI_Comm_size(tps->getTPSCommWorld(), &nprocs_);
  MPI_Comm_rank(tps->getTPSCommWorld(), &rank_);
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  tps->getRequiredInput("cycle-avg-joule-coupled/solve-em-every-n", solve_em_every_n_);
  tps->getRequiredInput("cycle-avg-joule-coupled/max-iters", max_iters_);
  tps->getInput("cycle-avg-joule-coupled/timing-frequency", timing_freq_, 100);
  bool axisym = false;
  tps->getInput("cycle-avg-joule-coupled/axisymmetric", axisym, false);
  tps->getInput("cycle-avg-joule-coupled/input-power", input_power_, -1.);
  tps->getInput("cycle-avg-joule-coupled/initial-input-power", initial_input_power_, -1.);

  if (axisym) {
    qmsa_solver_ = new QuasiMagnetostaticSolverAxiSym(em_opt_, tps);
  } else {
    qmsa_solver_ = new QuasiMagnetostaticSolver3D(em_opt_, tps);
  }
  flow_solver_ = new M2ulPhyS(inputFileName, tps);

#ifdef HAVE_GSLIB
  interp_flow_to_em_ = new FindPointsGSLIB(tps->getTPSCommWorld());
  interp_em_to_flow_ = new FindPointsGSLIB(tps->getTPSCommWorld());
#endif
}

CycleAvgJouleCoupling::CycleAvgJouleCoupling(string &inputFileName, TPS::Tps *tps, int max_out, bool axisym,
                                             double input_power, double initial_input_power)
    : em_opt_(), current_iter_(0.), input_power_(input_power), initial_input_power_(initial_input_power) {
  MPI_Comm_size(tps->getTPSCommWorld(), &nprocs_);
  MPI_Comm_rank(tps->getTPSCommWorld(), &rank_);
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  if (axisym) {
    qmsa_solver_ = new QuasiMagnetostaticSolverAxiSym(em_opt_, tps);
  } else {
    qmsa_solver_ = new QuasiMagnetostaticSolver3D(em_opt_, tps);
  }
  flow_solver_ = new M2ulPhyS(inputFileName, tps);

  solve_em_every_n_ = flow_solver_->getMaximumIterations();
  max_iters_ = max_out * solve_em_every_n_;
#ifdef HAVE_GSLIB
  interp_flow_to_em_ = new FindPointsGSLIB(tps->getTPSCommWorld());
  interp_em_to_flow_ = new FindPointsGSLIB(tps->getTPSCommWorld());
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
  const bool verbose = rank0_;
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

  // Set up GSLIB interpolators
  interp_flow_to_em_->Setup(*flow_mesh);
  interp_flow_to_em_->SetDefaultInterpolationValue(0.0);

  interp_em_to_flow_->Setup(*(qmsa_solver_->getMesh()));
  interp_em_to_flow_->SetDefaultInterpolationValue(0);

  // Determine numbers of points to interpolate to
  const ParFiniteElementSpace *em_fespace = qmsa_solver_->getFESpace();
  n_em_interp_nodes_ = 0;
  for (int i = 0; i < em_mesh->GetNE(); i++) {
    n_em_interp_nodes_ += em_fespace->GetFE(i)->GetNodes().GetNPoints();
  }

  // Determine numbers of points to interpolate to
  const ParFiniteElementSpace *flow_fespace = flow_solver_->GetFESpace();
  n_flow_interp_nodes_ = 0;
  for (int i = 0; i < flow_mesh->GetNE(); i++) {
    n_flow_interp_nodes_ += flow_fespace->GetFE(i)->GetNodes().GetNPoints();
  }

#else
  mfem_error("Cannot initialize interpolation without GSLIB support.");
#endif
}

void CycleAvgJouleCoupling::interpConductivityFromFlowToEM() {
  const bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Interpolating conductivity to EM mesh.\n");
  const ParMesh *em_mesh = qmsa_solver_->getMesh();
  const ParFiniteElementSpace *em_fespace = qmsa_solver_->getFESpace();

  const int NE = em_mesh->GetNE();
  const int dim = em_mesh->Dimension();

#ifdef HAVE_GSLIB
  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;

  vxyz.SetSize(n_em_interp_nodes_ * dim);

  int n0 = 0;  // running total of starting index
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = em_fespace->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = em_fespace->GetElementTransformation(i);

    const int nsp = ir.GetNPoints();

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + n0, nsp);
    Vector rowy(vxyz.GetData() + n0 + n_em_interp_nodes_, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + n0 + 2 * n_em_interp_nodes_, nsp);
    }
    n0 += nsp;

    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  assert(n0 == n_em_interp_nodes_);

  // Interpolate
  Vector conductivity_em(n_em_interp_nodes_);
  const ParGridFunction *conductivity_flow_gf = flow_solver_->GetPlasmaConductivityGF();
  interp_flow_to_em_->Interpolate(vxyz, *conductivity_flow_gf, conductivity_em);

  // Set grid function
  ParGridFunction *conductivity_em_gf = qmsa_solver_->getPlasmaConductivityGF();

  Array<int> vdofs;
  Vector elem_dof_vals;

  n0 = 0;
  for (int i = 0; i < NE; i++) {
    em_fespace->GetElementVDofs(i, vdofs);
    const int nsp = em_fespace->GetFE(i)->GetNodes().GetNPoints();
    assert(nsp == vdofs.Size());
    elem_dof_vals.SetSize(nsp);
    for (int j = 0; j < nsp; j++) {
      elem_dof_vals(j) = conductivity_em(n0 + j);
    }
    conductivity_em_gf->SetSubVector(vdofs, elem_dof_vals);
    n0 += nsp;
  }

  conductivity_em_gf->SetTrueVector();
  conductivity_em_gf->SetFromTrueVector();
#else
  mfem_error("Cannot interpolate without GSLIB support.");
#endif
}

void CycleAvgJouleCoupling::interpJouleHeatingFromEMToFlow() {
  const bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Interpolating Joule heating to flow mesh.\n");
  const ParMesh *flow_mesh = flow_solver_->GetMesh();
  const ParFiniteElementSpace *flow_fespace = flow_solver_->GetFESpace();

  const int NE = flow_mesh->GetNE();
  const int dim = flow_mesh->Dimension();

#ifdef HAVE_GSLIB
  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;

  vxyz.SetSize(n_flow_interp_nodes_ * dim);

  int n0 = 0;
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = flow_fespace->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = flow_fespace->GetElementTransformation(i);

    const int nsp = ir.GetNPoints();

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + n0, nsp);
    Vector rowy(vxyz.GetData() + n0 + n_flow_interp_nodes_, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + n0 + 2 * n_flow_interp_nodes_, nsp);
    }
    n0 += nsp;

    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  assert(n0 == n_flow_interp_nodes_);

  // Evaluate source grid function.
  Vector interp_vals(n_flow_interp_nodes_);

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
  this->solveBegin();
  double tlast = grvy_timer_elapsed_global();

  while (current_iter_ < max_iters_) {
    grvy_timer_begin(__func__);

    // periodically report on time/iteration
    if ((current_iter_ % timing_freq_) == 0) {
      if (rank0_) {
        double timePerIter = (grvy_timer_elapsed_global() - tlast) / timing_freq_;
        grvy_printf(ginfo, "Iteration = %i: wall clock time/iter = %.3f (secs)\n", current_iter_, timePerIter);
        tlast = grvy_timer_elapsed_global();
      }
    }

    this->solveStep();
    grvy_timer_end(__func__);
  }

  this->solveEnd();
}

void CycleAvgJouleCoupling::solveBegin() {
  flow_solver_->solveBegin();
  qmsa_solver_->solveBegin();
}

void CycleAvgJouleCoupling::solveStep() {
  // Run the em solver when it is due
  if (current_iter_ % solve_em_every_n_ == 0) {
    double delta_power = 0;
    if (input_power_ > 0) {
      delta_power = (input_power_ - initial_input_power_) * static_cast<double>(solve_em_every_n_) /
                    static_cast<double>(max_iters_);
    }
    interpConductivityFromFlowToEM();
    qmsa_solver_->solve();
    const double tot_jh = qmsa_solver_->totalJouleHeating();
    if (rank0_) {
      grvy_printf(GRVY_INFO, "The total input Joule heating = %.6e\n", tot_jh);
    }
    if (input_power_ > 0) {
      const double target_power = initial_input_power_ + (current_iter_ / solve_em_every_n_ + 1) * delta_power;
      const double ratio = target_power / tot_jh;
      qmsa_solver_->scaleJouleHeating(ratio);
      const double upd_jh = qmsa_solver_->totalJouleHeating();
      if (rank0_) {
        grvy_printf(GRVY_INFO, "The total input Joule heating after scaling = %.6e\n", upd_jh);
      }
    }
    interpJouleHeatingFromEMToFlow();
  }
  // Run a step of the flow solver
  flow_solver_->solveStep();
  // Increment the current iterate
  ++current_iter_;
}

void CycleAvgJouleCoupling::solveEnd() {
  flow_solver_->solveEnd();
  qmsa_solver_->solveEnd();
}
