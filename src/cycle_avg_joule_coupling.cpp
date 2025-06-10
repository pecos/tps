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
#include "loMach.hpp"
#include "quasimagnetostatic.hpp"

CycleAvgJouleCoupling::CycleAvgJouleCoupling(string &inputFileName, TPS::Tps *tps)
    : em_opt_(),
      qmsa_solver_(nullptr),
      flow_solver_(nullptr),
      efieldFEC_(nullptr),
      efieldFES_(nullptr),
      efieldFES1_(nullptr),
      efield_(nullptr),
      efieldR_(nullptr),
      efieldI_(nullptr),
      current_iter_(0.) {
  MPI_Comm_size(tps->getTPSCommWorld(), &nprocs_);
  MPI_Comm_rank(tps->getTPSCommWorld(), &rank_);
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  std::string plasma_solver;
  tps->getInput("cycle-avg-joule-coupled/plasma-solver", plasma_solver, std::string("compressible"));
  tps->getRequiredInput("cycle-avg-joule-coupled/solve-em-every-n", solve_em_every_n_);
  tps->getRequiredInput("cycle-avg-joule-coupled/max-iters", max_iters_);
  tps->getInput("cycle-avg-joule-coupled/timing-frequency", timing_freq_, 100);
  bool axisym = false;
  tps->getInput("cycle-avg-joule-coupled/axisymmetric", axisym, false);
  tps->getInput("cycle-avg-joule-coupled/input-power", input_power_, -1.);
  tps->getInput("cycle-avg-joule-coupled/initial-input-power", initial_input_power_, -1.);
  tps->getInput("cycle-avg-joule-coupled/fixed-conductivity", fixed_conductivity_, false);

  tps->getInput("cycle-avg-joule-coupled/oscillating-power", oscillating_power_, false);
  tps->getInput("cycle-avg-joule-coupled/input-power-amplitude",  power_amplitude_, 0.0);
  tps->getInput("cycle-avg-joule-coupled/input-power-period",  power_period_, 1.0);

  if (axisym) {
    qmsa_solver_ = new QuasiMagnetostaticSolverAxiSym(em_opt_, tps);
  } else {
    qmsa_solver_ = new QuasiMagnetostaticSolver3D(em_opt_, tps);
  }

  if (plasma_solver == "compressible") {
    flow_solver_ = new M2ulPhyS(inputFileName, tps);
  } else if (plasma_solver == "lomach") {
    flow_solver_ = new LoMachSolver(tps);
  } else {
    assert(false);
    exit(-1);
  }

#ifdef HAVE_GSLIB
  interp_flow_to_em_ = new FindPointsGSLIB(tps->getTPSCommWorld());
  interp_em_to_flow_ = new FindPointsGSLIB(tps->getTPSCommWorld());
#endif
}

CycleAvgJouleCoupling::CycleAvgJouleCoupling(string &inputFileName, TPS::Tps *tps, int max_out, bool axisym,
                                             double input_power, double initial_input_power)
    : em_opt_(),
      qmsa_solver_(nullptr),
      flow_solver_(nullptr),
      efieldFEC_(nullptr),
      efieldFES_(nullptr),
      efieldFES1_(nullptr),
      efield_(nullptr),
      efieldR_(nullptr),
      efieldI_(nullptr),
      current_iter_(0.),
      input_power_(input_power),
      initial_input_power_(initial_input_power) {
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

  tps->getRequiredInput("cycle-avg-joule-coupled/solve-em-every-n", solve_em_every_n_);
  max_iters_ = max_out * solve_em_every_n_;
#ifdef HAVE_GSLIB
  interp_flow_to_em_ = new FindPointsGSLIB(tps->getTPSCommWorld());
  interp_em_to_flow_ = new FindPointsGSLIB(tps->getTPSCommWorld());
#endif
}

CycleAvgJouleCoupling::~CycleAvgJouleCoupling() {
  if (efieldR_) delete efieldR_;
  if (efieldI_) delete efieldI_;
  if (efield_) delete efield_;
  if (efieldFES1_) delete efieldFES1_;
  if (efieldFES_) delete efieldFES_;
#ifdef HAVE_GSLIB
  delete interp_flow_to_em_;
  delete interp_em_to_flow_;
#endif
  delete flow_solver_;
  delete qmsa_solver_;
}

void CycleAvgJouleCoupling::initializeInterpolationData() {
  const bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing interpolation data.\n");

#ifdef HAVE_GSLIB
  ParMesh *flow_mesh = flow_solver_->getMesh();
  ParMesh *em_mesh = qmsa_solver_->getMesh();
  assert(flow_mesh != NULL);
  assert(em_mesh != NULL);

  assert(flow_mesh->GetNodes() != NULL);
  if (em_mesh->GetNodes() == NULL) {
    em_mesh->SetCurvature(1, false, -1, 0);
  }

  // Set up GSLIB interpolators
  interp_flow_to_em_->Setup(*flow_mesh);
  // interp_flow_to_em_->SetDefaultInterpolationValue(0.0);
  interp_flow_to_em_->SetDefaultInterpolationValue(0.1);

  interp_em_to_flow_->Setup(*(qmsa_solver_->getMesh()));
  interp_em_to_flow_->SetDefaultInterpolationValue(0);

  // Determine numbers of points to interpolate to
  const ParFiniteElementSpace *em_fespace = qmsa_solver_->getFESpace();
  n_em_interp_nodes_ = 0;
  for (int i = 0; i < em_mesh->GetNE(); i++) {
    n_em_interp_nodes_ += em_fespace->GetFE(i)->GetNodes().GetNPoints();
  }

  // Determine numbers of points to interpolate to
  const ParFiniteElementSpace *flow_fespace = flow_solver_->getFESpace();
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

#ifdef HAVE_GSLIB
  const ParMesh *em_mesh = qmsa_solver_->getMesh();
  const ParFiniteElementSpace *em_fespace = qmsa_solver_->getFESpace();

  const int NE = em_mesh->GetNE();
  const int dim = em_mesh->Dimension();

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
  const ParGridFunction *conductivity_flow_gf = flow_solver_->getPlasmaConductivityGF();
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
      if (elem_dof_vals(j) < 0.1) {
        elem_dof_vals(j) = 0.1;
      }
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

void CycleAvgJouleCoupling::interpolationPoints(Vector &vxyz, int n_interp_nodes, const ParFiniteElementSpace *fes) {
  const ParMesh *mesh = fes->GetParMesh();
  const int NE = mesh->GetNE();
  const int dim = mesh->Dimension();
  vxyz.SetSize(n_interp_nodes * dim);
  int n0 = 0;
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = fes->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = fes->GetElementTransformation(i);

    const int nsp = ir.GetNPoints();

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + n0, nsp);
    Vector rowy(vxyz.GetData() + n0 + n_interp_nodes, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + n0 + 2 * n_interp_nodes, nsp);
    }
    n0 += nsp;

    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  assert(n0 == n_interp_nodes);
}

void CycleAvgJouleCoupling::interpJouleHeatingFromEMToFlow() {
  const bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Interpolating Joule heating to flow mesh.\n");

#ifdef HAVE_GSLIB
  const ParFiniteElementSpace *flow_fespace = flow_solver_->getFESpace();

  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;
  interpolationPoints(vxyz, n_flow_interp_nodes_, flow_fespace);

  // Evaluate source grid function.
  Vector interp_vals(n_flow_interp_nodes_);

  const ParGridFunction *joule_heating_gf = qmsa_solver_->getJouleHeatingGF();
  assert(joule_heating_gf != NULL);

  interp_em_to_flow_->Interpolate(vxyz, *joule_heating_gf, interp_vals);

  ParGridFunction *joule_heating_flow = flow_solver_->getJouleHeatingGF();
  if (flow_fespace->IsDGSpace()) {
    joule_heating_flow->SetFromTrueDofs(interp_vals);
  } else {
    Array<int> vdofs;
    Vector elem_dof_vals;
    int n0 = 0;
    const int NE = flow_solver_->getMesh()->GetNE();
    for (int i = 0; i < NE; i++) {
      flow_fespace->GetElementDofs(i, vdofs);
      const int nsp = flow_fespace->GetFE(i)->GetNodes().GetNPoints();
      assert(nsp == vdofs.Size());
      elem_dof_vals.SetSize(nsp);
      for (int j = 0; j < nsp; j++) {
        elem_dof_vals(j) = interp_vals(n0 + j);
        if (elem_dof_vals(j) < 0.0) {
          elem_dof_vals(j) = 0.0;
        }
      }
      joule_heating_flow->SetSubVector(vdofs, elem_dof_vals);
      n0 += nsp;
    }
    joule_heating_flow->SetTrueVector();
    joule_heating_flow->SetFromTrueVector();
  }
#else
  mfem_error("Cannot interpolate without GSLIB support.");
#endif
}

void CycleAvgJouleCoupling::interpElectricFieldFromEMToFlow() {
  assert(efieldFES_);
  const bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Interpolating electric field to flow mesh.\n");

#ifdef HAVE_GSLIB
  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;
  interpolationPoints(vxyz, n_flow_interp_nodes_, efieldFES_);

  // Evaluate source grid function.
  Vector interp_vals(n_flow_interp_nodes_ * efield_ncomp_);

  const ParGridFunction *efield_real_gf = qmsa_solver_->getElectricFieldreal();
  interp_em_to_flow_->Interpolate(vxyz, *efield_real_gf, interp_vals);

  const ParFiniteElementSpace *flow_fespace = flow_solver_->getFESpace();
  if (flow_fespace->IsDGSpace()) {
    efieldR_->SetFromTrueDofs(interp_vals);
  } else {
    assert(false);
  }
  efieldR_->HostRead();

  const ParGridFunction *efield_imag_gf = qmsa_solver_->getElectricFieldimag();
  interp_em_to_flow_->Interpolate(vxyz, *efield_imag_gf, interp_vals);
  if (flow_fespace->IsDGSpace()) {
    efieldI_->SetFromTrueDofs(interp_vals);
  } else {
    assert(false);
  }
  efieldI_->HostRead();
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
    // update the power if necessary
    double delta_power = 0;
    if (input_power_ > 0) {
      delta_power = (input_power_ - initial_input_power_) * static_cast<double>(solve_em_every_n_) /
                    static_cast<double>(max_iters_);
    }

    // evaluate electric conductivity and interpolate it to EM mesh
    if (!fixed_conductivity_) flow_solver_->evaluatePlasmaConductivityGF();
    interpConductivityFromFlowToEM();

    // solve the EM system
    qmsa_solver_->solve();

    // report the "raw" Joule heating
    const double tot_jh = qmsa_solver_->totalJouleHeating();
    if (rank0_) {
      grvy_printf(GRVY_INFO, "The total input Joule heating = %.6e\n", tot_jh);
    }

    if (qmsa_solver_->evalRplasma()) {
      const double tot_I = qmsa_solver_->coilCurrent();

      // the effective plasma resistance = < Sjoule > / < I^2 >, where < * > is the cycle average
      //
      // the cycle-average of the current squared = 2 * Re(\hat{I})^2,
      // where \hat{I} is the Fourier coefficient of the driving current
      const double Rplasma = tot_jh / (2 * tot_I * tot_I);
      if (rank0_) {
        grvy_printf(GRVY_INFO, "The input current amplitude = %.6e\n", 2 * tot_I);
        grvy_printf(GRVY_INFO, "The effective plasma resistance = %.6e\n", Rplasma);
      }

      const double magnetic_energy = qmsa_solver_->magneticEnergy();
      const double Lplasma = 2 * magnetic_energy / (2 * tot_I * tot_I);
      if (rank0_) {
        grvy_printf(GRVY_INFO, "The magnetic field energy = %.6e\n", magnetic_energy);
        grvy_printf(GRVY_INFO, "The effective plasma inductance = %.6e\n", Lplasma);
      }
    }

    // scale the Joule heating (if we are controlling the power input)
    if (input_power_ > 0) {
      double target_power = initial_input_power_ + (current_iter_ / solve_em_every_n_ + 1) * delta_power;
      if (oscillating_power_) {
        const double tau = ((double)current_iter_) / power_period_;
        target_power = input_power_ + power_amplitude_ * sin(2 * M_PI * tau);
        if (rank0_) {
          grvy_printf(GRVY_INFO, "target_power = %.6e\n", target_power);
        }
      }
      const double ratio = target_power / tot_jh;
      qmsa_solver_->scaleJouleHeating(ratio);
      const double upd_jh = qmsa_solver_->totalJouleHeating();
      if (rank0_) {
        grvy_printf(GRVY_INFO, "current_iter = %d\n", current_iter_);
        grvy_printf(GRVY_INFO, "The total input Joule heating after scaling = %.6e\n", upd_jh);
      }
    }

    // interpolate the Joule heating to the flow mesh
    interpJouleHeatingFromEMToFlow();
    if (efieldFES_) interpElectricFieldFromEMToFlow();
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

/// Push solver variables to interface
void CycleAvgJouleCoupling::initInterface(TPS::Tps2Boltzmann &interface) {
  assert(!interface.IsInitialized());
  interface.init(flow_solver_);
  qmsa_solver_->setStoreE(true);
  // Get the FEC of the efield from the flow solver
  efieldFEC_ = flow_solver_->getFEC();
  // Set up efield function space
  if (efieldFES_) delete efieldFES_;
  efield_ncomp_ = interface.NeFieldComps() / 2;
  efieldFES_ =
      new mfem::ParFiniteElementSpace(flow_solver_->getMesh(), efieldFEC_, efield_ncomp_ * 2, mfem::Ordering::byNODES);
  if (efieldFES1_) delete efieldFES1_;
  efieldFES1_ =
      new mfem::ParFiniteElementSpace(flow_solver_->getMesh(), efieldFEC_, efield_ncomp_, mfem::Ordering::byNODES);
  if (efield_) delete efield_;
  efield_ = new mfem::ParGridFunction(efieldFES_);
  if (efieldR_) delete efieldR_;
  efieldR_ = new mfem::ParGridFunction(efieldFES1_, *efield_, 0);
  if (efieldI_) delete efieldI_;
  efieldI_ = new mfem::ParGridFunction(efieldFES1_, *efield_, efieldFES1_->GetVSize());
}

/// Push solver variables to interface
void CycleAvgJouleCoupling::push(TPS::Tps2Boltzmann &interface) {
  flow_solver_->push(interface);
  interface.interpolateFromNativeFES(*efield_, TPS::Tps2Boltzmann::Index::ElectricField);
}

/// Fetch solver variables from interface
void CycleAvgJouleCoupling::fetch(TPS::Tps2Boltzmann &interface) { flow_solver_->fetch(interface); }
