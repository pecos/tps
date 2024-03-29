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

///////////////////////////////
///  Add description later ////
///////////////////////////////

#include "loMach.hpp"

#include <grvy.h>
#include <hdf5.h>

#include <fstream>
#include <iomanip>
#include <limits>

#include "algebraicSubgridModels.hpp"
#include "algebraic_rans.hpp"
#include "calorically_perfect.hpp"
#include "gaussianInterpExtData.hpp"
#include "geometricSponge.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "tomboulides.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

LoMachSolver::LoMachSolver(TPS::Tps *tps)
    : tpsP_(tps),
      groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      rank0_(groupsMPI->isWorldRoot()) {
  // is this needed?
  groupsMPI->init();

  // set default solver state
  exit_status_ = NORMAL;

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }

  // for timer purposes
  tlast_ = 0.0;
}

LoMachSolver::~LoMachSolver() {
  // allocated in initialize()
  delete pvdc_;
  delete sfes_;
  delete sfec_;
  delete extData_;
  delete flow_;
  delete thermo_;
  delete sponge_;
  delete turbModel_;
  delete pmesh_;
  delete serial_mesh_;

  // allocated in constructor
  delete groupsMPI;
}

void LoMachSolver::initialize() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing loMach solver.\n");

  // Stash the order, just for convenience
  order = loMach_opts_.order;

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // Determine domain bounding box size
  {
    Mesh temp_mesh = Mesh(loMach_opts_.mesh_file.c_str());
    int dim = temp_mesh.Dimension();
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes(&temp_mesh, &fec);
    GridFunction coordsVert(&fes);
    temp_mesh.GetVertices(coordsVert);
    int nVert = coordsVert.Size() / dim;
    {
      double local_xmin = 1.0e18;
      double local_ymin = 1.0e18;
      double local_zmin = 1.0e18;
      double local_xmax = -1.0e18;
      double local_ymax = -1.0e18;
      double local_zmax = -1.0e18;

      double coords[3];
      for (int d = 0; d < 3; d++) {
        coords[d] = 0.0;
      }

      for (int n = 0; n < nVert; n++) {
        auto hcoords = coordsVert.HostRead();
        for (int d = 0; d < dim; d++) {
          coords[d] = hcoords[n + d * nVert];
        }
        local_xmin = min(coords[0], local_xmin);
        local_ymin = min(coords[1], local_ymin);
        if (dim == 3) {
          local_zmin = min(coords[2], local_zmin);
        }
        local_xmax = max(coords[0], local_xmax);
        local_ymax = max(coords[1], local_ymax);
        if (dim == 3) {
          local_zmax = max(coords[2], local_zmax);
        }
      }
      xmin_ = local_xmin;
      xmax_ = local_xmax;
      ymin_ = local_ymin;
      ymax_ = local_ymax;
      zmin_ = local_zmin;
      zmax_ = local_zmax;
    }

    if (loMach_opts_.periodicX) {
      loMach_opts_.x_trans = xmax_ - xmin_;
    }
    if (loMach_opts_.periodicY) {
      loMach_opts_.y_trans = ymax_ - ymin_;
    }
    if (loMach_opts_.periodicZ) {
      loMach_opts_.z_trans = zmax_ - zmin_;
    }
    if (loMach_opts_.periodicX || loMach_opts_.periodicY || loMach_opts_.periodicZ) {
      loMach_opts_.periodic = true;
    }
  }

  // Generate serial mesh, making it periodic if requested
  if (loMach_opts_.periodic) {
    Mesh temp_mesh = Mesh(loMach_opts_.mesh_file.c_str());
    Vector x_translation({loMach_opts_.x_trans, 0.0, 0.0});
    Vector y_translation({0.0, loMach_opts_.y_trans, 0.0});
    Vector z_translation({0.0, 0.0, loMach_opts_.z_trans});
    std::vector<Vector> translations = {x_translation, y_translation, z_translation};

    if (rank0_) {
      std::cout << " Making the mesh periodic using the following offsets:" << std::endl;
      std::cout << "   xTrans: " << loMach_opts_.x_trans << std::endl;
      std::cout << "   yTrans: " << loMach_opts_.y_trans << std::endl;
      std::cout << "   zTrans: " << loMach_opts_.z_trans << std::endl;
    }

    serial_mesh_ =
        new Mesh(std::move(Mesh::MakePeriodic(temp_mesh, temp_mesh.CreatePeriodicVertexMapping(translations))));
  } else {
    serial_mesh_ = new Mesh(loMach_opts_.mesh_file.c_str());
  }
  if (verbose) grvy_printf(ginfo, "Mesh read...\n");

  // Scale domain (NB: after periodicity applied)
  if (loMach_opts_.scale_mesh != 1) {
    if (verbose) grvy_printf(ginfo, "Scaling mesh factor of scale_mesh = %.6e\n", loMach_opts_.scale_mesh);
    serial_mesh_->EnsureNodes();
    GridFunction *nodes = serial_mesh_->GetNodes();
    *nodes *= loMach_opts_.scale_mesh;
  }

  // Stash mesh dimension (convenience)
  dim_ = serial_mesh_->Dimension();

  // Only support number of velocity components = spatial dimension for now
  nvel_ = dim_;

  // check if a simulation is being restarted
  if (loMach_opts_.io_opts_.enable_restart_) {
    // uniform refinement, user-specified number of times
    for (int l = 0; l < loMach_opts_.ref_levels; l++) {
      if (rank0_) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh_->UniformRefinement();
    }

    // read partitioning info from original decomposition (unless restarting from serial soln)
    nelemGlobal_ = serial_mesh_->GetNE();
    // nelemGlobal_ = mesh->GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      assert(!loMach_opts_.io_opts_.restart_serial_read_);
      // TODO(trevilo): Add support for serial read/write
      partitioning_file_hdf5("read", groupsMPI, nelemGlobal_, partitioning_);
    }

  } else {
    // remove previous solution
    if (rank0_) {
      string command = "rm -r ";
      command.append(loMach_opts_.io_opts_.output_dir_);
      int err = system(command.c_str());
      if (err != 0) {
        cout << "Error deleting previous data in " << loMach_opts_.io_opts_.output_dir_ << endl;
      }
    }

    // uniform refinement, user-specified number of times
    for (int l = 0; l < loMach_opts_.ref_levels; l++) {
      if (rank0_) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh_->UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh_->GetNE();
    if (nprocs_ > 1) {
      assert(serial_mesh_->Conforming());
      partitioning_ = Array<int>(serial_mesh_->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write", groupsMPI, nelemGlobal_, partitioning_);
    }

    // When starting from scratch time = 0 and iter = 0
    temporal_coeff_.time = 0.;
    iter = 0;
  }

  // Partition the mesh
  pmesh_ = new ParMesh(groupsMPI->getTPSCommWorld(), *serial_mesh_, partitioning_);
  if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");

  //-----------------------------------------------------
  // 2) Prepare the required finite elements (for resolution info)
  //    and other mesh related data
  //-----------------------------------------------------

  // TODO(trevilo): This code is necessary here b/c LoMachSolver owns
  // the grid resolution objects.  We should refactor them into a
  // "MeshHelper" class or some such thing.
  sfec_ = new H1_FECollection(order);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  if (verbose) grvy_printf(ginfo, "Spaces constructed...\n");

  // int sfes_truevsize = sfes_->GetTrueVSize();
  // if (verbose) grvy_printf(ginfo, "Got sizes...\n");

  /*
  gridScaleSml.SetSize(sfes_truevsize);
  gridScaleSml = 0.0;
  gridScaleXSml.SetSize(sfes_truevsize);
  gridScaleXSml = 0.0;
  gridScaleYSml.SetSize(sfes_truevsize);
  gridScaleYSml = 0.0;
  gridScaleZSml.SetSize(sfes_truevsize);
  gridScaleZSml = 0.0;

  // for paraview
  resolution_gf.SetSpace(sfes);
  resolution_gf = 0.0;
  */

  if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");

  // TODO(swh): not sure where this shoudl go but until dt calcs are
  // farmed out, leaving in loMach for time being
  // Determine the minimum element size.
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmin = min(pmesh_->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
  }
  // if (verbose) grvy_printf(ginfo, "Min element size found...\n");

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmax = max(pmesh_->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  }
  // if (verbose) grvy_printf(ginfo, "Max element size found...\n");

  if (rank0_) cout << "Maximum element size: " << hmax << "m" << endl;
  if (rank0_) cout << "Minimum element size: " << hmin << "m" << endl;

  //-----------------------------------------------------
  // 3) Prepare the sub-physics models/solvers
  //-----------------------------------------------------

  // Instantiate sponge
  sponge_ = new GeometricSponge(pmesh_, &loMach_opts_, tpsP_);

  // Instantiate external data
  extData_ = new GaussianInterpExtData(pmesh_, &loMach_opts_, tpsP_);

  // Instantiate turbulence model
  if (loMach_opts_.turb_opts_.turb_model_type_ == TurbulenceModelOptions::SMAGORINSKY) {
    turbModel_ = new AlgebraicSubgridModels(pmesh_, &loMach_opts_, tpsP_, 1);
  } else if (loMach_opts_.turb_opts_.turb_model_type_ == TurbulenceModelOptions::SIGMA) {
    turbModel_ = new AlgebraicSubgridModels(pmesh_, &loMach_opts_, tpsP_, 2);
  } else if (loMach_opts_.turb_opts_.turb_model_type_ == TurbulenceModelOptions::ALGEBRAIC_RANS) {
    turbModel_ = new AlgebraicRans(serial_mesh_, pmesh_, partitioning_, loMach_opts_.order, tpsP_);
  } else if (loMach_opts_.turb_opts_.turb_model_type_ == TurbulenceModelOptions::NONE) {
    // default
    turbModel_ = new ZeroTurbModel(pmesh_, loMach_opts_.order);
  } else {
    // Unknown choice... die
    if (rank0_) {
      grvy_printf(GRVY_ERROR, "Unknown loMach/turbulence-model option \n");
    }
    exit(ERROR);
  }

  // Instantiate thermochemical model
  if (loMach_opts_.thermo_solver == "constant-property") {
    thermo_ = new ConstantPropertyThermoChem(pmesh_, loMach_opts_.order, tpsP_);
  } else if (loMach_opts_.thermo_solver == "calorically-perfect") {
    thermo_ = new CaloricallyPerfectThermoChem(pmesh_, &loMach_opts_, temporal_coeff_, tpsP_);
  } else {
    // Unknown choice... die
    if (rank0_) {
      grvy_printf(GRVY_ERROR, "Unknown loMach/thermo-solver option > %s\n", loMach_opts_.thermo_solver.c_str());
    }
    exit(ERROR);
  }

  // Instantiate flow solver
  if (loMach_opts_.flow_solver == "zero-flow") {
    // No flow---set u = 0.  Primarily useful for testing thermochem models in isolation
    flow_ = new ZeroFlow(pmesh_, 1);
  } else if (loMach_opts_.flow_solver == "tomboulides") {
    // Tomboulides flow solver
    flow_ = new Tomboulides(pmesh_, loMach_opts_.order, loMach_opts_.order, temporal_coeff_, tpsP_);
  } else {
    // Unknown choice... die
    if (rank0_) {
      grvy_printf(GRVY_ERROR, "Unknown loMach/flow-solver option > %s\n", loMach_opts_.flow_solver.c_str());
    }
    exit(ERROR);
  }

  // Initialize time marching coefficients.  NB: dt will be reset
  // prior to time step, but must be initialized here in order to
  // avoid possible uninitialized usage when constructing operators in
  // model classes.
  temporal_coeff_.dt = 1.0;
  temporal_coeff_.dt3 = temporal_coeff_.dt2 = temporal_coeff_.dt1 = temporal_coeff_.dt;
  SetTimeIntegrationCoefficients(0);

  CFL = loMach_opts_.ts_opts_.cfl_;
  if (verbose) grvy_printf(ginfo, "got CFL...\n");

  // read-in external data if requested in bc setting
  extData_->initializeSelf();
  extData_->setup();
  flow_->initializeFromExtData(&extData_->toFlow_interface_);
  thermo_->initializeFromExtData(&extData_->toThermoChem_interface_);

  // Initialize model-owned data
  sponge_->initializeSelf();
  turbModel_->initializeSelf();
  flow_->initializeSelf();
  thermo_->initializeSelf();

  // Initialize restart read/write capability
  flow_->initializeIO(ioData);
  thermo_->initializeIO(ioData);

  const bool restart_serial =
      (loMach_opts_.io_opts_.restart_serial_read_ || loMach_opts_.io_opts_.restart_serial_write_);
  ioData.initializeSerial(rank0_, restart_serial, serial_mesh_, locToGlobElem, &partitioning_);
  MPI_Barrier(groupsMPI->getTPSCommWorld());
  if (verbose) grvy_printf(ginfo, "ioData.init thingy...\n");

  // If restarting, read restart files
  if (loMach_opts_.io_opts_.enable_restart_) {
    restart_files_hdf5("read");
  }

  // Exchange interface information
  turbModel_->initializeFromThermoChem(&thermo_->toTurbModel_interface_);
  turbModel_->initializeFromFlow(&flow_->toTurbModel_interface_);
  flow_->initializeFromTurbModel(&turbModel_->toFlow_interface_);
  thermo_->initializeFromTurbModel(&turbModel_->toThermoChem_interface_);
  flow_->initializeFromThermoChem(&thermo_->toFlow_interface_);
  thermo_->initializeFromFlow(&flow_->toThermoChem_interface_);
  flow_->initializeFromSponge(&sponge_->toFlow_interface_);
  thermo_->initializeFromSponge(&sponge_->toThermoChem_interface_);

  // static sponge
  sponge_->setup();

  // Finish initializing operators
  turbModel_->setup();
  turbModel_->initializeOperators();
  flow_->initializeOperators();
  thermo_->initializeOperators();
  // if(rank0_) {std::cout << "check: ops set..." << endl;}

  // TODO(trevilo): Enable averaging.  See note in loMach.hpp

  // Initialize visualization
  pvdc_ = new ParaViewDataCollection(loMach_opts_.io_opts_.output_dir_, pmesh_);
  pvdc_->SetDataFormat(VTKFormat::BINARY32);
  pvdc_->SetHighOrderOutput(true);
  pvdc_->SetLevelsOfDetail(order);

  turbModel_->initializeViz(*pvdc_);
  flow_->initializeViz(*pvdc_);
  thermo_->initializeViz(*pvdc_);
  sponge_->initializeViz(*pvdc_);
  extData_->initializeViz(*pvdc_);
}

void LoMachSolver::UpdateTimestepHistory(double dt) {
  // Rotate values in time step history
  // dthist[2] = dthist[1];
  // dthist[1] = dthist[0];
  // dthist[0] = dt;

  temporal_coeff_.dt3 = temporal_coeff_.dt2;
  temporal_coeff_.dt2 = temporal_coeff_.dt1;
  temporal_coeff_.dt1 = dt;
}

void LoMachSolver::initialTimeStep() {
  // TODO(trevilo): Compute initial dt from CFL
  // This calc was broken here, so it was removed temporarily.

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = loMach_opts_.ts_opts_.constant_dt_;
  if (dt_fixed > 0) {
    temporal_coeff_.dt = dt_fixed;
  }
}

void LoMachSolver::solveBegin() {
  setTimestep();
  if (rank0_) std::cout << "Initial dt = " << temporal_coeff_.dt << std::endl;

  pvdc_->SetCycle(iter);
  pvdc_->SetTime(temporal_coeff_.time);
  pvdc_->Save();
  if (rank0_ == true) std::cout << "Saving first step to paraview: " << iter << endl;

  iter_start_ = iter;
  if (rank0_ == true) {
    std::cout << "Starting main loop, from " << iter_start_ << " to " << loMach_opts_.max_steps_ << endl;
  }

  std::vector<std::string> thermo_header;
  thermo_->screenHeader(thermo_header);

  std::vector<std::string> flow_header;
  flow_->screenHeader(flow_header);

  // NB: Called on all ranks, but only rank 0 prints.  We assume any
  // necessary communication is handled by the flow class
  std::vector<double> thermo_screen_values;
  thermo_->screenValues(thermo_screen_values);

  std::vector<double> flow_screen_values;
  flow_->screenValues(flow_screen_values);

  if (rank0_) {
    // TODO(trevilo): Add state summary
    std::cout << std::endl;
    std::cout << std::setw(11) << "#     Step ";
    std::cout << std::setw(13) << "Time ";
    std::cout << std::setw(13) << "dt ";
    std::cout << std::setw(13) << "Wtime/Step ";
    for (size_t i = 0; i < thermo_header.size(); i++) {
      std::cout << std::setw(12) << thermo_header[i] << " ";
    }
    for (size_t i = 0; i < flow_header.size(); i++) {
      std::cout << std::setw(12) << flow_header[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "#==================================================================" << std::endl;

    std::cout << std::setw(10) << iter << " ";
    std::cout << std::setw(10) << std::scientific << temporal_coeff_.time << " ";
    std::cout << std::setw(10) << std::scientific << temporal_coeff_.dt << " ";
    std::cout << std::setw(10) << std::scientific << 0.0 << " ";
    for (size_t i = 0; i < thermo_screen_values.size(); i++) {
      std::cout << std::setw(10) << std::scientific << thermo_screen_values[i] << " ";
    }
    for (size_t i = 0; i < flow_screen_values.size(); i++) {
      std::cout << std::setw(10) << std::scientific << flow_screen_values[i] << " ";
    }
    std::cout << std::endl;
  }
}

void LoMachSolver::solveStep() {
  sw_step.Start();
  if (loMach_opts_.ts_opts_.integrator_type_ == LoMachTemporalOptions::CURL_CURL) {
    SetTimeIntegrationCoefficients(iter - iter_start_);
    thermo_->step();
    flow_->step();
    turbModel_->step();
  } else {
    if (rank0_) std::cout << "Time integration not updated." << endl;
    exit(1);
  }
  sw_step.Stop();

  UpdateTimestepHistory(temporal_coeff_.dt);
  temporal_coeff_.time += temporal_coeff_.dt;
  iter++;

  if ((iter % loMach_opts_.timing_frequency_) == 0) {
    double time_per_step = (sw_step.RealTime() - tlast_) / loMach_opts_.timing_frequency_;
    tlast_ = sw_step.RealTime();

    double max_time_per_step = 0.0;
    MPI_Reduce(&time_per_step, &max_time_per_step, 1, MPI_DOUBLE, MPI_MAX, 0, groupsMPI->getTPSCommWorld());

    // NB: Called on all ranks, but only rank 0 prints.  We assume any
    // necessary communication is handled by the flow class
    std::vector<double> thermo_screen_values;
    thermo_->screenValues(thermo_screen_values);

    std::vector<double> flow_screen_values;
    flow_->screenValues(flow_screen_values);

    if (rank0_) {
      // TODO(trevilo): Add state summary
      std::cout << std::setw(10) << iter << " ";
      std::cout << std::setw(10) << std::scientific << temporal_coeff_.time << " ";
      std::cout << std::setw(10) << std::scientific << temporal_coeff_.dt << " ";
      std::cout << std::setw(10) << std::scientific << max_time_per_step << " ";
      for (size_t i = 0; i < thermo_screen_values.size(); i++) {
        std::cout << std::setw(10) << std::scientific << thermo_screen_values[i] << " ";
      }
      for (size_t i = 0; i < flow_screen_values.size(); i++) {
        std::cout << std::setw(10) << std::scientific << flow_screen_values[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  // restart files
  if (iter % loMach_opts_.output_frequency_ == 0 && iter != 0) {
    // Write restart file!
    restart_files_hdf5("write");

    // Write visualization files
    pvdc_->SetCycle(iter);
    pvdc_->SetTime(temporal_coeff_.time);
    pvdc_->Save();
  }

  // check for DIE
  if (iter % loMach_opts_.io_opts_.exit_check_frequency_ == 0) {
    int early_exit = 0;
    if (rank0_) {
      if (file_exists("DIE")) {
        grvy_printf(gdebug, "Caught DIE file. Exiting...\n");
        remove("DIE");
        early_exit = 1;
      }
    }
    MPI_Bcast(&early_exit, 1, MPI_INT, 0, groupsMPI->getTPSCommWorld());
    if (early_exit == 1) exit(-1);
  }

  // update dt
  // if (dt_fixed < 0.0) {
  if (loMach_opts_.ts_opts_.constant_dt_ < 0.0) {
    updateTimestep();
  }
}

void LoMachSolver::solveEnd() {
  // evaluate error (if possible)
  const double flow_err = flow_->computeL2Error();
  if (flow_err >= 0.0) {
    if (rank0_) std::cout << "At time = " << temporal_coeff_.time << ", flow L2 error = " << flow_err << std::endl;
  }

  // paraview
  pvdc_->SetCycle(iter);
  pvdc_->SetTime(temporal_coeff_.time);
  if (rank0_ == true) std::cout << " Saving final step to paraview: " << iter << "... " << endl;
  pvdc_->Save();
  MPI_Barrier(groupsMPI->getTPSCommWorld());
  if (rank0_ == true) std::cout << " ...complete!" << endl;
}

void LoMachSolver::solve() {
  this->solveBegin();

  while (iter < loMach_opts_.max_steps_) {
    this->solveStep();
  }

  this->solveEnd();
}

void LoMachSolver::updateTimestep() {
  // minimum timestep to not waste comp time
  double dtMin = 1.0e-9;

  double Umax_lcl = 1.0e-12;
  double max_speed = Umax_lcl;
  double Umag;
  int Sdof = sfes_->GetNDofs();
  // TODO(trevilo): Let user set dtFactor
  double dtFactor = 1.0;
  auto dataU = flow_->getCurrentVelocity()->HostRead();

  // come in divided by order
  // const double *dataD = bufferGridScale->HostRead();
  const double *dataD = (turbModel_->getGridScale())->HostRead();

  for (int n = 0; n < Sdof; n++) {
    Umag = 0.0;
    Vector delta({dataD[n], dataD[n], dataD[n]});  // use smallest delta for all
    for (int eq = 0; eq < dim_; eq++) {
      Umag += (dataU[n + eq * Sdof] / delta[eq]) * (dataU[n + eq * Sdof] / delta[eq]);
    }
    Umag = std::sqrt(Umag);
    Umax_lcl = std::max(Umag, Umax_lcl);
  }

  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  double dtInst_conv = 0.5 * CFL / max_speed;

  // double *dGradU = flowClass->gradU.HostReadWrite();
  // double *dGradV = flowClass->gradV.HostReadWrite();
  // double *dGradW = flowClass->gradW.HostReadWrite();
  // double *dVisc = tcClass->viscSml.HostReadWrite();
  // double *rho = tcClass->rn.HostReadWrite();
  // Umax_lcl = 1.0e-12;
  // for (int n = 0; n < SdofInt; n++) {
  //   DenseMatrix gU;
  //   gU.SetSize(nvel_, dim_);
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(0, dir) = dGradU[n + dir * SdofInt];
  //   }
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(1, dir) = dGradV[n + dir * SdofInt];
  //   }
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(2, dir) = dGradW[n + dir * SdofInt];
  //   }
  //   double duMag = 0.0;
  //   for (int dir = 0; dir < dim_; dir++) {
  //     for (int eq = 0; eq < nvel_; eq++) {
  //       duMag += gU(eq, dir) * gU(eq, dir);
  //     }
  //   }
  //   duMag = sqrt(duMag);
  //   double viscVel = sqrt(duMag * dVisc[n] / rho[n]);
  //   viscVel /= dataD[n];
  //   Umax_lcl = std::max(viscVel, Umax_lcl);
  // }
  // MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  // double dtInst_visc = 0.5 * CFL / max_speed;

  // double dtInst = max(dtInst_conv, dtInst_visc);

  double dtInst = dtInst_conv;

  double &dt = temporal_coeff_.dt;
  if (dtInst > dt) {
    dt = dt * (1.0 + dtFactor);
    dt = std::min(dt, dtInst);
  } else if (dtInst < dt) {
    dt = dtInst;
  }

  if (dt < dtMin) {
    if (rank0_) {
      std::cout << " => Timestep running away!!!" << endl;
    }
    exit(ERROR);
  }
}

double LoMachSolver::computeCFL() {
  const double dt = temporal_coeff_.dt;

  double Umax_lcl = 1.0e-12;
  double max_speed = Umax_lcl;
  double Umag;
  int Sdof = sfes_->GetNDofs();

  // come in divided by order
  // double *dataD = bufferGridScale->HostReadWrite();
  double *dataD = (turbModel_->getGridScale())->HostReadWrite();
  auto dataU = flow_->getCurrentVelocity()->HostRead();

  for (int n = 0; n < Sdof; n++) {
    Umag = 0.0;
    // Vector delta({dataX[n], dataY[n], dataZ[n]});
    Vector delta({dataD[n], dataD[n], dataD[n]});  // use smallest delta for all
    for (int eq = 0; eq < dim_; eq++) {
      Umag += (dataU[n + eq * Sdof] / delta[eq]) * (dataU[n + eq * Sdof] / delta[eq]);
    }
    Umag = std::sqrt(Umag);
    Umax_lcl = std::max(Umag, Umax_lcl);
  }
  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  double CFL_conv = 2.0 * dt * max_speed;

  // double *dGradU = flowClass->gradU.HostReadWrite();
  // double *dGradV = flowClass->gradV.HostReadWrite();
  // double *dGradW = flowClass->gradW.HostReadWrite();
  // double *dVisc = tcClass->viscSml.HostReadWrite();
  // double *rho = tcClass->rn.HostReadWrite();
  // Umax_lcl = 1.0e-12;
  // for (int n = 0; n < SdofInt; n++) {
  //   DenseMatrix gU;
  //   gU.SetSize(nvel_, dim_);
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(0, dir) = dGradU[n + dir * SdofInt];
  //   }
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(1, dir) = dGradV[n + dir * SdofInt];
  //   }
  //   for (int dir = 0; dir < dim_; dir++) {
  //     gU(2, dir) = dGradW[n + dir * SdofInt];
  //   }
  //   double duMag = 0.0;
  //   for (int dir = 0; dir < dim_; dir++) {
  //     for (int eq = 0; eq < nvel_; eq++) {
  //       duMag += gU(eq, dir) * gU(eq, dir);
  //     }
  //   }
  //   duMag = sqrt(duMag);
  //   double viscVel = sqrt(duMag * dVisc[n] / rho[n]);
  //   viscVel /= dataD[n];
  //   Umax_lcl = std::max(viscVel, Umax_lcl);
  // }
  // MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  // double CFL_visc = 2.0 * dt * max_speed;

  // double CFL_here = max(CFL_conv, CFL_visc);
  double CFL_here = CFL_conv;

  return CFL_here;
}

void LoMachSolver::setTimestep() {
  double Umax_lcl = 1.0e-12;
  double max_speed = Umax_lcl;
  double Umag;
  int Sdof = sfes_->GetNDofs();

  CFL = loMach_opts_.ts_opts_.cfl_;

  // dt_fixed is initialized to -1, so if it is positive,
  // then the user requested a fixed dt run
  const double dt_fixed = loMach_opts_.ts_opts_.constant_dt_;
  if (dt_fixed > 0) {
    temporal_coeff_.dt = dt_fixed;
  } else {
    // auto dataU = flowClass->un_gf.HostRead();
    auto dataU = flow_->getCurrentVelocity()->HostRead();
    for (int n = 0; n < Sdof; n++) {
      Umag = 0.0;
      for (int eq = 0; eq < dim_; eq++) {
        Umag += dataU[n + eq * Sdof] * dataU[n + eq * Sdof];
      }
      Umag = std::sqrt(Umag);
      Umax_lcl = std::max(Umag, Umax_lcl);
    }
    MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
    double dtInst = CFL * hmin / (max_speed * (double)order);
    temporal_coeff_.dt = dtInst;
    std::cout << "dt from setTimestep: " << temporal_coeff_.dt << " max_speed: " << max_speed << endl;
  }
}

void LoMachSolver::SetTimeIntegrationCoefficients(int step) {
  // Maximum BDF order to use at current time step
  // step + 1 <= order <= max_bdf_order
  int bdf_order = std::min(step + 1, max_bdf_order);

  // on first call, initialize time step history
  if (step == 0) {
    // set dt2 and dt3 to nan, so that we'll get nan everywhere if they're used inappropriately
    temporal_coeff_.dt2 = temporal_coeff_.dt3 = std::numeric_limits<double>::signaling_NaN();

    temporal_coeff_.dt1 = temporal_coeff_.dt;
  }

  // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
  double rho1 = 0.0;
  if (bdf_order >= 2) {
    rho1 = temporal_coeff_.dt1 / temporal_coeff_.dt2;
  }

  // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
  double rho2 = 0.0;
  if (bdf_order == 3) {
    rho2 = temporal_coeff_.dt2 / temporal_coeff_.dt3;
  }

  if (step == 0 && bdf_order == 1) {
    temporal_coeff_.bd0 = 1.0;
    temporal_coeff_.bd1 = -1.0;
    temporal_coeff_.bd2 = 0.0;
    temporal_coeff_.bd3 = 0.0;
    temporal_coeff_.ab1 = 1.0;
    temporal_coeff_.ab2 = 0.0;
    temporal_coeff_.ab3 = 0.0;
  } else if (step >= 1 && bdf_order == 2) {
    temporal_coeff_.bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
    temporal_coeff_.bd1 = -(1.0 + rho1);
    temporal_coeff_.bd2 = pow(rho1, 2.0) / (1.0 + rho1);
    temporal_coeff_.bd3 = 0.0;
    temporal_coeff_.ab1 = 1.0 + rho1;
    temporal_coeff_.ab2 = -rho1;
    temporal_coeff_.ab3 = 0.0;
  } else if (step >= 2 && bdf_order == 3) {
    temporal_coeff_.bd0 = 1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
    temporal_coeff_.bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
    temporal_coeff_.bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
    temporal_coeff_.bd3 =
        -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1)) / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
    temporal_coeff_.ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
    temporal_coeff_.ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
    temporal_coeff_.ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
  }
}

void LoMachSolver::PrintTimingData() {
  double my_rt[2], rt_max[2];

  my_rt[0] = sw_setup.RealTime();
  my_rt[1] = sw_step.RealTime();

  MPI_Reduce(my_rt, rt_max, 2, MPI_DOUBLE, MPI_MAX, 0, pmesh_->GetComm());

  if (pmesh_->GetMyRank() == 0) {
    mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP"
              << "\n";

    mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0] << std::setw(10) << my_rt[1] << "\n";

    mfem::out << std::setprecision(8);
  }
}

// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {
  // if (verbose) grvy_printf(ginfo, "parsing solver options...\n");
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
  tpsP_->getInput("loMach/scale-mesh", loMach_opts_.scale_mesh, 1.0);
  assert(loMach_opts_.scale_mesh > 0);

  tpsP_->getRequiredInput("loMach/flow-solver", loMach_opts_.flow_solver);
  assert(loMach_opts_.flow_solver == "zero-flow" || loMach_opts_.flow_solver == "tomboulides");

  tpsP_->getInput("loMach/thermo-solver", loMach_opts_.thermo_solver, string("constant-property"));
  assert(loMach_opts_.thermo_solver == "constant-property" || loMach_opts_.thermo_solver == "calorically-perfect");

  tpsP_->getInput("loMach/order", loMach_opts_.order, 1);
  assert(loMach_opts_.order >= 1);

  tpsP_->getInput("loMach/ref_levels", loMach_opts_.ref_levels, 0);
  tpsP_->getInput("loMach/max_iter", loMach_opts_.max_iter, 100);
  tpsP_->getInput("loMach/rtol", loMach_opts_.rtol, 1.0e-8);
  tpsP_->getInput("loMach/nFilter", loMach_opts_.nFilter, 0);
  tpsP_->getInput("loMach/filterWeight", loMach_opts_.filterWeight, 1.0);
  tpsP_->getInput("loMach/filterTemp", loMach_opts_.filterTemp, false);
  tpsP_->getInput("loMach/filterVel", loMach_opts_.filterVel, false);
  tpsP_->getInput("loMach/thermalDiv", loMach_opts_.thermalDiv, true);
  tpsP_->getInput("loMach/realDiv", loMach_opts_.realDiv, false);
  tpsP_->getInput("loMach/solveTemp", loMach_opts_.solveTemp, true);

  // dump options to screen for user inspection
  if (rank0_) {
    loMach_opts_.print(std::cout);
  }

  // TODO(trevilo): maintain in loMach for now (via LoMachOptions)
  tpsP_->getInput("loMach/maxIters", loMach_opts_.max_steps_, 10);
  tpsP_->getInput("loMach/outputFreq", loMach_opts_.output_frequency_, 50);
  tpsP_->getInput("loMach/timingFreq", loMach_opts_.timing_frequency_, 100);

  // Turbulence model options
  loMach_opts_.turb_opts_.read(tpsP_, std::string("loMach"));

  // time integration controls
  loMach_opts_.ts_opts_.read(tpsP_);
  max_bdf_order = loMach_opts_.ts_opts_.bdf_order_;

  // I/O settings
  loMach_opts_.io_opts_.read(tpsP_);
  loadFromAuxSol = loMach_opts_.io_opts_.restart_variable_order_;

  // periodicity
  tpsP_->getInput("periodicity/enablePeriodic", loMach_opts_.periodic, false);
  tpsP_->getInput("periodicity/xTrans", loMach_opts_.x_trans, 1.0e12);
  tpsP_->getInput("periodicity/yTrans", loMach_opts_.y_trans, 1.0e12);
  tpsP_->getInput("periodicity/zTrans", loMach_opts_.z_trans, 1.0e12);
  tpsP_->getInput("periodicity/periodicX", loMach_opts_.periodicX, false);
  tpsP_->getInput("periodicity/periodicY", loMach_opts_.periodicY, false);
  tpsP_->getInput("periodicity/periodicZ", loMach_opts_.periodicZ, false);
}
