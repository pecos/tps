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

#include "logger.hpp"
#include "tomboulides.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

LoMachSolver::LoMachSolver(LoMachOptions loMach_opts, TPS::Tps *tps)
    : tpsP_(tps),
      loMach_opts_(loMach_opts),
      groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      rank0_(groupsMPI->isWorldRoot()) {
  pmesh_ = NULL;

  // is this needed?
  groupsMPI->init();

  // ini file options
  parseSolverOptions();
  parseSolverOptions2();
  loadFromAuxSol = config.RestartFromAux();

  // set default solver state
  exit_status_ = NORMAL;

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }
}

LoMachSolver::~LoMachSolver() {
  // allocated in initialize()
  delete bufferGridScaleZ;
  delete bufferGridScaleY;
  delete bufferGridScaleX;
  delete bufferGridScale;
  delete sfes;
  delete sfec;
  delete flow_;
  delete thermo_;
  delete pmesh_;

  // allocated in constructor
  delete groupsMPI;
}

void LoMachSolver::initialize() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing loMach solver.\n");

  if (loMach_opts_.uOrder == -1) {
    order = std::max(config.solOrder, 1);
    porder = order;
    double no;
    no = ceil(((double)order * 1.5));
    norder = int(no);
  } else {
    order = loMach_opts_.uOrder;
    porder = loMach_opts_.pOrder;
    norder = loMach_opts_.nOrder;
  }

  if (rank0_) {
    std::cout << "  " << endl;
    std::cout << " Order of solution-spaces " << endl;
    std::cout << " +velocity : " << order << endl;
    std::cout << " +pressure : " << porder << endl;
    std::cout << " +non-linear : " << norder << endl;
    std::cout << "  " << endl;
  }

  // grid periodicity
  Vector x_translation({config.GetXTrans(), 0.0, 0.0});
  Vector y_translation({0.0, config.GetYTrans(), 0.0});
  Vector z_translation({0.0, 0.0, config.GetZTrans()});
  std::vector<Vector> translations = {x_translation, y_translation, z_translation};

  if (rank0_) {
    std::cout << " Making the mesh periodic using the following offsets:" << std::endl;
    std::cout << "   xTrans: " << config.GetXTrans() << std::endl;
    std::cout << "   yTrans: " << config.GetYTrans() << std::endl;
    std::cout << "   zTrans: " << config.GetZTrans() << std::endl;
  }

  // Read the serial mesh (on each mpi rank)
  Mesh mesh = Mesh(loMach_opts_.mesh_file.c_str());
  Mesh *serial_mesh = &mesh;
  if (verbose) grvy_printf(ginfo, "Mesh read...\n");

  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  Mesh periodic_mesh = Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations));
  if (verbose) grvy_printf(ginfo, "Mesh made periodic (if applicable)...\n");

  // underscore stuff is terrible
  dim_ = mesh.Dimension();

  // HACK HACK HACK HARD CODE
  nvel_ = dim_;

  if (rank0_) {
    if (config.sgsModelType == 1) {
      std::cout << "Smagorinsky subgrid model active" << endl;
    } else if (config.sgsModelType == 2) {
      std::cout << "Sigma subgrid model active" << endl;
    }
  }

  MaxIters = config.GetNumIters();

  // check if a simulation is being restarted
  if (config.GetRestartCycle() > 0) {
    if (config.GetUniformRefLevels() > 0) {
      if (rank0_) {
        std::cerr << "ERROR: Uniform mesh refinement not supported upon restart." << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // read partitioning info from original decomposition (unless restarting from serial soln)
    nelemGlobal_ = serial_mesh->GetNE();
    // nelemGlobal_ = mesh->GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      assert(!config.isRestartSerialized("read"));
      // TODO(trevilo): Add support for serial read/write
      partitioning_file_hdf5("read", config, groupsMPI, nelemGlobal_, partitioning_);
    }

  } else {
    // remove previous solution
    if (rank0_) {
      string command = "rm -r ";
      command.append(config.GetOutputName());
      int err = system(command.c_str());
      if (err != 0) {
        cout << "Error deleting previous data in " << config.GetOutputName() << endl;
      }
    }

    // uniform refinement, user-specified number of times
    for (int l = 0; l < config.GetUniformRefLevels(); l++) {
      if (rank0_) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh->UniformRefinement();
      // mesh->UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh->GetNE();
    if (nprocs_ > 1) {
      assert(serial_mesh->Conforming());
      partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write", config, groupsMPI, nelemGlobal_, partitioning_);
    }

    // make sure these are actually hooked up!
    temporal_coeff_.time = 0.;
    iter = 0;
  }

  // 1c) Partition the mesh (see partitioning_ in M2ulPhyS)
  // pmesh__ = new ParMesh(MPI_COMM_WORLD, *mesh);
  pmesh_ = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
  if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");

  // instantiate physics models
  // turbClass = new TurbModel(pmesh_, &config, &loMach_opts_);

  // TODO(trevilo): Instantiate based in input options
  thermo_ = new ConstantPropertyThermoChem(pmesh_, 1, 1.0, 1.0);

  if (loMach_opts_.flow_solver == "zero-flow") {
    // No flow---set u = 0.  Primarily useful for testing thermochem models in isolation
    flow_ = new ZeroFlow(pmesh_, 1);
  } else if (loMach_opts_.flow_solver == "tomboulides") {
    // Tomboulides flow solver
    flow_ = new Tomboulides(pmesh_, loMach_opts_.order, loMach_opts_.order, temporal_coeff_);
  } else {
    // Unknown choice... die
    if (rank0_) {
      grvy_printf(GRVY_ERROR, "Unknown loMach/flow-solver option > %s\n", loMach_opts_.flow_solver.c_str());
    }
    exit(ERROR);
  }

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec = new H1_FECollection(order);
  sfes = new ParFiniteElementSpace(pmesh_, sfec);

  if (verbose) grvy_printf(ginfo, "Spaces constructed...\n");

  int sfes_truevsize = sfes->GetTrueVSize();
  if (verbose) grvy_printf(ginfo, "Got sizes...\n");

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

  if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");

  // Initialize time marching coefficients.  NB: dt will be reset
  // prior to time step, but must be initialized here in order to
  // avoid possible uninitialized usage when constructing operators in
  // model classes.
  temporal_coeff_.dt = 1.0;
  temporal_coeff_.dt3 = temporal_coeff_.dt2 = temporal_coeff_.dt1 = temporal_coeff_.dt;
  SetTimeIntegrationCoefficients(0);

  // Initialize model-owned data
  flow_->initializeSelf();
  thermo_->initializeSelf();

  // Initialize restart read/write capability
  flow_->initializeIO(ioData);

  // Exchange interface information
  flow_->initializeFromThermoChem(&thermo_->toFlow_interface_);
  // thermo_->initializeFromFlow(&flow_->interface);

  // Finish initializing operators
  flow_->initializeOperators();
  // thermo_->initializeOperators();

  // TODO(trevilo): Enable averaging.  See note in loMach.hpp

  ioData.initializeSerial(rank0_, (config.RestartSerial() != "no"), serial_mesh, locToGlobElem, &partitioning_);
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose) grvy_printf(ginfo, "ioData.init thingy...\n");

  CFL = config.GetCFLNumber();
  if (verbose) grvy_printf(ginfo, "got CFL...\n");

  // Determine the minimum element size.
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmin = min(pmesh_->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
  }
  if (verbose) grvy_printf(ginfo, "Min element size found...\n");

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < pmesh_->GetNE(); i++) {
      local_hmax = max(pmesh_->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  }
  if (verbose) grvy_printf(ginfo, "Max element size found...\n");

  // Build grid size vector and grid function
  bufferGridScale = new ParGridFunction(sfes);
  bufferGridScaleX = new ParGridFunction(sfes);
  bufferGridScaleY = new ParGridFunction(sfes);
  bufferGridScaleZ = new ParGridFunction(sfes);
  ParGridFunction dofCount(sfes);
  {
    int elndofs;
    Array<int> vdofs;
    Vector vals;
    Vector loc_data;
    int nSize = bufferGridScale->Size();
    Array<int> zones_per_vdof;
    zones_per_vdof.SetSize(sfes->GetVSize());
    zones_per_vdof = 0;
    Array<int> zones_per_vdofALL;
    zones_per_vdofALL.SetSize(sfes->GetVSize());
    zones_per_vdofALL = 0;

    double *data = bufferGridScale->HostReadWrite();
    double *count = dofCount.HostReadWrite();

    for (int i = 0; i < sfes->GetNDofs(); i++) {
      data[i] = 0.0;
      count[i] = 0.0;
    }

    // element loop
    for (int e = 0; e < sfes->GetNE(); ++e) {
      sfes->GetElementVDofs(e, vdofs);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = sfes->GetElementTransformation(e);
      const FiniteElement *el = sfes->GetFE(e);
      elndofs = el->GetDof();
      double delta;

      // element dof
      for (int dof = 0; dof < elndofs; ++dof) {
        const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
        tr->SetIntPoint(&ip);
        delta = pmesh_->GetElementSize(tr->ElementNo, 1);
        delta = delta / ((double)order);
        vals(dof) = delta;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        data[ldof + 0 * nSize] += vals[j];
      }

      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        zones_per_vdof[ldof]++;
      }
    }

    // Count the zones globally.
    GroupCommunicator &gcomm = bufferGridScale->ParFESpace()->GroupComm();
    gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
    gcomm.Bcast(zones_per_vdof);

    // Accumulate for all vdofs.
    gcomm.Reduce<double>(bufferGridScale->GetData(), GroupCommunicator::Sum);
    gcomm.Bcast<double>(bufferGridScale->GetData());

    // Compute means.
    for (int i = 0; i < nSize; i++) {
      const int nz = zones_per_vdof[i];
      if (nz) {
        data[i] /= nz;
      }
    }

    bufferGridScale->GetTrueDofs(gridScaleSml);
    bufferGridScaleX->GetTrueDofs(gridScaleXSml);
    bufferGridScaleY->GetTrueDofs(gridScaleYSml);
    bufferGridScaleZ->GetTrueDofs(gridScaleZSml);
  }

  // Determine domain bounding box size
  ParGridFunction coordsVert(sfes);
  pmesh_->GetVertices(coordsVert);
  int nVert = coordsVert.Size() / dim_;
  {
    double local_xmin = 1.0e18;
    double local_ymin = 1.0e18;
    double local_zmin = 1.0e18;
    double local_xmax = -1.0e18;
    double local_ymax = -1.0e18;
    double local_zmax = -1.0e18;
    for (int n = 0; n < nVert; n++) {
      auto hcoords = coordsVert.HostRead();
      double coords[3];
      for (int d = 0; d < dim_; d++) {
        coords[d] = hcoords[n + d * nVert];
      }
      local_xmin = min(coords[0], local_xmin);
      local_ymin = min(coords[1], local_ymin);
      if (dim_ == 3) {
        local_zmin = min(coords[2], local_zmin);
      }
      local_xmax = max(coords[0], local_xmax);
      local_ymax = max(coords[1], local_ymax);
      if (dim_ == 3) {
        local_zmax = max(coords[2], local_zmax);
      }
    }
    MPI_Allreduce(&local_xmin, &xmin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
    MPI_Allreduce(&local_ymin, &ymin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
    MPI_Allreduce(&local_zmin, &zmin, 1, MPI_DOUBLE, MPI_MIN, pmesh_->GetComm());
    MPI_Allreduce(&local_xmax, &xmax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
    MPI_Allreduce(&local_ymax, &ymax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
    MPI_Allreduce(&local_zmax, &zmax, 1, MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());
  }

  //
  // if (config.GetRestartCycle() == 0) initialTimeStep();
  if (rank0_) cout << "Maximum element size: " << hmax << "m" << endl;
  if (rank0_) cout << "Minimum element size: " << hmin << "m" << endl;
}

// void LoMachSolver::Setup(double dt) {
//   sw_setup.Start();

//   // Set initial time step in the history array
//   dthist[0] = dt;

//   // if (config.sgsExcludeMean == true) { bufferMeanUp = new ParGridFunction(fvfes); }
//   MPI_Barrier(MPI_COMM_WORLD);
//   if (rank0_) std::cout << "Attempting to setup bufferMeanUp..." << endl;
//   bufferMeanUp = new ParGridFunction(fvfes2);
//   MPI_Barrier(MPI_COMM_WORLD);
//   if (rank0_) std::cout << "...complete" << endl;

//   sw_setup.Stop();
//   // std::cout << "Check 29..." << std::endl;
// }

void LoMachSolver::UpdateTimestepHistory(double dt) {
  // Rotate values in time step history
  // dthist[2] = dthist[1];
  // dthist[1] = dthist[0];
  // dthist[0] = dt;

  temporal_coeff_.dt3 = temporal_coeff_.dt2;
  temporal_coeff_.dt2 = temporal_coeff_.dt1;
  temporal_coeff_.dt1 = dt;
}

void LoMachSolver::projectInitialSolution() {
  // This whole function was commented out.
  // Keep as no-op for now, but do we need it?
}

void LoMachSolver::initialTimeStep() {
  // TODO(trevilo): Compute initial dt from CFL
  // This calc was broken here, so it was removed temporarily.

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    temporal_coeff_.dt = dt_fixed;
  }
}

void LoMachSolver::solve() {
  // just restart here
  if (config.restart) {
    restart_files_hdf5("read");
  }

  setTimestep();
  if (iter == 0) {
    temporal_coeff_.dt = std::min(config.dt_initial, temporal_coeff_.dt);
  }

  // temporary hardcodes
  double CFL_actual;
  CFL = config.GetCFLNumber();

  int SdofInt = sfes->GetTrueVSize();
  int Sdof = sfes->GetNDofs();

  // dt_fixed is initialized to -1, so if it is positive,
  // then the user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    temporal_coeff_.dt = dt_fixed;
  }

  if (rank0_) std::cout << "Initial dt = " << temporal_coeff_.dt << std::endl;

  // trevilo: What is this doing?
  {
    double *res = bufferGridScale->HostReadWrite();
    double *data = resolution_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = res[i];
    }
  }

  ParGridFunction *u_gf = flow_->getCurrentVelocity();
  // const ParGridFunction *p_gf = flow_->GetCurrentPressure();

  ParaViewDataCollection pvdc("output", pmesh_);
  pvdc.SetDataFormat(VTKFormat::BINARY32);
  pvdc.SetHighOrderOutput(true);
  pvdc.SetLevelsOfDetail(order);
  pvdc.SetCycle(iter);
  pvdc.SetTime(temporal_coeff_.time);
  pvdc.RegisterField("velocity", u_gf);
  // pvdc.RegisterField("pressure", p_gf);
  pvdc.Save();
  if (rank0_ == true) std::cout << "Saving first step to paraview: " << iter << endl;

  int iter_start = iter + 1;
  if (rank0_ == true) std::cout << " Starting main loop, from " << iter_start << " to " << MaxIters << endl;

  if (rank0_ == true) {
    if (dt_fixed < 0) {
      std::cout << " " << endl;
      std::cout << " N     Time        dt      time/step    minT    maxT    max|U|" << endl;
      std::cout << "==================================================================" << endl;
    } else {
      std::cout << " " << endl;
      std::cout << " N     Time        cfl      time/step    minT    maxT    max|U|" << endl;
      std::cout << "==================================================================" << endl;
    }
  }

  double time_previous = 0.0;
  for (int step = iter_start; step <= MaxIters; step++) {
    iter = step;

    // trevilo: necessary?
    if (step > MaxIters) {
      break;
    }

    sw_step.Start();
    if (config.timeIntegratorType == 1) {
      SetTimeIntegrationCoefficients(step - iter_start);
      thermo_->step();
      flow_->step();
    } else if (config.timeIntegratorType == 2) {
      // staggeredTimeStep(time, dt, step, iter_start);
      if (rank0_) std::cout << "Time integration not updated." << endl;
      exit(1);
    } else if (config.timeIntegratorType == 3) {
      // deltaPStep(time, dt, step, iter_start);
      if (rank0_) std::cout << "Time integration not updated." << endl;
      exit(1);
    }
    sw_step.Stop();

    // if ((step < 10) || (step < 100 && step % 10 == 0) || (step % 100 == 0)) {
    //   double maxT, minT, maxU;
    //   double maxTall, minTall, maxUall;
    //   maxT = -1.0e12;
    //   minT = +1.0e12;
    //   maxU = 0.0;
    //   {
    //     double *d_T = tcClass->Tn.HostReadWrite();
    //     double *d_u = flowClass->un.HostReadWrite();
    //     for (int i = 0; i < SdofInt; i++) {
    //       maxT = max(d_T[i], maxT);
    //       minT = min(d_T[i], minT);
    //     }
    //     for (int i = 0; i < SdofInt; i++) {
    //       double Umag = d_u[i + 0 * SdofInt] * d_u[i + 0 * SdofInt] + d_u[i + 1 * SdofInt] * d_u[i + 1 * SdofInt] +
    //                     d_u[i + 2 * SdofInt] * d_u[i + 2 * SdofInt];
    //       Umag = sqrt(Umag);
    //       maxU = max(Umag, maxU);
    //     }
    //   }
    //   MPI_Allreduce(&minT, &minTall, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    //   MPI_Allreduce(&maxT, &maxTall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //   MPI_Allreduce(&maxU, &maxUall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    //   if (dt_fixed < 0) {
    //     if (rank0_ == true) {
    //       std::cout << " " << step << "    " << time << "   " << dt << "   " << sw_step.RealTime() - time_previous
    //                 << "   " << minTall << "   " << maxTall << "   " << maxUall << endl;
    //     }

    //   } else {
    //     CFL_actual = computeCFL(dt);
    //     if (rank0_ == true) {
    //       std::cout << " " << step << "    " << time << "   " << CFL_actual << "   "
    //                 << sw_step.RealTime() - time_previous << "   " << minTall << "   " << maxTall << "   " << maxUall
    //                 << endl;
    //     }
    //   }
    // }
    time_previous = sw_step.RealTime();

    // update dt
    if (dt_fixed < 0.0) {
      updateTimestep();
    }

    // restart files
    if (iter % config.itersOut == 0 && iter != 0) {
      restart_files_hdf5("write");
    }

    // check for DIE
    if (iter % 100 == 0) {
      if (rank0_) {
        if (file_exists("DIE")) {
          grvy_printf(gdebug, "Caught DIE file. Exiting...\n");
          remove("DIE");
          earlyExit = 1;
        }
      }
      MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (earlyExit == 1) exit(-1);
    }

    UpdateTimestepHistory(temporal_coeff_.dt);
    temporal_coeff_.time += temporal_coeff_.dt;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // paraview
  pvdc.SetCycle(iter);
  pvdc.SetTime(temporal_coeff_.time);
  if (rank0_ == true) std::cout << " Saving final step to paraview: " << iter << "... " << endl;
  pvdc.Save();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank0_ == true) std::cout << " ...complete!" << endl;
}

void LoMachSolver::updateTimestep() {
  // minimum timestep to not waste comp time
  double dtMin = 1.0e-9;

  double Umax_lcl = 1.0e-12;
  double max_speed = Umax_lcl;
  double Umag;
  int Sdof = sfes->GetNDofs();
  double dtFactor = config.dt_factor;
  auto dataU = flow_->getCurrentVelocity()->HostRead();

  // come in divided by order
  const double *dataD = bufferGridScale->HostRead();

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

double LoMachSolver::computeCFL(double dt) {
  double Umax_lcl = 1.0e-12;
  double max_speed = Umax_lcl;
  double Umag;
  int Sdof = sfes->GetNDofs();

  // come in divided by order
  double *dataD = bufferGridScale->HostReadWrite();
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
  int Sdof = sfes->GetNDofs();

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

void LoMachSolver::SetTimeIntegrationCoefficients(int step) {
  // Maximum BDF order to use at current time step
  // step + 1 <= order <= max_bdf_order
  int bdf_order = std::min(step + 1, max_bdf_order);

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
  double my_rt[6], rt_max[6];

  my_rt[0] = sw_setup.RealTime();
  my_rt[1] = sw_step.RealTime();
  my_rt[2] = sw_extrap.RealTime();
  my_rt[3] = sw_curlcurl.RealTime();
  my_rt[4] = sw_spsolve.RealTime();
  my_rt[5] = sw_hsolve.RealTime();

  MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, pmesh_->GetComm());

  if (pmesh_->GetMyRank() == 0) {
    mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP" << std::setw(10) << "EXTRAP" << std::setw(10)
              << "CURLCURL" << std::setw(10) << "PSOLVE" << std::setw(10) << "HSOLVE"
              << "\n";

    mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0] << std::setw(10) << my_rt[1] << std::setw(10)
              << my_rt[2] << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4] << std::setw(10) << my_rt[5]
              << "\n";

    mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10) << my_rt[1] / my_rt[1] << std::setw(10)
              << my_rt[2] / my_rt[1] << std::setw(10) << my_rt[3] / my_rt[1] << std::setw(10) << my_rt[4] / my_rt[1]
              << std::setw(10) << my_rt[5] / my_rt[1] << "\n";

    mfem::out << std::setprecision(8);
  }
}

// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {
  // if (verbose) grvy_printf(ginfo, "parsing solver options...\n");
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
  tpsP_->getRequiredInput("loMach/flow-solver", loMach_opts_.flow_solver);
  assert(loMach_opts_.flow_solver == "zero-flow" || loMach_opts_.flow_solver == "tomboulides");

  tpsP_->getInput("loMach/order", loMach_opts_.order, 1);
  tpsP_->getInput("loMach/uOrder", loMach_opts_.uOrder, -1);
  tpsP_->getInput("loMach/pOrder", loMach_opts_.pOrder, -1);
  tpsP_->getInput("loMach/nOrder", loMach_opts_.nOrder, -1);
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
}

///* move these to separate support class *//
void LoMachSolver::parseSolverOptions2() {
  // flow/numerics
  parseFlowOptions();

  // time integration controls
  parseTimeIntegrationOptions();

  // I/O settings
  parseIOSettings();

  // initial conditions
  parseICOptions();

  // changed the order of parsing in order to use species list.
  // boundary conditions. The number of boundaries of each supported type if
  // parsed first. Then, a section corresponding to each defined boundary is parsed afterwards
  parseBCInputs();

  // periodicity
  parsePeriodicInputs();
}

void LoMachSolver::parseFlowOptions() {
  std::map<std::string, int> sgsModel;
  sgsModel["none"] = 0;
  sgsModel["smagorinsky"] = 1;
  sgsModel["sigma"] = 2;

  tpsP_->getInput("loMach/order", config.solOrder, 2);
  tpsP_->getInput("loMach/integrationRule", config.integrationRule, 1);
  tpsP_->getInput("loMach/basisType", config.basisType, 1);
  tpsP_->getInput("loMach/maxIters", config.numIters, 10);
  tpsP_->getInput("loMach/outputFreq", config.itersOut, 50);
  tpsP_->getInput("loMach/timingFreq", config.timingFreq, 100);
  tpsP_->getInput("loMach/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP_->getInput("loMach/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  tpsP_->getInput("flow/axisymmetric", config.axisymmetric_, false);
  // axisymmetric not supported yet, so make sure we aren't trying to run it
  assert(!config.axisymmetric_);
  tpsP_->getInput("loMach/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("loMach/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("loMach/SutherlandPr", config.sutherland_.Pr, 0.71);

  tpsP_->getInput("loMach/constantViscosity", config.const_visc, -1.0);
  tpsP_->getInput("loMach/constantDensity", config.const_dens, -1.0);
  tpsP_->getInput("loMach/ambientPressure", config.amb_pres, 101325.0);
  tpsP_->getInput("loMach/openSystem", config.isOpen, true);

  tpsP_->getInput("loMach/enablePressureForcing", config.isForcing, false);
  if (config.isForcing) {
    for (int d = 0; d < 3; d++) tpsP_->getRequiredVecElem("loMach/pressureGrad", config.gradPress[d], d);
  }

  tpsP_->getInput("loMach/enableGravity", config.isGravity, false);
  if (config.isGravity) {
    for (int d = 0; d < 3; d++) tpsP_->getRequiredVecElem("loMach/gravity", config.gravity[d], d);
  }

  tpsP_->getInput("loMach/refinement_levels", config.ref_levels, 0);
  tpsP_->getInput("loMach/computeDistance", config.compute_distance, false);

  std::string type;
  tpsP_->getInput("loMach/sgsModel", type, std::string("none"));
  config.sgsModelType = sgsModel[type];
  tpsP_->getInput("loMach/sgsExcludeMean", config.sgsExcludeMean, false);

  double sgs_const = 0.;
  if (config.sgsModelType == 1) {
    sgs_const = 0.12;
  } else if (config.sgsModelType == 2) {
    sgs_const = 0.135;
  }
  tpsP_->getInput("loMach/sgsModelConstant", config.sgs_model_const, sgs_const);

  assert(config.solOrder > 0);
  assert(config.numIters >= 0);
  assert(config.itersOut > 0);
  assert(config.refLength > 0);
}

void LoMachSolver::parseTimeIntegrationOptions() {
  std::map<std::string, int> integrators;
  integrators["curlcurl"] = 1;
  integrators["staggeredTime"] = 2;
  integrators["deltaP"] = 3;

  std::string type;
  tpsP_->getInput("time/cfl", config.cflNum, 0.2);
  tpsP_->getInput("time/integrator", type, std::string("curlcurl"));
  tpsP_->getInput("time/enableConstantTimestep", config.constantTimeStep, false);
  tpsP_->getInput("time/initialTimestep", config.dt_initial, 1.0e-8);
  tpsP_->getInput("time/changeFactor", config.dt_factor, 0.05);
  tpsP_->getInput("time/dt_fixed", config.dt_fixed, -1.);
  tpsP_->getInput("time/maxSolverIteration", config.solver_iter, 100);
  tpsP_->getInput("time/solverRelTolerance", config.solver_tol, 1.0e-8);
  tpsP_->getInput("time/bdfOrder", config.bdfOrder, 2);
  tpsP_->getInput("time/abOrder", config.abOrder, 2);
  if (integrators.count(type) == 1) {
    config.timeIntegratorType = integrators[type];
  } else {
    grvy_printf(GRVY_ERROR, "Unknown time integrator > %s\n", type.c_str());
    exit(ERROR);
  }
  if (config.timeIntegratorType != 1) {
    config.bdfOrder = 1;
  }
  max_bdf_order = config.bdfOrder;
}

void LoMachSolver::parseIOSettings() {
  tpsP_->getInput("io/outdirBase", config.outputFile, std::string("output-default"));
  tpsP_->getInput("io/enableRestart", config.restart, false);
  tpsP_->getInput("io/exitCheckFreq", config.exit_checkFrequency_, 500);

  std::string restartMode;
  tpsP_->getInput("io/restartMode", restartMode, std::string("standard"));
  if (restartMode == "variableP") {
    config.restartFromAux = true;
  } else if (restartMode == "singleFileWrite") {
    config.restart_serial = "write";
  } else if (restartMode == "singleFileRead") {
    config.restart_serial = "read";
  } else if (restartMode == "singleFileReadWrite") {
    config.restart_serial = "readwrite";
  } else if (restartMode != "standard") {
    grvy_printf(GRVY_ERROR, "\nUnknown restart mode -> %s\n", restartMode.c_str());
    exit(ERROR);
  }
}

void LoMachSolver::parseICOptions() {
  tpsP_->getRequiredInput("initialConditions/rho", config.initRhoRhoVp[0]);
  tpsP_->getRequiredInput("initialConditions/rhoU", config.initRhoRhoVp[1]);
  tpsP_->getRequiredInput("initialConditions/rhoV", config.initRhoRhoVp[2]);
  tpsP_->getRequiredInput("initialConditions/rhoW", config.initRhoRhoVp[3]);
  tpsP_->getRequiredInput("initialConditions/temperature", config.initRhoRhoVp[4]);
  tpsP_->getInput("initialConditions/useFunction", config.useICFunction, false);
  tpsP_->getInput("initialConditions/useBoxFunction", config.useICBoxFunction, false);
  tpsP_->getInput("initialConditions/resetTemp", config.resetTemp, false);
}

void LoMachSolver::parseBCInputs() {
  // number of BC regions defined
  // int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // Wall.....................................
  std::map<std::string, WallType> wallMapping;
  wallMapping["inviscid"] = INV;
  wallMapping["slip"] = SLIP;
  wallMapping["viscous_adiabatic"] = VISC_ADIAB;
  wallMapping["viscous_isothermal"] = VISC_ISOTH;
  wallMapping["viscous_general"] = VISC_GNRL;

  config.wallBC.resize(numWalls);
  for (int i = 1; i <= numWalls; i++) {
    int patch;
    double temperature;
    std::string type;
    std::string basepath("boundaryConditions/wall" + std::to_string(i));
    tpsP_->getInput((basepath + "/useFunction").c_str(), config.useWallFunction, false);
    tpsP_->getInput((basepath + "/useBox").c_str(), config.useWallBox, false);
    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
    if (type == "viscous_isothermal") {
      // std::cout << "*Caught a wall type! " <<  i << " of " << numWalls << endl;
      tpsP_->getRequiredInput((basepath + "/temperature").c_str(), temperature);
      config.wallBC[i - 1].hvyThermalCond = ISOTH;
      config.wallBC[i - 1].elecThermalCond = ISOTH;
      config.wallBC[i - 1].Th = temperature;
      config.wallBC[i - 1].Te = temperature;
    } else if (type == "viscous_adiabatic") {
      config.wallBC[i - 1].hvyThermalCond = ADIAB;
      config.wallBC[i - 1].elecThermalCond = ADIAB;
      config.wallBC[i - 1].Th = -1.0;
      config.wallBC[i - 1].Te = -1.0;
    } else if (type == "viscous_general") {
      std::map<std::string, ThermalCondition> thmCondMap;
      thmCondMap["adiabatic"] = ADIAB;
      thmCondMap["isothermal"] = ISOTH;
      thmCondMap["sheath"] = SHTH;
      thmCondMap["none"] = NONE_THMCND;

      std::string hvyType, elecType;
      tpsP_->getRequiredInput((basepath + "/heavy_thermal_condition").c_str(), hvyType);
      config.wallBC[i - 1].hvyThermalCond = thmCondMap[hvyType];

      config.wallBC[i - 1].Th = -1.0;
      switch (config.wallBC[i - 1].hvyThermalCond) {
        case ISOTH: {
          double Th;
          tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Th);
          config.wallBC[i - 1].Th = Th;
        } break;
        case SHTH: {
          grvy_printf(GRVY_ERROR, "Wall%d: sheath condition is only supported for electron!\n", i);
          exit(-1);
        } break;
        case NONE_THMCND: {
          grvy_printf(GRVY_ERROR, "Wall%d: must specify heavies' thermal condition!\n", i);
          exit(-1);
        }
        default:
          break;
      }

      // NOTE(kevin): sheath can exist even for single temperature.
      if (config.twoTemperature) {
        tpsP_->getRequiredInput((basepath + "/electron_thermal_condition").c_str(), elecType);
        if (thmCondMap[elecType] == NONE_THMCND) {
          grvy_printf(GRVY_ERROR, "Wall%d: electron thermal condition must be specified for two-temperature plasma!\n",
                      i);
          exit(-1);
        }
        config.wallBC[i - 1].elecThermalCond = thmCondMap[elecType];
      } else {
        tpsP_->getInput((basepath + "/electron_thermal_condition").c_str(), elecType, std::string("none"));
        config.wallBC[i - 1].elecThermalCond = (thmCondMap[elecType] == SHTH) ? SHTH : NONE_THMCND;
      }
      config.wallBC[i - 1].Te = -1.0;
      switch (config.wallBC[i - 1].elecThermalCond) {
        case ISOTH: {
          double Te;
          tpsP_->getInput((basepath + "/electron_temperature").c_str(), Te, -1.0);
          if (Te < 0.0) {
            grvy_printf(GRVY_INFO, "Wall%d: no input for electron_temperature. Using temperature instead.\n", i);
            tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Te);
          }
          if (Te < 0.0) {
            grvy_printf(GRVY_ERROR, "Wall%d: invalid electron temperature: %.8E!\n", i, Te);
            exit(-1);
          }
          config.wallBC[i - 1].Te = Te;
        } break;
        case SHTH: {
          if (!config.ambipolar) {
            grvy_printf(GRVY_ERROR, "Wall%d: plasma must be ambipolar for sheath condition!\n", i);
            exit(-1);
          }
        }
        default:
          break;
      }
    }

    // NOTE(kevin): maybe redundant at this point. Kept this just in case.
    std::pair<int, WallType> patchType;
    patchType.first = patch;
    patchType.second = wallMapping[type];
    config.wallPatchType.push_back(patchType);
  }

  // Inlet......................................
  std::map<std::string, InletType> inletMapping;
  inletMapping["uniform"] = UNI_DENS_VEL;
  inletMapping["interpolate"] = INTERPOLATE;
  inletMapping["subsonic"] = SUB_DENS_VEL;  // remove the remaining for loMach
  inletMapping["nonreflecting"] = SUB_DENS_VEL_NR;
  inletMapping["nonreflectingConstEntropy"] = SUB_VEL_CONST_ENT;

  for (int i = 1; i <= numInlets; i++) {
    int patch;
    double density;
    // double rampTime;
    int rampSteps;
    std::string type;
    std::string basepath("boundaryConditions/inlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
    // all inlet BCs require 4 inputs (density + vel(3))
    {
      Array<double> uvw;
      tpsP_->getRequiredInput((basepath + "/density").c_str(), density);
      tpsP_->getRequiredVec((basepath + "/uvw").c_str(), uvw, 3);
      config.inletBC.Append(density);
      config.inletBC.Append(uvw, 3);
      config.velInlet[0] = uvw[0];
      config.velInlet[1] = uvw[1];
      config.velInlet[2] = uvw[2];
      config.densInlet = density;
      if (type == "interpolate" || type == "uniform") {
        tpsP_->getRequiredInput((basepath + "/rampSteps").c_str(), rampSteps);
        config.rampStepsInlet = rampSteps;
      }
    }
    // For multi-component gas, require (numActiveSpecies)-more inputs.
    if ((config.workFluid != DRY_AIR) && (config.numSpecies > 1)) {
      grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
      if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

      for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index
        double Ysp;
        // read mass fraction of species as listed in the input file.
        int inputSp = config.mixtureToInputMap[sp];
        std::string speciesBasePath(basepath + "/mass_fraction/species" + std::to_string(inputSp + 1));
        tpsP_->getRequiredInput(speciesBasePath.c_str(), Ysp);
        config.inletBC.Append(Ysp);
      }
    }
    std::pair<int, InletType> patchType;
    patchType.first = patch;
    patchType.second = inletMapping[type];
    config.inletPatchType.push_back(patchType);
  }

  // Outlet.......................................
  std::map<std::string, OutletType> outletMapping;
  outletMapping["subsonicPressure"] = SUB_P;
  outletMapping["resistInflow"] = RESIST_IN;
  outletMapping["nonReflectingPressure"] = SUB_P_NR;
  outletMapping["nonReflectingMassFlow"] = SUB_MF_NR;
  outletMapping["nonReflectingPointBasedMassFlow"] = SUB_MF_NR_PW;

  for (int i = 1; i <= numOutlets; i++) {
    int patch;
    double pressure, massFlow;
    std::string type;
    std::string basepath("boundaryConditions/outlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    if ((type == "subsonicPressure") || (type == "nonReflectingPressure")) {
      tpsP_->getRequiredInput((basepath + "/pressure").c_str(), pressure);
      config.outletBC.Append(pressure);
      config.outletPressure = pressure;
    } else if ((type == "nonReflectingMassFlow") || (type == "nonReflectingPointBasedMassFlow")) {
      tpsP_->getRequiredInput((basepath + "/massFlow").c_str(), massFlow);
      config.outletBC.Append(massFlow);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown outlet BC supplied at runtime -> %s", type.c_str());
      exit(ERROR);
    }

    std::pair<int, OutletType> patchType;
    patchType.first = patch;
    patchType.second = outletMapping[type];
    config.outletPatchType.push_back(patchType);
  }
}

void LoMachSolver::parsePeriodicInputs() {
  tpsP_->getInput("periodicity/enablePeriodic", config.periodic, false);
  tpsP_->getInput("periodicity/xTrans", config.xTrans, 1.0e12);
  tpsP_->getInput("periodicity/yTrans", config.yTrans, 1.0e12);
  tpsP_->getInput("periodicity/zTrans", config.zTrans, 1.0e12);
}
