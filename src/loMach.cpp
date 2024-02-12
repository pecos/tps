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

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

// temporary
// double mesh_stretching_func(const double y);
void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst);

LoMachSolver::LoMachSolver(LoMachOptions loMach_opts, TPS::Tps *tps)
    : groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      rank0_(groupsMPI->isWorldRoot()),
      tpsP_(tps),
      loMach_opts_(loMach_opts) {
  pmesh = NULL;

  // is this needed?
  groupsMPI->init();

  // ini file options
  parseSolverOptions();
  parseSolverOptions2();
  loadFromAuxSol = config.RestartFromAux();

  // incomp version
  /*
  if (config.const_visc > 0.0) { constantViscosity = true; }
  if (config.const_dens > 0.0) { constantDensity = true; }
  if ( constantViscosity == true && constantDensity == true ) {
    incompressibleSolve = true;
  }
  */

  // set default solver state
  exit_status_ = NORMAL;

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }

  // verify running on cpu
  /*
  if (tpsP_->getDeviceConfig() != "cpu") {
    if (mpi.Root()) {
      grvy_printf(GRVY_ERROR, "[ERROR] Low Mach simulation currently only supported on cpu.\n");
    }
    exit(1);
  }
  */

  // grvy_printf(GRVY_INFO, "Basic LoMachSolver empty \n");
  // plasma_conductivity_ = NULL;
  // plasma_conductivity_coef_ = NULL;
  // joule_heating_ = NULL;

  // instantiate phyics models
  /*
  turbClass = new TurbModel(pmesh,&config,&loMach_opts);
  tcClass = new ThermoChem(pmesh,&config,&loMach_opts);
  flowClass = new Flow(pmesh,&config,&loMach_opts);
  */
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
  Mesh *mesh_ptr = &mesh;
  Mesh *serial_mesh = &mesh;
  if (verbose) grvy_printf(ginfo, "Mesh read...\n");

  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  Mesh periodic_mesh = Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations));
  if (verbose) grvy_printf(ginfo, "Mesh made periodic (if applicable)...\n");

  // underscore stuff is terrible
  dim_ = mesh_ptr->Dimension();
  // int dim = pmesh->Dimension();
  dim = dim_;
  // if (verbose) grvy_printf(ginfo, "Got mesh dim...\n");

  // HACK HACK HACK HARD CODE
  nvel = dim;

  // 1b) Refine the serial mesh, if requested
  /*
  if (verbose && (loMach_opts_.ref_levels > 0)) {
    grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", loMach_opts_.ref_levels);
  }
  for (int l = 0; l < loMach_opts_.ref_levels; l++) {
    mesh_ptr->UniformRefinement();
  }
  */

  /*
  if (Mpi::Root())
  {
     printf("NL=%i NX=%i NY=%i NZ=%i dx+=%f\n", NL, NX, NY, NZ, LC * Re_tau);
     std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
  }
  */

  eqSystem = config.GetEquationSystem();
  // mixture = NULL;

  if (rank0_) {
    if (config.sgsModelType == 1) {
      std::cout << "Smagorinsky subgrid model active" << endl;
    } else if (config.sgsModelType == 2) {
      std::cout << "Sigma subgrid model active" << endl;
    }
    //} else if (config.sgsModelType == 3) {
    //  std::cout << "Dynamic Smagorinsky subgrid model active" << endl;
    //}
  }

  MaxIters = config.GetNumIters();
  max_speed = 0.;
  num_equation = 5;  // HARD CODE
  /*
  numSpecies = mixture->GetNumSpecies();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
  */
  // std::cout << " okay 2..." << endl;

  // check if a simulation is being restarted
  // std::cout << "RestartCycle: " << config.GetRestartCycle() << std::endl;
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
      /*
      if (config.isRestartSerialized("read")) {
        assert(serial_mesh->Conforming());
        //assert(mesh->Conforming());
        partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
        //partitioning_ = Array<int>(mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
        partitioning_file_hdf5("write");
      } else {
      */
      partitioning_file_hdf5("read", config, groupsMPI, nelemGlobal_, partitioning_);

      //}
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
    time = 0.;
    iter = 0;
  }

  // std::cout << " okay 3..." << endl;

  // If requested, evaluate the distance function (i.e., the distance to the nearest no-slip wall)
  /*
  distance_ = NULL;
  GridFunction *serial_distance = NULL;
  //if (config.compute_distance) {
  {
    //order = config.GetSolutionOrder();
    //dim = serial_mesh->Dimension();
    //basisType = config.GetBasisType();
    //DG_FECollection *tmp_fec = NULL;
    //if (basisType == 0) {
    //  tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLegendre);
    //} else if (basisType == 1) {
    //  tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    //}w

    order = loMach_opts_.uOrder;
    dim = serial_mesh->Dimension();
    H1_FECollection *tmp_fec = NULL;
    tmp_fec = new H1_FECollection(order, dim);

    // Serial FE space for scalar variable
    FiniteElementSpace *serial_fes = new FiniteElementSpace(serial_mesh, tmp_fec);

    // Serial grid function for scalar variable
    serial_distance = new GridFunction(serial_fes);

    // Build a list of wall patches
    Array<int> wall_patch_list;
    for (int i = 0; i < config.wallPatchType.size(); i++) {
      if (config.wallPatchType[i].second != WallType::INV) {
        wall_patch_list.Append(config.wallPatchType[i].first);
      }
    }

    if (serial_mesh->GetNodes() == NULL) {
      serial_mesh->SetCurvature(1);
    }

    //FiniteElementSpace *tmp_dfes = new FiniteElementSpace(serial_mesh, tmp_fec, dim, Ordering::byNODES);
    FiniteElementSpace *tmp_dfes = new FiniteElementSpace(serial_mesh, tmp_fec, dim);
    GridFunction coordinates(tmp_dfes);
    serial_mesh->GetNodes(coordinates);

    // Evaluate the distance function
    evaluateDistanceSerial(*serial_mesh, wall_patch_list, coordinates, *serial_distance);
  }
  */

  // 1c) Partition the mesh (see partitioning_ in M2ulPhyS)
  // pmesh_ = new ParMesh(MPI_COMM_WORLD, *mesh);
  pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
  // auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
  // delete mesh_ptr;
  // delete mesh;  // no longer need the serial mesh
  //  pmesh_->ReorientTetMesh();
  if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");

  // instantiate phyics models
  turbClass = new TurbModel(pmesh, &config, &loMach_opts_);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank0_) {std::cout << "turbClass instantiated" << endl;}
  tcClass = new ThermoChem(pmesh, &config, &loMach_opts_);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank0_) {std::cout << "tcClass instantiated" << endl;}
  flowClass = new Flow(pmesh, &config, &loMach_opts_);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank0_) {std::cout << "flowClass instantiated" << endl;}

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // velocity (DG_FECollection?)
  vfec = new H1_FECollection(order, dim);
  vfes = new ParFiniteElementSpace(pmesh, vfec, dim);

  // scalar
  sfec = new H1_FECollection(order);
  sfes = new ParFiniteElementSpace(pmesh, sfec);

  // pressure
  pfec = new H1_FECollection(porder);
  // pfec = new L2_FECollection(porder,dim);
  pfes = new ParFiniteElementSpace(pmesh, pfec);

  // dealias nonlinear term
  nfec = new H1_FECollection(norder, dim);
  nfes = new ParFiniteElementSpace(pmesh, nfec, dim);
  nfecR0 = new H1_FECollection(norder);
  nfesR0 = new ParFiniteElementSpace(pmesh, nfecR0);

  // full vector for compatability
  fvfes = new ParFiniteElementSpace(pmesh, vfec, num_equation);  //, Ordering::byNODES);
  fvfes2 = new ParFiniteElementSpace(pmesh, sfec, num_equation);

  // testing...
  // HCurlFESpace = new ND_ParFESpace(pmesh,order,pmesh->Dimension());
  if (verbose) grvy_printf(ginfo, "Spaces constructed...\n");

  int vfes_truevsize = vfes->GetTrueVSize();
  int pfes_truevsize = pfes->GetTrueVSize();
  int sfes_truevsize = sfes->GetTrueVSize();
  int nfes_truevsize = nfes->GetTrueVSize();
  int nfesR0_truevsize = nfesR0->GetTrueVSize();
  if (verbose) grvy_printf(ginfo, "Got sizes...\n");

  cur_step = 0;

  gridScaleSml.SetSize(sfes_truevsize);
  gridScaleSml = 0.0;
  gridScaleXSml.SetSize(sfes_truevsize);
  gridScaleXSml = 0.0;
  gridScaleYSml.SetSize(sfes_truevsize);
  gridScaleYSml = 0.0;
  gridScaleZSml.SetSize(sfes_truevsize);
  gridScaleZSml = 0.0;

  // for paraview
  // Qt_gf.SetSpace(sfes);
  // Qt_gf = 0.0;
  // viscTotal_gf.SetSpace(sfes);
  // viscTotal_gf = 0.0;
  // alphaTotal_gf.SetSpace(sfes);
  // alphaTotal_gf = 0.0;
  resolution_gf.SetSpace(sfes);
  resolution_gf = 0.0;

  if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");

  // PrintInfo();

  initSolutionAndVisualizationVectors();
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose) grvy_printf(ginfo, "init Sol and Vis okay...\n");

  // initialize sub modues
  turbClass->initialize();
  if (verbose) grvy_printf(ginfo, "init turbClass okay...\n");
  tcClass->initialize();
  if (verbose) grvy_printf(ginfo, "init tcClass okay...\n");
  flowClass->initialize();
  if (verbose) grvy_printf(ginfo, "init flowClass okay...\n");

  // setup pointers to external data
  turbClass->initializeExternal(flowClass->GetCurrentVelGradU(), flowClass->GetCurrentVelGradV(),
                                flowClass->GetCurrentVelGradW(), &resolution_gf);
  if (verbose) grvy_printf(ginfo, "init ext turbClass okay...\n");
  tcClass->initializeExternal(flowClass->GetCurrentVelocity(), turbClass->GetCurrentEddyViscosity(), numWalls,
                              numOutlets, numInlets);
  if (verbose) grvy_printf(ginfo, "init ext tcClass okay...\n");
  flowClass->initializeExternal(tcClass->GetCurrentTotalViscosity(), tcClass->GetCurrentDensity(),
                                tcClass->GetCurrentThermalDiv(), numWalls, numOutlets, numInlets);
  if (verbose) grvy_printf(ginfo, "init ext flowClass okay...\n");

  // Averaging handled through loMach
  average =
      new Averaging(Up, pmesh, sfec, sfes, vfes, fvfes, eqSystem, d_mixture, num_equation, dim, config, groupsMPI);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (verbose) grvy_printf(ginfo, "New Averaging okay...\n");

  // deprecated? average->read_meanANDrms_restart_files();
  // MPI_Barrier(MPI_COMM_WORLD);
  // if (verbose) grvy_printf(ginfo, "average initialized...\n");

  // register rms and mean sol into ioData
  if (average->ComputeMean()) {
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "setting up mean stuff...\n");

    // meanUp
    ioData.registerIOFamily("Time-averaged primitive vars", "/meanSolution", average->GetMeanUp(), false,
                            config.GetRestartMean());
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "..IOFamily okay\n");

    ioData.registerIOVar("/meanSolution", "meanDens", 0);
    ioData.registerIOVar("/meanSolution", "mean-u", 1);
    ioData.registerIOVar("/meanSolution", "mean-v", 2);
    if (nvel == 3) {
      ioData.registerIOVar("/meanSolution", "mean-w", 3);
      ioData.registerIOVar("/meanSolution", "mean-E", 4);
    } else {
      ioData.registerIOVar("/meanSolution", "mean-p", dim + 1);
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "..IOVar okay\n");

    /*
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      // Only for NS_PASSIVE.
      if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;
      // int inputSpeciesIndex = mixture->getInputIndexOf(sp);
      std::string speciesName = config.speciesNames[sp];
      ioData.registerIOVar("/meanSolution", "mean-Y" + speciesName, sp + nvel + 2);
    }
    */

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "On to rms...\n");

    // rms
    ioData.registerIOFamily("RMS velocity fluctuation", "/rmsData", average->GetRMS(), false, config.GetRestartMean());
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "..IOFamily (rms) okay\n");

    ioData.registerIOVar("/rmsData", "uu", 0);
    ioData.registerIOVar("/rmsData", "vv", 1);
    ioData.registerIOVar("/rmsData", "ww", 2);
    ioData.registerIOVar("/rmsData", "uv", 3);
    ioData.registerIOVar("/rmsData", "uw", 4);
    ioData.registerIOVar("/rmsData", "vw", 5);

    // MPI_Barrier(MPI_COMM_WORLD);
    // if (verbose) grvy_printf(ginfo, "..IOVar (rms) okay\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose) grvy_printf(ginfo, "mean setup good...\n");
  /**/

  ioData.initializeSerial(rank0_, (config.RestartSerial() != "no"), serial_mesh, locToGlobElem, &partitioning_);
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose) grvy_printf(ginfo, "ioData.init thingy...\n");

  /*
  projectInitialSolution();
  if (verbose) grvy_printf(ginfo, "initial sol projected...\n");
  */

  CFL = config.GetCFLNumber();
  if (verbose) grvy_printf(ginfo, "got CFL...\n");

  // Determine the minimum element size.
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < pmesh->GetNE(); i++) {
      local_hmin = min(pmesh->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
  }
  if (verbose) grvy_printf(ginfo, "Min element size found...\n");

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < pmesh->GetNE(); i++) {
      local_hmax = max(pmesh->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
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
    int vdim = sfes->GetVDim();
    int nSize = bufferGridScale->Size();
    Array<int> zones_per_vdof;
    zones_per_vdof.SetSize(sfes->GetVSize());
    zones_per_vdof = 0;
    Array<int> zones_per_vdofALL;
    zones_per_vdofALL.SetSize(sfes->GetVSize());
    zones_per_vdofALL = 0;

    double *data = bufferGridScale->HostReadWrite();
    double *dataTMP = bufferGridScaleX->HostReadWrite();
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
      double delta, d1;

      // element dof
      for (int dof = 0; dof < elndofs; ++dof) {
        const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
        tr->SetIntPoint(&ip);
        delta = pmesh->GetElementSize(tr->ElementNo, 1);
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
    // MPI_Allreduce(&zones_per_vdof, &zones_per_vdofALL, sfes->GetVSize(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Accumulate for all vdofs.
    gcomm.Reduce<double>(bufferGridScale->GetData(), GroupCommunicator::Sum);
    gcomm.Bcast<double>(bufferGridScale->GetData());
    // for (int i = 0; i < nSize; i++) { dataTMP[i] = data[i]; }
    // MPI_Allreduce(&bufferGridScaleX, &bufferGridScale, sfes->GetVSize(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
  pmesh->GetVertices(coordsVert);
  int nVert = coordsVert.Size() / dim;
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
    MPI_Allreduce(&local_xmin, &xmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
    MPI_Allreduce(&local_ymin, &ymin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
    MPI_Allreduce(&local_zmin, &zmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
    MPI_Allreduce(&local_xmax, &xmax, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
    MPI_Allreduce(&local_ymax, &ymax, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
    MPI_Allreduce(&local_zmax, &zmax, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  }

  // estimate initial dt => how the hell does this work?
  // Up->ExchangeFaceNbrData();

  //
  // if (config.GetRestartCycle() == 0) initialTimeStep();
  if (rank0_) cout << "Maximum element size: " << hmax << "m" << endl;
  if (rank0_) cout << "Minimum element size: " << hmin << "m" << endl;
  // if (rank0_) cout << "Initial time-step: " << dt << "s" << endl;

  // possible missing bit for restart stats?
  /*
  if (rank0_) {
    ios_base::openmode mode = std::fstream::trunc;
    if (config.GetRestartCycle() == 1) mode = std::fstream::app;

    histFile.open("history.hist", mode);
    if (!histFile.is_open()) {
      std::cout << "Could not open history file!" << std::endl;
    } else {
      if (histFile.tellp() == 0) {
        histFile << "time,iter,drdt,drudt,drvdt,drwdt,dredt";
        if (average->ComputeMean()) histFile << ",avrgSamples,mean_rho,mean_u,mean_v,mean_w,mean_p,uu,vv,ww,uv,uw,vw";
        histFile << std::endl;
      }
    }
  }
  */
}

void LoMachSolver::Setup(double dt) {
  sw_setup.Start();

  // setup sub models first
  turbClass->Setup(dt);
  tcClass->Setup(dt);
  flowClass->Setup(dt);

  // Set initial time step in the history array
  dthist[0] = dt;

  // if (config.sgsExcludeMean == true) { bufferMeanUp = new ParGridFunction(fvfes); }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank0_) std::cout << "Attempting to setup bufferMeanUp..." << endl;
  bufferMeanUp = new ParGridFunction(fvfes2);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank0_) std::cout << "...complete" << endl;

  sw_setup.Stop();
  // std::cout << "Check 29..." << std::endl;
}

void LoMachSolver::UpdateTimestepHistory(double dt) {
  // Rotate values in time step history
  dthist[2] = dthist[1];
  dthist[1] = dthist[0];
  dthist[0] = dt;
}

void LoMachSolver::projectInitialSolution() {
  // Initialize the state.

  // particular case: Euler vortex
  //   {
  //     void (*initialConditionFunction)(const Vector&, Vector&);
  //     t_final = 5.*  2./17.46;
  //     initialConditionFunction = &(this->InitialConditionEulerVortex);
  // initialConditionFunction = &(this->testInitialCondition);

  //     VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
  //     U->ProjectCoefficient(u0);
  //   }
  // if (rank0_) std::cout << "restart: " << config.GetRestartCycle() << std::endl;

  /*
#ifdef HAVE_MASA
  if (config.use_mms_) {
    initMasaHandler();
  }
#endif

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    if (config.use_mms_) {
#ifdef HAVE_MASA
      projectExactSolution(0.0, U);
      if (config.mmsSaveDetails_) projectExactSolution(0.0, masaU_);
#else
      mfem_error("Require MASA support to use MMS.");
#endif
    } else {
      uniformInitialConditions();
    }
  } else {
#ifdef HAVE_MASA
    if (config.use_mms_ && config.mmsSaveDetails_) projectExactSolution(0.0, masaU_);
#endif

    restart_files_hdf5("read");

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->UseRestartMode(true);
  }
  */

  /*
  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    // why would this be called if restart==0?  restart_files_hdf5("read");
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->UseRestartMode(true);
  }
  */
  // if (verbose) grvy_printf(ginfo, " PIS 1...\n");

  // initGradUp();

  // updatePrimitives();

  /*
  // update pressure grid function
  mixture->UpdatePressureGridFunction(press, Up);
  //if (verbose) grvy_printf(ginfo, " PIS 2...\n");

  // update plasma electrical conductivity
  if (tpsP_->isFlowEMCoupled()) {
    mixture->SetConstantPlasmaConductivity(plasma_conductivity_, Up);
  }
  //if (verbose) grvy_printf(ginfo, " PIS 3...\n");
  */

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    // Only save IC from fresh start.  On restart, will save viz at
    // next requested iter.  This avoids possibility of trying to
    // overwrite existing paraview data for the current iteration.

    // if (!(tpsP_->isVisualizationMode())) paraviewColl->Save();
  }
  // if (verbose) grvy_printf(ginfo, " PIS 4...\n");
}

void LoMachSolver::initialTimeStep() {
  auto dataU = U->HostReadWrite();
  int dof = vfes->GetNDofs();

  for (int n = 0; n < dof; n++) {
    Vector state(num_equation);
    for (int eq = 0; eq < num_equation; eq++) state[eq] = dataU[n + eq * dof];
  }

  double partition_C = max_speed;
  MPI_Allreduce(&partition_C, &max_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  dt = CFL * hmin / max_speed / static_cast<double>(dim);

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }
}

void LoMachSolver::solve() {
  // just restart here
  if (config.restart) {
    restart_files_hdf5("read");
    copyU();
  }

  // std::cout << "Check 6..." << std::endl;
  setTimestep();
  if (iter == 0) {
    dt = std::min(config.dt_initial, dt);
  }

  double Umax_lcl = 1.0e-12;
  max_speed = Umax_lcl;
  double Umag;
  int dof = vfes->GetNDofs();

  // temporary hardcodes
  // double t = 0.0;
  double CFL_actual;
  double t_final = 1.0;
  double dtFactor = config.dt_factor;
  // CFL = config.cflNum;
  CFL = config.GetCFLNumber();

  int SdofInt = sfes->GetTrueVSize();
  int Sdof = sfes->GetNDofs();

  // dt_fixed is initialized to -1, so if it is positive,
  // then the user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }

  Setup(dt);
  if (rank0_) std::cout << "Setup complete" << endl;

  updateU();
  // copyU(); // testing
  if (rank0_) std::cout << "Initial updateU complete" << endl;

  // for initial plot
  flowClass->updateGradientsOP(0.0);
  tcClass->updateGradientsOP(0.0);
  tcClass->updateDensity(0.0);
  tcClass->updateDiffusivity();

  // better ways to do this? just for plotting
  /*
  {
    double *visc = tcClass->bufferVisc->HostReadWrite();
    if (rank0_) std::cout << "okay 1" << endl;
    double *data = tcClass->viscTotal_gf.HostReadWrite();
    if (rank0_) std::cout << "okay 2" << endl;
    for (int i = 0; i < Sdof; i++) {
      data[i] = visc[i];
    }
  }
  //tcClass->GetCurrentTotalViscosity();
  if (rank0_) std::cout << "paraview visc pointer set" << endl;
  {
    double *alpha = tcClass->bufferAlpha->HostReadWrite();
    double *data = tcClass->alphaTotal_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = alpha[i];
    }
  }
  if (rank0_) std::cout << "paraview alpha pointer set" << endl;
  */
  {
    double *res = bufferGridScale->HostReadWrite();
    double *data = resolution_gf.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = res[i];
    }
  }

  ParGridFunction *r_gf = tcClass->GetCurrentDensity();
  ParGridFunction *t_gf = tcClass->GetCurrentTemperature();
  ParGridFunction *mu_gf = tcClass->GetCurrentTotalViscosity();
  // ParGridFunction *alpha_gf = tcClass->GetCurrentTotalThermalDiffusivity();
  ParGridFunction *u_gf = flowClass->GetCurrentVelocity();
  ParGridFunction *qt_gf = tcClass->GetCurrentThermalDiv();
  ParGridFunction *p_gf = flowClass->GetCurrentPressure();
  ParGridFunction *res_gf = GetCurrentResolution();

  ParaViewDataCollection pvdc("output", pmesh);
  pvdc.SetDataFormat(VTKFormat::BINARY32);
  pvdc.SetHighOrderOutput(true);
  pvdc.SetLevelsOfDetail(order);
  pvdc.SetCycle(iter);
  pvdc.SetTime(time);
  pvdc.RegisterField("resolution", res_gf);
  pvdc.RegisterField("density", r_gf);
  pvdc.RegisterField("temperature", t_gf);
  pvdc.RegisterField("viscosity", mu_gf);
  pvdc.RegisterField("velocity", u_gf);
  pvdc.RegisterField("pressure", p_gf);
  pvdc.RegisterField("Qt", qt_gf);
  // pvdc.RegisterField("alpha", alpha_gf);
  pvdc.Save();
  if (rank0_ == true) std::cout << "Saving first step to paraview: " << iter << endl;

  if (config.isOpen != true) {
    tcClass->computeSystemMass();
  }

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

    // std::cout << " step/Max: " << step << " " << MaxIters << endl;
    if (step > MaxIters) {
      break;
    }

    // if (verbose) grvy_printf(ginfo, "calling step...\n");
    sw_step.Start();
    if (config.timeIntegratorType == 1) {
      SetTimeIntegrationCoefficients(step - iter_start);

      flowClass->extrapolateState(step, abCoef);
      tcClass->extrapolateState(step, abCoef);

      flowClass->updateBC(step);
      tcClass->updateBC(step);

      flowClass->updateGradientsOP(1.0);
      tcClass->updateGradientsOP(1.0);

      tcClass->updateThermoP(dt);
      tcClass->updateDensity(1.0);

      turbClass->turbModelStep(time, dt, step, iter_start, bdfCoef);
      tcClass->updateDiffusivity();

      tcClass->thermoChemStep(time, dt, step, iter_start, bdfCoef);
      tcClass->updateDiffusivity();

      flowClass->flowStep(time, dt, step, iter_start, bdfCoef);

      // curlcurlStep(time, dt, step, iter_start);

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

    // if (Mpi::Root())
    if ((step < 10) || (step < 100 && step % 10 == 0) || (step % 100 == 0)) {
      double maxT, minT, maxU;
      double maxTall, minTall, maxUall;
      maxT = -1.0e12;
      minT = +1.0e12;
      maxU = 0.0;
      {
        double *d_T = tcClass->Tn.HostReadWrite();
        double *d_u = flowClass->un.HostReadWrite();
        for (int i = 0; i < SdofInt; i++) {
          maxT = max(d_T[i], maxT);
          minT = min(d_T[i], minT);
        }
        for (int i = 0; i < SdofInt; i++) {
          double Umag = d_u[i + 0 * SdofInt] * d_u[i + 0 * SdofInt] + d_u[i + 1 * SdofInt] * d_u[i + 1 * SdofInt] +
                        d_u[i + 2 * SdofInt] * d_u[i + 2 * SdofInt];
          Umag = sqrt(Umag);
          maxU = max(Umag, maxU);
        }
      }
      MPI_Allreduce(&minT, &minTall, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&maxT, &maxTall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&maxU, &maxUall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      if (dt_fixed < 0) {
        if (rank0_ == true) {
          std::cout << " " << step << "    " << time << "   " << dt << "   " << sw_step.RealTime() - time_previous
                    << "   " << minTall << "   " << maxTall << "   " << maxUall << endl;
        }

      } else {
        // CFL_actual = ComputeCFL(un_gf, dt);
        CFL_actual = computeCFL(dt);
        if (rank0_ == true) {
          std::cout << " " << step << "    " << time << "   " << CFL_actual << "   "
                    << sw_step.RealTime() - time_previous << "   " << minTall << "   " << maxTall << "   " << maxUall
                    << endl;
        }
      }
    }
    time_previous = sw_step.RealTime();

    // update dt
    if (dt_fixed < 0.0) {
      updateTimestep();
    }

    // restart files
    if (iter % config.itersOut == 0 && iter != 0) {
      updateU();
      restart_files_hdf5("write");
      average->write_meanANDrms_restart_files(iter, time);
    }

    // if(rank0_) std::cout << " Calling addSampleMean ..." << endl;
    average->addSampleMean(iter);
    // if(rank0_) std::cout << " ...and done" << endl;

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

    // if (!provisional)
    //{
    UpdateTimestepHistory(dt);
    time += dt;
    // if(rank0_) std::cout << "Time in Step after update: " << time << endl;
    // }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // paraview
  pvdc.SetCycle(iter);
  pvdc.SetTime(time);
  if (rank0_ == true) std::cout << " Saving final step to paraview: " << iter << "... " << endl;
  pvdc.Save();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank0_ == true) std::cout << " ...complete!" << endl;
}

void LoMachSolver::updateTimestep() {
  // minimum timestep to not waste comp time
  double dtMin = 1.0e-9;

  double Umax_lcl = 1.0e-12;
  max_speed = Umax_lcl;
  double Umag;
  int dof = vfes->GetNDofs();
  int Sdof = sfes->GetNDofs();
  double dtFactor = config.dt_factor;
  auto dataU = flowClass->un_gf.HostRead();

  // come in divided by order
  double *dataD = bufferGridScale->HostReadWrite();
  double *dataX = bufferGridScaleX->HostReadWrite();
  double *dataY = bufferGridScaleY->HostReadWrite();
  double *dataZ = bufferGridScaleZ->HostReadWrite();

  for (int n = 0; n < Sdof; n++) {
    Umag = 0.0;
    // Vector delta({dataX[n], dataY[n], dataZ[n]});
    Vector delta({dataD[n], dataD[n], dataD[n]});  // use smallest delta for all
    for (int eq = 0; eq < dim; eq++) {
      Umag += (dataU[n + eq * Sdof] / delta[eq]) * (dataU[n + eq * Sdof] / delta[eq]);
    }
    Umag = std::sqrt(Umag);
    Umax_lcl = std::max(Umag, Umax_lcl);
  }

  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  double dtInst_conv = 0.5 * CFL / max_speed;

  double *dGradU = flowClass->gradU.HostReadWrite();
  double *dGradV = flowClass->gradV.HostReadWrite();
  double *dGradW = flowClass->gradW.HostReadWrite();
  double *dVisc = tcClass->viscSml.HostReadWrite();
  double *rho = tcClass->rn.HostReadWrite();
  Umax_lcl = 1.0e-12;
  for (int n = 0; n < SdofInt; n++) {
    DenseMatrix gU;
    gU.SetSize(nvel, dim);
    for (int dir = 0; dir < dim; dir++) {
      gU(0, dir) = dGradU[n + dir * SdofInt];
    }
    for (int dir = 0; dir < dim; dir++) {
      gU(1, dir) = dGradV[n + dir * SdofInt];
    }
    for (int dir = 0; dir < dim; dir++) {
      gU(2, dir) = dGradW[n + dir * SdofInt];
    }
    double duMag = 0.0;
    for (int dir = 0; dir < dim; dir++) {
      for (int eq = 0; eq < nvel; eq++) {
        duMag += gU(eq, dir) * gU(eq, dir);
      }
    }
    duMag = sqrt(duMag);
    double viscVel = sqrt(duMag * dVisc[n] / rho[n]);
    viscVel /= dataD[n];
    Umax_lcl = std::max(viscVel, Umax_lcl);
  }
  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  double dtInst_visc = 0.5 * CFL / max_speed;

  double dtInst = max(dtInst_conv, dtInst_visc);

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
  // minimum timestep to not waste comp time
  double dtMin = 1.0e-9;

  double Umax_lcl = 1.0e-12;
  max_speed = Umax_lcl;
  double Umag;
  int dof = vfes->GetNDofs();
  int Sdof = sfes->GetNDofs();
  double dtFactor = config.dt_factor;

  // come in divided by order
  double *dataD = bufferGridScale->HostReadWrite();
  double *dataX = bufferGridScaleX->HostReadWrite();
  double *dataY = bufferGridScaleY->HostReadWrite();
  double *dataZ = bufferGridScaleZ->HostReadWrite();
  auto dataU = flowClass->un_gf.HostRead();

  for (int n = 0; n < Sdof; n++) {
    Umag = 0.0;
    // Vector delta({dataX[n], dataY[n], dataZ[n]});
    Vector delta({dataD[n], dataD[n], dataD[n]});  // use smallest delta for all
    for (int eq = 0; eq < dim; eq++) {
      Umag += (dataU[n + eq * Sdof] / delta[eq]) * (dataU[n + eq * Sdof] / delta[eq]);
    }
    Umag = std::sqrt(Umag);
    Umax_lcl = std::max(Umag, Umax_lcl);
  }
  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  double CFL_conv = 2.0 * dt * max_speed;

  double *dGradU = flowClass->gradU.HostReadWrite();
  double *dGradV = flowClass->gradV.HostReadWrite();
  double *dGradW = flowClass->gradW.HostReadWrite();
  double *dVisc = tcClass->viscSml.HostReadWrite();
  double *rho = tcClass->rn.HostReadWrite();
  Umax_lcl = 1.0e-12;
  for (int n = 0; n < SdofInt; n++) {
    DenseMatrix gU;
    gU.SetSize(nvel, dim);
    for (int dir = 0; dir < dim; dir++) {
      gU(0, dir) = dGradU[n + dir * SdofInt];
    }
    for (int dir = 0; dir < dim; dir++) {
      gU(1, dir) = dGradV[n + dir * SdofInt];
    }
    for (int dir = 0; dir < dim; dir++) {
      gU(2, dir) = dGradW[n + dir * SdofInt];
    }
    double duMag = 0.0;
    for (int dir = 0; dir < dim; dir++) {
      for (int eq = 0; eq < nvel; eq++) {
        duMag += gU(eq, dir) * gU(eq, dir);
      }
    }
    duMag = sqrt(duMag);
    double viscVel = sqrt(duMag * dVisc[n] / rho[n]);
    viscVel /= dataD[n];
    Umax_lcl = std::max(viscVel, Umax_lcl);
  }
  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  double CFL_visc = 2.0 * dt * max_speed;

  double CFL_here = max(CFL_conv, CFL_visc);

  return CFL_here;
}

void LoMachSolver::setTimestep() {
  double Umax_lcl = 1.0e-12;
  max_speed = Umax_lcl;
  double Umag;
  int dof = vfes->GetNDofs();
  int Sdof = sfes->GetNDofs();
  double dtFactor = config.dt_factor;

  auto dataU = flowClass->un_gf.HostRead();
  for (int n = 0; n < Sdof; n++) {
    Umag = 0.0;
    for (int eq = 0; eq < dim; eq++) {
      Umag += dataU[n + eq * Sdof] * dataU[n + eq * Sdof];
    }
    Umag = std::sqrt(Umag);
    Umax_lcl = std::max(Umag, Umax_lcl);
  }
  MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  double dtInst = CFL * hmin / (max_speed * (double)order);
  dt = dtInst;
  std::cout << "dt from setTimestep: " << dt << " max_speed: " << max_speed << endl;
}

// copies for compatability => is conserved (U) even necessary with this method?
void LoMachSolver::updateU() {
  // primitive state
  {
    double *dataUp = Up->HostReadWrite();

    const auto d_rn_gf = tcClass->rn_gf.Read();
    MFEM_FORALL(i, tcClass->rn_gf.Size(), { dataUp[i] = d_rn_gf[i]; });

    int vstart = sfes->GetNDofs();
    const auto d_un_gf = flowClass->un_gf.Read();
    MFEM_FORALL(i, flowClass->un_gf.Size(), { dataUp[i + vstart] = d_un_gf[i]; });

    int tstart = (1 + nvel) * (sfes->GetNDofs());
    const auto d_tn_gf = tcClass->Tn_gf.Read();
    MFEM_FORALL(i, tcClass->Tn_gf.Size(), { dataUp[i + tstart] = d_tn_gf[i]; });
  }
}

void LoMachSolver::copyU() {
  if (rank0_) std::cout << " Copying state to Up vector..." << endl;
  // primitive state
  {
    double *dataUp = Up->HostReadWrite();

    double *d_rn_gf = tcClass->rn_gf.ReadWrite();
    MFEM_FORALL(i, tcClass->rn_gf.Size(), { d_rn_gf[i] = dataUp[i]; });

    int vstart = sfes->GetNDofs();
    double *d_un_gf = flowClass->un_gf.ReadWrite();
    MFEM_FORALL(i, flowClass->un_gf.Size(), { d_un_gf[i] = dataUp[i + vstart]; });

    int tstart = (1 + nvel) * (sfes->GetNDofs());
    double *d_tn_gf = tcClass->Tn_gf.ReadWrite();
    MFEM_FORALL(i, tcClass->Tn_gf.Size(), { d_tn_gf[i] = dataUp[i + tstart]; });
  }
}

// move to utils!!!
/*
void LoMachSolver::ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
{

   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      //int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         curl.SetSize(3);
         curl(0) = grad(2, 1) - grad(1, 2);
         curl(1) = grad(0, 2) - grad(2, 0);
         curl(2) = grad(1, 0) - grad(0, 1);

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}
*/

// move to utils
/*
void LoMachSolver::vectorGrad3D(ParGridFunction &u, ParGridFunction &gu, ParGridFunction &gv, ParGridFunction &gw)
{

   ParGridFunction uSub;
   uSub.SetSpace(sfes);
   int nSize = uSub.Size();

   {
     double *dataSub = uSub.HostReadWrite();
     double *data = u.HostReadWrite();
     for (int i = 0; i < nSize; i++) {
       dataSub[i] = data[i + 0 * nSize];
     }
   }
   scalarGrad3D(uSub, gu);

   {
     double *dataSub = uSub.HostReadWrite();
     double *data = u.HostReadWrite();
     for (int i = 0; i < nSize; i++) {
       dataSub[i] = data[i + 1 * nSize];
     }
   }
   scalarGrad3D(uSub, gv);

   {
     double *dataSub = uSub.HostReadWrite();
     double *data = u.HostReadWrite();
     for (int i = 0; i < nSize; i++) {
       dataSub[i] = data[i + 2 * nSize];
     }
   }
   scalarGrad3D(uSub, gw);

}
*/

/*
void LoMachSolver::scalarGrad3D(ParGridFunction &u, ParGridFunction &gu)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize( 3 * (fes->GetVSize()));
   zones_per_vdof = 0;

   gu = 0.0;
   int nSize = u.Size();

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals1, vals2, vals3;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;

   // element loop
   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals1.SetSize(vdofs.Size());
      vals2.SetSize(vdofs.Size());
      vals3.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      //int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      // element dof
      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim,dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, 1);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         vals1(dof) = grad(0,0);
         vals2(dof) = grad(0,1);
         vals3(dof) = grad(0,2);

      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        gu(ldof + 0 * nSize) += vals1[j];
      }
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        gu(ldof + 1 * nSize) += vals2[j];
      }
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        gu(ldof + 2 * nSize) += vals3[j];
      }

      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        zones_per_vdof[ldof]++;
      }

   }

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(gu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(gu.GetData());

   // Compute means.
   for (int dir = 0; dir < dim; dir++) {
     for (int i = 0; i < u.Size(); i++) {
       const int nz = zones_per_vdof[i];
       if (nz) { gu(i + dir*nSize) /= nz; }
     }
   }

}
*/

/*
void LoMachSolver::computeLaplace(Vector u, Vector* Lu)
{

     // this is wasteful, add d2shape function routine...
     R1PM0_gf.SetFromTrueDofs(u);
     vectorGrad3D(R1PM0_gf,R1PM0a_gf,R1PM0b_gf,R1PM0c_gf);
     {
       double *data = R1PM0_gf.HostReadWrite();
       double *d1 = R1PM0a_gf.HostReadWrite();
       double *d2 = R1PM0b_gf.HostReadWrite();
       double *d3 = R1PM0c_gf.HostReadWrite();
       for (int i = 0; i < Sdof; i++) {
         data[i + 0*Sdof] = d1[i + 0*Sdof];
       }
       for (int i = 0; i < Sdof; i++) {
         data[i + 1*Sdof] = d2[i + 1*Sdof];
       }
       for (int i = 0; i < Sdof; i++) {
         data[i + 2*Sdof] = d3[i + 2*Sdof];
       }
     }
     vectorGrad3D(R1PM0_gf,R1PM0a_gf,R1PM0b_gf,R1PM0c_gf);
     {
       double *data = R0PM0_gf.HostReadWrite();
       double *d1 = R1PM0a_gf.HostReadWrite();
       double *d2 = R1PM0b_gf.HostReadWrite();
       double *d3 = R1PM0c_gf.HostReadWrite();
       for (int i = 0; i < Sdof; i++) {
         data[i] = 0.0;
       }
       for (int i = 0; i < Sdof; i++) {
         data[i + 0*Sdof] += d1[i + 0*Sdof];
       }
       for (int i = 0; i < Sdof; i++) {
         data[i + 1*Sdof] += d2[i + 1*Sdof];
       }
       for (int i = 0; i < Sdof; i++) {
         data[i + 2*Sdof] += d3[i + 2*Sdof];
       }
     }
     R0PM0_gf.GetTrueDofs(*Lu);
}
*/

/*
void LoMachSolver::scalarGrad3DV(Vector u, Vector* gu)
{
     R0PM0_gf.SetFromTrueDofs(u);
     scalarGrad3D(R0PM0_gf,R1PM0_gf);
     R1PM0_gf.GetTrueDofs(*gu);
}

void LoMachSolver::vectorGrad3DV(Vector u, Vector* gu, Vector* gv, Vector* gw)
{
     R1PM0_gf.SetFromTrueDofs(u);
     vectorGrad3D(R1PM0_gf,R1PM0a_gf,R1PM0b_gf,R1PM0c_gf);
     R1PM0a_gf.GetTrueDofs(*gu);
     R1PM0b_gf.GetTrueDofs(*gv);
     R1PM0c_gf.GetTrueDofs(*gw);
}
*/

/*
void LoMachSolver::ComputeCurl2D(ParGridFunction &u,
                                 ParGridFunction &cu,
                                 bool assume_scalar)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      //int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         if (assume_scalar)
         {
            curl.SetSize(2);
            curl(0) = grad(0, 1);
            curl(1) = -grad(0, 0);
         }
         else
         {
            curl.SetSize(2);
            curl(0) = grad(1, 0) - grad(0, 1);
            curl(1) = 0.0;
         }

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication.

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}
*/

/*
double LoMachSolver::ComputeCFL(ParGridFunction &u, double dt)
{

   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}
*/

/*
void LoMachSolver::EnforceCFL(double maxCFL, ParGridFunction &u, double &dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         // here
         double dtx, dty, dtz;
         //cflx = fabs(dt * ux(i) / hmin);
         //cfly = fabs(dt * uy(i) / hmin);
         dtx = maxCFL * hmin / fabs(ux(i));
         dty = maxCFL * hmin / fabs(uy(i));
         if (vdim == 3)
         {
           //cflz = fabs(dt * uz(i) / hmin);
           dtz = maxCFL * hmin / fabs(uz(i));
         }
         //cflm = cflx + cfly + cflz;
         //cflmax = fmax(cflmax, cflm);
         dt = fmin(dtx,dty);
         dt = fmin(dt,dtz);

      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return;
}
*/

void LoMachSolver::SetTimeIntegrationCoefficients(int step) {
  // Maximum BDF order to use at current time step
  // step + 1 <= order <= max_bdf_order
  int bdf_order = std::min(step + 1, max_bdf_order);

  // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
  double rho1 = 0.0;

  // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
  double rho2 = 0.0;

  rho1 = dthist[0] / dthist[1];

  if (bdf_order == 3) {
    rho2 = dthist[1] / dthist[2];
  }

  if (step == 0 && bdf_order == 1) {
    bd0 = 1.0;
    bd1 = -1.0;
    bd2 = 0.0;
    bd3 = 0.0;
    ab1 = 1.0;
    ab2 = 0.0;
    ab3 = 0.0;
  } else if (step >= 1 && bdf_order == 2) {
    bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
    bd1 = -(1.0 + rho1);
    bd2 = pow(rho1, 2.0) / (1.0 + rho1);
    bd3 = 0.0;
    ab1 = 1.0 + rho1;
    ab2 = -rho1;
    ab3 = 0.0;
  } else if (step >= 2 && bdf_order == 3) {
    bd0 = 1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
    bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
    bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
    bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1)) / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
    ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
    ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
    ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
  }

  abCoef = {ab1, ab2, ab3};
  bdfCoef = {bd0, bd1, bd2, bd3};
}

void LoMachSolver::PrintTimingData() {
  double my_rt[6], rt_max[6];

  my_rt[0] = sw_setup.RealTime();
  my_rt[1] = sw_step.RealTime();
  my_rt[2] = sw_extrap.RealTime();
  my_rt[3] = sw_curlcurl.RealTime();
  my_rt[4] = sw_spsolve.RealTime();
  my_rt[5] = sw_hsolve.RealTime();

  MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

  if (pmesh->GetMyRank() == 0) {
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

void LoMachSolver::PrintInfo() {
  int fes_size0 = vfes->GlobalVSize();
  int fes_size1 = pfes->GlobalVSize();

  if (pmesh->GetMyRank() == 0) {
    //      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
    mfem::out << "MFEM version: " << MFEM_VERSION << std::endl
              << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
              << "Velocity #DOFs: " << fes_size0 << std::endl
              << "Pressure #DOFs: " << fes_size1 << std::endl;
  }
}

/*
LoMachSolver::~LoMachSolver()
{

   delete FText_gfcoeff;
   delete g_bdr_form;
   delete FText_bdr_form;
   delete Text_gfcoeff;
   delete t_bdr_form;
   delete Text_bdr_form;
   delete f_form;

   delete mass_lf;
   delete N;
   delete Mv_form;
   delete Mt_form;
   delete Sp_form;
   delete D_form;
   delete Dt_form;
   delete G_form;
   delete H_form;
   delete Ht_form;

   delete MvInv;
   delete MtInv;
   delete HInv;
   delete HtInv;
   delete HInvPC;
   delete HtInvPC;
   delete MvInvPC;
   delete MtInvPC;

   delete SpInv;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete lor;

   delete vfec;
   delete pfec;
   delete sfec;
   delete sfec;

   delete vfes;
   delete pfes;
   delete sfes;
   delete sfes;
   delete fvfes;

   delete vfec_filter;
   delete vfes_filter;
   delete sfec_filter;
   delete sfes_filter;

}
*/

// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {
  // if (verbose) grvy_printf(ginfo, "parsing solver options...\n");
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
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

  // tpsP_->getInput("loMach/channelTest", loMach_opts_.channelTest, false);

  // tpsP_->getInput("em/atol", em_opts_.atol, 1.0e-10);
  // tpsP_->getInput("em/preconditioner_background_sigma", em_opts_.preconditioner_background_sigma, -1.0);
  // tpsP_->getInput("em/evaluate_magnetic_field", em_opts_.evaluate_magnetic_field, true);
  // tpsP_->getInput("em/nBy", em_opts_.nBy, 0);
  // tpsP_->getInput("em/yinterp_min", em_opts_.yinterp_min, 0.0);
  // tpsP_->getInput("em/yinterp_max", em_opts_.yinterp_max, 1.0);
  // tpsP_->getInput("em/By_file", em_opts_.By_file, std::string("By.h5"));
  // tpsP_->getInput("em/top_only", em_opts_.top_only, false);
  // tpsP_->getInput("em/bot_only", em_opts_.bot_only, false);

  // tpsP_->getInput("em/current_amplitude", em_opts_.current_amplitude, 1.0);
  // tpsP_->getInput("em/current_frequency", em_opts_.current_frequency, 1.0);
  // tpsP_->getInput("em/permeability", em_opts_.mu0, 1.0);

  // Vector default_axis(3);
  // default_axis[0] = 0;
  // default_axis[1] = 1;
  // default_axis[2] = 0;
  // tpsP_->getVec("em/current_axis", em_opts_.current_axis, 3, default_axis);

  // dump options to screen for user inspection
  if (rank0_) {
    loMach_opts_.print(std::cout);
  }
}

///* move these to separate support class *//
void LoMachSolver::parseSolverOptions2() {
  // tpsP->getRequiredInput("flow/mesh", config.meshFile);

  // flow/numerics
  parseFlowOptions();

  // time integration controls
  parseTimeIntegrationOptions();

  // statistics
  parseStatOptions();

  // I/O settings
  parseIOSettings();

  // RMS job management
  // parseRMSJobOptions();

  // heat source
  // parseHeatSrcOptions();

  // viscosity multiplier function
  parseViscosityOptions();

  // MMS (check before IC b/c if MMS, IC not required)
  // parseMMSOptions();

  // initial conditions
  // if (!config.use_mms_) { parseICOptions(); }
  parseICOptions();

  // add passive scalar forcings
  // parsePassiveScalarOptions();

  // fluid presets
  parseFluidPreset();

  // parseSystemType();

  // plasma conditions.
  /*
  config.gasModel = NUM_GASMODEL;
  config.transportModel = NUM_TRANSPORTMODEL;
  config.chemistryModel_ = NUM_CHEMISTRYMODEL;
  if (config.workFluid == USER_DEFINED) {
    parsePlasmaModels();
  } else {
    // parse options for other plasma presets.
  }
  */

  // species list.
  // parseSpeciesInputs();

  // if (config.workFluid == USER_DEFINED) {  // Transport model
  //   parseTransportInputs();
  // }

  // Reaction list
  // parseReactionInputs();

  // changed the order of parsing in order to use species list.
  // boundary conditions. The number of boundaries of each supported type if
  // parsed first. Then, a section corresponding to each defined boundary is parsed afterwards
  parseBCInputs();

  // sponge zone
  // parseSpongeZoneInputs();

  // Pack up input parameters for device objects.
  // packUpGasMixtureInput();

  // Radiation model
  // parseRadiationInputs();

  // periodicity
  parsePeriodicInputs();

  // post-process visualization inputs
  parsePostProcessVisualizationInputs();

  return;
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
  // tpsP_->getInput("flow/useRoe", config.useRoe, false);
  // tpsP_->getInput("flow/refLength", config.refLength, 1.0);
  tpsP_->getInput("loMach/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP_->getInput("loMach/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  // tpsP_->getInput("flow/axisymmetric", config.axisymmetric_, false);
  tpsP_->getInput("loMach/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("loMach/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("loMach/SutherlandPr", config.sutherland_.Pr, 0.71);

  tpsP_->getInput("loMach/constantViscosity", config.const_visc, -1.0);
  tpsP_->getInput("loMach/constantDensity", config.const_dens, -1.0);
  tpsP_->getInput("loMach/ambientPressure", config.amb_pres, 101325.0);
  tpsP_->getInput("loMach/openSystem", config.isOpen, true);
  // tpsP_->getInput("loMach/solveTemp", config.solveTemp, true);

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

  // Sutherland inputs default to dry air values
  /*
  tpsP_->getInput("flow/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("flow/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("flow/SutherlandPr", config.sutherland_.Pr, 0.71);
  */
}

void LoMachSolver::parseTimeIntegrationOptions() {
  // std::map<std::string, int> integrators;
  // integrators["forwardEuler"] = 1;
  // integrators["rk2"] = 2;
  // integrators["rk3"] = 3;
  // integrators["rk4"] = 4;
  // integrators["rk6"] = 6;

  std::map<std::string, int> integrators;
  integrators["curlcurl"] = 1;
  integrators["staggeredTime"] = 2;
  integrators["deltaP"] = 3;

  // std::cout << "getting time options" << endl;
  std::string type;
  tpsP_->getInput("time/cfl", config.cflNum, 0.2);
  // std::cout << "got cflNum" << config.cflNum << endl;
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

void LoMachSolver::parseStatOptions() {
  tpsP_->getInput("averaging/saveMeanHist", config.meanHistEnable, false);
  tpsP_->getInput("averaging/startIter", config.startIter, 0);
  tpsP_->getInput("averaging/sampleFreq", config.sampleInterval, 0);
  tpsP_->getInput("averaging/enableContinuation", config.restartMean, false);
  tpsP_->getInput("averaging/restartRMS", config.restartRMS, false);
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

void LoMachSolver::parseRMSJobOptions() {
  tpsP_->getInput("jobManagement/enableAutoRestart", config.rm_enableMonitor_, false);
  tpsP_->getInput("jobManagement/timeThreshold", config.rm_threshold_, 15 * 60);  // 15 minutes
  tpsP_->getInput("jobManagement/checkFreq", config.rm_checkFrequency_, 500);     // 500 iterations
}

/*
void M2ulPhyS::parseHeatSrcOptions() {
  int numHeatSources;
  tpsP->getInput("heatSource/numHeatSources", numHeatSources, 0);
  config.numHeatSources = numHeatSources;

  if (numHeatSources > 0) {
    config.numHeatSources = numHeatSources;
    config.heatSource = new heatSourceData[numHeatSources];

    for (int s = 0; s < numHeatSources; s++) {
      std::string base("heatSource" + std::to_string(s + 1));

      tpsP->getInput((base + "/isEnabled").c_str(), config.heatSource[s].isEnabled, false);

      if (config.heatSource[s].isEnabled) {
        tpsP->getRequiredInput((base + "/value").c_str(), config.heatSource[s].value);

        std::string type;
        tpsP->getRequiredInput((base + "/distribution").c_str(), type);
        if (type == "cylinder") {
          config.heatSource[s].type = type;
        } else {
          grvy_printf(GRVY_ERROR, "\nUnknown heat source distribution -> %s\n", type.c_str());
          exit(ERROR);
        }

        tpsP->getRequiredInput((base + "/radius").c_str(), config.heatSource[s].radius);
        config.heatSource[s].point1.SetSize(3);
        config.heatSource[s].point2.SetSize(3);

        tpsP->getRequiredVec((base + "/point1").c_str(), config.heatSource[s].point1, 3);
        tpsP->getRequiredVec((base + "/point2").c_str(), config.heatSource[s].point2, 3);
      }
    }
  }
}
*/

void LoMachSolver::parseViscosityOptions() {
  tpsP_->getInput("viscosityMultiplierFunction/isEnabled", config.linViscData.isEnabled, false);

  if (config.linViscData.isEnabled) {
    // config.linViscData.normal.SetSize(dim);
    // config.linViscData.point0.SetSize(dim);
    // config.linViscData.pointA.SetSize(dim);

    auto normal = config.linViscData.normal.HostWrite();
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[0], 0);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[1], 1);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[2], 2);

    auto point0 = config.linViscData.point0.HostWrite();
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/point", point0[0], 0);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/point", point0[1], 1);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/point", point0[2], 2);

    // auto pointInit = config.linViscData.pointInit.HostWrite();
    // tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[0], 0);
    // tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[1], 1);
    // tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[2], 2);

    tpsP_->getRequiredInput("viscosityMultiplierFunction/width", config.linViscData.width);
    tpsP_->getRequiredInput("viscosityMultiplierFunction/viscosityRatio", config.linViscData.viscRatio);

    tpsP_->getInput("viscosityMultiplierFunction/uniformMult", config.linViscData.uniformMult, 1.0);
    tpsP_->getInput("viscosityMultiplierFunction/cylinderX", config.linViscData.cylXradius, -1.0);
    tpsP_->getInput("viscosityMultiplierFunction/cylinderY", config.linViscData.cylYradius, -1.0);
    tpsP_->getInput("viscosityMultiplierFunction/cylinderZ", config.linViscData.cylZradius, -1.0);

    /**/
    auto pointA = config.linViscData.pointA.HostWrite();
    double pointA0, pointA1, pointA2;
    tpsP_->getInput("viscosityMultiplierFunction/pointA0", pointA0, 0.0);
    tpsP_->getInput("viscosityMultiplierFunction/pointA1", pointA1, 0.0);
    tpsP_->getInput("viscosityMultiplierFunction/pointA2", pointA2, 0.0);
    pointA[0] = pointA0;
    pointA[1] = pointA1;
    pointA[2] = pointA2;
    tpsP_->getInput("viscosityMultiplierFunction/viscosityRatioAnnulus", config.linViscData.viscRatioAnnulus, 1.0);
    tpsP_->getInput("viscosityMultiplierFunction/annulusRadius", config.linViscData.annulusRadius, 1.0e8);
    tpsP_->getInput("viscosityMultiplierFunction/annulusThickness", config.linViscData.annulusThickness, 1.0e-8);
    /**/
  }
}

/*
void M2ulPhyS::parseMMSOptions() {
  tpsP->getInput("mms/isEnabled", config.use_mms_, false);

  if (config.use_mms_) {
    tpsP->getRequiredInput("mms/name", config.mms_name_);
    tpsP->getInput("mms/compare_rhs", config.mmsCompareRhs_, false);
    tpsP->getInput("mms/save_details", config.mmsSaveDetails_, false);
  }
}
*/

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

/*
void M2ulPhyS::parsePassiveScalarOptions() {
  int numForcings;
  tpsP->getInput("passiveScalars/numScalars", numForcings, 0);
  for (int i = 1; i <= numForcings; i++) {
    // add new entry in arrayPassiveScalar
    config.arrayPassiveScalar.Append(new passiveScalarData);
    config.arrayPassiveScalar[i - 1]->coords.SetSize(3);

    std::string basepath("passiveScalar" + std::to_string(i));

    Array<double> xyz;
    tpsP->getRequiredVec((basepath + "/xyz").c_str(), xyz, 3);
    for (int d = 0; d < 3; d++) config.arrayPassiveScalar[i - 1]->coords[d] = xyz[d];

    double radius;
    tpsP->getRequiredInput((basepath + "/radius").c_str(), radius);
    config.arrayPassiveScalar[i - 1]->radius = radius;

    double value;
    tpsP->getRequiredInput((basepath + "/value").c_str(), value);
    config.arrayPassiveScalar[i - 1]->value = value;
  }
}
*/

void LoMachSolver::parseFluidPreset() {
  std::string fluidTypeStr;
  tpsP_->getInput("loMach/fluid", fluidTypeStr, std::string("dry_air"));
  if (fluidTypeStr == "dry_air") {
    config.workFluid = DRY_AIR;
    tpsP_->getInput("loMach/specific_heat_ratio", config.dryAirInput.specific_heat_ratio, 1.4);
    tpsP_->getInput("loMach/gas_constant", config.dryAirInput.gas_constant, 287.058);
  } else if (fluidTypeStr == "user_defined") {
    config.workFluid = USER_DEFINED;
  } else if (fluidTypeStr == "lte_table") {
    config.workFluid = LTE_FLUID;
    std::string thermo_file;
    tpsP_->getRequiredInput("loMach/lte/thermo_table", thermo_file);
    config.lteMixtureInput.thermo_file_name = thermo_file;
    std::string trans_file;
    tpsP_->getRequiredInput("loMach/lte/transport_table", trans_file);
    config.lteMixtureInput.trans_file_name = trans_file;
    std::string e_rev_file;
    tpsP_->getRequiredInput("loMach/lte/e_rev_table", e_rev_file);
    config.lteMixtureInput.e_rev_file_name = e_rev_file;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown fluid preset supplied at runtime -> %s", fluidTypeStr.c_str());
    exit(ERROR);
  }

  if ((rank0_ == true) && (config.workFluid != USER_DEFINED))
    cout << "Fluid is set to the preset '" << fluidTypeStr << "'. Input options in [plasma_models] will not be used."
         << endl;
}

/*
void M2ulPhyS::parseSystemType() {
  std::string systemType;
  tpsP->getInput("flow/equation_system", systemType, std::string("navier-stokes"));
  if (systemType == "euler") {
    config.eqSystem = EULER;
  } else if (systemType == "navier-stokes") {
    config.eqSystem = NS;
  } else if (systemType == "navier-stokes-passive") {
    config.eqSystem = NS_PASSIVE;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown equation_system -> %s", systemType.c_str());
    exit(ERROR);
  }
}

void M2ulPhyS::parsePlasmaModels() {
  tpsP->getInput("plasma_models/ambipolar", config.ambipolar, false);
  tpsP->getInput("plasma_models/two_temperature", config.twoTemperature, false);

  if (config.twoTemperature) {
    tpsP->getInput("plasma_models/electron_temp_ic", config.initialElectronTemperature, 300.0);
  } else {
    config.initialElectronTemperature = -1;
  }

  std::string gasModelStr;
  tpsP->getInput("plasma_models/gas_model", gasModelStr, std::string("pesfect_mixture"));
  if (gasModelStr == "pesfect_mixture") {
    config.gasModel = PERFECT_MIXTURE;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown gas_model -> %s", gasModelStr.c_str());
    exit(ERROR);
  }

  std::string transportModelStr;
  tpsP->getRequiredInput("plasma_models/transport_model", transportModelStr);
  if (transportModelStr == "argon_minimal") {
    config.transportModel = ARGON_MINIMAL;
  } else if (transportModelStr == "argon_mixture") {
    config.transportModel = ARGON_MIXTURE;
  } else if (transportModelStr == "constant") {
    config.transportModel = CONSTANT;
  }
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nUnknown transport_model -> %s", transportModelStr.c_str());
  //   exit(ERROR);
  // }

  std::string chemistryModelStr;
  tpsP->getInput("plasma_models/chemistry_model", chemistryModelStr, std::string(""));

  tpsP->getInput("plasma_models/const_plasma_conductivity", config.const_plasma_conductivity_, 0.0);

  // TODO(kevin): cantera wrapper
  // if (chemistryModelStr == "cantera") {
  //   config.chemistryModel_ = ChemistryModel::CANTERA;
  // }
}

void M2ulPhyS::parseSpeciesInputs() {
  // quick return if can't use these inputs
  if (config.workFluid == LTE_FLUID) {
    config.numSpecies = 1;
    return;
  }

  // Take parameters from input. These will be sorted into config attributes.
  DenseMatrix inputGasParams;
  Vector inputCV, inputCP;
  Vector inputInitialMassFraction;
  std::vector<std::string> inputSpeciesNames;
  DenseMatrix inputSpeciesComposition;

  // number of species defined
  if (config.eqSystem != NS_PASSIVE) {
    tpsP->getInput("species/numSpecies", config.numSpecies, 1);
    inputGasParams.SetSize(config.numSpecies, GasParams::NUM_GASPARAMS);
    inputInitialMassFraction.SetSize(config.numSpecies);
    inputSpeciesNames.resize(config.numSpecies);

    if (config.gasModel == PERFECT_MIXTURE) {
      inputCV.SetSize(config.numSpecies);
      // config.constantMolarCP.SetSize(config.numSpecies);
    }
  }


   // NOTE: for now, we force the user to set the background species as the last,
   // and the electron species as the second to last.

  config.numAtoms = 0;
  if ((config.numSpecies > 1) && (config.eqSystem != NS_PASSIVE)) {
    tpsP->getRequiredInput("species/background_index", config.backgroundIndex);

    tpsP->getRequiredInput("atoms/numAtoms", config.numAtoms);
    config.atomMW.SetSize(config.numAtoms);
    for (int a = 1; a <= config.numAtoms; a++) {
      std::string basepath("atoms/atom" + std::to_string(a));

      std::string atomName;
      tpsP->getRequiredInput((basepath + "/name").c_str(), atomName);
      tpsP->getRequiredInput((basepath + "/mass").c_str(), config.atomMW(a - 1));
      config.atomMap[atomName] = a - 1;
    }
  }

  // Gas Params
  if (config.workFluid != DRY_AIR) {
    assert(config.numAtoms > 0);
    inputSpeciesComposition.SetSize(config.numSpecies, config.numAtoms);
    inputSpeciesComposition = 0.0;
    for (int i = 1; i <= config.numSpecies; i++) {
      // double mw, charge;
      double formEnergy;
      std::string type, speciesName;
      std::vector<std::pair<std::string, std::string>> composition;
      std::string basepath("species/species" + std::to_string(i));

      tpsP->getRequiredInput((basepath + "/name").c_str(), speciesName);
      inputSpeciesNames[i - 1] = speciesName;

      tpsP->getRequiredPairs((basepath + "/composition").c_str(), composition);
      for (size_t c = 0; c < composition.size(); c++) {
        if (config.atomMap.count(composition[c].first)) {
          int atomIdx = config.atomMap[composition[c].first];
          inputSpeciesComposition(i - 1, atomIdx) = stoi(composition[c].second);
        } else {
          grvy_printf(GRVY_ERROR, "Requested atom %s for species %s is not available!\n", composition[c].first.c_str(),
                      speciesName.c_str());
        }
      }

      tpsP->getRequiredInput((basepath + "/formation_energy").c_str(), formEnergy);
      inputGasParams(i - 1, GasParams::FORMATION_ENERGY) = formEnergy;

      tpsP->getRequiredInput((basepath + "/initialMassFraction").c_str(), inputInitialMassFraction(i - 1));

      if (config.gasModel == PERFECT_MIXTURE) {
        tpsP->getRequiredInput((basepath + "/pesfect_mixture/constant_molar_cv").c_str(), inputCV(i - 1));
        // NOTE: For pesfect gas, CP will be automatically set from CV.
      }
    }

    Vector speciesMass(config.numSpecies), speciesCharge(config.numSpecies);
    inputSpeciesComposition.Mult(config.atomMW, speciesMass);
    inputSpeciesComposition.GetColumn(config.atomMap["E"], speciesCharge);
    speciesCharge *= -1.0;
    inputGasParams.SetCol(GasParams::SPECIES_MW, speciesMass);
    inputGasParams.SetCol(GasParams::SPECIES_CHARGES, speciesCharge);

    // Sort the gas params for mixture and save mapping.
    {
      config.gasParams.SetSize(config.numSpecies, GasParams::NUM_GASPARAMS);
      config.initialMassFractions.SetSize(config.numSpecies);
      config.speciesNames.resize(config.numSpecies);
      config.constantMolarCV.SetSize(config.numSpecies);

      config.speciesComposition.SetSize(config.numSpecies, config.numAtoms);
      config.speciesComposition = 0.0;

      // TODO(kevin): electron species is enforced to be included in the input file.
      bool isElectronIncluded = false;

      config.gasParams = 0.0;
      int paramIdx = 0;  // new species index.
      int targetIdx;
      for (int sp = 0; sp < config.numSpecies; sp++) {  // input file species index.
        if (sp == config.backgroundIndex - 1) {
          targetIdx = config.numSpecies - 1;
        } else if (inputSpeciesNames[sp] == "E") {
          targetIdx = config.numSpecies - 2;
          isElectronIncluded = true;
        } else {
          targetIdx = paramIdx;
          paramIdx++;
        }
        config.speciesMapping[inputSpeciesNames[sp]] = targetIdx;
        config.speciesNames[targetIdx] = inputSpeciesNames[sp];
        config.mixtureToInputMap[targetIdx] = sp;
        if (mpi.Root()) {
          std::cout << "name, input index, mixture index: " << config.speciesNames[targetIdx] << ", " << sp << ", "
                    << targetIdx << std::endl;
        }

        for (int param = 0; param < GasParams::NUM_GASPARAMS; param++)
          config.gasParams(targetIdx, param) = inputGasParams(sp, param);
      }
      assert(isElectronIncluded);

      for (int sp = 0; sp < config.numSpecies; sp++) {
        int inputIdx = config.mixtureToInputMap[sp];
        config.initialMassFractions(sp) = inputInitialMassFraction(inputIdx);
        config.constantMolarCV(sp) = inputCV(inputIdx);
        for (int a = 0; a < config.numAtoms; a++)
          config.speciesComposition(sp, a) = inputSpeciesComposition(inputIdx, a);
      }
    }
  }
}

void M2ulPhyS::parseTransportInputs() {
  switch (config.transportModel) {
    case ARGON_MINIMAL: {
      // Check if unsupported species are included.
      for (int sp = 0; sp < config.numSpecies; sp++) {
        if ((config.speciesNames[sp] != "Ar") && (config.speciesNames[sp] != "Ar.+1") &&
            (config.speciesNames[sp] != "E")) {
          grvy_printf(GRVY_ERROR, "\nArgon ternary mixture transport does not support the species: %s !",
                      config.speciesNames[sp].c_str());
          exit(ERROR);
        }
      }
      tpsP->getInput("plasma_models/transport_model/argon_minimal/third_order_thermal_conductivity",
                     config.thirdOrderkElectron, true);

      // pack up argon minimal transport input.
      {
        if (config.speciesMapping.count("Ar")) {
          config.argonTransportInput.neutralIndex = config.speciesMapping["Ar"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar' !\n");
          exit(ERROR);
        }
        if (config.speciesMapping.count("Ar.+1")) {
          config.argonTransportInput.ionIndex = config.speciesMapping["Ar.+1"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar.+1' !\n");
          exit(ERROR);
        }
        if (config.speciesMapping.count("E")) {
          config.argonTransportInput.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
          exit(ERROR);
        }

        config.argonTransportInput.thirdOrderkElectron = config.thirdOrderkElectron;

        // inputs for artificial transport multipliers.
        {
          tpsP->getInput("plasma_models/transport_model/artificial_multiplier/enabled",
                         config.argonTransportInput.multiply, false);
          if (config.argonTransportInput.multiply) {
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency",
                           config.argonTransportInput.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/diffusivity",
                           config.argonTransportInput.diffMult, 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/mobility",
                           config.argonTransportInput.mobilMult, 1.0);
          }
        }
      }
    } break;
    case ARGON_MIXTURE: {
      tpsP->getInput("plasma_models/transport_model/argon_mixture/third_order_thermal_conductivity",
                     config.thirdOrderkElectron, true);

      // pack up argon transport input.
      {
        if (config.speciesMapping.count("E")) {
          config.argonTransportInput.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
          exit(ERROR);
        }

        config.argonTransportInput.thirdOrderkElectron = config.thirdOrderkElectron;

        Array<ArgonSpcs> speciesType(config.numSpecies);
        identifySpeciesType(speciesType);
        identifyCollisionType(speciesType, config.argonTransportInput.collisionIndex);

        // inputs for artificial transport multipliers.
        {
          tpsP->getInput("plasma_models/transport_model/artificial_multiplier/enabled",
                         config.argonTransportInput.multiply, false);
          if (config.argonTransportInput.multiply) {
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency",
                           config.argonTransportInput.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/diffusivity",
                           config.argonTransportInput.diffMult, 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/mobility",
                           config.argonTransportInput.mobilMult, 1.0);
          }
        }
      }
    } break;
    case CONSTANT: {
      tpsP->getRequiredInput("plasma_models/transport_model/constant/viscosity", config.constantTransport.viscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/bulk_viscosity",
                             config.constantTransport.bulkViscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/thermal_conductivity",
                             config.constantTransport.thermalConductivity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/electron_thermal_conductivity",
                             config.constantTransport.electronThermalConductivity);
      std::string diffpath("plasma_models/transport_model/constant/diffusivity");
      // config.constantTransport.diffusivity.SetSize(config.numSpecies);
      std::string mtpath("plasma_models/transport_model/constant/momentum_transfer_frequency");
      // config.constantTransport.mtFreq.SetSize(config.numSpecies);
      for (int sp = 0; sp < config.numSpecies; sp++) {
        config.constantTransport.diffusivity[sp] = 0.0;
        config.constantTransport.mtFreq[sp] = 0.0;
      }
      for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index.
        int inputSp = config.mixtureToInputMap[sp];
        tpsP->getRequiredInput((diffpath + "/species" + std::to_string(inputSp + 1)).c_str(),
                               config.constantTransport.diffusivity[sp]);
        if (config.twoTemperature)
          tpsP->getRequiredInput((mtpath + "/species" + std::to_string(inputSp + 1)).c_str(),
                                 config.constantTransport.mtFreq[sp]);
      }
      config.constantTransport.electronIndex = -1;
      if (config.IsTwoTemperature()) {
        if (config.speciesMapping.count("E")) {
          config.constantTransport.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nConstant transport: two-temperature plasma requires the species 'E' !\n");
          exit(ERROR);
        }
      }
    } break;
    default:
      break;
  }
}

void M2ulPhyS::parseReactionInputs() {
  tpsP->getInput("reactions/number_of_reactions", config.numReactions, 0);
  if (config.numReactions > 0) {
    assert((config.workFluid != DRY_AIR) && (config.numSpecies > 1));
    config.reactionEnergies.SetSize(config.numReactions);
    config.reactionEquations.resize(config.numReactions);
    config.reactionModels.SetSize(config.numReactions);
    config.reactantStoich.SetSize(config.numSpecies, config.numReactions);
    config.productStoich.SetSize(config.numSpecies, config.numReactions);

    // config.reactionModelParams.resize(config.numReactions);
    config.detailedBalance.SetSize(config.numReactions);
    // config.equilibriumConstantParams.resize(config.numReactions);
  }
  config.rxnModelParamsHost.clear();

  for (int r = 1; r <= config.numReactions; r++) {
    std::string basepath("reactions/reaction" + std::to_string(r));

    std::string equation, model;
    tpsP->getRequiredInput((basepath + "/equation").c_str(), equation);
    config.reactionEquations[r - 1] = equation;
    tpsP->getRequiredInput((basepath + "/model").c_str(), model);

    double energy;
    tpsP->getRequiredInput((basepath + "/reaction_energy").c_str(), energy);
    config.reactionEnergies[r - 1] = energy;

    // NOTE: reaction inputs are stored directly into ChemistryInput.
    // Initialize the pointers with null.
    config.chemistryInput.reactionInputs[r - 1].modelParams = NULL;

    if (model == "arrhenius") {
      config.reactionModels[r - 1] = ARRHENIUS;

      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.rxnModelParamsHost.push_back(Vector({A, b, E}));

    } else if (model == "hoffert_lien") {
      config.reactionModels[r - 1] = HOFFERTLIEN;
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.rxnModelParamsHost.push_back(Vector({A, b, E}));

    } else if (model == "tabulated") {
      config.reactionModels[r - 1] = TABULATED_RXN;
      std::string inputPath(basepath + "/tabulated");
      readTable(inputPath, config.chemistryInput.reactionInputs[r - 1].tableInput);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown reaction_model -> %s", model.c_str());
      exit(ERROR);
    }

    bool detailedBalance;
    tpsP->getInput((basepath + "/detailed_balance").c_str(), detailedBalance, false);
    config.detailedBalance[r - 1] = detailedBalance;
    if (detailedBalance) {
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/E").c_str(), E);

      // NOTE(kevin): this array keeps max param in the indexing, as reactions can have different number of params.
      config.equilibriumConstantParams[0 + (r - 1) * gpudata::MAXCHEMPARAMS] = A;
      config.equilibriumConstantParams[1 + (r - 1) * gpudata::MAXCHEMPARAMS] = b;
      config.equilibriumConstantParams[2 + (r - 1) * gpudata::MAXCHEMPARAMS] = E;
    }

    Array<double> stoich(config.numSpecies);
    tpsP->getRequiredVec((basepath + "/reactant_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) {
      int inputSp = config.mixtureToInputMap[sp];
      config.reactantStoich(sp, r - 1) = stoich[inputSp];
    }
    tpsP->getRequiredVec((basepath + "/product_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) {
      int inputSp = config.mixtureToInputMap[sp];
      config.productStoich(sp, r - 1) = stoich[inputSp];
    }
  }

  // check conservations.
  {
    const int numReactions_ = config.numReactions;
    const int numSpecies_ = config.numSpecies;
    for (int r = 0; r < numReactions_; r++) {
      Vector react(numSpecies_), product(numSpecies_);
      config.reactantStoich.GetColumn(r, react);
      config.productStoich.GetColumn(r, product);

      // atom conservation.
      DenseMatrix composition;
      composition.Transpose(config.speciesComposition);
      Vector reactAtom(config.numAtoms), prodAtom(config.numAtoms);
      composition.Mult(react, reactAtom);
      composition.Mult(product, prodAtom);
      for (int a = 0; a < config.numAtoms; a++) {
        if (reactAtom(a) != prodAtom(a)) {
          grvy_printf(GRVY_ERROR, "Reaction %d does not conserve atom %d.\n", r, a);
          exit(-1);
        }
      }

      // mass conservation. (already ensured with atom but checking again.)
      double reactMass = 0.0, prodMass = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        // int inputSp = (*mixtureToInputMap_)[sp];
        reactMass += react(sp) * config.gasParams(sp, SPECIES_MW);
        prodMass += product(sp) * config.gasParams(sp, SPECIES_MW);
      }
      // This may be too strict..
      if (abs(reactMass - prodMass) > 1.0e-15) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve mass.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E =/= %.8E\n", reactMass, prodMass);
        exit(-1);
      }

      // energy conservation.
      // TODO(kevin): this will need an adjustion when radiation comes into play.
      double reactEnergy = 0.0, prodEnergy = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        // int inputSp = (*mixtureToInputMap_)[sp];
        reactEnergy += react(sp) * config.gasParams(sp, FORMATION_ENERGY);
        prodEnergy += product(sp) * config.gasParams(sp, FORMATION_ENERGY);
      }
      // This may be too strict..
      if (reactEnergy + config.reactionEnergies[r] != prodEnergy) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve energy.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E + %.8E = %.8E =/= %.8E\n", reactEnergy, config.reactionEnergies[r],
                    reactEnergy + config.reactionEnergies[r], prodEnergy);
        exit(-1);
      }
    }
  }

  // Pack up chemistry input for instantiation.
  {
    config.chemistryInput.model = config.chemistryModel_;
    const int numSpecies = config.numSpecies;
    if (config.speciesMapping.count("E")) {
      config.chemistryInput.electronIndex = config.speciesMapping["E"];
    } else {
      config.chemistryInput.electronIndex = -1;
    }

    int rxn_param_idx = 0;

    config.chemistryInput.numReactions = config.numReactions;
    for (int r = 0; r < config.numReactions; r++) {
      config.chemistryInput.reactionEnergies[r] = config.reactionEnergies[r];
      config.chemistryInput.detailedBalance[r] = config.detailedBalance[r];
      for (int sp = 0; sp < config.numSpecies; sp++) {
        config.chemistryInput.reactantStoich[sp + r * numSpecies] = config.reactantStoich(sp, r);
        config.chemistryInput.productStoich[sp + r * numSpecies] = config.productStoich(sp, r);
      }

      config.chemistryInput.reactionModels[r] = config.reactionModels[r];
      for (int p = 0; p < gpudata::MAXCHEMPARAMS; p++) {
        config.chemistryInput.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS] =
            config.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS];
      }

      if (config.reactionModels[r] != TABULATED_RXN) {
        assert(rxn_param_idx < config.rxnModelParamsHost.size());
        config.chemistryInput.reactionInputs[r].modelParams = config.rxnModelParamsHost[rxn_param_idx].Read();
        rxn_param_idx += 1;
      }
    }
  }
}
*/

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

/*
void M2ulPhyS::parseSpongeZoneInputs() {
  tpsP->getInput("spongezone/numSpongeZones", config.numSpongeRegions_, 0);

  if (config.numSpongeRegions_ > 0) {
    config.spongeData_ = new SpongeZoneData[config.numSpongeRegions_];

    for (int sz = 0; sz < config.numSpongeRegions_; sz++) {
      std::string base("spongezone" + std::to_string(sz + 1));

      std::string type;
      tpsP->getInput((base + "/type").c_str(), type, std::string("none"));

      if (type == "none") {
        grvy_printf(GRVY_ERROR, "\nUnknown sponge zone type -> %s\n", type.c_str());
        exit(ERROR);
      } else if (type == "planar") {
        config.spongeData_[sz].szType = SpongeZoneType::PLANAR;
      } else if (type == "annulus") {
        config.spongeData_[sz].szType = SpongeZoneType::ANNULUS;

        tpsP->getInput((base + "/r1").c_str(), config.spongeData_[sz].r1, 0.);
        tpsP->getInput((base + "/r2").c_str(), config.spongeData_[sz].r2, 0.);
      }

      config.spongeData_[sz].normal.SetSize(3);
      config.spongeData_[sz].point0.SetSize(3);
      config.spongeData_[sz].pointInit.SetSize(3);
      config.spongeData_[sz].targetUp.SetSize(5 + config.numSpecies + 2);  // always large enough now

      tpsP->getRequiredVec((base + "/normal").c_str(), config.spongeData_[sz].normal, 3);
      tpsP->getRequiredVec((base + "/p0").c_str(), config.spongeData_[sz].point0, 3);
      tpsP->getRequiredVec((base + "/pInit").c_str(), config.spongeData_[sz].pointInit, 3);
      tpsP->getInput((base + "/tolerance").c_str(), config.spongeData_[sz].tol, 1e-5);
      tpsP->getInput((base + "/multiplier").c_str(), config.spongeData_[sz].multFactor, 1.0);

      tpsP->getRequiredInput((base + "/targetSolType").c_str(), type);
      if (type == "userDef") {
        config.spongeData_[sz].szSolType = SpongeZoneSolution::USERDEF;
        auto hup = config.spongeData_[sz].targetUp.HostWrite();
        tpsP->getRequiredInput((base + "/density").c_str(), hup[0]);   // rho
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[1], 0);  // u
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[2], 1);  // v
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[3], 2);  // w
        tpsP->getRequiredInput((base + "/pressure").c_str(), hup[4]);  // P

        // For multi-component gas, require (numActiveSpecies)-more inputs.
        if (config.workFluid != DRY_AIR) {
          if (config.numSpecies > 1) {
            grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
            if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

            for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index.
              // read mass fraction of species as listed in the input file.
              int inputSp = config.mixtureToInputMap[sp];
              std::string speciesBasePath(base + "/mass_fraction/species" + std::to_string(inputSp + 1));
              tpsP->getRequiredInput(speciesBasePath.c_str(), hup[5 + sp]);
            }
          }

          if (config.twoTemperature) {
            tpsP->getInput((base + "/single_temperature").c_str(), config.spongeData_[sz].singleTemperature, false);
            if (!config.spongeData_[sz].singleTemperature) {
              tpsP->getRequiredInput((base + "/electron_temperature").c_str(), hup[5 + config.numSpecies]);  // P
            }
          }
        }

      } else if (type == "mixedOut") {
        config.spongeData_[sz].szSolType = SpongeZoneSolution::MIXEDOUT;
      } else {
        grvy_printf(GRVY_ERROR, "\nUnknown sponge zone type -> %s\n", type.c_str());
        exit(ERROR);
      }
    }
  }
}
*/

void LoMachSolver::parsePostProcessVisualizationInputs() {
  if (tpsP_->isVisualizationMode()) {
    tpsP_->getRequiredInput("post-process/visualization/prefix", config.postprocessInput.prefix);
    tpsP_->getRequiredInput("post-process/visualization/start-iter", config.postprocessInput.startIter);
    tpsP_->getRequiredInput("post-process/visualization/end-iter", config.postprocessInput.endIter);
    tpsP_->getRequiredInput("post-process/visualization/frequency", config.postprocessInput.freq);
  }
}

/*
void M2ulPhyS::parseRadiationInputs() {
  std::string type;
  std::string basepath("plasma_models/radiation_model");

  tpsP->getInput(basepath.c_str(), type, std::string("none"));
  std::string modelInputPath(basepath + "/" + type);

  if (type == "net_emission") {
    config.radiationInput.model = NET_EMISSION;

    std::string coefficientType;
    tpsP->getRequiredInput((modelInputPath + "/coefficient").c_str(), coefficientType);
    if (coefficientType == "tabulated") {
      config.radiationInput.necModel = TABULATED_NEC;
      std::string inputPath(modelInputPath + "/tabulated");
      readTable(inputPath, config.radiationInput.necTableInput);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown net emission coefficient type -> %s\n", coefficientType.c_str());
      exit(ERROR);
    }
  } else if (type == "none") {
    config.radiationInput.model = NONE_RAD;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown radiation model -> %s\n", type.c_str());
    exit(ERROR);
  }
}
*/

void LoMachSolver::parsePeriodicInputs() {
  tpsP_->getInput("periodicity/enablePeriodic", config.periodic, false);
  tpsP_->getInput("periodicity/xTrans", config.xTrans, 1.0e12);
  tpsP_->getInput("periodicity/yTrans", config.yTrans, 1.0e12);
  tpsP_->getInput("periodicity/zTrans", config.zTrans, 1.0e12);
}

/*
void M2ulPhyS::packUpGasMixtureInput() {
  if (config.workFluid == DRY_AIR) {
    config.dryAirInput.f = config.workFluid;
    config.dryAirInput.eq_sys = config.eqSystem;
#if defined(_HIP_)
    config.dryAirInput.visc_mult = config.visc_mult;
    config.dryAirInput.bulk_visc_mult = config.bulk_visc;
#endif
  } else if (config.workFluid == USER_DEFINED) {
    switch (config.gasModel) {
      case PERFECT_MIXTURE: {
        config.pesfectMixtureInput.f = config.workFluid;
        config.pesfectMixtureInput.numSpecies = config.numSpecies;
        if (config.speciesMapping.count("E")) {
          config.pesfectMixtureInput.isElectronIncluded = true;
        } else {
          config.pesfectMixtureInput.isElectronIncluded = false;
        }
        config.pesfectMixtureInput.ambipolar = config.ambipolar;
        config.pesfectMixtureInput.twoTemperature = config.twoTemperature;

        for (int sp = 0; sp < config.numSpecies; sp++) {
          for (int param = 0; param < (int)GasParams::NUM_GASPARAMS; param++) {
            config.pesfectMixtureInput.gasParams[sp + param * config.numSpecies] = config.gasParams(sp, param);
          }
          config.pesfectMixtureInput.molarCV[sp] = config.constantMolarCV(sp);
        }
      } break;
      default:
        grvy_printf(GRVY_ERROR, "Gas model is not specified!\n");
        exit(ERROR);
        break;
    }  // switch gasModel
  } else if (config.workFluid == LTE_FLUID) {
    config.lteMixtureInput.f = config.workFluid;
    // config.lteMixtureInput.thermo_file_name // already set in parseFluidPreset
    assert(config.numSpecies == 1);  // inconsistent to specify lte and carry species
    assert(!config.twoTemperature);  // inconsistent to specify lte and have two temperatures
  }
}
*/

void LoMachSolver::initSolutionAndVisualizationVectors() {
  // std::cout << " In initSol&Viz..." << endl;
  visualizationVariables_.clear();
  visualizationNames_.clear();

  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) {
    (*offsets)[k] = k * fvfes->GetNDofs();
  }

  u_block = new BlockVector(*offsets);
  up_block = new BlockVector(*offsets);

  // gradUp.SetSize(num_equation*dim*vfes->GetNDofs());
  // gradUp = new ParGridFunction(gradUpfes);

  U = new ParGridFunction(fvfes, u_block->HostReadWrite());
  Up = new ParGridFunction(fvfes, up_block->HostReadWrite());

  // dens = new ParGridFunction(sfes, Up->HostReadWrite());
  // vel = new ParGridFunction(vfes, Up->HostReadWrite() + sfes->GetNDofs());
  // std::cout << " check 8..." << endl;
  // temperature = new ParGridFunction(sfes, Up->HostReadWrite() + (1 + nvel) * sfes->GetNDofs()); // this may break...
  // std::cout << " check 9..." << endl;
  // press = new ParGridFunction(pfes);
  // std::cout << " check 10..." << endl;

  // if (config.isAxisymmetric()) {
  //   vtheta = new ParGridFunction(fes, Up->HostReadWrite() + 3 * fes->GetNDofs());
  // } else {
  //   vtheta = NULL;
  // }

  // electron_temp_field = NULL;
  // if (config.twoTemperature) {
  //   electron_temp_field = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());
  // }

  /*
  passiveScalar = NULL;
  if (eqSystem == NS_PASSIVE) {
    passiveScalar = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());
  } else {
    // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
    // will need to add full list of species.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      std::string speciesName = config.speciesNames[sp];
      visualizationVariables_.push_back(
          new ParGridFunction(fes, U->HostReadWrite() + (sp + nvel + 2) * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("partial_density_" + speciesName));
    }
  }
  */

  // add visualization variables if tps is run on post-process visualization mode.
  // TODO(kevin): maybe enable users to specify what to visualize.
  /*
  if (tpsP->isVisualizationMode()) {
    if (config.workFluid != DRY_AIR) {
      // species primitives.
      visualizationIndexes_.Xsp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("X_" + speciesName));
      }
      visualizationIndexes_.Ysp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("Y_" + speciesName));
      }
      visualizationIndexes_.nsp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("n_" + speciesName));
      }

      // transport properties.
      visualizationIndexes_.FluxTrns = visualizationVariables_.size();
      for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) {
        std::string fieldName;
        switch (t) {
          case FluxTrns::VISCOSITY:
            fieldName = "viscosity";
            break;
          case FluxTrns::BULK_VISCOSITY:
            fieldName = "bulk_viscosity";
            break;
          case FluxTrns::HEAVY_THERMAL_CONDUCTIVITY:
            fieldName = "thermal_cond_heavy";
            break;
          case FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY:
            fieldName = "thermal_cond_elec";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown flux transport property!");
            exit(ERROR);
            break;
        }
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(fieldName);
      }
      visualizationIndexes_.diffVel = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(nvelfes));
        visualizationNames_.push_back(std::string("diff_vel_" + speciesName));
      }
      visualizationIndexes_.SrcTrns = visualizationVariables_.size();
      for (int t = 0; t < SrcTrns::NUM_SRC_TRANS; t++) {
        std::string fieldName;
        switch (t) {
          case SrcTrns::ELECTRIC_CONDUCTIVITY:
            fieldName = "electric_cond";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown source transport property!");
            exit(ERROR);
            break;
        }
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(fieldName);
      }
      visualizationIndexes_.SpeciesTrns = visualizationVariables_.size();
      for (int t = 0; t < SpeciesTrns::NUM_SPECIES_COEFFS; t++) {
        std::string fieldName;
        switch (t) {
          case SpeciesTrns::MF_FREQUENCY:
            fieldName = "momentum_tranfer_freq";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown species transport property!");
            exit(ERROR);
            break;
        }
        for (int sp = 0; sp < numSpecies; sp++) {
          std::string speciesName = config.speciesNames[sp];
          visualizationVariables_.push_back(new ParGridFunction(fes));
          visualizationNames_.push_back(std::string(fieldName + "_" + speciesName));
        }
      }

      // chemistry reaction rates.
      visualizationIndexes_.rxn = visualizationVariables_.size();
      for (int r = 0; r < config.numReactions; r++) {
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("rxn_rate_" + std::to_string(r + 1)));
        // visualizationNames_.push_back(std::string("rxn_rate: " + config.reactionEquations[r]));
      }
    }  // if (config.workFluid != DRY_AIR)
  }    // if tpsP->isVisualizationMode()

  // If mms, add conserved and exact solution.
#ifdef HAVE_MASA
  if (config.use_mms_ && config.mmsSaveDetails_) {
    // for visualization.
    masaUBlock_ = new BlockVector(*offsets);
    masaU_ = new ParGridFunction(vfes, masaUBlock_->HostReadWrite());
    masaRhs_ = new ParGridFunction(*U);

    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, U->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("U" + std::to_string(eq)));
    }
    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, masaU_->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("mms_U" + std::to_string(eq)));
    }
    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, masaRhs_->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("RHS" + std::to_string(eq)));
    }
  }
#endif

  plasma_conductivity_ = NULL;
  if (tpsP->isFlowEMCoupled()) {
    plasma_conductivity_ = new ParGridFunction(fes);
    *plasma_conductivity_ = 0.0;
  }

  joule_heating_ = NULL;
  if (tpsP->isFlowEMCoupled()) {
    joule_heating_ = new ParGridFunction(fes);
    *joule_heating_ = 0.0;
  }
  */

  // define solution parameters for i/o
  /*
  ioData.registerIOFamily("Solution state variables", "/solution", U);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "rho-u", 1);
  ioData.registerIOVar("/solution", "rho-v", 2);
  if (nvel == 3) {
    ioData.registerIOVar("/solution", "rho-w", 3);
    ioData.registerIOVar("/solution", "rho-E", 4);
  } else {
    ioData.registerIOVar("/solution", "rho-E", 3);
  }
  */

  ioData.registerIOFamily("Solution state variables", "/solution", Up, false);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "u", 1);
  ioData.registerIOVar("/solution", "v", 2);
  if (nvel == 3) {
    ioData.registerIOVar("/solution", "w", 3);
    ioData.registerIOVar("/solution", "temperature", 4);
  } else {
    ioData.registerIOVar("/solution", "temperature", 3);
  }

  /*
  // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
  // will need to add full list of species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // Only for NS_PASSIVE.
    if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

    std::string speciesName = config.speciesNames[sp];
    ioData.registerIOVar("/solution", "rho-Y_" + speciesName, sp + nvel + 2);
  }

  if (config.twoTemperature) {
    ioData.registerIOVar("/solution", "rhoE_e", num_equation - 1);
  }
  */

  // compute factor to multiply viscosity when this option is active
  // spaceVaryViscMult = NULL;
  // this should be moved to Setup
  /*
  bufferViscMult = new ParGridFunction(sfes);
  {
    double *data = bufferViscMult->HostReadWrite();
    for (int i = 0; i < sfes->GetNDofs(); i++) { data[i] = 1.0; }
  }
  ParGridFunction coordsDof(vfes);
  pmesh->GetNodes(coordsDof);
  if (config.linViscData.isEnabled) {
    if (rank0_) std::cout << "Viscous sponge active" << endl;
    double *viscMult = bufferViscMult->HostReadWrite();
    double *hcoords = coordsDof.HostReadWrite();
    double wgt = 0.;
    for (int n = 0; n < sfes->GetNDofs(); n++) {
      double coords[3];
      for (int d = 0; d < dim; d++) { coords[d] = hcoords[n + d * sfes->GetNDofs()]; }
      viscSpongePlanar(coords, wgt);
      viscMult[n] = wgt + (config.linViscData.uniformMult - 1.0);
      //std::cout << pmesh->GetMyRank() << ")" << " Sponge weight: " << wgt << " coords: " << coords[0] << " " <<
  coords[1] << " " << coords[2] << endl;
    }
  }

  //HERE HERE HERE
  bufferViscMult->GetTrueDofs(viscMultSml);
  */

  /*
  int SdofInt = sfes->GetTrueVSize();
  Vector coordsVec;
  coordsVec.SetSize(3*SdofInt);
  Tn_gf.GetTrueDofs(coordsVec);
  if (config.linViscData.isEnabled) {
    double *dataViscMultSml = viscMultSml.HostReadWrite();
    double wgt = 0.;
    double coords[3];
    for (int n = 0; n < SdofInt; n++) {
      for (int d = 0; d < dim; d++) { coords[d] = coordsVec[n + d * SdofInt]; }
      viscSpongePlanar(coords, wgt);
      dataViscMultSml[n] = wgt;
    }
  }
  */

  /*
  paraviewColl->SetCycle(0);
  //std::cout << " check 15..." << endl;
  paraviewColl->SetTime(0.);
  //std::cout << " check 16..." << endl;

  paraviewColl->RegisterField("dens", dens);
  paraviewColl->RegisterField("vel", vel);
  //std::cout << " check 17..." << endl;
  paraviewColl->RegisterField("temp", temperature);
  //std::cout << " check 18..." << endl;
  paraviewColl->RegisterField("press", press);
  //std::cout << " check 19..." << endl;
  */

  // if (config.isAxisymmetric()) {
  //   paraviewColl->RegisterField("vtheta", vtheta);
  // }

  // if (eqSystem == NS_PASSIVE) {
  //   paraviewColl->RegisterField("passiveScalar", passiveScalar);
  // }

  // if (config.twoTemperature) {
  //   paraviewColl->RegisterField("Te", electron_temp_field);
  // }

  /*
  for (int var = 0; var < visualizationVariables_.size(); var++) {
    paraviewColl->RegisterField(visualizationNames_[var], visualizationVariables_[var]);
  }
  //std::cout << " check 20..." << endl;

  //if (spaceVaryViscMult != NULL) paraviewColl->RegisterField("viscMult", spaceVaryViscMult);
  //if (distance_ != NULL) paraviewColl->RegisterField("distance", distance_);

  paraviewColl->SetOwnData(true);
  //paraviewColl->Save();
  //std::cout << " check 21..." << endl;
  */
}

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst) {
  Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
  for (int i = 0; i < bffis->Size(); ++i) {
    dst->AddDomainIntegrator((*bffis)[i]);
  }
}
