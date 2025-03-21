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
/*
 * Implementation of the class Problem
 */

#include "M2ulPhyS.hpp"

#include "gslib_interpolator.hpp"
#include "utils.hpp"
#include "wallBC.hpp"

M2ulPhyS::M2ulPhyS(TPS::Tps *tps)
    : groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      rank0_(groupsMPI->isWorldRoot()),
      tpsP(tps) {
  parseSolverOptions2();
}

M2ulPhyS::M2ulPhyS(string &inputFileName, TPS::Tps *tps)
    : groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      rank_(groupsMPI->getTPSWorldRank()),
      rank0_(groupsMPI->isWorldRoot()),
      tpsP(tps) {
  // koomie TODO: refactor order so we can use standard parseSolverOptions call from Solver base class
#define NEWPARSER
#ifdef NEWPARSER
  parseSolverOptions2();
#else
  config.readInputFile(inputFileName);
#endif

  checkSolverOptions();

  initVariables();

  groupsMPI->init();

  // now that MPI groups have been initialized, we can finalize any additional
  // BC setup that requries coordination across processors
  if (bcIntegrator != NULL) {
    bcIntegrator->initBCs();
  }

  initIndirectionBC();

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

void M2ulPhyS::initMixtureAndTransportModels() {
  mixture = NULL;

  switch (config.GetWorkingFluid()) {
    case WorkingFluid::DRY_AIR:
      mixture = new DryAir(config, dim, nvel);

#if defined(_CUDA_) || defined(_HIP_)
      tpsGpuMalloc((void **)(&d_mixture), sizeof(DryAir));
      gpu::instantiateDeviceDryAir<<<1, 1>>>(config.dryAirInput, dim, nvel, d_mixture);

      tpsGpuMalloc((void **)&transportPtr, sizeof(DryAirTransport));
      gpu::instantiateDeviceDryAirTransport<<<1, 1>>>(d_mixture, config.GetViscMult(), config.GetBulkViscMult(),
                                                      config.sutherland_.C1, config.sutherland_.S0,
                                                      config.sutherland_.Pr, transportPtr);
#else
      transportPtr = new DryAirTransport(mixture, config);
#endif
      break;
    case WorkingFluid::USER_DEFINED:
      switch (config.GetGasModel()) {
        case GasModel::PERFECT_MIXTURE:
          mixture = new PerfectMixture(config, dim, nvel);
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)(&d_mixture), sizeof(PerfectMixture));
          gpu::instantiateDevicePerfectMixture<<<1, 1>>>(config.perfectMixtureInput, dim, nvel, d_mixture);
#endif
          break;
        default:
          mfem_error("GasModel not recognized.");
          break;
      }

      switch (config.GetTranportModel()) {
        case ARGON_MINIMAL:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ArgonMinimalTransport));
          gpu::instantiateDeviceArgonMinimalTransport<<<1, 1>>>(d_mixture, config.argonTransportInput, transportPtr);
#else
          transportPtr = new ArgonMinimalTransport(mixture, config);
#endif
          break;
        case ARGON_MIXTURE:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ArgonMixtureTransport));
          gpu::instantiateDeviceArgonMixtureTransport<<<1, 1>>>(d_mixture, config.argonTransportInput, transportPtr);
#else
          transportPtr = new ArgonMixtureTransport(mixture, config);
#endif
          break;
        case CONSTANT:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ConstantTransport));
          gpu::instantiateDeviceConstantTransport<<<1, 1>>>(d_mixture, config.constantTransport, transportPtr);
#else
          transportPtr = new ConstantTransport(mixture, config);
#endif
          break;
        default:
          mfem_error("TransportModel not recognized.");
          break;
      }
      switch (config.GetChemistryModel()) {
        default:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&chemistry_, sizeof(Chemistry));
          gpu::instantiateDeviceChemistry<<<1, 1>>>(d_mixture, config.chemistryInput, chemistry_);
#else
          chemistry_ = new Chemistry(mixture, config.chemistryInput);
#endif
          break;
      }
      break;
    case WorkingFluid::LTE_FLUID:
      int table_dim;
      tpsP->getInput("flow/lte/table_dim", table_dim, 2);
      if (table_dim == 2) {
#if defined(_GPU_)
        mfem_error("LTE_FLUID with 2D lookup tables not supported for GPU.");
#else
        // for 2D case, table read happens internally to LteMixture and LteTransport
        mixture = new LteMixture(config, dim, nvel);
        transportPtr = new LteTransport(mixture, config);
#endif
      } else if (table_dim == 1) {
        MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();

        // Initialize thermo TableInput (data read below)
        std::vector<TableInput> thermo_tables(3);
        for (size_t i = 0; i < thermo_tables.size(); i++) {
          thermo_tables[i].order = 1;
          thermo_tables[i].xLogScale = false;
          thermo_tables[i].fLogScale = false;
        }

        // Read data from hdf5 file containing 4 columns: T, energy, gas constant, speed of sound
        DenseMatrix thermo_data;
        bool success;
        success = h5ReadBcastMultiColumnTable(config.lteMixtureInput.thermo_file_name, std::string("T_energy_R_c"),
                                              TPSCommWorld, thermo_data, thermo_tables);
        if (!success) exit(ERROR);

        // Construct e -> T table from T -> e
        TableInput T_table_input;
        T_table_input.order = 1;
        T_table_input.xLogScale = false;
        T_table_input.fLogScale = false;
        T_table_input.Ndata = thermo_tables[0].Ndata;
        T_table_input.xdata = thermo_tables[0].fdata;
        T_table_input.fdata = thermo_tables[0].xdata;

        // LteMixture object valid on host
        mixture = new LteMixture(config.lteMixtureInput.f, dim, nvel, config.const_plasma_conductivity_,
                                 thermo_tables[0], thermo_tables[1], thermo_tables[2], T_table_input);

#if defined(_CUDA_) || defined(_HIP_)
        // Tables from above have host pointers.  Must get device
        // pointers here before instantiating device class
        for (int icol = 0; icol < 3; icol++) {
          const int nrow = thermo_tables[icol].Ndata;
          thermo_tables[icol].xdata = thermo_data.Read();
          thermo_tables[icol].fdata = thermo_data.Read() + (icol + 1) * nrow;
        }

        T_table_input.xdata = thermo_tables[0].fdata;
        T_table_input.fdata = thermo_tables[0].xdata;

        // LteMixture object valid on device
        tpsGpuMalloc((void **)(&d_mixture), sizeof(LteMixture));
        gpu::instantiateDeviceLteMixture<<<1, 1>>>(config.lteMixtureInput.f, dim, nvel,
                                                   config.const_plasma_conductivity_, thermo_tables[0],
                                                   thermo_tables[1], thermo_tables[2], T_table_input, d_mixture);
#endif

        // Initialize transport TableInput (data read below)
        std::vector<TableInput> trans_tables(3);
        for (size_t i = 0; i < trans_tables.size(); i++) {
          trans_tables[i].order = 1;
          trans_tables[i].xLogScale = false;
          trans_tables[i].fLogScale = false;
        }

        // Read data from hdf5 file containing 4 columns: T, mu, kappa, sigma
        // (i.e., temperature, dynamic viscosity, thermal conductivity, electrical conductivity)
        DenseMatrix trans_data;
        success = h5ReadBcastMultiColumnTable(config.lteMixtureInput.trans_file_name, std::string("T_mu_kappa_sigma"),
                                              TPSCommWorld, trans_data, trans_tables);
        if (!success) exit(ERROR);

          // Instantiate LteTransport class
#if defined(_CUDA_) || defined(_HIP_)
        // Tables from above have host pointers.  Must get device
        // pointers here before instantiating device class
        for (int icol = 0; icol < 3; icol++) {
          const int nrow = trans_tables[icol].Ndata;
          trans_tables[icol].xdata = trans_data.Read();
          trans_tables[icol].fdata = trans_data.Read() + (icol + 1) * nrow;
        }

        tpsGpuMalloc((void **)&transportPtr, sizeof(LteTransport));
        gpu::instantiateDeviceLteTransport<<<1, 1>>>(d_mixture, trans_tables[0], trans_tables[1], trans_tables[2],
                                                     transportPtr);
#else
        transportPtr = new LteTransport(mixture, trans_tables[0], trans_tables[1], trans_tables[2]);
#endif
      } else {
        mfem_error("flow/lte/table_dim must be 1 or 2.");
      }
      break;
    default:
      mfem_error("WorkingFluid not recognized.");
      break;
  }

  if (config.use_mixing_length) {
#if defined(_CUDA_) || defined(_HIP_)
    // dynamic_cast on a device-side pointer doesn't work, so instead
    // pass the TransportProperties pointer and cast it within
    // instantiateDeviceMixingLengthTransport
    TransportProperties *temporary_transport = transportPtr;
    tpsGpuMalloc((void **)&transportPtr, sizeof(MixingLengthTransport));
    gpu::instantiateDeviceMixingLengthTransport<<<1, 1>>>(d_mixture, config.mix_length_trans_input_,
                                                          temporary_transport, transportPtr);
#else
    // Build mixing length transport using whatever molecular transport we've already instantiated
    MolecularTransport *temporary_transport = dynamic_cast<MolecularTransport *>(transportPtr);
    transportPtr = new MixingLengthTransport(mixture, config, temporary_transport);
#endif
  }

  assert(mixture != NULL);
#if defined(_CUDA_) || defined(_HIP_)
#else
  d_mixture = mixture;
#endif
}

void M2ulPhyS::initVariables() {
#ifdef HAVE_GRVY
  grvy_timer_init("TPS");
#endif

  loadFromAuxSol = config.RestartFromAux();

  // Generate serial mesh, making it periodic if requested
  if (config.GetPeriodic()) {
    Mesh temp_mesh = Mesh(config.GetMeshFileName().c_str());
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

    serial_mesh =
        new Mesh(std::move(Mesh::MakePeriodic(temp_mesh, temp_mesh.CreatePeriodicVertexMapping(translations))));
  } else {
    serial_mesh = new Mesh(config.GetMeshFileName().c_str());
  }

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
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      if (config.isRestartSerialized("read")) {
        assert(serial_mesh->Conforming());
        partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
        partitioning_file_hdf5("write", groupsMPI, nelemGlobal_, partitioning_);
      } else {
        partitioning_file_hdf5("read", groupsMPI, nelemGlobal_, partitioning_);
      }
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
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh->GetNE();
    if (nprocs_ > 1) {
      assert(serial_mesh->Conforming());
      partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write", groupsMPI, nelemGlobal_, partitioning_);
      MPI_Barrier(groupsMPI->getTPSCommWorld());
    }

    time = 0.;
    iter = 0;
  }

  // If requested, evaluate the distance function (i.e., the distance to the nearest no-slip wall)
  distance_ = NULL;
  GridFunction *serial_distance = NULL;
  FiniteElementSpace *serial_fes = NULL;
  DG_FECollection *tmp_fec = NULL;
  if (config.compute_distance) {
    order = config.GetSolutionOrder();
    dim = serial_mesh->Dimension();
    basisType = config.GetBasisType();
    if (basisType == 0) {
      tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLegendre);
    } else if (basisType == 1) {
      tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    }

    // Serial FE space for scalar variable
    serial_fes = new FiniteElementSpace(serial_mesh, tmp_fec);

    // Serial grid function for scalar variable
    serial_distance = new GridFunction(serial_fes);

    // Build a list of wall patches
    Array<int> wall_patch_list;
    for (size_t i = 0; i < config.wallPatchType.size(); i++) {
      if (config.wallPatchType[i].second != WallType::INV) {
        wall_patch_list.Append(config.wallPatchType[i].first);
      }
    }

    if (serial_mesh->GetNodes() == NULL) {
      serial_mesh->SetCurvature(1);
    }

    FiniteElementSpace *tmp_dfes = new FiniteElementSpace(serial_mesh, tmp_fec, dim, Ordering::byNODES);
    GridFunction coordinates(tmp_dfes);
    serial_mesh->GetNodes(coordinates);

    // Evaluate the distance function
    if (!config.read_distance || !config.GetRestartCycle()) {
      if (rank0_) grvy_printf(ginfo, "Computing distance function\n");
      evaluateDistanceSerial(*serial_mesh, wall_patch_list, coordinates, *serial_distance);
    } else {
      // If distance function is read from restart, this will be overwritten later
      if (rank0_) grvy_printf(ginfo, "Distance function to be read from restart\n");
      *serial_distance = 0.0;
    }
    delete tmp_dfes;
  }

  // Instantiate parallel mesh
  mesh = new ParMesh(groupsMPI->getTPSCommWorld(), *serial_mesh, partitioning_);

  // If necessary, parallelize distance function
  if (config.compute_distance) {
    distance_ = new ParGridFunction(mesh, serial_distance, partitioning_);
    if (partitioning_ == NULL) {
      // In this case, ctor call above does not copy data, so explicitly call assignment
      *distance_ = *serial_distance;
    }
    // Done with serial_distance
    delete serial_distance;
    delete serial_fes;
    delete tmp_fec;

    distance_->ParFESpace()->ExchangeFaceNbrData();
    distance_->ExchangeFaceNbrData();
  }

  // only need serial mesh if on rank 0 and using single restart file option
  if (!rank0_ || (!config.isRestartSerialized("either"))) delete serial_mesh;

  // Paraview setup
  paraviewColl = new ParaViewDataCollection(config.GetOutputName(), mesh);
  paraviewColl->SetLevelsOfDetail(config.GetSolutionOrder());
  paraviewColl->SetHighOrderOutput(true);
  paraviewColl->SetPrecision(8);

  // This line is necessary to ensure that GetNodes does not return
  // NULL, which is required when we set up interpolation between
  // meshes.  We do it here b/c waiting until later (e.g., right
  // before interpolation set up) leads to a seg fault when running in
  // parallel, which is not yet understood.
  if (mesh->GetNodes() == NULL) {
    mesh->SetCurvature(1);
  }

#ifdef _GPU_
  loc_print.SetSize(1);
  loc_print = 0;
#endif

  // if we have a partitioning vector, use it to build local->global
  // element numbering map
  if (partitioning_ != NULL) {
    // Assumption: the map "local element id" -> "global element id" is
    // increasing, i.e. the local numbering preserves the element order from
    // the global numbering.
    //
    // NB: This is the same assumption made by the ParGridFunction
    // ctor if you pass the partitioning vector, at least as of mfem
    // v4.2.  See mfem/fem/pgridfunction.cpp, which is the source of
    // the above comment.

    locToGlobElem = new int[mesh->GetNE()];
    int lelem = 0;
    for (int gelem = 0; gelem < nelemGlobal_; gelem++) {
      if (rank_ == partitioning_[gelem]) {
        locToGlobElem[lelem] = gelem;
        lelem += 1;
      }
    }
  } else {
    locToGlobElem = NULL;
  }

  // partitioning summary
  {
    int maxElems;
    int minElems;
    int localElems = mesh->GetNE();
    MPI_Allreduce(&localElems, &minElems, 1, MPI_INT, MPI_MIN, groupsMPI->getTPSCommWorld());
    MPI_Allreduce(&localElems, &maxElems, 1, MPI_INT, MPI_MAX, groupsMPI->getTPSCommWorld());

    if (rank0_) {
      grvy_printf(GRVY_INFO, "number of elements on rank 0 = %i\n", localElems);
      grvy_printf(GRVY_INFO, "min elements/partition       = %i\n", minElems);
      grvy_printf(GRVY_INFO, "max elements/partition       = %i\n", maxElems);
    }
  }

  dim = mesh->Dimension();
  nvel = (config.isAxisymmetric() ? 3 : dim);
  if (config.isAxisymmetric()) assert(dim == 2);

  refLength = config.GetReferenceLength();

  eqSystem = config.GetEquationSystem();

  initMixtureAndTransportModels();

  switch (config.radiationInput.model) {
    case NET_EMISSION:
#if defined(_CUDA_) || defined(_HIP_)
      tpsGpuMalloc((void **)(&radiation_), sizeof(NetEmission));
      gpu::instantiateDeviceNetEmission<<<1, 1>>>(config.radiationInput, radiation_);
#else
      radiation_ = new NetEmission(config.radiationInput);
#endif
      break;
    case NONE_RAD:
      break;
    default:
      mfem_error("RadiationModel not recognized.");
      break;
  }

  order = config.GetSolutionOrder();

  MaxIters = config.GetNumIters();

  max_char_speed = 0.;

  // Kevin: if mixture is set to a preset such as DRY_AIR,
  // it will not read the input options for following variables,
  // instead applies preset values.
  numSpecies = mixture->GetNumSpecies();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
  // // Kevin: the code above replaces this num_equation determination.
  // switch (eqSystem) {
  //   case EULER:
  //     num_equation = 2 + dim;
  //     break;
  //   case NS:
  //     num_equation = 2 + dim;
  //     break;
  //   case NS_PASSIVE:
  //     num_equation = 2 + dim + 1;
  //     break;
  //   default:
  //     break;
  // }

  // initialize basis type and integration rule
  intRuleType = config.GetIntegrationRule();
  if (intRuleType == 0) {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLegendre);
  } else if (intRuleType == 1) {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLobatto);
  }

  basisType = config.GetBasisType();
  if (basisType == 0) {
    fec = new DG_FECollection(order, dim, BasisType::GaussLegendre);
    // fec  = new H1_FECollection(order, dim, BasisType::GaussLegendre);
  } else if (basisType == 1) {
    // This basis type includes end-nodes
    fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    // fec  = new H1_FECollection(order, dim, BasisType::GaussLobatto);
  }

  // FE Spaces
  fes = new ParFiniteElementSpace(mesh, fec);
  dfes = new ParFiniteElementSpace(mesh, fec, dim, Ordering::byNODES);
  nvelfes = new ParFiniteElementSpace(mesh, fec, nvel, Ordering::byNODES);
  vfes = new ParFiniteElementSpace(mesh, fec, num_equation, Ordering::byNODES);
  gradUpfes = new ParFiniteElementSpace(mesh, fec, num_equation * dim, Ordering::byNODES);

#if defined(_CUDA_) || defined(_HIP_)
  // prepare viscous sponge data to pass to gpu ctor
  viscositySpongeData vsd;
  vsd.enabled = config.linViscData.isEnabled;
  vsd.n[0] = vsd.n[1] = vsd.n[2] = 0.;
  vsd.p[0] = vsd.p[1] = vsd.p[2] = 0.;
  vsd.ratio = 1.0;
  vsd.width = 1.0;

  if (vsd.enabled) {
    for (int d = 0; d < dim; d++) {
      vsd.n[d] = config.GetLinearVaryingData().normal(d);
      vsd.p[d] = config.GetLinearVaryingData().point0(d);
    }
    vsd.ratio = config.GetLinearVaryingData().viscRatio;
    vsd.width = config.GetLinearVaryingData().width;
  }

  tpsGpuMalloc((void **)&d_fluxClass, sizeof(Fluxes));
  gpu::instantiateDeviceFluxes<<<1, 1>>>(d_mixture, eqSystem, transportPtr, num_equation, dim, config.isAxisymmetric(),
                                         config.GetSgsModelType(), config.GetSgsFloor(), config.GetSgsConstant(), vsd,
                                         d_fluxClass);

  tpsGpuMalloc((void **)&rsolver, sizeof(RiemannSolver));
  gpu::instantiateDeviceRiemann<<<1, 1>>>(num_equation, d_mixture, eqSystem, d_fluxClass, config.RoeRiemannSolver(),
                                          config.isAxisymmetric(), rsolver);

  // Note: This flux class is only used to compute the viscosity
  // multiplier function on the host.  If you want or need it for
  // anything else, be careful.  It does not have a valid pointer to a
  // transport class!
  fluxClass = new Fluxes(mixture, eqSystem, NULL, num_equation, dim, config.isAxisymmetric(), &config);

#else

  fluxClass = new Fluxes(mixture, eqSystem, transportPtr, num_equation, dim, config.isAxisymmetric(), &config);
  d_fluxClass = fluxClass;

  rsolver = new RiemannSolver(num_equation, mixture, eqSystem, d_fluxClass, config.RoeRiemannSolver(),
                              config.isAxisymmetric());
#endif

#ifdef _GPU_
  initIndirectionArrays();
#endif
  initSolutionAndVisualizationVectors();

  if (distance_ != NULL) {
    ioData.registerIOFamily("Distance function", "/distance", distance_, false, config.read_distance);
    ioData.registerIOVar("/distance", "distance", 0, config.read_distance);
  }

  average = new Averaging(config.avg_opts_, config.GetOutputName());
  average->registerField(std::string("primitive_state"), Up, true, 1, nvel);
  average->initializeVizForM2ulPhyS(fes, dfes, nvel);

  // NOTE: this should also be completed by the GasMixture class
  // Kevin: Do we need GasMixture class for this?
  // register rms and mean sol into ioData
  if (average->ComputeMean()) {
    // meanUp
    ioData.registerIOFamily("Time-averaged primitive vars", "/meanSolution",
                            average->GetMeanField(std::string("primitive_state")), false, config.GetRestartMean());
    ioData.registerIOVar("/meanSolution", "meanDens", 0);
    ioData.registerIOVar("/meanSolution", "mean-u", 1);
    ioData.registerIOVar("/meanSolution", "mean-v", 2);
    if (nvel == 3) {
      ioData.registerIOVar("/meanSolution", "mean-w", 3);
      ioData.registerIOVar("/meanSolution", "mean-E", 4);
    } else {
      ioData.registerIOVar("/meanSolution", "mean-p", dim + 1);
    }
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      // Only for NS_PASSIVE.
      if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

      // int inputSpeciesIndex = mixture->getInputIndexOf(sp);
      std::string speciesName = config.speciesNames[sp];
      ioData.registerIOVar("/meanSolution", "mean-Y" + speciesName, sp + nvel + 2);
    }

    // rms
    ioData.registerIOFamily("RMS velocity fluctuation", "/rmsData",
                            average->GetVariField(std::string("primitive_state")), false, config.GetRestartMean());
    if (nvel == 3) {
      ioData.registerIOVar("/rmsData", "uu", 0);
      ioData.registerIOVar("/rmsData", "vv", 1);
      ioData.registerIOVar("/rmsData", "ww", 2);
      ioData.registerIOVar("/rmsData", "uv", 3);
      ioData.registerIOVar("/rmsData", "uw", 4);
      ioData.registerIOVar("/rmsData", "vw", 5);
    } else if (nvel == 2) {
      ioData.registerIOVar("/rmsData", "uu", 0);
      ioData.registerIOVar("/rmsData", "vv", 1);
      ioData.registerIOVar("/rmsData", "uv", 2);
    } else {
      // only nvel = 2 or 3 supported
      assert(false);
    }
  }

  ioData.initializeSerial(rank0_, config.isRestartSerialized("either"), serial_mesh, locToGlobElem, &partitioning_);
  projectInitialSolution();

  // Boundary attributes in present partition
  Array<int> local_attr;
  getAttributesInPartition(local_attr);

  double *pTime;
  pTime = &time;

  bcIntegrator = NULL;
  if (local_attr.Size() > 0) {
    bcIntegrator = new BCintegrator(rank0_, groupsMPI, mesh, vfes, intRules, rsolver, dt, pTime, mixture, d_mixture,
                                    d_fluxClass, Up, gradUp, gpu_precomputed_data_.boundary_face_data, dim,
                                    num_equation, max_char_speed, config, local_attr, maxIntPoints, maxDofs, distance_);
  }

  // A->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  A = new DGNonLinearForm(rsolver, d_fluxClass, vfes, gradUpfes, gradUp, bcIntegrator, intRules, dim, num_equation,
                          mixture, gpu_precomputed_data_, maxIntPoints, maxDofs);
  if (local_attr.Size() > 0) A->AddBdrFaceIntegrator(bcIntegrator);

  {
    bool useLinearIntegration = false;
    //    if( basisType==1 && intRuleType==1 ) useLinearIntegration = true;

    faceIntegrator = new FaceIntegrator(intRules, rsolver, d_fluxClass, vfes, useLinearIntegration, dim, num_equation,
                                        gradUp, gradUpfes, max_char_speed, config.isAxisymmetric(), distance_);
  }
  A->AddInteriorFaceIntegrator(faceIntegrator);

  Aflux = new MixedBilinearForm(dfes, fes);
  domainIntegrator =
      new DomainIntegrator(d_fluxClass, intRules, intRuleType, dim, num_equation, config.isAxisymmetric());
  Aflux->AddDomainIntegrator(domainIntegrator);
  Aflux->Assemble();
  Aflux->Finalize();

  switch (config.GetTimeIntegratorType()) {
    case 1:
      timeIntegrator = new ForwardEulerSolver;
      break;
    case 2:
      timeIntegrator = new RK2Solver(1.0);
      break;
    case 3:
      timeIntegrator = new RK3SSPSolver;
      break;
    case 4:
      timeIntegrator = new RK4Solver;
      break;
    case 6:
      timeIntegrator = new RK6Solver;
      break;
    default:
      cout << "Unknown ODE solver type: " << config.GetTimeIntegratorType() << '\n';
  }

  gradUp_A = new GradNonLinearForm(gradUpfes, intRules, dim, num_equation);
  gradUp_A->AddInteriorFaceIntegrator(new GradFaceIntegrator(intRules, dim, num_equation));
  gradUp_A->AddBdrFaceIntegrator(new GradFaceIntegrator(intRules, dim, num_equation, bcIntegrator, config.useBCinGrad));

  rhsOperator =
      new RHSoperator(iter, dim, num_equation, order, eqSystem, max_char_speed, intRules, intRuleType, d_fluxClass,
                      mixture, d_mixture, chemistry_, transportPtr, radiation_, vfes, fes, gpu_precomputed_data_,
                      maxIntPoints, maxDofs, A, Aflux, mesh, spaceVaryViscMult, U, Up, gradUp, gradUpfes, gradUp_A,
                      bcIntegrator, config, plasma_conductivity_, joule_heating_, distance_);

  CFL = config.GetCFLNumber();
  rhsOperator->SetTime(time);
  timeIntegrator->Init(*rhsOperator);

  // Determine the minimum element size.
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < mesh->GetNE(); i++) {
      local_hmin = min(mesh->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
  }

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < mesh->GetNE(); i++) {
      local_hmax = max(mesh->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
  }

  // Determine domain bounding box size
  Vector bb_min;
  Vector bb_max;
  mesh->GetBoundingBox(bb_min, bb_max);

  xmin = bb_min[0];
  ymin = bb_min[1];
  (dim == 3) ? zmin = bb_min[2] : zmin = 0.0;

  xmax = bb_max[0];
  ymax = bb_max[1];
  (dim == 3) ? zmax = bb_max[2] : zmax = 0.0;

  // estimate initial dt
  Up->ExchangeFaceNbrData();
  gradUp->ExchangeFaceNbrData();

  if (config.GetRestartCycle() == 0) initialTimeStep();
  if (rank0_) cout << "Maximum element size: " << hmax << "m" << endl;
  if (rank0_) cout << "Minimum element size: " << hmin << "m" << endl;
  if (rank0_) cout << "Domain minimum: (" << xmin << ", " << ymin << ", " << zmin << ")" << endl;
  if (rank0_) cout << "Domain maximum: (" << xmax << ", " << ymax << ", " << zmax << ")" << endl;
  if (rank0_) cout << "Initial time-step: " << dt << "s" << endl;

  // t_final = MaxIters*dt;

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
}

void M2ulPhyS::initIndirectionArrays() {
  //-----------------------------------------------------------------
  // Element data
  //-----------------------------------------------------------------
  elementIndexingData &elem_data = gpu_precomputed_data_.element_indexing_data;

  elem_data.dof_offset.SetSize(vfes->GetNE());
  elem_data.dof_offset = -1;  // invalid
  auto h_dof_offset = elem_data.dof_offset.HostWrite();

  elem_data.dof_number.SetSize(vfes->GetNE());
  elem_data.dof_number = -1;  // invalid
  auto h_dof_number = elem_data.dof_number.HostWrite();

  std::vector<int> tempNodes;
  tempNodes.clear();

  for (int i = 0; i < vfes->GetNE(); i++) {
    const int dof = vfes->GetFE(i)->GetDof();

    // get the nodes IDs
    h_dof_offset[i] = tempNodes.size();
    h_dof_number[i] = dof;

    Array<int> dofs;
    vfes->GetElementVDofs(i, dofs);
    for (int n = 0; n < dof; n++) tempNodes.push_back(dofs[n]);
  }

  elem_data.dofs_list.SetSize(tempNodes.size());
  elem_data.dofs_list = 0;
  auto h_dofs_list = elem_data.dofs_list.HostWrite();
  for (int i = 0; i < elem_data.dofs_list.Size(); i++) {
    h_dofs_list[i] = tempNodes[i];
  }

  // count number of each type of element
  std::vector<int> tempNumElems;
  tempNumElems.clear();
  int dof1 = h_dof_number[0];
  int typeElems = 0;
  for (int el = 0; el < elem_data.dof_number.Size(); el++) {
    int dofi = h_dof_number[el];
    if (dofi == dof1) {
      typeElems++;
    } else {
      tempNumElems.push_back(typeElems);
      typeElems = 1;
      dof1 = dofi;
    }
  }
  tempNumElems.push_back(typeElems);
  elem_data.num_elems_of_type.SetSize(tempNumElems.size());
  elem_data.num_elems_of_type = 0;
  auto h_num_elems_of_type = elem_data.num_elems_of_type.HostWrite();
  for (int i = 0; i < (int)tempNumElems.size(); i++) h_num_elems_of_type[i] = tempNumElems[i];

  //-----------------------------------------------------------------
  // Interior faces
  //-----------------------------------------------------------------
  interiorFaceIntegrationData &face_data = gpu_precomputed_data_.interior_face_data;

  face_data.element_to_faces.SetSize(7 * vfes->GetNE());
  face_data.element_to_faces = 0;
  auto h_element_to_faces = face_data.element_to_faces.HostWrite();

  std::vector<double> shapes2, shapes1;
  shapes1.clear();
  shapes2.clear();

  face_data.el1.SetSize(mesh->GetNumFaces());
  face_data.el1 = 0;
  auto h_face_el1 = face_data.el1.HostWrite();

  face_data.el2.SetSize(mesh->GetNumFaces());
  face_data.el2 = 0;
  auto h_face_el2 = face_data.el2.HostWrite();

  face_data.num_quad.SetSize(mesh->GetNumFaces());
  face_data.num_quad = 0;
  auto h_face_num_quad = face_data.num_quad.HostWrite();

  face_data.el1_shape.UseDevice(true);
  face_data.el1_shape.SetSize(maxDofs * maxIntPoints * mesh->GetNumFaces());

  face_data.el2_shape.UseDevice(true);
  face_data.el2_shape.SetSize(maxDofs * maxIntPoints * mesh->GetNumFaces());

  face_data.quad_weight.UseDevice(true);
  face_data.quad_weight.SetSize(maxIntPoints * mesh->GetNumFaces());

  face_data.normal.UseDevice(true);
  face_data.normal.SetSize(dim * maxIntPoints * mesh->GetNumFaces());

  face_data.xyz.UseDevice(true);
  face_data.xyz.SetSize(dim * maxIntPoints * mesh->GetNumFaces());

  face_data.delta_el1.UseDevice(true);
  face_data.delta_el1.SetSize(mesh->GetNumFaces());

  face_data.delta_el2.UseDevice(true);
  face_data.delta_el2.SetSize(mesh->GetNumFaces());

  face_data.dist1.UseDevice(true);
  face_data.dist1.SetSize(maxIntPoints * mesh->GetNumFaces());

  face_data.dist2.UseDevice(true);
  face_data.dist2.SetSize(maxIntPoints * mesh->GetNumFaces());

  auto hshape1 = face_data.el1_shape.HostWrite();
  auto hshape2 = face_data.el2_shape.HostWrite();
  auto hweight = face_data.quad_weight.HostWrite();
  auto hnormal = face_data.normal.HostWrite();
  auto h_xyz = face_data.xyz.HostWrite();
  auto h_delta_el1 = face_data.delta_el1.HostWrite();
  auto h_delta_el2 = face_data.delta_el2.HostWrite();
  auto h_dist1 = face_data.dist1.HostWrite();
  auto h_dist2 = face_data.dist2.HostWrite();

  Vector xyz(dim);

  for (int face = 0; face < mesh->GetNumFaces(); face++) {
    FaceElementTransformations *tr;
    tr = mesh->GetInteriorFaceTransformations(face);
    if (tr != NULL) {
      Array<int> vdofs;
      Array<int> vdofs2;
      fes->GetElementVDofs(tr->Elem1No, vdofs);
      fes->GetElementVDofs(tr->Elem2No, vdofs2);

      {
        int nf = h_element_to_faces[7 * tr->Elem1No];
        if (nf < 0) nf = 0;
        h_element_to_faces[7 * tr->Elem1No + nf + 1] = face;
        nf++;
        h_element_to_faces[7 * tr->Elem1No] = nf;

        nf = h_element_to_faces[7 * tr->Elem2No];
        if (nf < 0) nf = 0;
        h_element_to_faces[7 * tr->Elem2No + nf + 1] = face;
        nf++;
        h_element_to_faces[7 * tr->Elem2No] = nf;
      }

      const FiniteElement *fe1 = fes->GetFE(tr->Elem1No);
      const FiniteElement *fe2 = fes->GetFE(tr->Elem2No);

      const int dof1 = fe1->GetDof();
      const int dof2 = fe2->GetDof();

      int intorder;
      if (tr->Elem2No >= 0) {
        intorder = (min(tr->Elem1->OrderW(), tr->Elem2->OrderW()) + 2 * max(fe1->GetOrder(), fe2->GetOrder()));
      } else {
        intorder = tr->Elem1->OrderW() + 2 * fe1->GetOrder();
      }
      if (fe1->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      const IntegrationRule *ir = &intRules->Get(tr->GetGeometryType(), intorder);

      h_face_el1[face] = tr->Elem1No;
      h_face_el2[face] = tr->Elem2No;
      h_face_num_quad[face] = ir->GetNPoints();

      h_delta_el1[face] = mesh->GetElementSize(tr->Elem1No, 1) / fe1->GetOrder();

      const int no2 = tr->Elem2->ElementNo;
      const int NE = vfes->GetNE();
      if (no2 >= NE) {
        // On shared face where element 2 is by different rank.  In this
        // case, mesh->GetElementSize fails.  Instead, compute spacing
        // directly using the ElementTransformation object.  The code
        // below is from the variant of Mesh::GetElementSize that takes an
        // ElementTransformation as input, rather than an element index.
        // We should simply call that function, but it is not public.
        ElementTransformation *T = tr->Elem2;
        DenseMatrix J(dim, dim);

        Geometry::Type geom = T->GetGeometryType();
        T->SetIntPoint(&Geometries.GetCenter(geom));
        Geometries.JacToPerfJac(geom, T->Jacobian(), J);

        // (dim-1) singular value is h_min, consistent with type=1 in GetElementSize
        h_delta_el2[face] = J.CalcSingularvalue(dim - 1) / fe2->GetOrder();
      } else {
        h_delta_el2[face] = mesh->GetElementSize(tr->Elem2No, 1) / fe2->GetOrder();
      }

      Vector shape1i, shape2i;
      shape1i.UseDevice(false);
      shape2i.UseDevice(false);
      shape1i.SetSize(dof1);
      shape2i.SetSize(dof2);

      Vector dist1, dist2;
      if (distance_ != NULL) {
        const ParFiniteElementSpace *dist_fes = distance_->ParFESpace();

        Array<int> dist_dofs1;
        // dist_fes->GetElementVDofs(tr->Elem1->ElementNo, dist_dofs1);
        dist_fes->GetElementVDofs(tr->Elem1No, dist_dofs1);
        dist1.SetSize(dist_dofs1.Size());
        distance_->GetSubVector(dist_dofs1, dist1);

        Array<int> dist_dofs2;
        if (no2 >= NE) {
          int Elem2NbrNo = no2 - NE;
          dist_fes->GetFaceNbrElementVDofs(Elem2NbrNo, dist_dofs2);
          dist2.SetSize(dist_dofs2.Size());
          distance_->FaceNbrData().GetSubVector(dist_dofs2, dist2);
        } else {
          dist_fes->GetElementVDofs(no2, dist_dofs2);
          dist2.SetSize(dist_dofs2.Size());
          distance_->GetSubVector(dist_dofs2, dist2);
        }
      }

      for (int k = 0; k < ir->GetNPoints(); k++) {
        const IntegrationPoint &ip = ir->IntPoint(k);
        tr->SetAllIntPoints(&ip);
        // shape functions
        fe1->CalcShape(tr->GetElement1IntPoint(), shape1i);
        fe2->CalcShape(tr->GetElement2IntPoint(), shape2i);
        for (int j = 0; j < dof1; j++) {
          hshape1[face * maxDofs * maxIntPoints + k * maxDofs + j] = shape1i[j];
        }
        for (int j = 0; j < dof2; j++) {
          hshape2[face * maxDofs * maxIntPoints + k * maxDofs + j] = shape2i[j];
        }

        hweight[face * maxIntPoints + k] = ip.weight;

        // Position in physical space
        tr->Transform(ip, xyz);

        // normals (multiplied by determinant of jacobian
        Vector nor;
        nor.UseDevice(false);
        nor.SetSize(dim);
        CalcOrtho(tr->Jacobian(), nor);
        for (int d = 0; d < dim; d++) {
          hnormal[face * dim * maxIntPoints + k * dim + d] = nor[d];
          h_xyz[face * dim * maxIntPoints + k * dim + d] = xyz[d];
        }

        // evaluate distance
        double d1 = 0;
        double d2 = 0;
        if (distance_ != NULL) {
          d1 = dist1 * shape1i;
          d2 = dist2 * shape2i;
        }

        h_dist1[face * maxIntPoints + k] = d1;
        h_dist2[face * maxIntPoints + k] = d2;
      }
    }
  }

  //-----------------------------------------------------------------
  // Boundary faces
  //-----------------------------------------------------------------
  boundaryFaceIntegrationData &bdry_face_data = gpu_precomputed_data_.boundary_face_data;

  // This is supposed to be number of boundary faces, and for
  // non-periodic cases it is.  But, for periodic meshes, it includes
  // periodic boundary faces---i.e., not only "true" boundary faces.

  // But, just switching to mesh->GetNFbyType(FaceType::Boundary) is
  // problematic, since the face index expected by, for example, tr =
  // mesh->GetBdrFaceTransformations(f) ranges over mesh->GetNBE().
  // Here we set up another array (abf_to_rbf) that maps from the
  // "actual" boundary faces to the index within the array of boundary
  // faces that includes the periodic faces.

  // See #199 for more info.
  const int NumBCelems = fes->GetNBE();

  // NB: *Must* call this here, as otherwise some faces are
  // erroneously included as boundary faces and asserts below may
  // fail
  mesh->ExchangeFaceNbrNodes();
  mesh->ExchangeFaceNbrData();

  if (NumBCelems > 0) {
    bdry_face_data.shape.UseDevice(true);
    bdry_face_data.shape.SetSize(NumBCelems * maxIntPoints * maxDofs);
    bdry_face_data.shape = 0.;
    auto hshapesBC = bdry_face_data.shape.HostWrite();

    bdry_face_data.normal.UseDevice(true);
    bdry_face_data.normal.SetSize(NumBCelems * maxIntPoints * dim);
    bdry_face_data.normal = 0.;
    auto h_face_normal = bdry_face_data.normal.HostWrite();

    bdry_face_data.xyz.UseDevice(true);
    bdry_face_data.xyz.SetSize(NumBCelems * maxIntPoints * dim);
    bdry_face_data.xyz = 0.;
    auto h_face_xyz = bdry_face_data.xyz.HostWrite();

    bdry_face_data.quad_weight.UseDevice(true);
    bdry_face_data.quad_weight.SetSize(NumBCelems * maxIntPoints);
    bdry_face_data.quad_weight = 0.;
    auto h_face_quad_weight = bdry_face_data.quad_weight.HostWrite();

    bdry_face_data.el.SetSize(NumBCelems);
    bdry_face_data.el = -1;
    auto h_face_el = bdry_face_data.el.HostWrite();

    bdry_face_data.num_quad.SetSize(NumBCelems);
    bdry_face_data.num_quad = 0;
    auto h_face_num_quad = bdry_face_data.num_quad.HostWrite();

    bdry_face_data.delta_el1.UseDevice(true);
    bdry_face_data.delta_el1.SetSize(NumBCelems);
    bdry_face_data.delta_el1 = 0.;
    auto h_bdry_delta_el1 = bdry_face_data.delta_el1.HostWrite();

    bdry_face_data.dist.UseDevice(true);
    bdry_face_data.dist.SetSize(NumBCelems * maxIntPoints);
    bdry_face_data.dist = 0.;
    auto h_bdry_dist = bdry_face_data.dist.HostWrite();

    const FiniteElement *fe;
    FaceElementTransformations *tr;
    // Mesh *mesh = fes->GetMesh();

    std::vector<int> uniqueElems;
    uniqueElems.clear();

    std::vector<int> rbf_to_abf;
    rbf_to_abf.clear();

    int numRealBCFaces = 0;

    for (int f = 0; f < NumBCelems; f++) {
      tr = mesh->GetBdrFaceTransformations(f);
      if (tr != NULL) {
        rbf_to_abf.push_back(f);

        numRealBCFaces += 1;
        fe = fes->GetFE(tr->Elem1No);

        const int elDof = fe->GetDof();
        int intorder;
        if (tr->Elem2No >= 0) {
          intorder = (min(tr->Elem1->OrderW(), tr->Elem2->OrderW()) + 2 * max(fe->GetOrder(), fe->GetOrder()));
        } else {
          intorder = tr->Elem1->OrderW() + 2 * fe->GetOrder();
        }
        if (fe->Space() == FunctionSpace::Pk) {
          intorder++;
        }
        const IntegrationRule *ir = &intRules->Get(tr->GetGeometryType(), intorder);

        h_face_el[f] = tr->Elem1No;
        h_face_num_quad[f] = ir->GetNPoints();
        h_bdry_delta_el1[f] = mesh->GetElementSize(tr->Elem1No, 1) / fe->GetOrder();

        bool inList = false;
        for (size_t n = 0; n < uniqueElems.size(); n++) {
          if (uniqueElems[n] == tr->Elem1No) inList = true;
        }
        if (!inList) uniqueElems.push_back(tr->Elem1No);

        Vector dist;
        if (distance_ != NULL) {
          const ParFiniteElementSpace *dist_fes = distance_->ParFESpace();

          Array<int> dist_dofs1;
          // dist_fes->GetElementVDofs(tr->Elem1->ElementNo, dist_dofs1);
          dist_fes->GetElementVDofs(tr->Elem1No, dist_dofs1);
          dist.SetSize(dist_dofs1.Size());
          distance_->GetSubVector(dist_dofs1, dist);
        }

        for (int q = 0; q < ir->GetNPoints(); q++) {
          const IntegrationPoint &ip = ir->IntPoint(q);
          tr->SetAllIntPoints(&ip);
          Vector nor;
          nor.UseDevice(false);
          nor.SetSize(dim);
          CalcOrtho(tr->Jacobian(), nor);
          h_face_quad_weight[f * maxIntPoints + q] = ip.weight;

          tr->Transform(ip, xyz);

          for (int d = 0; d < dim; d++) {
            h_face_normal[f * maxIntPoints * dim + q * dim + d] = nor[d];
            h_face_xyz[f * maxIntPoints * dim + q * dim + d] = xyz[d];
          }

          Vector shape1;
          shape1.UseDevice(false);
          shape1.SetSize(elDof);
          fe->CalcShape(tr->GetElement1IntPoint(), shape1);
          for (int n = 0; n < elDof; n++) hshapesBC[n + q * maxDofs + f * maxIntPoints * maxDofs] = shape1(n);

          // evaluate distance
          double d1 = 0;
          if (distance_ != NULL) {
            d1 = dist * shape1;
          }

          h_bdry_dist[f * maxIntPoints + q] = d1;
        }
      }
    }

    assert((int)rbf_to_abf.size() == mesh->GetNFbyType(FaceType::Boundary));
    assert((int)rbf_to_abf.size() == numRealBCFaces);

    bdry_face_data.rbf_to_abf.SetSize(rbf_to_abf.size());
    auto h_rbf_to_abf = bdry_face_data.rbf_to_abf.HostWrite();

    for (size_t i = 0; i < rbf_to_abf.size(); i++) {
      h_rbf_to_abf[i] = rbf_to_abf[i];
    }

    bdry_face_data.elements_to_faces.SetSize(7 * uniqueElems.size());
    bdry_face_data.elements_to_faces = -1;
    auto h_elements_to_faces = bdry_face_data.elements_to_faces.HostWrite();
    for (size_t i = 0; i < uniqueElems.size(); i++) {
      const int eli = uniqueElems[i];
      for (int f = 0; f < bdry_face_data.el.Size(); f++) {
        if (eli == h_face_el[f]) {
          // 0 + 7 * i = element number
          h_elements_to_faces[0 + 7 * i] = h_face_el[f];

          // 1 + 7 * i = number of boundary faces on this elem
          int numFace = h_elements_to_faces[1 + 7 * i];
          if (numFace == -1) numFace = 0;
          numFace++;

          // 1 + (j+1) + 7 * i = jth boundary face on ith unique boundary element
          h_elements_to_faces[1 + numFace + 7 * i] = f;

          // update number of faces
          h_elements_to_faces[1 + 7 * i] = numFace;
        }
      }
    }
  }

  //-----------------------------------------------------------------
  // Shared faces (i.e., interior faces at boundary of decomposition,
  // such that element1 and element2 live on different mpi ranks)
  //-----------------------------------------------------------------
  sharedFaceIntegrationData &shared_face_data = gpu_precomputed_data_.shared_face_data;

  mesh->ExchangeFaceNbrNodes();
  mesh->ExchangeFaceNbrData();
  const int Nshared = mesh->GetNSharedFaces();

  vfes->ExchangeFaceNbrData();
  gradUpfes->ExchangeFaceNbrData();

  if (distance_ != NULL) {
    distance_->ParFESpace()->ExchangeFaceNbrData();
    distance_->ExchangeFaceNbrData();
  }

  if (Nshared > 0) {
    shared_face_data.elem2_dofs.SetSize(Nshared * num_equation * maxDofs);
    shared_face_data.elem2_grad_dofs.SetSize(Nshared * num_equation * maxDofs * dim);

    shared_face_data.elem2_dofs = 0;
    shared_face_data.elem2_grad_dofs = 0;

    auto h_elem2_dofs = shared_face_data.elem2_dofs.HostReadWrite();
    auto h_elem2_grad_dofs = shared_face_data.elem2_grad_dofs.HostReadWrite();

    shared_face_data.el1_shape.UseDevice(true);
    shared_face_data.el2_shape.UseDevice(true);
    shared_face_data.quad_weight.UseDevice(true);
    shared_face_data.normal.UseDevice(true);
    shared_face_data.dist1.UseDevice(true);
    shared_face_data.dist2.UseDevice(true);

    shared_face_data.el1_shape.SetSize(Nshared * maxIntPoints * maxDofs);
    shared_face_data.el2_shape.SetSize(Nshared * maxIntPoints * maxDofs);
    shared_face_data.quad_weight.SetSize(Nshared * maxIntPoints);
    shared_face_data.normal.SetSize(Nshared * maxIntPoints * dim);
    shared_face_data.el1.SetSize(Nshared);
    shared_face_data.num_quad.SetSize(Nshared);
    shared_face_data.num_dof2.SetSize(Nshared);
    shared_face_data.dist1.SetSize(Nshared * maxIntPoints);
    shared_face_data.dist2.SetSize(Nshared * maxIntPoints);

    shared_face_data.el1_shape = 0.;
    shared_face_data.el2_shape = 0.;
    shared_face_data.quad_weight = 0.;
    shared_face_data.normal = 0.;
    shared_face_data.el1 = 0;
    shared_face_data.num_quad = 0;
    shared_face_data.num_dof2 = 0;
    shared_face_data.dist1 = 0.;
    shared_face_data.dist2 = 0.;

    auto h_shape1 = shared_face_data.el1_shape.HostWrite();
    auto h_shape2 = shared_face_data.el2_shape.HostWrite();
    auto h_face_quad_weight = shared_face_data.quad_weight.HostWrite();
    auto h_face_normal = shared_face_data.normal.HostWrite();
    auto h_face_el1 = shared_face_data.el1.HostWrite();
    auto h_face_num_quad = shared_face_data.num_quad.HostWrite();
    auto h_face_num_dof2 = shared_face_data.num_dof2.HostWrite();
    auto h_dist1 = shared_face_data.dist1.HostWrite();
    auto h_dist2 = shared_face_data.dist2.HostWrite();

    shared_face_data.xyz.UseDevice(true);
    shared_face_data.xyz.SetSize(Nshared * maxIntPoints * dim);
    shared_face_data.xyz = 0.;
    auto h_face_xyz = shared_face_data.xyz.HostWrite();

    shared_face_data.delta_el1.UseDevice(true);
    shared_face_data.delta_el1.SetSize(Nshared);
    shared_face_data.delta_el1 = 0.;
    auto h_shrd_delta_el1 = shared_face_data.delta_el1.HostWrite();

    shared_face_data.delta_el2.UseDevice(true);
    shared_face_data.delta_el2.SetSize(Nshared);
    shared_face_data.delta_el2 = 0.;
    auto h_shrd_delta_el2 = shared_face_data.delta_el2.HostWrite();

    std::vector<int> unicElems;
    unicElems.clear();

    Array<int> vdofs2, vdofsGrad;
    FaceElementTransformations *tr;
    for (int i = 0; i < Nshared; i++) {
      tr = mesh->GetSharedFaceTransformations(i, true);
      int Elem2NbrNo = tr->Elem2No - mesh->GetNE();

      const FiniteElement *fe1 = vfes->GetFE(tr->Elem1No);
      const FiniteElement *fe2 = vfes->GetFaceNbrFE(Elem2NbrNo);
      const int dof1 = fe1->GetDof();
      const int dof2 = fe2->GetDof();

      h_shrd_delta_el1[i] = mesh->GetElementSize(tr->Elem1No, 1) / fe1->GetOrder();

      // On shared face, element 2 is owned by a different rank.  In
      // this case, mesh->GetElementSize fails.  Instead, compute
      // spacing directly using the ElementTransformation object.  The
      // code below is from the variant of Mesh::GetElementSize that
      // takes an ElementTransformation as input, rather than an
      // element index.  We should simply call that function, but it
      // is not public.
      ElementTransformation *T = tr->Elem2;
      DenseMatrix J(dim, dim);

      Geometry::Type geom = T->GetGeometryType();
      T->SetIntPoint(&Geometries.GetCenter(geom));
      Geometries.JacToPerfJac(geom, T->Jacobian(), J);

      // (dim-1) singular value is h_min, consistent with type=1 in GetElementSize
      h_shrd_delta_el2[i] = J.CalcSingularvalue(dim - 1) / fe2->GetOrder();

      // vfes->GetElementVDofs(tr->Elem1No, vdofs1); // get these from nodesIDs
      vfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
      gradUpfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofsGrad);

      for (int n = 0; n < dof2; n++) {
        for (int eq = 0; eq < num_equation; eq++) {
          h_elem2_dofs[n + eq * maxDofs + i * num_equation * maxDofs] = vdofs2[n + eq * dof2];
          for (int d = 0; d < dim; d++) {
            int index = n + eq * maxDofs + d * num_equation * maxDofs + i * dim * num_equation * maxDofs;
            h_elem2_grad_dofs[index] = vdofsGrad[n + eq * dof2 + d * num_equation * dof2];
          }
        }
      }

      int intorder;
      if (tr->Elem2No >= 0) {
        intorder = (min(tr->Elem1->OrderW(), tr->Elem2->OrderW()) + 2 * max(fe1->GetOrder(), fe2->GetOrder()));
      } else {
        intorder = tr->Elem1->OrderW() + 2 * fe1->GetOrder();
      }
      if (fe1->Space() == FunctionSpace::Pk) {
        intorder++;
      }
      // IntegrationRules IntRules2(0, Quadrature1D::GaussLobatto);
      const IntegrationRule *ir = &intRules->Get(tr->GetGeometryType(), intorder);

      h_face_el1[i] = tr->Elem1No;
      h_face_num_quad[i] = ir->GetNPoints();
      h_face_num_dof2[i] = dof2;

      bool inList = false;
      for (size_t n = 0; n < unicElems.size(); n++) {
        if (unicElems[n] == tr->Elem1No) inList = true;
      }
      if (!inList) unicElems.push_back(tr->Elem1No);

      Vector dist1, dist2;
      if (distance_ != NULL) {
        const ParFiniteElementSpace *dist_fes = distance_->ParFESpace();

        Array<int> dist_dofs1;
        dist_fes->GetElementVDofs(tr->Elem1->ElementNo, dist_dofs1);
        dist1.SetSize(dist_dofs1.Size());
        distance_->GetSubVector(dist_dofs1, dist1);

        Array<int> dist_dofs2;
        dist_fes->GetFaceNbrElementVDofs(Elem2NbrNo, dist_dofs2);
        dist2.SetSize(dist_dofs2.Size());
        distance_->FaceNbrData().GetSubVector(dist_dofs2, dist2);
      }

      Vector shape1, shape2, nor;
      shape1.UseDevice(false);
      shape2.UseDevice(false);
      nor.UseDevice(false);
      shape1.SetSize(dof1);
      shape2.SetSize(dof2);
      nor.SetSize(dim);

      for (int q = 0; q < ir->GetNPoints(); q++) {
        const IntegrationPoint &ip = ir->IntPoint(q);
        tr->SetAllIntPoints(&ip);

        fe1->CalcShape(tr->GetElement1IntPoint(), shape1);
        fe2->CalcShape(tr->GetElement2IntPoint(), shape2);
        CalcOrtho(tr->Jacobian(), nor);

        for (int n = 0; n < dof1; n++) {
          h_shape1[i * maxIntPoints * maxDofs + q * maxDofs + n] = shape1[n];
        }
        h_face_quad_weight[i * maxIntPoints + q] = ip.weight;

        tr->Transform(ip, xyz);

        for (int d = 0; d < dim; d++) {
          h_face_normal[i * maxIntPoints * dim + q * dim + d] = nor[d];
          h_face_xyz[i * maxIntPoints * dim + q * dim + d] = xyz[d];
        }
        for (int n = 0; n < dof2; n++) {
          h_shape2[i * maxIntPoints * maxDofs + q * maxDofs + n] = shape2[n];
        }

        // evaluate distance
        double d1 = 0;
        double d2 = 0;
        if (distance_ != NULL) {
          d1 = dist1 * shape1;
          d2 = dist2 * shape2;
        }

        h_dist1[i * maxIntPoints + q] = d1;
        h_dist2[i * maxIntPoints + q] = d2;
      }
    }

    shared_face_data.shared_elements_to_shared_faces.SetSize(7 * unicElems.size());
    shared_face_data.shared_elements_to_shared_faces = -1;
    auto h_shared_elements_to_shared_faces = shared_face_data.shared_elements_to_shared_faces.HostWrite();
    for (size_t el = 0; el < unicElems.size(); el++) {
      const int eli = unicElems[el];
      for (int f = 0; f < shared_face_data.el1.Size(); f++) {
        if (eli == h_face_el1[f]) {
          h_shared_elements_to_shared_faces[0 + 7 * el] = h_face_el1[f];
          int numFace = h_shared_elements_to_shared_faces[1 + 7 * el];
          if (numFace == -1) numFace = 0;
          numFace++;
          h_shared_elements_to_shared_faces[1 + numFace + 7 * el] = f;
          h_shared_elements_to_shared_faces[1 + 7 * el] = numFace;
        }
      }
    }
  }
}

void M2ulPhyS::initIndirectionBC() {
  boundaryFaceIntegrationData &bdry_face_data = gpu_precomputed_data_.boundary_face_data;

  // This is supposed to be number of boundary faces, and for
  // non-periodic cases it is.  See #199 for more info.
  const int NumBCelems = fes->GetNBE();

  if (NumBCelems > 0) {
    bdry_face_data.bc_category.SetSize(NumBCelems);
    bdry_face_data.bc_category = NUM_BC_CATEGORIES;
    auto h_bc_category = bdry_face_data.bc_category.HostWrite();

    bdry_face_data.use_bc_in_grad.SetSize(NumBCelems);
    bdry_face_data.use_bc_in_grad = false;
    auto h_use_bc_in_grad = bdry_face_data.use_bc_in_grad.HostWrite();

    bdry_face_data.wall_bc_temperature.SetSize(NumBCelems);
    bdry_face_data.wall_bc_temperature = false;
    bdry_face_data.wall_bc_temperature.UseDevice(true);
    auto h_wall_bc_temperature = bdry_face_data.wall_bc_temperature.HostWrite();

    FaceElementTransformations *tr;
    Mesh *mesh = fes->GetMesh();

    for (int f = 0; f < NumBCelems; f++) {
      tr = mesh->GetBdrFaceTransformations(f);
      if (tr != NULL && bcIntegrator != NULL) {
        int attr = tr->Attribute;
        h_bc_category[f] = bcIntegrator->getAttributeCategory(attr);
        if (config.useBCinGrad && h_bc_category[f] == WALL) {
          std::unordered_map<int, BoundaryCondition *>::const_iterator wbci = bcIntegrator->wallBCmap.find(attr);
          if (wbci != bcIntegrator->wallBCmap.end()) {
            WallBC *wbc = dynamic_cast<WallBC *>(wbci->second);
            WallType wt = wbc->getType();
            if (wt == VISC_ISOTH) {
              fflush(stdout);
              h_use_bc_in_grad[f] = true;
              h_wall_bc_temperature[f] = wbc->getWallTemp();
            }
          }
        }
      }
    }
  }
}

M2ulPhyS::~M2ulPhyS() {
  if (rank0_) histFile.close();

  delete gradUp;

  delete gradUp_A;

  delete U;
  delete Up;

  delete average;

  delete rhsOperator;
  // delete domainIntegrator;
  delete Aflux;  // fails to delete (follow this)

  // delete faceIntegrator;
  delete A;  // fails to delete (follow this)

  delete timeIntegrator;
  // delete inlet/outlet integrators

#if defined(_CUDA_) || defined(_HIP_)
  gpu::freeDeviceRiemann<<<1, 1>>>(rsolver);
  tpsGpuFree(rsolver);

  gpu::freeDeviceFluxes<<<1, 1>>>(d_fluxClass);
  tpsGpuFree(d_fluxClass);

  gpu::freeDeviceChemistry<<<1, 1>>>(chemistry_);
  tpsGpuFree(chemistry_);

  gpu::freeDeviceRadiation<<<1, 1>>>(radiation_);
  tpsGpuFree(radiation_);

  gpu::freeDeviceTransport<<<1, 1>>>(transportPtr);
  tpsGpuFree(transportPtr);

  gpu::freeDeviceMixture<<<1, 1>>>(d_mixture);
  tpsGpuFree(d_mixture);
#else
  delete rsolver;
  delete chemistry_;
  delete radiation_;
  delete transportPtr;
#endif

  delete fluxClass;
  delete mixture;

  delete gradUpfes;
  delete vfes;
  delete nvelfes;
  delete dfes;
  delete fes;
  delete fec;
  delete intRules;

  // Since own data is true, this will delete the mesh and fields
  delete paraviewColl;
  // delete mesh;

  delete groupsMPI;

  delete[] locToGlobElem;

  // TODO(Uvilla) delete serial_mesh causes memory corruption
  // delete serial_mesh;

  //   if ( mpi.Root() && (config.RestartSerial() != "no") )
  //   {
  //     delete serial_mesh;
  //     delete serial_fes;
  //     delete serial_soln;
  //
  //     if( config.GetMeanSampleInterval()
  //     {
  //       delete serial_fesRMS;
  //       delete serial_RMS;
  //     }
  //   }

  // The fields registered in paraviewColl are views of u_block, up_block.
  // Therefore, this variables need to be deallocated at the end
  delete u_block;
  delete up_block;
  delete offsets;

#ifdef HAVE_GRVY
  if (rank0_) grvy_timer_summarize();
#endif
}

void M2ulPhyS::getAttributesInPartition(Array<int> &local_attr) {
  local_attr.DeleteAll();
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    bool attrInArray = false;
    for (int i = 0; i < local_attr.Size(); i++) {
      if (local_attr[i] == attr) attrInArray = true;
    }
    if (!attrInArray) local_attr.Append(attr);
  }
}

void M2ulPhyS::initSolutionAndVisualizationVectors() {
  visualizationVariables_.clear();
  visualizationNames_.clear();

  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) {
    (*offsets)[k] = k * vfes->GetNDofs();
  }

  u_block = new BlockVector(*offsets);
  up_block = new BlockVector(*offsets);

  // gradUp.SetSize(num_equation*dim*vfes->GetNDofs());
  gradUp = new ParGridFunction(gradUpfes);

  // TODO(Umberto) consider using new ParGridFunction(vfes, *u_block);
  U = new ParGridFunction(vfes, *u_block);
  Up = new ParGridFunction(vfes, *up_block);

  dens = new ParGridFunction(fes, *Up);
  vel = new ParGridFunction(dfes, *Up, fes->GetNDofs());

  if (config.isAxisymmetric()) {
    vtheta = new ParGridFunction(fes, *Up, 3 * fes->GetNDofs());
  } else {
    vtheta = NULL;
  }

  temperature = new ParGridFunction(fes, *Up, (1 + nvel) * fes->GetNDofs());

  electron_temp_field = NULL;
  if (config.twoTemperature) {
    electron_temp_field = new ParGridFunction(fes, *Up, (num_equation - 1) * fes->GetNDofs());
  }

  // this variable is purely for visualization
  press = new ParGridFunction(fes);

  passiveScalar = NULL;
  if (eqSystem == NS_PASSIVE) {
    passiveScalar = new ParGridFunction(fes, *Up, (num_equation - 1) * fes->GetNDofs());
  } else {
    // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
    // will need to add full list of species.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      std::string speciesName = config.speciesNames[sp];
      visualizationVariables_.push_back(new ParGridFunction(fes, *U, (sp + nvel + 2) * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("partial_density_" + speciesName));
    }
  }

  // add visualization variables if tps is run on post-process visualization mode.
  // TODO(kevin): maybe enable users to specify what to visualize.
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

  // define solution parameters for i/o
  ioData.registerIOFamily("Solution state variables", "/solution", U, true, true, fec);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "rho-u", 1);
  ioData.registerIOVar("/solution", "rho-v", 2);
  if (nvel == 3) {
    ioData.registerIOVar("/solution", "rho-w", 3);
    ioData.registerIOVar("/solution", "rho-E", 4);
  } else {
    ioData.registerIOVar("/solution", "rho-E", 3);
  }
  // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
  // will need to add full list of species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // Only for NS_PASSIVE.
    if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

    std::string speciesName = config.speciesNames[sp];
    if (config.io_opts_.enable_restart_from_lte_) {
      ioData.registerIOVar("/solution", "rho-Y_" + speciesName, sp + nvel + 2, false);
    } else {
      ioData.registerIOVar("/solution", "rho-Y_" + speciesName, sp + nvel + 2);
    }
  }
  if (config.twoTemperature) {
    if (config.io_opts_.enable_restart_from_lte_) {
      ioData.registerIOVar("/solution", "rhoE_e", num_equation - 1, false);
    } else {
      ioData.registerIOVar("/solution", "rhoE_e", num_equation - 1);
    }
  }

  // compute factor to multiply viscosity when this option is active
  spaceVaryViscMult = NULL;
  ParGridFunction coordsDof(dfes);
  mesh->GetNodes(coordsDof);
  if (config.linViscData.isEnabled) {
    spaceVaryViscMult = new ParGridFunction(fes);
    double *viscMult = spaceVaryViscMult->HostWrite();
    double wgt = 0.;
    for (int n = 0; n < fes->GetNDofs(); n++) {
      auto hcoords = coordsDof.HostRead();  // get coords

      double coords[3];
      for (int d = 0; d < dim; d++) {
        coords[d] = hcoords[n + d * vfes->GetNDofs()];
      }

      fluxClass->viscSpongePlanar(coords, wgt);
      viscMult[n] = wgt;
    }
  }

  paraviewColl->SetCycle(0);
  paraviewColl->SetTime(0.);

  paraviewColl->RegisterField("dens", dens);
  paraviewColl->RegisterField("vel", vel);
  if (config.isAxisymmetric()) {
    paraviewColl->RegisterField("vtheta", vtheta);
  }
  paraviewColl->RegisterField("temp", temperature);
  paraviewColl->RegisterField("press", press);
  if (eqSystem == NS_PASSIVE) {
    paraviewColl->RegisterField("passiveScalar", passiveScalar);
  }

  if (config.twoTemperature) {
    paraviewColl->RegisterField("Te", electron_temp_field);
  }

  for (size_t var = 0; var < visualizationVariables_.size(); var++) {
    paraviewColl->RegisterField(visualizationNames_[var], visualizationVariables_[var]);
  }

  if (spaceVaryViscMult != NULL) paraviewColl->RegisterField("viscMult", spaceVaryViscMult);

  if (distance_ != NULL) paraviewColl->RegisterField("distance", distance_);
  if (plasma_conductivity_ != NULL) paraviewColl->RegisterField("sigma", plasma_conductivity_);
  if (joule_heating_ != NULL) paraviewColl->RegisterField("Sjoule", joule_heating_);

  paraviewColl->SetOwnData(true);
  // paraviewColl->Save();
}

void M2ulPhyS::projectInitialSolution() {
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
  if (rank0_) std::cout << "restart: " << config.GetRestartCycle() << std::endl;

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

    if (config.io_opts_.enable_restart_from_lte_) {
      initilizeSpeciesFromLTE();
      Check_Undershoot();
    }

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->UseRestartMode(true);
  }

  initGradUp();

  // Exchange before computing primitives
  U->ParFESpace()->ExchangeFaceNbrData();
  U->ExchangeFaceNbrData();

  updatePrimitives();

  // update pressure grid function
  mixture->UpdatePressureGridFunction(press, Up);

  // update plasma electrical conductivity
  if (tpsP->isFlowEMCoupled()) {
    ParGridFunction *coordsDof = new ParGridFunction(dfes);
    mesh->GetNodes(*coordsDof);
    mixture->SetConstantPlasmaConductivity(plasma_conductivity_, Up, coordsDof);
    delete coordsDof;
  }

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    // Only save IC from fresh start.  On restart, will save viz at
    // next requested iter.  This avoids possibility of trying to
    // overwrite existing paraview data for the current iteration.
    if (!(tpsP->isVisualizationMode())) paraviewColl->Save();
  }

  // if restarting from LTE, write paraview and restart h5 immediately
  if (config.io_opts_.enable_restart_from_lte_ && !tpsP->isVisualizationMode()) {
    if (rank0_) std::cout << "Writing non-equilibrium restart files!" << std::endl;
    paraviewColl->Save();
    restart_files_hdf5("write");
  }
}

void M2ulPhyS::solveBegin() {
#ifdef HAVE_MASA
  // instantiate function for exact solution
  // NOTE: this has been taken care of at M2ulPhyS::initMasaHandler.
  // initMMSCoefficients();

  if (config.use_mms_) {
    checkSolutionError(time);
  }
#endif
}

void M2ulPhyS::solveStep() {
  timeIntegrator->Step(*U, time, dt);

  Check_NAN();
  if (mixture->GetWorkingFluid() == WorkingFluid::USER_DEFINED) Check_Undershoot();

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank0_) cout << "skata : " << " Check_Undershoot 2" << endl;

  if (!config.isTimeStepConstant()) {
    double dt_local = CFL * hmin / max_char_speed / static_cast<double>(dim);
    MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
  }

  iter++;

  const int vis_steps = config.GetNumItersOutput();
  if (iter % vis_steps == 0) {
#ifdef HAVE_MASA
    if (config.use_mms_) {
      if (config.mmsSaveDetails_) {
        rhsOperator->Mult(*U, *masaRhs_);
        projectExactSolution(time, masaU_);
      }
      checkSolutionError(time);
    } else {
      if (rank0_) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
    }
#else
    if (rank0_) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
#endif

    if (iter != MaxIters) {
      // auto hUp = Up->HostRead();
      Up->HostRead();
      mixture->UpdatePressureGridFunction(press, Up);

      restart_files_hdf5("write");

      paraviewColl->SetCycle(iter);
      paraviewColl->SetTime(time);
      paraviewColl->Save();
      // auto dUp = Up->ReadWrite();  // sets memory to GPU
      Up->ReadWrite();  // sets memory to GPU

      average->writeViz(iter, time, config.isMeanHistEnabled());
    }

    // make a separate routine! plane interp and dump here
    if (config.planeDump.isEnabled == true) {
#ifdef HAVE_GSLIB
      // Get the source field for the interpolation
      // TODO(shaering): improve option to select u, <u>, or <u'u'> for multiple at once
      ParGridFunction *u_gf;
      if (config.planeDump.conserved == true) {
        u_gf = GetSolutionGF();
      } else if (config.planeDump.primitive == true) {
        u_gf = getPrimitiveGF();
      } else if (config.planeDump.mean == true) {
        u_gf = average->GetMeanField(std::string("primitive_state"));
      } else if (config.planeDump.reynolds == true) {
        u_gf = average->GetVariField(std::string("primitive_state"));
      } else {
        grvy_printf(GRVY_ERROR, "\nSpecified interpolation type not supported.\n");
        exit(ERROR);
      }

      // Bounding box coordinates
      Vector bb0(3);
      bb0[0] = xmin;
      bb0[1] = ymin;
      bb0[2] = zmin;
      Vector bb1(3);
      bb1[0] = xmax;
      bb1[1] = ymax;
      bb1[2] = zmax;

      // The PlaneInterpolator class handles setting up the
      // interpolation points and performing the interpolation (using
      // FindPointsGSLIB)
      PlaneInterpolator interp(config.planeDump.point, config.planeDump.normal, bb0, bb1, config.planeDump.samples);
      interp.initializeFinder(mesh);
      interp.setInterpolationPoints();
      interp.interpolate(u_gf);

      string oname;
      oname = "./planeData/plane" + std::to_string(iter) + ".txt";
      interp.writeAscii(oname, rank0_);
#else
      grvy_printf(GRVY_ERROR, "\nPlane dump capability requires gslib support.\n");
      exit(ERROR);
#endif
    }  // plane dump
  }    // step check

  average->addSample(iter, d_mixture);
}

void M2ulPhyS::solveEnd() {
  if (iter == MaxIters) {
    // auto hUp = Up->HostRead();
    Up->HostRead();
    mixture->UpdatePressureGridFunction(press, Up);

    // write_restart_files();
    restart_files_hdf5("write");

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();

    average->writeViz(iter, time, config.isMeanHistEnabled());

#ifndef HAVE_MASA
    // // If HAVE_MASA is defined, this is handled above
    // void (*initialConditionFunction)(const Vector &, Vector &);
    // initialConditionFunction = &(this->InitialConditionEulerVortex);

    // VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    // const double error = U->ComputeLpError(2, u0);
    // if (rank0_) cout << "Solution error: " << error << endl;
#else
    if (config.use_mms_) {
      checkSolutionError(time, true);
    } else {
      if (rank0_) cout << "Final timestep iteration = " << MaxIters << endl;
    }
#endif

    if (rank0_) cout << "Final timestep iteration = " << MaxIters << endl;
  }
}

void M2ulPhyS::solve() {
  this->solveBegin();
  double tlast = grvy_timer_elapsed_global();
#ifdef HAVE_SLURM
  bool readyForRestart = false;
#endif

  // Integrate in time.
  while (iter < MaxIters) {
    grvy_timer_begin(__func__);

    // periodically report on time/iteratino
    if ((iter % config.timingFreq) == 0) {
      if (rank0_) {
        double timePerIter = (grvy_timer_elapsed_global() - tlast) / config.timingFreq;
        grvy_printf(ginfo, "Iteration = %i: wall clock time/iter = %.3f (secs)\n", iter, timePerIter);
        tlast = grvy_timer_elapsed_global();
      }
    }

    // Do the step
    this->solveStep();

#ifdef HAVE_SLURM
    // check if near end of a run and ready to submit restart
    if ((iter % config.rm_checkFreq() == 0) && (iter != MaxIters)) {
      readyForRestart = Check_JobResubmit();
      if (readyForRestart) {
        MaxIters = iter;
        SetStatus(JOB_RESTART);
        break;
      }
    }
#endif

    // periodically check for DIE file which requests to terminate early
    if (Check_ExitEarly(iter)) {
      MaxIters = iter;
      SetStatus(EARLY_EXIT);
      break;
    }
    grvy_timer_end(__func__);
  }  // <-- end main timestep iteration loop

  this->solveEnd();

  return;
}

// Initial conditions for debug/test case
void M2ulPhyS::InitialConditionEulerVortex(const Vector &x, Vector &y) {
  MFEM_ASSERT(x.Size() == 2, "");
  int equations = 4;
  if (x.Size() == 3) equations = 5;

  int problem = 1;
  DryAir *eqState = new DryAir();
  const double gamma = eqState->GetSpecificHeatRatio();
  const double Rg = eqState->GetGasConstant();

  double radius = 0, Minf = 0, beta = 0;
  if (problem == 1) {
    // "Fast vortex"
    radius = 0.2;
    Minf = 0.5;
    beta = 1. / 5.;

    radius = 0.5;
    Minf = 0.1;
  } else if (problem == 2) {
    // "Slow vortex"
    radius = 0.2;
    Minf = 0.05;
    beta = 1. / 50.;
  } else {
    mfem_error(
        "Cannot recognize problem."
        "Options are: 1 - fast vortex, 2 - slow vortex");
  }

  int numVortices = 3;
  Vector xc(numVortices), yc(numVortices);
  yc = 0.;
  for (int i = 0; i < numVortices; i++) {
    xc[i] = 2. * M_PI / static_cast<double>(numVortices + 1);
    xc[i] += static_cast<double>(i) * 2. * M_PI / static_cast<double>(numVortices);
  }

  const double Tt = 300.;
  const double Pt = 102200;

  const double funcGamma = 1. + 0.5 * (gamma - 1.) * Minf * Minf;

  const double temp_inf = Tt / funcGamma;
  const double pres_inf = Pt * pow(funcGamma, gamma / (gamma - 1.));
  const double vel_inf = Minf * sqrt(gamma * Rg * temp_inf);
  const double den_inf = pres_inf / (Rg * temp_inf);

  double r2rad = 0.0;

  const double shrinv1 = 1.0 / (gamma - 1.);

  double velX = 0.;
  double velY = 0.;
  double temp = 0.;
  for (int i = 0; i < numVortices; i++) {
    r2rad = (x(0) - xc[i]) * (x(0) - xc[i]);
    r2rad += (x(1) - yc[i]) * (x(1) - yc[i]);
    r2rad /= radius * radius;
    velX -= beta * (x(1) - yc[i]) / radius * exp(-0.5 * r2rad);
    velY += beta * (x(0) - xc[i]) / radius * exp(-0.5 * r2rad);
    temp += exp(-r2rad);
  }

  velX = vel_inf * (1 - velX);
  velY = vel_inf * velY;
  const double vel2 = velX * velX + velY * velY;

  const double specific_heat = Rg * gamma * shrinv1;
  temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / specific_heat * temp;

  const double den = den_inf * pow(temp / temp_inf, shrinv1);
  const double pres = den * Rg * temp;
  const double energy = shrinv1 * pres / den + 0.5 * vel2;

  y(0) = den;
  y(1) = den * velX;
  y(2) = den * velY;
  if (x.Size() == 3) y(3) = 0.;
  y(equations - 1) = den * energy;

  delete eqState;
}

// Initial conditions for debug/test case
void M2ulPhyS::testInitialCondition(const Vector &x, Vector &y) {
  DryAir *eqState = new DryAir();

  // Nice units
  const double vel_inf = 1.;
  const double den_inf = 1.3;
  const double Minf = 0.5;

  const double gamma = eqState->GetSpecificHeatRatio();
  // const double Rgas = eqState->GetGasConstant();

  const double pres_inf = (den_inf / gamma) * (vel_inf / Minf) * (vel_inf / Minf);

  y(0) = den_inf + 0.5 * (x(0) + 3) + 0.25 * (x(1) + 3);
  y(1) = y(0);
  y(2) = 0;
  y(3) = (pres_inf + x(0) + 0.2 * x(1)) / (gamma - 1.) + 0.5 * y(1) * y(1) / y(0);

  delete eqState;
}

// // NOTE: Use only for DRY_AIR.
// void M2ulPhyS::dryAirUniformInitialConditions() {
void M2ulPhyS::uniformInitialConditions() {
  double *data = U->HostWrite();
  double *dataUp = Up->HostWrite();
  double *dataGradUp = gradUp->HostWrite();

  int dof = vfes->GetNDofs();
  double *inputRhoRhoVp = config.GetConstantInitialCondition();

  // build initial state
  Vector initState(num_equation);
  initState = 0.;

  initState(0) = inputRhoRhoVp[0];
  initState(1) = inputRhoRhoVp[1];
  initState(2) = inputRhoRhoVp[2];
  if (nvel == 3) initState(3) = inputRhoRhoVp[3];

  if (mixture->GetWorkingFluid() == WorkingFluid::USER_DEFINED) {
    const int numSpecies = mixture->GetNumSpecies();
    const int numActiveSpecies = mixture->GetNumActiveSpecies();
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      // int inputIndex = mixture->getInputIndexOf(sp);
      initState(2 + nvel + sp) = inputRhoRhoVp[0] * config.initialMassFractions(sp);
    }

    // electron energy
    if (mixture->IsTwoTemperature()) {
      double ne = 0.;
      if (mixture->IsAmbipolar()) {
        for (int sp = 0; sp < numActiveSpecies; sp++)
          ne += mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) * initState(2 + nvel + sp) /
                mixture->GetGasParams(sp, GasParams::SPECIES_MW);
      } else {
        ne = initState(2 + nvel + numSpecies - 2) / mixture->GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
      }

      initState(num_equation - 1) = config.initialElectronTemperature * ne * mixture->getMolarCV(numSpecies - 2);
    }
  }

  // initial energy
  mixture->modifyEnergyForPressure(initState, initState, inputRhoRhoVp[4]);

  // TODO(marc): try to make this mixture independent
  //   DryAir *eqState = new DryAir(dim, num_equation);

  //   const double gamma = eqState->GetSpecificHeatRatio();
  //   const double rhoE =
  //       inputRhoRhoVp[4] / (gamma - 1.) + 0.5 *
  //                                             (inputRhoRhoVp[1] * inputRhoRhoVp[1] + inputRhoRhoVp[2] *
  //                                             inputRhoRhoVp[2] +
  //                                              inputRhoRhoVp[3] * inputRhoRhoVp[3]) /
  //                                             inputRhoRhoVp[0];

  Vector state;
  state.UseDevice(false);
  state.SetSize(num_equation);
  Vector Upi;
  Upi.UseDevice(false);
  Upi.SetSize(num_equation);

  mixture->GetPrimitivesFromConservatives(initState, Upi);

  for (int i = 0; i < dof; i++) {
    //     }
    for (int eq = 0; eq < num_equation; eq++) data[i + eq * dof] = initState(eq);
    for (int eq = 0; eq < num_equation; eq++) dataUp[i + eq * dof] = Upi[eq];

    for (int d = 0; d < dim; d++) {
      for (int eq = 0; eq < num_equation; eq++) {
        dataGradUp[i + eq * dof + d * num_equation * dof] = 0.;
      }
    }
  }

  //   delete eqState;
}

void M2ulPhyS::initGradUp() {
  double *dataGradUp = gradUp->HostWrite();
  int dof = vfes->GetNDofs();

  for (int i = 0; i < dof; i++) {
    for (int d = 0; d < dim; d++) {
      for (int eq = 0; eq < num_equation; eq++) {
        dataGradUp[i + eq * dof + d * num_equation * dof] = 0.;
      }
    }
  }
}

// NOTE(Mal): This is a method to be used when we restart from LTE simulation.
// It initilzes species mass densities based on LTE assumptions.
void M2ulPhyS::initilizeSpeciesFromLTE() {
  double *dataU = U->GetData();
  double *dataUp = Up->GetData();
  // double *dataGradUp = gradUp->HostWrite();

  int dof = vfes->GetNDofs();

  assert(mixture->GetWorkingFluid() == WorkingFluid::USER_DEFINED);

  std::string thermo_file;
  tpsP->getRequiredInput("flow/lte/thermo_table", thermo_file);
  config.lteMixtureInput.thermo_file_name = thermo_file;
  std::string trans_file;
  tpsP->getRequiredInput("flow/lte/transport_table", trans_file);
  config.lteMixtureInput.trans_file_name = trans_file;
  std::string e_rev_file;
  tpsP->getRequiredInput("flow/lte/e_rev_table", e_rev_file);
  config.lteMixtureInput.e_rev_file_name = e_rev_file;

  TableInterpolator2D *energy_table;
  TableInterpolator2D *R_table;
  TableInterpolator2D *c_table;
  TableInterpolator2D *T_table;

#if defined(HAVE_GSL) && !defined(_CUDA_) && !defined(_HIP_)

  energy_table = new GslTableInterpolator2D(config.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                            1,                                          /* density column */
                                            3 /* energy column */);
  R_table = new GslTableInterpolator2D(config.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                       1,                                          /* density column */
                                       6 /* mixture gas constant column */);

  c_table = new GslTableInterpolator2D(config.lteMixtureInput.thermo_file_name, 0, /* temperature column */
                                       1,                                          /* density column */
                                       8 /* speed of sound column */);

  T_table = new GslTableInterpolator2D(config.lteMixtureInput.e_rev_file_name, /* (energy,density) -> temp data */
                                       0,                                      /* energy column */
                                       1,                                      /* density column */
                                       2,                                      /* temperature column */
                                       3);                                     /* number of columns */
#else
  energy_table = NULL;
  R_table = NULL;
  c_table = NULL;
  T_table = NULL;
  mfem_error("Restart from LTE mixture requires GSL support.");
#endif

  Vector state(num_equation);
  Vector prim(num_equation);

  for (int i = 0; i < dof; i++) {
    for (int eq = 0; eq < num_equation; eq++) {
      state[eq] = dataU[i + eq * dof];
      prim[eq] = dataUp[i + eq * dof];
    }

    // Calculate species mass densities bases based on LTE at node level.
    mixture->GetSpeciesFromLTE(state.HostReadWrite(), prim.HostReadWrite(), energy_table, R_table, c_table, T_table);

    // Return new state at each node
    for (int eq = 0; eq < num_equation; eq++) {
      dataU[i + eq * dof] = state[eq];
      dataUp[i + eq * dof] = prim[eq];
    }
  }

  delete energy_table;
  delete R_table;
  delete c_table;
  delete T_table;
}

void M2ulPhyS::Check_NAN() {
  int local_print = 0;
  int dof = vfes->GetNDofs();

#ifdef _GPU_
  { local_print = M2ulPhyS::Check_NaN_GPU(U, dof * num_equation, loc_print); }
  if (local_print > 0) {
    cout << "Found a NaN!" << endl;
  }
#else
  const double *dataU = U->HostRead();

  // bool thereIsNan = false;

  for (int i = 0; i < dof; i++) {
    Vector iState(num_equation);
    for (int eq = 0; eq < num_equation; eq++) {
      if (std::isnan(dataU[i + eq * dof])) {
        // thereIsNan = true;
        cout << "NaN at node: " << i << " , Eq: " << eq << " partition: " << rank_ << endl;
        local_print++;
        // MPI_Abort(MPI_COMM_WORLD,1);
      }
      iState[eq] = dataU[i + eq * dof];
    }
    // if (~mixture->StateIsPhysical(iState)) {
    //   cout << "Unphysical at node: " << i << " partition: " << mpi.WorldRank() << endl;
    //   local_print++;
    // }
  }
#endif
  int print;
  MPI_Allreduce(&local_print, &print, 1, MPI_INT, MPI_SUM, groupsMPI->getTPSCommWorld());
  if (print > 0) {
    // auto hUp = Up->HostRead();  // get GPU data
    Up->HostRead();  // get GPU data
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
    MPI_Barrier(groupsMPI->getTPSCommWorld());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

int M2ulPhyS::Check_NaN_GPU(ParGridFunction *U, int lengthU, Array<int> &loc_print) {
  const double *dataU = U->Read();
  auto d_temp = loc_print.ReadWrite();

  MFEM_FORALL(n, lengthU, {
    double val = dataU[n];
    if (val != val) d_temp[0] += 1;
  });

  auto htemp = loc_print.HostRead();
  return htemp[0];
}

// Here, we force negative (unphysical) values of species to be zero.
// Negative values can occur due to numerical oscillations in the solution of the
// transport equations of scalars. The effect of oscilations can be mitigated by employing
// a TVD scheme. Van Leer's scheme is a common practice.
void M2ulPhyS::Check_Undershoot() {
  int dof = vfes->GetNDofs();
#ifdef _GPU_
  int nv = nvel;
  int nsp = numActiveSpecies;
  double *dataU = U->ReadWrite();

  MFEM_FORALL(i, dof, {
    for (int sp = 0; sp < nsp; sp++) {
      int eq = nv + 2 + sp;
      dataU[i + eq * dof] = max(dataU[i + eq * dof], 0.0);
    }
  });
#else
  double *dataU = U->HostReadWrite();
  for (int i = 0; i < dof; i++) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      int eq = nvel + 2 + sp;
      dataU[i + eq * dof] = max(dataU[i + eq * dof], 0.0);
    }
  }
#endif
}

void M2ulPhyS::initialTimeStep() {
  auto dataU = U->HostReadWrite();
  int dof = vfes->GetNDofs();

  for (int n = 0; n < dof; n++) {
    Vector state(num_equation);
    for (int eq = 0; eq < num_equation; eq++) state[eq] = dataU[n + eq * dof];
    double iC = mixture->ComputeMaxCharSpeed(state);
    if (iC > max_char_speed) max_char_speed = iC;
  }

  double partition_C = max_char_speed;
  MPI_Allreduce(&partition_C, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());

  dt = CFL * hmin / max_char_speed / static_cast<double>(dim);

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }
}

void M2ulPhyS::parseSolverOptions() { return; }

// koomie TODO: add parsing of all previously possible runtime inputs.
void M2ulPhyS::parseSolverOptions2() {
  tpsP->getRequiredInput("flow/mesh", config.meshFile);

  // flow/numerics
  parseFlowOptions();

  // time integration controls
  parseTimeIntegrationOptions();

  // statistics
  parseStatOptions();

  // I/O settings
  parseIOSettings();

  // RMS job management
  parseRMSJobOptions();

  // heat source
  parseHeatSrcOptions();

  // viscosity multiplier function
  parseViscosityOptions();

  // plane data
  parsePlaneDump();

  // MMS (check before IC b/c if MMS, IC not required)
  parseMMSOptions();

  // initial conditions
  if (!config.use_mms_) {
    parseICOptions();
  }

  // add passive scalar forcings
  parsePassiveScalarOptions();

  // fluid presets
  parseFluidPreset();

  parseSystemType();

  // plasma conditions.
  config.gasModel = NUM_GASMODEL;
  config.transportModel = NUM_TRANSPORTMODEL;
  config.chemistryModel_ = NUM_CHEMISTRYMODEL;
  if (config.workFluid == USER_DEFINED) {
    parsePlasmaModels();
  } else {
    // parse options for other plasma presets.
    config.const_plasma_conductivity_ = 50.0;
  }

  // species list.
  parseSpeciesInputs();

  if (config.workFluid == USER_DEFINED) {  // Transport model
    parseTransportInputs();
  }

  // Reaction list
  parseReactionInputs();

  // changed the order of parsing in order to use species list.
  // boundary conditions. The number of boundaries of each supported type if
  // parsed first. Then, a section corresponding to each defined boundary is parsed afterwards
  parseBCInputs();

  // changed the order of parsing in order to use species list.
  // sponge zone
  parseSpongeZoneInputs();

  // Pack up input parameters for device objects.
  packUpGasMixtureInput();

  // Radiation model
  parseRadiationInputs();

  // periodicity
  parsePeriodicInputs();

  // post-process visualization inputs
  parsePostProcessVisualizationInputs();

  return;
}

void M2ulPhyS::parseFlowOptions() {
  std::map<std::string, int> sgsModel;
  sgsModel["none"] = 0;
  sgsModel["smagorinsky"] = 1;
  sgsModel["sigma"] = 2;
  tpsP->getInput("flow/order", config.solOrder, 4);
  tpsP->getInput("flow/integrationRule", config.integrationRule, 1);
  tpsP->getInput("flow/basisType", config.basisType, 1);
  tpsP->getInput("flow/maxIters", config.numIters, 10);
  tpsP->getInput("flow/outputFreq", config.itersOut, 50);
  tpsP->getInput("flow/timingFreq", config.timingFreq, 100);
  tpsP->getInput("flow/useRoe", config.useRoe, false);
  tpsP->getInput("flow/refLength", config.refLength, 1.0);
  tpsP->getInput("flow/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP->getInput("flow/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  tpsP->getInput("flow/axisymmetric", config.axisymmetric_, false);
  tpsP->getInput("flow/enablePressureForcing", config.isForcing, false);
  if (config.isForcing) {
    for (int d = 0; d < 3; d++) tpsP->getRequiredVecElem("flow/pressureGrad", config.gradPress[d], d);
  }
  tpsP->getInput("flow/refinement_levels", config.ref_levels, 0);
  tpsP->getInput("flow/computeDistance", config.compute_distance, false);
  tpsP->getInput("flow/readDistance", config.read_distance, false);

  std::string type;
  tpsP->getInput("flow/sgsModel", type, std::string("none"));
  config.sgsModelType = sgsModel[type];

  double sgs_const = 0.;
  if (config.sgsModelType == 1) {
    sgs_const = 0.12;
  } else if (config.sgsModelType == 2) {
    sgs_const = 0.135;
  }
  tpsP->getInput("flow/sgsModelConstant", config.sgs_model_const, sgs_const);

  tpsP->getInput("flow/useMixingLength", config.use_mixing_length, false);
  if (config.use_mixing_length) {
    // Default value for mixing length is zero, which turns model off
    tpsP->getInput("flow/mixing-length/max-mixing-length", config.mix_length_trans_input_.max_mixing_length_, 0.0);
    tpsP->getInput("flow/mixing-length/Pr_ratio", config.mix_length_trans_input_.Prt_, 1.0);
    tpsP->getInput("flow/mixing-length/Let", config.mix_length_trans_input_.Let_, 1.0);
    tpsP->getInput("flow/mixing-length/bulk-multiplier", config.mix_length_trans_input_.bulk_multiplier_, 0.0);
  }

  assert(config.solOrder > 0);
  assert(config.numIters >= 0);
  assert(config.itersOut > 0);
  assert(config.refLength > 0);

  // Sutherland inputs default to dry air values
  tpsP->getInput("flow/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP->getInput("flow/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP->getInput("flow/SutherlandPr", config.sutherland_.Pr, 0.71);
}

void M2ulPhyS::parseTimeIntegrationOptions() {
  std::map<std::string, int> integrators;
  integrators["forwardEuler"] = 1;
  integrators["rk2"] = 2;
  integrators["rk3"] = 3;
  integrators["rk4"] = 4;
  integrators["rk6"] = 6;
  std::string type;
  tpsP->getInput("time/cfl", config.cflNum, 0.12);
  tpsP->getInput("time/integrator", type, std::string("rk4"));
  tpsP->getInput("time/enableConstantTimestep", config.constantTimeStep, false);
  tpsP->getInput("time/dt_fixed", config.dt_fixed, -1.);
  if (integrators.count(type) == 1) {
    config.timeIntegratorType = integrators[type];
  } else {
    grvy_printf(GRVY_ERROR, "Unknown time integrator > %s\n", type.c_str());
    exit(ERROR);
  }
}

void M2ulPhyS::parseStatOptions() { config.avg_opts_.read(tpsP); }

void M2ulPhyS::parseIOSettings() { config.io_opts_.read(tpsP); }

void M2ulPhyS::parseRMSJobOptions() {
  tpsP->getInput("jobManagement/enableAutoRestart", config.rm_enableMonitor_, false);
  tpsP->getInput("jobManagement/timeThreshold", config.rm_threshold_, 15 * 60);  // 15 minutes
  tpsP->getInput("jobManagement/checkFreq", config.rm_checkFrequency_, 500);     // 500 iterations
}

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

void M2ulPhyS::parseViscosityOptions() {
  tpsP->getInput("viscosityMultiplierFunction/isEnabled", config.linViscData.isEnabled, false);
  if (config.linViscData.isEnabled) {
    auto normal = config.linViscData.normal.HostWrite();
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[0], 0);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[1], 1);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/normal", normal[2], 2);

    auto point0 = config.linViscData.point0.HostWrite();
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/point", point0[0], 0);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/point", point0[1], 1);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/point", point0[2], 2);

    // auto pointInit = config.linViscData.pointInit.HostWrite();
    // tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[0], 0);
    // tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[1], 1);
    // tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[2], 2);

    tpsP->getRequiredInput("viscosityMultiplierFunction/width", config.linViscData.width);
    tpsP->getRequiredInput("viscosityMultiplierFunction/viscosityRatio", config.linViscData.viscRatio);
  }
}

void M2ulPhyS::parsePlaneDump() {
  tpsP->getInput("planeDump/isEnabled", config.planeDump.isEnabled, false);

  if (config.planeDump.isEnabled) {
    auto normal = config.planeDump.normal.HostWrite();
    tpsP->getRequiredVecElem("planeDump/norm", normal[0], 0);
    tpsP->getRequiredVecElem("planeDump/norm", normal[1], 1);
    tpsP->getRequiredVecElem("planeDump/norm", normal[2], 2);

    auto point = config.planeDump.point.HostWrite();
    tpsP->getRequiredVecElem("planeDump/point", point[0], 0);
    tpsP->getRequiredVecElem("planeDump/point", point[1], 1);
    tpsP->getRequiredVecElem("planeDump/point", point[2], 2);

    tpsP->getRequiredInput("planeDump/samples", config.planeDump.samples);

    // data to interpolate, only one selection is supported now
    tpsP->getInput("planeDump/conserved", config.planeDump.conserved, false);
    tpsP->getInput("planeDump/primitive", config.planeDump.primitive, false);
    tpsP->getInput("planeDump/mean", config.planeDump.mean, false);
    tpsP->getInput("planeDump/reynolds", config.planeDump.reynolds, false);
  }
}

void M2ulPhyS::parseMMSOptions() {
  tpsP->getInput("mms/isEnabled", config.use_mms_, false);

  if (config.use_mms_) {
    tpsP->getRequiredInput("mms/name", config.mms_name_);
    tpsP->getInput("mms/compare_rhs", config.mmsCompareRhs_, false);
    tpsP->getInput("mms/save_details", config.mmsSaveDetails_, false);
  }
}

void M2ulPhyS::parseICOptions() {
  tpsP->getRequiredInput("initialConditions/rho", config.initRhoRhoVp[0]);
  tpsP->getRequiredInput("initialConditions/rhoU", config.initRhoRhoVp[1]);
  tpsP->getRequiredInput("initialConditions/rhoV", config.initRhoRhoVp[2]);
  tpsP->getRequiredInput("initialConditions/rhoW", config.initRhoRhoVp[3]);
  tpsP->getRequiredInput("initialConditions/pressure", config.initRhoRhoVp[4]);
}

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

void M2ulPhyS::parseFluidPreset() {
  std::string fluidTypeStr;
  tpsP->getInput("flow/fluid", fluidTypeStr, std::string("dry_air"));
  if (fluidTypeStr == "dry_air") {
    config.workFluid = DRY_AIR;
    tpsP->getInput("flow/specific_heat_ratio", config.dryAirInput.specific_heat_ratio, 1.4);
    tpsP->getInput("flow/gas_constant", config.dryAirInput.gas_constant, 287.058);
  } else if (fluidTypeStr == "user_defined") {
    config.workFluid = USER_DEFINED;
  } else if (fluidTypeStr == "lte_table") {
    config.workFluid = LTE_FLUID;
    std::string thermo_file;
    tpsP->getRequiredInput("flow/lte/thermo_table", thermo_file);
    config.lteMixtureInput.thermo_file_name = thermo_file;
    std::string trans_file;
    tpsP->getRequiredInput("flow/lte/transport_table", trans_file);
    config.lteMixtureInput.trans_file_name = trans_file;
    std::string e_rev_file;
    tpsP->getRequiredInput("flow/lte/e_rev_table", e_rev_file);
    config.lteMixtureInput.e_rev_file_name = e_rev_file;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown fluid preset supplied at runtime -> %s", fluidTypeStr.c_str());
    exit(ERROR);
  }

  if (rank0_ && (config.workFluid != USER_DEFINED))
    cout << "Fluid is set to the preset '" << fluidTypeStr << "'. Input options in [plasma_models] will not be used."
         << endl;
}

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
  tpsP->getInput("plasma_models/gas_model", gasModelStr, std::string("perfect_mixture"));
  if (gasModelStr == "perfect_mixture") {
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
  printf("config.transportModel = %s\n", transportModelStr.c_str());
  fflush(stdout);
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

  /*
    NOTE: for now, we force the user to set the background species as the last,
    and the electron species as the second-to-last.
  */
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
      double levelDegeneracy;
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

      // tpsP->getRequiredInput((basepath + "/level_degeneracy").c_str(), levelDegeneracy);
      tpsP->getInput((basepath + "/level_degeneracy").c_str(), levelDegeneracy, 1.0);
      inputGasParams(i - 1, GasParams::SPECIES_DEGENERACY) = levelDegeneracy;

      tpsP->getRequiredInput((basepath + "/initialMassFraction").c_str(), inputInitialMassFraction(i - 1));

      if (config.gasModel == PERFECT_MIXTURE) {
        tpsP->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(), inputCV(i - 1));
        // NOTE: For perfect gas, CP will be automatically set from CV.
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
      // TODO(Mal): We need to find a better way to define indices of the species. Probably through input file.
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
        // TODO(trevilo): I am commenting this out (previously part of
        // if above) b/c it causes some tests to seg fault.  Need to
        // discuss with Malamas what he is wanting to achieve here and
        // figure out how to make it work.
        //
        // else if (inputSpeciesNames[sp] == "Ar.+1" ||  inputSpeciesNames[sp] == "Ar+1" ) {
        //   targetIdx = config.numSpecies - 3;
        // } else if (inputSpeciesNames[sp] == "Ar.+2" ||  inputSpeciesNames[sp] == "Ar+2" ) {
        //   targetIdx = config.numSpecies - 4;
        // } else if (inputSpeciesNames[sp] == "Ar2.+1" ||  inputSpeciesNames[sp] == "Ar2+1" ) {
        //   targetIdx = config.numSpecies - 5;
        // }

        config.speciesMapping[inputSpeciesNames[sp]] = targetIdx;
        config.speciesNames[targetIdx] = inputSpeciesNames[sp];
        config.mixtureToInputMap[targetIdx] = sp;
        if (rank0_) {
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
  tpsP->getInput("reactions/minimum_chemistry_temperature", config.chemistryInput.minimumTemperature, 0.0);
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
      readTableWrapper(inputPath, config.chemistryInput.reactionInputs[r - 1].tableInput);
    } else if (model == "bte") {
      config.reactionModels[r - 1] = GRIDFUNCTION_RXN;
      int index;
      tpsP->getRequiredInput((basepath + "/bte/index").c_str(), index);
      config.chemistryInput.reactionInputs[r - 1].indexInput = index;
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

      double max_e = max(reactEnergy, prodEnergy);
      const double delta = abs(reactEnergy + config.reactionEnergies[r] - prodEnergy);
      const double rel_delta = delta / max_e;
      if (rel_delta > 5e-16) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve energy.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E + %.8E = %.8E =/= %.8E\n", reactEnergy, config.reactionEnergies[r],
                    reactEnergy + config.reactionEnergies[r], prodEnergy);
        grvy_printf(GRVY_ERROR, "Difference = %.8E\n", delta);
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

    size_t rxn_param_idx = 0;

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

      if (config.reactionModels[r] == ARRHENIUS || config.reactionModels[r] == HOFFERTLIEN) {
        assert(rxn_param_idx < config.rxnModelParamsHost.size());
        config.chemistryInput.reactionInputs[r].modelParams = config.rxnModelParamsHost[rxn_param_idx].Read();
        rxn_param_idx += 1;
      }
    }
  }
}

void M2ulPhyS::parseBCInputs() {
  tpsP->getInput("boundaryConditions/useBCinGrad", config.useBCinGrad, false);

  // number of BC regions defined
  int numWalls, numInlets, numOutlets;
  tpsP->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // Wall Bcs
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

    tpsP->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP->getRequiredInput((basepath + "/type").c_str(), type);
    if (type == "viscous_isothermal") {
      tpsP->getRequiredInput((basepath + "/temperature").c_str(), temperature);
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
      tpsP->getRequiredInput((basepath + "/heavy_thermal_condition").c_str(), hvyType);
      config.wallBC[i - 1].hvyThermalCond = thmCondMap[hvyType];

      config.wallBC[i - 1].Th = -1.0;
      switch (config.wallBC[i - 1].hvyThermalCond) {
        case ISOTH: {
          double Th;
          tpsP->getRequiredInput((basepath + "/temperature").c_str(), Th);
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
        tpsP->getRequiredInput((basepath + "/electron_thermal_condition").c_str(), elecType);
        if (thmCondMap[elecType] == NONE_THMCND) {
          grvy_printf(GRVY_ERROR, "Wall%d: electron thermal condition must be specified for two-temperature plasma!\n",
                      i);
          exit(-1);
        }
        config.wallBC[i - 1].elecThermalCond = thmCondMap[elecType];
      } else {
        tpsP->getInput((basepath + "/electron_thermal_condition").c_str(), elecType, std::string("none"));
        config.wallBC[i - 1].elecThermalCond = (thmCondMap[elecType] == SHTH) ? SHTH : NONE_THMCND;
      }
      config.wallBC[i - 1].Te = -1.0;
      switch (config.wallBC[i - 1].elecThermalCond) {
        case ISOTH: {
          double Te;
          tpsP->getInput((basepath + "/electron_temperature").c_str(), Te, -1.0);
          if (Te < 0.0) {
            grvy_printf(GRVY_INFO, "Wall%d: no input for electron_temperature. Using temperature instead.\n", i);
            tpsP->getRequiredInput((basepath + "/temperature").c_str(), Te);
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

  // Inlet Bcs
  std::map<std::string, InletType> inletMapping;
  inletMapping["subsonic"] = SUB_DENS_VEL;
  inletMapping["subsonicFaceBasedX"] = SUB_DENS_VEL_FACE_X;
  inletMapping["subsonicFaceBasedY"] = SUB_DENS_VEL_FACE_Y;
  inletMapping["subsonicFaceBasedZ"] = SUB_DENS_VEL_FACE_Z;
  inletMapping["nonreflecting"] = SUB_DENS_VEL_NR;
  inletMapping["nonreflectingConstEntropy"] = SUB_VEL_CONST_ENT;

  for (int i = 1; i <= numInlets; i++) {
    int patch;
    double density;
    std::string type;
    std::string basepath("boundaryConditions/inlet" + std::to_string(i));

    tpsP->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP->getRequiredInput((basepath + "/type").c_str(), type);
    // all inlet BCs require 4 inputs (density + vel(3))
    {
      Array<double> uvw;
      tpsP->getRequiredInput((basepath + "/density").c_str(), density);
      tpsP->getRequiredVec((basepath + "/uvw").c_str(), uvw, 3);
      config.inletBC.Append(density);
      config.inletBC.Append(uvw, 3);
    }

    // NOTE: if there is a need, make face-based general by specifying tangent, leave
    // as either tan = x, y, or z for now as fixing requires changing BC interface
    // across the board
    // FaceBased requires an additional tangent definition cooresponding to the w-component in uvw
    // if (type == "subsonicFaceBased") {
    //  Array<double> tangent2;
    //  tpsP->getRequiredVec((basepath + "/w-tangent").c_str(), tangent2, 3);
    //  config.inletBC.Append(tangent2, 3);
    //}

    // For multi-component gas, require (numActiveSpecies)-more inputs.
    if ((config.workFluid != DRY_AIR) && (config.numSpecies > 1)) {
      grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
      if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

      for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index
        double Ysp;
        // read mass fraction of species as listed in the input file.
        int inputSp = config.mixtureToInputMap[sp];
        std::string speciesBasePath(basepath + "/mass_fraction/species" + std::to_string(inputSp + 1));
        tpsP->getRequiredInput(speciesBasePath.c_str(), Ysp);
        config.inletBC.Append(Ysp);
      }
    }
    std::pair<int, InletType> patchType;
    patchType.first = patch;
    patchType.second = inletMapping[type];
    config.inletPatchType.push_back(patchType);
  }

  // Outlet Bcs
  std::map<std::string, OutletType> outletMapping;
  outletMapping["subsonicPressure"] = SUB_P;
  outletMapping["nonReflectingPressure"] = SUB_P_NR;
  outletMapping["nonReflectingMassFlow"] = SUB_MF_NR;
  outletMapping["nonReflectingPointBasedMassFlow"] = SUB_MF_NR_PW;

  for (int i = 1; i <= numOutlets; i++) {
    int patch;
    double pressure, massFlow;
    std::string type;
    std::string basepath("boundaryConditions/outlet" + std::to_string(i));

    tpsP->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP->getRequiredInput((basepath + "/type").c_str(), type);

    if ((type == "subsonicPressure") || (type == "nonReflectingPressure")) {
      tpsP->getRequiredInput((basepath + "/pressure").c_str(), pressure);
      config.outletBC.Append(pressure);
    } else if ((type == "nonReflectingMassFlow") || (type == "nonReflectingPointBasedMassFlow")) {
      tpsP->getRequiredInput((basepath + "/massFlow").c_str(), massFlow);
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

void M2ulPhyS::parsePostProcessVisualizationInputs() {
  if (tpsP->isVisualizationMode()) {
    tpsP->getRequiredInput("post-process/visualization/prefix", config.postprocessInput.prefix);
    tpsP->getRequiredInput("post-process/visualization/start-iter", config.postprocessInput.startIter);
    tpsP->getRequiredInput("post-process/visualization/end-iter", config.postprocessInput.endIter);
    tpsP->getRequiredInput("post-process/visualization/frequency", config.postprocessInput.freq);
  }
}

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
      readTableWrapper(inputPath, config.radiationInput.necTableInput);
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

void M2ulPhyS::readTableWrapper(std::string inputPath, TableInput &result) {
  MPI_Comm TPSCommWorld = this->groupsMPI->getTPSCommWorld();
  std::string filename;
  tpsP->getInput((inputPath + "/x_log").c_str(), result.xLogScale, false);
  tpsP->getInput((inputPath + "/f_log").c_str(), result.fLogScale, false);
  tpsP->getInput((inputPath + "/order").c_str(), result.order, 1);
  tpsP->getRequiredInput((inputPath + "/filename").c_str(), filename);
  readTable(TPSCommWorld, filename, result.xLogScale, result.fLogScale, result.order, tableHost_, result);
}

void M2ulPhyS::parsePeriodicInputs() {
  tpsP->getInput("periodicity/enablePeriodic", config.periodic, false);
  tpsP->getInput("periodicity/xTrans", config.xTrans, 1.0e12);
  tpsP->getInput("periodicity/yTrans", config.yTrans, 1.0e12);
  tpsP->getInput("periodicity/zTrans", config.zTrans, 1.0e12);
}

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
        config.perfectMixtureInput.f = config.workFluid;
        config.perfectMixtureInput.numSpecies = config.numSpecies;
        if (config.speciesMapping.count("E")) {
          config.perfectMixtureInput.isElectronIncluded = true;
        } else {
          config.perfectMixtureInput.isElectronIncluded = false;
        }
        config.perfectMixtureInput.ambipolar = config.ambipolar;
        config.perfectMixtureInput.twoTemperature = config.twoTemperature;

        for (int sp = 0; sp < config.numSpecies; sp++) {
          for (int param = 0; param < (int)GasParams::NUM_GASPARAMS; param++) {
            config.perfectMixtureInput.gasParams[sp + param * config.numSpecies] = config.gasParams(sp, param);
          }
          config.perfectMixtureInput.molarCV[sp] = config.constantMolarCV(sp);
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

void M2ulPhyS::identifySpeciesType(Array<ArgonSpcs> &speciesType) {
  speciesType.SetSize(config.numSpecies);

  for (int sp = 0; sp < config.numSpecies; sp++) {
    speciesType[sp] = NONE_ARGSPCS;

    Vector spComp(config.numAtoms);
    config.speciesComposition.GetRow(sp, spComp);

    if (spComp(config.atomMap["Ar"]) == 1.0) {
      if (spComp(config.atomMap["E"]) == -1.0) {
        speciesType[sp] = AR1P;
      } else {
        bool argonMonomer = true;
        for (int a = 0; a < config.numAtoms; a++) {
          if (a == config.atomMap["Ar"]) continue;
          if (spComp(a) != 0.0) {
            argonMonomer = false;
            break;
          }
        }
        if (argonMonomer) {
          speciesType[sp] = AR;
        } else {
          // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
          std::string name = config.speciesNames[sp];
          grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                      name.c_str());
          exit(-1);
        }
      }
    } else if (spComp(config.atomMap["E"]) == 1.0) {
      bool electron = true;
      for (int a = 0; a < config.numAtoms; a++) {
        if (a == config.atomMap["E"]) continue;
        if (spComp(a) != 0.0) {
          electron = false;
          break;
        }
      }
      if (electron) {
        speciesType[sp] = ELECTRON;
      } else {
        // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
        std::string name = config.speciesNames[sp];
        grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                    name.c_str());
        exit(-1);
      }
    } else {
      // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
      std::string name = config.speciesNames[sp];
      grvy_printf(GRVY_ERROR, "The atom composition of species %s is not supported by ArgonMixtureTransport! \n",
                  name.c_str());
      exit(-1);
    }
  }

  // Check all species are identified.
  for (int sp = 0; sp < config.numSpecies; sp++) {
    if (speciesType[sp] == NONE_ARGSPCS) {
      // std::string name = speciesNames_[(*mixtureToInputMap_)[sp]];
      std::string name = config.speciesNames[sp];
      grvy_printf(GRVY_ERROR, "The species %s is not identified in ArgonMixtureTransport! \n", name.c_str());
      exit(-1);
    }
  }

  return;
}

void M2ulPhyS::identifyCollisionType(const Array<ArgonSpcs> &speciesType, ArgonColl *collisionIndex) {
  // collisionIndex_.resize(numSpecies);
  for (int spI = 0; spI < config.numSpecies; spI++) {
    // collisionIndex_[spI].resize(numSpecies - spI);
    for (int spJ = spI; spJ < config.numSpecies; spJ++) {
      // If not initialized, will raise an error.
      collisionIndex[spI + spJ * config.numSpecies] = NONE_ARGCOLL;

      const double pairType =
          config.gasParams(spI, GasParams::SPECIES_CHARGES) * config.gasParams(spJ, GasParams::SPECIES_CHARGES);
      if (pairType > 0.0) {  // Repulsive screened Coulomb potential
        collisionIndex[spI + spJ * config.numSpecies] = CLMB_REP;
      } else if (pairType < 0.0) {  // Attractive screened Coulomb potential
        collisionIndex[spI + spJ * config.numSpecies] = CLMB_ATT;
      } else {  // determines collision type by species pairs.
        if ((speciesType[spI] == AR) && (speciesType[spJ] == AR)) {
          collisionIndex[spI + spJ * config.numSpecies] = AR_AR;
        } else if (((speciesType[spI] == AR) && (speciesType[spJ] == AR1P)) ||
                   ((speciesType[spI] == AR1P) && (speciesType[spJ] == AR))) {
          collisionIndex[spI + spJ * config.numSpecies] = AR_AR1P;
        } else if (((speciesType[spI] == AR) && (speciesType[spJ] == ELECTRON)) ||
                   ((speciesType[spI] == ELECTRON) && (speciesType[spJ] == AR))) {
          collisionIndex[spI + spJ * config.numSpecies] = AR_E;
        } else {
          // std::string name1 = speciesNames_[(*mixtureToInputMap_)[spI]];
          // std::string name2 = speciesNames_[(*mixtureToInputMap_)[spJ]];
          std::string name1 = config.speciesNames[spI];
          std::string name2 = config.speciesNames[spJ];
          grvy_printf(GRVY_ERROR, "%s-%s is not supported in ArgonMixtureTransport! \n", name1.c_str(), name2.c_str());
          exit(-1);
        }
      }
    }
  }

  // Check all collision types are initialized.
  for (int spI = 0; spI < config.numSpecies; spI++) {
    for (int spJ = spI; spJ < config.numSpecies; spJ++) {
      if (collisionIndex[spI + spJ * config.numSpecies] == NONE_ARGCOLL) {
        // std::string name1 = speciesNames_[(*mixtureToInputMap_)[spI]];
        // std::string name2 = speciesNames_[(*mixtureToInputMap_)[spJ]];
        std::string name1 = config.speciesNames[spI];
        std::string name2 = config.speciesNames[spJ];
        grvy_printf(GRVY_ERROR, "%s-%s is not initialized in ArgonMixtureTransport! \n", name1.c_str(), name2.c_str());
        exit(-1);
      }
    }
  }

  return;
}

void M2ulPhyS::checkSolverOptions() const {
#ifdef _GPU_
  if (!config.isTimeStepConstant()) {
    if (rank0_) {
      std::cerr << "[ERROR]: GPU runs must use a constant time step: Please set DT_CONSTANT in input file."
                << std::endl;
      std::cerr << std::endl;
      exit(ERROR);
    }
  }
#endif

  // Axisymmetric solver does not yet support all options.  Check that
  // we are running a supported combination.
  if (config.isAxisymmetric()) {
    // Don't support passive scalars yet
    if (config.GetEquationSystem() == NS_PASSIVE) {
      if (rank0_) {
        std::cerr << "[ERROR]: Passive scalars not supported for axisymmetric simulations." << std::endl;
        std::cerr << std::endl;
        exit(ERROR);
      }
    }
    // Don't support Roe flux yet
    if (config.RoeRiemannSolver()) {
      if (rank0_) {
        std::cerr << "[ERROR]: Roe flux not supported for axisymmetric simulations. Please use flow/useRoe = 0."
                  << std::endl;
        std::cerr << std::endl;
        exit(ERROR);
      }
    }
    // Don't support non-reflecting BCs yet
    for (size_t i = 0; i < config.GetInletPatchType()->size(); i++) {
      std::pair<int, InletType> patchANDtype = (*config.GetInletPatchType())[i];
      if (patchANDtype.second != SUB_DENS_VEL) {
        if (rank0_) {
          std::cerr << "[ERROR]: Only SUB_DENS_VEL inlet supported for axisymmetric simulations." << std::endl;
          std::cerr << std::endl;
          exit(ERROR);
        }
      }
    }
    for (size_t i = 0; i < config.GetOutletPatchType()->size(); i++) {
      std::pair<int, OutletType> patchANDtype = (*config.GetOutletPatchType())[i];
      if (patchANDtype.second != SUB_P) {
        if (rank0_) {
          std::cerr << "[ERROR]: Only SUB_P outlet supported for axisymmetric simulations." << std::endl;
          std::cerr << std::endl;
          exit(ERROR);
        }
      }
    }
  }

#ifndef HAVE_MASA
  if (config.use_mms_) {
    if (rank0_) {
      std::cerr << "[ERROR]: Require MASA support to run manufactured solutions." << std::endl;
      std::cerr << std::endl;
      exit(ERROR);
    }
  }
#endif

  // Warn user if they requested fixed dt without setting enableConstantTimestep
  if ((config.GetFixedDT() > 0) && (!config.isTimeStepConstant())) {
    if (rank0_) {
      std::cerr << "[WARNING]: Setting dt_fixed overrides enableConstantTimestep." << std::endl;
      std::cerr << std::endl;
    }
  }

  return;
}

void M2ulPhyS::updatePrimitives() {
  const double *data = U->HostRead();
  double *dataUp = Up->HostWrite();
  int dof = vfes->GetNDofs();

  // Change this double * state = new double[num_equation] and same for Upi?
  Vector state;
  state.UseDevice(false);
  state.SetSize(num_equation);
  Vector Upi;
  Upi.UseDevice(false);
  Upi.SetSize(num_equation);

  for (int i = 0; i < dof; i++) {
    for (int eq = 0; eq < num_equation; eq++) state(eq) = data[i + eq * dof];
    mixture->GetPrimitivesFromConservatives(state, Upi);
    for (int eq = 0; eq < num_equation; eq++) dataUp[i + eq * dof] = Upi[eq];
  }
}

void M2ulPhyS::visualization() {
  double tlast = grvy_timer_elapsed_global();

#ifdef HAVE_MASA
  if (config.use_mms_) {
    checkSolutionError(time);
  }
#endif

  int fileIter = config.postprocessInput.startIter;

  // Read solution files per frequency.
  while (fileIter <= config.postprocessInput.endIter) {
    grvy_timer_begin(__func__);

    // periodically report on time/iteratino
    if (rank0_) {
      double timePerIter = (grvy_timer_elapsed_global() - tlast);
      grvy_printf(ginfo, "Iteration = %i: wall clock time/snapshot = %.3f (secs)\n", fileIter, timePerIter);
      tlast = grvy_timer_elapsed_global();
    }

    // int oldIter = iter;
    size_t digits = 8;
    int precision = digits - std::min(digits, std::to_string(fileIter).size());
    // pad leading zeros to iter.
    std::string iterStr = std::string(precision, '0').append(std::to_string(fileIter));
    std::string filename(config.postprocessInput.prefix + "-" + iterStr + ".h5");
    // iter and time will be set from the file.
    restart_files_hdf5("read", filename);
    // NOTE(kevin): file iter does not have to be the same as actual timestep.
    // assert(oldIter == iter);  // make sure iter in the solution file matches its filename.

    // use RhsOperator::updatePrimitives and updateGradients which uses both gpu and cpu.
    rhsOperator->updateGradients(*U, false);

    // update pressure grid function
    mixture->UpdatePressureGridFunction(press, Up);

    updateVisualizationVariables();

    Check_NAN();

#ifdef HAVE_MASA
    if (config.use_mms_) {
      if (config.mmsSaveDetails_) {
        rhsOperator->Mult(*U, *masaRhs_);
        projectExactSolution(time, masaU_);
      }
      checkSolutionError(time);
    } else {
      if (rank0_) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
    }
#else
    if (rank0_) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
#endif

    // set iter and time based on the file.
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();

    average->writeViz(iter, time, config.isMeanHistEnabled());

    average->addSample(iter, d_mixture);

    // periodically check for DIE file which requests to terminate early
    if (Check_ExitEarly(fileIter)) {
      fileIter = config.postprocessInput.endIter;
      SetStatus(EARLY_EXIT);
      break;
    }

    fileIter += config.postprocessInput.freq;

    grvy_timer_end(__func__);
  }  // <-- end main timestep iteration loop

  if (rank0_) cout << "Final timestep iteration = " << config.postprocessInput.endIter << endl;

  return;
}

void M2ulPhyS::updateVisualizationVariables() {
#ifdef _GPU_
  grvy_printf(GRVY_ERROR, "Post-process visualization is not implemented for gpu!");
  exit(ERROR);
#endif  // _GPU_

  // TODO(kevin): The routine here currently only supports cpu path, though it is written in a gpu-compatible way.
  // Will require some minor #ifdef additions to implement gpu path.

  double *dataU = U->GetData();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();
  const int ndofs = vfes->GetNDofs();
  const int _dim = dim;
  const int _nvel = nvel;
  const int _num_equation = num_equation;
  const int _numSpecies = numSpecies;
  const int _numReactions = config.numReactions;

  GasMixture *in_mix = mixture;
  TransportProperties *in_transport = transportPtr;
  Chemistry *in_chem = chemistry_;
  const bool isDryAir = (config.workFluid == DRY_AIR);

  const int nVisual = visualizationVariables_.size();
  const AuxiliaryVisualizationIndexes visualIdxs = visualizationIndexes_;
  double *dataVis[gpudata::MAXVISUAL];
  for (int vis = 0; vis < nVisual; vis++) dataVis[vis] = visualizationVariables_[vis]->GetData();

  for (int n = 0; n < ndofs; n++) {
    double state[gpudata::MAXEQUATIONS];
    double prim[gpudata::MAXEQUATIONS];
    double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double Efield[gpudata::MAXDIM];
    for (int eq = 0; eq < _num_equation; eq++) {
      state[eq] = dataU[n + eq * ndofs];
      prim[eq] = dataUp[n + eq * ndofs];
      for (int d = 0; d < _dim; d++)
        gradUpn[eq + d * _num_equation] = dataGradUp[n + eq * ndofs + d * _num_equation * ndofs];
    }
    // TODO(kevin): EM coupling update.
    for (int d = 0; d < _dim; d++) Efield[d] = 0.0;

    if (!isDryAir) {
      // update species primitives.
      double Xsp[gpudata::MAXSPECIES];
      double Ysp[gpudata::MAXSPECIES];
      double nsp[gpudata::MAXSPECIES];
      in_mix->computeSpeciesPrimitives(state, Xsp, Ysp, nsp);

      for (int sp = 0; sp < _numSpecies; sp++) {
        dataVis[visualIdxs.Xsp + sp][n] = Xsp[sp];
        dataVis[visualIdxs.Ysp + sp][n] = Ysp[sp];
        dataVis[visualIdxs.nsp + sp][n] = nsp[sp];
      }

      // update flux transport properties.
      double fluxTrns[FluxTrns::NUM_FLUX_TRANS];
      double diffVel[gpudata::MAXSPECIES * gpudata::MAXDIM];
      double distance = 0;
      if (distance_ != NULL) distance = (*distance_)[n];

      double radius = -1;
      if (config.isAxisymmetric()) {
        ParGridFunction *xyz = new ParGridFunction(dfes);
        mesh->GetNodes(*xyz);
        radius = (*xyz)[n + 0 * ndofs];
      }

      in_transport->ComputeFluxTransportProperties(state, gradUpn, Efield, radius, distance, fluxTrns, diffVel);
      for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) {
        dataVis[visualIdxs.FluxTrns + t][n] = fluxTrns[t];
      }
      for (int sp = 0; sp < _numSpecies; sp++) {
        for (int v = 0; v < _nvel; v++) dataVis[visualIdxs.diffVel + sp][n + v * ndofs] = diffVel[sp + v * _numSpecies];
      }

      // update source transport properties.
      double srcTrns[SrcTrns::NUM_SRC_TRANS];
      double speciesTrns[gpudata::MAXSPECIES * SpeciesTrns::NUM_SPECIES_COEFFS];
      in_transport->ComputeSourceTransportProperties(state, prim, gradUpn, Efield, distance, srcTrns, speciesTrns,
                                                     diffVel, nsp);
      for (int t = 0; t < SrcTrns::NUM_SRC_TRANS; t++) {
        dataVis[visualIdxs.SrcTrns + t][n] = srcTrns[t];
      }
      for (int t = 0; t < SpeciesTrns::NUM_SPECIES_COEFFS; t++) {
        for (int sp = 0; sp < _numSpecies; sp++) {
          dataVis[visualIdxs.SpeciesTrns + sp + t * _numSpecies][n] = speciesTrns[sp + t * _numSpecies];
        }
      }

      // update chemistry.
      double Th = 0., Te = 0.;
      Th = prim[1 + _nvel];
      Te = (in_mix->IsTwoTemperature()) ? prim[_num_equation - 1] : Th;
      double kfwd[gpudata::MAXREACTIONS], kC[gpudata::MAXREACTIONS];
      in_chem->computeForwardRateCoeffs(nsp, Th, Te, n, kfwd);
      in_chem->computeEquilibriumConstants(Th, Te, kC);
      // get reaction rates
      double progressRates[gpudata::MAXREACTIONS];
      for (int r = 0; r < _numReactions; r++) progressRates[r] = 0.0;
      in_chem->computeProgressRate(nsp, kfwd, kC, progressRates);
      for (int r = 0; r < _numReactions; r++) {
        dataVis[visualIdxs.rxn + r][n] = progressRates[r];
      }
    }  // if (!isDryAir)
  }    // for (int n = 0; n < ndofs; n++)
}

void M2ulPhyS::evaluatePlasmaConductivityGF() {
  assert(plasma_conductivity_ != NULL);
  double *d_pc = plasma_conductivity_->Write();

  const double *d_Up = Up->Read();
  const double *d_U = U->Read();
  const double *d_gradUp = gradUp->Read();

  const double *d_distance = NULL;
  if (distance_ != NULL) {
    d_distance = distance_->Read();
  }

  TransportProperties *d_transport = transportPtr;

  const int nnodes = vfes->GetNDofs();
  const int _dim = dim;
  const int _nvel = nvel;
  const int _num_equation = num_equation;
  const int _numSpecies = numSpecies;

#if defined(_GPU_)
  MFEM_FORALL(n, nnodes, {
    double upn[gpudata::MAXEQUATIONS];
    double Un[gpudata::MAXEQUATIONS];
    double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
#else
  double upn[gpudata::MAXEQUATIONS];
  double Un[gpudata::MAXEQUATIONS];
  double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
  for (int n = 0; n < nnodes; n++) {
#endif
    for (int eq = 0; eq < _num_equation; eq++) {
      upn[eq] = d_Up[n + eq * nnodes];
      Un[eq] = d_U[n + eq * nnodes];
      for (int d = 0; d < _dim; d++)
        gradUpn[eq + d * _num_equation] = d_gradUp[n + eq * nnodes + d * _num_equation * nnodes];
    }
    // TODO(kevin): update E-field with EM coupling.
    // E-field can have azimuthal component.
    double Efield[gpudata::MAXDIM];
    for (int v = 0; v < _nvel; v++) Efield[v] = 0.0;

    double globalTransport[gpudata::MAXSPECIES];
    double speciesTransport[gpudata::MAXSPECIES * SpeciesTrns::NUM_SPECIES_COEFFS];
    // NOTE: diffusion has nvel components, as E-field can have azimuthal component.
    double diffusionVelocity[gpudata::MAXSPECIES * gpudata::MAXDIM];
    for (int v = 0; v < _nvel; v++)
      for (int sp = 0; sp < _numSpecies; sp++) diffusionVelocity[sp + v * _numSpecies] = 0.0;
    double ns[gpudata::MAXSPECIES];

    double dist = 0.0;
    if (d_distance != NULL) dist = d_distance[n];
    d_transport->ComputeSourceTransportProperties(Un, upn, gradUpn, Efield, dist, globalTransport, speciesTransport,
                                                  diffusionVelocity, ns);

    d_pc[n] = globalTransport[SrcTrns::ELECTRIC_CONDUCTIVITY];
#if defined(_GPU_)
  });  // MFEM_FORALL(n, nnodes, {
#else
  }  // for (int n = 0; n < nnodes; n++)
#endif
}
