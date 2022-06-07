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

M2ulPhyS::M2ulPhyS(MPI_Session &_mpi, TPS::Tps *tps) : mpi(_mpi) {
  tpsP = tps;
  nprocs_ = mpi.WorldSize();
  rank_ = mpi.WorldRank();
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  groupsMPI = new MPI_Groups(&mpi);

  parseSolverOptions2();
}

M2ulPhyS::M2ulPhyS(MPI_Session &_mpi, string &inputFileName, TPS::Tps *tps) : mpi(_mpi) {
  tpsP = tps;
  nprocs_ = mpi.WorldSize();
  rank_ = mpi.WorldRank();
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  groupsMPI = new MPI_Groups(&mpi);

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

  // set default solver state
  exit_status_ = NORMAL;

  // This example depends on this ordering of the space.
  // MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
#ifdef _GPU_
  // write to GPU global memory
  gradUp->ReadWrite();
  U->ReadWrite();
  Up->ReadWrite();
  auto vgradUp = gradUp->ReadWrite();
  auto v_U = U->ReadWrite();
  auto vUp = Up->ReadWrite();
#endif  // _GPU_

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }
}

void M2ulPhyS::initVariables() {
#ifdef HAVE_GRVY
  grvy_timer_init("TPS");
#endif

  loadFromAuxSol = config.RestartFromAux();

  // check if a simulation is being restarted
  if (config.GetRestartCycle() > 0) {
    // read serial mesh and corresponding partition file (the hdf file that was generated
    // when starting from scratch).

    serial_mesh = new Mesh(config.GetMeshFileName().c_str());

    if (config.GetUniformRefLevels() > 0) {
      if (mpi.Root()) {
        std::cerr << "ERROR: Uniform mesh refinement not supported upon restart." << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // read partitioning info from original decomposition (unless restarting from serial soln)
    nelemGlobal_ = serial_mesh->GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      if (config.RestartSerial() == "read") {
        assert(serial_mesh->Conforming());
        partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
        partitioning_file_hdf5("write");
      } else {
        partitioning_file_hdf5("read");
      }
    }

    mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh, partitioning_);

    // only need serial mesh if on rank 0 and using single restart file option
    if (!mpi.Root() || (config.RestartSerial() == "no")) delete serial_mesh;

    // Paraview setup
    paraviewColl = new ParaViewDataCollection(config.GetOutputName(), mesh);
    paraviewColl->SetLevelsOfDetail(config.GetSolutionOrder());
    paraviewColl->SetHighOrderOutput(true);
    paraviewColl->SetPrecision(8);
    // paraviewColl->SetDataFormat(VTKFormat::ASCII);

  } else {
    // remove previous solution
    if (mpi.Root()) {
      string command = "rm -r ";
      command.append(config.GetOutputName());
      int err = system(command.c_str());
      if (err != 0) {
        cout << "Error deleting previous data in " << config.GetOutputName() << endl;
      }
    }

    serial_mesh = new Mesh(config.GetMeshFileName().c_str());

    // uniform refinement, user-specified number of times
    for (int l = 0; l < config.GetUniformRefLevels(); l++) {
      if (mpi.Root()) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      serial_mesh->UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh->GetNE();
    if (nprocs_ > 1) {
      assert(serial_mesh->Conforming());
      partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write");
      MPI_Barrier(MPI_COMM_WORLD);
    }

    mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh, partitioning_);

    // we only need serial mesh if on rank 0 and using the serial restart file option
    if (!mpi.Root() || (config.RestartSerial() == "no")) delete serial_mesh;

    // VisIt setup
    //     visitColl = new VisItDataCollection(config.GetOutputName(), mesh);
    //     visitColl->SetPrefixPath(config.GetOutputName());
    //     visitColl->SetPrecision(8);

    // Paraview setup
    paraviewColl = new ParaViewDataCollection(config.GetOutputName(), mesh);
    paraviewColl->SetLevelsOfDetail(config.GetSolutionOrder());
    paraviewColl->SetHighOrderOutput(true);
    paraviewColl->SetPrecision(8);
    // paraviewColl->SetDataFormat(VTKFormat::ASCII);

    time = 0.;
    iter = 0;
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
      if (mpi.WorldRank() == partitioning_[gelem]) {
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
    MPI_Allreduce(&localElems, &minElems, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localElems, &maxElems, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

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

  mixture = NULL;
  switch (config.GetWorkingFluid()) {
    case WorkingFluid::DRY_AIR:
      mixture = new DryAir(config, dim, nvel);
      transportPtr = new DryAirTransport(mixture, config);
      break;
    case WorkingFluid::TEST_BINARY_AIR:
      mixture = new TestBinaryAir(config, dim, nvel);
      transportPtr = new TestBinaryAirTransport(mixture, config);
      break;
    case WorkingFluid::USER_DEFINED:
      switch (config.GetGasModel()) {
        case GasModel::PERFECT_MIXTURE:
          mixture = new PerfectMixture(config, dim, nvel);
          break;
        default:
          mfem_error("GasModel not recognized.");
          break;
      }
      switch (config.GetTranportModel()) {
        case ARGON_MINIMAL:
          transportPtr = new ArgonMinimalTransport(mixture, config);
          break;
        case ARGON_MIXTURE:
          transportPtr = new ArgonMixtureTransport(mixture, config);
          break;
        case CONSTANT:
          transportPtr = new ConstantTransport(mixture, config);
          break;
        default:
          mfem_error("TransportModel not recognized.");
          break;
      }
      switch (config.GetChemistryModel()) {
        default:
          chemistry_ = new Chemistry(mixture, config);
          break;
      }
      break;
    default:
      mfem_error("WorkingFluid not recognized.");
      break;
  }
  assert(mixture != NULL);

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
  vfes = new ParFiniteElementSpace(mesh, fec, num_equation, Ordering::byNODES);
  gradUpfes = new ParFiniteElementSpace(mesh, fec, num_equation * dim, Ordering::byNODES);

  initIndirectionArrays();
  initSolutionAndVisualizationVectors();

  average = new Averaging(Up, mesh, fec, fes, dfes, vfes, eqSystem, mixture, num_equation, dim, config, groupsMPI);
  average->read_meanANDrms_restart_files();

  // NOTE: this should also be completed by the GasMixture class
  // Kevin: Do we need GasMixture class for this?
  // register rms and mean sol into ioData
  if (average->ComputeMean()) {
    // meanUp
    ioData.registerIOFamily("Time-averaged primitive vars", "/meanSolution", average->GetMeanUp(), false,
                            config.GetRestartMean());
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

      int inputSpeciesIndex = mixture->getInputIndexOf(sp);
      std::string speciesName = config.speciesNames[inputSpeciesIndex];
      ioData.registerIOVar("/meanSolution", "mean-Y" + speciesName, sp + nvel + 2);
    }

    // rms
    ioData.registerIOFamily("RMS velocity fluctuation", "/rmsData", average->GetRMS(), false, config.GetRestartMean());
    ioData.registerIOVar("/rmsData", "uu", 0);
    ioData.registerIOVar("/rmsData", "vv", 1);
    ioData.registerIOVar("/rmsData", "ww", 2);
    ioData.registerIOVar("/rmsData", "uv", 3);
    ioData.registerIOVar("/rmsData", "uw", 4);
    ioData.registerIOVar("/rmsData", "vw", 5);
  }

  ioData.initializeSerial(mpi.Root(), (config.RestartSerial() != "no"), serial_mesh);
  projectInitialSolution();

  fluxClass = new Fluxes(mixture, eqSystem, transportPtr, num_equation, dim, config.isAxisymmetric());

  alpha = 0.5;
  isSBP = config.isSBP();

  // Create Riemann Solver
  rsolver =
      new RiemannSolver(num_equation, mixture, eqSystem, fluxClass, config.RoeRiemannSolver(), config.isAxisymmetric());

  // Boundary attributes in present partition
  Array<int> local_attr;
  getAttributesInPartition(local_attr);

  bcIntegrator = NULL;
  if (local_attr.Size() > 0) {
    bcIntegrator = new BCintegrator(groupsMPI, mesh, vfes, intRules, rsolver, dt, mixture, fluxClass, Up, gradUp,
                                    shapesBC, normalsWBC, intPointsElIDBC, dim, num_equation, max_char_speed, config,
                                    local_attr, maxIntPoints, maxDofs);
  }

  // A->SetAssemblyLevel(AssemblyLevel::PARTIAL);

  A = new DGNonLinearForm(fluxClass, vfes, gradUpfes, gradUp, bcIntegrator, intRules, dim, num_equation, mixture,
                          gpuArrays, maxIntPoints, maxDofs);
  if (local_attr.Size() > 0) A->AddBdrFaceIntegrator(bcIntegrator);

  {
    bool useLinearIntegration = false;
    //    if( basisType==1 && intRuleType==1 ) useLinearIntegration = true;

    faceIntegrator = new FaceIntegrator(intRules, rsolver, fluxClass, vfes, useLinearIntegration, dim, num_equation,
                                        gradUp, gradUpfes, max_char_speed, config.isAxisymmetric());
  }
  A->AddInteriorFaceIntegrator(faceIntegrator);
  if (isSBP) {
    SBPoperator = new SBPintegrator(mixture, fluxClass, intRules, dim, num_equation, alpha);
    A->AddDomainIntegrator(SBPoperator);
  }

  Aflux = new MixedBilinearForm(dfes, fes);
  domainIntegrator = new DomainIntegrator(fluxClass, intRules, intRuleType, dim, num_equation, config.isAxisymmetric());
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

  //   gradUp_A = new ParNonlinearForm(gradUpfes);
  //   gradUp_A->AddInteriorFaceIntegrator(
  //       new GradFaceIntegrator(intRules, dim, num_equation) );
  gradUp_A = new GradNonLinearForm(
#ifdef _GPU_
      vfes,
#else
      gradUpfes,
#endif
      intRules, dim, num_equation, gpuArrays, maxIntPoints, maxDofs);
  gradUp_A->AddInteriorFaceIntegrator(new GradFaceIntegrator(intRules, dim, num_equation));

  rhsOperator =
      new RHSoperator(iter, dim, num_equation, order, eqSystem, max_char_speed, intRules, intRuleType, fluxClass,
                      mixture, chemistry_, transportPtr, vfes, gpuArrays, maxIntPoints, maxDofs, A, Aflux, mesh,
                      spaceVaryViscMult, U, Up, gradUp, gradUpfes, gradUp_A, bcIntegrator, isSBP, alpha, config);

  CFL = config.GetCFLNumber();
  rhsOperator->SetTime(time);
  timeIntegrator->Init(*rhsOperator);

  // Determine the minimum element size.
  {
    double local_hmin = mesh->GetElementSize(0, 1);
    for (int i = 1; i < mesh->GetNE(); i++) {
      // if(sqrt(mesh->GetElementVolume(i))<1e-3) cout<<sqrt(mesh->GetElementVolume(i))<<endl;
      local_hmin = min(mesh->GetElementSize(i, 1), local_hmin);
      // local_hmin = min(sqrt(mesh->GetElementVolume(i)), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
  }

  // estimate initial dt
  Up->ExchangeFaceNbrData();
  gradUp->ExchangeFaceNbrData();

  if (config.GetRestartCycle() == 0) initialTimeStep();
  if (mpi.Root()) cout << "Initial time-step: " << dt << "s" << endl;

  // t_final = MaxIters*dt;

  if (mpi.Root()) {
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
  gpuArrays.posDofIds.SetSize(2 * vfes->GetNE());
  gpuArrays.posDofIds = 0;
  auto hposDofIds = gpuArrays.posDofIds.HostWrite();

  std::vector<int> tempNodes;
  tempNodes.clear();

  for (int i = 0; i < vfes->GetNE(); i++) {
    const int dof = vfes->GetFE(i)->GetDof();

    // get the nodes IDs
    hposDofIds[2 * i] = tempNodes.size();
    hposDofIds[2 * i + 1] = dof;
    Array<int> dofs;
    vfes->GetElementVDofs(i, dofs);
    for (int n = 0; n < dof; n++) tempNodes.push_back(dofs[n]);
  }

  gpuArrays.nodesIDs.SetSize(tempNodes.size());
  gpuArrays.nodesIDs = 0;
  auto hnodesIDs = gpuArrays.nodesIDs.HostWrite();
  for (int i = 0; i < gpuArrays.nodesIDs.Size(); i++) hnodesIDs[i] = tempNodes[i];

  // count number of each type of element
  std::vector<int> tempNumElems;
  tempNumElems.clear();
  int dof1 = hposDofIds[1];
  int typeElems = 0;
  for (int el = 0; el < gpuArrays.posDofIds.Size() / 2; el++) {
    int dofi = hposDofIds[2 * el + 1];
    if (dofi == dof1) {
      typeElems++;
    } else {
      tempNumElems.push_back(typeElems);
      typeElems = 1;
      dof1 = dofi;
    }
  }
  tempNumElems.push_back(typeElems);
  gpuArrays.numElems.SetSize(tempNumElems.size());
  gpuArrays.numElems = 0;
  auto hnumElems = gpuArrays.numElems.HostWrite();
  for (int i = 0; i < (int)tempNumElems.size(); i++) hnumElems[i] = tempNumElems[i];

  {
    // Initialize vectors and arrays to be used in
    // face integrations
    //     const int maxIntPoints = 49; // corresponding to square face with p=5
    //     const int maxDofs = 216; //HEX with p=5

    gpuArrays.elemFaces.SetSize(7 * vfes->GetNE());
    gpuArrays.elemFaces = 0;
    auto helemFaces = gpuArrays.elemFaces.HostWrite();

    std::vector<double> shapes2, shapes1;
    shapes1.clear();
    shapes2.clear();

    Mesh *mesh = vfes->GetMesh();

    gpuArrays.elems12Q.SetSize(3 * mesh->GetNumFaces());
    gpuArrays.elems12Q = 0;
    auto helems12Q = gpuArrays.elems12Q.HostWrite();

    gpuArrays.shapeWnor1.UseDevice(true);
    gpuArrays.shapeWnor1.SetSize((maxDofs + 1 + dim) * maxIntPoints * mesh->GetNumFaces());

    gpuArrays.shape2.UseDevice(true);
    gpuArrays.shape2.SetSize(maxDofs * maxIntPoints * mesh->GetNumFaces());

    auto hshapeWnor1 = gpuArrays.shapeWnor1.HostWrite();
    auto hshape2 = gpuArrays.shape2.HostWrite();

    for (int face = 0; face < mesh->GetNumFaces(); face++) {
      FaceElementTransformations *tr;
      tr = mesh->GetInteriorFaceTransformations(face);
      if (tr != NULL) {
        Array<int> vdofs;
        Array<int> vdofs2;
        fes->GetElementVDofs(tr->Elem1No, vdofs);
        fes->GetElementVDofs(tr->Elem2No, vdofs2);

        {
          int nf = helemFaces[7 * tr->Elem1No];
          if (nf < 0) nf = 0;
          helemFaces[7 * tr->Elem1No + nf + 1] = face;
          nf++;
          helemFaces[7 * tr->Elem1No] = nf;

          nf = helemFaces[7 * tr->Elem2No];
          if (nf < 0) nf = 0;
          helemFaces[7 * tr->Elem2No + nf + 1] = face;
          nf++;
          helemFaces[7 * tr->Elem2No] = nf;
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

        helems12Q[3 * face] = tr->Elem1No;
        helems12Q[3 * face + 1] = tr->Elem2No;
        helems12Q[3 * face + 2] = ir->GetNPoints();

        Vector shape1i, shape2i;
        shape1i.UseDevice(false);
        shape2i.UseDevice(false);
        shape1i.SetSize(dof1);
        shape2i.SetSize(dof2);
        for (int k = 0; k < ir->GetNPoints(); k++) {
          const IntegrationPoint &ip = ir->IntPoint(k);
          tr->SetAllIntPoints(&ip);
          // shape functions
          fe1->CalcShape(tr->GetElement1IntPoint(), shape1i);
          fe2->CalcShape(tr->GetElement2IntPoint(), shape2i);
          for (int j = 0; j < dof1; j++)
            hshapeWnor1[face * (maxDofs + dim + 1) * maxIntPoints + j + k * (dim + 1 + maxDofs)] = shape1i[j];
          for (int j = 0; j < dof2; j++) hshape2[face * maxDofs * maxIntPoints + j + k * maxDofs] = shape2i[j];

          hshapeWnor1[face * (maxDofs + dim + 1) * maxIntPoints + maxDofs + k * (dim + 1 + maxDofs)] = ip.weight;
          // normals (multiplied by determinant of jacobian
          Vector nor;
          nor.UseDevice(false);
          nor.SetSize(dim);
          CalcOrtho(tr->Jacobian(), nor);
          for (int d = 0; d < dim; d++)
            hshapeWnor1[face * (maxDofs + dim + 1) * maxIntPoints + maxDofs + 1 + d + k * (maxDofs + dim + 1)] = nor[d];
        }
      }
    }
  }

  //  BC  integration arrays
  if (fes->GetNBE() > 0) {
    const int NumBCelems = fes->GetNBE();

    shapesBC.UseDevice(true);
    shapesBC.SetSize(NumBCelems * maxIntPoints * maxDofs);
    shapesBC = 0.;
    auto hshapesBC = shapesBC.HostWrite();

    normalsWBC.UseDevice(true);
    normalsWBC.SetSize(NumBCelems * maxIntPoints * (dim + 1));
    normalsWBC = 0.;
    auto hnormalsWBC = normalsWBC.HostWrite();

    intPointsElIDBC.SetSize(NumBCelems * 2);
    intPointsElIDBC = 0;
    auto hintPointsElIDBC = intPointsElIDBC.HostWrite();

    const FiniteElement *fe;
    FaceElementTransformations *tr;
    Mesh *mesh = fes->GetMesh();

    for (int f = 0; f < NumBCelems; f++) {
      tr = mesh->GetBdrFaceTransformations(f);
      if (tr != NULL) {
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

        hintPointsElIDBC[2 * f] = ir->GetNPoints();
        hintPointsElIDBC[2 * f + 1] = tr->Elem1No;

        for (int q = 0; q < ir->GetNPoints(); q++) {
          const IntegrationPoint &ip = ir->IntPoint(q);
          tr->SetAllIntPoints(&ip);
          Vector nor;
          nor.UseDevice(false);
          nor.SetSize(dim);
          CalcOrtho(tr->Jacobian(), nor);
          hnormalsWBC[dim + q * (dim + 1) + f * maxIntPoints * (dim + 1)] = ip.weight;
          for (int d = 0; d < dim; d++) hnormalsWBC[d + q * (dim + 1) + f * maxIntPoints * (dim + 1)] = nor[d];

          Vector shape1;
          shape1.UseDevice(false);
          shape1.SetSize(elDof);
          fe->CalcShape(tr->GetElement1IntPoint(), shape1);
          for (int n = 0; n < elDof; n++) hshapesBC[n + q * maxDofs + f * maxIntPoints * maxDofs] = shape1(n);
        }
      }
    }
  } else {
    shapesBC.SetSize(1);
    shapesBC = 0.;

    normalsWBC.SetSize(1);
    normalsWBC = 0.;

    intPointsElIDBC.SetSize(1);
    intPointsElIDBC = 0.;
  }

  {
    // fill out gradient shape function arrays and their indirections
    std::vector<double> temp;
    std::vector<int> positions;
    temp.clear();
    positions.clear();

    for (int el = 0; el < vfes->GetNE(); el++) {
      positions.push_back((int)temp.size());

      const FiniteElement *elem = vfes->GetFE(el);
      ElementTransformation *Tr = vfes->GetElementTransformation(el);
      const int elDof = elem->GetDof();

      // element volume integral
      int intorder = 2 * elem->GetOrder();
      if (intRuleType == 1 && elem->GetGeomType() == Geometry::SQUARE) intorder--;  // when Gauss-Lobatto
      const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);

      positions.push_back(ir->GetNPoints());

      for (int i = 0; i < ir->GetNPoints(); i++) {
        IntegrationPoint ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        // Calculate the shape functions
        Vector shape;
        shape.UseDevice(false);
        shape.SetSize(elDof);
        Vector dshapeVec;
        dshapeVec.UseDevice(false);
        dshapeVec.SetSize(elDof * dim);
        dshapeVec = 0.;
        DenseMatrix dshape(dshapeVec.HostReadWrite(), elDof, dim);
        elem->CalcShape(ip, shape);
        elem->CalcPhysDShape(*Tr, dshape);
        double detJac = Tr->Jacobian().Det() * ip.weight;

        for (int n = 0; n < elDof; n++) temp.push_back(shape[n]);
        for (int d = 0; d < dim; d++)
          for (int n = 0; n < elDof; n++) temp.push_back(dshape(n, d));
        temp.push_back(detJac);
      }
    }

    gpuArrays.elemShapeDshapeWJ.UseDevice(true);
    gpuArrays.elemShapeDshapeWJ.SetSize((int)temp.size());
    gpuArrays.elemShapeDshapeWJ = 0.;
    auto helemShapeDshapeWJ = gpuArrays.elemShapeDshapeWJ.HostWrite();
    for (int i = 0; i < gpuArrays.elemShapeDshapeWJ.Size(); i++) helemShapeDshapeWJ[i] = temp[i];
    gpuArrays.elemShapeDshapeWJ.Read();

    gpuArrays.elemPosQ_shapeDshapeWJ.SetSize((int)positions.size());
    for (int i = 0; i < gpuArrays.elemPosQ_shapeDshapeWJ.Size(); i++)
      gpuArrays.elemPosQ_shapeDshapeWJ[i] = positions[i];
    gpuArrays.elemPosQ_shapeDshapeWJ.Read();
  }

#ifdef _GPU_
  auto dnodesID = gpuArrays.nodesIDs.Read();
  auto dnumElems = gpuArrays.numElems.Read();
  auto dposDofIds = gpuArrays.posDofIds.Read();

  auto dshapeWnor1 = gpuArrays.shapeWnor1.Read();
  auto dshape2 = gpuArrays.shape2.Read();

  auto delemFaces = gpuArrays.elemFaces.Read();
  auto delems12Q = gpuArrays.elems12Q.Read();

  auto dshapesBC = shapesBC.Read();
  auto dnormalsBC = normalsWBC.Read();
  auto dintPointsELIBC = intPointsElIDBC.Read();
#endif
}

M2ulPhyS::~M2ulPhyS() {
  if (mpi.Root()) histFile.close();

  delete gradUp;

  delete gradUp_A;

  // ks (aug 2021) - following two lines cause unknown pointer errors with
  // MFEM 4.3 (not 4.2). Commenting out for now.

  // delete u_block;
  // delete up_block;
  delete offsets;

  delete U;
  delete Up;

  delete average;

  delete rhsOperator;
  // delete domainIntegrator;
  delete Aflux;  // fails to delete (follow this)
  if (isSBP) delete SBPoperator;
  // delete faceIntegrator;
  delete A;  // fails to delete (follow this)

  delete timeIntegrator;
  // delete inlet/outlet integrators

  delete rsolver;
  delete fluxClass;
  delete transportPtr;
  delete chemistry_;
  delete mixture;
  delete gradUpfes;
  delete vfes;
  delete dfes;
  delete fes;
  delete fec;
  delete intRules;

  delete paraviewColl;
  // delete mesh;

  delete groupsMPI;

  delete[] locToGlobElem;

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

#ifdef HAVE_GRVY
  if (mpi.WorldRank() == 0) grvy_timer_summarize();
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
  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) {
    (*offsets)[k] = k * vfes->GetNDofs();
  }
  u_block = new BlockVector(*offsets);
  up_block = new BlockVector(*offsets);

  // gradUp.SetSize(num_equation*dim*vfes->GetNDofs());
  gradUp = new ParGridFunction(gradUpfes);

  U = new ParGridFunction(vfes, u_block->HostReadWrite());
  Up = new ParGridFunction(vfes, up_block->HostReadWrite());

  dens = new ParGridFunction(fes, Up->HostReadWrite());
  vel = new ParGridFunction(dfes, Up->HostReadWrite() + fes->GetNDofs());

  if (config.isAxisymmetric()) {
    vtheta = new ParGridFunction(fes, Up->HostReadWrite() + 3 * fes->GetNDofs());
  } else {
    vtheta = NULL;
  }

  temperature = new ParGridFunction(fes, Up->HostReadWrite() + (1 + nvel) * fes->GetNDofs());

  // this variable is purely for visualization
  press = new ParGridFunction(fes);

  passiveScalar = NULL;
  if (eqSystem == NS_PASSIVE) {
    passiveScalar = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());
  } else {
    // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
    // will need to add full list of species.
    visualizationVariables.resize(numActiveSpecies);
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      visualizationVariables[sp] = new ParGridFunction(fes, U->HostReadWrite() + (sp + nvel + 2) * fes->GetNDofs());
    }
  }

  // If mms, add conserved and exact solution.
#ifdef HAVE_MASA
  if (config.use_mms_ && config.mmsSaveDetails_) {
    // for visualization.
    masaUBlock_ = new BlockVector(*offsets);
    masaU_ = new ParGridFunction(vfes, masaUBlock_->HostReadWrite());
    masaRhs_ = new ParGridFunction(*U);

    for (int eq = 0; eq < num_equation; eq++)
      visualizationVariables.push_back(new ParGridFunction(fes, U->HostReadWrite() + eq * fes->GetNDofs()));
    for (int eq = 0; eq < num_equation; eq++)
      visualizationVariables.push_back(new ParGridFunction(fes, masaU_->HostReadWrite() + eq * fes->GetNDofs()));
    for (int eq = 0; eq < num_equation; eq++)
      visualizationVariables.push_back(new ParGridFunction(fes, masaRhs_->HostReadWrite() + eq * fes->GetNDofs()));
  }
#endif

  // define solution parameters for i/o
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
  // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
  // will need to add full list of species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // Only for NS_PASSIVE.
    if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

    int inputSpeciesIndex = mixture->getInputIndexOf(sp);
    std::string speciesName = config.speciesNames[inputSpeciesIndex];
    ioData.registerIOVar("/solution", "rho-Y_" + speciesName, sp + nvel + 2);
  }

  // compute factor to multiply viscosity when this option is active
  spaceVaryViscMult = NULL;
  ParGridFunction coordsDof(dfes);
  mesh->GetNodes(coordsDof);
  if (config.linViscData.isEnabled) {
    spaceVaryViscMult = new ParGridFunction(fes);
    double *viscMult = spaceVaryViscMult->HostWrite();
    for (int n = 0; n < fes->GetNDofs(); n++) {
      double alpha = 1.;
      auto hcoords = coordsDof.HostRead();  // get coords
      double dist_pi = 0., dist_p0 = 0., dist_pi0 = 0.;
      for (int d = 0; d < dim; d++) {
        dist_pi += config.GetLinearVaryingData().normal(d) *
                   (config.GetLinearVaryingData().pointInit(d) - hcoords[n + d * vfes->GetNDofs()]);
        dist_p0 += config.GetLinearVaryingData().normal(d) *
                   (config.GetLinearVaryingData().point0(d) - hcoords[n + d * vfes->GetNDofs()]);
        dist_pi0 += config.GetLinearVaryingData().normal(d) *
                    (config.GetLinearVaryingData().pointInit(d) - config.GetLinearVaryingData().point0(d));
      }

      if (dist_pi > 0. && dist_p0 < 0) {
        alpha += (config.GetLinearVaryingData().viscRatio - 1.) / dist_pi0 * dist_pi;
      }
      viscMult[n] = alpha;
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
  } else if (numActiveSpecies > 0) {
    // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
    // will need to add full list of species.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      int inputSpeciesIndex = mixture->getInputIndexOf(sp);
      std::string speciesName = config.speciesNames[inputSpeciesIndex];
      paraviewColl->RegisterField("partial_density_" + speciesName, visualizationVariables[sp]);
    }
  }

// If mms, add exact solution.
#ifdef HAVE_MASA
  if (config.use_mms_ && config.mmsSaveDetails_) {
    for (int eq = 0; eq < num_equation; eq++)
      paraviewColl->RegisterField("U" + std::to_string(eq), visualizationVariables[numActiveSpecies + eq]);
    for (int eq = 0; eq < num_equation; eq++)
      paraviewColl->RegisterField("mms_U" + std::to_string(eq),
                                  visualizationVariables[numActiveSpecies + num_equation + eq]);
    for (int eq = 0; eq < num_equation; eq++)
      paraviewColl->RegisterField("RHS" + std::to_string(eq),
                                  visualizationVariables[numActiveSpecies + 2 * num_equation + eq]);
  }
#endif

  if (spaceVaryViscMult != NULL) paraviewColl->RegisterField("viscMult", spaceVaryViscMult);

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
  std::cout << "restart: " << config.GetRestartCycle() << std::endl;
  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    uniformInitialConditions();
#ifdef HAVE_MASA
    if (config.use_mms_) {
      initMasaHandler();

      projectExactSolution(0.0, U);
      if (config.mmsSaveDetails_) projectExactSolution(0.0, masaU_);
    }
#endif
  } else {
    if (config.RestartHDFConversion())
      read_restart_files();
    else
      restart_files_hdf5("read");

    if (mpi.Root()) Cache_Paraview_Timesteps();

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    // to be used with future MFEM version...
    // paraviewColl->SetRestartMode(true);
  }

  initGradUp();

  updatePrimitives();

  // update pressure grid function
  mixture->UpdatePressureGridFunction(press, Up);

  paraviewColl->Save();
}

void M2ulPhyS::solve() {
  double tlast = grvy_timer_elapsed_global();

#ifdef HAVE_MASA
  // instantiate function for exact solution
  // NOTE: this has been taken care of at M2ulPhyS::initMasaHandler.
  // initMMSCoefficients();

  if (config.use_mms_) {
    checkSolutionError(time);
  }
#endif

#ifdef HAVE_SLURM
  bool readyForRestart = false;
#endif

  // Integrate in time.
  while (iter < MaxIters) {
    grvy_timer_begin(__func__);

    // periodically report on time/iteratino
    if ((iter % config.timingFreq) == 0) {
      if (mpi.Root()) {
        double timePerIter = (grvy_timer_elapsed_global() - tlast) / config.timingFreq;
        grvy_printf(ginfo, "Iteration = %i: wall clock time/iter = %.3f (secs)\n", iter, timePerIter);
        tlast = grvy_timer_elapsed_global();
      }
    }

    timeIntegrator->Step(*U, time, dt);

    Check_NAN();

    if (!config.isTimeStepConstant()) {
      double dt_local = CFL * hmin / max_char_speed / static_cast<double>(dim);
      MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
    }

    iter++;

    const int vis_steps = config.GetNumItersOutput();
    if (iter % vis_steps == 0) {
      // dump history
      writeHistoryFile();

#ifdef HAVE_MASA
      if (config.use_mms_) {
        if (config.mmsSaveDetails_) {
          rhsOperator->Mult(*U, *masaRhs_);
          projectExactSolution(time, masaU_);
        }
        checkSolutionError(time);
      } else {
        if (mpi.Root()) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
      }
#else
      if (mpi.Root()) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
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

        average->write_meanANDrms_restart_files(iter, time);
      }
    }

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

    average->addSampleMean(iter);

    // periodically check for DIE file which requests to terminate early
    if (Check_ExitEarly(iter)) {
      MaxIters = iter;
      SetStatus(EARLY_EXIT);
      break;
    }

    grvy_timer_end(__func__);
  }  // <-- end main timestep iteration loop

  if (iter == MaxIters) {
    // auto hUp = Up->HostRead();
    Up->HostRead();
    mixture->UpdatePressureGridFunction(press, Up);

    // write_restart_files();
    restart_files_hdf5("write");

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();

    average->write_meanANDrms_restart_files(iter, time);

#ifndef HAVE_MASA
    // // If HAVE_MASA is defined, this is handled above
    // void (*initialConditionFunction)(const Vector &, Vector &);
    // initialConditionFunction = &(this->InitialConditionEulerVortex);

    // VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    // const double error = U->ComputeLpError(2, u0);
    // if (mpi.Root()) cout << "Solution error: " << error << endl;
#else
    if (config.use_mms_) {
      checkSolutionError(time, true);
    } else {
      if (mpi.Root()) cout << "Final timestep iteration = " << MaxIters << endl;
    }
#endif

    if (mpi.Root()) cout << "Final timestep iteration = " << MaxIters << endl;
  }

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
  if (dim == 3) initState(3) = inputRhoRhoVp[3];

  if (mixture->GetWorkingFluid() == WorkingFluid::USER_DEFINED) {
    const int numSpecies = mixture->GetNumSpecies();
    const int numActiveSpecies = mixture->GetNumActiveSpecies();
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      int inputIndex = mixture->getInputIndexOf(sp);
      initState(2 + dim + sp) = inputRhoRhoVp[0] * config.initialMassFractions(inputIndex);
    }

    // electron energy
    if (mixture->IsTwoTemperature()) {
      double ne = 0.;
      if (mixture->IsAmbipolar()) {
        for (int sp = 0; sp < numActiveSpecies; sp++)
          ne += mixture->GetGasParams(sp, GasParams::SPECIES_CHARGES) * initState(2 + dim + sp) /
                mixture->GetGasParams(sp, GasParams::SPECIES_MW);
      } else {
        ne = initState(2 + dim + numSpecies - 2) / mixture->GetGasParams(numSpecies - 2, GasParams::SPECIES_MW);
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

// void M2ulPhyS::uniformInitialConditions() {
//   if (config.GetWorkingFluid() == DRY_AIR) {
//     dryAirUniformInitialConditions();
//     return;
//   }
//
//   std::string basepath("initialConditions");
//   Vector initCondition(num_equation);
//   for (int eq = 0; eq < num_equation; eq++) {
//     tpsP->getRequiredInput((basepath + "/Q" + std::to_string(eq+1)).c_str(), initCondition[eq]);
//   }
//
//   double *data = U->HostWrite();
//   double *dataUp = Up->HostWrite();
//   double *dataGradUp = gradUp->HostWrite();
//
//   int dof = vfes->GetNDofs();
//
//   Vector state;
//   state.UseDevice(false);
//   state.SetSize(num_equation);
//   Vector Upi;
//   Upi.UseDevice(false);
//   Upi.SetSize(num_equation);
//
//   for (int i = 0; i < dof; i++) {
//     for (int eq = 0; eq < num_equation; eq++) {
//       data[i + eq * dof] = initCondition(eq);
//       state(eq) = data[i + eq * dof];
//       for (int d = 0; d < dim; d++) {
//         dataGradUp[i + eq * dof + d * num_equation * dof] = 0.;
//       }
//     }
//
//     mixture->GetPrimitivesFromConservatives(state, Upi);
//     for (int eq = 0; eq < num_equation; eq++) dataUp[i + eq * dof] = Upi[eq];
//   }
//
//   return;
// }

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

void M2ulPhyS::write_restart_files() {
  string serialName = "restart_p";
  serialName.append(to_string(order));
  serialName.append("_");
  serialName.append(config.GetOutputName());
  serialName.append(".sol");

  string fileName = groupsMPI->getParallelName(serialName);
  ofstream file(fileName, std::ofstream::trunc);
  file.precision(8);

  // write cycle and time
  file << iter << " " << time << " " << dt << endl;

  // double *data = Up->GetData();
  const double *data = Up->HostRead();  // get data from GPU
  int dof = vfes->GetNDofs();

  for (int i = 0; i < dof * num_equation; i++) {
    file << data[i] << endl;
  }

  Up->Write();  // sets data back to GPU

  file.close();
}

void M2ulPhyS::read_restart_files() {
  if (mpi.Root()) {
    cout << endl;
    cout << "================================================" << endl;
    cout << "| Restarting simulation" << endl;
    cout << "================================================" << endl;
  }

  if (loadFromAuxSol) {
    cerr << "ERROR: Restart from auxOrder is not supported with ascii-based restarts." << endl;
    cerr << "To change order, convert the ascii-based restart file to hdf5 and then change the order." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  assert(!loadFromAuxSol);

  string serialName = "restart_p";
  serialName.append(to_string(order));
  serialName.append("_");
  serialName.append(config.GetOutputName());
  serialName.append(".sol");

  string fileName = groupsMPI->getParallelName(serialName);
  ifstream file(fileName);

  if (!file.is_open()) {
    cout << "Could not open file \"" << fileName << "\"" << endl;
    return;
  } else {
    double *data;
    data = Up->GetData();

    string line;
    // read time and iters
    {
      getline(file, line);
      istringstream ss(line);
      string word;
      ss >> word;
      iter = stoi(word);

      if (mpi.Root()) cout << "--> restart iter = " << iter << endl;
      config.SetRestartCycle(iter);

      ss >> word;
      time = stof(word);

      ss >> word;
      dt = stof(word);
    }

    int lines = 0;
    while (getline(file, line)) {
      istringstream ss(line);
      string word;
      ss >> word;

      data[lines] = stof(word);
      lines++;
    }
    file.close();

    double *dataUp = Up->GetData();

    // fill out U
    double *dataU = U->GetData();
    // double gamma = mixture->GetSpecificHeatRatio();
    int dof = vfes->GetNDofs();
    if (lines != dof * num_equation) {
      cout << "# of lines in files does not match domain size" << endl;
    } else {
      dof = vfes->GetNDofs();
      for (int i = 0; i < dof; i++) {
        Vector primitiveState(num_equation);
        Vector conservedState(num_equation);
        for (int eq = 0; eq < num_equation; eq++) primitiveState[eq] = dataUp[i + eq * dof];
        mixture->GetConservativesFromPrimitives(primitiveState, conservedState);
        for (int eq = 0; eq < num_equation; eq++) dataU[i + eq * dof] = conservedState[eq];
      }
    }
  }

  // load data to GPU
  // auto dUp = Up->ReadWrite();
  // auto dU = U->ReadWrite();
  Up->ReadWrite();
  U->ReadWrite();
  //  if( loadFromAuxSol ) auto dausUp = aux_Up->ReadWrite();
}

void M2ulPhyS::Check_NAN() {
  int local_print = 0;
  int dof = vfes->GetNDofs();

#ifdef _GPU_
  { local_print = M2ulPhyS::Check_NaN_GPU(U, dof * num_equation, loc_print); }
#else
  const double *dataU = U->HostRead();

  // bool thereIsNan = false;

  for (int i = 0; i < dof; i++) {
    Vector iState(num_equation);
    for (int eq = 0; eq < num_equation; eq++) {
      if (std::isnan(dataU[i + eq * dof])) {
        // thereIsNan = true;
        cout << "NaN at node: " << i << " partition: " << mpi.WorldRank() << endl;
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
  MPI_Allreduce(&local_print, &print, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (print > 0) {
    // auto hUp = Up->HostRead();  // get GPU data
    Up->HostRead();  // get GPU data
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

int M2ulPhyS::Check_NaN_GPU(ParGridFunction *U, int lengthU, Array<int> &loc_print) {
  const double *dataU = U->Read();
  auto d_temp = loc_print.Write();

  MFEM_FORALL(n, lengthU, {
    double val = dataU[n];
    if (val != val) d_temp[0] += 1;
  });

  auto htemp = loc_print.HostRead();
  return htemp[0];
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
  MPI_Allreduce(&partition_C, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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

  return;
}

void M2ulPhyS::parseFlowOptions() {
  tpsP->getInput("flow/order", config.solOrder, 4);
  tpsP->getInput("flow/integrationRule", config.integrationRule, 1);
  tpsP->getInput("flow/basisType", config.basisType, 1);
  tpsP->getInput("flow/maxIters", config.numIters, 10);
  tpsP->getInput("flow/outputFreq", config.itersOut, 50);
  tpsP->getInput("flow/timingFreq", config.timingFreq, 100);
  tpsP->getInput("flow/useRoe", config.useRoe, false);
  tpsP->getInput("flow/useSumByParts", config.SBP, false);
  tpsP->getInput("flow/refLength", config.refLength, 1.0);
  tpsP->getInput("flow/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP->getInput("flow/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  tpsP->getInput("flow/axisymmetric", config.axisymmetric_, false);
  tpsP->getInput("flow/enablePressureForcing", config.isForcing, false);
  if (config.isForcing) {
    for (int d = 0; d < 3; d++) tpsP->getRequiredVecElem("flow/pressureGrad", config.gradPress[d], d);
  }
  tpsP->getInput("flow/refinement_levels", config.ref_levels, 0);

  assert(config.solOrder > 0);
  assert(config.numIters >= 0);
  assert(config.itersOut > 0);
  assert(config.refLength > 0);
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

void M2ulPhyS::parseStatOptions() {
  tpsP->getInput("averaging/saveMeanHist", config.meanHistEnable, false);
  tpsP->getInput("averaging/startIter", config.startIter, 0);
  tpsP->getInput("averaging/sampleFreq", config.sampleInterval, 0);
  tpsP->getInput("averaging/enableContinuation", config.restartMean, false);
}

void M2ulPhyS::parseIOSettings() {
  tpsP->getInput("io/outdirBase", config.outputFile, std::string("output-default"));
  tpsP->getInput("io/enableRestart", config.restart, false);
  tpsP->getInput("io/exitCheckFreq", config.exit_checkFrequency_, 500);

  std::string restartMode;
  tpsP->getInput("io/restartMode", restartMode, std::string("standard"));
  if (restartMode == "variableP") {
    config.restartFromAux = true;
  } else if (restartMode == "singleFileWrite") {
    config.restart_serial = "write";
  } else if (restartMode == "singleFileRead") {
    config.restart_serial = "read";
  } else if (restartMode != "standard") {
    grvy_printf(GRVY_ERROR, "\nUnknown restart mode -> %s\n", restartMode.c_str());
    exit(ERROR);
  }
}

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
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[0], 0);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[1], 1);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[2], 2);

    auto point0 = config.linViscData.point0.HostWrite();
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[0], 0);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[1], 1);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[2], 2);

    auto pointInit = config.linViscData.pointInit.HostWrite();
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[0], 0);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[1], 1);
    tpsP->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[2], 2);

    tpsP->getRequiredInput("viscosityMultiplierFunction/viscosityRatio", config.linViscData.viscRatio);
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
  } else if (fluidTypeStr == "test_binary_air") {
    config.workFluid = TEST_BINARY_AIR;
  } else if (fluidTypeStr == "user_defined") {
    config.workFluid = USER_DEFINED;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown fluid preset supplied at runtime -> %s", fluidTypeStr.c_str());
    exit(ERROR);
  }

  if (config.workFluid != USER_DEFINED)
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
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nUnknown transport_model -> %s", transportModelStr.c_str());
  //   exit(ERROR);
  // }

  std::string chemistryModelStr;
  tpsP->getInput("plasma_models/chemistry_model", chemistryModelStr, std::string(""));

  // TODO(kevin): cantera wrapper
  // if (chemistryModelStr == "cantera") {
  //   config.chemistryModel_ = ChemistryModel::CANTERA;
  // }
}

void M2ulPhyS::parseSpeciesInputs() {
  // number of species defined
  if (config.eqSystem != NS_PASSIVE) {
    tpsP->getInput("species/numSpecies", config.numSpecies, 1);
    config.gasParams.SetSize(config.numSpecies, GasParams::NUM_GASPARAMS);
    // config.speciesNames.SetSize(config.numSpecies);
    config.initialMassFractions.SetSize(config.numSpecies);
    config.speciesNames.resize(config.numSpecies);

    if (config.gasModel == PERFECT_MIXTURE) {
      config.constantMolarCV.SetSize(config.numSpecies);
      config.constantMolarCP.SetSize(config.numSpecies);
    }
  }

  /*
    NOTE: for now, we force the user to set the background species as the last,
    and the electron species as the second to last.
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
    config.speciesComposition.SetSize(config.numSpecies, config.numAtoms);
    config.speciesComposition = 0.0;
    for (int i = 1; i <= config.numSpecies; i++) {
      // double mw, charge;
      double formEnergy;
      std::string type, speciesName;
      std::vector<std::pair<std::string, std::string>> composition;
      std::string basepath("species/species" + std::to_string(i));

      tpsP->getRequiredInput((basepath + "/name").c_str(), speciesName);
      config.speciesNames[i - 1] = speciesName;

      tpsP->getRequiredPairs((basepath + "/composition").c_str(), composition);
      for (size_t c = 0; c < composition.size(); c++) {
        if (config.atomMap.count(composition[c].first)) {
          int atomIdx = config.atomMap[composition[c].first];
          config.speciesComposition(i - 1, atomIdx) = stoi(composition[c].second);
        } else {
          grvy_printf(GRVY_ERROR, "Requested atom %s for species %s is not available!\n", composition[c].first.c_str(),
                      speciesName.c_str());
        }
      }

      tpsP->getRequiredInput((basepath + "/formation_energy").c_str(), formEnergy);
      config.gasParams(i - 1, GasParams::FORMATION_ENERGY) = formEnergy;

      tpsP->getRequiredInput((basepath + "/initialMassFraction").c_str(), config.initialMassFractions(i - 1));

      //// require initial electron temperature
      // if (speciesName == "E")
      //  tpsP->getRequiredInput((basepath + "/initialElectronTemperature").c_str(),
      //                         config.initialElectronTemperature);

      if (config.gasModel == PERFECT_MIXTURE) {
        tpsP->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(),
                               config.constantMolarCV(i - 1));
        // NOTE: For perfect gas, CP will be automatically set from CV.
      }
    }

    Vector speciesMass(config.numSpecies), speciesCharge(config.numSpecies);
    config.speciesComposition.Mult(config.atomMW, speciesMass);
    config.speciesComposition.GetColumn(config.atomMap["E"], speciesCharge);
    speciesCharge *= -1.0;
    config.gasParams.SetCol(GasParams::SPECIES_MW, speciesMass);
    config.gasParams.SetCol(GasParams::SPECIES_CHARGES, speciesCharge);
  }
}

void M2ulPhyS::parseTransportInputs() {
  switch (config.transportModel) {
    case ARGON_MINIMAL:
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
      break;
    case ARGON_MIXTURE:
      tpsP->getInput("plasma_models/transport_model/argon_mixture/third_order_thermal_conductivity",
                     config.thirdOrderkElectron, true);
      break;
    case CONSTANT: {
      tpsP->getRequiredInput("plasma_models/transport_model/constant/viscosity", config.constantTransport.viscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/bulk_viscosity",
                             config.constantTransport.bulkViscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/thermal_conductivity",
                             config.constantTransport.thermalConductivity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/electron_thermal_conductivity",
                             config.constantTransport.electronThermalConductivity);
      std::string diffpath("plasma_models/transport_model/constant/diffusivity");
      config.constantTransport.diffusivity.SetSize(config.numSpecies);
      std::string mtpath("plasma_models/transport_model/constant/momentum_transfer_frequency");
      config.constantTransport.mtFreq.SetSize(config.numSpecies);
      config.constantTransport.diffusivity = 0.0;
      config.constantTransport.mtFreq = 0.0;
      for (int sp = 1; sp <= config.numSpecies; sp++) {
        tpsP->getRequiredInput((diffpath + "/species" + std::to_string(sp)).c_str(),
                               config.constantTransport.diffusivity(sp - 1));
        if (config.twoTemperature)
          tpsP->getRequiredInput((mtpath + "/species" + std::to_string(sp)).c_str(),
                                 config.constantTransport.mtFreq(sp - 1));
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

    config.reactionModelParams.resize(config.numReactions);
    config.detailedBalance.SetSize(config.numReactions);
    config.equilibriumConstantParams.resize(config.numReactions);
  }

  for (int r = 1; r <= config.numReactions; r++) {
    std::string basepath("reactions/reaction" + std::to_string(r));

    // TODO(kevin): make tps input parser accessible to all classes.
    // TODO(kevin): reaction classes read input options directly in their initialization.
    std::string equation, model;
    tpsP->getRequiredInput((basepath + "/equation").c_str(), equation);
    config.reactionEquations[r - 1] = equation;
    tpsP->getRequiredInput((basepath + "/model").c_str(), model);

    double energy;
    tpsP->getRequiredInput((basepath + "/reaction_energy").c_str(), energy);
    config.reactionEnergies[r - 1] = energy;

    if (model == "arrhenius") {
      config.reactionModels[r - 1] = ARRHENIUS;
      config.reactionModelParams[r - 1].resize(3);
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.reactionModelParams[r - 1] = {A, b, E};

    } else if (model == "hoffert_lien") {
      config.reactionModels[r - 1] = HOFFERTLIEN;
      config.reactionModelParams[r - 1].resize(3);
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.reactionModelParams[r - 1] = {A, b, E};

    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown reaction_model -> %s", model.c_str());
      exit(ERROR);
    }

    bool detailedBalance;
    tpsP->getInput((basepath + "/detailed_balance").c_str(), detailedBalance, false);
    config.detailedBalance[r - 1] = detailedBalance;
    if (detailedBalance) {
      config.equilibriumConstantParams[r - 1].resize(3);
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/E").c_str(), E);
      config.equilibriumConstantParams[r - 1] = {A, b, E};
    }

    Array<double> stoich(config.numSpecies);
    tpsP->getRequiredVec((basepath + "/reactant_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) config.reactantStoich(sp, r - 1) = stoich[sp];
    tpsP->getRequiredVec((basepath + "/product_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) config.productStoich(sp, r - 1) = stoich[sp];
  }
}

void M2ulPhyS::parseBCInputs() {
  // number of BC regions defined
  int numWalls, numInlets, numOutlets;
  tpsP->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // Wall Bcs
  std::map<std::string, WallType> wallMapping;
  wallMapping["inviscid"] = INV;
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
    // For multi-component gas, require (numActiveSpecies)-more inputs.
    if ((config.workFluid != DRY_AIR) && (config.numSpecies > 1)) {
      grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
      if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

      for (int sp = 1; sp <= config.numSpecies; sp++) {
        double Ysp;
        // read mass fraction of species as listed in the input file.
        std::string speciesBasePath(basepath + "/mass_fraction/species" + std::to_string(sp));
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
      config.spongeData_[sz].targetUp.SetSize(5);

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

            for (int sp = 1; sp <= config.numSpecies; sp++) {
              // read mass fraction of species as listed in the input file.
              std::string speciesBasePath(base + "/mass_fraction/species" + std::to_string(sp));
              tpsP->getRequiredInput(speciesBasePath.c_str(), hup[4 + sp]);
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

void M2ulPhyS::checkSolverOptions() const {
#ifdef _GPU_
  if (!config.isTimeStepConstant()) {
    if (mpi.Root()) {
      std::cerr << "[ERROR]: GPU runs must use a constant time step: Please set DT_CONSTANT in input file."
                << std::endl;
      std::cerr << std::endl;
      exit(ERROR);
    }
  }

  if (config.isAxisymmetric()) {
    if (mpi.Root()) {
      std::cerr << "[ERROR]: Axisymmetric simulations not supported on GPU." << std::endl;
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
      if (mpi.Root()) {
        std::cerr << "[ERROR]: Passive scalars not supported for axisymmetric simulations." << std::endl;
        std::cerr << std::endl;
        exit(ERROR);
      }
    }
    // Don't support Roe flux yet
    if (config.RoeRiemannSolver()) {
      if (mpi.Root()) {
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
        if (mpi.Root()) {
          std::cerr << "[ERROR]: Only SUB_DENS_VEL inlet supported for axisymmetric simulations." << std::endl;
          std::cerr << std::endl;
          exit(ERROR);
        }
      }
    }
    for (size_t i = 0; i < config.GetOutletPatchType()->size(); i++) {
      std::pair<int, OutletType> patchANDtype = (*config.GetOutletPatchType())[i];
      if (patchANDtype.second != SUB_P) {
        if (mpi.Root()) {
          std::cerr << "[ERROR]: Only SUB_P outlet supported for axisymmetric simulations." << std::endl;
          std::cerr << std::endl;
          exit(ERROR);
        }
      }
    }
  }

#ifndef HAVE_MASA
  if (config.use_mms_) {
    if (mpi.Root()) {
      std::cerr << "[ERROR]: Require MASA support to run manufactured solutions." << std::endl;
      std::cerr << std::endl;
      exit(ERROR);
    }
  }
#endif

  // Warn user if they requested fixed dt without setting enableConstantTimestep
  if ((config.GetFixedDT() > 0) && (!config.isTimeStepConstant())) {
    if (mpi.Root()) {
      std::cerr << "[WARNING]: Setting dt_fixed overrides enableConstantTimestep." << std::endl;
      std::cerr << std::endl;
    }
  }

  return;
}

void M2ulPhyS::updatePrimitives() {
  double *data = U->HostWrite();
  double *dataUp = Up->HostWrite();
  int dof = vfes->GetNDofs();

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
