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
/*
 * Implementation of the class Problem
 */

#include "M2ulPhyS.hpp"

#include <sstream>

M2ulPhyS::M2ulPhyS(MPI_Session &_mpi, string &inputFileName) : mpi(_mpi) {
  nprocs_ = mpi.WorldSize();
  rank_ = mpi.WorldRank();
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  groupsMPI = new MPI_Groups(&mpi);

  config.readInputFile(inputFileName);

#ifdef _GPU_
  if (!config.isTimeStepConstant()) {
    if (mpi.Root()) {
      std::cerr << "[ERROR]: GPU runs must use a constant time step: Please set DT_CONSTANT in input file."
                << std::endl;
      std::cerr << std::endl;
      exit(ERROR);
    }
  }
#endif

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
    for (unsigned int gelem = 0; gelem < nelemGlobal_; gelem++) {
      if (mpi.WorldRank() == partitioning_[gelem]) {
        locToGlobElem[lelem] = gelem;
        lelem += 1;
      }
    }
  } else {
    locToGlobElem = NULL;
  }

  cout << "Process " << mpi.WorldRank() << " # elems " << mesh->GetNE() << endl;

  dim = mesh->Dimension();

  refLength = config.GetReferenceLength();

  eqSystem = config.GetEquationSystem();

  mixture = NULL;
  switch( config.GetWorkingFluid() ){
    case WorkingFluid::DRY_AIR:
      mixture = new DryAir( config, dim);
      mixture->setViscMult(config.GetViscMult());
      mixture->setBulkViscMult(config.GetBulkViscMult());
      break;
    case WorkingFluid::USER_DEFINED:
      break;
  }
  assert(mixture!=NULL);

  order = config.GetSolutionOrder();

  MaxIters = config.GetNumIters();

  max_char_speed = 0.;

  // NOTE: num_equations to be given by the gasMixture
  switch (eqSystem) {
    case EULER:
      num_equation = 2 + dim;
      break;
    case NS:
      num_equation = 2 + dim;
      break;
    case NS_PASSIVE:
      num_equation = 2 + dim + 1;
      break;
    default:
      break;
  }

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

  average = new Averaging(Up, mesh, fec, fes, dfes, vfes, eqSystem, num_equation, dim, config, groupsMPI);
  average->read_meanANDrms_restart_files();

  // NOTE: this should also be completed by the GasMixture class
  // register rms and mean sol into ioData
  if (average->ComputeMean()) {
    // meanUp
    ioData.registerIOFamily("Time-averaged primitive vars", "/meanSolution", average->GetMeanUp(), false,
                            config.GetRestartMean());
    ioData.registerIOVar("/meanSolution", "meanDens", 0);
    ioData.registerIOVar("/meanSolution", "mean-u", 1);
    ioData.registerIOVar("/meanSolution", "mean-v", 2);
    if (dim == 3) {
      ioData.registerIOVar("/meanSolution", "mean-w", 3);
      ioData.registerIOVar("/meanSolution", "mean-E", 4);
    } else {
      ioData.registerIOVar("/meanSolution", "mean-p", 3);
    }
    if (eqSystem == NS_PASSIVE) ioData.registerIOVar("/meanSolution", "mean-Z", num_equation - 1);

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

  fluxClass = new Fluxes(mixture, eqSystem, num_equation, dim);

  alpha = 0.5;
  isSBP = config.isSBP();

  // Create Riemann Solver
  rsolver = new RiemannSolver(num_equation, mixture, eqSystem, fluxClass, config.RoeRiemannSolver());

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

  A = new DGNonLinearForm(vfes, gradUpfes, gradUp, bcIntegrator, intRules, dim, num_equation, mixture, gpuArrays,
                          maxIntPoints, maxDofs);
  if (local_attr.Size() > 0) A->AddBdrFaceIntegrator(bcIntegrator);

  {
    bool useLinearIntegration = false;
    //    if( basisType==1 && intRuleType==1 ) useLinearIntegration = true;

    faceIntegrator = new FaceIntegrator(intRules, rsolver, fluxClass, vfes, useLinearIntegration, dim, num_equation,
                                        gradUp, gradUpfes, max_char_speed);
  }
  A->AddInteriorFaceIntegrator(faceIntegrator);
  if (isSBP) {
    SBPoperator = new SBPintegrator(mixture, fluxClass, intRules, dim, num_equation, alpha);
    A->AddDomainIntegrator(SBPoperator);
  }

  Aflux = new MixedBilinearForm(dfes, fes);
  domainIntegrator = new DomainIntegrator(fluxClass, intRules, intRuleType, dim, num_equation);
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

  rhsOperator = new RHSoperator(iter, dim, num_equation, order, eqSystem, max_char_speed, intRules, intRuleType,
                                fluxClass, mixture, vfes, gpuArrays, maxIntPoints, maxDofs, A, Aflux, mesh,
                                spaceVaryViscMult, Up, gradUp, gradUpfes, gradUp_A, bcIntegrator, isSBP, alpha, config);

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
  temperature = new ParGridFunction(fes, Up->HostReadWrite() + (1 + dim) * fes->GetNDofs());
  
  // this variable is purely for visualization
  press = new ParGridFunction(fes);
  
  passiveScalar = NULL;
  if (eqSystem == NS_PASSIVE)
    passiveScalar = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());

  // define solution parameters for i/o
  ioData.registerIOFamily("Solution state variables", "/solution", U);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "rho-u", 1);
  ioData.registerIOVar("/solution", "rho-v", 2);
  if (dim == 3) {
    ioData.registerIOVar("/solution", "rho-w", 3);
    ioData.registerIOVar("/solution", "rho-E", 4);
  } else {
    ioData.registerIOVar("/solution", "rho-E", 3);
  }
  if (eqSystem == NS_PASSIVE) ioData.registerIOVar("/solution", "rho-Z", num_equation - 1);

  // compute factor to multiply viscosity when this option is active
  spaceVaryViscMult = NULL;
  ParGridFunction coordsDof(dfes);
  mesh->GetNodes(coordsDof);
  if (config.GetLinearVaryingData().viscRatio > 0.) {
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
  paraviewColl->RegisterField("temp",temperature);
  paraviewColl->RegisterField("press", press);
  if (eqSystem == NS_PASSIVE) paraviewColl->RegisterField("passiveScalar", passiveScalar);

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

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    uniformInitialConditions();
#ifdef _MASA_
    initMasaHandler("exact", dim, config.GetEquationSystem(), config.GetViscMult());
    void (*initialConditionFunction)(const Vector &, double, Vector &);
    initialConditionFunction = &(this->MASA_exactSol);
    VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    u0.SetTime(0.0);
    U->ProjectCoefficient(u0);
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
  
  // update pressure grid function
  mixture->UpdatePressureGridFunction(press,Up);
}

void M2ulPhyS::Iterate() {
#ifdef HAVE_GRVY
  const int iterQuery = 100;
  double tlast = grvy_timer_elapsed_global();
#endif

#ifdef _MASA_
  // instantiate function for exact solution
  void (*exactSolnFunctionDen)(const Vector &, double, Vector &);
  void (*exactSolnFunctionVel)(const Vector &, double, Vector &);
  void (*exactSolnFunctionPre)(const Vector &, double, Vector &);
  exactSolnFunctionDen = &(this->MASA_exactDen);
  exactSolnFunctionVel = &(this->MASA_exactVel);
  exactSolnFunctionPre = &(this->MASA_exactPre);
  VectorFunctionCoefficient DenMMS(1, exactSolnFunctionDen);
  VectorFunctionCoefficient VelMMS(dim, exactSolnFunctionVel);
  VectorFunctionCoefficient PreMMS(1, exactSolnFunctionPre);

  // and dump error before we take any steps
  DenMMS.SetTime(time);
  VelMMS.SetTime(time);
  PreMMS.SetTime(time);
  const double errorDen = dens->ComputeLpError(2, DenMMS);
  const double errorVel = vel->ComputeLpError(2, VelMMS);
  const double errorPre = press->ComputeLpError(2, PreMMS);
  if (mpi.Root())
    cout << "time step: " << iter << ", physical time " << time << "s"
         << ", Dens. error: " << errorDen << " Vel. " << errorVel << " press. " << errorPre << endl;
#endif

  bool readyForRestart = false;

  // Integrate in time.
  while (iter < MaxIters) {
#ifdef HAVE_GRVY
    grvy_timer_begin(__func__);
    if ((iter % iterQuery) == 0) {
      if (mpi.Root()) {
        double timePerIter = (grvy_timer_elapsed_global() - tlast) / iterQuery;
        grvy_printf(ginfo, "Iteration = %i: wall clock time/iter = %.3f (secs)\n", iter, timePerIter);
        tlast = grvy_timer_elapsed_global();
      }
      writeHistoryFile();
    }
#endif

    timeIntegrator->Step(*U, time, dt);

    Check_NAN();

    if (!config.isTimeStepConstant()) {
      double dt_local = CFL * hmin / max_char_speed / static_cast<double>(dim);
      MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
    }

    iter++;

    const int vis_steps = config.GetNumItersOutput();
    if (iter % vis_steps == 0) {
#ifdef _MASA_
      rhsOperator->updatePrimitives(*U);
      DenMMS.SetTime(time);
      VelMMS.SetTime(time);
      PreMMS.SetTime(time);
      const double errorDen = dens->ComputeLpError(2, DenMMS);
      const double errorVel = vel->ComputeLpError(2, VelMMS);
      const double errorPre = press->ComputeLpError(2, PreMMS);
      if (mpi.Root())
        cout << "time step: " << iter << ", physical time " << time << "s"
             << ", Dens. error: " << errorDen << " Vel. " << errorVel << " press. " << errorPre << endl;
#else
      if (mpi.Root()) cout << "time step: " << iter << ", physical time " << time << "s" << endl;
#endif

      if (iter != MaxIters) {
        restart_files_hdf5("write");

        auto hUp = Up->HostRead();
        mixture->UpdatePressureGridFunction(press,Up);
        
        paraviewColl->SetCycle(iter);
        paraviewColl->SetTime(time);
        paraviewColl->Save();
        auto dUp = Up->ReadWrite();  // sets memory to GPU

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

#ifdef HAVE_GRVY
    grvy_timer_end(__func__);
#endif
  }  // <-- end main timestep iteration loop

  if (iter == MaxIters) {
    // write_restart_files();
    restart_files_hdf5("write");

    auto hUp = Up->HostRead();
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();

    average->write_meanANDrms_restart_files(iter, time);

#ifndef _MASA_
    // If _MASA_ is defined, this is handled above
    void (*initialConditionFunction)(const Vector &, Vector &);
    initialConditionFunction = &(this->InitialConditionEulerVortex);

    VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    const double error = U->ComputeLpError(2, u0);
    if (mpi.Root()) cout << "Solution error: " << error << endl;
#endif

    if (mpi.Root()) cout << "Final timestep iteration = " << MaxIters << endl;
  }

  return;
}

#ifdef _MASA_
void M2ulPhyS::MASA_exactSol(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1], x[2], tin);  // rho
  y(1) = y[0] * MASA::masa_eval_exact_u<double>(x[0], x[1], x[2], tin);
  y(2) = y[0] * MASA::masa_eval_exact_v<double>(x[0], x[1], x[2], tin);
  y(3) = y[0] * MASA::masa_eval_exact_w<double>(x[0], x[1], x[2], tin);
  y(4) = MASA::masa_eval_exact_p<double>(x[0], x[1], x[2], tin) / (gamma - 1.);

  double k = 0.;
  for (int d = 0; d < x.Size(); d++) k += y[1 + d] * y[1 + d];
  k *= 0.5 / y[0];
  y[4] += k;
}

void M2ulPhyS::MASA_exactDen(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1], x[2], tin);  // rho
}

void M2ulPhyS::MASA_exactVel(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_u<double>(x[0], x[1], x[2], tin);
  y(1) = MASA::masa_eval_exact_v<double>(x[0], x[1], x[2], tin);
  y(2) = MASA::masa_eval_exact_w<double>(x[0], x[1], x[2], tin);
}

void M2ulPhyS::MASA_exactPre(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_p<double>(x[0], x[1], x[2], tin);
}
#endif

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

void M2ulPhyS::uniformInitialConditions() {
  double *data = U->HostWrite();
  double *dataUp = Up->HostWrite();
  double *dataGradUp = gradUp->HostWrite();

  int dof = vfes->GetNDofs();
  double *inputRhoRhoVp = config.GetConstantInitialCondition();

  DryAir *eqState = new DryAir(dim,num_equation);

  const double gamma = eqState->GetSpecificHeatRatio();
  const double rhoE =
      inputRhoRhoVp[4] / (gamma - 1.) + 0.5 *
                                            (inputRhoRhoVp[1] * inputRhoRhoVp[1] + inputRhoRhoVp[2] * inputRhoRhoVp[2] +
                                             inputRhoRhoVp[3] * inputRhoRhoVp[3]) /
                                            inputRhoRhoVp[0];

  Vector state;
  state.UseDevice(false);
  state.SetSize(num_equation);
  Vector Upi;
  Upi.UseDevice(false);
  Upi.SetSize(num_equation);
  
  for (int i = 0; i < dof; i++) {
    data[i] = inputRhoRhoVp[0];
    data[i + dof] = inputRhoRhoVp[1];
    data[i + 2 * dof] = inputRhoRhoVp[2];
    if (dim == 3) data[i + 3 * dof] = inputRhoRhoVp[3];
    data[i + (1 + dim) * dof] = rhoE;
    if (eqSystem == NS_PASSIVE) data[i + (num_equation - 1) * dof] = 0.;
    
    for(int eq=0;eq<num_equation;eq++) state(eq) = data[i + eq*dof];
    eqState->GetPrimitivesFromConservatives(state,Upi);
    for(int eq=0;eq<num_equation;eq++) dataUp[i+eq*dof] = Upi[eq];

//     dataUp[i] = data[i];
//     dataUp[i + dof] = data[i + dof] / data[i];
//     dataUp[i + 2 * dof] = data[i + 2 * dof] / data[i];
//     if (dim == 3) dataUp[i + 3 * dof] = data[i + 3 * dof] / data[i];
//     dataUp[i + (1 + dim) * dof] = inputRhoRhoVp[4];
//     if (eqSystem == NS_PASSIVE) dataUp[i + (num_equation - 1) * dof] = 0.;

    for (int d = 0; d < dim; d++) {
      for (int eq = 0; eq < num_equation; eq++) {
        dataGradUp[i + eq * dof + d * num_equation * dof] = 0.;
      }
    }
  }

  delete eqState;
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
    double gamma = mixture->GetSpecificHeatRatio();
    int dof = vfes->GetNDofs();
    if (lines != dof * num_equation) {
      cout << "# of lines in files does not match domain size" << endl;
    } else {
      dof = vfes->GetNDofs();
      for (int i = 0; i < dof; i++) {
        double p = dataUp[i + (1 + dim) * dof];
        double r = dataUp[i];
        Array<double> vel(dim);
        for (int d = 0; d < dim; d++) vel[d] = dataUp[i + (d + 1) * dof];
        double k = 0.;
        for (int d = 0; d < dim; d++) k += vel[d] * vel[d];
        double rE = p / (gamma - 1.) + 0.5 * r * k;
        dataU[i] = r;
        for (int d = 0; d < dim; d++) dataU[i + (d + 1) * dof] = r * vel[d];
        dataU[i + (1 + dim) * dof] = rE;
        if (eqSystem == NS_PASSIVE) dataU[i + (num_equation - 1) * dof] = r * dataUp[i + (num_equation - 1) * dof];
      }
    }
  }

  // load data to GPU
  auto dUp = Up->ReadWrite();
  auto dU = U->ReadWrite();
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
    for (int eq = 0; eq < num_equation; eq++) {
      if (std::isnan(dataU[i + eq * dof])) {
        // thereIsNan = true;
        cout << "NaN at node: " << i << " partition: " << mpi.WorldRank() << endl;
        local_print++;
        // MPI_Abort(MPI_COMM_WORLD,1);
      }
    }
  }
#endif
  int print;
  MPI_Allreduce(&local_print, &print, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (print > 0) {
    auto hUp = Up->HostRead();  // get GPU data
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
