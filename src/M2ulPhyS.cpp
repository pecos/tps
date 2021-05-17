/* 
 * Implementation of the class Problem
 */

#include "M2ulPhyS.hpp"
#include <sstream>

M2ulPhyS::M2ulPhyS(MPI_Session &_mpi,
                   string &inputFileName):
mpi(_mpi)
{
  groupsMPI = new MPI_Groups(&mpi);
  
  config.readInputFile(inputFileName);
  
  initVariables();
  
  groupsMPI->init();
  
  // This example depends on this ordering of the space.
  //MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
  
}


void M2ulPhyS::initVariables()
{
  loadFromAuxSol = config.RestartFromAux();
  auxOrder = config.RestartFromAux();
  if( loadFromAuxSol && auxOrder<1 )
  {
    cout<<"Option AUX_ORDER must be set when using RESTART_FROM_AUX"<<endl;
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  
  dumpAuxSol = false;
  if( auxOrder>0 ) dumpAuxSol =true;
  
  // check if a simulations is being restarted
  Mesh *tempmesh;
  if( config.GetRestartCycle()>0 )
  {
    // read serial mesh and partition. Hopefully the same
    // partitions will be created in the same order
    tempmesh = new Mesh(config.GetMeshFileName().c_str() );
    mesh = new ParMesh(MPI_COMM_WORLD,*tempmesh);
    tempmesh->Clear();
    
    // Paraview setup
    paraviewColl = new ParaViewDataCollection(config.GetOutputName(), mesh);
    paraviewColl->SetLevelsOfDetail( config.GetSolutionOrder() );
    paraviewColl->SetHighOrderOutput(true);
    paraviewColl->SetPrecision(8);
    
    if( mpi.Root() )
    {
      cout<<"================================================"<<endl;
      cout<<"| Restarting simulation at iteration "<<config.GetRestartCycle()<<endl;
      cout<<"================================================"<<endl;
    }
    
  }else
  {
    //remove previous solution
    if( mpi.Root() )
    {
      string command = "rm -r ";
      command.append( config.GetOutputName() );
      int err = system(command.c_str());
      if( err==1 )
      {
        cout<<"Error deleting previous data in "<<config.GetOutputName()<<endl;
      }
    }
    
    tempmesh = new Mesh(config.GetMeshFileName().c_str() );
    mesh = new ParMesh(MPI_COMM_WORLD,*tempmesh);
    tempmesh->Clear();
    
    // VisIt setup
//     visitColl = new VisItDataCollection(config.GetOutputName(), mesh);
//     visitColl->SetPrefixPath(config.GetOutputName());
//     visitColl->SetPrecision(8);
    
    // Paraview setup
    paraviewColl = new ParaViewDataCollection(config.GetOutputName(), mesh);
    paraviewColl->SetLevelsOfDetail( config.GetSolutionOrder() );
    paraviewColl->SetHighOrderOutput(true);
    paraviewColl->SetPrecision(8);
    
    time = 0.;
    iter = 0;
  }
  
  cout<<"Process "<<mpi.WorldRank()<<" # elems "<< mesh->GetNE()<<endl;
  
  dim = mesh->Dimension();
  
  refLength = config.GetReferenceLength();
  
  eqSystem = config.GetEquationSystem();
  
  eqState = new EquationOfState(config.GetWorkingFluid());
  eqState->setViscMult(config.GetViscMult() );
  
  order = config.GetSolutionOrder();
  
  MaxIters = config.GetNumIters();
  
  max_char_speed = 0.;
  
  switch(eqSystem)
  {
    case EULER:
      num_equation = 2 + dim;
      break;
    case NS:
      num_equation = 2 + dim;
      break;
    default:
      break;
  }
  
  // initialize basis type and integration rule
  intRuleType = config.GetIntegrationRule();
  if( intRuleType == 0 )
  {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLegendre);
  }else if( intRuleType == 1 )
  {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLobatto);
  }
  
  basisType = config.GetBasisType();
  if( basisType == 0 )
  {
    fec  = new DG_FECollection(order, dim, BasisType::GaussLegendre);
    //fec  = new H1_FECollection(order, dim, BasisType::GaussLegendre);
    if( loadFromAuxSol || dumpAuxSol ) 
      aux_fec = new DG_FECollection(1, dim, BasisType::GaussLegendre);
    
  }else if ( basisType == 1 )
  {
    // This basis type includes end-nodes
    fec  = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    //fec  = new H1_FECollection(order, dim, BasisType::GaussLobatto);
    if( loadFromAuxSol || dumpAuxSol ) 
      aux_fec = new DG_FECollection(1, dim, BasisType::GaussLobatto);
  }
  
  // FE Spaces
  fes  = new ParFiniteElementSpace(mesh, fec);
  dfes = new ParFiniteElementSpace(mesh, fec, dim, Ordering::byNODES);
  vfes = new ParFiniteElementSpace(mesh, fec, num_equation, Ordering::byNODES);
  if( loadFromAuxSol || dumpAuxSol ) aux_vfes = new ParFiniteElementSpace(mesh, 
                                                aux_fec, num_equation, Ordering::byNODES);
  gradUpfes = new ParFiniteElementSpace(mesh, fec, num_equation*dim, Ordering::byNODES);
  
  initSolutionAndVisualizationVectors();
  projectInitialSolution();
  
  average = new Averaging(Up,
                          mesh,
                          fec,
                          fes,
                          dfes,
                          vfes,
                          num_equation,
                          dim,
                          config,
                          groupsMPI);
  average->read_meanANDrms_restart_files();
  
  fluxClass = new Fluxes(eqState,
                         eqSystem,
                         num_equation,
                         dim);
  
  alpha = 0.5;
  isSBP = config.isSBP();
  
  // Create Riemann Solver
  rsolver = new RiemannSolver(num_equation, 
                              eqState, 
                              fluxClass,
                              config.RoeRiemannSolver() );
  
  A = new ParNonlinearForm(vfes);
  {
    bool useLinearIntegration = false;
    if( basisType==1 && intRuleType==1 ) useLinearIntegration = true;
    
    faceIntegrator = new FaceIntegrator(intRules, 
                                      rsolver,
                                      fluxClass, 
                                      vfes,
                                      useLinearIntegration,
                                      dim, 
                                      num_equation, 
                                      gradUp,
                                      gradUpfes,
                                      max_char_speed);
  }
  A->AddInteriorFaceIntegrator( faceIntegrator );
  if( isSBP )
  {
    SBPoperator = new SBPintegrator(eqState,fluxClass,intRules,
                                    dim,num_equation, alpha);
    A->AddDomainIntegrator( SBPoperator );
  }
  
  // Boundary attributes in present partition
  Array<int> local_attr;
  getAttributesInPartition(local_attr);
  
  bcIntegrator = NULL;
  if( local_attr.Size()>0 )
  {
    bcIntegrator = new BCintegrator(groupsMPI,
                                    mesh,
                                    vfes,
                                    intRules,
                                    rsolver, 
                                    dt,
                                    eqState,
                                    fluxClass,
                                    Up,
                                    gradUp,
                                    dim,
                                    num_equation,
                                    max_char_speed,
                                    config, 
                                    local_attr );
    A->AddBdrFaceIntegrator( bcIntegrator );
  }
  
  Aflux = new MixedBilinearForm(dfes, fes);
  domainIntegrator = new DomainIntegrator(fluxClass,intRules, intRuleType, 
                                          dim, num_equation);
  Aflux->AddDomainIntegrator( domainIntegrator );
  Aflux->Assemble();
  
  switch (config.GetTimeIntegratorType() )
  {
    case 1: timeIntegrator = new ForwardEulerSolver; break;
    case 2: timeIntegrator = new RK2Solver(1.0); break;
    case 3: timeIntegrator = new RK3SSPSolver; break;
    case 4: timeIntegrator = new RK4Solver; break;
    case 6: timeIntegrator = new RK6Solver; break;
    default:
        cout << "Unknown ODE solver type: " << config.GetTimeIntegratorType() << '\n';
  }
  
  gradUp_A = new ParNonlinearForm(gradUpfes);
  gradUp_A->AddInteriorFaceIntegrator(
      new GradFaceIntegrator(intRules, dim, num_equation) );
  
  rhsOperator = new RHSoperator(time,
                                dim,
                                num_equation,
                                order,
                                eqSystem,
                                max_char_speed,
                                intRules,
                                intRuleType,
                                fluxClass,
                                eqState,
                                vfes,
                                A, 
                                Aflux,
                                mesh,
                                Up,
                                gradUp,
                                gradUpfes,
                                gradUp_A,
                                bcIntegrator,
                                isSBP, 
                                alpha,
                                config );
  
  CFL = config.GetCFLNumber();
  rhsOperator->SetTime(time);
  timeIntegrator->Init( *rhsOperator );
  
  // Determine the minimum element size.
   {
      double local_hmin = mesh->GetElementSize(0, 1);
      for (int i = 1; i < mesh->GetNE(); i++)
      {
        if(sqrt(mesh->GetElementVolume(i))<1e-3) cout<<sqrt(mesh->GetElementVolume(i))<<endl;
        local_hmin = min(mesh->GetElementSize(i, 1), local_hmin);
        //local_hmin = min(sqrt(mesh->GetElementVolume(i)), local_hmin);
      }
      MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
   }

  // estimate initial dt
  gradUp->ExchangeFaceNbrData();
    
  if( config.GetRestartCycle()==0 ) initialTimeStep();
  if( mpi.Root() ) cout<<"Initial time-step: "<<dt<<"s"<<endl;
  
  //t_final = MaxIters*dt;
}


M2ulPhyS::~M2ulPhyS()
{
  if( loadFromAuxSol || dumpAuxSol )
  {
    delete aux_Up;
    delete aux_Up_data;
    delete aux_vfes;
    delete aux_fec;
  }
  
  delete gradUp;
  
  delete u_block;
  delete up_block;
  delete offsets;
  
  delete U;
  delete Up;
  
  delete average;
  
  //delete rhsOperator;
  //delete domainIntegrator;
  delete Aflux; // fails to delete (follow this)
  if( isSBP ) delete SBPoperator;
  //delete faceIntegrator;
  delete A; // fails to delete (follow this)
  
  delete timeIntegrator;
  // delete inlet/outlet integrators
  
  delete rsolver;
  delete fluxClass;
  delete eqState;
  delete gradUpfes;
  delete vfes;
  delete dfes;
  delete fes;
  delete fec;
  delete intRules;
  
  delete paraviewColl;
  //delete mesh;
  
  delete groupsMPI;
}


void M2ulPhyS::getAttributesInPartition(Array<int>& local_attr)
{
  local_attr.DeleteAll();
  for(int bel=0;bel<vfes->GetNBE(); bel++)
  {
    int attr = vfes->GetBdrAttribute(bel);
    bool attrInArray = false;
    for(int i=0;i<local_attr.Size();i++)
    {
      if( local_attr[i]==attr ) attrInArray = true;
    }
    if( !attrInArray ) local_attr.Append( attr );
  }
}


void M2ulPhyS::initSolutionAndVisualizationVectors()
{
  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) { (*offsets)[k] = k * vfes->GetNDofs(); }
  u_block = new BlockVector(*offsets);
  up_block = new BlockVector(*offsets);
  
  //gradUp.SetSize(num_equation*dim*vfes->GetNDofs());
  gradUp = new ParGridFunction(gradUpfes);
  
  U  = new ParGridFunction(vfes, u_block->GetData());
  Up = new ParGridFunction(vfes, up_block->GetData());
  
  if( loadFromAuxSol || dumpAuxSol )
  {
    aux_Up_data = new double[num_equation*aux_vfes->GetNDofs()];
    aux_Up = new ParGridFunction(aux_vfes, aux_Up_data);
  }
  
  dens = new ParGridFunction(fes, Up->GetData());
  vel = new ParGridFunction(dfes, Up->GetData()+fes->GetNDofs() );
  press = new ParGridFunction(fes,
                Up->GetData()+(num_equation-1)*fes->GetNDofs() );
  
  paraviewColl->SetCycle(0);
  paraviewColl->SetTime(0.);
  
  paraviewColl->RegisterField("dens",dens);
  paraviewColl->RegisterField("vel",vel);
  paraviewColl->RegisterField("press",press);
  
  paraviewColl->SetOwnData(true);
  //paraviewColl->Save();
}

void M2ulPhyS::projectInitialSolution()
{
  // Initialize the state.
  
  // particular case: Euler vortex
//   {
//     void (*initialConditionFunction)(const Vector&, Vector&);
//     t_final = 5.*  2./17.46;
//     initialConditionFunction = &(this->InitialConditionEulerVortex);
      //initialConditionFunction = &(this->testInitialCondition);
    
//     VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
//     U->ProjectCoefficient(u0);
//   }

  if( config.GetRestartCycle()==0 && !loadFromAuxSol )
  {
    uniformInitialConditions();
#ifdef _MASA_
    initMasaHandler("exact",dim);
    void (*initialConditionFunction)(const Vector&, Vector&);
    initialConditionFunction = &(this->MASA_initialCondition);
    VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    //cout<<"x y z rho"<<endl;
    U->ProjectCoefficient(u0);
#endif
  }else
  {
    read_restart_files();
    
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
  }
  
  initGradUp();
}


void M2ulPhyS::Iterate()
{
  // Integrate in time.
  while( iter<MaxIters )
  {
    timeIntegrator->Step(*U, time, dt);
  
    Check_NAN();
    
    if( !config.isTimeStepConstant() )
    {
      double dt_local = CFL * hmin / max_char_speed /(double)dim;
      MPI_Allreduce(&dt_local, &dt,
                        1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
    }
    
    iter++;

    const int vis_steps = config.GetNumItersOutput();
    if( iter % vis_steps == 0 )
    {
      if(mpi.Root()) cout <<"time step: "<<iter<<", physical time "<<time<<"s"<< endl;
      
      write_restart_files();
      
      paraviewColl->SetCycle(iter);
      paraviewColl->SetTime(time);
      paraviewColl->Save();
      
      average->write_meanANDrms_restart_files();
    }
    
    average->addSampleMean(iter);
  }
  
  
  if( iter==MaxIters )
  {
    
    write_restart_files();
      
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
    
    average->write_meanANDrms_restart_files();
      
    void (*initialConditionFunction)(const Vector&, Vector&);
    initialConditionFunction = &(this->InitialConditionEulerVortex);

    VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    const double error = U->ComputeLpError(2, u0);
    cout << "Solution error: " << error << endl;
    
  }
}

#ifdef _MASA_
void M2ulPhyS::MASA_initialCondition(const Vector& x, Vector& y)
{
  MFEM_ASSERT(x.Size() == 3, "");

  EquationOfState eqState( DRY_AIR );
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) =      MASA::masa_eval_exact_rho<double>(x[0],x[1],x[2]); // rho
  y(1) = y[0]*MASA::masa_eval_exact_u<double>(x[0],x[1],x[2]);
  y(2) = y[0]*MASA::masa_eval_exact_v<double>(x[0],x[1],x[2]);
  y(3) = y[0]*MASA::masa_eval_exact_w<double>(x[0],x[1],x[2]);
  y(4) = MASA::masa_eval_exact_p<double>(x[0],x[1],x[2])/(gamma-1.);

  double k = 0.;
  for(int d=0;d<x.Size();d++) k += y[1+d]*y[1+d];
  k *= 0.5/y[0];
  y[4] += k;
}
#endif

// Initial conditions for debug/test case
void M2ulPhyS::InitialConditionEulerVortex(const Vector& x, Vector& y)
{
  MFEM_ASSERT(x.Size() == 2, "");
  int equations = 4;
  if(x.Size()==3) equations = 5;
  
  int problem = 1;
  EquationOfState *eqState = new EquationOfState(DRY_AIR);
  const double gamma = eqState->GetSpecificHeatRatio();
  const double Rg = eqState->GetGasConstant();

   double radius = 0, Minf = 0, beta = 0;
   if (problem == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
      
      radius = 0.5;
      Minf = 0.1;
   }
   else if (problem == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - fast vortex, 2 - slow vortex");
   }

   int numVortices = 3;
   Vector xc(numVortices),yc(numVortices);
   yc = 0.;
   for(int i=0;i<numVortices;i++)
   {
     xc[i] = 2.*M_PI/double(numVortices+1);
     xc[i]+= double(i)*2.*M_PI/double(numVortices);
   }

   const double Tt = 300.;
   const double Pt = 102200;
   
   const double funcGamma = 1.+0.5*(gamma-1.)*Minf*Minf;
   
   const double temp_inf = Tt/funcGamma;
   const double pres_inf = Pt*pow(funcGamma,gamma/(gamma-1.));
   const double vel_inf = Minf*sqrt(gamma*Rg*temp_inf);
   const double den_inf = pres_inf/(Rg*temp_inf);
   

   double r2rad = 0.0;

   const double shrinv1 = 1.0 / ( gamma - 1.);

   double velX = 0.;
   double velY = 0.;
   double temp = 0.;
   for(int i=0;i<numVortices;i++)
   {
    r2rad  = (x(0)-xc[i])*(x(0)-xc[i]);
    r2rad += (x(1)-yc[i])*(x(1)-yc[i]);
    r2rad /= radius*radius;
    velX -= beta*(x(1)-yc[i])/radius*exp( -0.5*r2rad);
    velY += beta*(x(0)-xc[i])/radius*exp( -0.5*r2rad);
    temp += exp(-r2rad);
  }
   
   velX = vel_inf*(1 - velX);
   velY = vel_inf*velY;
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = Rg * gamma * shrinv1;
   temp = temp_inf -0.5*(vel_inf*beta)*(vel_inf*beta)/specific_heat*temp;

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * Rg * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   if(x.Size()==3) y(3) = 0.;
   y(equations-1) = den * energy;
   
   delete eqState;
}


// Initial conditions for debug/test case
void M2ulPhyS::testInitialCondition(const Vector& x, Vector& y)
{
   EquationOfState *eqState = new EquationOfState(DRY_AIR);

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.3;
   const double Minf = 0.5;
   
   const double gamma = eqState->GetSpecificHeatRatio();
   //const double Rgas = eqState->GetGasConstant();

   const double pres_inf = (den_inf / gamma) * (vel_inf / Minf) *
                           (vel_inf / Minf);

   y(0) = den_inf + 0.5*(x(0)+3) +0.25*(x(1)+3);
   y(1) = y(0);
   y(2) = 0;
   y(3) = (pres_inf+x(0)+0.2*x(1))/(gamma-1.) + 0.5*y(1)*y(1)/y(0);
   
   delete eqState;
}

void M2ulPhyS::uniformInitialConditions()
{
  double *data = U->GetData();
  double *dataUp = Up->GetData();
  double *dataGradUp = gradUp->GetData();
  
  int dof = vfes->GetNDofs();
  double *inputRhoRhoVp = config.GetConstantInitialCondition();
  
  EquationOfState *eqState = new EquationOfState(DRY_AIR);
  const double gamma = eqState->GetSpecificHeatRatio();
  const double rhoE = inputRhoRhoVp[4]/(gamma-1.)+
                    0.5*(inputRhoRhoVp[1]*inputRhoRhoVp[1] +
                         inputRhoRhoVp[2]*inputRhoRhoVp[2] +
                         inputRhoRhoVp[3]*inputRhoRhoVp[3]
                    )/inputRhoRhoVp[0];
  
  for(int i=0; i<dof; i++)
  {
    data[i       ] = inputRhoRhoVp[0];
    data[i +  dof] = inputRhoRhoVp[1];
    data[i +2*dof] = inputRhoRhoVp[2];
    if(dim==3) data[i +3*dof] = inputRhoRhoVp[3];
    data[i +(num_equation-1)*dof] = rhoE;
    
    dataUp[i       ] = data[i];
    dataUp[i +  dof] = data[i+  dof]/data[i];
    dataUp[i +2*dof] = data[i+2*dof]/data[i];
    if(dim==3) dataUp[i +3*dof] = data[i+3*dof]/data[i];
    dataUp[i +(num_equation-1)*dof] = inputRhoRhoVp[4];
    
    for(int d=0;d<dim;d++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {
        dataGradUp[i+eq*dof + d*num_equation*dof] = 0.;
      }
    }
  }
  
  delete eqState;
}

void M2ulPhyS::initGradUp()
{
  double *dataGradUp = gradUp->GetData();
  int dof = vfes->GetNDofs();
  
  for(int i=0; i<dof; i++)
  {
    for(int d=0;d<dim;d++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {
        dataGradUp[i+eq*dof + d*num_equation*dof] = 0.;
      }
    }
  }
}


void M2ulPhyS::interpolateOrder2Aux()
{
  double *data_Up = Up->GetData();
  double *data_aux = aux_Up->GetData();

  // fill out coords
  for(int el=0;el<vfes->GetNE();el++) // lin_vfes and vfes should have the same num of elems
  {
    // get aux. points
    const IntegrationRule aux_ir = aux_vfes->GetFE(el)->GetNodes();
    ElementTransformation *aux_Tr = aux_vfes->GetElementTransformation(el);
    
    DenseMatrix aux_pos;
    aux_Tr->Transform(aux_ir,aux_pos);
    
    Array<int> aux_dofs;
    aux_vfes->GetElementVDofs(el, aux_dofs);
    
    Array<int> p_dofs;
    vfes->GetElementVDofs(el, p_dofs);
    
    // Order p element transformation
    ElementTransformation *p_Tr = vfes->GetElementTransformation(el);
    const FiniteElement *p_fe = vfes->GetFE(el);
    
    for(int aux_n=0;aux_n<aux_ir.GetNPoints();aux_n++)
    {
      IntegrationPoint transformed_IP;
      Vector coords;
      coords.SetSize(dim);
      for(int d=0;d<dim;d++) coords[d] = aux_pos(aux_n,d);
      
      p_Tr->TransformBack(coords, transformed_IP);
      
      p_Tr->SetIntPoint( &transformed_IP);
      
      Vector shape;
      shape.SetSize( p_fe->GetDof() );
      
      p_fe->CalcShape(transformed_IP, shape);
      
      // interpolate values
      int aux_index = aux_dofs[aux_n];
      const int aux_Ndofs = aux_vfes->GetNDofs();
      const int p_Ndofs = vfes->GetNDofs();
      for(int eq=0;eq<num_equation;eq++)
      {
        data_aux[aux_index + eq*aux_Ndofs] = 0.;
        for(int i=0;i<shape.Size();i++)
        {
          int indexHO = p_dofs[i];
          data_aux[aux_index + eq*aux_Ndofs] += 
              shape[i]*data_Up[indexHO+eq*p_Ndofs];
        }
      }
    }
  }
}

void M2ulPhyS::interpolateAux2Order()
{
  double *data_Up = Up->GetData();
  double *data_aux = aux_Up->GetData();

  // fill out coords
  for(int el=0;el<vfes->GetNE();el++) // aux_vfes and vfes should have the same num of elems
  {
    // get order p points
    const IntegrationRule ir = vfes->GetFE(el)->GetNodes();
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    
    DenseMatrix pos;
    Tr->Transform(ir,pos);
    
    Array<int> aux_dofs;
    aux_vfes->GetElementVDofs(el, aux_dofs);
    
    Array<int> p_dofs;
    vfes->GetElementVDofs(el, p_dofs);
    
    // linear element transformation
    ElementTransformation *aux_Tr = aux_vfes->GetElementTransformation(el);
    const FiniteElement *aux_fe = aux_vfes->GetFE(el);
    
    // loop over HO nodes
    for(int n=0;n<ir.GetNPoints();n++)
    {
      IntegrationPoint transformed_IP;
      Vector coords;
      coords.SetSize(dim);
      for(int d=0;d<dim;d++) coords[d] = pos(n,d);
      
      aux_Tr->TransformBack(coords, transformed_IP);
      
      aux_Tr->SetIntPoint( &transformed_IP);
      
      Vector shape;
      shape.SetSize( aux_fe->GetDof() );
      
      aux_fe->CalcShape(transformed_IP, shape);
      
      // interpolate values
      const int p_index = p_dofs[n];
      const int aux_Ndofs = aux_vfes->GetNDofs();
      const int p_Ndofs = vfes->GetNDofs();
      for(int eq=0;eq<num_equation;eq++)
      {
        data_Up[p_index + eq*p_Ndofs] = 0.;
        for(int i=0;i<shape.Size();i++)
        {
          int aux_index = aux_dofs[i];
          data_Up[p_index + eq*p_Ndofs] += 
              shape[i]*data_aux[aux_index+eq*aux_Ndofs];
        }
      }
    }
  }
}



void M2ulPhyS::write_restart_files()
{
  string serialName = "restart_p";
  serialName.append( to_string(order) );
  serialName.append("_");
  serialName.append( config.GetOutputName() );
  serialName.append( ".sol" );
    
  string fileName = groupsMPI->getParallelName( serialName );
  ofstream file( fileName, std::ofstream::trunc );
  file.precision(8);
  
  // write cycle and time
  file<<iter<<" "<<time<<" "<<dt<<endl;
  
  double *data = Up->GetData();
  int dof = vfes->GetNDofs();
  
  for(int i=0;i<dof*num_equation;i++)
  {
    file << data[i] <<endl;
  }
  
  file.close();
  
  if( dumpAuxSol )
  {
    interpolateOrder2Aux();
    serialName = "restart_p";
    serialName.append( to_string(auxOrder) );
    serialName.append("_");
    serialName.append( config.GetOutputName() );
    serialName.append( ".sol" );
      
    fileName = groupsMPI->getParallelName( serialName );
    file.open( fileName, std::ofstream::trunc );
    file.precision(8);
    
    // write cycle and time
    file<<iter<<" "<<time<<" "<<dt<<endl;
    
    double *data = aux_Up->GetData();
    int dof = aux_vfes->GetNDofs();
    
    for(int i=0;i<dof*num_equation;i++)
    {
      file << data[i] <<endl;
    }
    
    file.close();
  }
}


void M2ulPhyS::read_restart_files()
{
  string serialName = "restart_p";
  if( loadFromAuxSol )
  {
    serialName.append( to_string(auxOrder) );
  }else
  {
    serialName.append( to_string(order) );
  }
  serialName.append("_");
  serialName.append( config.GetOutputName() );
  serialName.append( ".sol" );
  
  string fileName = groupsMPI->getParallelName( serialName );
  ifstream file( fileName );
  
  if( !file.is_open() )
  {
    cout<< "Could not open file \""<<fileName<<"\""<<endl;
    return;
  }else
  {
    double *data;
    if( loadFromAuxSol )
    {
      data = aux_Up->GetData();
    }else
    {
      data = Up->GetData();
    }
    
    string line;
    // read time and iters
    {
      getline(file,line);
      istringstream ss(line);
      string word;
      ss >> word;
      iter = stoi( word );
      
      ss >> word;
      time = stof( word );
      
      ss >> word;
      dt = stof( word );
    }
  
    int lines = 0;
    while( getline(file,line) )
    {
      istringstream ss(line);
      string word;
      ss >> word;
      
      data[lines] = stof( word );
      lines++;
    }
    file.close();
    
    if( loadFromAuxSol ) interpolateAux2Order();
    
    double *dataUp = Up->GetData();
    
    // fill out U
    double *dataU = U->GetData();
    double gamma = eqState->GetSpecificHeatRatio();
    int dof = vfes->GetNDofs();
    if( loadFromAuxSol ) dof = aux_vfes->GetNDofs();
    if( lines!=dof*num_equation )
    {
      cout<<"# of lines in files does not match domain size"<<endl;
    }else
    {
      dof = vfes->GetNDofs();
      for(int i=0;i<dof;i++)
      {
        double p = dataUp[i + (num_equation-1)*dof];
        double r = dataUp[i];
        Array<double> vel(dim);
        for(int d=0;d<dim;d++) vel[d] = dataUp[i+(d+1)*dof];
        double k = 0.;
        for(int d=0;d<dim;d++) k += vel[d]*vel[d];
        double rE = p/(gamma-1.) + 0.5*r*k;
        dataU[i] = r;
        for(int d=0;d<dim;d++) dataU[i+(d+1)*dof] = r*vel[d];
        dataU[i+(num_equation-1)*dof] = rE;
      }
    }
  }
}


void M2ulPhyS::Check_NAN()
{
  double *dataU = U->GetData();
  int dof = vfes->GetNDofs();
  
  //bool thereIsNan = false;
  int local_print = 0;
  for(int i=0;i<dof;i++)
  {
    for(int eq=0;eq<num_equation;eq++)
    {
      if( std::isnan(dataU[i+eq*dof]) )
      {
        //thereIsNan = true;
        //cout<<"NaN at node: "<<i<<" partition: "<<mpi.WorldRank()<<endl;
        local_print++;
        //MPI_Abort(MPI_COMM_WORLD,1);
      }
    }
  }
  int print;
  MPI_Allreduce(&local_print, &print,
                       1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if( print>0 )
  {
    paraviewColl->Save();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
  }
}



void M2ulPhyS::initialTimeStep()
{
  double *dataU = U->GetData();
  int dof = vfes->GetNDofs();
  
  for(int n=0;n<dof;n++)
  {
    Vector state(num_equation);
    for(int eq=0;eq<num_equation;eq++) state[eq] = dataU[n+eq*dof];
    double iC = eqState->ComputeMaxCharSpeed(state,dim);
    if( iC>max_char_speed ) max_char_speed = iC;
  }
  
  double partition_C = max_char_speed;
  MPI_Allreduce(&partition_C, &max_char_speed,
                       1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  dt = CFL * hmin / max_char_speed /(double)dim;
}
