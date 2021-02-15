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
  }else if ( basisType == 1 )
  {
    // This basis type includes end-nodes
    fec  = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    //fec  = new H1_FECollection(order, dim, BasisType::GaussLobatto);
  }
  
  // FE Spaces
  fes  = new ParFiniteElementSpace(mesh, fec);
  dfes = new ParFiniteElementSpace(mesh, fec, dim, Ordering::byNODES);
  vfes = new ParFiniteElementSpace(mesh, fec, num_equation, Ordering::byNODES);
  gradUpfes = new ParFiniteElementSpace(mesh, fec, num_equation*dim, Ordering::byNODES);
  
  initSolutionAndVisualizationVectors();
  projectInitialSolution();
  
  fluxClass = new Fluxes(eqState,
                         eqSystem,
                         num_equation,
                         dim);
  
  alpha = 0.5;
  isSBP = config.isSBP();
  
  // Create Riemann Solver
  rsolver = new RiemannSolver(num_equation, eqState);
  
  A = new ParNonlinearForm(vfes);
  faceIntegrator = new FaceIntegrator(intRules, 
                                      rsolver,
                                      fluxClass, 
                                      vfes,
                                      dim, 
                                      num_equation, 
                                      gradUp,
                                      gradUpfes,
                                      max_char_speed);
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
  
  rhsOperator = new RHSoperator(dim,
                                num_equation,
                                eqSystem,
                                max_char_speed,
                                intRules,
                                intRuleType,
                                fluxClass,
                                eqState,
                                vfes,
                                A, 
                                Aflux,
                                Up,
                                gradUp,
                                gradUpfes,
                                gradUp_A,
                                bcIntegrator,
                                isSBP, 
                                alpha );
  
  CFL = config.GetCFLNumber();
  rhsOperator->SetTime(time);
  timeIntegrator->Init( *rhsOperator );
  
  // Determine the minimum element size.
   hmin = 0.0;
   {
      //double local_hmin = mesh->GetElementSize(0, 1);
     double local_hmin = sqrt(mesh->GetElementVolume(0));
      for (int i = 1; i < mesh->GetNE(); i++)
      {
        if(sqrt(mesh->GetElementVolume(i))<1e-3) cout<<sqrt(mesh->GetElementVolume(i))<<endl;
        //local_hmin = min(mesh->GetElementSize(i, 1), local_hmin);
        local_hmin = min(sqrt(mesh->GetElementVolume(i)), local_hmin);
      }
      MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
   }

  // estimate initial dt
  {
    gradUp->ExchangeFaceNbrData();
    // Find a safe dt, using a temporary vector. Calling Mult() computes the
    // maximum char speed at all quadrature points on all faces.
    Vector z(A->Width());
    A->Mult(*U, z);
    dt = CFL * hmin / max_char_speed / (2*order+1);
    t_final = MaxIters*dt;
  }
}


M2ulPhyS::~M2ulPhyS()
{
  delete paraviewColl;
  //delete visitColl;
  
  delete gradUp;
  
  delete u_block;
  delete up_block;
  delete offsets;
  
  delete U;
  delete Up;
  
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
    //void (*initialConditionFunction)(const Vector&, Vector&);
//     t_final = 5.*  2./17.46;
//     initialConditionFunction = &(this->InitialConditionEulerVortex);
      //initialConditionFunction = &(this->testInitialCondition);
    
//     VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
//     U->ProjectCoefficient(u0);
//   }

  if( config.GetRestartCycle()==0 )
  {
    uniformInitialConditions();
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
  bool done = false;
  while( !done )
  {
    // adjusts time step to finish at t_final
    double dt_real;
    double dt_local = min(dt, t_final - time);
    
    MPI_Allreduce(&dt_local, &dt_real,
                       1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());

    timeIntegrator->Step(*U, time, dt_real);
  
    Check_NAN();
    
    dt = CFL * hmin / max_char_speed /(double)dim;
    iter++;

    const int vis_steps = config.GetNumItersOutput();
    done = (time >= t_final - 1e-8*dt);
    if (done || iter % vis_steps == 0)
    {
        if(mpi.Root()) cout << "time step: " << iter << ", progress(%): " << 100.*time/t_final << endl;
        
        { // DEBUG ONLY!
//           GridFunction uk(fes, u_block->GetBlock(3));
//           ostringstream sol_name;
//           sol_name << config.GetOutputName() <<"-" << 2 << "-final.gf";
//           ofstream sol_ofs(sol_name.str().c_str());
//           sol_ofs.precision(8);
//           sol_ofs << uk;
//           exit(0);
          
          write_restart_files();
          
          paraviewColl->SetCycle(iter);
          paraviewColl->SetTime(time);
          paraviewColl->Save();
          
          //vel->SaveVTK(file,"vel",3);
          //press->SaveVTK(file,"press",3);
          
//           visitColl->SetCycle(iter);
//           visitColl->SetTime(time);
//           visitColl->Save();
        }
    }
  }
  
  if(time == t_final)
  {
    void (*initialConditionFunction)(const Vector&, Vector&);
    initialConditionFunction = &(this->InitialConditionEulerVortex);

    VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
    const double error = U->ComputeLpError(2, u0);
    cout << "Solution error: " << error << endl;
    
//       string fileName(config.GetOutputName());
//       fileName.append(".mesh");
//       ofstream vtkmesh(fileName);
//       mesh->Print(vtkmesh);
//       vtkmesh.close();
  }
}


// Initial conditions for debug/test case
void M2ulPhyS::InitialConditionEulerVortex(const Vector& x, Vector& y)
{
  MFEM_ASSERT(x.Size() == 2, "");
  
  int problem = 2;
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

   const double xc = 0.0, yc = 0.0;

   const double Tt = 300.;
   const double Pt = 102200;
   
   const double funcGamma = 1.+0.5*(gamma-1.)*Minf*Minf;
   
   const double temp_inf = Tt/funcGamma;
   const double pres_inf = Pt*pow(funcGamma,gamma/(gamma-1.));
   const double vel_inf = Minf*sqrt(gamma*Rg*temp_inf);
   const double den_inf = pres_inf/(Rg*temp_inf);
   

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / ( gamma - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = Rg * gamma * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * Rg * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
   
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


void M2ulPhyS::write_restart_files()
{
  string fileName = groupsMPI->getParallelName( "restart.sol" );
  ofstream file( fileName, std::ofstream::trunc );
  
  // write cycle and time
  file<<iter<<" "<<time<<endl;
  
  double *data = Up->GetData();
  int dof = vfes->GetNDofs();
  
  for(int i=0;i<dof*num_equation;i++)
  {
    file << data[i] <<endl;
  }
  
  file.close();
}

void M2ulPhyS::read_restart_files()
{
  string fileName = groupsMPI->getParallelName( "restart.sol" );
  ifstream file( fileName );
  
  if( !file.is_open() )
  {
    cout<< "Could not open file \""<<fileName<<"\""<<endl;
    return;
  }else
  {
    double *dataUp = Up->GetData();
    
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
    }
  
    int lines = 0;
    while( getline(file,line) )
    {
      istringstream ss(line);
      string word;
      ss >> word;
      
      dataUp[lines] = stof( word );
      lines++;
    }
    file.close();
    
    // fill out U
    double *dataU = U->GetData();
    double gamma = eqState->GetSpecificHeatRatio();
    int dof = vfes->GetNDofs();
    if( lines!=dof*num_equation )
    {
      cout<<"# of lines in files does not match domain size"<<endl;
    }else
    {
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
      if( isnan(dataU[i+eq*dof]) )
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
