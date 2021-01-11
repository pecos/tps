/* 
 * Implementation of the class Problem
 */

#include "M2ulPhyS.hpp"

M2ulPhyS::M2ulPhyS(string &inputFileName)
{
  config.readInputFile(inputFileName);
  
  initVariables();
  
  // This example depends on this ordering of the space.
  MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
  
}


void M2ulPhyS::initVariables()
{
  
  //mesh = new Mesh(config.GetMeshFileName(),1,1);
  mesh = new Mesh(config.GetMeshFileName());
  mesh->PrintCharacteristics();
  dim = mesh->Dimension();
  
  eqSystem = config.GetEquationSystem();
  
  order = config.GetSolutionOrder();
  
  MaxIters = config.GetNumIters();
  
  max_char_speed = 0.;
  
  switch(eqSystem)
  {
    case EULER:
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
    //fec  = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    fec  = new H1_FECollection(order, dim, BasisType::GaussLobatto);
  }
  
  // FE Spaces
  fes  = new FiniteElementSpace(mesh, fec);
  dfes = new FiniteElementSpace(mesh, fec, dim, Ordering::byNODES);
  vfes = new FiniteElementSpace(mesh, fec, num_equation, Ordering::byNODES);
  
  eqState = new EquationOfState(config.GetWorkingFluid());
  
  fluxClass = new Fluxes(eqState);
  
  alpha = 0.5;
  isSBP = config.isSBP();
  
  // Create Riemann Solver
  rsolver = new RiemannSolver(num_equation, eqState);
  
  A = new NonlinearForm(vfes);
  faceIntegrator = new FaceIntegrator(intRules, rsolver, dim, 
                                      num_equation, max_char_speed);
  A->AddInteriorFaceIntegrator( faceIntegrator );
  if( isSBP )
  {
    SBPoperator = new SBPintegrator(eqState,fluxClass,intRules,
                                    dim,num_equation, alpha);
    A->AddDomainIntegrator( SBPoperator );
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
  
  cout << "Number of unknowns: " << vfes->GetVSize() << endl;
  cout << "Number of DoF:      " << vfes->GetNDofs() << endl;
  
  rhsOperator = new RHSoperator(dim,
                                eqSystem,
                                max_char_speed,
                                intRules,intRuleType,
                                fluxClass,
                                eqState,
                                vfes,
                                A, 
                                Aflux,
                                isSBP, alpha
                               );
  
  initSolutionAndVisualizationVectors();
  projectInitialSolution();
  
  time = 0.;
  CFL = config.GetCFLNumber();
//   switch(order)
//   {
//     case 4: CFL = 0.12; break;
//     default: break;
//   }
  rhsOperator->SetTime(time);
  timeIntegrator->Init( *rhsOperator );
  
  // Determine the minimum element size.
   hmin = 0.0;
   {
      hmin = mesh->GetElementSize(0, 1);
      for (int i = 1; i < mesh->GetNE(); i++)
      {
         hmin = min(mesh->GetElementSize(i, 1), hmin);
      }
   }

  // estimate initial dt
  {
    // Find a safe dt, using a temporary vector. Calling Mult() computes the
    // maximum char speed at all quadrature points on all faces.
    Vector z(A->Width());
    A->Mult(*sol, z);
    dt = CFL * hmin / max_char_speed / (2*order+1);
  }
}


M2ulPhyS::~M2ulPhyS()
{
  delete u_block;
  delete offsets;
  
  delete sol;
  
  delete rhsOperator;
  delete domainIntegrator;
  //delete Aflux; // fails to delete (follow)
  if( isSBP ) delete SBPoperator;
  delete faceIntegrator;
  //delete A; // fails to delete (follow)
  delete rsolver;
  delete fluxClass;
  delete eqState;
  delete vfes;
  delete dfes;
  delete fes;
  delete fec;
  delete intRules;
  
  delete mesh;
}

void M2ulPhyS::initSolutionAndVisualizationVectors()
{
  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) { (*offsets)[k] = k * vfes->GetNDofs(); }
  u_block = new BlockVector(*offsets);
  
//   {
//     ofstream mesh_ofs("vortex.mesh");
//     mesh_ofs.precision(8);
//     mesh_ofs << mesh;
//   }
  
  sol = new GridFunction(vfes, u_block->GetData());

  // Momentum grid function on dfes for visualization.
  //GridFunction mom(dfes, u_block->GetData() + (*offsets)[1]);
}

void M2ulPhyS::projectInitialSolution()
{
  // Initialize the state.
  
  void (*initialConditionFunction)(const Vector&, Vector&);
  
  // particular case: Euler vortex
  {
    t_final = 5.*  2./17.46;
    initialConditionFunction = &(this->InitialConditionEulerVortex);
  }
  //initialConditionFunction = &(this->testInitialCondition);
  
  VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
  sol->ProjectCoefficient(u0);
}


void M2ulPhyS::Iterate()
{
  // Integrate in time.
  bool done = false;
  for (int ti = 0; !done; )
  {
    // adjusts time step to finish at t_final
    double dt_real = min(dt, t_final - time);

    timeIntegrator->Step(*sol, time, dt_real);
    //dt = CFL * hmin / max_char_speed / (2*order+1);
    dt = CFL * hmin / max_char_speed /(double)dim;
    ti++;

    const int vis_steps = 50;
    done = (time >= t_final - 1e-8*dt);
    if (done || ti % vis_steps == 0)
    {
        cout << "time step: " << ti << ", progress(%): " << 100.*time/t_final << endl;
    }
  }
  
  if (time == t_final)
   {
     void (*initialConditionFunction)(const Vector&, Vector&);
      initialConditionFunction = &(this->InitialConditionEulerVortex);
  
      VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
      const double error = sol->ComputeLpError(2, u0);
      cout << "Solution error: " << error << endl;
      
      string fileName(config.GetOutputName());
      fileName.append(".mesh");
      ofstream vtkmesh(fileName);
      mesh->Print(vtkmesh);
      vtkmesh.close();
      
      // 9. Save the final solution. This output can be viewed later using GLVis:
      //    "glvis -m vortex.mesh -g vortex-1-final.gf".
      for (int k = 0; k < num_equation; k++)
      {
          GridFunction uk(fes, u_block->GetBlock(k));
          ostringstream sol_name;
          sol_name << config.GetOutputName() <<"-" << k << "-final.gf";
          ofstream sol_ofs(sol_name.str().c_str());
          sol_ofs.precision(8);
          sol_ofs << uk;
      }
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

   y(0) = den_inf + x(0);
   y(1) = y(0);
   y(2) = 0;
   y(3) = pres_inf/(gamma-1.) + 0.5*y(1)*y(1)/y(0);
   cout<<y(3)<<endl;
   
   delete eqState;
}

