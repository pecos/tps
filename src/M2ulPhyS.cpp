/* 
 * Implementation of the class Problem
 */

#include "M2ulPhyS.hpp"

M2ulPhyS::M2ulPhyS(Mesh &_mesh,
                 int _order,
                 int solverType,
                 double _t_final,
                 Equations _eqSystem,
                 WorkingFluid _fluid): 
  dim(_mesh.Dimension()),
  order(_order),
  eqSystem(_eqSystem),
  mesh(_mesh),
  t_final(_t_final)
{
  
  // This example depends on this ordering of the space.
  MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
  
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
  intRuleType = 0;
  if( intRuleType == 0 )
  {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLegendre);
  }else if( intRuleType == 1 )
  {
    intRules = new IntegrationRules(0, Quadrature1D::GaussLobatto);
  }
  
  basisType = 0;
  if( basisType == 0 )
  {
    fec  = new DG_FECollection(order, dim, BasisType::GaussLegendre);
  }else if ( basisType == 1 )
  {
    fec  = new DG_FECollection(order, dim, BasisType::GaussLobatto);
  }
  
  // FE Spaces
  fes  = new FiniteElementSpace(&mesh, fec);
  dfes = new FiniteElementSpace(&mesh, fec, dim, Ordering::byNODES);
  vfes = new FiniteElementSpace(&mesh, fec, num_equation, Ordering::byNODES);
  
  eqState = new EquationOfState(_fluid);
  
  fluxClass = new Fluxes(eqState);
  
  // Create Riemann Solver
  rsolver = new RiemannSolver(num_equation, eqState);
  
  A = new NonlinearForm(vfes);
  faceIntegrator = new FaceIntegrator(intRules, rsolver, dim, num_equation, max_char_speed);
  A->AddInteriorFaceIntegrator( faceIntegrator );
  
  Aflux = new MixedBilinearForm(dfes, fes);
  domainIntegrator = new DomainIntegrator(intRules, dim, num_equation);
  Aflux->AddDomainIntegrator( domainIntegrator );
  Aflux->Assemble();
  //Me_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE())
  
  switch (solverType)
  {
    case 1: timeIntegrator = new ForwardEulerSolver; break;
    case 2: timeIntegrator = new RK2Solver(1.0); break;
    case 3: timeIntegrator = new RK3SSPSolver; break;
    case 4: timeIntegrator = new RK4Solver; break;
    case 6: timeIntegrator = new RK6Solver; break;
    default:
        cout << "Unknown ODE solver type: " << solverType << '\n';
  }
  
  cout << "Number of unknowns: " << vfes->GetVSize() << endl;
  cout << "Number of DoF:      " << vfes->GetNDofs() << endl;
  
  rhsOperator = new RHSoperator(dim,
                                eqSystem,
                                max_char_speed,
                                fluxClass,
                                eqState,
                                vfes,
                                A, 
                                Aflux);
  
  initSolutionAndVisualizationVectors();
  projectInitialSolution();
  
  time = 0.;
  CFL = 1.;
  switch(order)
  {
    case 3: CFL = 0.12; break;
    default: break;
  }
  rhsOperator->SetTime(time);
  timeIntegrator->Init( *rhsOperator );
  
  // Determine the minimum element size.
   hmin = 0.0;
   {
      hmin = mesh.GetElementSize(0, 1);
      for (int i = 1; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
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
  
  //**** DEBUG
  //dt = 0.001;
}

M2ulPhyS::~M2ulPhyS()
{
  delete sol;
  delete u_block;
  delete offsets;
  delete rhsOperator;
  delete domainIntegrator;
  //delete Aflux; // fails to delete (follow)
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
}

void M2ulPhyS::initSolutionAndVisualizationVectors()
{
  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) { (*offsets)[k] = k * vfes->GetNDofs(); }
  u_block = new BlockVector(*offsets);
  
  sol = new GridFunction(vfes, u_block->GetData());

  // Momentum grid function on dfes for visualization.
  //GridFunction mom(dfes, u_block->GetData() + (*offsets)[1]);
}

void M2ulPhyS::projectInitialSolution()
{
  // Initialize the state.
  
  void (*initialConditionFunction)(const Vector&, Vector&);
  initialConditionFunction = &(this->InitialConditionTest);
  
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
        cout << "time step: " << ti << ", time: " << time << endl;
//         if (visualization)
//         {
//           sout << "solution\n" << mesh << mom << flush;
//         }
    }
  }
  
  if (t_final == 2.0)
   {
     void (*initialConditionFunction)(const Vector&, Vector&);
      initialConditionFunction = &(this->InitialConditionTest);
  
      VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
      const double error = sol->ComputeLpError(2, u0);
      cout << "Solution error: " << error << endl;
   }
}


// Initial conditions for debug/test case
void M2ulPhyS::InitialConditionTest(const Vector& x, Vector& y)
{
  MFEM_ASSERT(x.Size() == 2, "");
  
  int problem = 2;
  EquationOfState *eqState = new EquationOfState();

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

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;
   
   const double specific_heat_ratio = eqState->GetSpecificHeatRatio();
   const double gas_constant = eqState->GetGasConstant();

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio) * (vel_inf / Minf) *
                           (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * eqState->GetGasConstant());

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / ( specific_heat_ratio - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
}

