#include <limits>
#include <math.h>

#include <mfem/general/forall.hpp>

#include "../src/thermo_chem_base.hpp"
#include "../src/split_flow_base.hpp"
#include "../src/tomboulides.hpp"
#include "../src/tps_mfem_wrap.hpp"

using namespace mfem;

/// Used to set the velocity IC (and to check error)
void vel_exact_tg(const Vector &x, double t, Vector &u) {
  const double nu = 1.0;
  const double F = std::exp(-2 * nu * t);

  u(0) =  F * std::sin(x[0]) * std::cos(x[1]);
  u(1) = -F * std::cos(x[0]) * std::sin(x[1]);
}

void updateTimeIntegrationCoefficients(int step, double *dthist, timeCoefficients &time_coeff) {
  using std::pow;

  // Maximum BDF order to use at current time step
  // step + 1 <= order <= max_bdf_order
  int bdf_order = std::min(step + 1, 3);

  // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
  double rho1 = 0.0;

  // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
  double rho2 = 0.0;

  rho1 = dthist[0] / dthist[1];

  if (bdf_order == 3) {
    rho2 = dthist[1] / dthist[2];
  }

  if (step == 0 && bdf_order == 1) {
    time_coeff.bd0 = 1.0;
    time_coeff.bd1 = -1.0;
    time_coeff.bd2 = 0.0;
    time_coeff.bd3 = 0.0;
    time_coeff.ab1 = 1.0;
    time_coeff.ab2 = 0.0;
    time_coeff.ab3 = 0.0;
  } else if (step >= 1 && bdf_order == 2) {
    time_coeff.bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
    time_coeff.bd1 = -(1.0 + rho1);
    time_coeff.bd2 = pow(rho1, 2.0) / (1.0 + rho1);
    time_coeff.bd3 = 0.0;
    time_coeff.ab1 = 1.0 + rho1;
    time_coeff.ab2 = -rho1;
    time_coeff.ab3 = 0.0;
  } else if (step >= 2 && bdf_order == 3) {
    time_coeff.bd0 = 1.0 + rho1 / (1.0 + rho1) + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
    time_coeff.bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
    time_coeff.bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
    time_coeff.bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1)) / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
    time_coeff.ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
    time_coeff.ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
    time_coeff.ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
  }
}



/**
 * @brief Test that the interface design for the low Mach split time
 * integrators builds and runs.  For these purposes, we solve a 2D
 * Taylor-Green problem.  Specifically, the domain is
 *
 * \Omega = (0, 2\pi) \times (0, 2\pi).
 *
 * The solution is
 *
 * u(x,y) = F(t) sin(x) cos(y),  v(x,y) = -F(t) cos(x) sin(y),
 *
 * where
 *
 * F(t) = exp(-2 \nu t),
 *
 * and \nu is the kinematic viscosity.
 */
int main(int argc, char *argv[]) {
  // Initialize mpi
  Mpi::Init(argc, argv);
  Hypre::Init();

  int ref_levels = 2;
  int order = 3;
  int problem = 0; // 0 = TG, 1 = lid-driven cavity
  double dt = -1.0;
  double t_final = 1.0;


  //---------------------------------------------------------------------
  // Parse command line options
  OptionsParser args(argc, argv);
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order", "Order (degree) of the finite elements.");
  args.AddOption(&problem, "-p", "--problem", "0 = Taylor-Green, 1 = lid-driven cavity.");
  args.AddOption(&dt, "-dt", "--time-step", "Time step.");
  args.AddOption(&t_final, "-tf", "--final-time", "Final time.");

  args.Parse();
  if (!args.Good()) {
    if (Mpi::Root()) {
      args.PrintUsage(mfem::out);
    }
    return 1;
  }
  if (Mpi::Root()) {
    args.PrintOptions(mfem::out);
  }

  // Create the mesh
  Mesh *mesh = new Mesh("./meshes/inline-quad.mesh");

  // Make it periodic
  Mesh periodic_mesh;
  ParMesh *pmesh;
  if (problem == 0) {
    // problem 0 is Taylor-Green on (0,2\pi)x(0,2\pi)
    Vector x_translation({1.0, 0.0});
    Vector y_translation({0.0, 1.0});
    std::vector<Vector> translations = {x_translation, y_translation};

    Mesh periodic_mesh = Mesh::MakePeriodic(*mesh,mesh->CreatePeriodicVertexMapping(translations));

    // Scale domain to (0,L)x(0,L)
    periodic_mesh.EnsureNodes();
    GridFunction *nodes = periodic_mesh.GetNodes();
    *nodes *= (2 * M_PI);

    // Refine the periodic mesh
    for (int i = 0; i < ref_levels; ++i) {
      periodic_mesh.UniformRefinement();
    }

    // And finally, decompose it
    pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);

  } else if (problem == 1) {
    // problem 1 is a lid-driven cavity on (0,1)x(0,1)
    for (int i = 0; i < ref_levels; ++i) {
      mesh->UniformRefinement();
    }
    pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  } else {
    std::cout << "Unknown problem requested.  Exiting." << std::endl;
    return 1;
  }
  delete mesh;

  // Get minimum element size
  double hmin;
  double local_hmin = 1.0e18;
  for (int i = 0; i < pmesh->GetNE(); i++) {
    local_hmin = std::min(pmesh->GetElementSize(i, 1), local_hmin);
  }
  MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

  // polynomial orders
  const int sorder = order;
  const int vorder = order;
  const int porder = order;

  // time step coefficients (1st order)
  timeCoefficients time_coeff;
  time_coeff.time = 0.0;
  if (dt > 0.0) {
    time_coeff.dt = dt;
  } else {
    time_coeff.dt = 0.5 * hmin / vorder; // max velocity magn is 1
  }

  double dthist[3];
  dthist[0] = dthist[1] = dthist[2] = time_coeff.dt;
  updateTimeIntegrationCoefficients(0, dthist, time_coeff);

  // Density and viscosity to use
  const double rho = 1.0;
  const double mu = 1.0;

  // Set up the flow and thermo classes
  FlowBase *flow;
  ThermoChemModelBase *thermo;

  // Instantiate model classes
  flow = new Tomboulides(pmesh, vorder, porder, time_coeff);
  thermo = new ConstantPropertyThermoChem(pmesh, sorder, rho, mu);

  // Initialize -- after this, solution fields have been allocated
  flow->initializeSelf();
  thermo->initializeSelf();

  // Get the velocity field and set the IC
  ParGridFunction *u_gf = flow->getCurrentVelocity();
  VectorFunctionCoefficient u_excoeff(2, vel_exact_tg);
  u_excoeff.SetTime(0.0);

  // If TG problem set the IC
  if (problem == 0) {
    u_gf->ProjectCoefficient(u_excoeff);
  }

  if (problem == 1) {
    Vector zero(2);
    zero = 0.0;

    Vector onex(2);
    onex[0] = 1.0;

    Array<int> top_attr(pmesh->bdr_attributes.Max());
    top_attr = 0;
    top_attr[2] = 1;

    Tomboulides *tflow = dynamic_cast<Tomboulides*>(flow);

    tflow->addVelDirichletBC(onex, top_attr);

    Array<int> rest_attr(pmesh->bdr_attributes.Max());
    rest_attr = 1;
    rest_attr[2] = 0;
    tflow->addVelDirichletBC(zero, rest_attr);

    return 1;
  }

#if 0
  // Set above to 1 to ensure that assert catches uninitialized interface
  flow->initializeOperators();
#endif

  // Exchange interface information
  flow->initializeFromThermoChem(&thermo->interface);
  thermo->initializeFromFlow(&flow->interface);

  // And now we can finish initializing the flow
  flow->initializeOperators();

  // Write IC to paraview
  ParaViewDataCollection *pd = nullptr;
  pd = new ParaViewDataCollection("tomboulides_tg_res", pmesh);
  pd->SetPrefixPath("./");
  pd->RegisterField("velocity", u_gf);
  // pd->RegisterField("pressure", p_gf);
  pd->SetLevelsOfDetail(vorder);
  pd->SetDataFormat(VTKFormat::BINARY);
  pd->SetHighOrderOutput(true);
  pd->SetCycle(0);
  pd->SetTime(time_coeff.time);
  pd->Save();

  // Time march to t = 1.0
  int iter = 0;
  while (time_coeff.time < t_final) {
    if (Mpi::Root()) {
      std::cout << "Step " << iter << ": time = " << time_coeff.time << std::endl;
    }
    // Take a time step
    thermo->step();  // Thermo does nothing here
    flow->step();
    time_coeff.time += time_coeff.dt;
    iter++;

    // update dt history
    dthist[2] = dthist[1];
    dthist[1] = dthist[0];
    dthist[0] = time_coeff.dt;

    // update coefficients (dt is constant here)
    updateTimeIntegrationCoefficients(iter, dthist, time_coeff);
  }
  if (Mpi::Root()) {
    std::cout << "Final step " << iter << ": time = " << time_coeff.time << std::endl;
  }

  u_excoeff.SetTime(time_coeff.time);
  const double err_u = u_gf->ComputeL2Error(u_excoeff);
  if (Mpi::Root()) {
    std::cout << "Velocity error at final time = " << err_u << std::endl;
  }


  // Write some visualization files
  pd->SetCycle(1);
  pd->SetTime(time_coeff.time);
  pd->Save();

  // clean up
  delete pd;
  delete thermo;
  delete flow;
  delete pmesh;

  return 0;
}
