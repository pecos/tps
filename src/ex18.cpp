//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -r 2 -o 1 -s 3
//       ex18 -p 1 -r 1 -o 3 -s 4
//       ex18 -p 1 -r 0 -o 5 -s 6
//       ex18 -p 2 -r 1 -o 1 -s 3
//       ex18 -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "M2ulPhyS.hpp"


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
//    const int dim = mesh.Dimension();
// 
//    MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");
   
   M2ulPhyS solver( mesh,
                    order,
                    ode_solver_type,
                    t_final,
                    EULER,
                    DRY_AIR);

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      //mesh.UniformRefinement();
   }
   mesh.PrintCharacteristics();

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.

   // 6. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   

   // Output the initial solution.
   {
//       ofstream mesh_ofs("vortex.mesh");
//       mesh_ofs.precision(precision);
//       mesh_ofs << mesh;
// 
//       for (int k = 0; k < num_equation; k++)
//       {
//          GridFunction uk(&fes, u_block.GetBlock(k));
//          ostringstream sol_name;
//          sol_name << "vortex-" << k << "-init.gf";
//          ofstream sol_ofs(sol_name.str().c_str());
//          sol_ofs.precision(precision);
//          sol_ofs << uk;
//       }
   }

   // 7. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   //FE_Evolution euler(vfes, A, Aflux.SpMat());

   // Visualize the density
   socketstream sout;
//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
// 
//       sout.open(vishost, visport);
//       if (!sout)
//       {
//          cout << "Unable to connect to GLVis server at "
//               << vishost << ':' << visport << endl;
//          visualization = false;
//          cout << "GLVis visualization disabled.\n";
//       }
//       else
//       {
//          sout.precision(precision);
//          sout << "solution\n" << mesh << mom;
//          sout << "pause\n";
//          sout << flush;
//          cout << "GLVis visualization paused."
//               << " Press space (in the GLVis window) to resume it.\n";
//       }
//    }


   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();
   
   solver.Iterate();

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-1-final.gf".
//    for (int k = 0; k < num_equation; k++)
//    {
//       GridFunction uk(&fes, u_block.GetBlock(k));
//       ostringstream sol_name;
//       sol_name << "vortex-" << k << "-final.gf";
//       ofstream sol_ofs(sol_name.str().c_str());
//       sol_ofs.precision(precision);
//       sol_ofs << uk;
//    }

   // 10. Compute the L2 solution error summed for all components.
//    if (t_final == 2.0)
//    {
//       const double error = sol.ComputeLpError(2, u0);
//       cout << "Solution error: " << error << endl;
//    }


   return 0;
}
