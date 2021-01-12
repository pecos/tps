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
   const char *inputFile = "../data/periodic-square.mesh";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&inputFile, "-run", "--runFile",
                  "Name of the input file with run options.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   
//    for (int lev = 0; lev < 1; lev++)
//    {
//       mesh.UniformRefinement();
//    }
//    mesh.PrintCharacteristics();
   
   string inputFileName(inputFile);
   
   M2ulPhyS solver( inputFileName );



   // Visualize the density
   socketstream sout;


   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();
   
   solver.Iterate();

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;


   return 0;
}
