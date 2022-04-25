/* A utility to transfer tps solutions from one mesh to another */

// #include "M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  MPI_Session mpi(argc, argv);

  // 1. Parse command line options
  const char *baseMeshFile = "example.mesh";
  const char *outputMeshFile = "output-example.mesh";
  int order = 1;
  int numRefine = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&baseMeshFile, "-b", "--base-mesh", "Base mesh file to use.");
  args.AddOption(&outputMeshFile, "-r", "--refined-mesh", "Refined mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.AddOption(&numRefine, "-n", "--number", "Number of refinements");
  // args.ParseCheck();

  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);
  std::cout << "output filename: " << outputMeshFile << std::endl;

  // 3. Read the serial mesh from the given mesh file.
  Mesh serial_mesh(baseMeshFile);

  // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
  //    this mesh once in parallel to increase the resolution.
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear(); // the serial mesh is no longer needed
  for (int n = 0; n < numRefine; n++) {
    mesh.UniformRefinement();
    mesh.FinalizeTopology();
    if (n<numRefine-1) mesh.Finalize(true);
  }
  mesh.Finalize();

  mesh.Save(outputMeshFile);

  return 0;
}
