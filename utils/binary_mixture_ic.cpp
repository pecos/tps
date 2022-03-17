/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  MPI_Session mpi(argc, argv);

  // Set the method's default parameters.
  const char *input_file = "coarse.run";

  // Parse command-line options.
  OptionsParser args(argc, argv);
  args.AddOption(&input_file, "-r1", "--runFile1", "runFile");

  args.Parse();
  if (!args.Good())
    {
      args.PrintUsage(cout);
      return 1;
    }
  args.PrintOptions(cout);

  // Instantiate M2ulPhyS classes for *both* the coarse and fine

  // NB: the M2ulPhyS ctor calls M2ulPhyS::initVariables, which reads
  // the restart files, assuming that RESTART_CYCLE is set in the
  // input file.  So, we require that RESTART_CYCLE is set in the
  // *source* run file....
  string inputFileName(input_file);
  M2ulPhyS srcField( mpi, inputFileName );
  RunConfiguration& srcConfig = srcField.GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::TEST_BINARY_AIR);
  assert(~srcConfig.GetRestartCycle());

  // Get meshes
  ParMesh* mesh_1 = srcField.GetMesh();

  const int dim = mesh_1->Dimension();

  // GSLIB only works in 2 and 3 dimensions
  assert(dim>1);

  if (mesh_1->GetNodes() == NULL) { mesh_1->SetCurvature(1); }
  const int mesh_poly_deg =
    mesh_1->GetNodes()->FESpace()->GetElementOrder(0);
  cout << "Source mesh curvature: "
       << mesh_1->GetNodes()->OwnFEC()->Name() << endl;

  // 2) Set up source field
  FiniteElementCollection *src_fec = NULL;
  ParFiniteElementSpace *src_fes = NULL;
  ParGridFunction *src_state = NULL;

  src_fec = srcField.GetFEC();
  src_fes = srcField.GetFESpace();
  src_state = srcField.GetSolutionGF();

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

  double *dataU = src_state->GetData();
  int NDof = src_fes->GetNDofs();

  double gamma = 1.4;
  double T0 = 293.0;
  double p0 = 101300;
  double gas_constant = 287.058;

  double rho = p0 / gas_constant / T0;
  double rhoU = 0.0;
  double rhoE = 101300. / (gamma - 1.) + 0.5 * rhoU * rhoU / rho;
  double pi = atan(1.0) * 4.0;
  for (int i = 0; i < NDof; i++) {
    // std::cout << "Grid point " << i << ": ";
    // for (int d = 0; d < dim; d++) {
    //   std::cout << coordinates[i + d * NDof] << ", ";
    // }
    // std::cout << std::endl;

    dataU[i] = rho;
    for (int d = 0; d < dim; d++) {
      dataU[i + (d+1) * NDof] = rhoU;
    }
    dataU[i + (dim+1) * NDof] = rhoE;
    dataU[i + (dim+2) * NDof] = rho * ( 0.5 + 0.5 * sin( 2.0 * pi * coordinates[i + 0 * NDof] / 10.0 ) );
    // dataU[i + (dim+3) * NDof] = 0.5 - 0.5 * sin( 2.0 * pi * coordinates[i + 0 * NDof] / 10.0 );
    // if (coordinates[i + 0 * NDof]>0) {
    //   dataU[i + (dim+2) * NDof] = 0.0;
    //   dataU[i + (dim+3) * NDof] = 1.0;
    // } else {
    //   dataU[i + (dim+2) * NDof] = 1.0;
    //   dataU[i + (dim+3) * NDof] = 0.0;
    // }
  }

  RHSoperator rhsOperator = srcField.getRHSoperator();
  rhsOperator.updatePrimitives(*src_state);

  // Write restart files
  srcField.writeHDF5();

  return 0;
}
