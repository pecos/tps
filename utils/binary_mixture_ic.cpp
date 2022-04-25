/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  TPS::Tps tps(argc, argv);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), tps.getInputFilename(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==USER_DEFINED);
  assert(~srcConfig.GetRestartCycle());

  GasMixture *mixture = srcField->getMixture();
  int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  ArgonMinimalTransport transport = ArgonMinimalTransport(mixture, srcConfig);

  // Get meshes
  ParMesh* mesh_1 = srcField->GetMesh();

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

  src_fec = srcField->GetFEC();
  src_fes = srcField->GetFESpace();
  src_state = srcField->GetSolutionGF();

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

  double *dataU = src_state->GetData();
  int NDof = src_fes->GetNDofs();

  std::string basepath("utils/binary_initial_condition");

  double T0, p0, u0, Lx, Ly;
  double kx, ky;
  tps.getRequiredInput((basepath + "/temperature").c_str(), T0);
  tps.getRequiredInput((basepath + "/pressure").c_str(), p0);
  tps.getRequiredInput((basepath + "/velocity/u").c_str(), u0);
  tps.getRequiredInput((basepath + "/domain_length/x").c_str(), Lx);
  tps.getRequiredInput((basepath + "/domain_length/y").c_str(), Ly);
  tps.getRequiredInput((basepath + "/wavenumber/x").c_str(), kx);
  tps.getRequiredInput((basepath + "/wavenumber/y").c_str(), ky);
  grvy_printf(GRVY_INFO, "\n Temperature: %.8E\n", T0);
  grvy_printf(GRVY_INFO, "\n Pressure: %.8E\n", p0);
  grvy_printf(GRVY_INFO, "\n X velocity: %.8E\n", u0);
  grvy_printf(GRVY_INFO, "\n X domain length: %.8E\n", Lx);
  grvy_printf(GRVY_INFO, "\n Y domain length: %.8E\n", Ly);
  grvy_printf(GRVY_INFO, "\n Wave number - x: %.4E\n", kx);
  grvy_printf(GRVY_INFO, "\n Wave number - y: %.4E\n", ky);

  double time;
  tps.getRequiredInput((basepath + "/reference_solution/time").c_str(), time);
  grvy_printf(GRVY_INFO, "\n Reference solution is at time = : %.8E\n", time);
  std::string refFileName;
  tps.getInput((basepath + "/reference_solution/filename").c_str(), refFileName, std::string("argonMinimal.binary.ref.h5"));
  grvy_printf(GRVY_INFO, "\n Reference solution filename : %s\n", refFileName.c_str());

  double nTotal = p0 / UNIVERSALGASCONSTANT / T0;
  double rho = nTotal * mixture->GetGasParams(numSpecies - 1, GasParams::SPECIES_MW);
  double rhoU = rho * u0;
  double rhoE = nTotal * mixture->getMolarCV(numSpecies - 1) * T0 + 0.5 * rhoU * rhoU / rho;

  // double rho = p0 / gas_constant / T0;
  // double rhoU = rho * u0;
  // double rhoE = 101300. / (gamma - 1.) + 0.5 * rhoU * rhoU / rho;
  double pi = atan(1.0) * 4.0;
  for (int i = 0; i < NDof; i++) {
    double Y = 0.5 + 0.45 * cos( 2.0 * pi * kx * coordinates[i + 0 * NDof] / Lx )
                          * cos( 2.0 * pi * ky * coordinates[i + 1 * NDof] / Ly );
    // std::cout << "Grid point " << i << ": ";
    // for (int d = 0; d < dim; d++) {
    //   std::cout << coordinates[i + d * NDof] << ", ";
    // }
    // std::cout << std::endl;

    dataU[i] = rho;
    dataU[i + 1 * NDof] = rhoU;
    for (int d = 1; d < dim; d++) {
      dataU[i + (d + 1) * NDof] = 0.0;
    }
    dataU[i + (dim + 1) * NDof] = rhoE;
    dataU[i + (dim + 2) * NDof] = rho * Y;
    dataU[i + (dim + 2 + numSpecies - 2) * NDof] = 0.0;
    // dataU[i + (dim+3) * NDof] = 0.5 - 0.5 * sin( 2.0 * pi * coordinates[i + 0 * NDof] / 10.0 );
    // if (coordinates[i + 0 * NDof]>0) {
    //   dataU[i + (dim+2) * NDof] = 0.0;
    //   dataU[i + (dim+3) * NDof] = 1.0;
    // } else {
    //   dataU[i + (dim+2) * NDof] = 1.0;
    //   dataU[i + (dim+3) * NDof] = 0.0;
    // }
  }

  RHSoperator rhsOperator = srcField->getRHSoperator();
  rhsOperator.updatePrimitives(*src_state);

  // Write restart files
  srcField->writeHDF5();

  Vector conservedState(num_equation);
  int idx = 0;
  for (int eq = 0; eq < num_equation; eq++) {
    conservedState(eq) = dataU[idx + eq * NDof];
  }

  Vector diffusivity;
  transport.computeMixtureAverageDiffusivity(conservedState, diffusivity);
  double Dia = diffusivity(transport.getIonIndex());
  grvy_printf(GRVY_INFO, "\n Ar-Ar+ binary diffusivity : %.8E\n", Dia);

  double decay = exp(- (4.0 * pi * pi * (kx * kx / Lx / Lx + ky * ky / Ly / Ly)) * Dia * time);
  double distance = u0 * time;
  grvy_printf(GRVY_INFO, "\n Decay : %.8E\n", decay);
  grvy_printf(GRVY_INFO, "\n Convection distance : %.8E\n", distance);

  // Set up reference solution (only for doubly-periodic diffusion case)
  for (int i = 0; i < NDof; i++) {
    double Y = 0.5 + 0.45 * decay * cos( 2.0 * pi * kx * (coordinates[i + 0 * NDof] - distance) / Lx )
                                  * cos( 2.0 * pi * ky * coordinates[i + 1 * NDof] / Ly );
    // std::cout << "Grid point " << i << ": ";
    // for (int d = 0; d < dim; d++) {
    //   std::cout << coordinates[i + d * NDof] << ", ";
    // }
    // std::cout << std::endl;

    dataU[i] = rho;
    dataU[i + 1 * NDof] = rhoU;
    for (int d = 1; d < dim; d++) {
      dataU[i + (d + 1) * NDof] = 0.0;
    }
    dataU[i + (dim + 1) * NDof] = rhoE;
    dataU[i + (dim + 2) * NDof] = rho * Y;
    dataU[i + (dim + 2 + numSpecies - 2) * NDof] = 0.0;
  }

  rhsOperator.updatePrimitives(*src_state);

  // Write restart files
  srcField->writeHDF5(refFileName);

  return 0;
}
