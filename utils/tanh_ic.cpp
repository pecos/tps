/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getInputFilename(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==USER_DEFINED);
  assert(!srcConfig.GetRestartCycle());

  GasMixture *mixture = srcField->getMixture();
  //int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  // Get meshes
  ParMesh* mesh_1 = srcField->getMesh();

  const int dim = mesh_1->Dimension();

  // GSLIB only works in 2 and 3 dimensions
  assert(dim>1);

  if (mesh_1->GetNodes() == NULL) { mesh_1->SetCurvature(1); }
  // const int mesh_poly_deg =
  //   mesh_1->GetNodes()->FESpace()->GetElementOrder(0);
  cout << "Source mesh curvature: "
       << mesh_1->GetNodes()->OwnFEC()->Name() << endl;

  // 2) Set up source field
  const FiniteElementCollection *src_fec = NULL;
  ParFiniteElementSpace *src_fes = NULL;
  ParGridFunction *src_state = NULL;

  src_fec = srcField->getFEC();
  src_fes = srcField->getFESpace();
  src_state = srcField->GetSolutionGF();

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

  double *dataU = src_state->HostWrite();
  int NDof = src_fes->GetNDofs();

  std::string basepath("utils/tanh_initial_condition");
  bool constantPressure;
  double p0;
  tps.getInput((basepath + "/constant_pressure").c_str(), constantPressure, false);
  if (constantPressure) tps.getRequiredInput((basepath + "/pressure").c_str(), p0);

  Vector sol1(num_equation), sol2(num_equation);
  for (int eq = 0; eq < num_equation; eq++) {
    tps.getRequiredInput((basepath + "/solution1/Q" + std::to_string(eq+1)).c_str(), sol1[eq]);
    tps.getRequiredInput((basepath + "/solution2/Q" + std::to_string(eq+1)).c_str(), sol2[eq]);
  }
  if (constantPressure) {
    grvy_printf(GRVY_INFO, "\n Setting total energy for the pressure: %.8E\n", p0);
    mixture->modifyEnergyForPressure(sol1, sol1, p0, false);
    mixture->modifyEnergyForPressure(sol2, sol2, p0, false);
  }
  for (int eq = 0; eq < num_equation; eq++) {
    grvy_printf(GRVY_INFO, "\n Solution 1 - Q%d: %.8E\n", (eq+1), sol1[eq]);
  }
  for (int eq = 0; eq < num_equation; eq++) {
    grvy_printf(GRVY_INFO, "\n Solution 2 - Q%d: %.8E\n", (eq+1), sol2[eq]);
  }

  double offset, scale;
  tps.getRequiredInput((basepath + "/offset").c_str(), offset);
  tps.getRequiredInput((basepath + "/scale").c_str(), scale);
  grvy_printf(GRVY_INFO, "\n offset: %.8E\n", offset);
  grvy_printf(GRVY_INFO, "\n scale: %.8E\n", scale);

  for (int i = 0; i < NDof; i++) {
    double tanhFactor = 0.5 + 0.5 * tanh((coordinates[i + 0 * NDof] - offset) / scale);
    // std::cout << "Grid point " << i << ": ";
    // for (int d = 0; d < dim; d++) {
    //   std::cout << coordinates[i + d * NDof] << ", ";
    // }
    // std::cout << std::endl;

    for (int eq = 0; eq < num_equation; eq++) {
      dataU[i + eq * NDof] = sol1[eq] * (1.0 - tanhFactor) + sol2[eq] * tanhFactor;
    }
  }

  RHSoperator *rhsOperator = srcField->getRHSoperator();
  rhsOperator->updatePrimitives(*src_state);

  // Write restart files
  srcField->writeHDF5();

  // reference solution
  double time;
  tps.getRequiredInput((basepath + "/reference_solution/time").c_str(), time);
  grvy_printf(GRVY_INFO, "\n Reference solution is at time = : %.8E\n", time);
  std::string refFileName;
  tps.getInput((basepath + "/reference_solution/filename").c_str(), refFileName, std::string("tanh_ic.ref.h5"));
  grvy_printf(GRVY_INFO, "\n Reference solution filename : %s\n", refFileName.c_str());

  double u0 = sol1[1] / sol1[0]; // assume sol2 has the same speed.
  double distance = u0 * time;
  for (int i = 0; i < NDof; i++) {
    double tanhFactor = 0.5 + 0.5 * tanh((coordinates[i + 0 * NDof] - distance - offset) / scale);
    // std::cout << "Grid point " << i << ": ";
    // for (int d = 0; d < dim; d++) {
    //   std::cout << coordinates[i + d * NDof] << ", ";
    // }
    // std::cout << std::endl;

    for (int eq = 0; eq < num_equation; eq++) {
      dataU[i + eq * NDof] = sol1[eq] * (1.0 - tanhFactor) + sol2[eq] * tanhFactor;
    }
  }

  rhsOperator->updatePrimitives(*src_state);

  // Write restart files
  srcField->writeHDF5(refFileName);

  return 0;
}
