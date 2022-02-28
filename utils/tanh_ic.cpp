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

  std::string basepath("utils/tanh_initial_condition");
  Vector sol1(num_equation), sol2(num_equation);

  for (int eq = 0; eq < num_equation; eq++) {
    tps.getRequiredInput((basepath + "/solution1/Q" + std::to_string(eq+1)).c_str(), sol1[eq]);
    tps.getRequiredInput((basepath + "/solution2/Q" + std::to_string(eq+1)).c_str(), sol2[eq]);
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

  RHSoperator rhsOperator = srcField->getRHSoperator();
  rhsOperator.updatePrimitives(*src_state);

  // Write restart files
  srcField->writeHDF5();

  return 0;
}
