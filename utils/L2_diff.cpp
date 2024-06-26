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

  M2ulPhyS *srcField1 = new M2ulPhyS( tps.getInputFilename(), &tps);
  M2ulPhyS *srcField2 = new M2ulPhyS( tps.getInputFilename(), &tps);
  //RunConfiguration& srcConfig = srcField1->GetConfig();

  GasMixture *mixture = srcField1->getMixture();
  //int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  // Get meshes
  ParMesh* mesh_1 = srcField1->getMesh();

  const int dim = mesh_1->Dimension();

  // GSLIB only works in 2 and 3 dimensions
  assert(dim>1);

  if (mesh_1->GetNodes() == NULL) { mesh_1->SetCurvature(1); }
  // const int mesh_poly_deg =
  //   mesh_1->GetNodes()->FESpace()->GetElementOrder(0);
  cout << "Source mesh curvature: "
       << mesh_1->GetNodes()->OwnFEC()->Name() << endl;

  std::string basepath("utils/L2_diff");
  std::string filename1, filename2;
  tps.getRequiredInput((basepath + "/filename1").c_str(), filename1);
  tps.getRequiredInput((basepath + "/filename2").c_str(), filename2);
  grvy_printf(GRVY_INFO, "\n Filename 1 : %s\n", filename1.c_str());
  grvy_printf(GRVY_INFO, "\n Filename 2 : %s\n", filename2.c_str());
  srcField1->readHDF5(filename1);
  srcField2->readHDF5(filename2);

  srcField1->updatePrimitives();
  srcField2->updatePrimitives();

  // 2) Set up source field
  const FiniteElementCollection *src_fec = NULL;
  ParFiniteElementSpace *src_fes = NULL;
  ParGridFunction *src_state1 = NULL;
  ParGridFunction *src_state2 = NULL;

  src_fec = srcField1->getFEC();
  src_fes = srcField1->getFESpace();
  src_state1 = srcField1->GetSolutionGF();
  src_state2 = srcField2->GetSolutionGF();
  ParGridFunction diff(*src_state1), zeros(*src_state1);

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

  double *dataU1 = src_state1->HostReadWrite();
  double *dataU2 = src_state2->GetData();
  double *dataDiff = diff.HostReadWrite();
  double *dataZeros = zeros.HostReadWrite();
  int NDof = src_fes->GetNDofs();

  for (int i = 0; i < NDof; i++) {
    for (int eq = 0; eq < num_equation; eq++) {
      dataDiff[i + eq * NDof] = dataU1[i + eq * NDof] - dataU2[i + eq * NDof];
      dataZeros[i + eq * NDof] = 0.0;
    }
  }

  //IntegrationRules *intRules = srcField1->getIntegrationRules();
  VectorGridFunctionCoefficient diffCoeff(&diff), zeroCoeff(&zeros);
  VectorGridFunctionCoefficient state2Coeff(src_state2);
  double error = src_state1->ComputeLpError(2, state2Coeff);
  grvy_printf(GRVY_INFO, "\n L2 norm of difference : %.8E\n", error);

  error = diff.ComputeLpError(2, zeroCoeff);
  grvy_printf(GRVY_INFO, "\n L2 norm of difference : %.8E\n", error);

  double norm = src_state1->ComputeLpError(2, zeroCoeff);
  grvy_printf(GRVY_INFO, "\n L2 norm of solution : %.8E\n", norm);
  grvy_printf(GRVY_INFO, "\n Relative error : %.8E\n", error / norm);

  ParGridFunction *src_prim1 = srcField1->getPrimitiveGF();
  ParGridFunction *src_prim2 = srcField2->getPrimitiveGF();

  ParGridFunction *press1 = srcField1->getPressureGF();
  ParGridFunction *press2 = srcField2->getPressureGF();
  mixture->UpdatePressureGridFunction(press1, src_prim1);
  mixture->UpdatePressureGridFunction(press2, src_prim2);

  double *dataUp1 = src_prim1->HostReadWrite();
  double *dataUp2 = src_prim2->HostReadWrite();
  double *dataP1 = press1->HostReadWrite();
  double *dataP2 = press2->HostReadWrite();
  for (int i = 0; i < NDof; i++) {
    for (int eq = 0; eq < num_equation; eq++) {
      dataU1[i + eq * NDof] = dataDiff[i + eq * NDof];
      dataUp1[i + eq * NDof] -= dataUp2[i + eq * NDof];
      dataP1[i + eq * NDof] -= dataP2[i + eq * NDof];
    }
  }

  srcField1->writeParaview(0, 0.0);

  // double norm = ComputeGlobalLpNorm(2, diffCoeff, *mesh_1, *intRules);
  // diff.ProjectCoefficient(diffCoeff);

  return 0;
}
