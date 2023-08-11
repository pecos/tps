/* A utility to transfer tps solutions from one mesh to another */

#include "M2ulPhyS.hpp"
#include "mfem.hpp"
#include "tps.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);

  // Set the method's default parameters.
  const char *src_input_file = "coarse.run";
  const char *tar_input_file = "fine.run";

  // Parse command-line options.
  OptionsParser args(argc, argv);
  args.AddOption(&src_input_file, "-r1", "--runFile1",
                 "runFile for source case");
  args.AddOption(&tar_input_file, "-r2", "--runFile2",
                 "runFile for target case");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Instantiate M2ulPhyS classes for *both* the coarse and fine

  // NB: the M2ulPhyS ctor calls M2ulPhyS::initVariables, which reads
  // the restart files, assuming that RESTART_CYCLE is set in the
  // input file.  So, we require that RESTART_CYCLE is set in the
  // *source* run file....
  string srcFileName(src_input_file);

  tps.parseInputFile(srcFileName);
  tps.chooseDevices();

  M2ulPhyS srcField(srcFileName, &tps);
  RunConfiguration &srcConfig = srcField.GetConfig();
  assert(srcConfig.GetRestartCycle() > 0);

  tps.closeInputFile();

  ParMesh *mesh_1 = srcField.getMesh();
  const int dim = mesh_1->Dimension();

  // but, we require that RESTART_CYCLE is *not* set in the
  // target run file, since the target restart files do not exist yet.
  string tarFileName(tar_input_file);

  tps.parseInputFile(tarFileName);

  M2ulPhyS tarField(tarFileName, &tps);
  RunConfiguration &tarConfig = tarField.GetConfig();
  assert(tarConfig.GetRestartCycle() == 0);

  tps.closeInputFile();

  // Get meshes
  ParMesh *mesh_2 = tarField.getMesh();

  // const int dim = mesh_1->Dimension();

  // GSLIB only works in 2 and 3 dimensions
  assert(dim > 1);
  // Input meshes must have same dimension
  assert(mesh_2->Dimension() == dim);

  if (mesh_1->GetNodes() == NULL) {
    mesh_1->SetCurvature(1);
  }
  if (mesh_2->GetNodes() == NULL) {
    mesh_2->SetCurvature(1);
  }

  cout << "Source mesh curvature: " << mesh_1->GetNodes()->OwnFEC()->Name()
       << endl
       << "Target mesh curvature: " << mesh_2->GetNodes()->OwnFEC()->Name()
       << endl;

  // 2) Set up source field
  const FiniteElementCollection *src_fec = NULL;
  ParGridFunction *func_source = NULL;

  src_fec = srcField.getFEC();
  func_source = srcField.GetSolutionGF();

  // 3) Some checks
  const Geometry::Type gt = mesh_2->GetNodalFESpace()->GetFE(0)->GetGeomType();
  MFEM_VERIFY(gt != Geometry::PRISM, "Wedge elements are not currently "
                                     "supported.");
  MFEM_VERIFY(mesh_2->GetNumGeometries(mesh_2->Dimension()) == 1,
              "Mixed meshes"
              "are not currently supported.");

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  // Setup the FiniteElementSpace and GridFunction on the target mesh.
  const FiniteElementCollection *tar_fec = tarField.getFEC();
  ParFiniteElementSpace *tar_fes = tarField.getFESpace();
  ParGridFunction *func_target = tarField.GetSolutionGF();
  std::cout << "Target FE collection: " << tar_fec->Name() << std::endl;

  const int NE = mesh_2->GetNE();
  const int nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints();
  const int tar_ncomp = func_target->VectorDim();
  cout << "tar_ncomp = " << tar_ncomp << std::endl;

  // Generate list of points where the grid function will be evaluated.
  Vector vxyz;

  // NB: We assume DG here!
  vxyz.SetSize(nsp * NE * dim);
  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = tar_fes->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = tar_fes->GetElementTransformation(i);

    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(vxyz.GetData() + i * nsp, nsp);
    Vector rowy(vxyz.GetData() + i * nsp + NE * nsp, nsp);
    Vector rowz;
    if (dim == 3) {
      rowz.SetDataAndSize(vxyz.GetData() + i * nsp + 2 * NE * nsp, nsp);
    }
    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim == 3) {
      pos.GetRow(2, rowz);
    }
  }
  const int nodes_cnt = vxyz.Size() / dim;

  // Evaluate source grid function.
  Vector interp_vals(nodes_cnt * tar_ncomp);
  FindPointsGSLIB finder(MPI_COMM_WORLD);
  finder.Setup(*mesh_1);
  finder.Interpolate(vxyz, *func_source, interp_vals);

  // Project the interpolated values to the target FiniteElementSpace.
  // NB: Assume DG
  func_target->SetFromTrueDofs(interp_vals);

  // Write restart files
  tarField.writeHDF5();

  // Free the internal gslib data.
  finder.FreeData();

  return 0;
}
