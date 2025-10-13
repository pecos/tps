/* A utility to transfer tps solutions from one mesh to another */

#include "loMach.hpp"
#include "M2ulPhyS.hpp"
#include "mfem.hpp"
#include "tps.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);
  TPS::Tps tar_tps(MPI_COMM_WORLD);

  // Set the method's default parameters.
  const char *src_input_file = "coarse.ini";
  const char *tar_input_file = "";
  const char *tar_mesh_h1 = "";
  const char *pv_output_dir = "pv_output";
  int order = 1;

  // Parse command-line options.
  OptionsParser args(argc, argv);
  args.AddOption(&src_input_file, "-r1", "--runFile1",
                 "TPS input file for source case---i.e., the original mesh and "
                 "solution (required)");
  args.AddOption(&tar_input_file, "-r2", "--runFile2",
                 "TPS input file for target case (optional--must use -r2 or "
                 "-mh1, but not both)");
  args.AddOption(&tar_mesh_h1, "-mh1", "--mesh-h1",
                 "Mesh to interpolate to (using H1 space) (optional--must use "
                 "-mh1 or -r2, but not both)");
  args.AddOption(&pv_output_dir, "-out", "--paraview-output",
                 "Paraview output directory (-mh1 only)");
  args.AddOption(&order, "-p", "--order-h1", "FEM order (H1 only)");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  string srcFileName(src_input_file);
  string tarFileName(tar_input_file);
  string tarMeshH1(tar_mesh_h1);

  // We have to have set one of these...
  if (tarFileName.empty() && tarMeshH1.empty()) {
    args.PrintUsage(cout);
    return 1;
  }

  // but not both
  if (!tarFileName.empty() && !tarMeshH1.empty()) {
    args.PrintUsage(cout);
    return 1;
  }

  // Instantiate M2ulPhyS class for the "coarse" (i.e., source) case

  // Note that the M2ulPhyS ctor is responsible for reading the
  // restart file, assuming that io/enableRestart = True in the tps
  // input file.  So, you need to set this option if you want to
  // interpolate from an existing restart file.  Otherwise, the
  // M2ulPhyS ctor will initialize the field (to uniform by default)
  // and then the code below will interpolate using that field.
  tps.parseInputFile(srcFileName);
  tps.chooseDevices();

  const std::string &src_solver_type = tps.getSolverType();

  TPS::PlasmaSolver *srcField = nullptr;
  if (src_solver_type == "flow") {
    srcField = new M2ulPhyS(srcFileName, &tps);
    //RunConfiguration &srcConfig = srcField->GetConfig();
    //assert(srcConfig.GetRestartCycle() > 0);
  } else if (src_solver_type == "loMach") {
    srcField = new LoMachSolver(&tps);
    srcField->parseSolverOptions();
    srcField->initialize();
  }
  tps.closeInputFile();

  ParMesh *mesh_1 = srcField->getMesh();
  const int dim = mesh_1->Dimension();

  // Instantiate the "fine" (i.e., target) mesh.
  // This must be done via the tps input file if the goal is to
  // generate a tps restart on a new mesh.  Alternatively, if just a
  // mesh is specified, using the option --mesh-h1, that mesh is read
  // and decomposed here.
  TPS::PlasmaSolver *tarField = nullptr;
  ParMesh *mesh_2 = nullptr;
  if (!tarFileName.empty()) {
    // Instantiate M2ulPhyS class for the "fine" (i.e., target) case
    // Note that the M2ulPhyS ctor is responsible for reading the
    // restart file, assuming that io/enableRestart = True in the tps
    // input file.  So, assuming there is not an existing restart file
    // for the target mesh, you should not set this option. If
    // it is set, M2ulPhyS will try to read a file that doesn't exist
    // and give an error.
    tar_tps.parseInputFile(tarFileName);

    const std::string &tar_solver_type = tar_tps.getSolverType();
    assert(src_solver_type == tar_solver_type);

    if (src_solver_type == "flow") {
      tarField = new M2ulPhyS(tarFileName, &tar_tps);
    } else if (src_solver_type == "loMach") {
      tarField = new LoMachSolver(&tar_tps);
      tarField->parseSolverOptions();
      tarField->initialize();
    }
    tar_tps.closeInputFile();

    mesh_2 = tarField->getMesh();
  } else {
    // Otherwise, read the mesh directly

    // All mpi ranks read the serial mesh
    Mesh serial_mesh(tarMeshH1.c_str());

    // Decompose
    mesh_2 = new ParMesh(MPI_COMM_WORLD, serial_mesh);
  }

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
       << endl;
  cout << "Target mesh curvature: " << mesh_2->GetNodes()->OwnFEC()->Name()
       << endl;

  // 2) Set up source field
  const FiniteElementCollection *src_fec = NULL;
  ParGridFunction *func_source = NULL;

  src_fec = srcField->getFEC();

  // 3) Some checks
  const Geometry::Type gt = mesh_2->GetNodalFESpace()->GetFE(0)->GetGeomType();
  MFEM_VERIFY(gt != Geometry::PRISM,
              "Wedge elements are not currently supported.");
  MFEM_VERIFY(mesh_2->GetNumGeometries(mesh_2->Dimension()) == 1,
              "Mixed meshes are not currently supported.");

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  // For every IOFamily that has been registered with the 'source',
  // interpolate it to the 'target' mesh
  std::vector<IOFamily>& src_io_families = srcField->getIODataOrganizer().getIOFamilies();

  for (size_t ifield = 0; ifield < src_io_families.size(); ifield++) {
    //func_source = srcField->GetSolutionGF();
    func_source = src_io_families[ifield].getParGridFunction();


    // Setup the FiniteElementSpace and GridFunction on the target mesh.
    const FiniteElementCollection *tar_fec = nullptr;
    ParFiniteElementSpace *tar_fes = nullptr;
    ParGridFunction *func_target = nullptr;

    if (!tarFileName.empty()) {
      tar_fec = tarField->getFEC();
      tar_fes = tarField->getFESpace();
      std::vector<IOFamily>& tar_io_families = tarField->getIODataOrganizer().getIOFamilies();
      func_target = tar_io_families[ifield].getParGridFunction();
      tar_fes = func_target->ParFESpace();// = tar_io_families[ifield].getParGridFunction();
      //func_target = tarField->GetSolutionGF();
    } else {
      const int num_variables = func_source->VectorDim();
      tar_fec = new H1_FECollection(order, dim);
      tar_fes = new ParFiniteElementSpace(mesh_2, tar_fec, num_variables);
      func_target = new ParGridFunction(tar_fes);
    }

    std::cout << "Target FE collection: " << tar_fec->Name() << std::endl;

    const int NE = mesh_2->GetNE();
    const int nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints();
    const int tar_ncomp = func_target->VectorDim();
    cout << "tar_ncomp = " << tar_ncomp << std::endl;

    // Generate list of points where the grid function will be evaluated.
    Vector vxyz;

    // The method for getting the necessary points here works for both
    // H1 and L2 fields.  However, for H1, it will contain duplicate
    // points, which means we must take care when setting the
    // corresponding ParGridFunction below.
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

    // FIXME: Ensure this functions correctly for H1
    if (tar_fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) {
      func_target->SetFromTrueDofs(interp_vals);
    } else if (tar_fec->GetContType() == FiniteElementCollection::CONTINUOUS) {
      // Fill solution element-by-element
      Array<int> vdofs;
      Vector elem_dof_vals(nsp * tar_ncomp);

      for (int i = 0; i < mesh_2->GetNE(); i++) {
        tar_fes->GetElementVDofs(i, vdofs);
        for (int j = 0; j < nsp; j++) {
          for (int d = 0; d < tar_ncomp; d++) {
            // Arrange values byNodes
            int idx = d * nsp * NE + i * nsp + j;
            elem_dof_vals(j + d * nsp) = interp_vals(idx);
          }
        }
        func_target->SetSubVector(vdofs, elem_dof_vals);
      }
      func_target->SetFromTrueVector();
    } else {
      std::cout << "FEC not understood" << std::endl;
      exit(1);
    }

    if (tarFileName.empty()) {
      // Dump paraview for visualization
      ParaViewDataCollection pvc(pv_output_dir, mesh_2);
      pvc.SetLevelsOfDetail(order);
      pvc.SetHighOrderOutput(true);
      pvc.SetPrecision(8);

      pvc.SetCycle(0);
      pvc.SetTime(0.0);

      ParFiniteElementSpace fes(mesh_2, tar_fec, 1);
      int ndofs = fes.GetNDofs();
      for (int ivar = 0; ivar < tar_ncomp; ivar++) {
        string U("U_");
        pvc.RegisterField(U + to_string(ivar),
                          new ParGridFunction(&fes, func_target->HostReadWrite() +
                                              ivar * ndofs));
      }
      pvc.Save();
    }

    // Free the internal gslib data.
    finder.FreeData();

    // If we own them, delete fe collection, etc
    if (tarFileName.empty()) {
      delete tar_fes;
      delete tar_fec;
      delete func_target;
    }
  }

  if (!tarFileName.empty()) {
    tarField->restart_files_hdf5("write");
  }



  // delete the target M2ulPhyS class
  delete tarField;
  delete srcField;

  return 0;
}
