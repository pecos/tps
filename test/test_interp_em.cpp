/* A utility to transfer tps solutions from one mesh to another */

#include "../src/tps.hpp"
#include "../src/cycle_avg_joule_coupling.hpp"
#include "../src/tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

double true_plasma_conductivity(const double r, const double z) {
  if (r < 0.5) {
    return (1 + z) * (1 - 2 * r) * (1 - 2 * r);
  } else {
    return 0.0;
  }
}

void set_flow_plasma_conductivity(const ParFiniteElementSpace *vfes, ParGridFunction *pc) {
  double *data = pc->HostWrite();

  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  // get coords
  const FiniteElementCollection *fec = vfes->FEColl();
  ParMesh *mesh = vfes->GetParMesh();

  int dim = mesh->Dimension();
  assert(dim == 2);

  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParGridFunction coordsDof(&dfes);
  mesh->GetNodes(coordsDof);

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);
    Vector x(dim);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for (int d = 0; d < dim; d++) x[d] = coordsDof[index + d * dof];

      const double radius = x[0];
      const double z = x[1];

      data[index] = true_plasma_conductivity(radius, z);
    }
  }
}

int check_em_plasma_conductivity(const ParFiniteElementSpace *vfes, ParGridFunction *pc) {
  int ierr = 0;
  //const double rtol = 2e-15;
  const double rtol = 3e-14;
  const double atol = 3e-16;

  const double *data = pc->HostRead();

  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  // get coords
  const FiniteElementCollection *fec = vfes->FEColl();
  ParMesh *mesh = vfes->GetParMesh();

  int dim = mesh->Dimension();
  assert(dim == 2);

  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParGridFunction coordsDof(&dfes);
  mesh->GetNodes(coordsDof);

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);
    Vector x(dim);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for (int d = 0; d < dim; d++) x[d] = coordsDof[index + d * dof];

      const double radius = x[0];
      const double z = x[1];

      double err = 0.0;
      if (radius < 0.5) {
        const double true_val = true_plasma_conductivity(radius, z);
        err = std::abs(data[index] - true_val);
        if ((err / true_val > rtol) && (err > atol)){
          printf("Error at r = %.3e, z = %.3e, value = %.6e, expected = %.6e, rel_err = %.6e, abs_err = %.6e\n", radius, z, data[index], true_val, err / true_val, err);
          ierr++;
        }
      } else {
        // true value is 0
        err = std::abs(data[index]);
        if (err > atol) {
          printf("Error at r = %.3e, z = %.3e, value = %.6e, expected = %.6e\n", radius, z, data[index], 0.0);
          ierr++;
        }
      }
    }
  }
  return ierr;
}


int main (int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);

  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();
  tps.chooseSolver();

  std::cout << "tps.getSolverType() = " << tps.getSolverType() << std::endl;
  assert(tps.getSolverType() == "cycle-avg-joule-coupled");

  CycleAvgJouleCoupling *solver = new CycleAvgJouleCoupling(tps.getInputFilename(), &tps);

  solver->parseSolverOptions();
  solver->initialize();


  // Check flow -> EM
  ParFiniteElementSpace *vfes = solver->getFlowSolver()->getFESpace();
  ParGridFunction *flow_pc = solver->getFlowSolver()->getPlasmaConductivityGF();

  // Set fields to known function
  set_flow_plasma_conductivity(vfes, flow_pc);

  // interpolation
  solver->initializeInterpolationData();
  solver->interpConductivityFromFlowToEM();

  // check results
  ParFiniteElementSpace *em_fes = solver->getEMSolver()->getFESpace();
  ParGridFunction *em_pc = solver->getEMSolver()->getPlasmaConductivityGF();

  int ierr = check_em_plasma_conductivity(em_fes, em_pc);


  // Check EM->flow
  ParGridFunction *em_jh = solver->getEMSolver()->getJouleHeatingGF();

  // Set fields to known function
  set_flow_plasma_conductivity(em_fes, em_jh);
  ierr += check_em_plasma_conductivity(em_fes, em_jh);

  // interpolation
  solver->interpJouleHeatingFromEMToFlow();

  // check results
  ParGridFunction *flow_jh = solver->getFlowSolver()->getJouleHeatingGF();

  ierr += check_em_plasma_conductivity(vfes, flow_jh);


  if (ierr != 0) {
    printf("Found %d errors!\n", ierr);
  } else {
    printf("Success: No errors found.\n");
  }
  return ierr;
}
