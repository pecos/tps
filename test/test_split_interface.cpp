#include <limits>

#include <mfem/general/forall.hpp>

#include "../src/thermo_chem_base.hpp"
#include "../src/split_flow_base.hpp"
#include "../src/tps_mfem_wrap.hpp"

using namespace mfem;

/**
 * @brief Test that the interface design for the low Mach split time
 * integrators builds and runs a trivial program.
 */
int main(int argc, char *argv[]) {
  // Initialize mpi
  Mpi::Init(argc, argv);

  // Create a mesh (doesn't really matter what it is)
  Mesh *mesh = new Mesh("./meshes/cyl-tet-coarse.msh");
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  // polynomial orders
  const int sorder = 2;
  const int vorder = 3;

  // Density and viscosity to use
  const double rho = 42.0;
  const double mu = 1.0e-6;

  // Set up the flow and thermo classes
  FlowBase *flow;
  ThermoChemModelBase *thermo;

  flow = new ZeroFlow(pmesh, vorder);
  thermo = new ConstantPropertyThermoChem(pmesh, sorder, rho, mu);

  flow->initializeSelf();
  thermo->initializeSelf();

  flow->initializeFromThermoChem(&thermo->toFlow_interface_);
  thermo->initializeFromFlow(&flow->toThermoChem_interface);

  // Take a time step (actually does nothing for these classes, but still)
  thermo->step();
  flow->step();

  // Make sure when we get the density via the flow interface it is correct
  const ParGridFunction *rho_field = flow->getThermoInterface()->density;

  const double tol = std::numeric_limits<double>::epsilon();
  int ierr = 0;
  {
    Vector trho;
    rho_field->GetTrueDofs(trho);
    const double *d_rho = trho.Read();
    double err = 0.0;
    double *perr = &err;
    MFEM_FORALL(i, trho.Size(), { perr[0] += abs(d_rho[i] - 42.0); });
    if (std::abs(err) > tol) {
      ierr += 1;
    }
  }

  // Make sure when we get the velocity via the thermo interface it is correct
  if (thermo->getFlowInterface()->swirl_supported) {
    ierr += 1;
  }

  const ParGridFunction *u_field = thermo->getFlowInterface()->velocity;

  {
    Vector tu;
    u_field->GetTrueDofs(tu);
    const double *d_u = tu.Read();
    double err = 0.0;
    double *perr = &err;
    MFEM_FORALL(i, tu.Size(), { perr[0] += abs(d_u[i]); });
    if (std::abs(err) > tol) {
      ierr += 1;
    }
  }

  // clean up
  delete thermo;
  delete flow;
  delete pmesh;

  return ierr;
}
