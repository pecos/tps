#include <tps_config.h>
#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>
#include <hdf5.h>

using namespace mfem;
using namespace std;

// This function checks the distance function computed for
// meshes/axi-pipe.msh, assuming that only the outer right boundary
// (at x=1) is set to be a wall, as in inputs/pipe.axisym.viscous.ini.
// For this case, the distance function is simply 1-x.
int checkTrivialDistance(const ParGridFunction &distance, const ParGridFunction &coordinates) {
  int err = 0;
  const double rel_tol = 1e-13;
  const double abs_tol = 1e-20;
  for (int i = 0; i < distance.Size(); i++) {
    const double dist_computed = distance[i];
    const double dist_expected = 1.0 - coordinates[i];

    if (dist_expected > 0) {
      const double rel_err = std::abs(dist_computed - dist_expected)/std::abs(dist_expected);
      if( rel_err > rel_tol) err++;
    } else {
      if (dist_computed > abs_tol) err++;
    }
  }
  return err;
}

int main (int argc, char *argv[]) {
  // This test is intended to be run with inputs/pipe.axisym.viscous.ini
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);

  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  // Instantiate flow solver (NB: distance fcn happens internally)
  M2ulPhyS *flow = new M2ulPhyS(tps.getInputFilename(), &tps);

  // Get mesh coordinates
  ParMesh *mesh = flow->GetMesh();
  ParFiniteElementSpace *dfes = flow->GetVectorFES();
  ParGridFunction coordinates(dfes);
  mesh->GetNodes(coordinates);

  // Get distance function
  const ParGridFunction *distance = flow->getDistanceFcn();

  // Compare distance function to expected result
  int ierr = checkTrivialDistance(*distance, coordinates);

  // Clean up
  delete flow;

  return ierr;
}
