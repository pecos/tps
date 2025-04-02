#include <iostream>

#include "../src/io.hpp"
#include "../src/loMach.hpp"
#include "../src/loMach_options.hpp"
#include "../src/mesh_base.hpp"
#include "../src/reactingFlow.hpp"
#include "../src/tps_mfem_wrap.hpp"
#include "../src/tps.hpp"

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(argc, argv);
  MPI_Comm tpsComm = MPI_COMM_WORLD;

  TPS::Tps tps(tpsComm);

  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  // Load just enough options to get through the ReactingFlow ctor
  LoMachOptions loMach_opts;
  tps.getRequiredInput("loMach/mesh", loMach_opts.mesh_file);

  // Instantiate the mesh
  const int order = 1;
  MeshBase *meshData = new MeshBase(&tps, &loMach_opts, order);
  meshData->initializeMesh();
  ParMesh *pmesh = meshData->getMesh();

  // Instantiate the reacting flow class
  temporalSchemeCoefficients temporal_coeff;
  temporal_coeff.order = 1;

  ReactingFlow *thermo = new ReactingFlow(pmesh, &loMach_opts, temporal_coeff, &tps);
  thermo->initializeSelf();

  int nSpecies;
  tps.getInput("plasma_models/species_number", nSpecies, 3);
  std::vector<double> Y0(nSpecies);
  for (int i = 1; i <= nSpecies; i++) {
    std::string basepath("species/species" + std::to_string(i));
    tps.getRequiredInput((basepath + "/initialMassFraction").c_str(), Y0[i-1]);
  }

  double T0;
  tps.getInput("initialConditions/temperature", T0, 300.0);

  std::cout << "Initial state:" << std::endl;
  for (int sp = 0; sp < nSpecies; sp++) {
    std::cout << "  Y[" << sp << "] = " << Y0[sp] << std::endl;
  }
  std::cout << "  T0 = " << T0 << std::endl;

  bool quasineutral;
  tps.getInput("plasma_models/ambipolar", quasineutral, false);

  int nActiveSpecies = nSpecies - 1;
  if (quasineutral) {
    nActiveSpecies -= 1;
  }
  int nState = nActiveSpecies + 1;

  double *YT = new double[nState];
  double *rhs = new double[nState];

  for (int sp = 0; sp < nActiveSpecies; sp++) {
    YT[sp] = Y0[sp];
  }
  YT[nActiveSpecies] = T0;

  // only allow constant time stepping
  double dt;
  tps.getRequiredInput("time/dt_fixed", dt);

  int nsub;
  tps.getInput("loMach/reactingFlow/sub-steps", nsub, 1);

  double time = 0;

  // // Forward Euler (with substepping)
  // dt /= nsub;
  // int max_iters = 20 * 50;
  // for(int step = 0; step < max_iters; ++step) {
  //   std::cout << step << " " << time << " " << YT[0] << " " << YT[nActiveSpecies] << std::endl;
  //   thermo->evaluateReactingSource(YT, 0, rhs);
  //   for (int i = 0; i < nState; i++) {
  //     std::cout << "  rhs[" << i << "] = " << rhs[i] << std::endl;
  //     YT[i] += dt * rhs[i];
  //     std::cout << "  YT[" << i << "] = " << YT[i] << std::endl;
  //   }
  //   time += dt;
  // }

  // return 0;

  // Backward Euler (fully implicit---i.e., nonlinearly implicit)
  int max_iters;
  tps.getInput("time/steps", max_iters, 10000);
  double *dYT = new double[nState];
  double *YT1 = new double[nState];
  double *YT0 = new double[nState];
  double *rhs1 = new double[nState];

  mfem::DenseMatrix Jac(nState);

  double eps = 1e-7;
  for(int step = 0; step < max_iters; ++step) {
    std::cout << step << " " << time;
    for (int i = 0; i < nState; i++) {
      std::cout << " " << YT[i];
    }
    std::cout << std::endl;

    for (int i = 0; i < nState; i++) {
      dYT[i] = 0.0;
      YT0[i] = YT[i];
    }

    // Evaluate RHS and Jacobian (via finite difference)
    thermo->evaluateReactingSource(YT, 0, rhs);
    for (int i = 0; i < nState; i++) {

      for (int j = 0; j < nState; j++) {
        YT1[j] = YT[j];
      }
      YT1[i] *= (1 + eps);

      thermo->evaluateReactingSource(YT1, 0, rhs1);

      for (int j = 0; j < nState; j++) {
        Jac(j,i) = (rhs1[j] - rhs[j]) / (YT1[i] - YT[i]);
      }
    }

    for (int i = 0; i < nState; i++) {
      rhs[i] *= -dt;

      for (int j = 0; j < nState; j++) {
        Jac(j,i) *= -dt;
      }
      Jac(i,i) += 1.0;
    }

    double res_norm0 = 0;
    for (int i = 0; i < nState; i++) {
      res_norm0 += rhs[i] * rhs[i];
    }
    res_norm0 = sqrt(res_norm0);

    std::cout << "Beginning Newton iteration with r0 = " << res_norm0 << std::endl;

    bool converged = false;
    const int IMAX = 200;
    //for (int iiter = 0; iiter < IMAX; iiter++) {
    int iiter = 0;

    double res_norm = res_norm0;
    while (iiter < IMAX && res_norm > 1e-12 && (res_norm / res_norm0) > 1e-9) {
      // Solve the system, afterward, rhs contains the solution
      mfem::LinearSolve(Jac, rhs, 1.e-9);

      for (int i = 0; i < nState; i++) {
        if (iiter < 100) {
          //dYT[i] = -0.1 * rhs[i];
          dYT[i] = -rhs[i];
        } else {
          dYT[i] = -rhs[i];
        }
        YT[i] += dYT[i];
      }

      // Evaluate RHS and Jacobian (via finite difference)
      thermo->evaluateReactingSource(YT, 0, rhs);
      for (int i = 0; i < nState; i++) {

        for (int j = 0; j < nState; j++) {
          YT1[j] = YT[j];
        }
        YT1[i] *= (1 + eps);

        thermo->evaluateReactingSource(YT1, 0, rhs1);

        for (int j = 0; j < nState; j++) {
          Jac(j,i) = (rhs1[j] - rhs[j]) / (YT1[i] - YT[i]);
        }
      }

      for (int i = 0; i < nState; i++) {
        rhs[i] *= -dt;
        rhs[i] += YT[i] - YT0[i];

        for (int j = 0; j < nState; j++) {
          Jac(j,i) *= -dt;
        }
        Jac(i,i) += 1.0;
      }

      res_norm = 0;
      for (int i = 0; i < nState; i++) {
        res_norm += rhs[i] * rhs[i];
      }
      res_norm = sqrt(res_norm);

      std::cout << iiter << ": r = " << res_norm << ", r/r0 = " << res_norm / res_norm0 << std::endl;

      // if (res_norm < 1e-14 || (res_norm / res_norm0 < 1e-7)) {
      //   converged = true;
      //   break;
      // }
      iiter += 1;
    }
    if (iiter >= IMAX) {
      std::cout << "WARNING: Newton solve did not converge." << std::endl;
      for (int i = 0; i < nState; i++) {
        std::cout << "YT[" << i << "] = " << YT[i] << std::endl;
      }
    }
    time += dt;
  }

  delete thermo;

  return 0;
}
