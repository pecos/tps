// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-

#include <cmath>
#include <limits>

#include <mfem.hpp>

#include "../src/tps.hpp"
#include "../src/M2ulPhyS.hpp"

int pingFaceNonlinearForm(DGNonLinearForm *A, ParGridFunction *U, MPI_Session mpi) {
  int ierr = 0;

  ParFiniteElementSpace *pfes = U->ParFESpace();

  ParGridFunction Up(*U);
  ParGridFunction dU(*U);
  dU = 0.;

  Vector r0, rp;
  r0.SetSize(A->Height());
  r0 = 0.;
  rp.SetSize(A->Height());
  rp = 0.;

  A->Mult(*U, r0);
  Operator &J0 = A->GetGradient(*U);

  //double perturbation[3] = {1e-3, 1e-4, 1e-5};
  double perturbation[3] = {1e-2, 1e-3, 1e-4};
  //double perturbation[3] = {1e-1, 1e-2, 1e-3};
  double err[3];

  // build the perturbation
  HypreParVector U_tdof = *U->GetTrueDofs();
  U->SetFromTrueDofs(U_tdof);
  Up.SetFromTrueDofs(U_tdof);
  HypreParVector Up_tdof = *Up.GetTrueDofs();
  HypreParVector dU_tdof = *dU.GetTrueDofs();

  for (int irank = 0; irank < mpi.WorldSize(); irank++) { // for each rank

    int ndof = -1;
    if (mpi.WorldRank() == irank) {
      ndof = pfes->GetTrueVSize();
    }

    // tell everyone how many dofs we're going to loop over
    MPI_Bcast( &ndof, 1, MPI_INT, irank, MPI_COMM_WORLD);

    for (int idof = 0; idof < ndof; idof++) {
      for (int ref = 0; ref < 3; ref++ ) {
        dU_tdof = 0.;
        if (mpi.WorldRank() == irank) {
          dU_tdof(idof) = U_tdof(idof) * perturbation[ref];
        }
        Up_tdof = U_tdof;
        Up_tdof += dU_tdof;
        Up.SetFromTrueDofs(Up_tdof);
        dU.SetFromTrueDofs(dU_tdof);

        rp = 0.;
        A->Mult(Up, rp);

        Vector dr;
        dr.SetSize(A->Height());
        dr = 0.;

        J0.Mult(dU, dr);

        Vector delta(rp);
        delta -= r0;

        Vector Jerr(delta);
        Jerr -= dr;

        // We only check error convergence rate if relative difference is large enough
        if (Jerr.Norml2() / delta.Normlinf() > 1e-13) {
          err[ref] = Jerr.Norml2();
        } else {
          err[ref] = -1;
        }
      }

      if (err[0] > 0) {
        const double rate = std::log(err[2] / err[1]) / std::log(0.1);

        if (rate < 1.95 || rate > 2.05) {
          std::cout << "myrank = " << mpi.WorldRank() << ", irank = " << irank << ", idof = " << idof << std::endl;
          std::cout << "error = " << err[0] << " " << err[1] << " " << err[2] << std::endl;
          std::cout << "rate = " << rate << std::endl;
          ierr += 1;
        }
      }
    }

  }

  // return sum of errors over all ranks
  int sum_err;
  MPI_Allreduce(&ierr, &sum_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // // only serial right now!
  // for (int ii = 0; ii < U->Size(); ii++) {
  //   rp = 0.;

  //   dU = 0.;
  //   dU[ii] = (*U)[ii] * perturbation;

  //   Up = *U;
  //   Up += dU;

  //   A->Mult(Up, rp);

  //   Vector dr;
  //   dr.SetSize(A->Height());
  //   dr = 0.;
  //   J0.Mult(dU, dr);

  //   Vector delta(rp);
  //   delta -= r0;
  //   delta -= dr;

  //   double rel_err = delta.Norml2()/dr.Norml2();
  //   //std::cout << "Component ii = " << ii<< ", relative_error = " << rel_err << std::endl;

  //   if (rel_err > tol) {
  //     ierr += 1;
  //   }
  // }

  //std::cout << "Perturbation = " << perturbation << std::endl;
  if (mpi.Root()) std::cout << "pingFaceNonlinearForm: Number of errors = " << sum_err << std::endl;
  return sum_err;
}


int main(int argc, char *argv[]) {
  int ierr = 0;

  {
    TPS::Tps tps;
    tps.parseCommandLineArgs(argc, argv);
    tps.parseInput();
    tps.chooseDevices();

    M2ulPhyS *flow = new M2ulPhyS(tps.getMPISession(), tps.getInputFilename(), &tps);
    DGNonLinearForm *A = flow->GetFaceIntegrator();
    ParGridFunction *U = flow->GetSolutionGF();

    ierr += pingFaceNonlinearForm(A, U, tps.getMPISession());
  }

  std::cout << "ierr = " << ierr << std::endl;
  return ierr;
}

