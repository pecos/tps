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
#ifndef FLOW_LINEAR_SOLVER_HPP_
#define FLOW_LINEAR_SOLVER_HPP_

#include <mpi.h>

#include <mfem.hpp>

class flowLinearSolver : public mfem::Solver {
 private:
  mfem::GMRESSolver gmres_solver_;
  mfem::Solver *prec_;

 public:
  flowLinearSolver(const mfem::ParFiniteElementSpace *vfes) : gmres_solver_(vfes->GetComm()) {
    ParMesh *pmesh = vfes->GetParMesh();
    int dim = pmesh->Dimension();
    int ngeom = pmesh->GetNumGeometries(dim);
    int max_ngeom;
    MPI_Allreduce(&ngeom, &max_ngeom, 1, MPI_INT, MPI_MAX, vfes->GetComm());

    if (max_ngeom == 1) {
      // Block ILU preconditioner
      // NB: does not work on hybrid meshes
      int block_size = vfes->GetFE(0)->GetDof();
      prec_ = new mfem::BlockILU(block_size, mfem::BlockILU::Reordering::MINIMUM_DISCARDED_FILL);
    } else {
      // Regular ILU preconditioner
      mfem::HypreILU *ilu = new mfem::HypreILU();
      ilu->SetLevelOfFill(0);
      prec_ = ilu;
    }

    // TODO(trevilo): Allow user to set thse
    gmres_solver_.iterative_mode = false;
    gmres_solver_.SetRelTol(1e-9);
    gmres_solver_.SetAbsTol(1e-14);
    gmres_solver_.SetMaxIter(1000);
    // gmres_solver_.SetPrintLevel(1);
    gmres_solver_.SetPrintLevel(0);
    gmres_solver_.SetPreconditioner(*prec_);
  }

  ~flowLinearSolver() { delete prec_; }

  void SetOperator(const mfem::Operator &op) { gmres_solver_.SetOperator(op); }

  virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const { gmres_solver_.Mult(x, y); }
};

#endif  // FLOW_LINEAR_SOLVER_HPP_
