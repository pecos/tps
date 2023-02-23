// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
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
#ifndef BOUNDARYCONDITION_HPP_
#define BOUNDARYCONDITION_HPP_

#include <grvy.h>
#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "riemann_solver.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;

class BoundaryCondition {
 protected:
  RiemannSolver *rsolver;
  GasMixture *mixture;
  Equations eqSystem;
  ParFiniteElementSpace *vfes;
  IntegrationRules *intRules;
  double &dt;
  const int dim_;
  const int nvel_;
  const int num_equation_;
  const int patchNumber;
  const double refLength;

  bool axisymmetric_;
  bool BCinit;

  Array<int> listElems;  // list of boundary elements (position in the BC array)

  Array<int> offsetsBoundaryU;

  Vector face_flux_;

 public:
  BoundaryCondition(RiemannSolver *_rsolver, GasMixture *_mixture, Equations _eqSystem, ParFiniteElementSpace *_vfes,
                    IntegrationRules *_intRules, double &dt, const int _dim, const int _num_equation,
                    const int _patchNumber, const double _refLength, bool axisym);
  virtual ~BoundaryCondition();

  virtual void computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                              Vector &bdrFlux) = 0;

  // holding function for any miscellaneous items needed to initialize BCs
  // prior to use (and require MPI)
  virtual void initBCs() = 0;

  virtual void updateMean(IntegrationRules *intRules, ParGridFunction *Up) = 0;

  // aggregate boundary area
  double aggregateArea(int bndry_attr, MPI_Comm bc_comm);
  // aggregate boundary face count
  int aggregateBndryFaces(int bndry_attr, MPI_Comm bc_comm);

  // integration of BC on GPU
  void setElementList(Array<int> &listElems);

  virtual void integrationBC(Vector &y,        // output
                             const Vector &x,  // conservative vars (input)
                             const elementIndexingData &elem_index_data, ParGridFunction *Up,
                             ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC, Array<int> &intPointsElIDBC,
                             const int &maxIntPoints, const int &maxDofs) = 0;

  static void copyValues(const Vector &orig, Vector &target, const double &mult);
};

#endif  // BOUNDARYCONDITION_HPP_
