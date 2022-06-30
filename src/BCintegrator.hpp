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
#ifndef BCINTEGRATOR_HPP_
#define BCINTEGRATOR_HPP_

#include <tps_config.h>

#include "BoundaryCondition.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "mpi_groups.hpp"
#include "riemann_solver.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "unordered_map"

using namespace mfem;

// Boundary face term: <F.n(u),[w]>
class BCintegrator : public NonlinearFormIntegrator {
 protected:
  MPI_Groups *groupsMPI;

  RunConfiguration &config;

  RiemannSolver *rsolver;
  GasMixture *mixture;
  Fluxes *fluxClass;

  double &max_char_speed;
  IntegrationRules *intRules;

  ParMesh *mesh;

  // pointer to finite element space
  ParFiniteElementSpace *vfes;

  // pointer to primitive varibales
  ParGridFunction *Up;

  ParGridFunction *gradUp;

  Vector &shapesBC;
  Vector &normalsWBC;
  Array<int> &intPointsElIDBC;

  const int dim;
  const int num_equation;

  const int &maxIntPoints;
  const int &maxDofs;

  std::unordered_map<int, BoundaryCondition *> inletBCmap;
  std::unordered_map<int, BoundaryCondition *> outletBCmap;
  std::unordered_map<int, BoundaryCondition *> wallBCmap;

  // void calcMeanState();
  void computeBdrFlux(const int attr, Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                      Vector &bdrFlux);

 public:
  BCintegrator(MPI_Groups *_groupsMPI, ParMesh *_mesh, ParFiniteElementSpace *_vfes, IntegrationRules *_intRules,
               RiemannSolver *rsolver_, double &_dt, GasMixture *mixture, GasMixture *d_mixture, Fluxes *_fluxClass,
               ParGridFunction *_Up, ParGridFunction *_gradUp, Vector &_shapesBC, Vector &_normalsWBC,
               Array<int> &_intPointsElIDBC, const int _dim, const int _num_equation, double &_max_char_speed,
               RunConfiguration &_runFile, Array<int> &local_bdr_attr, const int &_maxIntPoints, const int &_maxDofs);
  ~BCintegrator();

  virtual void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
  void initBCs();

  void updateBCMean(ParGridFunction *Up);
  void integrateBCs(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds);

  // GPU functions
  static void retrieveGradientsData_gpu(ParGridFunction *gradUp, DenseTensor &elGradUp, Array<int> &vdofs,
                                        const int &num_equation, const int &dim, const int &totalDofs,
                                        const int &elDofs);
};

#endif  // BCINTEGRATOR_HPP_
