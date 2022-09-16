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
#ifndef WALLBC_HPP_
#define WALLBC_HPP_

#include <tps_config.h>

#include "BoundaryCondition.hpp"
#include "dataStructures.hpp"
#include "fluxes.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;

class WallBC : public BoundaryCondition {
 private:
  // The type of wall
  const WallType wallType_;
  const WallData wallData_;

  GasMixture *d_mixture_;  // only used in the device.
  Fluxes *fluxClass;

  // TODO(kevin): eventually replace this with wallPrim.
  double wallTemp_;

  // TODO(kevin): can be moved to BoundaryCondition.
  BoundaryViscousFluxData bcFlux_;
  BoundaryPrimitiveData bcState_;

  const Array<int> &intPointsElIDBC;
  const int &maxIntPoints_;

  Array<int> wallElems;
  void buildWallElemsArray(const Array<int> &intPointsElIDBC);

  void computeINVwallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux);
  void computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                Vector &bdrFlux);
  void computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius,
                                 Vector &bdrFlux);
  void computeGeneralWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux);

 public:
  WallBC(RiemannSolver *rsolver_, GasMixture *_mixture, GasMixture *d_mixture, Equations _eqSystem, Fluxes *_fluxClass,
         ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt, const int _dim,
         const int _num_equation, int _patchNumber, WallType _bcType, const WallData _inputData,
         const Array<int> &intPointsElIDBC, const int &maxIntPoints, bool axisym);
  ~WallBC();

  void computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector transip, int ibdrN, 
                      Vector &bdrFlux);

  virtual void initBCs();

  virtual void updateMean(IntegrationRules *intRules, ParGridFunction *Up) {}

  // functions for BC integration on GPU

  virtual void integrationBC(Vector &y,  // output
                             const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                             ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                             Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs);

  void integrateWalls_gpu(Vector &y,  // output
                          const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds, Vector &shapesBC,
                          Vector &normalsWBC, Array<int> &intPointsElIDBC, const int &maxDofs);

  void interpWalls_gpu(const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds, ParGridFunction *Up,
                       ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC, Array<int> &intPointsElIDBC,
                       const int &maxDofs);

#ifdef _GPU_
  static MFEM_HOST_DEVICE void computeInvWallState(const double *u1, double *u2, const double *nor, const int &dim,
                                                   const int &num_equation, const int &thrd, const int &maxThreads) {
    MFEM_SHARED double momNormal, norm;
    MFEM_SHARED double unitNor[3];

    if (thrd == maxThreads - 1) {
      norm = 0.;
      for (int d = 0; d < dim; d++) norm += nor[d] * nor[d];
      norm = sqrt(norm);
    }
    MFEM_SYNC_THREAD;

    for (int d = thrd; d < dim; d += maxThreads) unitNor[d] = nor[d] / norm;
    MFEM_SYNC_THREAD;

    if (thrd == 0) {
      momNormal = 0.;
      for (int d = 0; d < dim; d++) momNormal += unitNor[d] * u1[d + 1];
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) u2[thrd] = u1[thrd];
    MFEM_SYNC_THREAD;

    for (int d = thrd; d < dim; d += maxThreads) {
      u2[d + 1] = u1[d + 1] - 2. * momNormal * unitNor[d];
    }
  }

  static MFEM_HOST_DEVICE void computeInvWallState_gpu_serial(const double *u1, double *u2, const double *nor,
                                                              const int &dim, const int &num_equation) {
    double momNormal, norm;
    double unitNor[3];

    norm = 0.;
    for (int d = 0; d < dim; d++) norm += nor[d] * nor[d];
    norm = sqrt(norm);

    for (int d = 0; d < dim; d++) unitNor[d] = nor[d] / norm;

    momNormal = 0.;
    for (int d = 0; d < dim; d++) momNormal += unitNor[d] * u1[d + 1];

    for (int eq = 0; eq < num_equation; eq++) u2[eq] = u1[eq];

    for (int d = 0; d < dim; d++) {
      u2[d + 1] = u1[d + 1] - 2. * momNormal * unitNor[d];
    }
  }

  static MFEM_HOST_DEVICE void computeIsothermalState(const double *u1, double *u2, const double *nor,
                                                      const double &wallTemp, const double &gamma, const double &Rg,
                                                      const int &dim, const int &num_equation,
                                                      const WorkingFluid &fluid, const int &thrd,
                                                      const int &maxThreads) {
    if (thrd < num_equation) u2[thrd] = u1[thrd];
    MFEM_SYNC_THREAD;

    if (fluid == WorkingFluid::DRY_AIR) {
      DryAir::computeStagnantStateWithTemp_gpu(u1, u2, wallTemp, gamma, Rg, num_equation, dim, thrd, maxThreads);
    }
  }

  static MFEM_HOST_DEVICE void computeIsothermalState_gpu_serial(const double *u1, double *u2, const double *nor,
                                                                 const double &wallTemp, const double &gamma,
                                                                 const double &Rg, const int &dim,
                                                                 const int &num_equation, const WorkingFluid &fluid) {
    for (int eq = 0; eq < num_equation; eq++) {
      u2[eq] = u1[eq];
    }

    if (fluid == WorkingFluid::DRY_AIR) {
      DryAir::computeStagnantStateWithTemp_gpu_serial(u1, u2, wallTemp, gamma, Rg, num_equation, dim);
    }
  }

#endif
};

#endif  // WALLBC_HPP_
