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
#ifndef INLETBC_HPP_
#define INLETBC_HPP_

#include <tps_config.h>

#include "BoundaryCondition.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "logger.hpp"
#include "mpi_groups.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;

class InletBC : public BoundaryCondition {
 private:
  MPI_Groups *groupsMPI;

  GasMixture *d_mixture_;  // used only in the device

  const InletType inletType_;

  // In/out conditions specified in the configuration file
  Vector inputState;
  int numActiveSpecies_ = -1;

  // Mean boundary state
  Vector meanUp;

  Vector boundaryU;
  int bdrN;
  bool bdrUInit;

  // boundary mean calculation variables
  Vector bdrUp;          // Up at
  Array<int> bdrElemsQ;  // element dofs and face num. of integration points
  Array<int> bdrDofs;    // indexes of the D
  Vector bdrShape;       // shape functions evaluated at the integration points
  const int &maxIntPoints_;
  const int &maxDofs_;

  // local vector for mean calculation
  Vector localMeanUp;
  // global mean sum (in parallel computations
  Vector glob_sum;

  // Unit trangent vector 1 & 2
  Vector tangent1;
  Vector tangent2;
  Vector inverseNorm2cartesian;

  void initBdrElemsShape();
  void initBoundaryU(ParGridFunction *Up);

  void subsonicReflectingDensityVelocity(Vector &normal, Vector &stateIn, Vector &bdrFlux);
  void subsonicNonReflectingDensityVelocity(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);
  void subsonicNonReflectingTemperatureVelocity(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);
  void subsonicNonReflectingTemperatureVelocityUser(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector transip, Vector &bdrFlux);  

  virtual void updateMean(IntegrationRules *intRules, ParGridFunction *Up);

 public:
  InletBC(MPI_Groups *_groupsMPI, Equations _eqSystem, RiemannSolver *rsolver_, GasMixture *_mixture,
          GasMixture *d_mixture, ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt, const int _dim,
          const int _num_equation, int _patchNumber, double _refLength, InletType _bcType,
          const Array<double> &_inputData, const int &_maxIntPoints, const int &maxDofs, bool axisym);
  ~InletBC();

  //  void computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector &bdrFlux);
  void computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, double radius, Vector transip,
                      Vector &bdrFlux);

  virtual void initBCs();

  virtual void integrationBC(Vector &y,  // output
                             const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                             ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                             Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs);

  static void updateMean_gpu(ParGridFunction *Up, Vector &localMeanUp, const int _num_equation, const int numBdrElems,
                             const int totalDofs, Vector &bdrUp, Array<int> &bdrElemsQ, Array<int> &bdrDofs,
                             Vector &bdrShape, const int &maxIntPoints, const int &maxDofs);

  // functions for BC integration on GPU

  void integrateInlets_gpu(Vector &y,  // output
                           const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds, Vector &shapesBC,
                           Vector &normalsWBC, Array<int> &intPointsElIDBC, Array<int> &listElems,
                           Array<int> &offsetsBoundaryU);
  void interpInlet_gpu(const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds, Vector &shapesBC,
                       Vector &normalsWBC, Array<int> &intPointsElIDBC, Array<int> &listElems,
                       Array<int> &offsetsBoundaryU);

#ifdef _GPU_
  MFEM_HOST_DEVICE void pluginInputState(const double *inputState, double *u2, const int nvel,
                                         const int numActiveSpecies) {
    u2[0] = inputState[0];
    for (int v = 0; v < nvel; v++) u2[1 + v] = inputState[0] * inputState[1 + v];
    if (numActiveSpecies > 0) {
      for (int sp = 0; sp < numActiveSpecies; sp++) u2[nvel + 2 + sp] = inputState[4 + sp];
    }
    return;
  }

  static MFEM_HOST_DEVICE void computeSubDenseVel(const double *u1, double *u2, const double *nor,
                                                  const double *inputState, const double &gamma, const double &Rg,
                                                  const int &dim, const int &num_equation, const WorkingFluid &fluid,
                                                  const Equations &eqSystem, const int &thrd, const int &maxThread) {
    // assumes there at least as many threads as number of equations
    MFEM_SHARED double KE[3];
    MFEM_SHARED double p;

    if (thrd < dim) KE[thrd] = 0.5 * u1[1 + thrd] * u1[1 + thrd] / u1[0];
    if (dim != 3 && (thrd == num_equation - 1)) KE[2] = 0.;
    MFEM_SYNC_THREAD;

    if (fluid == WorkingFluid::DRY_AIR) {
      if (thrd == 0) p = DryAir::pressure(u1, &KE[0], gamma, dim, num_equation);
      MFEM_SYNC_THREAD;

      if (thrd == 0) u2[0] = inputState[0];
      if (thrd == 1) u2[1] = inputState[0] * inputState[1];
      if (thrd == 2) u2[2] = inputState[0] * inputState[2];
      if (dim == 3 && thrd == 3) u2[3] = inputState[0] * inputState[3];

      if (eqSystem == NS_PASSIVE && thrd == num_equation - 1) {
        u2[thrd] = 0.;
      }
      MFEM_SYNC_THREAD;

      DryAir::modifyEnergyForPressure_gpu(u2, u2, p, gamma, Rg, num_equation, dim, thrd, maxThread);
    }
  }

  static MFEM_HOST_DEVICE void computeSubDenseVel_gpu_serial(const double *u1, double *u2, const double *nor,
                                                             const double *inputState, const double &gamma,
                                                             const double &Rg, const int &dim, const int &num_equation,
                                                             const WorkingFluid &fluid) {
    // assumes there at least as many threads as number of equations
    double KE[3];
    double p;

    for (int d = 0; d < dim; d++) KE[d] = 0.5 * u1[1 + d] * u1[1 + d] / u1[0];
    if (dim != 3) KE[2] = 0.;

    if (fluid == WorkingFluid::DRY_AIR) {
      p = DryAir::pressure(u1, &KE[0], gamma, dim, num_equation);

      u2[0] = inputState[0];

      for (int d = 0; d < dim; d++) u2[1 + d] = inputState[0] * inputState[1 + d];

      DryAir::modifyEnergyForPressure_gpu_serial(u2, u2, p, gamma, Rg, num_equation, dim);
    }
  }

#endif
};

#endif  // INLETBC_HPP_
