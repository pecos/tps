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
#ifndef OUTLETBC_HPP_
#define OUTLETBC_HPP_

#include <tps_config.h>

#include <mfem.hpp>

#include "BoundaryCondition.hpp"
#include "dataStructures.hpp"
#include "logger.hpp"
#include "mpi_groups.hpp"

using namespace mfem;

class OutletBC : public BoundaryCondition {
 private:
  MPI_Groups *groupsMPI;

  const OutletType outletType;

  // In/out conditions specified in the configuration file
  const Array<double> inputState;

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
  const int &maxIntPoints;
  const int &maxDofs;

  // local vector for mean calculation
  Vector localMeanUp;
  // global mean sum (in parallel computations
  Vector glob_sum;

  // area of the outlet
  double area;
  bool parallelAreaComputed;

  // Unit trangent vector 1 & 2
  Vector tangent1;
  Vector tangent2;
  Vector inverseNorm2cartesian;

  void initBdrElemsShape();
  void initBoundaryU(ParGridFunction *Up);

  void subsonicReflectingPressure(Vector &normal, Vector &stateIn, Vector &bdrFlux);

  void subsonicNonReflectingPressure(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);

  void subsonicNonRefMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);

  void subsonicNonRefPWMassFlow(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);

  virtual void updateMean(IntegrationRules *intRules, ParGridFunction *Up);

  void computeParallelArea();

 public:
  OutletBC(MPI_Groups *_groupsMPI, Equations _eqSystem, RiemannSolver *rsolver_, GasMixture *mixture,
           ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt, const int _dim,
           const int _num_equation, int _patchNumber, double _refLength, OutletType _bcType,
           const Array<double> &_inputData, const int &_maxIntPoints, const int &maxDofs);
  ~OutletBC();

  // computes state that imposes the boundary
  void computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux);
  virtual void initBCs();

  virtual void integrationBC(Vector &y,  // output
                             const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                             ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                             Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs);

  static void updateMean_gpu(ParGridFunction *Up, Vector &localMeanUp, const int _num_equation, const int numBdrElems,
                             const int totalDofs, Vector &bdrUp, Array<int> &bdrElemsQ, Array<int> &bdrDofs,
                             Vector &bdrShape, const int &maxIntPoints, const int &maxDofs);

  // functions for BC integration on GPU

  static void integrateOutlets_gpu(const OutletType type, Equations &eqSystem, const Array<double> &inputState,
                                   const double &dt,
                                   Vector &y,  // output
                                   const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                                   ParGridFunction *Up, ParGridFunction *gradUp, Vector &meanUp, Vector &boundaryU,
                                   Vector &interpolated_Ubdr, Vector &interpolatedGradUpbdr_, Vector &tangent1,
                                   Vector &tangent2, Vector &inverseNorm2cartesian, Vector &shapesBC,
                                   Vector &normalsWBC, Array<int> &intPointsElIDBC, Array<int> &listElems,
                                   Array<int> &offsetBoundaryU, const int &maxIntPoints, const int &maxDofs,
                                   const int &dim, const int &num_equation, GasMixture *mixture,
                                   const double &refLength, const double &area);

  static void interpOutlet_gpu(const OutletType type, const Array<double> &inputState, Vector &interpolated_Ubdr,
                               Vector &interpolatedGradUpbdr_, const Vector &x, const Array<int> &nodesIDs,
                               const Array<int> &posDofIds, ParGridFunction *Up, ParGridFunction *gradUp,
                               Vector &shapesBC, Vector &normalsWBC, Array<int> &intPointsElIDBC, Array<int> &listElems,
                               Array<int> &offsetsBoundaryU, const int &maxIntPoints, const int &maxDofs,
                               const int &dim, const int &num_equation);

#ifdef _GPU_  // GPU functions
  static MFEM_HOST_DEVICE void computeSubPressure(const int &thrd, const double *u1, double *u2, const double *nor,
                                                  const double &press, const double &gamma, const int &dim,
                                                  const int &num_equation, const Equations &eqSystem) {
    if (thrd == 1 + dim) {
      // NOTE: this still assumes ideal gas equation.
      double k = 0.;
      for (int d = 0; d < dim; d++) k += u1[1 + d] * u1[1 + d];
      u2[thrd] = press / (gamma - 1.) + 0.5 * k / u1[0];
    } else {
      u2[thrd] = u1[thrd];
    }
  }

  static MFEM_HOST_DEVICE void computeNRSubPress(const int &thrd, const int &n, const double *u1, const double *gradUp,
                                                 const double *meanUp, const double &dt, double *u2, double *boundaryU,
                                                 const double *inputState, const double *nor, const double *d_tang1,
                                                 const double *d_tang2, const double *d_inv, const double &refLength,
                                                 const double &gamma, const double &Rg, const int &elDof,
                                                 const int &dim, const int &num_equation, const Equations &eqSystem) {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[20];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK, dpdn;
    MFEM_SHARED double L1, L2, L3, L4, L5, L6;
    MFEM_SHARED double d1, d2, d3, d4, d5;
    MFEM_SHARED double bdrFlux[20], uN[20], newU[20];

    if (thrd < dim) unitNorm[thrd] = nor[thrd];
    if (thrd == num_equation - 1) {
      mod = 0.;
      for (int d = 0; d < dim; d++) mod += nor[d] * nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;

    if (thrd < dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;

    // mean vel. outlet tangent and normal
    if (thrd < 3) {
      meanVel[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 0) meanVel[0] += unitNorm[d] * meanUp[d + 1];
        if (thrd == 1) meanVel[1] += d_tang1[d] * meanUp[d + 1];
        if (dim == 3 && thrd == dim - 1) {
          meanVel[2] += d_tang2[d] * meanUp[d + 1];
        }
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) {
      normGrad[thrd] = 0.;
      for (int d = 0; d < dim; d++) normGrad[thrd] += unitNorm[d] * gradUp[thrd + d * num_equation];
    }
    MFEM_SYNC_THREAD;
    if (thrd == 0) {
      // NOTE: assumes DryAir!
      double T = DryAir::temperatureFromConservative(u1, gamma, Rg, dim, num_equation);
      dpdn = Rg * (T * normGrad[0] + u1[0] * normGrad[1 + dim]);
    }
    MFEM_SYNC_THREAD;

    if (thrd == 0) speedSound = sqrt(gamma * Rg * meanUp[1 + dim]);  // assumes perfect gas
    if (thrd == elDof - 1) {
      meanK = 0.;
      for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;

    // compute outgoing characteristics
    if (thrd == 1) {
      L2 = speedSound * speedSound * normGrad[0] - dpdn;
      L2 *= meanVel[0];
    }

    if (thrd == 2) {
      L3 = 0;
      for (int d = 0; d < dim; d++) L3 += d_tang1[d] * normGrad[1 + d];
      L3 *= meanVel[0];
    }

    if (thrd == 3) {
      L4 = 0.;
      if (dim == 3) {
        for (int d = 0; d < dim; d++) L4 += d_tang2[d] * normGrad[1 + d];
        L4 *= meanVel[0];
      }
    }

    if (thrd == 0) {
      L5 = 0.;
      for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
      L5 = dpdn + meanUp[0] * speedSound * L5;
      L5 *= meanVel[0] + speedSound;
    }

    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      L6 = meanVel[0] * normGrad[num_equation - 1];
    }

    // estimate ingoing characteristic
    if (thrd == elDof - 1) {
      const double sigma = speedSound / refLength;
      double meanP = Rg * meanUp[0] * meanUp[1 + dim];  // NOTE: assumes DryAir
      L1 = sigma * (meanP - inputState[0]);
    }
    MFEM_SYNC_THREAD;

    if (thrd == 0) d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
    if (thrd == 1) d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
    if (thrd == 2) d3 = L3;
    if (thrd == 4) d4 = L4;
    if (thrd == 5) d5 = 0.5 * (L5 + L1);
    MFEM_SYNC_THREAD;

    if (thrd == 0) bdrFlux[0] = d1;
    if (thrd == 1) bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
    if (thrd == 2) bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
    if (thrd == elDof - 1 && dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
    if (thrd == 1 + dim) {
      bdrFlux[1 + dim] = meanUp[0] * meanVel[0] * d2;
      bdrFlux[1 + dim] += meanUp[0] * meanVel[1] * d3;
      if (dim == 3) bdrFlux[1 + dim] += meanUp[0] * meanVel[2] * d4;
      bdrFlux[1 + dim] += meanK * d1 + d5 / (gamma - 1.);
    }
    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      bdrFlux[num_equation - 1] = d1 * u1[num_equation - 1] / u1[0] + u1[0] * L6;
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) u2[thrd] = boundaryU[thrd + n * num_equation];
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) uN[thrd] = u2[thrd];
    if (thrd > 0 && thrd < dim + 1) {
      uN[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 1) uN[1] += u2[1 + d] * unitNorm[d];
        if (thrd == 2) uN[2] += u2[1 + d] * d_tang1[d];
        if (dim == 3 && thrd == 3) uN[3] += u2[1 + d] * d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) newU[thrd] = uN[thrd] - dt * bdrFlux[thrd];
    MFEM_SYNC_THREAD;

    double sum = 0.;
    if (thrd < dim) {
      for (int j = 0; j < dim; j++) sum += d_inv[thrd + j * dim] * newU[1 + j];
    }
    MFEM_SYNC_THREAD;
    if (thrd < dim) newU[thrd + 1] = sum;
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) boundaryU[thrd + n * num_equation] = newU[thrd];
  }

  static MFEM_HOST_DEVICE void computeNRSubMassFlow(
      const int &thrd, const int &n, const double *u1, const double *gradUp, const double *meanUp, const double &dt,
      double *u2, double *boundaryU, const double *inputState, const double *nor, const double *d_tang1,
      const double *d_tang2, const double *d_inv, const double &refLength, const double &area, const double &gamma,
      const double &Rg, const int &elDof, const int &dim, const int &num_equation, const Equations &eqSystem) {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[20];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK, dpdn;
    MFEM_SHARED double L1, L2, L3, L4, L5, L6;
    MFEM_SHARED double d1, d2, d3, d4, d5;
    MFEM_SHARED double bdrFlux[20], uN[20], newU[20];

    if (thrd < dim) unitNorm[thrd] = nor[thrd];
    if (thrd == num_equation - 1) {
      mod = 0.;
      for (int d = 0; d < dim; d++) mod += nor[d] * nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;

    if (thrd < dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;

    // mean vel. outlet tangent and normal
    if (thrd < 3) {
      meanVel[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 0) meanVel[0] += unitNorm[d] * meanUp[d + 1];
        if (thrd == 1) meanVel[1] += d_tang1[d] * meanUp[d + 1];
        if (dim == 3 && thrd == dim - 1) {
          meanVel[2] += d_tang2[d] * meanUp[d + 1];
        }
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) {
      normGrad[thrd] = 0.;
      for (int d = 0; d < dim; d++) normGrad[thrd] += unitNorm[d] * gradUp[thrd + d * num_equation];
    }
    MFEM_SYNC_THREAD;
    if (thrd == 0) {
      // NOTE: assumes DryAir!
      double T = DryAir::temperatureFromConservative(u1, gamma, Rg, dim, num_equation);
      dpdn = Rg * (T * normGrad[0] + u1[0] * normGrad[1 + dim]);
    }
    MFEM_SYNC_THREAD;

    // NOTE: assumes DryAir!
    if (thrd == 0) speedSound = sqrt(gamma * Rg * meanUp[1 + dim]);
    if (thrd == elDof - 1) {
      meanK = 0.;
      for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;

    // compute outgoing characteristics
    if (thrd == 1) {
      L2 = speedSound * speedSound * normGrad[0] - dpdn;
      L2 *= meanVel[0];
    }

    if (thrd == 2) {
      L3 = 0;
      for (int d = 0; d < dim; d++) L3 += d_tang1[d] * normGrad[1 + d];
      L3 *= meanVel[0];
    }

    if (thrd == 3) {
      L4 = 0.;
      if (dim == 3) {
        for (int d = 0; d < dim; d++) L4 += d_tang2[d] * normGrad[1 + d];
        L4 *= meanVel[0];
      }
    }

    if (thrd == 0) {
      L5 = 0.;
      for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
      L5 = dpdn + meanUp[0] * speedSound * L5;
      L5 *= meanVel[0] + speedSound;
    }

    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      L6 = meanVel[0] * normGrad[num_equation - 1];
    }

    // estimate ingoing characteristic
    if (thrd == elDof - 1) {
      const double sigma = speedSound / refLength;
      L1 = -sigma * (meanVel[0] - inputState[0] / meanUp[0] / area);
      L1 *= meanUp[0] * speedSound;
    }
    MFEM_SYNC_THREAD;

    if (thrd == 0) d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
    if (thrd == 1) d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
    if (thrd == 2) d3 = L3;
    if (thrd == 4) d4 = L4;
    if (thrd == 5) d5 = 0.5 * (L5 + L1);
    MFEM_SYNC_THREAD;

    if (thrd == 0) bdrFlux[0] = d1;
    if (thrd == 1) bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
    if (thrd == 2) bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
    if (thrd == elDof - 1 && dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
    if (thrd == 1 + dim) {
      bdrFlux[1 + dim] = meanUp[0] * meanVel[0] * d2;
      bdrFlux[1 + dim] += meanUp[0] * meanVel[1] * d3;
      if (dim == 3) bdrFlux[1 + dim] += meanUp[0] * meanVel[2] * d4;
      bdrFlux[1 + dim] += meanK * d1 + d5 / (gamma - 1.);
    }
    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      bdrFlux[num_equation - 1] = d1 * u1[num_equation - 1] / u1[0] + u1[0] * L6;
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) u2[thrd] = boundaryU[thrd + n * num_equation];
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) uN[thrd] = u2[thrd];
    if (thrd > 0 && thrd < dim + 1) {
      uN[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 1) uN[1] += u2[1 + d] * unitNorm[d];
        if (thrd == 2) uN[2] += u2[1 + d] * d_tang1[d];
        if (dim == 3 && thrd == 3) uN[3] += u2[1 + d] * d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) newU[thrd] = uN[thrd] - dt * bdrFlux[thrd];
    MFEM_SYNC_THREAD;

    double sum = 0.;
    if (thrd < dim) {
      for (int j = 0; j < dim; j++) sum += d_inv[thrd + j * dim] * newU[1 + j];
    }
    MFEM_SYNC_THREAD;
    if (thrd < dim) newU[thrd + 1] = sum;
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) boundaryU[thrd + n * num_equation] = newU[thrd];
  }

  static MFEM_HOST_DEVICE void computeNR_PW_SubMF(const int &thrd, const int &n, const double *u1, const double *gradUp,
                                                  const double *meanUp, const double &dt, double *u2, double *boundaryU,
                                                  const double *inputState, const double *nor, const double *d_tang1,
                                                  const double *d_tang2, const double *d_inv, const double &refLength,
                                                  const double &area, const double &gamma, const double &Rg,
                                                  const int &elDof, const int &dim, const int &num_equation,
                                                  const Equations &eqSystem) {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[20];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK, dpdn;
    MFEM_SHARED double L1, L2, L3, L4, L5, L6;
    MFEM_SHARED double d1, d2, d3, d4, d5;
    MFEM_SHARED double bdrFlux[20], uN[20], newU[20];

    if (thrd < dim) unitNorm[thrd] = nor[thrd];
    if (thrd == num_equation - 1) {
      mod = 0.;
      for (int d = 0; d < dim; d++) mod += nor[d] * nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;

    if (thrd < dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;

    // mean vel. outlet tangent and normal
    if (thrd < 3) {
      meanVel[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 0) meanVel[0] += unitNorm[d] * meanUp[d + 1];
        if (thrd == 1) meanVel[1] += d_tang1[d] * meanUp[d + 1];
        if (dim == 3 && thrd == dim - 1) {
          meanVel[2] += d_tang2[d] * meanUp[d + 1];
        }
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) {
      normGrad[thrd] = 0.;
      for (int d = 0; d < dim; d++) normGrad[thrd] += unitNorm[d] * gradUp[thrd + d * num_equation];
    }
    MFEM_SYNC_THREAD;
    if (thrd == 0) {
      // NOTE: assumes DryAir!
      double T = DryAir::temperatureFromConservative(u1, gamma, Rg, dim, num_equation);
      dpdn = Rg * (T * normGrad[0] + u1[0] * normGrad[1 + dim]);
    }
    MFEM_SYNC_THREAD;

    if (thrd == 0) speedSound = sqrt(gamma * Rg * meanUp[1 + dim]);  // assumes perfect gas
    if (thrd == elDof - 1) {
      meanK = 0.;
      for (int d = 0; d < dim; d++) meanK += meanUp[1 + d] * meanUp[1 + d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;

    // compute outgoing characteristics
    if (thrd == 1) {
      L2 = speedSound * speedSound * normGrad[0] - dpdn;
      L2 *= meanVel[0];
    }

    if (thrd == 2) {
      L3 = 0;
      for (int d = 0; d < dim; d++) L3 += d_tang1[d] * normGrad[1 + d];
      L3 *= meanVel[0];
    }

    if (thrd == 3) {
      L4 = 0.;
      if (dim == 3) {
        for (int d = 0; d < dim; d++) L4 += d_tang2[d] * normGrad[1 + d];
        L4 *= meanVel[0];
      }
    }

    if (thrd == 0) {
      L5 = 0.;
      for (int d = 0; d < dim; d++) L5 += unitNorm[d] * normGrad[1 + d];
      L5 = dpdn + meanUp[0] * speedSound * L5;
      L5 *= meanVel[0] + speedSound;
    }

    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      L6 = meanVel[0] * normGrad[num_equation - 1];
    }

    // estimate ingoing characteristic
    if (thrd == elDof - 1) {
      double normVel = 0.;
      for (int d = 0; d < dim; d++) normVel += u1[1 + d] * unitNorm[d];
      normVel /= u1[0];
      const double sigma = speedSound / refLength;
      L1 = -sigma * (normVel - inputState[0] / meanUp[0] / area);
      L1 *= meanUp[0] * speedSound;
    }
    MFEM_SYNC_THREAD;

    if (thrd == 0) d1 = (L2 + 0.5 * (L5 + L1)) / speedSound / speedSound;
    if (thrd == 1) d2 = 0.5 * (L5 - L1) / meanUp[0] / speedSound;
    if (thrd == 2) d3 = L3;
    if (thrd == 4) d4 = L4;
    if (thrd == 5) d5 = 0.5 * (L5 + L1);
    MFEM_SYNC_THREAD;

    if (thrd == 0) bdrFlux[0] = d1;
    if (thrd == 1) bdrFlux[1] = meanVel[0] * d1 + meanUp[0] * d2;
    if (thrd == 2) bdrFlux[2] = meanVel[1] * d1 + meanUp[0] * d3;
    if (thrd == elDof - 1 && dim == 3) bdrFlux[3] = meanVel[2] * d1 + meanUp[0] * d4;
    if (thrd == 1 + dim) {
      bdrFlux[1 + dim] = meanUp[0] * meanVel[0] * d2;
      bdrFlux[1 + dim] += meanUp[0] * meanVel[1] * d3;
      if (dim == 3) bdrFlux[1 + dim] += meanUp[0] * meanVel[2] * d4;
      bdrFlux[1 + dim] += meanK * d1 + d5 / (gamma - 1.);
    }
    if (thrd == num_equation - 1 && eqSystem == NS_PASSIVE) {
      bdrFlux[num_equation - 1] = d1 * u1[num_equation - 1] / u1[0] + u1[0] * L6;
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) u2[thrd] = boundaryU[thrd + n * num_equation];
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) uN[thrd] = u2[thrd];
    if (thrd > 0 && thrd < dim + 1) {
      uN[thrd] = 0.;
      for (int d = 0; d < dim; d++) {
        if (thrd == 1) uN[1] += u2[1 + d] * unitNorm[d];
        if (thrd == 2) uN[2] += u2[1 + d] * d_tang1[d];
        if (dim == 3 && thrd == 3) uN[3] += u2[1 + d] * d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) newU[thrd] = uN[thrd] - dt * bdrFlux[thrd];
    MFEM_SYNC_THREAD;

    double sum = 0.;
    if (thrd < dim) {
      for (int j = 0; j < dim; j++) sum += d_inv[thrd + j * dim] * newU[1 + j];
    }
    MFEM_SYNC_THREAD;
    if (thrd < dim) newU[thrd + 1] = sum;
    MFEM_SYNC_THREAD;

    if (thrd < num_equation) boundaryU[thrd + n * num_equation] = newU[thrd];
  }

#endif  // __GPU__
};

#endif  // OUTLETBC_HPP_
