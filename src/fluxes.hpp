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
#ifndef FLUXES_HPP_
#define FLUXES_HPP_

/*
 * Class that deals with flux operations
 */

#include <tps_config.h>

#include <general/forall.hpp>
#include <mfem.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"

enum Equations { EULER, NS, MHD };

using namespace mfem;

class Fluxes {
 private:
  EquationOfState *eqState;

  Equations &eqSystem;

  const int &dim;

  const int &num_equations;
  double Rg;
  Vector gradT;
  Vector vel;
  Vector vtmp;
  DenseMatrix stress;

 public:
  Fluxes(EquationOfState *_eqState, Equations &_eqSystem, const int &_num_equations, const int &_dim);

  void ComputeTotalFlux(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux);

  void ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux);

  void ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux);

  // Compute the split fersion of the flux for SBP operations
  // Output matrices a_mat, c_mat need not have the right size
  void ComputeSplitFlux(const Vector &state, DenseMatrix &a_mat, DenseMatrix &c_mat);

  // GPU functions
  static void convectiveFluxes_gpu(const Vector &x, DenseTensor &flux, const double &gamma, const int &dof,
                                   const int &dim, const int &num_equation);
  static void viscousFluxes_gpu(const Vector &x, ParGridFunction *gradUp, DenseTensor &flux, const double &gamma,
                                const double &Rg,  // gas constant
                                const double &Pr,  // Prandtl number
                                const double &viscMult, const double &bulkViscMult,
                                const ParGridFunction *spaceVaryViscMult, const linearlyVaryingVisc &linViscData,
                                const int &dof, const int &dim, const int &num_equation);

#ifdef _GPU_
  static MFEM_HOST_DEVICE void viscousFlux_gpu(double *vFlux, const double *Un, const double *gradUpn,
                                               const double &gamma, const double &Rg, const double &viscMult,
                                               const double &bulkViscMult, const double &Pr, const int &thrd,
                                               const int &maxThreads, const int &dim, const int &num_equation) {
    MFEM_SHARED double KE[3], vel[3], divV;
    MFEM_SHARED double stress[3][3];
    MFEM_SHARED double gradT[3];

    if (thrd < 3) KE[thrd] = 0.;
    if (thrd < dim) KE[thrd] = 0.5 * Un[1 + thrd] * Un[1 + thrd] / Un[0];
    // if(thrd<num_equation) for(int d=0;d<dim;d++) vFlux[thrd+d*num_equation] = 0.;
    for (int eq = thrd; eq < num_equation; eq += maxThreads) {
      for (int d = 0; d < dim; d++) vFlux[eq + d * num_equation] = 0.;
    }
    MFEM_SYNC_THREAD;

    const double p = EquationOfState::pressure(&Un[0], &KE[0], gamma, dim, num_equation);
    const double temp = p / Un[0] / Rg;
    double visc = EquationOfState::GetViscosity_gpu(temp);
    visc *= viscMult;
    const double k = EquationOfState::GetThermalConductivity_gpu(visc, gamma, Rg, Pr);

    for (int i = thrd; i < dim; i += maxThreads) {
      for (int j = 0; j < dim; j++) {
        stress[i][j] = gradUpn[1 + j + i * num_equation] + gradUpn[1 + i + j * num_equation];
      }
      // temperature gradient
      gradT[i] = temp * (gradUpn[1 + dim + i * num_equation] / p - gradUpn[0 + i * num_equation] / Un[0]);

      vel[i] = Un[1 + i] / Un[0];
    }
    if (thrd == maxThreads - 1) {
      divV = 0.;
      for (int i = 0; i < dim; i++) divV += gradUpn[1 + i + i * num_equation];
    }
    MFEM_SYNC_THREAD;

    for (int i = thrd; i < dim; i += maxThreads) stress[i][i] += (bulkViscMult - 2. / 3.) * divV;
    MFEM_SYNC_THREAD;

    for (int i = thrd; i < dim; i += maxThreads) {
      for (int j = 0; j < dim; j++) {
        vFlux[1 + i + j * num_equation] = visc * stress[i][j];
        // energy equation
        vFlux[num_equation - 1 + i * num_equation] += visc * vel[j] * stress[i][j];
      }
    }
    MFEM_SYNC_THREAD;
    for (int i = thrd; i < dim; i += maxThreads) {
      vFlux[num_equation - 1 + i * num_equation] += k * gradT[i];
    }
    MFEM_SYNC_THREAD;
  }
#endif
};

#endif  // FLUXES_HPP_
