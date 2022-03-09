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

#include <mfem.hpp>
#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "transport_properties.hpp"

using namespace mfem;

// TODO: In order to avoid repeated primitive variable evaluation,
// Fluxes and RiemannSolver should take Vector Up (on the evaulation point) as input argument,
// and FaceIntegrator should have a pointer to ParGridFunction *Up.
// Also should be able to have Up more than number of equations,
// while gradUp is evaluated only for the first num_equation variables.
// Need to discuss further.
class Fluxes {
 private:
  GasMixture *mixture;

  TransportProperties *transport;

  Equations &eqSystem;

  const int &dim;

  const int &num_equation;
  double Rg;
  Vector gradT;
  Vector vel;
  Vector vtmp;
  DenseMatrix stress;

 public:
  Fluxes(GasMixture *_mixture, Equations &_eqSystem, TransportProperties *_transport, const int &_num_equation,
         const int &_dim);

  Equations GetEquationSystem() { return eqSystem; }

  void ComputeTotalFlux(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux);

  void ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux);

  void ComputeViscousFluxes(const Vector &state, const DenseMatrix &gradUp, DenseMatrix &flux);

  // Compute the split fersion of the flux for SBP operations
  // Output matrices a_mat, c_mat need not have the right size
  void ComputeSplitFlux(const Vector &state, DenseMatrix &a_mat, DenseMatrix &c_mat);

  // GPU functions
  void convectiveFluxes_gpu(const Vector &x, DenseTensor &flux, const Equations &eqSystem, GasMixture *mixture,
                                   const int &dof, const int &dim, const int &num_equation);
  void viscousFluxes_gpu(const Vector &x, ParGridFunction *gradUp, DenseTensor &flux, const Equations &eqSystem,
                                GasMixture *mixture, const ParGridFunction *spaceVaryViscMult,
                                const linearlyVaryingVisc &linViscData, const int &dof, const int &dim,
                                const int &num_equation);

#ifdef _GPU_
  static MFEM_HOST_DEVICE void viscousFlux_gpu(double *vFlux, const double *Un, const double *gradUpn,
                                               const Equations &eqSystem, const TransportModel &transpModel, 
                                               const double *gasParams,
                                               const double &gamma, const double &Rg,
                                               const double &viscMult, const double &bulkViscMult, const double &Pr,
                                               const double &Sc, const int &dim,
                                               const int &num_equation, const int &numActiveSpecies, const int &numSpecies,
                                               const int &thrd, const int &maxThreads) {
    MFEM_SHARED double vel[3], divV, KE[3];
    MFEM_SHARED double stress[3][3];
    MFEM_SHARED double gradT[3];
    MFEM_SHARED double transportBuffer[10]; // WARNING: size set to 10. Check that it is enough!!
    MFEM_SHARED double diffusionVelocity[15 * 3];
    MFEM_SHARED double speciesEnthalpies[15];

    if (thrd < 3) KE[thrd] = 0.;
    if (thrd < dim) KE[thrd] = 0.5 * Un[1 + thrd] * Un[1 + thrd] / Un[0];
    // if(thrd<num_equation) for(int d=0;d<dim;d++) vFlux[thrd+d*num_equation] = 0.;
    for (int eq = thrd; eq < num_equation; eq += maxThreads) {
      for (int d = 0; d < dim; d++) vFlux[eq + d * num_equation] = 0.;
    }
    MFEM_SYNC_THREAD;
    
    { // TODO: transform this to an if statement
      DryAir::computeSpeciesEnthalpies_gpu(&speciesEnthalpies[0],numSpecies,thrd, maxThreads);
    }
    
    switch (transpModel){
      case TransportModel::DRY_AIR_TRNSP:
	{
	  double cp_div_pr = Rg * gamma / (gamma - 1.);
          cp_div_pr /= Pr;
          DryAirTransport::ComputeFluxTransportProperties_gpu(Un,
                                                              gradUpn,
                                                              gasParams,
                                                              transportBuffer,
                                                              diffusionVelocity,
                                                              dim,
                                                              num_equation,
                                                              numActiveSpecies,
                                                              numSpecies,
                                                              Rg,
                                                              cp_div_pr,
                                                              gamma,
                                                              Sc,
                                                              viscMult,
                                                              bulkViscMult,
                                                              thrd, maxThreads );
	}
        break;
      default:
        printf("[ERROR] Fluxes::viscousFlux_gpu() transport model not yet supported.");
        break;
    }
    MFEM_SYNC_THREAD;
    
    const double visc = transportBuffer[GlobalTrnsCoeffs::VISCOSITY];
    double k = transportBuffer[GlobalTrnsCoeffs::HEAVY_THERMAL_CONDUCTIVITY];

    for (int i = thrd; i < dim; i += maxThreads) {
      for (int j = 0; j < dim; j++) {
        stress[i][j] = gradUpn[1 + j + i * num_equation] + gradUpn[1 + i + j * num_equation];
      }
      // temperature gradient
      gradT[i] = gradUpn[1 + dim + i * num_equation];

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
        vFlux[1 + dim + i * num_equation] += visc * vel[j] * stress[i][j];
      }
    }
    MFEM_SYNC_THREAD;
    for (int i = thrd; i < dim; i += maxThreads) {
      vFlux[1 + dim + i * num_equation] += k * gradT[i];
    }
    MFEM_SYNC_THREAD;
    
    // Transport terms
    for (int sp = thrd; sp < numActiveSpecies; sp += maxThreads) {
      for (int d = 0; d < dim; d++) {
        vFlux[2 +dim + sp + d*num_equation] = - Un[dim + 2 + sp] * diffusionVelocity[sp+d*numActiveSpecies];
        vFlux[1 + dim + d * num_equation] -=  speciesEnthalpies[sp] * diffusionVelocity[sp+d*numActiveSpecies];
      }
    }

//     if (eqSystem == NS_PASSIVE) {
//       for (int d = thrd; d < dim; d += maxThreads)
//         vFlux[num_equation - 1 + d * num_equation] += visc / Sc * gradUpn[num_equation - 1 + d * num_equation];
//     }
  }
#endif
};

#endif  // FLUXES_HPP_
