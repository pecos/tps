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

#ifndef GPU_CONSTRUCTOR_HPP_
#define GPU_CONSTRUCTOR_HPP_

/** @file
 * @brief Contains utilities for aiding in instantiating various
 * classes on the device.
 *
 * These utilities should be used, for example, for polymorphic
 * classes with virtual functions that need to be called on the
 * device.  In this situation, for the virtual function table to be
 * correct, the class object must be constructed on the device.  To
 * achieve this, the functions declared in this file use placement new
 * (since HIP does not support device-side new at this time; cuda
 * does, but to achieve a unified approach, it is not used here).
 * Because of the use of placement new, the required memory must be
 * explicitly allocated prior to the placement new, and then freed
 * after explicitly calling the relevant destructor on the device.
 * This file also provides macros for this.
 */

#include <tps_config.h>

#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "gas_transport.hpp"
#include "lte_mixture.hpp"
#include "lte_transport_properties.hpp"
#include "mixing_length_transport.hpp"
#include "radiation.hpp"
#include "riemann_solver.hpp"
#include "transport_properties.hpp"

/*!
  @def tpsGpuMalloc(ptr, size)
  Translates to a call to cudaMalloc(ptr, size) or hipMalloc(ptr, size)
  depending on if cuda or hip is being used.  Otherwise compiles
  to assert(false).

  @def tpsGpuFree(ptr)
  Translates to a call to cudaFree(ptr) or hipFree(ptr) depending on
  if cuda or hip is being used.  Otherwise compiles to assert(false).
*/
#if defined(_CUDA_)
// cuda malloc and free
#define tpsGpuMalloc(ptr, size) cudaMalloc(ptr, size)
#define tpsGpuFree(ptr) cudaFree(ptr)
#elif defined(_HIP_)
// hip malloc and free
#define tpsGpuMalloc(ptr, size) hipMalloc(ptr, size)
#define tpsGpuFree(ptr) hipFree(ptr)
#else
// these should be unreachable if properly used, if not, assert(false)
#define tpsGpuMalloc(ptr, size) assert(false)
#define tpsGpuFree(ptr) assert(false)
#endif

namespace gpu {

#if defined(_CUDA_) || defined(_HIP_)
//! Instantiate DryAir object on the device with placement new
__global__ void instantiateDeviceDryAir(const DryAirInput inputs, int _dim, int nvel, void *mix);

//! Instantiate PerfectMixture object on the device with placement new
__global__ void instantiateDevicePerfectMixture(const PerfectMixtureInput inputs, int _dim, int nvel, void *mix);

//! Instantiate LteMixture object on the device with placement new
__global__ void instantiateDeviceLteMixture(WorkingFluid f, int _dim, int nvel, double pc,
                                            TableInput energy_table_input, TableInput R_table_input,
                                            TableInput c_table_input, TableInput T_table_input, void *mix);

//! Instantiate DryAirTransport object on the device with placement new
__global__ void instantiateDeviceDryAirTransport(GasMixture *mixture, const double viscosity_multiplier,
                                                 const double bulk_viscosity, const double C1, const double S0,
                                                 const double Pr, void *transport);

//! Instantiate ConstantTransport object on the device with placement new
__global__ void instantiateDeviceConstantTransport(GasMixture *mixture, const constantTransportData inputs,
                                                   void *trans);

//! Instantiate GasMinimalTransport object on the device with placement new
__global__ void instantiateDeviceGasMinimalTransport(GasMixture *mixture, const GasTransportInput inputs, void *trans);

//! Instantiate GasMixtureTransport object on the device with placement new
__global__ void instantiateDeviceGasMixtureTransport(GasMixture *mixture, const GasTransportInput inputs, void *trans);

//! Instantiate ConstantTransport object on the device with placement new
__global__ void instantiateDeviceLteTransport(GasMixture *mixture, TableInput mu_table_input,
                                              TableInput kappa_table_input, TableInput sigma_table_input, void *trans);

//! Instantiate MixingLengthTransport object on the device with placement new
__global__ void instantiateDeviceMixingLengthTransport(GasMixture *mixture, const mixingLengthTransportData inputs,
                                                       TransportProperties *mol_trans, void *trans);

//! Instantiate Fluxes object on the device with placement new
__global__ void instantiateDeviceFluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                        const int _num_equation, const int _dim, bool axisym, int sgs_model,
                                        double sgs_floor, double sgs_const, viscositySpongeData vsd, void *f);

//! Instantiate RiemannSolver object on the device with placement new
__global__ void instantiateDeviceRiemann(int _num_equation, GasMixture *_mixture, Equations _eqSystem,
                                         Fluxes *_fluxClass, bool _useRoe, bool axisym, void *r);

//! Instantiate Chemistry object on the device with placement new
__global__ void instantiateDeviceChemistry(GasMixture *mixture, const ChemistryInput inputs, void *chem);

//! Instantiate NetEmission object on the device with placement new
__global__ void instantiateDeviceNetEmission(const RadiationInput inputs, void *radiation);

//! Explicit call to GasMixture destructor on the device
__global__ void freeDeviceMixture(GasMixture *mix);

//! Explicit call to TransportProperties destructor on the device
__global__ void freeDeviceTransport(TransportProperties *transport);

//! Explicit call to Fluxes destructor on the device
__global__ void freeDeviceFluxes(Fluxes *f);

//! Explicit call to RiemannSolver destructor on the device
__global__ void freeDeviceRiemann(RiemannSolver *r);

//! Explicit call to Chemistry destructor on the device
__global__ void freeDeviceChemistry(Chemistry *chem);

//! Explicit call to Radiation destructor on the device
__global__ void freeDeviceRadiation(Radiation *radiation);

//! Set the data to a GridFunctionReaction
__global__ void deviceSetGridFunctionReactionData(const double *data, int size, GridFunctionReaction *reaction);
__global__ void deviceSetChemistryReactionData(const double *data, int size, Chemistry *chem);

#endif  // cuda or hip
}  // namespace gpu

#endif  // GPU_CONSTRUCTOR_HPP_
