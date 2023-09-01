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

/** @file
 * @copydoc gpu_constructor.hpp
 */

#include "gpu_constructor.hpp"

namespace gpu {

#if defined(_CUDA_) || defined(_HIP_)
//---------------------------------------------------
// Constructors first
//---------------------------------------------------
__global__ void instantiateDeviceDryAir(const DryAirInput inputs, int _dim, int nvel, void *mix) {
  mix = new (mix) DryAir(inputs, _dim, nvel);
}

__global__ void instantiateDevicePerfectMixture(const PerfectMixtureInput inputs, int _dim, int nvel, void *mix) {
  mix = new (mix) PerfectMixture(inputs, _dim, nvel);
}

__global__ void instantiateDeviceLteMixture(WorkingFluid f, int _dim, int nvel, double pc,
                                            TableInput energy_table_input, TableInput R_table_input,
                                            TableInput c_table_input, TableInput T_table_input, void *mix) {
  mix = new (mix) LteMixture(f, _dim, nvel, pc, energy_table_input, R_table_input, c_table_input, T_table_input);
}

__global__ void instantiateDeviceDryAirTransport(GasMixture *mixture, const double viscosity_multiplier,
                                                 const double bulk_viscosity, const double C1, const double S0,
                                                 const double Pr, void *transport) {
  transport = new (transport) DryAirTransport(mixture, viscosity_multiplier, bulk_viscosity, C1, S0, Pr);
}

__global__ void instantiateDeviceConstantTransport(GasMixture *mixture, const constantTransportData inputs,
                                                   void *trans) {
  trans = new (trans) ConstantTransport(mixture, inputs);
}
__global__ void instantiateDeviceArgonMinimalTransport(GasMixture *mixture, const ArgonTransportInput inputs,
                                                       void *trans) {
  trans = new (trans) ArgonMinimalTransport(mixture, inputs);
}
__global__ void instantiateDeviceArgonMixtureTransport(GasMixture *mixture, const ArgonTransportInput inputs,
                                                       void *trans) {
  trans = new (trans) ArgonMixtureTransport(mixture, inputs);
}

__global__ void instantiateDeviceLteTransport(GasMixture *mixture, TableInput mu_table_input,
                                              TableInput kappa_table_input, TableInput sigma_table_input, void *trans) {
  trans = new (trans) LteTransport(mixture, mu_table_input, kappa_table_input, sigma_table_input);
}

__global__ void instantiateDeviceMixingLengthTransport(GasMixture *mixture, const mixingLengthTransportData inputs,
                                                       TransportProperties *mol_trans, void *trans) {
  // can't dynamic_cast on the device, so regular cast here
  MolecularTransport *temporary_transport = (MolecularTransport *)mol_trans;
  trans = new (trans) MixingLengthTransport(mixture, inputs, temporary_transport);
}

__global__ void instantiateDeviceFluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                        const int _num_equation, const int _dim, bool axisym, int sgs_model,
                                        double sgs_floor, double sgs_const, viscositySpongeData vsd, void *f) {
  f = new (f)
      Fluxes(_mixture, _eqSystem, _transport, _num_equation, _dim, axisym, sgs_model, sgs_floor, sgs_const, vsd);
}

__global__ void instantiateDeviceRiemann(int _num_equation, GasMixture *_mixture, Equations _eqSystem,
                                         Fluxes *_fluxClass, bool _useRoe, bool axisym, void *r) {
  r = new (r) RiemannSolver(_num_equation, _mixture, _eqSystem, _fluxClass, _useRoe, axisym);
}

__global__ void instantiateDeviceChemistry(GasMixture *mixture, const ChemistryInput inputs, void *chem) {
  chem = new (chem) Chemistry(mixture, inputs);
}

__global__ void instantiateDeviceNetEmission(const RadiationInput inputs, void *radiation) {
  radiation = new (radiation) NetEmission(inputs);
}

//---------------------------------------------------
// And then destructors
//---------------------------------------------------
__global__ void freeDeviceFluxes(Fluxes *f) { f->~Fluxes(); }
__global__ void freeDeviceRiemann(RiemannSolver *r) { r->~RiemannSolver(); }
__global__ void freeDeviceMixture(GasMixture *mix) { mix->~GasMixture(); }
__global__ void freeDeviceTransport(TransportProperties *transport) { transport->~TransportProperties(); }
__global__ void freeDeviceChemistry(Chemistry *chem) {
  if (chem != NULL) chem->~Chemistry();
}
__global__ void freeDeviceRadiation(Radiation *radiation) {
  if (radiation != NULL) radiation->~Radiation();
}

#endif  // cuda or hip
}  // namespace gpu
