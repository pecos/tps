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

#include "gpu_constructor.hpp"

namespace gpu {

#if defined(_CUDA_)

// CUDA supports device new/delete
// __global__ void instantiateDeviceMixture(const WorkingFluid f, const Equations eq_sys,
//                                          const double viscosity_multiplier, const double bulk_viscosity, int _dim,
//                                          int nvel, GasMixture **mix) {
//  // *mix = new DryAir(f, eq_sys, viscosity_multiplier, bulk_viscosity, _dim, nvel);
//  *mix = new DryAir(inputs, _dim, nvel);
//}
__global__ void instantiateDeviceDryAir(const DryAirInput inputs, int _dim, int nvel, GasMixture **mix) {
  *mix = new DryAir(inputs, _dim, nvel);
}

__global__ void instantiateDeviceDryAirTransport(GasMixture *mixture, const double viscosity_multiplier,
                                                 const double bulk_viscosity, TransportProperties **trans) {
  *trans = new DryAirTransport(mixture, viscosity_multiplier, bulk_viscosity);
}

__global__ void instantiateDevicePerfectMixture(const PerfectMixtureInput inputs, int _dim, int nvel,
                                                GasMixture **mix) {
  *mix = new PerfectMixture(inputs, _dim, nvel);
}

__global__ void instantiateDeviceConstantTransport(GasMixture *mixture, const constantTransportData inputs,
                                                   TransportProperties **trans) {
  *trans = new ConstantTransport(mixture, inputs);
}

__global__ void instantiateDeviceArgonMinimalTransport(GasMixture *mixture, const ArgonTransportInput inputs,
                                                       TransportProperties **trans) {
  *trans = new ArgonMinimalTransport(mixture, inputs);
}

__global__ void instantiateDeviceArgonMixtureTransport(GasMixture *mixture, const ArgonTransportInput inputs,
                                                       TransportProperties **trans) {
  *trans = new ArgonMixtureTransport(mixture, inputs);
}

__global__ void instantiateDeviceFluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                        const int _num_equation, const int _dim, bool axisym, Fluxes **f) {
  *f = new Fluxes(_mixture, _eqSystem, _transport, _num_equation, _dim, axisym);
}

__global__ void instantiateDeviceRiemann(int _num_equation, GasMixture *_mixture, Equations _eqSystem,
                                         Fluxes *_fluxClass, bool _useRoe, bool axisym, RiemannSolver **r) {
  *r = new RiemannSolver(_num_equation, _mixture, _eqSystem, _fluxClass, _useRoe, axisym);
}

__global__ void instantiateDeviceChemistry(GasMixture *mixture, const ChemistryInput inputs, Chemistry **chem) {
  *chem = new Chemistry(mixture, inputs);
}

__global__ void freeDeviceMixture(GasMixture *mix) { delete mix; }
__global__ void freeDeviceTransport(TransportProperties *trans) { delete trans; }
__global__ void freeDeviceFluxes(Fluxes *f) { delete f; }
__global__ void freeDeviceRiemann(RiemannSolver *r) { delete r; }
__global__ void freeDeviceChemistry(Chemistry *chem) { delete chem; }

#elif defined(_HIP_)

// __global__ void instantiateDeviceMixture(const WorkingFluid f, const Equations eq_sys,
//                                          const double viscosity_multiplier, const double bulk_viscosity, int _dim,
//                                          int nvel, void *mix) {
//  mix = new (mix) DryAir(inputs, _dim, nvel);
//}
__global__ void instantiateDeviceDryAir(const DryAirInput inputs, int _dim, int nvel, void *mix) {
  mix = new (mix) DryAir(inputs, _dim, nvel);
}

__global__ void instantiateDeviceDryAirTransport(GasMixture *mixture, const double viscosity_multiplier,
                                                 const double bulk_viscosity, void *transport) {
  transport = new (transport) DryAirTransport(mixture, viscosity_multiplier, bulk_viscosity);
}

__global__ void instantiateDevicePerfectMixture(const PerfectMixtureInput inputs, int _dim, int nvel, void *mix) {
  mix = new (mix) PerfectMixture(inputs, _dim, nvel);
}

__global__ void instantiateDeviceFluxes(GasMixture *_mixture, Equations _eqSystem, TransportProperties *_transport,
                                        const int _num_equation, const int _dim, bool axisym, void *f) {
  f = new (f) Fluxes(_mixture, _eqSystem, _transport, _num_equation, _dim, axisym);
}

__global__ void instantiateDeviceRiemann(int _num_equation, GasMixture *_mixture, Equations _eqSystem,
                                         Fluxes *_fluxClass, bool _useRoe, bool axisym, void *r) {
  r = new (r) RiemannSolver(_num_equation, _mixture, _eqSystem, _fluxClass, _useRoe, axisym);
}

__global__ void freeDeviceMixture(GasMixture *mix) {
  mix->~GasMixture();
}  // explicit destructor call b/c placement new above
__global__ void freeDeviceTransport(TransportProperties *transport) { transport->~TransportProperties(); }
__global__ void freeDeviceFluxes(Fluxes *f) { f->~Fluxes(); }
__global__ void freeDeviceRiemann(RiemannSolver *r) { r->~RiemannSolver(); }

#endif  // defined(_CUDA_)

// NOTE(kevin): Do not use this. For some unknown reason, this wrapper causes a memory issue, at a random place far
// after this instantiation.
void assignMixture(const DryAirInput inputs, const int dim, const int nvel, GasMixture *dMixture) {
#if defined(_CUDA_)
  GasMixture **d_mixture_tmp;
  cudaMalloc((void **)&d_mixture_tmp, sizeof(GasMixture **));
  instantiateDeviceDryAir<<<1, 1>>>(inputs, dim, nvel, d_mixture_tmp);
  cudaMemcpy(&dMixture, d_mixture_tmp, sizeof(GasMixture *), cudaMemcpyDeviceToHost);
  cudaFree(d_mixture_tmp);
#elif defined(_HIP_)
  hipMalloc((void **)&dMixture, sizeof(DryAir));
  instantiateDeviceDryAir<<<1, 1>>>(inputs, dim, nvel, dMixture);
#endif
}

}  // namespace gpu
