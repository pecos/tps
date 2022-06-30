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

#include <hdf5.h>
#include <tps_config.h>

#include <fstream>
#include <mfem/general/forall.hpp>

// #include "BCintegrator.hpp"
#include "argon_transport.hpp"
// #include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "dataStructures.hpp"
// #include "dgNonlinearForm.hpp"
// #include "domain_integrator.hpp"
#include "equation_of_state.hpp"
// #include "faceGradientIntegration.hpp"
// #include "face_integrator.hpp"
#include "fluxes.hpp"
// #include "gradNonLinearForm.hpp"
// #include "io.hpp"
#include "logger.hpp"
#include "mpi_groups.hpp"
// #include "rhs_operator.hpp"
#include "riemann_solver.hpp"
#include "run_configuration.hpp"
// #include "sbp_integrators.hpp"
#include "solver.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;

#if defined(_CUDA_)
__global__ void instantiateDeviceMixture(const DryAirInput inputs, int _dim,
                                         int nvel, GasMixture **mix);
#elif defined(_HIP_)
__global__ void instantiateDeviceMixture(const DryAirInput inputs, int _dim,
                                         int nvel, void *mix);
#endif


void assignMixture(const DryAirInput inputs, const int dim, const int nvel, GasMixture *dMixture);

#endif  // GPU_CONSTRUCTOR_HPP_
