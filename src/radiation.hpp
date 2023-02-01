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
#ifndef RADIATION_HPP_
#define RADIATION_HPP_

#include <tps_config.h>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

class Radiation {
 protected:
 public:
  MFEM_HOST_DEVICE Radiation() {}

  MFEM_HOST_DEVICE virtual ~Radiation() {}

  // Currently has the minimal format required for NEC model.
  MFEM_HOST_DEVICE virtual double computeEnergySink(const double &T_h) {
    printf("computeEnergySink not implemented");
    assert(false);
    return 0;
  }
};

class NetEmission : public Radiation {
 private:
  const double PI_ = PI;

  // NOTE(kevin): currently only takes tabulated data.
  LinearTable necTable_;

 public:
  MFEM_HOST_DEVICE NetEmission(const RadiationInput &inputs);

  MFEM_HOST_DEVICE ~NetEmission();

  MFEM_HOST_DEVICE double computeEnergySink(const double &T_h) override { return -4.0 * PI_ * necTable_.eval(T_h); }
};

#endif  // RADIATION_HPP_
