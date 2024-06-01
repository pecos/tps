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
#ifndef CASES_HPP_
#define CASES_HPP_

/** @file
 * @brief A place for case-specific bc and ic. All case names specified in the input
 * file must be added to if-list in cases.cpp along with the actual ic/bc.
 */

#include <assert.h>
#include <hdf5.h>

#include <string>

#include "tps_mfem_wrap.hpp"

typedef std::function<void(const mfem::Vector &, double, mfem::Vector &)> vfptr;
vfptr vel_ic(std::string ic_string_);
vfptr vel_bc(std::string type);

typedef std::function<double(const mfem::Vector &, double)> sfptr;
sfptr temp_ic(std::string ic_string_);
sfptr temp_bc(std::string type);

void vel_exact_tgv2d(const mfem::Vector &x, double t, mfem::Vector &u);
void vel_tgv2d_uniform(const mfem::Vector &x, double t, mfem::Vector &u);
void vel_channel(const mfem::Vector &x, double t, mfem::Vector &u);
void vel_exact_pipe(const mfem::Vector &x, double t, mfem::Vector &u);

double temp_rt3d(const mfem::Vector &x, double t);

#endif  // CASES_HPP_
