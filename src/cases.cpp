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

// #include "cases.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "M2ulPhyS.hpp"
#include "tps_mfem_wrap.hpp"


/// Used to set the velocity IC (and to check error)
void vel_exact_tgv2d(const Vector &x, double t, Vector &u) {
  const double nu = 1.0;
  const double F = std::exp(-2 * nu * t);

  u(0) = F * std::sin(x[0]) * std::cos(x[1]);
  u(1) = -F * std::cos(x[0]) * std::sin(x[1]);
}

/// Used to set the velocity IC with TG field and uniform
void vel_tgv2d_uniform(const Vector &x, double t, Vector &u) {
  const double u0 = 1.0;
  const double F = 0.1;
  const double PI = 3.14159265359;
  double twoPi = 2.0 * PI;

  u(0) = u0;
  u(1) = 0.0;

  u(0) += +F * std::sin(twoPi * x[0]) * std::cos(twoPi * x[1]);
  u(1) += -F * std::cos(twoPi * x[0]) * std::sin(twoPi * x[1]);
}

/// Used to set the channel IC
void vel_channel(const Vector &x, double t, Vector &u) {
  double PI = 3.14159265359;
  double Lx = 25.0;
  double Ly = 2.0;
  double Lz = 9.4;
  double Umean = 1.0;
  double uInt = 0.1;
  int nModes = 4;
  double uM;
  double ax, by, cz;
  double AA, BB, CC;
  double wall;

  // expects channel height (-1,1)
  wall = (1.0 - std::pow(x(1), 8.0));
  u(0) = Umean * wall;
  u(1) = 0.0;
  u(2) = 0.0;

  for (int n = 1; n <= nModes; n++) {
    ax = 4.0 * PI / Lx * (double)n;
    by = 2.0 * PI / Ly * (double)n;
    cz = 2.0 * PI / Lz * (double)n;

    AA = 1.0;
    BB = 1.0;
    CC = -(AA * ax + BB * by) / cz;

    uM = uInt / (double)n;

    u(0) += uM * AA * cos(ax * (x(0) + (double)(n - 1) * Umean)) * sin(by * x(1)) *
            sin(cz * (x(2) + 0.5 * (double)(n - 1) * Umean)) * wall;
    u(1) += uM * BB * sin(ax * (x(0) + (double)(n - 1) * Umean)) * cos(by * x(1)) *
            sin(cz * (x(2) + 0.5 * (double)(n - 1) * Umean)) * wall;
    u(2) += uM * CC * sin(ax * (x(0) + (double)(n - 1) * Umean)) * sin(by * x(1)) *
            cos(cz * (x(2) + 0.5 * (double)(n - 1) * Umean)) * wall;
  }
}


// typedef int (*fptr)();

/// Add cases to selection here
fptr vel_ic(std::string ic_string_) {
    if (ic_string_ == "tgv2d") {
      return vel_exact_tgv2d;
    } else if (ic_string_ == "tgv2d_uniform") {
      return vel_tgv2d_uniform;
    } else if (ic_string_ == "channel") {
      return vel_channel;
    }
}
