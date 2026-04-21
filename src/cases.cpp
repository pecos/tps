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
 * @copydoc cases.hpp
 */

#include "cases.hpp"

#include <grvy.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "tps_mfem_wrap.hpp"
#include "utils.hpp"

using namespace mfem;

/// generic user-specified vel ic
void velIC_user(const Vector &x, double t, Vector &u) {
  u(0) = 0.0;
  u(1) = 0.0;
  u(2) = 0.0;
}

/// generic user-specified vel bc
void velBC_user(const Vector &x, double t, Vector &u) {
  u(0) = 0.0;
  u(1) = 0.0;
  u(2) = 0.0;
}

/// generic user-specified temp ic
double tempIC_user(const Vector &coords, double t) {
  double temp;
  temp = -1.0;
  return temp;
}

/// Used to set the velocity IC (and to check error)
void vel_exact_tgv2d(const Vector &x, double t, Vector &u) {
  const double nu = 1.0;
  const double F = std::exp(-2 * nu * t);

  u(0) = F * std::sin(x[0]) * std::cos(x[1]);
  u(1) = -F * std::cos(x[0]) * std::sin(x[1]);
  if (u.Size() == 3) u(2) = 0.0;
}

/// Used to set the velocity IC with TG field and uniform
void vel_tgv2d_uniform(const Vector &x, double t, Vector &u) {
  const double u0 = 1.0;
  const double F = 0.1;
  const double PI = 3.14159265359;
  double twoPi = 2.0 * PI;

  u(0) = u0;
  u(1) = 0.0;
  if (u.Size() == 3) u(2) = 0.0;

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

/// Add ic cases to selection here
vfptr vel_ic(std::string ic_string_) {
  if (ic_string_ == "tgv2d") {
    return vel_exact_tgv2d;
  } else if (ic_string_ == "tgv2d_uniform") {
    return vel_tgv2d_uniform;
  } else if (ic_string_ == "channel") {
    return vel_channel;
  } else if (ic_string_ == "user") {
    return velIC_user;
  } else {
    grvy_printf(GRVY_ERROR, "Attempting to use unknown vel_ic");
    exit(ERROR);
    return vel_exact_tgv2d;
  }
}

/// Used to for pipe flow test case
void vel_exact_pipe(const Vector &x, double t, Vector &u) {
  u(0) = 0.0;
  u(1) = 2.0 * (1 - x[0] * x[0]);
}

/// Add bc cases to selection here
vfptr vel_bc(std::string type) {
  if (type == "fully-developed-pipe") {
    return vel_exact_pipe;
  } else if (type == "user") {
    return velBC_user;
  } else {
    grvy_printf(GRVY_ERROR, "Attempting to use unknown vel_bc");
    exit(ERROR);
    return vel_exact_pipe;
  }
}

/// Rayleigh-Taylor ic
double temp_rt3d(const Vector &x, double t) {
  double CC = 0.05;
  double twoPi = 6.28318530718;
  double yWidth = 0.1;
  double yInt, dy, wt;
  double temp, dT;
  double Tlo = 100.0;
  double Thi = 1500.0;

  yInt = std::cos(twoPi * x[0]) + std::cos(twoPi * x[2]);
  yInt *= CC;
  yInt += 4.0;

  dy = x[1] - yInt;
  dT = Thi - Tlo;

  wt = 0.5 * (tanh(-dy / yWidth) + 1.0);
  temp = Tlo + wt * dT;

  return temp;
}

/// Hot/Cold wall channel
double temp_channel(const Vector &coords, double t) {
  double Thi = 400.0;
  double Tlo = 200.0;
  double y = coords(1);
  double temp;
  temp = Tlo + 0.5 * (y + 0.1) * (Thi - Tlo);
  return temp;
}

/// Bouyancy-driven cavity
double temp_lequereBox(const Vector &coords, double t) {
  double Thi = 480.0;
  double Tlo = 120.0;
  double Tmean;
  double x = coords(0);
  double temp;

  Tmean = 0.5 * (Thi + Tlo);
  temp = Tmean + x * (Thi - Tlo);

  return temp;
}

/// Add temp ic cases to selection here
sfptr temp_ic(std::string ic_string_) {
  if (ic_string_ == "rt3D") {
    return temp_rt3d;
  } else if (ic_string_ == "channel") {
    return temp_channel;
  } else if (ic_string_ == "lequere-box") {
    return temp_lequereBox;
  } else if (ic_string_ == "user") {
    return tempIC_user;
  } else {
    grvy_printf(GRVY_ERROR, "Attempting to use unknown temp_ic");
    exit(ERROR);
    return temp_rt3d;
  }
}
