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

#ifndef QMS2FLOW1D_HPP_
#define QMS2FLOW1D_HPP_

#include "tps_mfem_wrap.hpp"
#include "tps.hpp"

namespace TPS {

// Need to:
// [X] - Initialize vector of integer size with argument given in Python
//       (Number of 1D grid points)
// [X] - Form vector of that size in C++
//       (To receive plasma conductivity/send Joule heating)
// [X] - Receive 1D plasma conductivity from Python
// [ ] - Expand to 2D plasma conductivity in C++
// [ ] - Solve for 2D Joule heating with axisymmetric solver
// [ ] - Reduce to 1D Joule heating in C++
// [X] - Send 1D Joule heating to Python

class Qms2Flow1d {
 public:
  Qms2Flow1d(Tps *tps) {}
  void initialize(int n_1d_);
  void print_all_1d();

  const Vector &PlasmaConductivity1d() const { return plasma_conductivity_1d; }
  Vector &PlasmaConductivity1d() { return plasma_conductivity_1d; }

  const Vector &JouleHeating1d() const { return joule_heating_1d; }
  Vector &JouleHeating1d() { return joule_heating_1d; }

  const Vector &Coordinates1d() const { return coordinates_1d; }
  Vector &Coordinates1d() { return coordinates_1d; }

  const Vector &Radius1d() const { return radius_1d; }
  Vector &Radius1d() { return radius_1d; }

 private:
  int n_1d;
  Vector plasma_conductivity_1d;
  Vector joule_heating_1d;
  Vector coordinates_1d;
  Vector radius_1d;

  //  r is radial location, R is torch radius, r_c is modeling parameter
  double radial_profile(double r, double R, double r_c);

  //  Binary search to find interval
  int find_z_interval(double z, Vector &z_coords);

  //  Interpolation of 1d values along centerline
  double interpolate_z(double z, Vector &z_coords, Vector &values);
};

}  // namespace TPS
#endif  // QMS2FLOW1D_HPP_
