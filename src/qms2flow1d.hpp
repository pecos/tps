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
#include "em_options.hpp"
#include "quasimagnetostatic.hpp"
#include "gslib_interpolator.hpp"

namespace TPS {

class Qms2Flow1d {
 public:
  Qms2Flow1d(Tps *tps);
  ~Qms2Flow1d();
  //  Creates vectors of required size for 1d flow and initializes EM solver
  void initialize(const int n_1d_);
  //  Expands 1d conductivity, solves EM, and reduces 2d Joule heating
  void solve();
  //  Prints the four 1d vectors
  void print_all_1d();

  const Vector &PlasmaConductivity1d() const { return *cond_1d; }
  Vector &PlasmaConductivity1d() { return *cond_1d; }

  const Vector &JouleHeating1d() const { return *joule_1d; }
  Vector &JouleHeating1d() { return *joule_1d; }

  const Vector &Coordinates1d() const { return *z_coords_1d; }
  Vector &Coordinates1d() { return *z_coords_1d; }

  const Vector &TorchRadius1d() const { return *radius_1d; }
  Vector &TorchRadius1d() { return *radius_1d; }

  void set_n_interp(const int n_interp_) { n_interp = n_interp_; }
  void set_length(const double length_) { length = length_; }

  //  Reduces 2d Joule heating by taking radial totals at 1d coordinates
  void set_joule_heating_1d();

  //  Useful for comparison/convergence
  double total_joule_heating_1d() { return linear_integral(*joule_1d, *z_coords_1d); }
  double total_joule_heating_2d() { return qmsa->totalJouleHeating(); }

 private:
  Tps *tpsP_;
  ElectromagneticOptions *em_opts;
  QuasiMagnetostaticSolverAxiSym *qmsa;
  //  Number of points in 1d flow solver
  int n_1d;
  Vector *cond_1d;
  Vector *joule_1d;
  //  1d coordinates are assumed to be sorted
  Vector *z_coords_1d;
  //  Torch radius at z coordinates
  Vector *radius_1d;
  //  Plasma torch length [m]. Joule heating is zero for z > L
  double length = 0.315;

  //  GSLib based interpolator from FE solution
  LineInterpolator *l_interp;
    //  Number of interpolation points for radial integrals
  int n_interp;
  //  Trapezoidal rule
  double linear_integral(Vector values, Vector coords);
  double radial_integral(Vector values, Vector r_coords);

  //  r: radial location, torch_r: torch radius
  //  r_c: modeling parameter 0 <= r_c <= R
  double radial_profile(double r, double torch_r, double r_c) const;
  //  Binary search to find interval
  int find_z_interval(double z) const;
  //  Interpolation of 1d values along centerline via binary search
  //  Unrelated to GSLib interpolation of FE solution
  double interpolate_z(double z, const Vector &values) const;
  //  Multiplies interpolant by radial profile
  double expand_cond_2d(double r, double z) const;

  //  Expands 1d conductivity into 2d via linear interpolation
  //  along z-axis multiplied by a radial profile
  void set_plasma_conductivity_2d();
};

}  // namespace TPS
#endif  // QMS2FLOW1D_HPP_
