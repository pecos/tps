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

namespace TPS {

class Qms2Flow1d {
 public:
  Qms2Flow1d(Tps *tps);
  ~Qms2Flow1d();
  void initialize(int n_1d_);
  void solve();  //  Set the plasma conductivity and solve the EM field
  void print_all_1d();

  const Vector &PlasmaConductivity1d() const { return *cond_1d; }
  Vector &PlasmaConductivity1d() { return *cond_1d; }

  const Vector &JouleHeating1d() const { return *joule_1d; }
  Vector &JouleHeating1d() { return *joule_1d; }

  const Vector &Coordinates1d() const { return *z_coords_1d; }
  Vector &Coordinates1d() { return *z_coords_1d; }

  const Vector &TorchRadius1d() const { return *radius_1d; }
  Vector &TorchRadius1d() { return *radius_1d; }

 private:
  Tps *tpsP_;
  ElectromagneticOptions *em_opts;
  QuasiMagnetostaticSolverAxiSym *qmsa;

  int n_1d;
  Vector *cond_1d;
  Vector *joule_1d;
  Vector *z_coords_1d;  //  1d coordinates are assumed to be sorted
  Vector *radius_1d;  //  Torch radius at z coordinates

  /** 
   * r: radial location
   * torch_r: torch radius
   * r_c: modeling parameter, 0 <= r_c <= R
   */
  double radial_profile(double r, double torch_r, double r_c) const;
  //  Binary search to find interval
  int find_z_interval(double z) const;
  //  Interpolation of 1d values along centerline
  double interpolate_z(double z, const Vector &values) const;

  double expand_cond_2d(double r, double z) const;

  void set_plasma_conductivity_2d();

  int n_interp;
  void set_joule_heating_1d(const int n_interp);
};

}  // namespace TPS
#endif  // QMS2FLOW1D_HPP_
