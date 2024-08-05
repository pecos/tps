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

#include "qms2flow1d.hpp"
#include "data_exchange_utils.hpp"

#ifdef HAVE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#include <iostream>
#include <cmath>
#include <assert.h>

namespace TPS {

void Qms2Flow1d::initialize(int n_1d_) {
  n_1d = n_1d_;
  plasma_conductivity_1d = mfem::Vector(n_1d_);
  joule_heating_1d = mfem::Vector(n_1d_);
  coordinates_1d = mfem::Vector(n_1d_);
  radius_1d = mfem::Vector(n_1d_);
}

void print_vec(Vector &vec) {
  for (int i = 0; i < vec.Size(); i++) {
    std::cout << i << ": " << vec[i] << endl;
  }
}

void Qms2Flow1d::print_all_1d() {
  std::cout << "---\nCoordinates:\n---" << endl;
  print_vec(coordinates_1d);
  std::cout << "---\nRadius:\n---" << endl;
  print_vec(radius_1d);
  std::cout << "---\nPlasma Conductivity:\n---" << endl;
  print_vec(plasma_conductivity_1d);
  std::cout << "---\nJoule Heating:\n---" << endl;
  print_vec(joule_heating_1d);
}

double Qms2Flow1d::radial_profile(double r, double R, double r_c) {
  //  Constant over [0, r_c]. Zero beyond torch radius R
  //  Cosine for cont. connection between the two
  assert(0 <= r);
  double out = 0;
  if (0 <= r && r <= r_c) {
    out = 1;
  } else if (r_c < r && r < R) {
    out = 1 + cos(M_PI*(r - r_c)/(R - r_c));
    out *= 0.5;
  }
  return out;
}

int Qms2Flow1d::find_z_interval(double z, Vector &z_coords) {
  int n_points = z_coords.Size(), i_min;
  if (z < z_coords[1]) {
    i_min = 0;
  } else if (z > z_coords[n_points - 1]) {
    i_min = n_points - 2;
  } else {
    i_min = 1;
    int i_max = n_points - 2;
    while (i_max > i_min + 1) {
      int i_test = round(0.5*(i_max - i_min) + i_min);
      if (z == z_coords[i_test]) {
        i_min = i_test;
        break;
      } else if (z < z_coords[i_test]) {
        i_max = i_test;
      } else {
        i_min = i_test;
      }
    }
  }
  return i_min;
}

double Qms2Flow1d::interpolate_z(double z, Vector &z_coords, Vector &values) {
  int ind = Qms2Flow1d::find_z_interval(z, z_coords);
  double interp = (values[ind+1] - values[ind])/(z_coords[ind+1] - z_coords[ind]);
  interp *= (z - z_coords[ind]);
  interp += values[ind];
  return interp;
}

}  // namespace TPS

#ifdef HAVE_PYTHON
namespace py = pybind11;

namespace tps_wrappers {
void qms2flow1d(py::module &m) {
  py::class_<TPS::Qms2Flow1d>(m, "Qms2Flow1d")
      .def(py::init<TPS::Tps *>())
      .def("initialize", &TPS::Qms2Flow1d::initialize)
      .def("print_all", &TPS::Qms2Flow1d::print_all_1d)
      .def("Coordinates1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Coordinates1d(), false));
          })
      .def("Radius1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Radius1d(), false));
          })
      .def("PlasmaConductivity1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.PlasmaConductivity1d(), false));
          })
      .def("JouleHeating1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUDataRead>(new TPS::CPUDataRead(interface.JouleHeating1d()));
          });
}

}  // namespace tps_wrappers

#endif
