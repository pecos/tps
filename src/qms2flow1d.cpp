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

Qms2Flow1d::Qms2Flow1d(Tps *tps) {
  tpsP_ = tps;
  em_opts = nullptr;
  qmsa = nullptr;
  l_interp = nullptr;

  n_1d = 0;
  cond_1d = nullptr;
  joule_1d = nullptr;
  z_coords_1d = nullptr;
  radius_1d = nullptr;
  n_interp = 0;
}

Qms2Flow1d::~Qms2Flow1d() {
  delete cond_1d;
  delete joule_1d;
  delete z_coords_1d;
  delete radius_1d;

  delete em_opts;
  delete qmsa;
  delete l_interp;
}

void Qms2Flow1d::initialize(const int n_1d_) {
  n_1d = n_1d_;
  cond_1d = new Vector(n_1d);
  joule_1d = new Vector(n_1d);
  z_coords_1d = new Vector(n_1d);
  radius_1d = new Vector(n_1d);

  em_opts = new ElectromagneticOptions();
  qmsa = new QuasiMagnetostaticSolverAxiSym(*em_opts, tpsP_);
  l_interp = new LineInterpolator();

  n_interp = 100;

  qmsa->parseSolverOptions();
  qmsa->initialize();
  //  Ensuring nodes is required for GSLib interpolation
  qmsa->getMesh()->EnsureNodes();
}

void Qms2Flow1d::solve() {
  set_plasma_conductivity_2d();
  qmsa->solve();
  set_joule_heating_1d();
}

void Qms2Flow1d::print_all_1d() {
  std::cout << "\n--- 1d Coordinates ---" << endl;
  z_coords_1d->Print();
  std::cout << "\n--- Torch Radius ---" << endl;
  radius_1d->Print();
  std::cout << "\n--- 1d Plasma Conductivity ---" << endl;
  cond_1d->Print();
  std::cout << "\n--- 1d Joule Heating ---" << endl;
  joule_1d->Print();
}

double Qms2Flow1d::radial_profile(double r, double torch_r, double r_c) const {
  //  Constant over [0, r_c]. Zero beyond torch_r
  //  Cosine for continuous connection between the two
  assert(0 <= r && 0 <= torch_r && 0 <= r_c);
  double out = 0;
  if (0 <= r && r <= r_c) {
    out = 1;
  } else if (r_c < r && r < torch_r) {
    out = 1 + cos(M_PI*(r - r_c)/(torch_r - r_c));
    out *= 0.5;
  }
  return out;
}

int Qms2Flow1d::find_z_interval(double z) const {
  int i_min;
  if (z < z_coords_1d->Elem(1)) {
    i_min = 0;
  } else if (z > z_coords_1d->Elem(n_1d - 1)) {
    i_min = n_1d - 2;
  } else {
    i_min = 1;
    int i_max = n_1d - 2;
    while (i_max > i_min + 1) {
      int i_test = round(0.5*(i_max - i_min) + i_min);
      if (z == z_coords_1d->Elem(i_test)) {
        i_min = i_test;
        break;
      } else if (z < z_coords_1d->Elem(i_test)) {
        i_max = i_test;
      } else {
        i_min = i_test;
      }
    }
  }
  return i_min;
}

double Qms2Flow1d::interpolate_z(double z, const Vector &values) const {
  const int ind = find_z_interval(z);
  double interp = (values[ind+1] - values[ind])/(z_coords_1d->Elem(ind+1) - z_coords_1d->Elem(ind));
  interp *= (z - z_coords_1d->Elem(ind));
  interp += values[ind];
  return interp;
}

double Qms2Flow1d::expand_cond_2d(double r, double z) const {
  const double sigma_1d = interpolate_z(z, *cond_1d);
  const double torch_radius = interpolate_z(z, *radius_1d);
  double sigma = radial_profile(r, torch_radius, 0.7*torch_radius);
  sigma *= sigma_1d;
  return sigma;
}

void Qms2Flow1d::set_plasma_conductivity_2d() {
  ParFiniteElementSpace *fes = qmsa->getFESpace();
  ParMesh *pmesh = qmsa->getMesh();

  ParFiniteElementSpace *coord_fes = new ParFiniteElementSpace(pmesh, fes->FEColl(), 2);
  ParGridFunction *coordsDof = new ParGridFunction(coord_fes);
  pmesh->GetNodes(*coordsDof);

  ParGridFunction *pc = qmsa->getPlasmaConductivityGF();
  double *plasma_conductivity_gf = pc->HostWrite();

  const int n_nodes = fes->GetNDofs();
  double r, z;
  for (int i = 0; i < n_nodes; i++) {
    r = coordsDof->Elem(i);
    z = coordsDof->Elem(i + n_nodes);
    plasma_conductivity_gf[i] = expand_cond_2d(r, z);
  }

  delete coord_fes;
  delete coordsDof;
}

double Qms2Flow1d::linear_integral(Vector values, Vector coords) {
  assert(values.Size() == coords.Size());
  const int n_minus_one = values.Size() - 1;

  //  x_i - x_i-1
  Vector temp = Vector(coords, 1, n_minus_one);
  Vector diff = temp;
  temp = Vector(coords, 0, n_minus_one);
  diff -= temp;

  //  (v_i + v_i-1)/2 * (x_i - x_i-1)
  temp = Vector(values, 0, n_minus_one);
  Vector summands = temp;
  temp = Vector(values, 1, n_minus_one);
  summands += temp;
  summands *= 0.5;
  summands *= diff;

  return summands.Sum();
}

//  Includes 2pi r Jacobian
double Qms2Flow1d::radial_integral(Vector values, Vector r_coords) {
  Vector input = values;
  input *= r_coords;
  return 2*M_PI*linear_integral(input, r_coords);
}

void Qms2Flow1d::set_joule_heating_1d() {
  double z, torch_r;
  l_interp->setn(n_interp);
  l_interp->initializeFinder(qmsa->getMesh());

  for (int i = 0; i < n_1d; i++) {
    z = z_coords_1d->Elem(i);

    if (z > length) {
      joule_1d->Elem(i) = 0;
    } else {
      torch_r = radius_1d->Elem(i);

      l_interp->getStart()->Elem(0) = 0;
      l_interp->getStart()->Elem(1) = z;
      l_interp->getEnd()->Elem(0) = torch_r;
      l_interp->getEnd()->Elem(1) = z;

      l_interp->setInterpolationPoints();
      l_interp->interpolate(qmsa->getJouleHeatingGF());

      Vector values = l_interp->getSolution();
      Vector r_coords = Vector(l_interp->getInterpolationPoints(), 0, n_interp);

      joule_1d->Elem(i) = radial_integral(values, r_coords);
    }
  }
}

}  // namespace TPS

#ifdef HAVE_PYTHON
namespace py = pybind11;

namespace tps_wrappers {
void qms2flow1d(py::module &m) {
  py::class_<TPS::Qms2Flow1d>(m, "Qms2Flow1d")
      .def(py::init<TPS::Tps *>())
      .def("initialize", &TPS::Qms2Flow1d::initialize, "Initialize the 1d flow-axisymmetric EM interface",
          py::arg("n_1d"))
      .def("solve", &TPS::Qms2Flow1d::solve, "Solve the 1d Joule heating based on the 1d plasma conductivity")
      .def("print_all", &TPS::Qms2Flow1d::print_all_1d,
          "Print the 1d z coordinates, torch radius, plasma conductivity, and Joule heating from C++")
      .def("total_joule_2d", &TPS::Qms2Flow1d::total_joule_heating_2d,
          "Calculate the total Joule heating from the 2d solution")
      .def("total_joule_1d", &TPS::Qms2Flow1d::total_joule_heating_1d,
          "Calculate the total Joule heating from the 1d solution")
      .def("calculate_joule", &TPS::Qms2Flow1d::set_joule_heating_1d,
          "Calculates the 1d Joule heating without resolving the EM problem")
      .def("set_n_interp", &TPS::Qms2Flow1d::set_n_interp,
          "Sets number of interpolation points for radial integrals",
          py::arg("n_interp"))
      .def("set_length", &TPS::Qms2Flow1d::set_length,
          "Set length of plasma torch (Joule heating is zero in the plume)",
          py::arg("length"))
      .def("Coordinates1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Coordinates1d(), false));
          }, "1d z coordinates memory location buffered read/write access")
      .def("TorchRadius1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.TorchRadius1d(), false));
          }, "1d torch radius memory location buffered read/write access")
      .def("PlasmaConductivity1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.PlasmaConductivity1d(), false));
          }, "1d plasma conductivity memory location buffered read/write access")
      .def("JouleHeating1d",
          [](TPS::Qms2Flow1d &interface) {
            return std::unique_ptr<TPS::CPUDataRead>(new TPS::CPUDataRead(interface.JouleHeating1d()));
          }, "1d Joule heating memory location buffered read access");
}

}  // namespace tps_wrappers

#endif
