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
/*
 * Implementation of the class Tps2Bolzmann
 */

#include "tps2Boltzmann.hpp"

#ifdef HAVE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#include <cstddef>
#include <cstdlib>

namespace TPS {

class CPUDataRead {
 public:
  CPUDataRead(const mfem::Vector &v) : data_(v.HostRead()), size_(v.Size()), stride_(1) {}
  double *data() const { return const_cast<double *>(data_); }
  size_t size() const { return size_; }
  size_t stride() const { return stride_; }

 private:
  const double *data_;
  size_t size_;
  size_t stride_;
};

class CPUData {
 public:
  CPUData(mfem::Vector &v, bool rw) : data_(rw ? v.HostReadWrite() : v.HostWrite()), size_(v.Size()), stride_(1) {}
  double *data() { return data_; }
  size_t size() const { return size_; }
  size_t stride() const { return stride_; }

 private:
  double *data_;
  size_t size_;
  size_t stride_;
};

Tps2Boltzmann::Tps2Boltzmann(Tps *tps) : NIndexes(7), tps_(tps), all_fes_(nullptr) {
  // Assert we have a couple solver;
  assert(tps->isFlowEMCoupled());

  tps->getRequiredInput("species/numSpecies", nspecies_);
  // TODO(Umberto): Get the number of reactions for the solver
  tps->getRequiredInput("boltzmannInterface/nreactions", nreactions_);
  tps->getRequiredInput("boltzmannInterface/order", order_);
  tps->getRequiredInput("boltzmannInterface/basisType", basis_type_);
  assert(basis_type_ == 0 || basis_type_ == 1);

  offsets.SetSize(NIndexes + 1);
  ncomps.SetSize(NIndexes + 1);
}

void Tps2Boltzmann::init(M2ulPhyS *flowSolver) {
  std::cout << "Tps2Boltzmann::init is called" << std::endl;
  mfem::ParMesh *pmesh(flowSolver->GetMesh());
  fec_ = new mfem::L2_FECollection(order_, pmesh->Dimension(), basis_type_);
  switch (pmesh->Dimension()) {
    case 2:
      nEfieldComps_ = 2;
      break;
    case 3:
      nEfieldComps_ = 6;
      break;
    default:
      std::abort();
  }

  nfields_ = nspecies_ + nEfieldComps_ + 4 + nreactions_;

  all_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nfields_, mfem::Ordering::byNODES);
  species_densities_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nspecies_, mfem::Ordering::byNODES);
  efield_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nEfieldComps_, mfem::Ordering::byNODES);
  scalar_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_);
  reaction_rates_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nreactions_, mfem::Ordering::byNODES);

  list_fes_ = new mfem::ParFiniteElementSpace *[NIndexes + 1];
  list_fes_[Index::ElectricField] = efield_fes_;
  ncomps[Index::ElectricField] = nEfieldComps_;
  list_fes_[Index::SpeciesDensities] = species_densities_fes_;
  ncomps[Index::SpeciesDensities] = nspecies_;
  list_fes_[Index::HeavyTemperature] = scalar_fes_;
  ncomps[Index::HeavyTemperature] = 1;
  list_fes_[Index::ElectronTemperature] = scalar_fes_;
  ncomps[Index::ElectronTemperature] = 1;
  list_fes_[Index::ElectronMobility] = scalar_fes_;
  ncomps[Index::ElectronMobility] = 1;
  list_fes_[Index::ElectronDiffusion] = scalar_fes_;
  ncomps[Index::ElectronDiffusion] = 1;
  list_fes_[Index::ReactionRates] = reaction_rates_fes_;
  ncomps[Index::ReactionRates] = nreactions_;
  list_fes_[Index::All] = all_fes_;
  ncomps[Index::All] = nfields_;

  mfem::ParGridFunction *all = new mfem::ParGridFunction(all_fes_);
  fields_ = new mfem::ParGridFunction *[NIndexes + 1];
  offsets[0] = 0;
  for (std::size_t index(0); index < NIndexes; ++index) {
    fields_[index] = new mfem::ParGridFunction(list_fes_[index], *all, offsets[index]);
    offsets[index + 1] = offsets[index] + fields_[index]->Size();
  }
  fields_[Index::All] = all;

  // Native spaces
  mfem::FiniteElementCollection *fec_native(flowSolver->GetFEC());
  species_densities_native_fes_ =
      new mfem::ParFiniteElementSpace(pmesh, fec_native, nspecies_, mfem::Ordering::byNODES);
  efield_native_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_native, nEfieldComps_, mfem::Ordering::byNODES);
  scalar_native_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_native);
  reaction_rates_native_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_native, nreactions_, mfem::Ordering::byNODES);

  list_native_fes_ = new mfem::ParFiniteElementSpace *[NIndexes + 1];
  list_native_fes_[Index::ElectricField] = efield_native_fes_;
  list_native_fes_[Index::SpeciesDensities] = species_densities_native_fes_;
  list_native_fes_[Index::HeavyTemperature] = scalar_native_fes_;
  list_native_fes_[Index::ElectronTemperature] = scalar_native_fes_;
  list_native_fes_[Index::ElectronMobility] = scalar_native_fes_;
  list_native_fes_[Index::ElectronDiffusion] = scalar_native_fes_;
  list_native_fes_[Index::ReactionRates] = reaction_rates_native_fes_;

  // Interpolator
  auto assembly_level = mfem::AssemblyLevel::FULL;
  scalar_interpolator_ = new mfem::ParDiscreteLinearOperator(scalar_native_fes_, scalar_fes_);
  scalar_interpolator_->AddDomainInterpolator(new mfem::IdentityInterpolator());
  scalar_interpolator_->SetAssemblyLevel(assembly_level);
  scalar_interpolator_->Assemble();
}

void Tps2Boltzmann::interpolateFromNativeFES(const ParGridFunction &input, Tps2Boltzmann::Index index) {
  if (ncomps[index] == 1) {
    scalar_interpolator_->Mult(input, *(fields_[index]));
  } else {
    const int loc_size_native = list_native_fes_[index]->GetNDofs();
    const int loc_size = list_fes_[index]->GetNDofs();
    for (int icomp(0); icomp < ncomps[index]; ++icomp) {
      const mfem::Vector view_input(const_cast<mfem::ParGridFunction &>(input), icomp * loc_size_native,
                                    loc_size_native);
      mfem::Vector view_field(*(fields_[index]), icomp * loc_size, loc_size);
      scalar_interpolator_->Mult(view_input, view_field);
    }
  }
}

Tps2Boltzmann::~Tps2Boltzmann() {
  // Delete views
  for (std::size_t i(0); i < NIndexes + 1; ++i) delete fields_[i];
  // Delete array
  delete[] fields_;

  // Delete interpolators
  delete scalar_interpolator_;

  // Delete view Native Finite Element Spaces
  delete species_densities_native_fes_;
  delete efield_native_fes_;
  delete scalar_native_fes_;
  delete reaction_rates_native_fes_;
  delete[] list_native_fes_;

  // Delete view Finite Element Spaces
  delete species_densities_fes_;
  delete efield_fes_;
  delete scalar_fes_;
  delete reaction_rates_fes_;
  delete[] list_fes_;

  // Delete monolithic function space
  delete all_fes_;

  // Delete finite element collection
  delete fec_;
}

}  // namespace TPS

#ifdef HAVE_PYTHON
namespace py = pybind11;

namespace tps_wrappers {
void tps2bolzmann(py::module &m) {
  // Can by read in numpy as np.array(data_instance, copy = False)
  py::class_<TPS::CPUDataRead>(m, "CPUDataRead", py::buffer_protocol())
      .def_buffer([](TPS::CPUDataRead &d) -> py::buffer_info {
        return py::buffer_info(d.data(),                                /*pointer to buffer*/
                               sizeof(double),                          /*size of one element*/
                               py::format_descriptor<double>::format(), /*python struct-style format descriptor*/
                               1,                                       /*number of dimensions*/
                               {d.size()},                              /*buffer dimension(s)*/
                               {d.stride() * sizeof(const double)},     /*Stride in bytes for each index*/
                               true /*read only*/);
      });

  py::class_<TPS::CPUData>(m, "CPUData", py::buffer_protocol()).def_buffer([](TPS::CPUData &d) -> py::buffer_info {
    return py::buffer_info(d.data(),                                /*pointer to buffer*/
                           sizeof(double),                          /*size of one element*/
                           py::format_descriptor<double>::format(), /*python struct-style format descriptor*/
                           1,                                       /*number of dimensions*/
                           {d.size()},                              /*buffer dimension(s)*/
                           {d.stride() * sizeof(const double)},     /*Stride in bytes for each index*/
                           false /*writetable*/);
  });

  py::enum_<TPS::Tps2Boltzmann::Index>(m, "t2bIndex")
      .value("SpeciesDensities", TPS::Tps2Boltzmann::Index::SpeciesDensities)
      .value("ElectricField", TPS::Tps2Boltzmann::Index::ElectricField)
      .value("HeavyTemperature", TPS::Tps2Boltzmann::Index::HeavyTemperature)
      .value("ElectronTemperature", TPS::Tps2Boltzmann::Index::ElectronTemperature)
      .value("ElectronMobility", TPS::Tps2Boltzmann::Index::ElectronMobility)
      .value("ElectronDiffusion", TPS::Tps2Boltzmann::Index::ElectronDiffusion)
      .value("ReactionRates", TPS::Tps2Boltzmann::Index::ReactionRates);

  py::class_<TPS::Tps2Boltzmann>(m, "Tps2Boltzmann")
      .def(py::init<TPS::Tps *>())
      .def("HostRead",
           [](const TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
             return std::unique_ptr<TPS::CPUDataRead>(new TPS::CPUDataRead(interface.Field(index)));
           })
      .def("HostWrite",
           [](TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
             return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Field(index), false));
           })
      .def("HostReadWrite", [](TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
        return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Field(index), true));
      });
}
}  // namespace tps_wrappers
#endif
