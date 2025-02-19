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

#include <math.h>

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

void idenity_fun(const Vector &x, Vector &out) {
  for (int i(0); i < x.Size(); ++i) out[i] = x[i];
}

Tps2Boltzmann::Tps2Boltzmann(Tps *tps)
    : NIndexes(7), tps_(tps), all_fes_(nullptr), use_h1fec_(false), save_to_paraview_dc(false), paraview_dc(nullptr) {
  // Assert we have a couple solver;
  assert(tps->isFlowEMCoupled());

  tps->getRequiredInput("species/numSpecies", nspecies_);
  nreactions_ = _countBTEReactions();
  tps->getRequiredInput("boltzmannInterface/order", order_);
  tps->getRequiredInput("boltzmannInterface/basisType", basis_type_);
  assert(basis_type_ == 0 || basis_type_ == 1);

  tps->getRequiredInput("em/current_frequency", EfieldAngularFreq_);
  EfieldAngularFreq_ *= 2. * M_PI;

  use_h1fec_ = tps->getInput("boltzmannInterface/h1fec", false);
  save_to_paraview_dc = tps->getInput("boltzmannInterface/save_to_paraview", false);

  offsets.SetSize(NIndexes + 1);
  ncomps.SetSize(NIndexes + 1);
}

int Tps2Boltzmann::_countBTEReactions() {
  int total_reactions(0);
  int bte_reactions(0);
  tps_->getRequiredInput("reactions/number_of_reactions", total_reactions);
  reaction_eqs_.reserve(total_reactions);
  for (int r(0); r < total_reactions; ++r) {
    std::string basepath("reactions/reaction" + std::to_string(r + 1));
    std::string equation, model;
    tps_->getRequiredInput((basepath + "/equation").c_str(), equation);
    tps_->getRequiredInput((basepath + "/model").c_str(), model);
    if (model == "bte") {
      ++bte_reactions;
      reaction_eqs_.push_back(equation);
    }
  }
  return bte_reactions;
}

void Tps2Boltzmann::init(TPS::PlasmaSolver *flowSolver) {
  mfem::ParMesh *pmesh(flowSolver->getMesh());
  if (use_h1fec_)
    fec_ = new mfem::H1_FECollection(order_, pmesh->Dimension(), basis_type_);
  else
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
  const mfem::FiniteElementCollection *fec_native(flowSolver->getFEC());
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

  scalar_interpolator_to_nativeFES_ = new mfem::ParDiscreteLinearOperator(scalar_fes_, scalar_native_fes_);
  scalar_interpolator_to_nativeFES_->AddDomainInterpolator(new mfem::IdentityInterpolator());
  scalar_interpolator_to_nativeFES_->SetAssemblyLevel(assembly_level);
  scalar_interpolator_to_nativeFES_->Assemble();

  // Spatial coordinates
  spatial_coord_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, pmesh->Dimension(), mfem::Ordering::byNODES);
  spatial_coordinates_ = new mfem::ParGridFunction(spatial_coord_fes_);
  mfem::VectorFunctionCoefficient coord_fun(pmesh->Dimension(),
                                            std::function<void(const Vector &, Vector &)>(idenity_fun));
  spatial_coordinates_->ProjectCoefficient(coord_fun);

  if (save_to_paraview_dc) {
    paraview_dc = new mfem::ParaViewDataCollection("interface", pmesh);
    paraview_dc->SetPrefixPath("BoltzmannInterface");
    paraview_dc->SetDataFormat(VTKFormat::BINARY);
    paraview_dc->RegisterField("Heavy temperature", &(this->Field(TPS::Tps2Boltzmann::Index::HeavyTemperature)));
    paraview_dc->RegisterField("Electron temperature", &(this->Field(TPS::Tps2Boltzmann::Index::ElectronTemperature)));
    paraview_dc->RegisterField("Electric field", &(this->Field(TPS::Tps2Boltzmann::Index::ElectricField)));
    paraview_dc->RegisterField("Species", &(this->Field(TPS::Tps2Boltzmann::Index::SpeciesDensities)));
    paraview_dc->RegisterField("Reaction rates", &(this->Field(TPS::Tps2Boltzmann::Index::ReactionRates)));
  }
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

void Tps2Boltzmann::interpolateToNativeFES(ParGridFunction &output, Index index) {
  if (ncomps[index] == 1) {
    scalar_interpolator_to_nativeFES_->Mult(*(fields_[index]), output);
  } else {
    const int loc_size_native = list_native_fes_[index]->GetNDofs();
    const int loc_size = list_fes_[index]->GetNDofs();
    for (int icomp(0); icomp < ncomps[index]; ++icomp) {
      mfem::Vector view_output(output, icomp * loc_size_native, loc_size_native);
      mfem::Vector view_field(*(fields_[index]), icomp * loc_size, loc_size);
      scalar_interpolator_to_nativeFES_->Mult(view_field, view_output);
    }
  }
}

void Tps2Boltzmann::saveDataCollection(int cycle, double time) {
  if (paraview_dc) {
    paraview_dc->SetCycle(cycle);
    paraview_dc->SetTime(time);
    paraview_dc->Save();
  }
}

Tps2Boltzmann::~Tps2Boltzmann() {
  // Delete views
  for (std::size_t i(0); i < NIndexes + 1; ++i) delete fields_[i];
  // Delete array
  delete[] fields_;

  // Delete interpolators
  delete scalar_interpolator_to_nativeFES_;
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

  delete spatial_coord_fes_;
  delete spatial_coordinates_;

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
      .def("HostReadSpatialCoordinates",
           [](const TPS::Tps2Boltzmann &interface) {
             return std::unique_ptr<TPS::CPUDataRead>(new TPS::CPUDataRead(interface.SpatialCoordinates()));
           })
      .def("HostRead",
           [](const TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
             return std::unique_ptr<TPS::CPUDataRead>(new TPS::CPUDataRead(interface.Field(index)));
           })
      .def("HostWrite",
           [](TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
             return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Field(index), false));
           })
      .def("HostReadWrite",
           [](TPS::Tps2Boltzmann &interface, TPS::Tps2Boltzmann::Index index) {
             return std::unique_ptr<TPS::CPUData>(new TPS::CPUData(interface.Field(index), true));
           })
      .def("EfieldAngularFreq", &TPS::Tps2Boltzmann::EfieldAngularFreq)
      .def("timeStep", &TPS::Tps2Boltzmann::timeStep)
      .def("currentTime", &TPS::Tps2Boltzmann::currentTime)
      .def("Nspecies", &TPS::Tps2Boltzmann::Nspecies)
      .def("NeFiledComps", &TPS::Tps2Boltzmann::NeFieldComps)
      .def("nComponents", &TPS::Tps2Boltzmann::nComponents)
      .def("saveDataCollection", &TPS::Tps2Boltzmann::saveDataCollection, "Save the data collection in Paraview format",
           py::arg("cycle"), py::arg("time"))
      .def("getReactionEquation", &TPS::Tps2Boltzmann::getReactionEquation, "Return the equation of the reaction",
           py::arg("index"));
}
}  // namespace tps_wrappers
#endif
