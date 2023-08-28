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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif


namespace TPS {

Tps2Boltzmann::Tps2Boltzmann(Tps *tps): 
        NIndexes(7),
        tps_(tps) {
        // Assert we have a couple solver;
        assert(tps->isFlowEMCoupled());

        tps->getRequiredInput("species/numSpecies", nspecies_);
        //TODO: Get the number of reactions for the solver
        tps->getRequiredInput("boltzmannInterface/nreactios", nreactions_); 
        nfields_ = nspecies_ + 5 + nreactions_;
        int order;
        tps->getRequiredInput("boltzmannInterface/order", order);
        int basis_type;
        tps->getRequiredInput("boltzmannInterface/basis_type", basis_type);
        assert(basis_type==0 || basis_type==1);


        offsets.SetSize(NIndexes+1);
        mfem::ParMesh * pmesh( tps->getFluidMesh() );
        assert( pmesh );
        fec_ = new mfem::L2_FECollection(order, pmesh->Dimension(), basis_type);

        all_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nfields_, mfem::Ordering::byNODES);
        species_densities_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_,nspecies_ , mfem::Ordering::byNODES);
        scalar_fes_ =  new mfem::ParFiniteElementSpace(pmesh, fec_);
        reaction_rates_fes_ = new mfem::ParFiniteElementSpace(pmesh, fec_, nreactions_ , mfem::Ordering::byNODES);

        mfem::ParFiniteElementSpace ** list_fes = new mfem::ParFiniteElementSpace*[NIndexes];
        list_fes[Index::SpeciesDensities] = species_densities_fes_;
        list_fes[Index::ElectricField] = scalar_fes_;
        list_fes[Index::HeavyTemperature] = scalar_fes_;
        list_fes[Index::ElectronTemperature] = scalar_fes_;
        list_fes[Index::ElectronMobility] = scalar_fes_;
        list_fes[Index::ElectronDiffusion] = scalar_fes_;
        list_fes[Index::ReactionRates] = reaction_rates_fes_;

        all_ = new mfem::ParGridFunction(all_fes_);
        fields_ = new  mfem::ParGridFunction*[NIndexes];
        offsets[0] = 0;
        for (std::size_t index(0); index < NIndexes; ++ index) {
            fields_[index]= new mfem::ParGridFunction(list_fes[index], *all_, offsets[index]);
            offsets[index+1] = offsets[index] + fields_[index]->Size();
        }

        delete[] list_fes;
}

Tps2Boltzmann::~Tps2Boltzmann() {

        // Delete views
        for(std::size_t i(0); i<NIndexes; ++i)
            delete fields_[i];
        // Delete array
        delete[] fields_;

        // Delete monolithic field
        delete all_; 

        // Delete view Finite Element Spaces
        delete species_densities_fes_;
        delete scalar_fes_;
        delete reaction_rates_fes_;

        // Delete monolithic function space
        delete all_fes_;

        // Delete finite element collection
        delete fec_;
}

} /*end namespace*/

#ifdef HAVE_PYTHON
namespace py = pybind11;

namespace tps_wrappers {
    void tps2bolzmann(py::module& m) {
  py::enum_<TPS::Tps2Boltzmann::Index>(m, "t2bIndex")
    .value("SpeciesDensities", TPS::Tps2Boltzmann::Index::SpeciesDensities)
    .value("ElectricField", TPS::Tps2Boltzmann::Index::ElectricField)
    .value("HeavyTemperature", TPS::Tps2Boltzmann::Index::HeavyTemperature)
    .value("ElectronTemperature", TPS::Tps2Boltzmann::Index::ElectronTemperature)
    .value("ElectronMobility", TPS::Tps2Boltzmann::Index::ElectronMobility)
    .value("ElectronDiffusion", TPS::Tps2Boltzmann::Index::ElectronDiffusion)
    .value("ReactionRates", TPS::Tps2Boltzmann::Index::ReactionRates)
  py::class_<TPS::Tps2Boltzmann>(m, "Tps2Bolzmann")
    .def(py::init<TPS::Tps *>());
}
}
#endif