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
#ifndef TPS2BOLTZMANN_HPP_
#define TPS2BOLTZMANN_HPP_

/** @file
 * Top-level TPS class
 */

#include <grvy.h>
#include <tps_config.h>

#undef PACKAGE_NAME
#undef PACKAGE_VERSION
#undef PACKAGE_TARNAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING

#include "tps_mfem_wrap.hpp"
#undef PACKAGE_NAME
#undef PACKAGE_VERSION
#undef PACKAGE_TARNAME
#undef PACKAGE_BUGREPORT
#undef PACKAGE_STRING

#include <string>
#include <map>

#include "tps.hpp"


namespace TPS {
    class Tps2Boltzmann{
        public:

        enum Index {
        //! Species densities (up to 6 species) [1/m^3]: TPS --> Bolzmann
        SpeciesDensities=0,
        //! Amplitute of electric field [V/m]: TPS --> Bolzmann
        ElectricField=1,
        //! Heavy temperature [K]: TPS --> Bolzmann
        HeavyTemperature=2,
        //! Electron temperature [eV]: TPS <--> Bolzmann
        ElectronTemperature=3,
        //! Electron mobility: TPS <-- Bolzmann
        ElectronMobility=4,
        //! Electron diffusion: TPS <-- Bolzmann
        ElectronDiffusion=5,
        //! Reaction rates: TPS <-- Bolzmann
        ReactionRates=6
        };

        //! Total number of fields
        const std::size_t NIndexes;


        Tps2Boltzmann(Tps *tps);

        const mfem::ParGridFunction & All() const;
        mfem::ParGridFunction & All();

        const mfem::ParGridFunction & Field(Index index) const;
        mfem::ParGridFunction & Field(Index index);

        ~Tps2Boltzmann();


        private:
        Tps *tps_;

        int nspecies_;
        int nreactions_;
        int nvfields;
        int nfields_;
        mfem::Array<int> offsets;

        mfem::FiniteElementCollection * fec_;
        mfem::ParFiniteElementSpace * all_fes_;
        mfem::ParFiniteElementSpace * species_densities_fes_;
        mfem::ParFiniteElementSpace * scalar_fes_;
        mfem::ParFiniteElementSpace * reaction_rates_fes_;

        //! all variables
        mfem::ParGridFunction * all_; 
        //! array of fields see *Index for how to address this
        mfem::ParGridFunction ** fields_; 
    };
}

#endif