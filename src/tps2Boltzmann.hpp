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

#include <map>
#include <string>

#include "tps.hpp"

class M2ulPhyS;

namespace TPS {
// TODO(Umberto):
/*
- Add dictionaries for species
- Reshape arrays as matrices
*/
class Tps2Boltzmann {
 public:
  enum Index {
    //! Componentes of electric field [V/m]: TPS --> Bolzmann
    ElectricField = 0,
    //! Species densities (up to 6 species) [1/m^3]: TPS --> Bolzmann
    SpeciesDensities = 1,
    //! Heavy temperature [K]: TPS --> Bolzmann
    HeavyTemperature = 2,
    //! Electron temperature [K]: TPS <--> Bolzmann
    ElectronTemperature = 3,
    //! Electron mobility: TPS <-- Bolzmann
    ElectronMobility = 4,
    //! Electron diffusion: TPS <-- Bolzmann
    ElectronDiffusion = 5,
    //! Reaction rates: TPS <-- Bolzmann
    ReactionRates = 6,
    //! All variables
    All = 7
  };

  //! Total number of fields
  const std::size_t NIndexes;

  Tps2Boltzmann(Tps *tps);
  void init(TPS::PlasmaSolver *flowSolver);
  bool IsInitialized() const { return all_fes_ != nullptr; }

  const mfem::ParFiniteElementSpace &Fes(Index index) const { return *(list_fes_[index]); }
  mfem::ParFiniteElementSpace &Fes(Index index) { return *(list_fes_[index]); }

  const mfem::ParFiniteElementSpace &NativeFes(Index index) const { return *(list_native_fes_[index]); }
  mfem::ParFiniteElementSpace &NativeFes(Index index) { return *(list_native_fes_[index]); }

  const mfem::ParGridFunction &SpatialCoordinates() const { return *spatial_coordinates_; }
  mfem::ParGridFunction &SpatialCoordinates() { return *spatial_coordinates_; }

  const mfem::ParGridFunction &Field(Index index) const { return *(fields_[index]); }
  mfem::ParGridFunction &Field(Index index) { return *(fields_[index]); }

  void interpolateFromNativeFES(const ParGridFunction &input, Index index);
  void interpolateToNativeFES(ParGridFunction &output, Index index);

  //! Get the angular Frequency \omega of the electrical field:
  //! E(t) = Er*cos(\omega t) + Ei*sin(\omega t)
  double EfieldAngularFreq() { return EfieldAngularFreq_; }
  int Nspecies() const { return nspecies_; }
  int NeFieldComps() const { return nEfieldComps_; }
  int nComponents(Index index) const { return ncomps[index]; }
  std::string getReactionEquation(int index) const { return reaction_eqs_[index]; }

  void setTimeStep(double dt) { timestep_ = dt; }
  void setCurrentTime(double time) { currentTime_ = time; }

  double timeStep() const { return timestep_; }
  double currentTime() const { return currentTime_; }

  void saveDataCollection(int cycle, double time);

  ~Tps2Boltzmann();

 private:
  int _countBTEReactions();
  Tps *tps_;

  int nspecies_;
  int nEfieldComps_;
  int nreactions_;
  int nfields_;
  int order_;
  int basis_type_;
  mfem::Array<int> offsets;
  mfem::Array<int> ncomps;

  mfem::FiniteElementCollection *fec_;

  //! Function spaces in the Boltzmann interface
  mfem::ParFiniteElementSpace *all_fes_;
  mfem::ParFiniteElementSpace *species_densities_fes_;
  mfem::ParFiniteElementSpace *efield_fes_;
  mfem::ParFiniteElementSpace *scalar_fes_;
  mfem::ParFiniteElementSpace *reaction_rates_fes_;
  mfem::ParFiniteElementSpace **list_fes_;

  mfem::ParFiniteElementSpace *spatial_coord_fes_;

  //! Function spaces using the native TPS fec
  mfem::ParFiniteElementSpace *species_densities_native_fes_;
  mfem::ParFiniteElementSpace *efield_native_fes_;
  mfem::ParFiniteElementSpace *scalar_native_fes_;
  mfem::ParFiniteElementSpace *reaction_rates_native_fes_;
  mfem::ParFiniteElementSpace **list_native_fes_;

  //! Linear interpolator between native TPS fec to Interface fec
  mfem::ParDiscreteLinearOperator *scalar_interpolator_;
  mfem::ParDiscreteLinearOperator *scalar_interpolator_to_nativeFES_;

  //! array of fields see *Index for how to address this
  mfem::ParGridFunction **fields_;
  mfem::ParGridFunction *spatial_coordinates_;

  double EfieldAngularFreq_;
  double timestep_;
  double currentTime_;

  bool save_to_paraview_dc;
  mfem::ParaViewDataCollection *paraview_dc;
  std::vector<std::string> reaction_eqs_;
};
}  // namespace TPS

#endif  // TPS2BOLTZMANN_HPP_
