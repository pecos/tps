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
 * Implementation of the Boltzmann interface routines for M2ulPhyS
 */

#include "M2ulPhyS.hpp"
#include "tps2Boltzmann.hpp"

// CPU version (just for starting up)
void M2ulPhyS::push(TPS::Tps2Boltzmann &interface) {
  std::cout << "Enter M2ulPhyS::push" << std::endl;
  assert(interface.IsInitialized());

  int nscalardofs = vfes->GetNDofs();

  const double *solver_data = U->HostRead();

  mfem::ParGridFunction *species =
      new mfem::ParGridFunction(&interface.NativeFes(TPS::Tps2Boltzmann::Index::SpeciesDensities));
  mfem::ParGridFunction *heavyTemperature =
      new mfem::ParGridFunction(&interface.NativeFes(TPS::Tps2Boltzmann::Index::HeavyTemperature));
  mfem::ParGridFunction *electronTemperature =
      new mfem::ParGridFunction(&interface.NativeFes(TPS::Tps2Boltzmann::Index::ElectronTemperature));

  double *species_data = species->HostWrite();
  double *heavyTemperature_data = heavyTemperature->HostWrite();
  double *electronTemperature_data = electronTemperature->HostWrite();

  double state_local[gpudata::MAXEQUATIONS];
  double species_local[gpudata::MAXSPECIES];

  PerfectMixture * pmixture = dynamic_cast<PerfectMixture *>(mixture);
  assert(pmixture);

  std::cout << "Start to loop" << std::endl;
  for (int i = 0; i < nscalardofs; i++) {
    for (int eq = 0; eq < num_equation; eq++) state_local[eq] = solver_data[i + eq * nscalardofs];

    pmixture->computeNumberDensities(state_local, species_local);

    pmixture->computeTemperaturesBase(state_local, species_local, species_local[mixture->GetiElectronIndex()],
                                     species_local[mixture->GetiBackgroundIndex()], heavyTemperature_data[i],
                                     electronTemperature_data[i]);

    for (int sp = 0; sp < interface.Nspecies(); sp++) species_data[i + sp * nscalardofs] = AVOGADRONUMBER * species_local[sp];
  }

  std::cout << "Call interpolate" << std::endl;
  interface.interpolateFromNativeFES(*species, TPS::Tps2Boltzmann::Index::SpeciesDensities);
  interface.interpolateFromNativeFES(*heavyTemperature, TPS::Tps2Boltzmann::Index::HeavyTemperature);
  interface.interpolateFromNativeFES(*electronTemperature, TPS::Tps2Boltzmann::Index::ElectronTemperature);

  std::cout << "Delete native GridFunctions" << std::endl;

  delete species;
  delete heavyTemperature;
  delete electronTemperature;

  std::cout << "Exit M2ulPhyS::push" << std::endl;
}

void M2ulPhyS::fetch(TPS::Tps2Boltzmann &interface) { return; }