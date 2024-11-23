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

#include "thermo_chem_base.hpp"

#include "tps.hpp"

using namespace mfem;

ConstantPropertyThermoChem::ConstantPropertyThermoChem(ParMesh *pmesh, int sorder, double rho, double mu)
    : pmesh_(pmesh), sorder_(sorder), rho_(rho), mu_(mu) {}

ConstantPropertyThermoChem::ConstantPropertyThermoChem(ParMesh *pmesh, int sorder, TPS::Tps *tps)
    : pmesh_(pmesh), sorder_(sorder) {
  assert(tps != nullptr);
  tps->getInput("loMach/constprop/rho", rho_, 1.0);
  tps->getInput("loMach/constprop/mu", mu_, 1.0);
}

ConstantPropertyThermoChem::~ConstantPropertyThermoChem() {
  delete thermal_divergence_;
  delete pressure_divergence_;  
  delete viscosity_;
  delete density_;
  delete fes_;
  delete fec_;
}

void ConstantPropertyThermoChem::initializeSelf() {
  fec_ = new H1_FECollection(sorder_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_);

  density_ = new ParGridFunction(fes_);
  viscosity_ = new ParGridFunction(fes_);
  thermal_divergence_ = new ParGridFunction(fes_);
  pressure_divergence_ = new ParGridFunction(fes_);  
  pressure_ = new ParGridFunction(fes_);
  temperature_ = new ParGridFunction(fes_);    

  *density_ = rho_;
  *viscosity_ = mu_;
  *thermal_divergence_ = 0.0;
  *pressure_divergence_ = 0.0;  
  *pressure_ = thermo_pressure_;
  *temperature_ = 300.0;

  toFlow_interface_.density = density_;
  toFlow_interface_.temperature = temperature_;    
  toFlow_interface_.viscosity = viscosity_;
  toFlow_interface_.thermal_divergence = thermal_divergence_;
  toFlow_interface_.pressure_divergence = pressure_divergence_;  

  toTurbModel_interface_.density = density_;
}
