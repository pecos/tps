// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
#ifndef MASA_HANDLER_HPP_
#define MASA_HANDLER_HPP_

#include <masa.h>

#include <mfem.hpp>

#include "fluxes.hpp"  // for enum Equations
#include "run_configuration.hpp"
#include "equation_of_state.hpp"

using namespace mfem;
using namespace std;

// void initMasaHandler(string name, int dim, const Equations& eqn, const double& viscMult, const std::string mms_name);

class MasaHandler {
 protected:
  int dim_, numEquation_;
  RunConfiguration* config_;

  // TODO: hope to incorporate den, vel, press to Up.
  ParGridFunction *U_, *Up_, *dens_, *vel_, *press_;
 public:
  MasaHandler(const int dim, const int numEquation, RunConfiguration& config, ParGridFunction *U, ParGridFunction *Up,
              ParGridFunction *dens, ParGridFunction *vel, ParGridFunction *press);
  ~MasaHandler() {}

  virtual void exactSolnFunction(const Vector &x, double tin, Vector &y) = 0;

  void checkSanity();
  void projectExactSolution();
};

class DryAirMMS : public MasaHandler {
 private:
  DryAir *dryAir_;
 public:
  DryAirMMS(const int dim, const int numEquation, RunConfiguration& config, ParGridFunction *U, ParGridFunction *Up,
                   ParGridFunction *dens, ParGridFunction *vel, ParGridFunction *press);
  ~DryAirMMS();

  virtual void exactSolnFunction(const Vector &x, double tin, Vector &y);
};

class NS2DCompressible : public DryAirMMS {
 public:
  NS2DCompressible(const int dim, const int numEquation, RunConfiguration& config, ParGridFunction *U, ParGridFunction *Up,
                   ParGridFunction *dens, ParGridFunction *vel, ParGridFunction *press);
  ~NS2DCompressible() {};
};

class Euler3DTransient : public DryAirMMS {
 public:
  Euler3DTransient(const int dim, const int numEquation, RunConfiguration& config, ParGridFunction *U, ParGridFunction *Up,
                   ParGridFunction *dens, ParGridFunction *vel, ParGridFunction *press);
  ~Euler3DTransient() {};
};

class NS3DTransient : public DryAirMMS {
 public:
  NS3DTransient(const int dim, const int numEquation, RunConfiguration& config, ParGridFunction *U, ParGridFunction *Up,
                ParGridFunction *dens, ParGridFunction *vel, ParGridFunction *press);
  ~NS3DTransient() {};
};

#endif  // MASA_HANDLER_HPP_
