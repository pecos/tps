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

#include "static_thermo.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>
#include <mfem/general/forall.hpp>

#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "radiation.hpp"
#include "tps.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

static double radius(const Vector &pos) { return pos[0]; }
static FunctionCoefficient radius_coeff(radius);

StaticThermo::StaticThermo(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, temporalSchemeCoefficients &time_coeff, TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), time_coeff_(time_coeff) {
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  tps->getInput("loMach/axisymmetric", axisym_, false);

  // Initialize thermo TableInput, use this to get other properties from fixed T
  std::vector<TableInput> thermo_tables(5);
  for (size_t i = 0; i < thermo_tables.size(); i++) {
    thermo_tables[i].order = 1;
    thermo_tables[i].xLogScale = false;
    thermo_tables[i].fLogScale = false;
  }

  std::string table_file;
  tps->getRequiredInput("loMach/staticthermo/table-file", table_file);

  // Read data from hdf5 file containing 6 columns: T, mu, kappa, sigma, Rgas, Cp
  DenseMatrix table_data;
  bool success;
  success = h5ReadBcastMultiColumnTable(table_file, std::string("T_mu_kap_sig_R_Cp"), pmesh_->GetComm(), table_data,
                                        thermo_tables);
  if (!success) exit(ERROR);
  mu_table_ = new LinearTable(thermo_tables[0]);
  kappa_table_ = new LinearTable(thermo_tables[1]);
  sigma_table_ = new LinearTable(thermo_tables[2]);
  Rgas_table_ = new LinearTable(thermo_tables[3]);
  Cp_table_ = new LinearTable(thermo_tables[4]);

  tpsP_->getInput("loMach/ambientPressure", ambient_pressure_, 101325.0);
  thermo_pressure_ = ambient_pressure_;

  // adopt this setting as the fixed temperature
  tpsP_->getInput("initialConditions/temperature", T_ic_, 300.0);
}

StaticThermo::~StaticThermo() {
  // allocated in initializeOperators
  delete temp_;

  // allocated in initializeSelf
  delete sfes_;
  delete sfec_;
}

void StaticThermo::initializeSelf() {
  if (rank0_) grvy_printf(ginfo, "Initializing StaticThermo solver.\n");

  //-----------------------------------------------------
  // 1) Prepare the required finite element objects
  //-----------------------------------------------------
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  temp_ = new ParGridFunction(sfes_);
  *temp_ = T_ic_;
  
  int sfes_truevsize = sfes_->GetTrueVSize();

  Qt_.SetSize(sfes_truevsize);
  Qt_ = 0.0;
  Qt_gf_.SetSpace(sfes_);
  Qt_gf_ = 0.0;

  rn_.SetSize(sfes_truevsize);
  rn_ = 1.0;

  rn_gf_.SetSpace(sfes_);
  rn_gf_ = 1.0;

  Rgas_.SetSize(sfes_truevsize);
  Rgas_ = 0.0;

  Cp_.SetSize(sfes_truevsize);
  Cp_ = 0.0;

  visc_.SetSize(sfes_truevsize);
  visc_ = 0.0;

  kappa_.SetSize(sfes_truevsize);
  kappa_ = 0.0;

  mu_gf_.SetSpace(sfes_);
  mu_gf_ = 0.0;

  kappa_gf_.SetSpace(sfes_);
  kappa_gf_ = 0.0;

  Rgas_gf_.SetSpace(sfes_);
  Rgas_gf_ = 0.0;

  Cp_gf_.SetSpace(sfes_);
  Cp_gf_ = 0.0;

  if (rank0_) grvy_printf(ginfo, "StaticThermo vectors and gf initialized...\n");

  // exports
  toFlow_interface_.density = &rn_gf_;
  toFlow_interface_.viscosity = &mu_gf_;
  toFlow_interface_.thermal_divergence = &Qt_gf_;
  toTurbModel_interface_.density = &rn_gf_;

  // Initialize the gas constant and Cp
  {
    const double *d_T = temp_->Read();
    
    double *d_Rgas = Rgas_.Write();
    MFEM_FORALL(i, temp_->Size(), { d_Rgas[i] = Rgas_table_->eval(d_T[i]); });

    double *d_Cp = Cp_.Write();
    MFEM_FORALL(i, temp_->Size(), { d_Cp[i] = Cp_table_->eval(d_T[i]); });
  }
  Rgas_gf_.SetFromTrueDofs(Rgas_);
  Cp_gf_.SetFromTrueDofs(Cp_);

  // Initialize the transport properties
  {
    const double *d_T = temp_->Read();
    
    double *d_visc = visc_.Write();
    MFEM_FORALL(i, temp_->Size(), { d_visc[i] = mu_table_->eval(d_T[i]); });

    double *d_kap = kappa_.Write();
    MFEM_FORALL(i, temp_->Size(), { d_kap[i] = kappa_table_->eval(d_T[i]); });
  }
  mu_gf_.SetFromTrueDofs(visc_);
  kappa_gf_.SetFromTrueDofs(kappa_);
}

void StaticThermo::initializeOperators() {
  const double dt_ = time_coeff_.dt;

  // compute density
  updateDensity();
}

void StaticThermo::initializeViz(ParaViewDataCollection &pvdc) {
  pvdc.RegisterField("temperature", temp_);
  pvdc.RegisterField("density", &rn_gf_);
  pvdc.RegisterField("kappa", &kappa_gf_);
  pvdc.RegisterField("Qt", &Qt_gf_);
  pvdc.RegisterField("Rgas", &Rgas_gf_);
  pvdc.RegisterField("Cp", &Cp_gf_);
}

void StaticThermo::updateDensity() {
  // Update the density using the ideal gas law
  rn_ = thermo_pressure_;
  rn_ /= Rgas_;
  rn_ /= *temp_;
  rn_gf_.SetFromTrueDofs(rn_);
}
