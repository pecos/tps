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
#include "run_configuration.hpp"

#include <fstream>
#include <sstream>
#include <string>

RunConfiguration::RunConfiguration() {
  // Initialize with default values
  partFile = "partition";

  ref_levels = 0;
  timeIntegratorType = 4;
  solOrder = 4;
  sgsModelType = 0;
  sgsFloor = 0.;

  integrationRule = 1;
  basisType = 1;

  cflNum = 0.12;
  constantTimeStep = false;
  dt_fixed = -1.0;
  numIters = 10;
  useRoe = false;
  restart_hdf5_conversion = false;
  restart_cycle = 0;
  singleRestartFile = false;

  sampleInterval = 0;
  startIter = 0;
  restartMean = false;
  meanHistEnable = false;

  itersOut = 50;
  workFluid = DRY_AIR;
  eqSystem = EULER;
  axisymmetric_ = false;
  visc_mult = 1.;
  bulk_visc = 0.;
  refLength = 1.;
  for (int ii = 0; ii < 5; ii++) initRhoRhoVp[ii] = 0.;

  linViscData.normal.UseDevice(true);
  linViscData.point0.UseDevice(true);
  linViscData.pointInit.UseDevice(true);

  linViscData.normal.SetSize(3);
  linViscData.point0.SetSize(3);
  linViscData.pointInit.SetSize(3);

  linViscData.normal = 0.;
  linViscData.point0 = 0.;
  linViscData.pointInit = 0.;
  linViscData.viscRatio = 0.;

  planeDump.normal.UseDevice(true);
  planeDump.point.UseDevice(true);

  planeDump.normal.SetSize(3);
  planeDump.point.SetSize(3);

  planeDump.normal = 0.;
  planeDump.point = 0.;
  planeDump.samples = 0;

  //   initSpongeData();
  numSpongeRegions_ = 0;
  spongeData_ = NULL;

  heatSource = NULL;

  isForcing = false;
  for (int ii = 0; ii < 3; ii++) gradPress[ii] = 0.;

  isGravity = false;
  for (int ii = 0; ii < 3; ii++) gravity[ii] = 0.;
  
  arrayPassiveScalar.DeleteAll();

  // Resource manager monitoring
  rm_enableMonitor_ = false;
  rm_threshold_ = 15 * 60;  // 15 minutes
  rm_checkFrequency_ = 25;  // 25 iterations

  initialElectronTemperature = 0.;

  twoTemperature = false;
  ambipolar = false;
}

RunConfiguration::~RunConfiguration() {
  for (int i = 0; i < arrayPassiveScalar.Size(); i++) {
    delete arrayPassiveScalar[i];
  }

  if (heatSource != NULL) delete[] heatSource;

  if (spongeData_ != NULL) delete[] spongeData_;
}

// convenience function to determine whether a restart is serialized (using a
// single HDF5 file)
bool RunConfiguration::isRestartSerialized(string mode) {
  if (io_opts_.restart_serial_read_ && io_opts_.restart_serial_write_) {
    return true;
  } else if (mode == "read") {
    return io_opts_.restart_serial_read_;
  } else if (mode == "write") {
    return io_opts_.restart_serial_write_;
  } else if (mode == "either") {
    return (io_opts_.restart_serial_read_ || io_opts_.restart_serial_write_);
  } else if (mode == "both") {
    return (io_opts_.restart_serial_read_ && io_opts_.restart_serial_write_);
  } else {
    // should never get here
    assert(false);
  }
  // should never get here, but have to return to avoid "missing return statement" error
  return false;
}

// convenience function to determine whether a restart is partitioned
// (ie. parallel).
bool RunConfiguration::isRestartPartitioned(string mode) { return (!isRestartSerialized(mode)); }

void RunConfiguration::initSpongeData() {
  //   spongeData.multFactor = 1.;
  //
  //   spongeData.normal.UseDevice(true);
  //   spongeData.point0.UseDevice(true);
  //   spongeData.pointInit.UseDevice(true);
  //   spongeData.targetUp.UseDevice(true);
  //
  //   spongeData.normal.SetSize(3);
  //   spongeData.point0.SetSize(3);
  //   spongeData.pointInit.SetSize(3);
  //   spongeData.targetUp.SetSize(5);
  //
  //   spongeData.normal = 0.;
  //   spongeData.point0 = 0.;
  //   spongeData.pointInit = 0.;
  //   spongeData.targetUp = 0.;
  //
  //   spongeData.tol = 1e-5;
  //   spongeData.szType = SpongeZoneSolution::NONE;
}

Array<double> RunConfiguration::GetInletData(int i) {
  int length = 4;
  if ((workFluid != DRY_AIR) && (numSpecies > 1)) length += numSpecies;
  Array<double> data(length);
  for (int j = 0; j < length; j++) data[j] = inletBC[j + length * i];

  return data;
}

Array<double> RunConfiguration::GetOutletData(int out) {
  Array<double> data(1);
  data[0] = outletBC[out];

  return data;
}

WallData RunConfiguration::GetWallData(int w) { return wallBC[w]; }
