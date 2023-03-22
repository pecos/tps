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

  cflNum = 0.1;
  constantTimeStep = false;
  dt_fixed = -1.0;
  numIters = 10;
  useRoe = false;
  restart = false;
  restart_hdf5_conversion = false;
  restart_serial = "no";
  restart_cycle = 0;
  restartFromAux = false;
  singleRestartFile = false;

  sampleInterval = 0;
  startIter = 0;
  restartMean = false;
  meanHistEnable = false;

  restartBdr = false;
  
  itersOut = 50;
  workFluid = DRY_AIR;
  eqSystem = EULER;
  axisymmetric_ = false;
  visc_mult = 1.;
  bulk_visc = 0.;
  refLength = 1.;
  SBP = false;
  for (int ii = 0; ii < 5; ii++) initRhoRhoVp[ii] = 0.;

  linViscData.normal.UseDevice(true);
  linViscData.point0.UseDevice(true);
  //linViscData.pointInit.UseDevice(true);

  linViscData.normal.SetSize(3);
  linViscData.point0.SetSize(3);
  //linViscData.pointInit.SetSize(3);

  linViscData.normal = 0.;
  linViscData.point0 = 0.;
  //linViscData.pointInit = 0.;
  linViscData.viscRatio = 0.;

  //   initSpongeData();
  numSpongeRegions_ = 0;
  spongeData_ = NULL;

  heatSource = NULL;

  isForcing = false;
  for (int ii = 0; ii < 3; ii++) gradPress[ii] = 0.;

  arrayPassiveScalar.DeleteAll();

  // Resource manager monitoring
  rm_enableMonitor_ = false;
  rm_threshold_ = 15 * 60;     // 15 minutes
  rm_checkFrequency_ = 25;     // 25 iterations
  exit_checkFrequency_ = 500;  // 500 iterations

  initialElectronTemperature = 0.;

  twoTemperature = false;
  ambipolar = false;

  periodic = false;
  xTrans = 1.0e12;
  yTrans = 1.0e12;
  zTrans = 1.0e12;  
  
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
  if (restart_serial == "readwrite") {
    return true;
  } else if (mode == "read") {
    if (restart_serial == "read") return true;
  } else if (mode == "write") {
    if (restart_serial == "write") return true;
  }

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

// Note from ks: this method to be deprecated. Superceded by
// M2ulPhyS::parseSolverOptions2()
void RunConfiguration::readInputFile(std::string inpuFileName) {
  // runfile
  ifstream runFile(inpuFileName);

  if (!runFile.is_open()) {
    cout << "Could not open runfile" << endl;
    return;
  }

  string line;

  while (getline(runFile, line)) {
    istringstream ss(line);
    // check if it is a commented line
    if (!(line.c_str()[0] == '#') && line.size() > 0) {
      string word;
      ss >> word;

      // mesh file
      if (word.compare("MESH") == 0) {
        ss >> word;
        meshFile.clear();
        meshFile = word;

        //       // linear mesh file
        //       }else if( word.compare("LIN_MESH")==0 )
        //       {
        //         ss >> word;
        //         lin_meshFile.clear();
        //         lin_meshFile = word;

        // number of uniform refinements
      } else if (word.compare("REF_LEVELS") == 0) {
        ss >> word;
        ref_levels = stoi(word);

        // partition file name
      } else if (word.compare("PART_FILE") == 0) {
        ss >> word;
        partFile.clear();
        partFile = word;

        // output file name
      } else if (word.compare("OUTPUT_NAME") == 0) {
        ss >> word;
        outputFile.clear();
        outputFile = word;

        // max num. of iters
      } else if (word.compare("NMAX") == 0) {
        ss >> word;
        numIters = stoi(word);

        // cycles at restart
      } else if (word.compare("RESTART_CYCLE") == 0) {
        ss >> word;
        restart = true;
        // restart_cycle = stoi(word);   // use value straight from restart file instead (ks mod)

        // restart from aux. sol
      } else if (word.compare("RESTART_CONVERSION") == 0) {
        ss >> word;
        restart_hdf5_conversion = true;
	
      } else if (word.compare("RESTART_SERIAL") == 0) {
        ss >> word;
        assert((word == "read") || (word == "write"));
        restart_serial = word;

        // restart from aux. sol
      } else if (word.compare("RESTART_FROM_AUX") == 0) {
        restartFromAux = true;

        // write a single restart file (rather than 1 from each MPI task)
      } else if (word.compare("SINGLE_RESTART_FILE") == 0) {
        singleRestartFile = true;

        // mean and RMS calculation
      } else if (word.compare("CALC_MEAN_RMS") == 0) {
        ss >> word;
        startIter = stoi(word);

        ss >> word;
        sampleInterval = stoi(word);

      } else if (word.compare("SAVE_MEAN_HIST") == 0) {
        meanHistEnable = true;

      } else if (word.compare("CONTINUE_MEAN_CALC") == 0) {
        restartMean = true;

        // output interval
      } else if (word.compare("ITERS_OUT") == 0) {
        ss >> word;
        itersOut = stoi(word);

        // Riemann solver
      } else if (word.compare("USE_ROE") == 0) {
        ss >> word;
        if (stoi(word) == 1) useRoe = true;

        // solution order
      } else if (word.compare("POL_ORDER") == 0) {
        ss >> word;
        solOrder = stoi(word);

        // Integration rule
      } else if (word.compare("INT_RULE") == 0) {
        ss >> word;
        integrationRule = stoi(word);

        // Basis type
      } else if (word.compare("BASIS_TYPE") == 0) {
        ss >> word;
        basisType = stoi(word);

        // CFL number
      } else if (word.compare("CFL") == 0) {
        ss >> word;
        cflNum = stod(word);

        // constant time-step
      } else if (word.compare("DT_CONSTANT") == 0) {
        constantTimeStep = true;

        // fixed time-step
      } else if (word.compare("DT_FIXED") == 0) {
        constantTimeStep = true;
        ss >> word;
        dt_fixed = stod(word);

        // time integrator
      } else if (word.compare("TIME_INTEGRATOR") == 0) {
        ss >> word;
        timeIntegratorType = stoi(word);

        // resource manager controls
      } else if (word.compare("ENABLE_AUTORESTART") == 0) {
        rm_enableMonitor_ = true;

      } else if (word.compare("RM_THRESHOLD") == 0) {
        ss >> word;
        rm_threshold_ = stoi(word);

      } else if (word.compare("RM_CHECK_FREQUENCY") == 0) {
        ss >> word;
        rm_checkFrequency_ = stoi(word);

      } else if (word.compare("EXIT_CHECK_FREQUENCY") == 0) {
        ss >> word;
        exit_checkFrequency_ = stoi(word);

      // working fluid
      } else if (word.compare("FLUID") == 0) {
        ss >> word;
        int typeFluid = stoi(word);
        switch (typeFluid) {
          case 0:
            workFluid = DRY_AIR;
            break;
          default:
            workFluid = USER_DEFINED;
            break;
        }
      } else if (word.compare("EQ_SYSTEM") == 0) {
        ss >> word;
        int systemType = stoi(word);
        switch (systemType) {
          case 0:
            eqSystem = EULER;
            break;
          case 1:
            eqSystem = NS;
            break;
          case 2:
            eqSystem = NS_PASSIVE;
            break;
          default:
            break;
        }
      } else if (word.compare("IS_SBP") == 0) {
        ss >> word;
        int isSBP = stoi(word);
        if (isSBP == 1) SBP = true;

        // Initial solution RHO
      } else if (word.compare("INIT_RHO") == 0) {
        ss >> word;
        initRhoRhoVp[0] = stod(word);

        // Initial solution RHOVX
      } else if (word.compare("INIT_RHOVX") == 0) {
        ss >> word;
        initRhoRhoVp[1] = stod(word);

        // Initial solution RHOVY
      } else if (word.compare("INIT_RHOVY") == 0) {
        ss >> word;
        initRhoRhoVp[2] = stod(word);

        // Initial solution RHOVZ
      } else if (word.compare("INIT_RHOVZ") == 0) {
        ss >> word;
        initRhoRhoVp[3] = stod(word);

        // Initial solution P
      } else if (word.compare("INIT_P") == 0) {
        ss >> word;
        initRhoRhoVp[4] = stod(word);

        // viscity factor
      } else if (word.compare("VISC_MULT") == 0) {
        ss >> word;
        visc_mult = stod(word);

      } else if (word.compare("BULK_VISC_MULT") == 0) {
        ss >> word;
        bulk_visc = stod(word);

        // Reference length
      } else if (word.compare("REF_LENGTH") == 0) {
        ss >> word;
        refLength = stod(word);

      } else if (word.compare("GRAD_PRESSURE") == 0) {
        ss >> word;
        gradPress[0] = stod(word);

        ss >> word;
        gradPress[1] = stod(word);

        ss >> word;
        gradPress[2] = stod(word);
        isForcing = true;

      } else if (word.compare("LV_PLANE_NORM") == 0) {
        auto normal = linViscData.normal.HostWrite();
        ss >> word;
        normal[0] = stod(word);
        ss >> word;
        normal[1] = stod(word);
        ss >> word;
        normal[2] = stod(word);
        // make sure modulus = 1
        double modulus = 0.;
        for (int d = 0; d < 3; d++) modulus += normal[d] * normal[d];
        modulus = sqrt(modulus);
        for (int d = 0; d < 3; d++) normal[d] /= modulus;

      } else if (word.compare("LV_PLANE_P0") == 0) {
        auto point0 = linViscData.point0.HostWrite();
        ss >> word;
        point0[0] = stod(word);
        ss >> word;
        point0[1] = stod(word);
        ss >> word;
        point0[2] = stod(word);

	/*}  else if (word.compare("LV_PLANE_PINIT") == 0) {
        auto pointInit = linViscData.pointInit.HostWrite();
        ss >> word;
        pointInit[0] = stod(word);
        ss >> word;
        pointInit[1] = stod(word);
        ss >> word;
        pointInit[2] = stod(word); 
        */

      } else if (word.compare("LV_VISC_RATIO") == 0) {
        ss >> word;
        linViscData.viscRatio = stod(word);

      } else if (word.compare("INLET") == 0) {
        ss >> word;
        std::pair<int, InletType> patchANDtype;
        patchANDtype.first = stoi(word);

        ss >> word;
        int typ = stoi(word);
        switch (typ) {
          case 0:
            patchANDtype.second = SUB_DENS_VEL;
            break;
          case 1:
            patchANDtype.second = SUB_DENS_VEL_NR;
            break;
          case 2:
            patchANDtype.second = SUB_VEL_CONST_ENT;
            break;
        }
        inletPatchType.push_back(patchANDtype);

        for (int i = 0; i < 4; i++) {
          ss >> word;
          inletBC.Append(stod(word));
        }

      } else if (word.compare("OUTLET") == 0) {
        ss >> word;
        std::pair<int, OutletType> patchANDtype;
        patchANDtype.first = stoi(word);

        ss >> word;
        int typ = stoi(word);
        switch (typ) {
          case 0:
            patchANDtype.second = SUB_P;
            break;
          case 1:
            patchANDtype.second = SUB_P_NR;
            break;
          case 2:
            patchANDtype.second = SUB_MF_NR;
            break;
          case 3:
            patchANDtype.second = SUB_MF_NR_PW;
            break;
        }
        outletPatchType.push_back(patchANDtype);

        ss >> word;
        outletBC.Append(stod(word));

      } else if (word.compare("WALL") == 0) {
        ss >> word;
        std::pair<int, WallType> patchType;
        patchType.first = stoi(word);
        ss >> word;
        switch (stoi(word)) {
          case 0:
            patchType.second = INV;
            break;
          case 1:
            patchType.second = VISC_ADIAB;
            break;
          case 2:
            patchType.second = VISC_ISOTH;
            break;
          case 3:	    
            patchType.second = SLIP;
            break;	    
        }
        wallPatchType.push_back(patchType);
        WallData wData;
        if (patchType.second == VISC_ISOTH) {
          ss >> word;
          wData.Th = stod(word);
        } else {
          wData.Th = -1.0;
        }
        wallBC.push_back(wData);
      } else if (word.compare("PASSIVE_SCALAR") == 0) {
        int arrSize = arrayPassiveScalar.Size();
        arrayPassiveScalar.Append(new passiveScalarData);

        arrayPassiveScalar[arrSize]->value = 0.;
        ss >> word;
        arrayPassiveScalar[arrSize]->coords.SetSize(3);
        arrayPassiveScalar[arrSize]->coords = 0.;
        arrayPassiveScalar[arrSize]->coords[0] = stod(word);
        ss >> word;
        arrayPassiveScalar[arrSize]->coords[1] = stod(word);
        ss >> word;
        arrayPassiveScalar[arrSize]->coords[2] = stod(word);
        ss >> word;
        arrayPassiveScalar[arrSize]->radius = stod(word);
        ss >> word;
        arrayPassiveScalar[arrSize]->value = stod(word);
      } else if (word.compare("NUM_SPECIES") == 0) {
        ss >> word;
        numSpecies = stoi(word);
        gasParams.SetSize(numSpecies, GasParams::NUM_GASPARAMS);
        for (int sp = 0; sp < numSpecies; sp++)
          for (int param = 0; param < GasParams::NUM_GASPARAMS; param++) {
            ss >> word;
            gasParams(sp, param) = stod(word);
          }
      } else if (word.compare("AMBIPOLAR") == 0) {
        ambipolar = false;
        ss >> word;
        if (word.compare("TRUE") == 0) ambipolar = true;
      } else if (word.compare("TWO_TEMPERATURE") == 0) {
        twoTemperature = false;
        ss >> word;
        if (word.compare("TRUE") == 0) twoTemperature = true;
      } else if (word.compare("GAS_MODEL") == 0) {
        ss >> word;
        assert((stoi(word) >= 0) && (stoi(word) < GasModel::NUM_GASMODEL));
        gasModel = (GasModel)stoi(word);
      } else if (word.compare("TRANSPORT_MODEL") == 0) {
        ss >> word;
        assert((stoi(word) >= 0) && (stoi(word) < TransportModel::NUM_TRANSPORTMODEL));
        transportModel = (TransportModel)stoi(word);
      } else if (word.compare("CHEMISTRY_MODEL") == 0) {
        ss >> word;
        assert((stoi(word) >= 0) && (stoi(word) < ChemistryModel::NUM_CHEMISTRYMODEL));
        chemistryModel_ = (ChemistryModel)stoi(word);
      }
    }
  }

  runFile.close();
}

Array<double> RunConfiguration::GetInletData(int i) {

  // WARNIING:  the structure of the inputs is stupid and will break as-is
  // for combinations of different bc types and specified species 
  int length = 4;
  //  int length = 5;
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
