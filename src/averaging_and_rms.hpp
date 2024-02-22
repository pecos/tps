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
#ifndef AVERAGING_AND_RMS_HPP_
#define AVERAGING_AND_RMS_HPP_

#include <tps_config.h>

#include <mfem/general/forall.hpp>
#include <string>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

class Averaging {
 private:
  // Primitive variables
  ParGridFunction *Up;
  ParMesh *mesh;
  const FiniteElementCollection *fec;
  ParFiniteElementSpace *fes;
  ParFiniteElementSpace *dfes;
  const int num_equation;
  const int dim;
  const int nvel;
  RunConfiguration &config;

  // FES for RMS
  ParFiniteElementSpace *rmsFes;
  int numRMS;

  // time averaged primitive variables
  ParGridFunction *meanUp;
  ParGridFunction *rms;
  ParGridFunction *diss;

  // time averaged p, rho, vel (pointers to meanUp) for Visualization
  ParGridFunction *meanP, *meanRho, *meanV;
  ParGridFunction *meanScalar;

  ParaViewDataCollection *paraviewMean = NULL;

  int samplesMean;
  int samplesRMS;

  // iteration interval between samples
  int sampleInterval;
  int startMean;
  bool computeMean;

  void initiMeanAndRMS();

 public:
  Averaging(ParGridFunction *_Up, ParMesh *_mesh, ParFiniteElementSpace *_fes, ParFiniteElementSpace *_dfes,
            RunConfiguration &_config);
  ~Averaging();

  void addSampleMean(const int &iter, GasMixture *mixture = nullptr);
  void write_meanANDrms_restart_files(const int &iter, const double &time);
  void read_meanANDrms_restart_files();

  int GetSamplesMean() { return samplesMean; }
  int GetSamplesRMS() { return samplesRMS; }
  int GetSamplesInterval() { return sampleInterval; }
  bool ComputeMean() { return computeMean; }

  ParGridFunction *GetMeanUp() { return meanUp; }
  ParGridFunction *GetRMS() { return rms; }

  void SetSamplesMean(int &samples) { samplesMean = samples; }
  void SetSamplesRMS(int &samples) { samplesRMS = samples; }
  void SetSamplesInterval(int &interval) { sampleInterval = interval; }

  void addSample(GasMixture *mixture);
};

#endif  // AVERAGING_AND_RMS_HPP_
