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

class AveragingFamily {
 public:
  int rms_start_index_;
  int rms_components_;

  const ParGridFunction *instantaneous_fcn_;
  ParGridFunction *mean_fcn_;
  ParGridFunction *rms_fcn_;

  AveragingFamily(const ParGridFunction *instant, ParGridFunction *mean, ParGridFunction *rms, int rms_start_index = 0,
                  int rms_components = 1) {
    rms_start_index_ = rms_start_index;
    rms_components_ = rms_components;
    instantaneous_fcn_ = instant;
    mean_fcn_ = mean;
    rms_fcn_ = rms;
  }

  // Move constructor (required for emplace_back)
  AveragingFamily(AveragingFamily &&fam) {
    this->rms_start_index_ = fam.rms_start_index_;
    this->rms_components_ = fam.rms_components_;

    // Copy the pointers
    this->instantaneous_fcn_ = fam.instantaneous_fcn_;
    this->mean_fcn_ = fam.mean_fcn_;
    this->rms_fcn_ = fam.rms_fcn_;

    // Set incoming mean_fcn_ to null, which ensures memory that
    // this->mean_fcn_ now points to is not deleted when the
    // destructor of fam is called
    fam.mean_fcn_ = nullptr;
    fam.rms_fcn_ = nullptr;
  }

  ~AveragingFamily() {
    delete mean_fcn_;
    delete rms_fcn_;
  }
};

class Averaging {
 private:
  bool rank0_;
  std::vector<AveragingFamily> avg_families_;

  // time averaged p, rho, vel (pointers to meanUp) for Visualization
  ParGridFunction *meanP, *meanRho, *meanV;
  ParGridFunction *meanScalar;

  std::string mean_output_name_;
  ParaViewDataCollection *paraviewMean = NULL;

  int samplesMean;
  int samplesRMS;

  // iteration interval between samples
  int sampleInterval;
  int startMean;
  bool computeMean;

 public:
  Averaging(RunConfiguration &config);
  ~Averaging();

  void registerField(const ParGridFunction *field_to_average, bool compute_rms = true, int rms_start_index = 0,
                     int rms_components = 1);
  void initializeViz(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes, int nvel);

  void addSampleMean(const int &iter, GasMixture *mixture = nullptr);
  void write_meanANDrms_restart_files(const int &iter, const double &time, bool save_mean_hist);
  void read_meanANDrms_restart_files();

  int GetSamplesMean() { return samplesMean; }
  int GetSamplesRMS() { return samplesRMS; }
  int GetSamplesInterval() { return sampleInterval; }
  bool ComputeMean() { return computeMean; }

  // TODO(trevilo): Specific to single ParGridFunction case
  ParGridFunction *GetMeanUp() { return avg_families_[0].mean_fcn_; }
  ParGridFunction *GetRMS() { return avg_families_[0].rms_fcn_; }

  void SetSamplesMean(int &samples) { samplesMean = samples; }
  void SetSamplesRMS(int &samples) { samplesRMS = samples; }
  void SetSamplesInterval(int &interval) { sampleInterval = interval; }

  void addSample(GasMixture *mixture);
};

#endif  // AVERAGING_AND_RMS_HPP_
