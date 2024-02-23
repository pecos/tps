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

#include <string>

#include "tps_mfem_wrap.hpp"

// Forward declarations
class GasMixture;
class RunConfiguration;

using namespace mfem;
using namespace std;

/**
 * @brief Stores statistics fields (mean and variances) that are computed by Averaging
 *
 * This is a helper class for class Averaging.  It holds the
 * statistics fields to update and a pointer to the instantaneous
 * field being averaged.
 */
class AveragingFamily {
 public:
  /** Variable index to "start" variances
   *
   * For example, if state vector is [rho, u, v, w, T] and you only
   * want the velocity covariance matrix, rms_start_index_ should be 1
   */
  int rms_start_index_;

  /** Number of variables for covariance calculation
   *
   * For example, if state vector is [rho, u, v, w, T] and you only
   * want the velocity covariance matrix, rms_components_ should be
   * 3. This will lead to 6 covariance being computed (uu, vv, ww, uv,
   * uw, and vw).
   *
   * It is assumed that the variance variables are contiguous---i.e.,
   * there is no support for getting uT only.
   */
  int rms_components_;

  /** Pointer to function containing the instantaneous field being averaged (not owned) */
  const ParGridFunction *instantaneous_fcn_;

  /** Pointer to mean field (owned) */
  ParGridFunction *mean_fcn_;

  /** Pointer to variance field (owned) */
  ParGridFunction *rms_fcn_;

  /**
   * @brief Constructor
   *
   * Notes: the mean and rms grid functions should be allocated before
   * constructing the AveragingFamily, but AveragingFamily then takes
   * ownership.
   */
  AveragingFamily(const ParGridFunction *instant, ParGridFunction *mean, ParGridFunction *rms, int rms_start_index = 0,
                  int rms_components = 1) {
    rms_start_index_ = rms_start_index;
    rms_components_ = rms_components;
    instantaneous_fcn_ = instant;
    mean_fcn_ = mean;
    rms_fcn_ = rms;
  }

  /// Move constructor (required for emplace_back)
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

  /// Destructor (deletes the mean and rms fields)
  ~AveragingFamily() {
    delete mean_fcn_;
    delete rms_fcn_;
  }
};

/**
 * @brief Implements in situ statistics (mean and covariance) calculations
 *
 */
class Averaging {
 private:
  /// vector of families for averaging
  std::vector<AveragingFamily> avg_families_;

  /// Indicate if this is rank 0
  bool rank0_;

  /// True if mean calculation is requested
  bool compute_mean_;

  /// Time steps between updating stats with a new sample
  int sample_interval_;

  /// Time step at which to start computing stats
  int step_start_mean_;

  /// Number of samples used for the mean so far
  int ns_mean_;

  /// Number of samples used for the variances so var
  int ns_vari_;

  /// Visualization directory (i.e., paraview dumped to mean_output_name_)
  std::string mean_output_name_;

  /// mfem paraview data collection, used to write viz files
  ParaViewDataCollection *pvdc_ = nullptr;

  /// time averaged p, rho, vel (pointers to meanUp) for visualization (M2ulPhyS only!)
  ParGridFunction *meanP = nullptr;
  ParGridFunction *meanRho = nullptr;
  ParGridFunction *meanV = nullptr;

 public:
  /// Constructor
  Averaging(RunConfiguration &config);

  /// Destructor
  ~Averaging();

  /**
   * @brief Register a field for averaging
   *
   * @param field_to_average Pointer to the instantaneous data that will be used to update the statistics
   * @param compute_rms Set to true to compute variances.  Otherwise only mean is computed.
   * @param rms_start_index Variable index at which to start variances (see AveragingFamily)
   * @param rms_components Number of variables in variances (see AveragingFamily)
   */
  void registerField(const ParGridFunction *field_to_average, bool compute_rms = true, int rms_start_index = 0,
                     int rms_components = 1);

  /**
   * @brief Initialize visualiztion for statistics
   *
   * Note: This method is current specific to M2ulPhyS
   */
  void initializeViz(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes, int nvel);

  // TODO(trevilo): Specific to single ParGridFunction case
  ParGridFunction *GetMeanUp() { return avg_families_[0].mean_fcn_; }
  ParGridFunction *GetRMS() { return avg_families_[0].rms_fcn_; }

  void addSampleMean(const int &iter, GasMixture *mixture = nullptr);
  void addSample(GasMixture *mixture);

  void write_meanANDrms_restart_files(const int &iter, const double &time, bool save_mean_hist);

  int GetSamplesMean() { return ns_mean_; }
  int GetSamplesRMS() { return ns_vari_; }
  int GetSamplesInterval() { return sample_interval_; }
  bool ComputeMean() { return compute_mean_; }

  void SetSamplesMean(int &samples) { ns_mean_ = samples; }
  void SetSamplesRMS(int &samples) { ns_vari_ = samples; }
  void SetSamplesInterval(int &interval) { sample_interval_ = interval; }
};

#endif  // AVERAGING_AND_RMS_HPP_
