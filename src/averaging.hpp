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
#ifndef AVERAGING_HPP_
#define AVERAGING_HPP_

#include <tps_config.h>

#include <string>

#include "tps_mfem_wrap.hpp"

// Forward declarations
class GasMixture;

namespace TPS {
class Tps;
}

using namespace mfem;
using namespace std;

/**
 * @brief Options controlling behavior of Averaging class
 */
class AveragingOptions {
 public:
  AveragingOptions();

  int sample_interval_; /**< Time steps between updating stats with a new sample */
  int step_start_mean_; /**< Time step at which to start computing stats */

  bool save_mean_history_;        /**< Whether to save viz files for the evolution of the mean */
  bool enable_mean_continuation_; /**< Enable / disable continuation of statistics calculations from restart file */
  bool zero_variances_;           /**< Enable / disable zeroing out the variances at the beginning of a run */

  void read(TPS::Tps *tps, std::string prefix = std::string(""));
};

/**
 * @brief Stores statistics fields (mean and variances) that are computed by Averaging
 *
 * This is a helper class for class Averaging.  It holds the
 * statistics fields to update and a pointer to the instantaneous
 * field being averaged.
 */
class AveragingFamily {
 public:
  /// The family name.  Should identify the data being averaged (e.g., "temperature")
  std::string name_;

  /** Variable index to "start" variances
   *
   * For example, if state vector is [rho, u, v, w, T] and you only
   * want the velocity covariance matrix, vari_start_index_ should be 1
   */
  int vari_start_index_;

  /** Number of variables for covariance calculation
   *
   * For example, if state vector is [rho, u, v, w, T] and you only
   * want the velocity covariance matrix, vari_components_ should be
   * 3. This will lead to 6 covariance being computed (uu, vv, ww, uv,
   * uw, and vw).
   *
   * It is assumed that the variance variables are contiguous---i.e.,
   * there is no support for getting uT only.
   */
  int vari_components_;

  /** Pointer to function containing the instantaneous field being averaged (not owned) */
  const ParGridFunction *instantaneous_fcn_;

  /** Pointer to mean field (owned) */
  ParGridFunction *mean_fcn_;

  /** Pointer to variance field (owned) */
  ParGridFunction *vari_fcn_;

  /**
   * @brief Constructor
   *
   * Notes: the mean and vari grid functions should be allocated before
   * constructing the AveragingFamily, but AveragingFamily then takes
   * ownership.
   */
  AveragingFamily(std::string name, const ParGridFunction *instant, ParGridFunction *mean, ParGridFunction *vari,
                  int vari_start_index = 0, int vari_components = 1) {
    name_ = name;
    vari_start_index_ = vari_start_index;
    vari_components_ = vari_components;
    instantaneous_fcn_ = instant;
    mean_fcn_ = mean;
    vari_fcn_ = vari;
  }

  /// Move constructor (required for emplace_back)
  AveragingFamily(AveragingFamily &&fam) {
    this->name_ = fam.name_;
    this->vari_start_index_ = fam.vari_start_index_;
    this->vari_components_ = fam.vari_components_;

    // Copy the pointers
    this->instantaneous_fcn_ = fam.instantaneous_fcn_;
    this->mean_fcn_ = fam.mean_fcn_;
    this->vari_fcn_ = fam.vari_fcn_;

    // Set incoming mean_fcn_ to null, which ensures memory that
    // this->mean_fcn_ now points to is not deleted when the
    // destructor of fam is called
    fam.mean_fcn_ = nullptr;
    fam.vari_fcn_ = nullptr;
  }

  /// Destructor (deletes the mean and vari fields)
  ~AveragingFamily() {
    delete mean_fcn_;
    delete vari_fcn_;
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

  /// flag to continue mean from restart
  bool enable_mean_continuation_;

  /// flag to restart variances
  bool zero_variances_;

  /// mfem paraview data collection, used to write viz files
  ParaViewDataCollection *pvdc_ = nullptr;

  /// time averaged p, rho, vel (pointers to meanUp) for visualization (M2ulPhyS only!)
  ParGridFunction *meanP = nullptr;
  ParGridFunction *meanRho = nullptr;
  ParGridFunction *meanV = nullptr;

 public:
  /// Constructor
  Averaging(AveragingOptions &opts, std::string output_name);

  /// Destructor
  ~Averaging();

  /**
   * @brief Register a field for averaging
   *
   * @param name Name for the data being averaged (e.g., "temperature", or "primitive-state")
   * @param field_to_average Pointer to the instantaneous data that will be used to update the statistics
   * @param compute_vari Set to true to compute variances.  Otherwise only mean is computed.
   * @param vari_start_index Variable index at which to start variances (see AveragingFamily)
   * @param vari_components Number of variables in variances (see AveragingFamily)
   */
  void registerField(std::string name, const ParGridFunction *field_to_average, bool compute_vari = true,
                     int vari_start_index = 0, int vari_components = 1);

  /**
   * @brief Initialize visualiztion for statistics
   */
  void initializeViz();

  /**
   * @brief Add a sample to the statistics
   *
   * Checks iter to see if it is time to add a sample.  If not,
   * returns.  If so, farms work out to appropriate addSampleInternal
   * variant.
   */
  void addSample(const int &iter, GasMixture *mixture = nullptr);

  /**
   * @brief Write paraview visualization files with statistics
   */
  void writeViz(const int &iter, const double &time, bool save_mean_hist);

  /**
   * @brief Internal implementation of sample addition
   *
   * This method should be private, but is not b/c of a cuda
   * requirement on device lambdas.  Instead of this method, you
   * should call addSample (with mixture = nullptr).
   */
  void addSampleInternal();

  /**
   * @brief Internal implementation of sample addition (M2ulPhyS version)
   *
   * This method should be private, but is not b/c of a cuda
   * requirement on device lambdas.  Instead of this method, you
   * should call addSample (with mixture = a valid GasMixture object).
   */
  void addSampleInternal(GasMixture *mixture);

  /**
   * @brief Initialize visualiztion for statistics (M2ulPhyS version)
   *
   * Note: This method is included to provide backward compatibility
   * with how stats viz files were originally labeled.  It should only
   * be used inside M2ulPhyS.
   */
  void initializeVizForM2ulPhyS(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes, int nvel);

  int getFamilyIndex(std::string name) const {
    for (size_t i = 0; i < avg_families_.size(); i++) {
      if (avg_families_[i].name_ == name) return i;
    }
    return -1;
  }

  /**
   * @brief Get pointer to mean field for a family
   *
   * @param name Name of the family
   */
  ParGridFunction *GetMeanField(std::string name) {
    const int i = getFamilyIndex(name);
    assert(i >= 0);
    return avg_families_[i].mean_fcn_;
  }

  /**
   * @brief Get pointer to variance field for a family
   *
   * @param name Name of the family
   */
  ParGridFunction *GetVariField(std::string name) {
    const int i = getFamilyIndex(name);
    assert(i >= 0);
    return avg_families_[i].vari_fcn_;
  }

  int GetSamplesMean() { return ns_mean_; }
  int GetSamplesRMS() { return ns_vari_; }
  int GetSamplesInterval() { return sample_interval_; }

  bool ComputeMean() { return compute_mean_; }
  bool ContinueMean() { return enable_mean_continuation_; }
  bool RestartRMS() { return zero_variances_; }

  void SetSamplesMean(int &samples) { ns_mean_ = samples; }
  void SetSamplesRMS(int &samples) { ns_vari_ = samples; }
  void SetSamplesInterval(int &interval) { sample_interval_ = interval; }
};

#endif  // AVERAGING_HPP_
