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

#include "loMach_options.hpp"

#include "tps.hpp"

SubGridModelOptions::SubGridModelOptions() : sgs_model_string_("none"), sgs_model_constant_(0.0), exclude_mean_(false) {
  sgs_model_map_["none"] = NONE;
  sgs_model_map_["smagorinsky"] = SMAGORINSKY;
  sgs_model_map_["sigma"] = SIGMA;

  sgs_model_type_ = sgs_model_map_[sgs_model_string_];
}

void SubGridModelOptions::read(TPS::Tps *tps, std::string prefix) {
  // At the moment, SGS model options are under either "flow" or
  // "loMach", so we must have a prefix
  assert(!prefix.empty());

  std::string basename;
  if (!prefix.empty()) {
    basename = prefix + "/";
  }

  tps->getInput((basename + "sgsModel").c_str(), sgs_model_string_, std::string("none"));
  tps->getInput((basename + "sgsExcludeMean").c_str(), exclude_mean_, false);

  // Set model type
  sgs_model_type_ = sgs_model_map_[sgs_model_string_];

  // Set default constant based on model type
  double default_sgs_const = 0.;
  if (sgs_model_type_ == SMAGORINSKY) {
    default_sgs_const = 0.12;
  } else if (sgs_model_type_ == SIGMA) {
    default_sgs_const = 0.135;
  }

  // Get model constant (set to default if not present)
  tps->getInput((basename + "sgsModelConstant").c_str(), sgs_model_constant_, default_sgs_const);
}

LoMachTemporalOptions::LoMachTemporalOptions()
    : integrator_string_("curlcurl"), enable_constant_dt_(false), cfl_(-1.0), constant_dt_(-1.0), bdf_order_(3) {
  integrator_map_["curlcurl"] = CURL_CURL;
  integrator_map_["staggeredTime"] = STAGGERED_TIME;
  integrator_map_["deltaP"] = DELTA_P;
}

void LoMachTemporalOptions::read(TPS::Tps *tps, std::string prefix) {
  std::string basename;
  if (!prefix.empty()) {
    basename = prefix + "/time";
  } else {
    basename = "time";
  }

  tps->getInput((basename + "/cfl").c_str(), cfl_, 0.2);
  tps->getInput((basename + "/integrator").c_str(), integrator_string_, std::string("curlcurl"));

  integrator_type_ = integrator_map_[integrator_string_];

  // At the moment, only curl-curl approach is supported
  assert(integrator_type_ == CURL_CURL);

  tps->getInput((basename + "/enableConstantTimestep").c_str(), enable_constant_dt_, false);
  tps->getInput((basename + "/dt_fixed").c_str(), constant_dt_, -1.0);
  tps->getInput((basename + "/dt_initial").c_str(), initial_dt_, 1.0e-8);
  tps->getInput((basename + "/dt_min").c_str(), minimum_dt_, 1.0e-12);      
  tps->getInput((basename + "/dt_max").c_str(), maximum_dt_, 1.0);    
  tps->getInput((basename + "/bdfOrder").c_str(), bdf_order_, 3);
}
