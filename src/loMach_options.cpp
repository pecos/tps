#include "loMach_options.hpp"

#include "tps.hpp"

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

  tps->getInput((basename + "/bdfOrder").c_str(), bdf_order_, 3);
}
