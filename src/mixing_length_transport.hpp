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
#ifndef MIXING_LENGTH_TRANSPORT_HPP_
#define MIXING_LENGTH_TRANSPORT_HPP_

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace std;
using namespace mfem;

class MixingLengthTransport : public TransportProperties {
 protected:
  // for turbulent transport calculations
  const double max_mixing_length_;  // user-specifed maximum mixing length
  const double Prt_;                // eddy Prandtl number
  const double Let_;                // eddy Lewis number

  // for molecular transport (owned)
  TransportProperties *molecular_transport_;

 public:
  MixingLengthTransport(GasMixture *mix, RunConfiguration &runfile, TransportProperties *molecular_transport);
  MFEM_HOST_DEVICE MixingLengthTransport(GasMixture *mix, const mixingLengthTransportData &inputs,
                                         TransportProperties *molecular_transport);

  MFEM_HOST_DEVICE virtual ~MixingLengthTransport() { delete molecular_transport_; }

  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, const Vector &Efield,
                                              double distance, Vector &transportBuffer, DenseMatrix &diffusionVelocity);
  MFEM_HOST_DEVICE virtual void ComputeFluxTransportProperties(const double *state, const double *gradUp,
                                                               const double *Efield, double distance,
                                                               double *transportBuffer, double *diffusionVelocity);
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                const Vector &Efield, double distance, Vector &globalTransport,
                                                DenseMatrix &speciesTransport, DenseMatrix &diffusionVelocity,
                                                Vector &n_sp);
  MFEM_HOST_DEVICE virtual void ComputeSourceTransportProperties(const double *state, const double *Up,
                                                                 const double *gradUp, const double *Efield,
                                                                 double distance, double *globalTransport,
                                                                 double *speciesTransport, double *diffusionVelocity,
                                                                 double *n_sp);

  MFEM_HOST_DEVICE void GetViscosities(const double *conserved, const double *primitive, double distance,
                                       double *visc) override;
};

MFEM_HOST_DEVICE inline void MixingLengthTransport::GetViscosities(const double *conserved, const double *primitive,
                                                                   double distance, double *visc) {
  molecular_transport_->GetViscosities(conserved, primitive, distance, visc);
}


#endif  // MIXING_LENGTH_TRANSPORT_HPP_
