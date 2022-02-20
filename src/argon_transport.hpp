// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
#ifndef ARGON_TRANSPORT_HPP_
#define ARGON_TRANSPORT_HPP_

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <tps_config.h>
#include <grvy.h>
#include "utils.hpp"

#include <mfem.hpp>
#include <mfem/general/forall.hpp>
#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "transport_properties.hpp"
#include "collision_integrals.hpp"

using namespace mfem;
using namespace std;
// using namespace charged;
// using namespace argon;

class ArgonMinimalTransport : public TransportProperties {
 protected:
   int electronIndex_ = -1;
   int ionIndex_ = -1;
   int neutralIndex_ = -1;

   const double kB_ = BOLTZMANNCONSTANT;
   const double eps0_ = VACUUMPERMITTIVITY;
   const double qe_ = ELECTRONCHARGE;
   const double R_ = UNIVERSALGASCONSTANT;
   const double debyeFactor_ = kB_ * eps0_ / qe_ / qe_;
   const double PI_ = 4.0 * atan(1.0);
   const double qeOverkB_ = qe_ / kB_;

   // standard Chapman-Enskog coefficients
   const double viscosityFactor_ = 5. / 16. * sqrt(PI_ * kB_);
   const double kOverEtaFactor_ = 15. / 4. * kB_;
   const double diffusivityFactor_ = 3. / 16. * sqrt(2.0 * PI_ * kB_) / AVOGADRONUMBER;

   Vector mw_;
   double muAE_;
   double muAI_;
   double muEI_;

   bool thirdOrderkElectron_;

 public:
  ArgonMinimalTransport(GasMixture *_mixture, RunConfiguration &_runfile);

  ~ArgonMinimalTransport(){};

  // Currently, transport properties are evaluated in flux and source term separately.
  // Flux does not take primitive variables as input, rather evaluate them whenever needed.
  // ComputeFluxTransportProperties also evaluates required primitive variables,
  // but do not return it as output.
  // TODO: need to discuss whether to reuse computed primitive variables in flux evaluation,
  // or in general evaluation of primitive variables.
  virtual void ComputeFluxTransportProperties(const Vector &state, const DenseMatrix &gradUp, Vector &transportBuffer,
                                              DenseMatrix &diffusionVelocity);
  // Vector &outputUp);

  // Source term will be constructed using ForcingTerms, which have pointers to primitive variables.
  // So we can use them in evaluating transport properties.
  // If this routine evaluate additional primitive variables, can return them just as the routine above.
  virtual void ComputeSourceTransportProperties(const Vector &state, const Vector &Up, const DenseMatrix &gradUp,
                                                Vector &transportBuffer, DenseMatrix &diffusionVelocity){};

  double computeThirdOrderElectronThermalConductivity(const Vector &X_sp, const double debyeLength, const double Te, const double nondimTe);

  // These are used to compute third-order electron thermal conductivity based on standard Chapman--Enskog method.
  double L11ee(const Vector &Q2) { return Q2(0); }
  double L11ea(const Vector &Q1) { return 6.25 * Q1(0) - 15. * Q1(1) + 12. * Q1(2); }
  double L12ee(const Vector &Q2) { return 1.75 * Q2(0) - 2.0 * Q2(1); }
  double L12ea(const Vector &Q1) { return 10.9375 * Q1(0) - 39.375 * Q1(1) + 57. * Q1(2) - 30. * Q1(3); }
  double L22ee(const Vector &Q2) { return 4.8125 * Q2(0) - 7.0 * Q2(1) + 5. * Q2(2); }
  double L22ea(const Vector &Q1) { return 19.140625 * Q1(0) - 91.875 * Q1(1) + 199.5 * Q1(2) - 210. * Q1(3) + 90. * Q1(4); }
};



#endif  // ARGON_TRANSPORT_HPP_
