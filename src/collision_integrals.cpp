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

#include "collision_integrals.hpp"

namespace collision {

namespace charged {
/*
  collision integrals Q^(l,r) for charged species, based on screened Coulomb potential (attractive/repulsive).
  Functions are fitted from the tabulated data of:
  1. Mason, E. A., Munn, R. J., & Smith, F. J. (1967). Transport coefficients of ionized gases. Physics of Fluids,
  10(8), 1827–1832. https://doi.org/10.1063/1.1762365
  2. Devoto, R. S. (1973). Transport coefficients of ionized argon. Physics of Fluids, 16(5), 616–623.
  https://doi.org/10.1063/1.1694396

  Functions take a non-dimensionalized temperature based on Debye length,
  and return collision integrals in unit of pi * (Debye length)^2.
  For the exact formulation, see equations (A.3) and (A.4) in the following reference:
  Munafò, A., Alberti, A., Pantano, C., Freund, J. B., & Panesi, M. (2020). A computational model for nanosecond pulse
  laser-plasma interactions. Journal of Computational Physics, 406, 109190. https://doi.org/10.1016/j.jcp.2019.109190
*/

MFEM_HOST_DEVICE double att11(const double &Tp) {
  return 0.2150 * pow(log(1.0 + 5.2194 * pow(Tp, 1.0472)), 1.2435) / Tp / Tp;
}

MFEM_HOST_DEVICE double att12(const double &Tp) {
  return 0.0991 * pow(log(1.0 + 7.4684 * pow(Tp, 1.0155)), 1.1536) / Tp / Tp;
}

MFEM_HOST_DEVICE double att13(const double &Tp) {
  return 0.0616 * pow(log(1.0 + 7.8271 * pow(Tp, 0.9452)), 1.1105) / Tp / Tp;
}

MFEM_HOST_DEVICE double att14(const double &Tp) {
  return 0.0308 * pow(log(1.0 + 13.9567 * pow(Tp, 0.9511)), 1.1803) / Tp / Tp;
}

MFEM_HOST_DEVICE double att15(const double &Tp) {
  return 0.0232 * pow(log(1.0 + 13.7888 * pow(Tp, 0.9148)), 1.1532) / Tp / Tp;
}

MFEM_HOST_DEVICE double att22(const double &Tp) {
  return 0.2423 * pow(log(1.0 + 4.6796 * pow(Tp, 1.3290)), 1.1279) / Tp / Tp;
}

MFEM_HOST_DEVICE double att23(const double &Tp) {
  return 0.1221 * pow(log(1.0 + 8.7542 * pow(Tp, 1.3875)), 1.1110) / Tp / Tp;
}

MFEM_HOST_DEVICE double att24(const double &Tp) {
  return 0.0619 * pow(log(1.0 + 18.2538 * pow(Tp, 1.4341)), 1.1618) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep11(const double &Tp) {
  return 0.3904 * pow(log(1.0 + 0.9100 * pow(Tp, 1.1025)), 1.0544) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep12(const double &Tp) {
  return 0.1547 * pow(log(1.0 + 1.6597 * pow(Tp, 1.1725)), 0.9792) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep13(const double &Tp) {
  return 0.0814 * pow(log(1.0 + 2.5815 * pow(Tp, 1.1948)), 0.9570) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep14(const double &Tp) {
  return 0.0683 * pow(log(1.0 + 1.9774 * pow(Tp, 1.2033)), 0.8264) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep15(const double &Tp) {
  return 0.0346 * pow(log(1.0 + 4.5177 * pow(Tp, 1.2132)), 0.9294) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep22(const double &Tp) {
  return 0.4128 * pow(log(1.0 + 1.2436 * pow(Tp, 1.1830)), 1.0123) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep23(const double &Tp) {
  return 0.2203 * pow(log(1.0 + 1.8832 * pow(Tp, 1.2059)), 0.9851) / Tp / Tp;
}

MFEM_HOST_DEVICE double rep24(const double &Tp) {
  return 0.1323 * pow(log(1.0 + 2.7248 * pow(Tp, 1.2129)), 0.9847) / Tp / Tp;
}

}  // namespace charged

// Argon collision integrals
// Q_(species, species)^(l,r)
// Takes T in Kelvin, returns in unit of m^2.
namespace argon {

MFEM_HOST_DEVICE double ArAr11(const double &T) {
  // Reference : fitted from tabulated data of Amdur, I., & Mason, E. A. (1958). Properties of gases at very high
  // temperatures. Physics of Fluids, 1(5), 370–383. https://doi.org/10.1063/1.1724353
  return 2.2910e-18 * pow(T, -0.3032);
}

MFEM_HOST_DEVICE double ArAr22(const double &T) {
  // Reference : Liu, W. S., Whitten, B. T., & Glass, I. I. (1978). Ionizing argon boundary layers. Part 1. Quasi-steady
  // flat-plate laminar boundary-layer flows. Journal of Fluid Mechanics, 87(4), 609–640.
  // https://doi.org/10.1017/S0022112078001792
  return 1.7e-18 * pow(T, -0.25);
}

// argon neutral (Ar) - argon positive ion (Ar1P)
MFEM_HOST_DEVICE double ArAr1P11(const double &T) {
  // Reference: fitted from tabulated data of Devoto, R. S. (1973). Transport coefficients of ionized argon. Physics of
  // Fluids, 16(5), 616–623. https://doi.org/10.1063/1.1694396
  return 4.574321e-18 * pow(T, -0.1805);
}

// General form of the fit used for the e-Ar collision integrals
MFEM_HOST_DEVICE double logT_fit(const double logT, const double coeff[9]) {
  // Evaluate powers of log(T) from -1 through 7
  double logT_power[9];
  logT_power[0] = 1. / logT;
  logT_power[1] = 1.;
  for (int k = 0; k < 7; k++) {
    logT_power[k + 2] = logT_power[k + 1] * logT;
  }

  // fit = sum_k c_k (log(T)^k) for k=-1 to 7
  double fit = 0.0;
  for (int k = 0; k < 9; k++) {
    fit += coeff[k] * logT_power[k];
  }

  return fit;
}

/*
  e-Ar (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,Ar}^(1), elastic momentum transfer cross section, is determined by a 7-parameter shifted MERT model,
  fitted over BSR LXCat dataset.
*/
MFEM_HOST_DEVICE double eAr11(const double &T) {
  const double logT = log(T);
  const double coeff[9] = {6.36254140e-18, 1.84835040e-18,  -5.87727093e-18, 3.20023027e-18, -8.50509054e-19,
                           1.28163820e-19, -1.11712910e-20, 5.25649382e-22,  -1.03296658e-23};
  return logT_fit(logT, coeff);
}

MFEM_HOST_DEVICE double eAr12(const double &T) {
  const double logT = log(T);
  const double coeff[9] = {1.91338172e-17, 5.45418129e-18,  -1.78361685e-17, 9.75657946e-18, -2.61115722e-18,
                           3.98310268e-19, -3.53503678e-20, 1.70375066e-21,  -3.45211955e-23};
  return logT_fit(logT, coeff);
}

MFEM_HOST_DEVICE double eAr13(const double &T) {
  const double logT = log(T);
  const double coeff[9] = {3.04685398e-17, 8.39750994e-18,  -2.88132528e-17, 1.60147037e-17, -4.34837891e-18,
                           6.73136845e-19, -6.06704580e-20, 2.97216168e-21,  -6.12760944e-23};
  return logT_fit(logT, coeff);
}

MFEM_HOST_DEVICE double eAr14(const double &T) {
  const double logT = log(T);
  const double coeff[9] = {3.90777949e-17, 1.04696956e-17,  -3.73774204e-17, 2.10610498e-17, -5.79029566e-18,
                           9.07573157e-19, -8.28466766e-20, 4.11188110e-21,  -8.59225098e-23};
  return logT_fit(logT, coeff);
}

MFEM_HOST_DEVICE double eAr15(const double &T) {
  const double logT = log(T);
  const double coeff[9] = {4.41333290e-17, 1.15696010e-17,  -4.25651305e-17, 2.42442440e-17, -6.73359258e-18,
                           1.06641697e-18, -9.83933863e-20, 4.93775812e-21,  -1.04362372e-22};
  return logT_fit(logT, coeff);
}

}  // namespace argon

}  // namespace collision
