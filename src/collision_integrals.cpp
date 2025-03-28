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

// Nitrogen collision integrals
// Q_(species, species)^(l,r)
// Takes T in Kelvin, returns in unit of m^2.
// All r=2 uses "isotropic scattering assumption", see: Sherman, M. P. (1965). Transport properties of partially ionized nitrogen. NASA: R65SD43.
namespace nitrogen {

MFEM_HOST_DEVICE double NiNi11(const double &T) {
  // Reference : Levin et. al. (1990). "Collision Integrals and High Temperature Transport Properties for N-N, O-O, and N-O".
  double logT = log(T);
  double c[5];
  c[0] = -43.674589506069054;
  c[1] = -1.894798783975738;
  c[2] = -0.003096113560006;
  c[3] = -12.737655547787776;
  c[4] = 2.673418610304956;  
  double expOmega = c[0] + c[1]*pow(logT,c[3]) + c[2]*pow(logT,c[4]);
  return log(expOmega);
}

MFEM_HOST_DEVICE double NiNi22(const double &T) {
  // Reference : Levin et. al. (1990). "Collision Integrals and High Temperature Transport Properties for N-N, O-O, and N-O".  
  double logT = log(T);
  double c[5];
  c[0] = -43.501008926336119;
  c[1] = -0.184736895556468;
  c[2] = -0.005592899012521;
  c[3] = -11.604191683888978;
  c[4] = 2.417462743199025;
  double expOmega = c[0] + c[1]*pow(logT,c[3]) + c[2]*pow(logT,c[4]);
  return log(expOmega);
}

// N-N+: NO DATA (using N-N)
MFEM_HOST_DEVICE double NiNi1P11(const double &T) {
  double logT = log(T);
  double c[5];
  c[0] = -43.674589506069054;
  c[1] = -1.894798783975738;
  c[2] = -0.003096113560006;
  c[3] = -12.737655547787776;
  c[4] = 2.673418610304956;  
  double expOmega = c[0] + c[1]*pow(logT,c[3]) + c[2]*pow(logT,c[4]);
  return log(expOmega);  
}

MFEM_HOST_DEVICE double N2N211(const double &T) {
  // Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
  double logT = log(T);
  double c[3];
  c[0] = -47.866523949904071;
  c[1] = 0.531494355003653;
  c[2] = 0.010748072674994;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;      
  return log(expOmega);  
}

MFEM_HOST_DEVICE double N2N222(const double &T) {
  // Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
  double logT = log(T);
  double c[3];
  c[0] = -47.200458725512135;
  c[1] = 0.544720304287795;
  c[2] = 0.010160903112939;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;    
  return log(expOmega);  
}

MFEM_HOST_DEVICE double N2N21P11(const double &T) {
  // Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
  double logT = log(T);
  double c[3];
  c[0] = -47.318323876430846;
  c[1] = 0.661701532231132;
  c[2] = 0.011601814193470;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;  
  return log(expOmega);  
}

MFEM_HOST_DEVICE double N2Ni1P11(const double &T) {
  // Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
  double logT = log(T);
  double c[3];
  c[0] = -47.855538582567668;
  c[1] = 0.856571008228926;
  c[2] = -0.018389550018588;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}

// N-N2+ : NO DATA (using N2-Ni+)
MFEM_HOST_DEVICE double NiN21P11(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -47.855538582567668;
  c[1] = 0.856571008228926;
  c[2] = -0.018389550018588;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}
  
  
/*
  e-Ar (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,Ar}^(1), elastic momentum transfer cross section, 
  fitted over IAA LXCat dataset.
*/
MFEM_HOST_DEVICE double eNi11(const double &T) {
  double logT = log(T);
  double c[8];
  c[0] = -2.003882301835628;
  c[1] = 1.532253343478155;
  c[2] = -0.505597421163568;
  c[3] = 0.091480325046587;
  c[4] = -0.009821686553564;
  c[5] = 0.000626838643976;
  c[6] = -0.000022046833696;
  c[7] = 0.000000329903530;   
  for(int i=0; i<8; i++) c[i] *= 1000.0;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT + c[2]*pow(logT,3) + c[3]*pow(logT,4) + c[4]*pow(logT,5) + c[5]*pow(logT,6) + c[6]*pow(logT,7) + c[7]*pow(logT,8);
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eNi12(const double &T) {
  double logT = log(T);
  double c[8];
  c[0] = -1.956159066307375;
  c[1] = 1.557221870910142;
  c[2] = -0.533922416988002;
  c[3] = 0.100141637408542;
  c[4] = -0.011119131479736;
  c[5] = 0.000732252201714;
  c[6] = -0.000026518334060;
  c[7] = 0.000000407775769;
  for(int i=0; i<8; i++) c[i] *= 1000.0;  
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT + c[2]*pow(logT,3) + c[3]*pow(logT,4) + c[4]*pow(logT,5) + c[5]*pow(logT,6) + c[6]*pow(logT,7) + c[7]*pow(logT,8);
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eNi13(const double &T) {
  double logT = log(T);
  double c[8];
  c[0] = -4.514549781113058;
  c[1] = 3.776124850464214;
  c[2] = -1.427950653286536;
  c[3] = 0.289498661175164;
  c[4] = -0.034291085489797;
  c[5] = 0.002389263184571;
  c[6] = -0.000091052188327;
  c[7] = 0.000001467616631;
  for(int i=0; i<8; i++) c[i] *= 100.0;    
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT + c[2]*pow(logT,3) + c[3]*pow(logT,4) + c[4]*pow(logT,5) + c[5]*pow(logT,6) + c[6]*pow(logT,7) + c[7]*pow(logT,8);
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eNi14(const double &T) {
  double logT = log(T);
  double c[8];
  c[0] = 7.850307064335757;
  c[1] = -6.131538141527012;
  c[2] = 1.925885314608013;
  c[3] = -0.333041600780530;
  c[4] = 0.034213058505903;
  c[5] = -0.002083820862016;
  c[6] = 0.000069580148773;
  c[7] = -0.000000982075309;
  for(int i=0; i<8; i++) c[i] *= 100.0;  
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT + c[2]*pow(logT,3) + c[3]*pow(logT,4) + c[4]*pow(logT,5) + c[5]*pow(logT,6) + c[6]*pow(logT,7) + c[7]*pow(logT,8);
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eNi15(const double &T) {
  double logT = log(T);
  double c[8];
  c[0] = 9.350737481569830;  
  c[1] = -7.411741684073336;
  c[2] = 2.380778500283301;
  c[3] = -0.420842759318950;
  c[4] = 0.044194959058462;
  c[5] = -0.002753071048199;
  c[6] = 0.000094064763399;
  c[7] = -0.000001358467597;
  for(int i=0; i<8; i++) c[i] *= 100.0;    
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT + c[2]*pow(logT,3) + c[3]*pow(logT,4) + c[4]*pow(logT,5) + c[5]*pow(logT,6) + c[6]*pow(logT,7) + c[7]*pow(logT,8);
  return log(expOmega);  
}

/*
  e-Ar (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,Ar}^(1), elastic momentum transfer cross section,
  fitted over IAA LXCat dataset.
*/
MFEM_HOST_DEVICE double eN211(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -57.365347800354527;
  c[1] = 2.237881478639873;
  c[2] = -0.064095690853854;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eN212(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -57.103935279579566;
  c[1] = 2.310012393118432;
  c[2] = -0.069886269572701;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eN213(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -57.352822536318413;
  c[1] = 2.444566251016784;
  c[2] = -0.078578082485722;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eN214(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -57.643870857111693;
  c[1] = 2.561369597252678;
  c[2] = -0.086168461442770;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}

MFEM_HOST_DEVICE double eN215(const double &T) {
  double logT = log(T);
  double c[3];
  c[0] = -57.774233348256821;
  c[1] = 2.632018476967198;
  c[2] = -0.091227910380667;
  double expOmega = c[0] + c[1]*logT + c[2]*logT*logT;
  return log(expOmega);  
}
  
}  // namespace nitrogen

  
}  // namespace collision
