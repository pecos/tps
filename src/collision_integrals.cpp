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

/*
  e-Ar (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,Ar}^(1), elastic momentum transfer cross section, is determined by a 7-parameter shifted MERT model,
  fitted over BSR LXCat dataset.
*/
MFEM_HOST_DEVICE double eAr11(const double &T) {
  const double logT = log(T);
  if (T < 1.2e4) {
    return 5.8664e-22 * logT * logT * logT - 6.3417e-21 * logT * logT + 3.2083e-21 * logT + 9.0686e-20;
  } else {
    double tmp = logT - 10.9082;
    return 12.1818e-20 * exp(-0.4186 * tmp * tmp) + 8.6949e-22;
  }
}

MFEM_HOST_DEVICE double eAr12(const double &T) {
  const double logT = log(T);
  if (T < 1.0e4) {
    return 5.0435e-22 * logT * logT * logT - 4.0041e-21 * logT * logT - 1.3234e-20 * logT + 1.1966e-19;
  } else {
    double tmp = logT - 10.5348;
    return 12.5836e-20 * exp(-0.5116 * tmp * tmp) + 8.7254e-22;
  }
}

MFEM_HOST_DEVICE double eAr13(const double &T) {
  const double logT = log(T);
  if (T < 8.2e3) {
    return 4.3150e-22 * logT * logT * logT - 2.1312e-21 * logT * logT - 2.5311e-20 * logT + 1.3866e-19;
  } else {
    double tmp = logT - 10.2802;
    return 12.9711e-20 * exp(-0.5725 * tmp * tmp) + 1.6371e-21;
  }
}

MFEM_HOST_DEVICE double eAr14(const double &T) {
  const double logT = log(T);
  if (T < 7.1e3) {
    return 3.9545e-22 * logT * logT * logT - 1.1198e-21 * logT * logT - 3.1302e-20 * logT + 1.4507e-19;
  } else {
    double tmp = logT - 10.0853;
    return 13.2903e-20 * exp(-0.6150 * tmp * tmp) + 1.7854e-21;
  }
}

MFEM_HOST_DEVICE double eAr15(const double &T) {
  const double logT = log(T);
  if (T < 6.0e3) {
    return 2.8521e-22 * logT * logT * logT + 9.9567e-22 * logT * logT - 4.2614e-20 * logT + 1.6026e-19;
  } else {
    double tmp = logT - 9.9275;
    return 13.4901e-20 * exp(-0.6295 * tmp * tmp) + 4.6041e-22;
  }
}

}  // namespace argon

// Nitrogen collision integrals
// Q_(species, species)^(l,r)
// Takes T in Kelvin, returns in unit of m^2.
// All r=2 uses "isotropic scattering assumption", see: Sherman, M. P. (1965). Transport properties of partially ionized nitrogen. NASA: R65SD43.
namespace nitrogen {

// Reference : Levin et. al. (1990). "Collision Integrals and High Temperature Transport Properties for N-N, O-O, and N-O".
MFEM_HOST_DEVICE double NiNi11(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -34.495061828795336;
  c[1] = -4.216558629806175;
  c[2] = 0.704162022307954;
  c[3] = -0.052865133429065;
  c[4] = 0.001398268912999;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  */

  // see: Capitelli 2000    
  double logT = log(T);
  double c[2];
  c[0] = -41.999345922993626;
  c[1] = -0.294552697364472;
  double expOmega = c[0] + c[1]*logT;
  return exp(expOmega);
  
}

// Reference : Levin et. al. (1990). "Collision Integrals and High Temperature Transport Properties for N-N, O-O, and N-O".
MFEM_HOST_DEVICE double NiNi22(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -36.815543020557080;
  c[1] = -2.950557093855711;
  c[2] = 0.464808279754732;
  c[3] = -0.033419105245876;
  c[4] = 0.000830667315518;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  */

  // see: Su 2023
  double logT = log(T);
  double c[2];
  c[0] = -42.134170027961517;
  c[1] = -0.264742466936063;
  double expOmega = c[0] + c[1]*logT;
  return exp(expOmega);
  
}

// N-N+: NO DATA (using N-N)
// see: su 2023  
MFEM_HOST_DEVICE double NiNi1P11(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -39.100221303876083;
  c[1] = -4.216563895965268;
  c[2] = 0.704162971673771;
  c[3] = -0.052865207953036;
  c[4] = 0.001398271066244;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  */
  
  // su...
  double logT = log(T);
  double c[5];
  c[0] = -38.031557701992455;
  c[1] = -1.615788958431633;
  c[2] = 0.232342802535233;
  c[3] = -0.015601342343555;
  c[4] = 0.000385790678744;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  
}

// Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
// see also: Capitelli 2000    
MFEM_HOST_DEVICE double N2N211(const double &T) {
  
  /*
  double logT = log(T);
  double c[5];
  c[0] = -1.028860574431997;
  c[1] = 0.314037073072164;
  c[2] = -0.059056910722227;
  c[3] = 0.004811832484639;
  c[4] = -0.000145066239802;
  double expOmega = 100.0 * (c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4));
  return exp(expOmega);
  */

  // su2023
  double logT = log(T);
  double c[2];
  c[0] = -42.225712551892755;
  c[1] = -0.229958945507976;
  double expOmega = c[0] + c[1]*logT;
  return exp(expOmega);
  
}

// Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"
// see also: Capitelli 2000    
MFEM_HOST_DEVICE double N2N222(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -52.253488600180702;
  c[1] = 7.253691451264671;
  c[2] = -1.582492289893343;
  c[3] = 0.140011494486633;
  c[4] = -0.004476593767047;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  */
  
  // su2023
  double logT = log(T);
  double c[2];
  c[0] = -42.077467697607034;
  c[1] = -0.229160550336776;
  double expOmega = c[0] + c[1]*logT;
  return exp(expOmega);
  
}

// Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"  
MFEM_HOST_DEVICE double N2N21P11(const double &T) {
  
  double logT = log(T);
  double c[5];
  c[0] = -96.770585022102779;
  c[1] = 29.148723006721372;
  c[2] = -5.530060545847180;
  c[3] = 0.455326558461576;
  c[4] = -0.013842724823926;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
    
}

// Reference : A. V. Phelps. (1991). "Cross Sections and Swarm Coefficients for Nitrogen Ions and Neutrals in N2 and Argon Ions and Neutrals in Ar for Energies from 0.1 eV to 10 keV"  
MFEM_HOST_DEVICE double N2Ni1P11(const double &T) {

  double logT = log(T);
  double c[7];
  c[0] = -7.070758182949830;
  c[1] = 4.845317363412498;
  c[2] = -1.458758839937497;
  c[3] = 0.233156378931474;
  c[4] = -0.020896682219994;
  c[5] = 0.000996062674946;
  c[6] = -0.000019731218085;
  double expOmega = 100.0 * (c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6));
  return exp(expOmega);       
  
}

// N-N2+ : NO DATA (using N2-N+)
// NOTE: this one doesnt really matter as N2 is all N by the temp of N2+  
MFEM_HOST_DEVICE double NiN21P11(const double &T) {

  double logT = log(T);
  double c[7];
  c[0] = -7.070758182949830;
  c[1] = 4.845317363412498;
  c[2] = -1.458758839937497;
  c[3] = 0.233156378931474;
  c[4] = -0.020896682219994;
  c[5] = 0.000996062674946;
  c[6] = -0.000019731218085;
  double expOmega = 100.0 * (c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6));
  return exp(expOmega);  

}

// N2-N : NO DATA (using N2-N+)
// see: su2023  
MFEM_HOST_DEVICE double N2Ni11(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -1.017786766181694;
  c[1] = 0.310545068504458;
  c[2] = -0.057851309103005;
  c[3] = 0.004669770398097;
  c[4] = -0.000140132790060;
  double expOmega = 100.0 * (c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4));
  return exp(expOmega);
  */
  
  // su
  double logT = log(T);
  double c[5];
  c[0] = 24.496049925340895;
  c[1] = -27.786968224881310;
  c[2] = 4.261190501951376;
  c[3] = -0.293139208272160;
  c[4] = 0.007561754781801;
  double expOmega = 100.0 * (c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4));
  return exp(expOmega);  
  
}

// N2-N : NO DATA  (using N-N 22)
// see: su2023
MFEM_HOST_DEVICE double N2Ni22(const double &T) {

  /*
  double logT = log(T);
  double c[5];
  c[0] = -39.100221303876083;
  c[1] = -4.216563895965268;
  c[2] = 0.704162971673771;
  c[3] = -0.052865207953036;
  c[4] = 0.001398271066244;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);
  */
  
  // su
  double logT = log(T);
  double c[5];
  c[0] = -62.712016967753847;
  c[1] = 7.212207597803123;
  c[2] = -0.988167069227977;
  c[3] = 0.056255946061936;
  c[4] = -0.001145315858929;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4);
  return exp(expOmega);  
  
}

  
/*
  e-N (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,N}^(1), elastic momentum transfer cross section, 
  fitted over IAA LXCat dataset.
*/
MFEM_HOST_DEVICE double eNi11(const double &T) {

  double logT = log(T);
  double c[7];
  c[0] = 2.583657310241357;
  c[1] = -3.338700252668392;
  c[2] = 1.384354837198594;
  c[3] = -0.284334080031242;
  c[4] = 0.031157728257488;
  c[5] = -0.001750672149711;
  c[6] = 0.000039758329849;
  for(int i=0; i<7; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6);
  return exp(expOmega);
     
}

MFEM_HOST_DEVICE double eNi12(const double &T) {
  
  double logT = log(T);
  double c[7];
  c[0] = -1.934755833917042;
  c[1] = 1.396606776346035;
  c[2] = -0.422114799937392;
  c[3] = 0.067093051236947;
  c[4] = -0.005928358157850;
  c[5] = 0.000276540278651;
  c[6] = -0.000005326503633;
  for(int i=0; i<7; i++) c[i] *= 1000.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6);
  return exp(expOmega);
  
}

MFEM_HOST_DEVICE double eNi13(const double &T) {
  
  double logT = log(T);
  double c[7];
  c[0] = -2.837349005539165;
  c[1] = 2.184292700049795;
  c[2] = -0.700859738170989;
  c[3] = 0.118480284536834;
  c[4] = -0.011150119826904;
  c[5] = 0.000554606302530;
  c[6] = -0.000011402466225;
  for(int i=0; i<7; i++) c[i] *= 1000.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6);
  return exp(expOmega);
  
}

MFEM_HOST_DEVICE double eNi14(const double &T) {
  
  double logT = log(T);
  double c[7];
  c[0] = -1.673254218486384;
  c[1] = 1.339721615950149;
  c[2] = -0.448694918786985;
  c[3] = 0.078770108123025;
  c[4] = -0.007668193507749;
  c[5] = 0.000393314841564;
  c[6] = -0.000008317289766;
  for(int i=0; i<7; i++) c[i] *= 1000.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6);
  return exp(expOmega);
  
}

MFEM_HOST_DEVICE double eNi15(const double &T) {
  
  double logT = log(T);
  double c[7];
  c[0] = -1.245791078716272;
  c[1] = 1.535014697501557;
  c[2] = -0.740292376788452;
  c[3] = 0.162391831811548;
  c[4] = -0.018468300738766;
  c[5] = 0.001064873335057;
  c[6] = -0.000024703118057;
  for(int i=0; i<7; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5) + c[6]*pow(logT,6);
  return exp(expOmega);
  
}

MFEM_HOST_DEVICE double eN211(const double &T) {

  double logT = log(T);
  double c[6];
  c[0] = -3.847278097767338;
  c[1] = 2.151944606452283;
  c[2] = -0.537317123236616;
  c[3] = 0.066165032113028;
  c[4] = -0.004009615832468;
  c[5] = 0.000095579640766;
  for(int i=0; i<6; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5);
  return exp(expOmega);
     
}

MFEM_HOST_DEVICE double eN212(const double &T) {
  
  double logT = log(T);
  double c[6];
  c[0] = -5.337534425696322;
  c[1] = 3.238333171420736;
  c[2] = -0.845936604069987;
  c[3] = 0.109071857868453;
  c[4] = -0.006934205079810;
  c[5] = 0.000173872831529;
  for(int i=0; i<6; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5);
  return exp(expOmega);
  
}

MFEM_HOST_DEVICE double eN213(const double &T) {
  
  double logT = log(T);
  double c[6];
  c[0] = -6.527006679994851;
  c[1] = 4.109423857482659;
  c[2] = -1.096380242694045;
  c[3] = 0.144466323778224;
  c[4] = -0.009393664358810;
  c[5] = 0.000241114854285;
  for(int i=0; i<6; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5);
  return exp(expOmega);  
  
}

MFEM_HOST_DEVICE double eN214(const double &T) {
  
  double logT = log(T);
  double c[6];
  c[0] = -6.527006679994851;
  c[1] = 4.109423857482659;
  c[2] = -1.096380242694045;
  c[3] = 0.144466323778224;
  c[4] = -0.009393664358810;
  c[5] = 0.000241114854285;
  for(int i=0; i<6; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5);
  return exp(expOmega);  
  
}

MFEM_HOST_DEVICE double eN215(const double &T) {
  
  double logT = log(T);
  double c[6];
  c[0] = -5.131403805671025;
  c[1] = 3.295203439069785;
  c[2] = -0.914505280894294;
  c[3] = 0.125300650936993;
  c[4] = -0.008464520705942;
  c[5] = 0.000225459901920;
  for(int i=0; i<6; i++) c[i] *= 100.0;
  double expOmega = c[0] + c[1]*logT + c[2]*pow(logT,2) + c[3]*pow(logT,3) + c[4]*pow(logT,4) + c[5]*pow(logT,5);
  return exp(expOmega);  
  
}
  
}  // namespace nitrogen

  
}  // namespace collision
