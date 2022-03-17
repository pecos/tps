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
#ifndef COLLISION_INTEGRALS_HPP_
#define COLLISION_INTEGRALS_HPP_

#include <math.h>

using namespace std;

namespace collision
{

namespace charged
{
/*
  collision integrals Q^(l,r) for charged species, based on screened Coulomb potential (attractive/repulsive).
  Functions are fitted from the tabulated data of:
  1. Mason, E. A., Munn, R. J., & Smith, F. J. (1967). Transport coefficients of ionized gases. Physics of Fluids, 10(8), 1827–1832. https://doi.org/10.1063/1.1762365
  2. Devoto, R. S. (1973). Transport coefficients of ionized argon. Physics of Fluids, 16(5), 616–623. https://doi.org/10.1063/1.1694396

  Functions take a non-dimensionalized temperature based on Debye length,
  and return collision integrals in unit of pi * (Debye length)^2.
  For the exact formulation, see equations (A.3) and (A.4) in the following reference:
  Munafò, A., Alberti, A., Pantano, C., Freund, J. B., & Panesi, M. (2020). A computational model for nanosecond pulse laser-plasma interactions. Journal of Computational Physics, 406, 109190. https://doi.org/10.1016/j.jcp.2019.109190
*/

double att11(const double Tp);
double att12(const double Tp);
double att13(const double Tp);
double att14(const double Tp);
double att15(const double Tp);

double att22(const double Tp);
double att23(const double Tp);
double att24(const double Tp);

double rep11(const double Tp);
double rep12(const double Tp);
double rep13(const double Tp);
double rep14(const double Tp);
double rep15(const double Tp);

double rep22(const double Tp);
double rep23(const double Tp);
double rep24(const double Tp);

} // charged

// Argon collision integrals
// Q_(species, species)^(l,r)
// Takes T in Kelvin, returns in unit of m^2.
namespace argon
{

double ArAr22(const double T);

// argon neutral (Ar) - argon positive ion (Ar1P)
double ArAr1P11(const double T);

/*
  e-Ar (l,r) are fitted over numerical quadrature of definitions.
  Q_{e,Ar}^(1), elastic momentum transfer cross section, is determined by a 7-parameter shifted MERT model,
  fitted over BSR LXCat dataset.
*/
double eAr11(const double T);
double eAr12(const double T);
double eAr13(const double T);
double eAr14(const double T);
double eAr15(const double T);

} // argon


} // collision
#endif  // COLLISION_INTEGRALS_HPP_
