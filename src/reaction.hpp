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
#ifndef REACTION_HPP_
#define REACTION_HPP_

/*
  Reaction class determines reaction rate coefficient of a single reaction.
  The set of reaction classes is handled by Chemistry class.
*/

#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

using namespace mfem;
using namespace std;

class Reaction {
 protected:
 public:
  const ReactionModel reactionModel;
  MFEM_HOST_DEVICE Reaction(ReactionModel rm) : reactionModel(rm) {}

  MFEM_HOST_DEVICE virtual ~Reaction() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         [[maybe_unused]] const int &dofindex,
                                                         const bool isElectronInvolved = false,
                                                         [[maybe_unused]] const double *nsp = 0) {
    printf("computeRateCoefficient not implemented");
    return 0;
  }
};

class Arrhenius : public Reaction {
 private:
  // coefficients for modified Arrhenius model: k = A * T^b * exp( - E / R / T )
  double A_;
  double b_;
  double E_;

 public:
  MFEM_HOST_DEVICE Arrhenius(const double &A, const double &b, const double &E);

  MFEM_HOST_DEVICE virtual ~Arrhenius() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         [[maybe_unused]] const int &dofindex,
                                                         const bool isElectronInvolved = false,
                                                         [[maybe_unused]] const double *nsp = 0);
};

class HoffertLien : public Reaction {
  /*
    referece: Hoffert, M. I., & Lien, H. (1967). Quasi-one-dimensional, nonequilibrium gas dynamics of partially ionized
    two-temperature argon. Physics of Fluids, 10(8), 1769–1777. https://doi.org/10.1063/1.1762356
  */
 private:
  // coefficients for modified Arrhenius model: k = A * T^b * exp( - E / R / T )
  double A_;
  double b_;
  double E_;

 public:
  MFEM_HOST_DEVICE HoffertLien(const double &A, const double &b, const double &E);

  MFEM_HOST_DEVICE virtual ~HoffertLien() {}

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         [[maybe_unused]] const int &dofindex,
                                                         const bool isElectronInvolved = false,
                                                         [[maybe_unused]] const double *nsp = 0);
};

class Tabulated : public Reaction {
 private:
  TableInterpolator *table_ = NULL;

 public:
  MFEM_HOST_DEVICE Tabulated(const TableInput &input);

  MFEM_HOST_DEVICE virtual ~Tabulated();

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e,
                                                         [[maybe_unused]] const int &dofindex,
                                                         const bool isElectronInvolved = false,
                                                         [[maybe_unused]] const double *nsp = 0);
};

class GridFunctionReaction : public Reaction {
 private:
  const double *data_;
  const int comp_;
  int size_;

 public:
  MFEM_HOST_DEVICE GridFunctionReaction(int comp);

  MFEM_HOST_DEVICE virtual ~GridFunctionReaction();

  void setGridFunction(const mfem::GridFunction &f);

  MFEM_HOST_DEVICE void setData(const double *data, int size);

  MFEM_HOST_DEVICE virtual double computeRateCoefficient([[maybe_unused]] const double &T_h,
                                                         [[maybe_unused]] const double &T_e, const int &dofindex,
                                                         [[maybe_unused]] const bool isElectronInvolved = false,
                                                         [[maybe_unused]] const double *nsp = 0);
};

class RadiativeDecay : public Reaction {
 private:
  const double big = 1.0e+50;
  const double small = 1.0e-50;

  // Length parameters used for the evaluation of the escape factors
  const double Rcyl;  // Radius of a cylinder
  double Lcyl;        // Length of a cylinder

  std::string upper_sp_name;
  std::string lower_sp_name;

  int rank0_;

  int iAr_u;
  int iAr_l;

  // Parameters
  const double Mr_Ar = 0.039948;       // [kg/mol]
  const double M_Ar = 6.63352088e-26;  // Mr_Ar/spc.N_A # [kg] mass of argon atom (6.63352088e-26 kg)
  const double SPEEDOFLIGHT = 299792458.;

  // Variables
  int NumOfInteral_lvl_u;
  std::vector<double> *E_lvl_u;
  std::vector<double> *g_lvl_u;
  std::vector<double> n_sp_lvl_u;
  std::vector<std::vector<double>> *Aji;
  std::vector<double> effAcoef_lvl;
  std::vector<double> *E_lvl_l;
  std::vector<double> *g_lvl_l;
  std::vector<double> n_sp_lvl_l;
  int numSpecies_;

  // Data of internal levels of lumped species
  // ground state
  std::vector<double> E_lvl_g = {0.0};
  std::vector<double> g_lvl_g = {1.0};

  // 4s metastables
  std::vector<double> E_lvl_m = {1114246.8116913952,
                                 1131113.0237639823};  // [J/mol] // {11.54835442, 11.72316039}; // [eV]
  std::vector<double> g_lvl_m = {5.0, 1.0};

  // 4s resonance
  std::vector<double> E_lvl_r = {1121506.2040552883,
                                 1141235.3742507447};  // [J/mol] // {11.62359272, 11.82807116}; // [eV]
  std::vector<double> g_lvl_r = {3.0, 3.0};
  std::vector<std::vector<double>> Aji_r_g = {{132000000.0}, {532000000.0}};

  // 4p states
  std::vector<double> E_lvl_4p = {1245337.6579411437, 1280653.4893638478, 1261614.7730293325, 1263463.1280640187,
                                  1269085.454762629,  1270883.3460389085, 1281579.837318737,  1283469.8354227678,
                                  1285942.7139612488, 1300611.3568123293};  // [J/mol]
  // {12.90701530, 13.27303810, 13.07571571, 13.09487256, 13.15314387,
  // 13.17177770, 13.28263902, 13.30222747, 13.32785705, 13.47988682}; // [eV]
  std::vector<double> g_lvl_4p = {3.0, 1.0, 7.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 1.0};
  std::vector<std::vector<double>> Aji_4p_m = {
      {18900000.0, 980000.0},  {33000000.0, 0.0}, {9300000.0, 0.0},       {5200000.0, 2430000.0},
      {24500000.0, 0.0},       {0.0, 0.0},        {630000.0, 18600000.0}, {3800000.0, 0.0},
      {6400000.0, 11700000.0}, {0.0, 0.0}};

  std::vector<std::vector<double>> Aji_4p_r = {{5400000.0, 190000.0},   {0.0, 0.0},
                                               {21500000.0, 1470000.0}, {25000000.0, 1060000.0},
                                               {4900000.0, 5000000.0},  {40000000.0, 8643.18384420115},
                                               {22000.0, 13900000.0},   {8500000.0, 22300000.0},
                                               {1830000.0, 15300000.0}, {236000.0, 45000000.0}};

  // Excited levels based on NIST database
  // Racah          Paschen   index
  // Ar               1s1       n/a     Ground state

  // Ar(4s[3/2]2)     1s5       0       1st metastable
  // Ar(4s[3/2]1)     1s4       0       1st resonacne

  // Ar(4s'[1/2]0)    1s3       1       2nd metastable
  // Ar(4s'[1/2]1)    1s2       1       2nd resonacne

  // Ar(4p[1/2]1)     2p10	    0       4p states
  // Ar(4p[5/2]3)     2p9	      1
  // Ar(4p[5/2]2)     2p8	      2
  // Ar(4p[3/2]1)     2p7	      3
  // Ar(4p[3/2]2)     2p6	      4
  // Ar(4p[1/2]0)     2p5	      5
  // Ar(4p'[3/2]1)    2p4	      6
  // Ar(4p'[3/2]2)    2p3	      7
  // Ar(4p'[1/2]1)    2p2       8
  // Ar(4p'[1/2]0)    2p1       9

  // Transitions based on NIST database
  // wavelength   upper level             lower level       A coef
  // 0            Ar(4s[3/2]2)      ->        Ar          0.0
  // 106.666      Ar(4s[3/2]1)      ->        Ar          132000000.0
  // 0            Ar(4s'[1/2]0)     ->        Ar          0.0
  // 104.822      Ar(4s'[1/2]1)     ->        Ar          532000000.0

  // 912.547      Ar(4p[1/2]1)      ->    Ar(4s[3/2]2)    18900000.0
  // 966.043      Ar(4p[1/2]1)      ->    Ar(4s[3/2]1)    5400000.0
  // 1047.292     Ar(4p[1/2]1)      ->    Ar(4s'[1/2]0)   980000.0
  // 1149.125     Ar(4p[1/2]1)      ->    Ar(4s'[1/2]1)   190000.0

  // 811.754      Ar(4p[5/2]3)      ->    Ar(4s[3/2]2)    33000000.0
  // 0            Ar(4p[5/2]3)      ->    Ar(4s[3/2]1)    0.0
  // 0            Ar(4p[5/2]3)      ->    Ar(4s'[1/2]0)   0.0
  // 0            Ar(4p[5/2]3)      ->    Ar(4s'[1/2]1)   0.0

  // 801.699      Ar(4p[5/2]2)      ->    Ar(4s[3/2]2)    9300000.0
  // 842.696      Ar(4p[5/2]2)      ->    Ar(4s[3/2]1)    21500000.0
  // 0            Ar(4p[5/2]2)      ->    Ar(4s'[1/2]0)   0.0
  // 978.719      Ar(4p[5/2]2)      ->    Ar(4s'[1/2]1)   1470000.0

  // 772.589      Ar(4p[3/2]1)      ->    Ar(4s[3/2]2)    5200000.0
  // 810.592      Ar(4p[3/2]1)      ->    Ar(4s[3/2]1)    25000000.0
  // 867.032      Ar(4p[3/2]1)      ->    Ar(4s'[1/2]0)   2430000.0
  // 935.678      Ar(4p[3/2]1)      ->    Ar(4s'[1/2]1)   1060000.0

  // 763.720      Ar(4p[3/2]2)      ->    Ar(4s[3/2]2)    24500000.0
  // 800.836      Ar(4p[3/2]2)      ->    Ar(4s[3/2]1)    4900000.0
  // 0            Ar(4p[3/2]2)      ->    Ar(4s'[1/2]0)   0.0
  // 922.703      Ar(4p[3/2]2)      ->    Ar(4s'[1/2]1)   5000000.0

  // 0            Ar(4p[1/2]0)      ->    Ar(4s[3/2]2)    0.0
  // 751.672      Ar(4p[1/2]0)      ->    Ar(4s[3/2]1)    40000000.0
  // 0            Ar(4p[1/2]0)      ->    Ar(4s'[1/2]0)   0.0
  // 858.042      Ar(4p[1/2]0)      ->    Ar(4s'[1/2]1)   8643.18384420115

  // 714.901      Ar(4p'[3/2]1)     ->    Ar(4s[3/2]2)    630000.0
  // 747.322      Ar(4p'[3/2]1)     ->    Ar(4s[3/2]1)    22000.0
  // 795.036      Ar(4p'[3/2]1)     ->    Ar(4s'[1/2]0)   18600000.0
  // 852.378      Ar(4p'[3/2]1)     ->    Ar(4s'[1/2]1)   13900000.0

  // 706.917      Ar(4p'[3/2]2)     ->    Ar(4s[3/2]2)    3800000.0
  // 738.601      Ar(4p'[3/2]2)     ->    Ar(4s[3/2]1)    8500000.0
  // 0            Ar(4p'[3/2]2)     ->    Ar(4s'[1/2]0)   0.0
  // 841.052      Ar(4p'[3/2]2)     ->    Ar(4s'[1/2]1)   22300000.0

  // 696.735      Ar(4p'[1/2]1)     ->    Ar(4s[3/2]2)    6400000.0
  // 727.494      Ar(4p'[1/2]1)     ->    Ar(4s[3/2]1)    1830000.0
  // 772.633      Ar(4p'[1/2]1)     ->    Ar(4s'[1/2]0)   11700000.0
  // 826.679      Ar(4p'[1/2]1)     ->    Ar(4s'[1/2]1)   15300000.0

  // 0            Ar(4p'[1/2]0)     ->    Ar(4s[3/2]2)    0.0
  // 667.912      Ar(4p'[1/2]0)     ->    Ar(4s[3/2]1)    236000.0
  // 0            Ar(4p'[1/2]0)     ->    Ar(4s'[1/2]0)   0.0
  // 750.59       Ar(4p'[1/2]0)     ->    Ar(4s'[1/2]1)   45000000.0

 public:
  MFEM_HOST_DEVICE RadiativeDecay(const double _R, const std::map<std::string, int> *_speciesMapping,
                                  const std::vector<std::string> *_speciesNames, const int *numSpecies,
                                  const double *_reactantStoich, const double *_productStoich);

  MFEM_HOST_DEVICE virtual ~RadiativeDecay();

  MFEM_HOST_DEVICE virtual double computeRateCoefficient(const double &T_h, const double &T_e, const int &dofindex,
                                                         [[maybe_unused]] const bool isElectronInvolved = false,
                                                         const double *nsp = 0);

  MFEM_HOST_DEVICE void GetEinsteinACoefficient(const double &T_h, const double &T_e, double *effAcoef);

  MFEM_HOST_DEVICE void GetNumDensityOfInteralLevels(const int NumOfInteral_lvl, const double &n_sp, const double &T_e,
                                                     double *n_sp_internal);

  MFEM_HOST_DEVICE double escapeFactCalc(const double &n_i, const double &E_j, const double &E_i, const double &g_j,
                                         const double &g_i, const double &A_ji, const double &T_g);
};

#endif  // REACTION_HPP_
