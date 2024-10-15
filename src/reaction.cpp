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

#include "reaction.hpp"

using namespace mfem;
using namespace std;

MFEM_HOST_DEVICE Arrhenius::Arrhenius(const double &A, const double &b, const double &E)
    : Reaction(ARRHENIUS), A_(A), b_(b), E_(E) {}

MFEM_HOST_DEVICE double Arrhenius::computeRateCoefficient(const double &T_h, const double &T_e,
                                                          [[maybe_unused]] const int &dofindex,
                                                          const bool isElectronInvolved,
                                                          [[maybe_unused]] const double *nsp) {
  double temp = (isElectronInvolved) ? T_e : T_h;

  return A_ * pow(temp, b_) * exp(-E_ / UNIVERSALGASCONSTANT / temp);
}

MFEM_HOST_DEVICE HoffertLien::HoffertLien(const double &A, const double &b, const double &E)
    : Reaction(HOFFERTLIEN), A_(A), b_(b), E_(E) {}

MFEM_HOST_DEVICE double HoffertLien::computeRateCoefficient(const double &T_h, const double &T_e,
                                                            [[maybe_unused]] const int &dofindex,
                                                            const bool isElectronInvolved,
                                                            [[maybe_unused]] const double *nsp) {
  double temp = (isElectronInvolved) ? T_e : T_h;
  double tempFactor = E_ / BOLTZMANNCONSTANT / temp;

  return A_ * pow(temp, b_) * (tempFactor + 2.0) * exp(-tempFactor);
}

MFEM_HOST_DEVICE Tabulated::Tabulated(const TableInput &input) : Reaction(TABULATED_RXN) {
  switch (input.order) {
    case 1: {
      table_ = new LinearTable(input);
    } break;
    default: {
      printf("Given interpolation order is not supported for TableInterpolator!");
      assert(false);
    } break;
  }
}

MFEM_HOST_DEVICE Tabulated::~Tabulated() { delete table_; }

MFEM_HOST_DEVICE double Tabulated::computeRateCoefficient(const double &T_h, const double &T_e,
                                                          [[maybe_unused]] const int &dofindex,
                                                          const bool isElectronInvolved,
                                                          [[maybe_unused]] const double *nsp) {
  double temp = (isElectronInvolved) ? T_e : T_h;
  return table_->eval(temp);
}

MFEM_HOST_DEVICE GridFunctionReaction::GridFunctionReaction(int comp)
    : Reaction(GRIDFUNCTION_RXN), data_(nullptr), comp_(comp), size_(0) {}

MFEM_HOST_DEVICE GridFunctionReaction::~GridFunctionReaction() {}

MFEM_HOST_DEVICE void GridFunctionReaction::setData(const double *data, int size) {
  data_ = data + comp_ * size_;
  size_ = size;
}

void GridFunctionReaction::setGridFunction(const mfem::GridFunction &f) {
  size_ = f.FESpace()->GetNDofs();
  assert(comp_ < f.FESpace()->GetVDim());
  assert(f.FESpace()->GetOrdering() == mfem::Ordering::byNODES);
#if defined(_CUDA_) || defined(_HIP_)
  data_ = f.Read() + comp_ * size_;
#else
  data_ = f.HostRead() + comp_ * size_;
#endif
}

MFEM_HOST_DEVICE double GridFunctionReaction::computeRateCoefficient([[maybe_unused]] const double &T_h,
                                                                     [[maybe_unused]] const double &T_e,
                                                                     const int &dofindex,
                                                                     [[maybe_unused]] const bool isElectronInvolved,
                                                                     [[maybe_unused]] const double *nsp) {
  if (data_) {
    assert(dofindex < size_);
    return data_[dofindex];
  } else {
    return 0.;
  }
}

#ifndef _GPU_
// Radiative decay portion: extracted from commit (de27f14)
MFEM_HOST_DEVICE RadiativeDecay::RadiativeDecay(const double _R, const std::map<std::string, int> *_speciesMapping,
                                                const std::vector<std::string> *_speciesNames, const int *numSpecies,
                                                const double *_reactantStoich, const double *_productStoich)
    : Reaction(RADIATIVE_DECAY), Rcyl(_R) {
  rank0_ = Mpi::Root();
  Lcyl = 2.0 * Rcyl;
  numSpecies_ = *numSpecies;

  int numOfReactants = 0, numOfProducts = 0;
  for (int sp = 0; sp < numSpecies_; sp++) {
    numOfReactants = numOfReactants + _reactantStoich[sp];
    numOfProducts = numOfProducts + _productStoich[sp];
  }
  assert(numOfReactants == 1);
  assert(numOfProducts == 1);

  for (int sp = 0; sp < numSpecies_; sp++) {
    if (_reactantStoich[sp] > 0) upper_sp_name = _speciesNames->at(sp);
    if (_productStoich[sp] > 0) lower_sp_name = _speciesNames->at(sp);
  }

  iAr_u = _speciesMapping->at(upper_sp_name);
  iAr_l = _speciesMapping->at(lower_sp_name);

  bool flag =
      ((upper_sp_name == "Ar_r" && lower_sp_name == "Ar") || (upper_sp_name == "Ar_p" && lower_sp_name == "Ar_r") ||
       (upper_sp_name == "Ar_p" && lower_sp_name == "Ar_m") || upper_sp_name == "Ar");
  assert(flag);

  // Allowed radiative transitions
  if (upper_sp_name == "Ar_r" || upper_sp_name == "Ar") {
    NumOfInteral_lvl_u = E_lvl_r.size();
    E_lvl_u = &E_lvl_r;
    g_lvl_u = &g_lvl_r;
    n_sp_lvl_u.resize(E_lvl_r.size());
    E_lvl_l = &E_lvl_g;
    g_lvl_l = &g_lvl_g;
    n_sp_lvl_l.resize(E_lvl_g.size());
    Aji = &Aji_r_g;
  } else if (upper_sp_name == "Ar_p" && lower_sp_name == "Ar_m") {
    NumOfInteral_lvl_u = E_lvl_4p.size();
    E_lvl_u = &E_lvl_4p;
    g_lvl_u = &g_lvl_4p;
    n_sp_lvl_u.resize(E_lvl_4p.size());
    E_lvl_l = &E_lvl_m;
    g_lvl_l = &g_lvl_m;
    n_sp_lvl_l.resize(E_lvl_m.size());
    Aji = &Aji_4p_m;
  } else if (upper_sp_name == "Ar_p" && lower_sp_name == "Ar_r") {
    NumOfInteral_lvl_u = E_lvl_4p.size();
    E_lvl_u = &E_lvl_4p;
    g_lvl_u = &g_lvl_4p;
    n_sp_lvl_u.resize(E_lvl_4p.size());
    E_lvl_l = &E_lvl_r;
    g_lvl_l = &g_lvl_r;
    n_sp_lvl_l.resize(E_lvl_r.size());
    Aji = &Aji_4p_r;

  } else {
    if (rank0_) {
      printf("Specified radiative reaction no supported!");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    assert(false);
  }

  effAcoef_lvl.resize(NumOfInteral_lvl_u);
  std::fill(effAcoef_lvl.begin(), effAcoef_lvl.end(), 0.0);
}

MFEM_HOST_DEVICE RadiativeDecay::~RadiativeDecay() {}

MFEM_HOST_DEVICE double RadiativeDecay::computeRateCoefficient(const double &T_h, const double &T_e,
                                                               const int &dofindex,
                                                               [[maybe_unused]] const bool isElectronInvolved,
                                                               const double *nsp) {
  double n_sp_u = nsp[iAr_u];  // Find the index corresponding to the correct species
  double n_sp_l = nsp[iAr_l];

  GetNumDensityOfInteralLevels(n_sp_lvl_u.size(), n_sp_u, T_e, &n_sp_lvl_u[0]);
  GetNumDensityOfInteralLevels(n_sp_lvl_l.size(), n_sp_l, T_e, &n_sp_lvl_l[0]);

  GetEinsteinACoefficient(T_h, T_e, &effAcoef_lvl[0]);

  // The effective Einstein A Coefficient of a lumped level is number-density
  // weighted over the internal levels.
  double effAcoef = 0.0;
  for (int i_lvl = 0; i_lvl < NumOfInteral_lvl_u; i_lvl++) {
    effAcoef += n_sp_lvl_u[i_lvl] * effAcoef_lvl[i_lvl];
  }
  effAcoef = effAcoef / (n_sp_u + small);

  effAcoef = max(min(effAcoef, 1.0), 0.0);
  return effAcoef;
}

MFEM_HOST_DEVICE void RadiativeDecay::GetNumDensityOfInteralLevels(const int NumOfInteral_lvl, const double &n_sp,
                                                                   const double &T_e, double *n_sp_internal) {
  // n_sp in [mol/m^3] is the number concentration of a lumped state/level.
  // Here, we evaluate the number concentration of the internal levels assuming that they
  // are Boltzmann distributed.
  // E_lvl_u in [J/mol]
  // T_e in [K] is the Electron / Excitation Temperature
  // Should we use T_h instead?

  double Q_n = 0.0;  // partition function of internal levels
  for (int i_lvl = 0; i_lvl < NumOfInteral_lvl; i_lvl++) {
    Q_n += (*g_lvl_u)[i_lvl] *
           exp(-(*E_lvl_u)[i_lvl] / UNIVERSALGASCONSTANT / (T_e + small));  // E0 in J/mol, so e/kT = E0/NA/RT
  }

  for (int i_lvl = 0; i_lvl < NumOfInteral_lvl; i_lvl++) {
    n_sp_internal[i_lvl] = (*g_lvl_u)[i_lvl] * exp(-(*E_lvl_u)[i_lvl] / UNIVERSALGASCONSTANT / (T_e + small)) * n_sp /
                           Q_n;  // E0 in J/mol, so e/kT = E0/NA/RT
  }
}

MFEM_HOST_DEVICE void RadiativeDecay::GetEinsteinACoefficient(const double &T_h, const double &T_e, double *effAcoef) {
  for (int i_lvl = 0; i_lvl < NumOfInteral_lvl_u; i_lvl++) {
    for (int itrans = 0; itrans < int((*Aji)[i_lvl].size()); itrans++) {
      double Acoef = (*Aji)[i_lvl][itrans];
      double eta = escapeFactCalc(n_sp_lvl_l[itrans], (*E_lvl_u)[i_lvl], (*E_lvl_l)[itrans], (*g_lvl_u)[i_lvl],
                                  (*g_lvl_l)[itrans], Acoef, T_h);
      effAcoef[i_lvl] = effAcoef[i_lvl] + eta * Acoef;
    }
  }
}

MFEM_HOST_DEVICE double RadiativeDecay::escapeFactCalc(const double &n_i, const double &E_j, const double &E_i,
                                                       const double &g_j, const double &g_i, const double &A_ji,
                                                       const double &T_g) {
  //  Calculations for escape factor
  // i -> lower level
  // j -> upper level
  // Ar(j) -> Ar(i) + hv
  // R =  n_j * A_ji * eta

  //  n_i in [mol/m^3] is the number density of the upper level
  //  E_i in [J/mol] is the energy of the upper level
  //  g_i in [] is the degeneracy of the upper level
  //  A_ji in [1/s] is the Einstein A coeficient of transition j->i

  //  T_g in [K] is the Temperature

  double eta = 1.0;

  // wavelength of transition
  double lambda_0 = PLANCKCONSTANT * SPEEDOFLIGHT / ((E_j - E_i) / AVOGADRONUMBER);

  // absorption coefficient at line center, for Doppler absorption
  double k0 = pow(lambda_0, 3) * (n_i * AVOGADRONUMBER) * g_j * A_ji * sqrt(M_Ar) /
              (8.0 * PI * g_i * sqrt(2.0 * BOLTZMANNCONSTANT * PI * T_g));

  double q0 = Rcyl;
  double Lq = Lcyl / (2 * q0);

  /// compute escape factor

  // Mewe (1967)
  // eta = ( 2.0 - exp(-1e-3 * k0 * Rcyl) ) / (1.0 + k0*Rcyl) ;

  if ((k0 * (Lcyl / 2) > 1) && (k0 * q0 > 1)) {
    // Chai & Kwon Doppler lineshape
    eta = (2.0 / (sqrt(PI * log(k0 * Lcyl / 2.0)) * k0 * Lcyl) / (2.0 * pow(Lq, 2) + 2.0) +
           1.0 / (sqrt(PI * log(k0 * q0)) * k0 * 2.0 * q0) * (Lq / (pow(Lq, 2) + 1) + atan(Lq)));

    // Iordanova/Holstein
    // eta = 1.6 / (k0*R * pow((PI * log(k0*R)),2) ) ;

    // Golubovskii et al. Lorentz lineshape
    // eta = (1.0 / ( sqrt(PI*k0*Lcyl/2) ) * (2.0/3.0 - 2.0 * pow(Lq,1.5) / (3.0 * pow((pow(Lq,2) + 1),0.75)) )
    //  + 1.0/(2.0*sqrt(PI*k0*q0))* 4.0*Lq*sqrt(Lq) / (3.0 * pow((pow(Lq,2) + 1),0.75)));

  } else {
    eta = 1.0;
  }

  eta = min(eta, 1.0);

  return eta;
}

#else
// Radiative decay portion: extracted from commit (de27f14)
MFEM_HOST_DEVICE RadiativeDecay::RadiativeDecay(const double _R, const std::map<std::string, int> *_speciesMapping,
                                                const std::vector<std::string> *_speciesNames, const int *numSpecies,
                                                const double *_reactantStoich, const double *_productStoich)
    : Reaction(RADIATIVE_DECAY), Rcyl(_R) {
  rank0_ = Mpi::Root();

  if (rank0_) {
    printf("Radiative decay not supported on GPU yet!");
  }
}

MFEM_HOST_DEVICE RadiativeDecay::~RadiativeDecay() {}

MFEM_HOST_DEVICE double RadiativeDecay::computeRateCoefficient(const double &T_h, const double &T_e,
                                                               const int &dofindex,
                                                               [[maybe_unused]] const bool isElectronInvolved,
                                                               const double *nsp) {
  double effAcoef = 0.0;
  return effAcoef;
}

MFEM_HOST_DEVICE void RadiativeDecay::GetNumDensityOfInteralLevels(const int NumOfInteral_lvl, const double &n_sp,
                                                                   const double &T_e, double *n_sp_internal) {}

MFEM_HOST_DEVICE void RadiativeDecay::GetEinsteinACoefficient(const double &T_h, const double &T_e, double *effAcoef) {}

MFEM_HOST_DEVICE double RadiativeDecay::escapeFactCalc(const double &n_i, const double &E_j, const double &E_i,
                                                       const double &g_j, const double &g_i, const double &A_ji,
                                                       const double &T_g) {
  double eta = 1.0;
  return eta;
}
#endif
