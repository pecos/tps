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
                                                          const bool isElectronInvolved) {
  double temp = (isElectronInvolved) ? T_e : T_h;

  return A_ * pow(temp, b_) * exp(-E_ / UNIVERSALGASCONSTANT / temp);
}

MFEM_HOST_DEVICE HoffertLien::HoffertLien(const double &A, const double &b, const double &E)
    : Reaction(HOFFERTLIEN), A_(A), b_(b), E_(E) {}

MFEM_HOST_DEVICE double HoffertLien::computeRateCoefficient(const double &T_h, const double &T_e,
                                                            [[maybe_unused]] const int &dofindex,
                                                            const bool isElectronInvolved) {
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
                                                          const bool isElectronInvolved) {
  double temp = (isElectronInvolved) ? T_e : T_h;
  return table_->eval(temp);
}

MFEM_HOST_DEVICE GridFunctionReaction::GridFunctionReaction(int comp)
    : Reaction(GRIDFUNCTION_RXN), data(nullptr), comp(comp), size_(0) {}

MFEM_HOST_DEVICE GridFunctionReaction::~GridFunctionReaction() {}

void GridFunctionReaction::setGridFunctionData(std::shared_ptr<mfem::ParGridFunction> &f) {
  f_ = f;
  size_ = f->FESpace()->GetNDofs();
  assert(comp < f->FESpace()->GetVDim() );
  assert(f->FESpace()->GetOrdering() == mfem::Ordering::byNODES);
#ifdef _GPU_
  data = f_->Read() + comp * size_;
#else
  data = f_->HostRead() + comp * size_;
#endif
}

MFEM_HOST_DEVICE double GridFunctionReaction::computeRateCoefficient([[maybe_unused]] const double &T_h,
                                                                     [[maybe_unused]] const double &T_e,
                                                                     const int &dofindex,
                                                                     [[maybe_unused]] const bool isElectronInvolved) {
  if (data) {
    assert(dofindex < size_);
    return data[dofindex];
  }
  else 
    return 0.;
}
