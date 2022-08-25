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

#include "table.hpp"

MFEM_HOST_DEVICE TableInterpolator::TableInterpolator(const int &Ndata, const double *xdata, const double *fdata, const bool &xLogScale, const bool &fLogScale)
    : Ndata_(Ndata), xLogScale_(xLogScale), fLogScale_(fLogScale) {
  assert((xdata != NULL) && (fdata != NULL));
  for (int k = 0; k < Ndata_; k++) {
    xdata_[k] = xdata[k];
    fdata_[k] = fdata[k];
  }
}

// Find the data interval where the input value lies within.
// The algorithm is a copy version of std::upper_bound, which is similar to binary search.
// This has O(log_2(Ndata)) complexity.
MFEM_HOST_DEVICE int TableInterpolator::findInterval(const double &xEval) {
  int count = Ndata_;
  int first = 0;
  int it, step;

  // Find the first index which value is larger than xEval.
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (xEval > xdata_[it]) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  // if xEval is outside the range, first has either 0 or Ndata_. Limit the value.
  first = max(1, min(Ndata_-1, first));
  // We want the left index of the interval.
  return first - 1;
}

//////////////////////////////////////////////////////
//////// Linear interpolation
//////////////////////////////////////////////////////

MFEM_HOST_DEVICE LinearTable::LinearTable(const TableInput &input) : TableInterpolator(input.Ndata, input.xdata, input.fdata, input.xLogScale, input.fLogScale) {
  for (int k = 0; k < Ndata_-1; k++) {
    a_[k] = (fLogScale_) ? log(fdata_[k]) : fdata_[k];
    double df = (fLogScale_) ? (log(fdata_[k+1]) - log(fdata_[k])) : (fdata_[k+1] - fdata_[k]);
    b_[k] = (xLogScale_) ? df / (log(xdata_[k+1]) - log(xdata_[k])) : df / (xdata_[k+1] - xdata_[k]);
    a_[k] -= (xLogScale_) ? b_[k] * log(xdata_[k]) : b_[k] * xdata_[k];
  }
}

MFEM_HOST_DEVICE double LinearTable::eval(const double &xEval) {
  int index = findInterval(xEval);
  double xt = (xLogScale_) ? log(xEval) : xEval;
  double ft = a_[index] + b_[index] * xt;
  if (fLogScale_) ft = exp(ft);

  return ft;
}
