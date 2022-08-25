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
#ifndef TABLE_HPP_
#define TABLE_HPP_

#include <math.h>

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"

using namespace std;

class TableInterpolator {
protected:
  double xdata_[gpudata::MAXTABLE];
  double fdata_[gpudata::MAXTABLE];
  int Ndata_;

  bool xLogScale_;
  bool fLogScale_;

public:
  MFEM_HOST_DEVICE TableInterpolator(const int &Ndata, const double *xdata, const double *fdata, const bool &xLogScale, const bool &fLogScale);

  MFEM_HOST_DEVICE virtual ~TableInterpolator() {}

  MFEM_HOST_DEVICE int findInterval(const double &xEval);

  MFEM_HOST_DEVICE virtual double eval(const double &xEval) {
    printf("TableInterpolator not initialized!");
    return nan("");
  }
};

//////////////////////////////////////////////////////
//////// Linear interpolation
//////////////////////////////////////////////////////

class LinearTable : public TableInterpolator {
private:
  // f[j](x) = a + b * x
  double a_[gpudata::MAXTABLE];
  double b_[gpudata::MAXTABLE];

public:
  MFEM_HOST_DEVICE LinearTable(const TableInput &input);

  MFEM_HOST_DEVICE virtual ~LinearTable() {}

  MFEM_HOST_DEVICE virtual double eval(const double &xEval);
};

#endif  // TABLE_HPP_
