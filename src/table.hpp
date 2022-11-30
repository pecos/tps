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
#include <tps_config.h>

#ifdef HAVE_GSL
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_spline2d.h>
#endif  // HAVE_GSL

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"

class TableInterpolator {
 protected:
  double xdata_[gpudata::MAXTABLE];
  double fdata_[gpudata::MAXTABLE];
  int Ndata_;

  bool xLogScale_;
  bool fLogScale_;

 public:
  MFEM_HOST_DEVICE TableInterpolator(const int &Ndata, const double *xdata, const double *fdata, const bool &xLogScale,
                                     const bool &fLogScale);

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

/** \brief 2-D interpolation base class
 *
 * Specifies interface for two-dimensional table look-up (i.e.,
 * interpolation).
 */
class TableInterpolator2D {
 protected:
  double *xdata_;
  double *ydata_;
  double *fdata_;

  unsigned int nx_;
  unsigned int ny_;

 public:
  TableInterpolator2D();
  TableInterpolator2D(unsigned int nx, unsigned int ny);
  virtual ~TableInterpolator2D();

  /// Reset size and re-allocate data arrays
  void resize(unsigned int nx, unsigned int ny);

  /// Interpolate fcn to (x,y) --- must be implemented in derived class
  virtual double eval(const double &x, const double &y) {
    printf("TableInterpolator2D not initialized!");
    return -1.0;
  }
};

#ifdef HAVE_GSL

/** \brief 2-D interpolation using GSL
 *
 * Provided 2-D table look-up using the Gnu Scientific Library.  See
 * https://www.gnu.org/software/gsl/doc/html/interp.html# for GSL
 * documentation.
 */
class GslTableInterpolator2D : public TableInterpolator2D {
 protected:
  const gsl_interp2d_type *itype_;
  gsl_spline2d *spline_;
  gsl_interp_accel *xacc_, *yacc_;

 public:
  GslTableInterpolator2D(std::string plato_file, int xcol, int ycol, int fcol);
  GslTableInterpolator2D(unsigned int nx, unsigned int ny, const double *xdata, const double *ydata,
                         const double *fdata);
  virtual ~GslTableInterpolator2D();

  /// Interpolate function to (x,y)
  virtual double eval(const double &x, const double &y) { return gsl_spline2d_eval(spline_, x, y, xacc_, yacc_); }
};

#endif  // HAVE_GSL

#endif  // TABLE_HPP_
