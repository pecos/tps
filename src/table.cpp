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

#include <stdio.h>

using namespace std;

MFEM_HOST_DEVICE TableInterpolator::TableInterpolator(const int &Ndata, const double *xdata, const double *fdata,
                                                      const bool &xLogScale, const bool &fLogScale)
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
  first = max(1, min(Ndata_ - 1, first));
  // We want the left index of the interval.
  return first - 1;
}

//////////////////////////////////////////////////////
//////// Linear interpolation
//////////////////////////////////////////////////////

MFEM_HOST_DEVICE LinearTable::LinearTable(const TableInput &input)
    : TableInterpolator(input.Ndata, input.xdata, input.fdata, input.xLogScale, input.fLogScale) {
  assert(input.order == 1);
  for (int k = 0; k < Ndata_ - 1; k++) {
    a_[k] = (fLogScale_) ? log(fdata_[k]) : fdata_[k];
    double df = (fLogScale_) ? (log(fdata_[k + 1]) - log(fdata_[k])) : (fdata_[k + 1] - fdata_[k]);
    b_[k] = (xLogScale_) ? df / (log(xdata_[k + 1]) - log(xdata_[k])) : df / (xdata_[k + 1] - xdata_[k]);
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

/** \brief Default ctor
 *
 * Sizes set to zero.  Nothing allocated.
 */
TableInterpolator2D::TableInterpolator2D() : nx_(0), ny_(0) {}

/** \brief Constructs 2-D interpolation base class
 *
 * Allocates memory for required data arrays based on input sizes.
 * Memory is uninitialized.
 */
TableInterpolator2D::TableInterpolator2D(unsigned int nx, unsigned int ny) : nx_(nx), ny_(ny) {
  xdata_ = new double[nx_];
  ydata_ = new double[ny_];
  fdata_ = new double[nx_ * ny_];
}

/** \brief 2-D interpolation base class destructor
 *
 * Frees memory allocated by ctor
 */
TableInterpolator2D::~TableInterpolator2D() {
  delete[] xdata_;
  delete[] ydata_;
  delete[] fdata_;
}

/**
 * Resets sizes and re-allocates data.  Use with care.  Any
 * pre-existing data is destroyed.  Resuling allocated arrays are not
 * initialized.
 */
void TableInterpolator2D::resize(unsigned int nx, unsigned int ny) {
  if (nx_ > 0) {
    delete[] xdata_;
  }
  if (ny_ > 0) {
    delete[] ydata_;
  }
  if (nx_ > 0 && ny_ > 0) {
    delete[] fdata_;
  }

  nx_ = nx;
  ny_ = ny;
  xdata_ = new double[nx_];
  ydata_ = new double[ny_];
  fdata_ = new double[nx_ * ny_];
}

#ifdef HAVE_GSL

/** \brief Constructs GSL 2-D interpolation object
 *
 * Reads data from plato_file
 */
GslTableInterpolator2D::GslTableInterpolator2D(std::string plato_file, int xcol, int ycol, int fcol, int ncol)
    : TableInterpolator2D(), itype_(gsl_interp2d_bilinear) {
  // assert that incoming sizes make sense
  assert(xcol >= 0 && xcol < ncol);
  assert(ycol >= 0 && ycol < ncol);
  assert(fcol >= 0 && fcol < ncol);

  // open plato file
  FILE *table_input_file;
  table_input_file = fopen(plato_file.c_str(), "r");
  if (!table_input_file) {
    std::cout << "Unable to open " << plato_file << std::endl;
    assert(false);
  }

  // read size info from plato file
  unsigned int nxt, nyt;
  int ierr = fscanf(table_input_file, "%u %u\n", &nxt, &nyt);
  assert(ierr == 2);

  // set table sizes and allocate data arrays
  resize(nxt, nyt);

  // read data
  char stmp;
  double *ftmp = new double[ncol];
  for (unsigned int jj = 0; jj < ny_; ++jj) {
    for (unsigned int ii = 0; ii < nx_; ++ii) {
      ierr = 0;
      for (int c = 0; c < ncol; ++c) {
        ierr += fscanf(table_input_file, "%lf", &ftmp[c]);
      }
      assert(ierr == ncol);

      // NB: We assume that ftmp[xcol] is same for each jj
      xdata_[ii] = ftmp[xcol];
      ydata_[jj] = ftmp[ycol];
      fdata_[jj * nx_ + ii] = ftmp[fcol];
    }
  }

  delete[] ftmp;

  // close file
  fclose(table_input_file);

  // set up gsl objects
  spline_ = gsl_spline2d_alloc(itype_, nx_, ny_);

  xacc_ = gsl_interp_accel_alloc();
  yacc_ = gsl_interp_accel_alloc();

  gsl_spline2d_init(spline_, xdata_, ydata_, fdata_, nx_, ny_);
}

/** \brief Constructs GSL 2-D interpolation object
 *
 * Initializes data using input pointers.  Allocates and initializes
 * all required GSL objects.
 */
GslTableInterpolator2D::GslTableInterpolator2D(unsigned int nx, unsigned int ny, const double *xdata,
                                               const double *ydata, const double *fdata)
    : TableInterpolator2D(nx, ny), itype_(gsl_interp2d_bilinear) {
  for (unsigned int i = 0; i < nx_; ++i) {
    xdata_[i] = xdata[i];
  }
  for (unsigned int i = 0; i < ny_; ++i) {
    ydata_[i] = ydata[i];
  }
  for (unsigned int i = 0; i < nx_ * ny_; ++i) {
    fdata_[i] = fdata[i];
  }

  spline_ = gsl_spline2d_alloc(itype_, nx_, ny_);

  xacc_ = gsl_interp_accel_alloc();
  yacc_ = gsl_interp_accel_alloc();

  gsl_spline2d_init(spline_, xdata_, ydata_, fdata_, nx_, ny_);
}

/** \brief GSL 2-D interpolation destructor
 *
 * Frees GSL objects allocated by constructor
 */
GslTableInterpolator2D::~GslTableInterpolator2D() {
  gsl_interp_accel_free(yacc_);
  gsl_interp_accel_free(xacc_);
  gsl_spline2d_free(spline_);
}

#endif  // HAVE_GSL
