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
#ifndef GSLIB_INTERPOLATOR_HPP_
#define GSLIB_INTERPOLATOR_HPP_

#include <tps_config.h>

#include "tps_mfem_wrap.hpp"

/** @file
 *
 * Provides utilities for interpolating fields to arbitrary * sets of
 * points.  These utilities make use of the * mfem::FindPointsGSLIB
 * class to do the actual work.
 */

class InterpolatorBase {
 protected:
  // Spatial dimensions
  int dim_;

  // Coordinates of points to interpolate to
  mfem::Vector xyz_;

  // Interpolated values
  mfem::Vector soln_;

#ifdef HAVE_GSLIB
  // mfem interpolator
  mfem::FindPointsGSLIB *finder_;
#endif

 public:
  /** Construct interpolation object.
   *
   * Nothing is allocated or initialized.  You must call
   * initializeFinder and setInterpolationPoints prior to interpolate.
   */
  InterpolatorBase();

  /** Destructor.  Frees the FindPointsGSLIB object */
  virtual ~InterpolatorBase();

  /** Initialize the internal FindPointsGSLIB object */
  void initializeFinder(mfem::ParMesh *mesh);

  /** Sets the interpolation points from user input */
  void setInterpolationPoints(mfem::Vector xyz);

  virtual void setInterpolationPoints();

  /** Interpolation the input ParGridFunction to the desired points.
   *
   * Must call initializeFinder and setInterpolationPoints prior.
   */
  void interpolate(mfem::ParGridFunction *u);

  /** Write the data to an ascii file */
  virtual void writeAscii(std::string oname, bool rank0 = true) const;
};

class PlaneInterpolator : public InterpolatorBase {
 private:
  // point and normal that define the plane
  mfem::Vector point_;
  mfem::Vector normal_;

  // corners of bounding box to intersect with plane
  mfem::Vector bb0_;
  mfem::Vector bb1_;

  // number of points per side
  int n_;

 public:
  PlaneInterpolator();
  PlaneInterpolator(mfem::Vector point, mfem::Vector normal, mfem::Vector bb0, mfem::Vector bb1, int n);
  virtual ~PlaneInterpolator();

  void setInterpolationPoints() final;
  void writeAscii(std::string oname, bool rank0 = true) const final;
};

class LineInterpolator : public InterpolatorBase {
 private:
  //  Endpoints of line
  mfem::Vector start_;
  mfem::Vector end_;

  //  Number of points on line
  int n_;

 public:
  LineInterpolator();
  LineInterpolator(mfem::Vector start, mfem::Vector end, int n);
  virtual ~LineInterpolator();

  /** Sets evenly spaced interpolation points
   * 
   * point_i = start_ + (end_ - start_)*i/(n_ - 1)
   */
  void setInterpolationPoints() final;
  void writeAscii(std::string oname, bool rank0 = true) const final;
};

#endif  // GSLIB_INTERPOLATOR_HPP_
