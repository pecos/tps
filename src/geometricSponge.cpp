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

#include "geometricSponge.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "sponge_base.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

GeometricSponge::GeometricSponge(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps)
    : tpsP_(tps), loMach_opts_(loMach_opts), pmesh_(pmesh) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  tpsP_->getInput("spongeMultiplier/uniform", uniform.isEnabled, false);
  tpsP_->getInput("spongeMultiplier/plane", plane.isEnabled, false);
  tpsP_->getInput("spongeMultiplier/cylinder", cylinder.isEnabled, false);
  tpsP_->getInput("spongeMultiplier/annulus", annulus.isEnabled, false);

  if (uniform.isEnabled) {
    tpsP_->getRequiredInput("spongeMultiplier/uniformMult", uniform.mult);
  }

  if (plane.isEnabled) {
    tpsP_->getRequiredVecElem("spongeMultiplier/planeNormal", plane.normal[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplier/planeNormal", plane.normal[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplier/planeNormal", plane.normal[2], 2);
    tpsP_->getRequiredVecElem("spongeMultiplier/planePoint", plane.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplier/planePoint", plane.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplier/planePoint", plane.point[2], 2);
    tpsP_->getRequiredInput("spongeMultiplier/planeWidth", plane.width);
    tpsP_->getRequiredInput("spongeMultiplier/planeMult", plane.mult);
  }

  if (cylinder.isEnabled) {
    tpsP_->getRequiredVecElem("spongeMultiplier/cylinderPoint", cylinder.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplier/cylinderPoint", cylinder.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplier/cylinderPoint", cylinder.point[2], 2);
    tpsP_->getInput("spongeMultiplier/cylinderRadiusX", cylinder.radiusX, -1.0);
    tpsP_->getInput("spongeMultiplier/cylinderRadiusY", cylinder.radiusY, -1.0);
    tpsP_->getInput("spongeMultiplier/cylinderRadiusZ", cylinder.radiusZ, -1.0);
    tpsP_->getInput("spongeMultiplier/cylinderWidth", cylinder.width, 1.0e-8);
    tpsP_->getRequiredInput("spongeMultiplier/cylinderMult", cylinder.mult);
  }

  if (annulus.isEnabled) {
    tpsP_->getRequiredVecElem("spongeMultiplier/annulusPoint", annulus.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplier/annulusPoint", annulus.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplier/annulusPoint", annulus.point[2], 2);
    tpsP_->getInput("spongeMultiplier/annulusRadiusX", annulus.radiusX, -1.0);
    tpsP_->getInput("spongeMultiplier/annulusRadiusY", annulus.radiusY, -1.0);
    tpsP_->getInput("spongeMultiplier/annulusRadiusZ", annulus.radiusZ, -1.0);
    tpsP_->getInput("spongeMultiplier/annulusWidth", annulus.width, 1.0e-8);
    tpsP_->getRequiredInput("spongeMultiplier/annulusMult", annulus.mult);
  }

  if (!uniform.isEnabled && !plane.isEnabled && !cylinder.isEnabled && !annulus.isEnabled) {
    if (rank0_) {
      std::cout << "No sponge enabled.  Supported types: unifrom, plane, cylinder, and annulus" << endl;
    }
  }
}

GeometricSponge::~GeometricSponge() {
  delete sfec_;
  delete sfes_;
  delete vfec_;
  delete vfes_;
}

void GeometricSponge::initializeSelf() {
  rank_ = pmesh_->GetMyRank();
  rank0_ = false;
  if (rank_ == 0) {
    rank0_ = true;
  }
  dim_ = pmesh_->Dimension();

  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing Sponge.\n");

  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  vfec_ = new H1_FECollection(order_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  Sdof_ = sfes_->GetNDofs();

  mult_gf_.SetSpace(sfes_);

  // exports
  toFlow_interface_.visc_multiplier = &mult_gf_;
  toThermoChem_interface_.diff_multiplier = &mult_gf_;
  toTurbModel_interface_.diff_multiplier = &mult_gf_;
}

void GeometricSponge::initializeViz(ParaViewDataCollection &pvdc) { pvdc.RegisterField("sponge", &mult_gf_); }

void GeometricSponge::setup() {
  ParGridFunction coordsDof(vfes_);
  pmesh_->GetNodes(coordsDof);

  double *data = mult_gf_.HostReadWrite();
  const double *hcoords = coordsDof.HostRead();

  for (int i = 0; i < Sdof_; i++) {
    double coords[3];
    for (int d = 0; d < dim_; d++) {
      coords[d] = hcoords[i + d * Sdof_];
    }

    double wgt = 1.0;
    if (uniform.isEnabled) spongeUniform(wgt);
    if (plane.isEnabled) spongePlane(coords, wgt);
    if (cylinder.isEnabled) spongeCylinder(coords, wgt);
    if (annulus.isEnabled) spongeAnnulus(coords, wgt);
    data[i] = wgt;
  }
}

void GeometricSponge::spongeUniform(double &wgt) {
  double factor, wgt_lcl;
  factor = uniform.mult;
  factor = max(factor, 1.0);
  wgt_lcl = factor;
  wgt = max(wgt, wgt_lcl);
}

void GeometricSponge::spongePlane(double *x, double &wgt) {
  double normal[3];
  double point[3];
  double s[3];
  double factor, width, dist, wgt0;
  double wgt_lcl;

  for (int d = 0; d < dim_; d++) normal[d] = plane.normal[d];
  for (int d = 0; d < dim_; d++) point[d] = plane.point[d];
  width = plane.width;
  factor = plane.mult;

  // get settings
  factor = max(factor, 1.0);

  // distance from plane
  dist = 0.;
  for (int d = 0; d < dim_; d++) s[d] = (x[d] - point[d]);
  for (int d = 0; d < dim_; d++) dist += s[d] * normal[d];

  // weight
  wgt0 = 0.5 * (tanh(0.0 / width - 2.0) + 1.0);
  wgt_lcl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
  wgt_lcl = (wgt_lcl - wgt0) / (1.0 - wgt0);
  wgt_lcl = std::max(wgt_lcl, 0.0);
  wgt_lcl *= (factor - 1.0);
  wgt_lcl += 1.0;
  wgt = max(wgt, wgt_lcl);
}

// TODO(swh): add suppport for arbitrary orientation of cylinder axis
void GeometricSponge::spongeCylinder(double *xGlobal, double &wgt) {
  // C++ standard frowns on use of non-compile time arrays, e.g.
  // point[dim_], hence the requirement for hard-codes here and
  // elsewhere in this class
  double point[3];
  double factor, width, dist, wgt0;
  double wgtCyl;

  // cylinder orientation and size
  double cylX = cylinder.radiusX;
  double cylY = cylinder.radiusY;
  double cylZ = cylinder.radiusZ;
  width = cylinder.width;

  // offset origin of cylinder
  double x[3];  // dim_]; not able to use variable length arrays
  for (int i = 0; i < dim_; i++) {
    point[i] = cylinder.point[i];
  }
  for (int i = 0; i < dim_; i++) {
    x[i] = xGlobal[i] - point[i];
  }

  factor = cylinder.mult;
  wgt0 = 0.5 * (tanh(0.0 / width - 2.0) + 1.0);

  // cylinder aligned with one axis
  if (cylX > 0.0) {
    dist = x[1] * x[1] + x[2] * x[2];
    dist = std::sqrt(dist);
    dist = dist - cylX;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);

  } else if (cylY > 0.0) {
    dist = x[0] * x[0] + x[2] * x[2];
    dist = std::sqrt(dist);
    dist = dist - cylY;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);

  } else if (cylZ > 0.0) {
    dist = x[0] * x[0] + x[1] * x[1];
    dist = std::sqrt(dist);
    dist = dist - cylZ;
    wgtCyl = 0.5 * (tanh(dist / width - 2.0) + 1.0);
    wgtCyl = (wgtCyl - wgt0) * 1.0 / (1.0 - wgt0);
    wgtCyl = std::max(wgtCyl, 0.0);
    wgtCyl *= (factor - 1.0);
    wgtCyl += 1.0;
    wgt = std::max(wgt, wgtCyl);
  }
}

// TODO(swh): NOT GENERAL, only for annulus aligned with y
void GeometricSponge::spongeAnnulus(double *xGlobal, double &wgt) {
  double wgtAnn;
  double s[3];
  double x[3];

  // TODO(swh): offset origin of annulus
  for (int i = 0; i < dim_; i++) {
    x[i] = xGlobal[i];  // - point[i];
  }

  double centerAnnulus[3];
  for (int d = 0; d < dim_; d++) {
    centerAnnulus[d] = annulus.point[d];
  }
  double rad1 = annulus.radiusY;
  double rad2 = annulus.width;
  double factor = annulus.mult;
  double wgt0 = 0.5 * (tanh(0.0 / rad2 - 2.0) + 1.0);

  // distance to center
  double dist1 = x[0] * x[0] + x[2] * x[2];
  dist1 = std::sqrt(dist1);

  // coordinates of nearest point on annulus center ring
  for (int d = 0; d < dim_; d++) s[d] = (rad1 / dist1) * x[d];
  s[1] = centerAnnulus[1];

  // distance to ring
  for (int d = 0; d < dim_; d++) s[d] = x[d] - s[d];
  double dist2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
  dist2 = std::sqrt(dist2);

  // if (x[1] <= centerAnnulus[1]+rad2) {
  //   std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << endl;
  // }

  // if (dist2 <= rad2) {
  //   std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << " fA: " << factorA << endl;
  // }

  // sponge weight
  wgtAnn = 0.5 * (tanh(10.0 * (1.0 - dist2 / rad2)) + 1.0);
  wgtAnn = (wgtAnn - wgt0) * 1.0 / (1.0 - wgt0);
  wgtAnn = std::max(wgtAnn, 0.0);
  wgtAnn *= (factor - 1.0);
  wgtAnn += 1.0;
  wgt = std::max(wgt, wgtAnn);
}

void GeometricSponge::step() {
  // empty for now, use for any updates/modifications
  // during simulation e.g. ramping down with time
}
