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

#include "sponge_base.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "utils.hpp"
#include "geometricSponge.hpp"

using namespace mfem;
using namespace mfem::common;

GeometricSponge::GeometricSponge(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps)
    : tpsP_(tps), pmesh_(pmesh), loMach_opts_(loMach_opts) {
  
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  tpsP_->getInput("spongeMultiplierFunction/uniform", uniform.isEnabled, false);
  tpsP_->getInput("spongeMultiplierFunction/plane", plane.isEnabled, false);
  tpsP_->getInput("spongeMultiplierFunction/cylinder", cylinder.isEnabled, false);
  tpsP_->getInput("spongeMultiplierFunction/annulus", annulus.isEnabled, false);    
  
  if (uniform.isEnabled) {
    tpsP_->getRequiredInput("spongeMultiplierFunction/uniformMult", uniform.mult);    
  }

  if (plane.isEnabled) {  
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planeNormal", plane.normal[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planeNormal", plane.normal[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planeNormal", plane.normal[2], 2);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planePoint", plane.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planePoint", plane.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/planePoint", plane.point[2], 2);    
    tpsP_->getRequiredInput("spongeMultiplierFunction/planeWidth", plane.width);
    tpsP_->getRequiredInput("spongeMultiplierFunction/planeMult", plane.mult);
  } 

  if (cylinder.isEnabled) {      
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/cylinderPoint", cylinder.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/cylinderPoint", cylinder.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/cylinderPoint", cylinder.point[2], 2);
    tpsP_->getInput("spongeMultiplierFunction/cylinderRadiusX", cylinder.radiusX, -1.0);
    tpsP_->getInput("spongeMultiplierFunction/cylinderRadiusY", cylinder.radiusY, -1.0);
    tpsP_->getInput("spongeMultiplierFunction/cylinderRadiusZ", cylinder.radiusZ, -1.0);    
    tpsP_->getInput("spongeMultiplierFunction/cylinderWidth", cylinder.width, 1.0e-8);    
    tpsP_->getRequiredInput("spongeMultiplierFunction/cylinderMult", cylinder.mult);    
  }
  
  if (annulus.isEnabled) {      
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/annulusPoint", annulus.point[0], 0);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/annulusPoint", annulus.point[1], 1);
    tpsP_->getRequiredVecElem("spongeMultiplierFunction/annulusPoint", annulus.point[2], 2);
    tpsP_->getInput("spongeMultiplierFunction/annulusRadiusX", annulus.radiusX, -1.0);
    tpsP_->getInput("spongeMultiplierFunction/annulusRadiusY", annulus.radiusY, -1.0);
    tpsP_->getInput("spongeMultiplierFunction/annulusRadiusZ", annulus.radiusZ, -1.0);    
    tpsP_->getInput("spongeMultiplierFunction/annulusWidth", annulus.width, 1.0e-8);    
    tpsP_->getRequiredInput("spongeMultiplierFunction/annulusMult", annulus.mult); 
  }

  // TODO: add checks and error msg for invalid sponge settings
  
}

void GeometricSponge::initializeSelf() {

  rank_ = pmesh_->GetMyRank();
  rank0_ = false;
  if (rank_ == 0) {
    rank0_ = true;
  }
  dim_ = pmesh_->Dimension();
  nvel_ = dim_;

  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing Sponge.\n");

  /*
  if (loMach_opts_->uOrder == -1) {
    // order_ = std::max(config_->solOrder, 1);
    double no;
    no = ceil(((double)order_ * 1.5));
  } else {
    order_ = loMach_opts_->uOrder;
  }
  */

  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  int sfes_truevsize = sfes_->GetTrueVSize();

  mult.SetSize(sfes_truevsize);
  mult_gf.SetSpace(sfes_);

  // exports
  toFlow_interface_.visc_multiplier = &mult_gf;
  toThermoChem_interface_.diff_multiplier = &mult_gf;
}

void GeometricSponge::setup() {

  ParGridFunction coordsDof(sfes_);
  pmesh_->GetNodes(coordsDof);
  
  double *data = mult_gf.HostReadWrite();
  double *hcoords = coordsDof.HostReadWrite();
  double wgt = 1.0;
  for (int n = 0; n < sfes_->GetNDofs(); n++) {    
    double coords[3];
    for (int d = 0; d < dim_; d++) {
      coords[d] = hcoords[n + d * sfes_->GetNDofs()];
    }    
    if(uniform.isEnabled) spongeUniform(wgt);
    if(plane.isEnabled) spongePlane(coords, wgt);
    if(cylinder.isEnabled) spongeCylinder(coords, wgt);              
    if(annulus.isEnabled) spongeAnnulus(coords, wgt);          
    data[n] = wgt; 
  }
  // mult_gf.GetTrueDofs(mult);
  
}

void GeometricSponge::spongeUniform(double &wgt) {
  
  double factor, wgt_lcl;
  factor = uniform.mult;
  factor = max(factor, 1.0);  
  wgt_lcl = factor;
  wgt = max(wgt, wgt_lcl);
  
}

void GeometricSponge::spongePlane(double *x, double &wgt) {
 
  double normal[dim_];
  double point[dim_];
  double s[dim_];
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

//  TODO: add suppport for arbitrary orientation of cylinder axis
void GeometricSponge::spongeCylinder(double *xGlobal, double &wgt) {

  double normal[dim_];
  double point[dim_];
  double s[dim_];
  double factor, width, dist, wgt0;  
  double wgtCyl;  
  
  // cylinder orientation and size
  double cylX = cylinder.radiusX;
  double cylY = cylinder.radiusY;
  double cylZ = cylinder.radiusZ;
  width = cylinder.width;
    
  // offset origin of cylinder
  double x[dim_];
  for(int i=0; i<dim_; i++) { point[i] = cylinder.point[i]; }  
  for(int i=0; i<dim_; i++) { x[i] = xGlobal[i] - point[i]; }

  factor = cylinder.mult;  

  // cylinder aligned with one axis
  if ( cylX > 0.0) {
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

// TODO: NOT GENERAL, only for annulus aligned with y
void GeometricSponge::spongeAnnulus(double *xGlobal, double &wgt) {  

  double wgtAnn, wgt0;
  double normal[dim_];
  double point[dim_];
  double s[dim_];
  double x[dim_];  
  
  double centerAnnulus[3];  
  for (int d = 0; d < dim_; d++) {
    centerAnnulus[d] = annulus.point[d];
  }
  double rad1 = annulus.radiusY; // ?
  double rad2 = annulus.width;  
  double factor = annulus.mult;
  
  // distance to center
  double dist1 = x[0]*x[0] + x[2]*x[2];
  dist1 = std::sqrt(dist1);

  // coordinates of nearest point on annulus center ring
  for (int d = 0; d < dim_; d++) s[d] = (rad1/dist1) * x[d];
  s[1] = centerAnnulus[1];
  
  // distance to ring
  for (int d = 0; d < dim_; d++) s[d] = x[d] - s[d];  
  double dist2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  dist2 = std::sqrt(dist2);

  //if (x[1] <= centerAnnulus[1]+rad2) { 
  //  std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << endl;
  //}  

  //if (dist2 <= rad2) { 
  //  std::cout << " rad ratio: " << rad1/dist1 << " ring dist: " << dist2 << " fA: " << factorA << endl;
  //}  
  
  // sponge weight
  wgtAnn = 0.5 * (tanh( 10.0*(1.0 - dist2/rad2) ) + 1.0);
  wgtAnn = (wgtAnn - wgt0) * 1.0/(1.0 - wgt0);
  wgtAnn = std::max(wgtAnn,0.0);
  wgtAnn *= (factor - 1.0);
  wgtAnn += 1.0;
  wgt = std::max(wgt,wgtAnn);
 
}

void GeometricSponge::step() {
  // empty for now, use for any updates/modifications
  // during simulation e.g. ramping down with time
}


