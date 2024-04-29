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

#include "algebraicSubgridModels.hpp"

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

using namespace mfem;
using namespace mfem::common;

AlgebraicSubgridModels::AlgebraicSubgridModels(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps,
                                               ParGridFunction *gridScale, int sModel)
    : tpsP_(tps), loMach_opts_(loMach_opts), pmesh_(pmesh) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  dim_ = pmesh_->Dimension();
  nvel_ = dim_;
  order_ = loMach_opts_->order;
  gridScale_ = gridScale;
  sModel_ = sModel;

  // read-ins
  double sgs_const = 0.;
  if (sModel_ == 1) {
    sgs_const = 0.09;
  } else if (sModel_ == 2) {
    sgs_const = 0.135;
  }
  tpsP_->getInput("loMach/sgsModelConstant", sgs_model_const_, sgs_const);
}

AlgebraicSubgridModels::~AlgebraicSubgridModels() {
  // delete bufferGridScale_;
  delete sfec_;
  delete sfes_;
  delete vfec_;
  delete vfes_;
}

void AlgebraicSubgridModels::initializeSelf() {
  // bool verbose = rank0;
  if (rank0_) grvy_printf(ginfo, "Initializing AlgebraicSubgridModels solver.\n");

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  // vector
  vfec_ = new H1_FECollection(order_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  int vfes_truevsize = vfes_->GetTrueVSize();
  int sfes_truevsize = sfes_->GetTrueVSize();

  Sdof_ = sfes_->GetNDofs();
  SdofInt_ = sfes_->GetTrueVSize();

  subgridVisc_.SetSize(sfes_truevsize);
  subgridVisc_gf_.SetSpace(sfes_);
  delta_.SetSize(sfes_truevsize);

  gradU_.SetSize(vfes_truevsize);
  gradV_.SetSize(vfes_truevsize);
  gradW_.SetSize(vfes_truevsize);
  rn_.SetSize(sfes_truevsize);

  // gridScale_.SetSize(sfes_truevsize);
  // gridScaleX_.SetSize(sfes_truevsize);
  // gridScaleY_.SetSize(sfes_truevsize);
  // gridScaleZ_.SetSize(sfes_truevsize);
  // resolution_gf_.SetSpace(sfes_);

  if (verbose) grvy_printf(ginfo, "AlgebraicSubgridModels vectors and gf initialized...\n");

  // exports
  toFlow_interface_.eddy_viscosity = &subgridVisc_gf_;
  toThermoChem_interface_.eddy_viscosity = &subgridVisc_gf_;
}

void AlgebraicSubgridModels::initializeOperators() {
  // By calling step here, we initialize the subgrid viscosity using
  // the initial velocity gradient
  this->step();
}

void AlgebraicSubgridModels::initializeViz(ParaViewDataCollection &pvdc) {
  // pvdc.RegisterField("resolution", &resolution_gf_);
  pvdc.RegisterField("muT", &subgridVisc_gf_);
}

void AlgebraicSubgridModels::setup() {
  /*
  /// Grid-related ///

  // Build grid size vector and grid function
  bufferGridScale_ = new ParGridFunction(sfes_);
  // bufferGridScaleX = new ParGridFunction(sfes);
  // bufferGridScaleY = new ParGridFunction(sfes);
  // bufferGridScaleZ = new ParGridFunction(sfes);
  ParGridFunction dofCount(sfes_);
  {
    int elndofs;
    Array<int> vdofs;
    Vector vals;
    Vector loc_data;
    int nSize = bufferGridScale_->Size();
    Array<int> zones_per_vdof;
    zones_per_vdof.SetSize(sfes_->GetVSize());
    zones_per_vdof = 0;
    Array<int> zones_per_vdofALL;
    zones_per_vdofALL.SetSize(sfes_->GetVSize());
    zones_per_vdofALL = 0;

    double *data = bufferGridScale_->HostReadWrite();
    double *count = dofCount.HostReadWrite();

    for (int i = 0; i < sfes_->GetNDofs(); i++) {
      data[i] = 0.0;
      count[i] = 0.0;
    }

    // element loop
    for (int e = 0; e < sfes_->GetNE(); ++e) {
      sfes_->GetElementVDofs(e, vdofs);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = sfes_->GetElementTransformation(e);
      const FiniteElement *el = sfes_->GetFE(e);
      elndofs = el->GetDof();
      double delta;

      // element dof
      for (int dof = 0; dof < elndofs; ++dof) {
        const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
        tr->SetIntPoint(&ip);
        delta = pmesh_->GetElementSize(tr->ElementNo, 1);
        delta = delta / ((double)order_);
        vals(dof) = delta;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        data[ldof + 0 * nSize] += vals[j];
      }

      for (int j = 0; j < vdofs.Size(); j++) {
        int ldof = vdofs[j];
        zones_per_vdof[ldof]++;
      }
    }

    // Count the zones globally.
    GroupCommunicator &gcomm = bufferGridScale_->ParFESpace()->GroupComm();
    gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
    gcomm.Bcast(zones_per_vdof);

    // Accumulate for all vdofs.
    gcomm.Reduce<double>(bufferGridScale_->GetData(), GroupCommunicator::Sum);
    gcomm.Bcast<double>(bufferGridScale_->GetData());

    // Compute means.
    for (int i = 0; i < nSize; i++) {
      const int nz = zones_per_vdof[i];
      if (nz) {
        data[i] /= nz;
      }
    }

    bufferGridScale_->GetTrueDofs(gridScale_);
    // bufferGridScaleX->GetTrueDofs(gridScaleXSml);
    // bufferGridScaleY->GetTrueDofs(gridScaleYSml);
    // bufferGridScaleZ->GetTrueDofs(gridScaleZSml);
  }

  resolution_gf_ = *bufferGridScale_;
  */
}

void AlgebraicSubgridModels::step() {
  // std::cout << "In SGS step: " << sModel_ << "..." << endl;
  if (sModel_ != 1 && sModel_ != 2) {
    return;
  }
  // std::cout << " giddy up" << endl;

  // gather necessary information from other classes
  (flow_interface_->gradU)->GetTrueDofs(gradU_);
  (flow_interface_->gradV)->GetTrueDofs(gradV_);
  (flow_interface_->gradW)->GetTrueDofs(gradW_);
  (thermoChem_interface_->density)->GetTrueDofs(rn_);

  subgridVisc_ = 0.0;
  const double *dGradU = gradU_.HostRead();
  const double *dGradV = gradV_.HostRead();
  const double *dGradW = gradW_.HostRead();
  const double *rho = rn_.HostRead();
  const double *del = gridScale_->HostRead();
  double *data = subgridVisc_.HostReadWrite();

  if (sModel_ == 1) {
    // std::cout << "In Smag loop " << endl;
    for (int i = 0; i < SdofInt_; i++) {
      double nu_sgs = 0.;
      DenseMatrix gradUp;
      gradUp.SetSize(nvel_, dim_);
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(0, dir) = dGradU[i + dir * SdofInt_];
      }
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(1, dir) = dGradV[i + dir * SdofInt_];
      }
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(2, dir) = dGradW[i + dir * SdofInt_];
      }
      sgsSmag(gradUp, del[i], nu_sgs);
      // std::cout << "gradU diag:" << gradUp(0,0) << " " << gradUp(1,1) << " " << gradUp(1,1) << "| nuT: " << nu_sgs <<
      // endl;
      data[i] = rho[i] * nu_sgs;
    }

  } else if (sModel_ == 2) {
    for (int i = 0; i < SdofInt_; i++) {
      double nu_sgs = 0.;
      DenseMatrix gradUp;
      gradUp.SetSize(nvel_, dim_);
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(0, dir) = dGradU[i + dir * SdofInt_];
      }
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(1, dir) = dGradV[i + dir * SdofInt_];
      }
      for (int dir = 0; dir < dim_; dir++) {
        gradUp(2, dir) = dGradW[i + dir * SdofInt_];
      }
      sgsSigma(gradUp, del[i], nu_sgs);
      data[i] = rho[i] * nu_sgs;
    }
  }

  subgridVisc_gf_.SetFromTrueDofs(subgridVisc_);
}

/**
Basic Smagorinksy subgrid model with user-specified coefficient
*/
void AlgebraicSubgridModels::sgsSmag(const DenseMatrix &gradUp, double delta, double &nu) {
  Vector Sij(6);
  double Smag = 0.;
  double Cd;
  // double l_floor; // TODO(swh): re-instate
  double d_model;
  Cd = sgs_model_const_;

  // gradUp is in (eq,dim) form
  Sij[0] = gradUp(0, 0);
  Sij[1] = gradUp(1, 1);
  Sij[2] = gradUp(2, 2);
  Sij[3] = 0.5 * (gradUp(0, 1) + gradUp(1, 0));
  Sij[4] = 0.5 * (gradUp(0, 2) + gradUp(2, 0));
  Sij[5] = 0.5 * (gradUp(1, 2) + gradUp(2, 1));

  // strain magnitude with silly sqrt(2) factor
  for (int i = 0; i < 3; i++) Smag += Sij[i] * Sij[i];
  for (int i = 3; i < 6; i++) Smag += 2.0 * Sij[i] * Sij[i];
  Smag = sqrt(2.0 * Smag);

  // eddy viscosity with delta shift
  // l_floor = config->GetSgsFloor();
  // d_model = Cd * max(delta-l_floor,0.0);
  d_model = Cd * delta;
  nu = d_model * d_model * Smag;
}

/**
NOT TESTED: Sigma subgrid model following Nicoud et.al., "Using singular values to build a
subgrid-scale model for large eddy simulations", PoF 2011.
*/
void AlgebraicSubgridModels::sgsSigma(const DenseMatrix &gradUp, double delta, double &nu) {
  DenseMatrix Qij(dim_, dim_);
  DenseMatrix du(dim_, dim_);
  DenseMatrix B(dim_, dim_);
  Vector ev(dim_);
  Vector sigma(dim_);
  double Cd;
  double sml = 1.0e-12;
  double pi = 3.14159265359;
  double onethird = 1. / 3.;
  double d_model, d4;
  double p1, p2, p, q, detB, r, phi;
  Cd = sgs_model_const_;

  // Qij = u_{k,i}*u_{k,j}
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      Qij(i, j) = 0.;
    }
  }
  for (int k = 0; k < dim_; k++) {
    for (int j = 0; j < dim_; j++) {
      for (int i = 0; i < dim_; i++) {
        Qij(i, j) += gradUp(k, i) * gradUp(k, j);
      }
    }
  }

  // shifted grid scale, d should really be sqrt of J^T*J
  // l_floor = config->GetSgsFloor();
  // d_model = max((delta-l_floor), sml);
  d_model = delta;
  d4 = pow(d_model, 4);
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      Qij(i, j) *= d4;
    }
  }

  // eigenvalues for symmetric pos-def 3x3
  p1 = Qij(0, 1) * Qij(0, 1) + Qij(0, 2) * Qij(0, 2) + Qij(1, 2) * Qij(1, 2);
  q = onethird * (Qij(0, 0) + Qij(1, 1) + Qij(2, 2));
  p2 = (Qij(0, 0) - q) * (Qij(0, 0) - q) + (Qij(1, 1) - q) * (Qij(1, 1) - q) + (Qij(2, 2) - q) * (Qij(2, 2) - q) +
       2.0 * p1;
  p = std::sqrt(max(p2, 0.0) / 6.0);
  // cout << "p1, q, p2, p: " << p1 << " " << q << " " << p2 << " " << p << endl; fflush(stdout);

  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      B(i, j) = Qij(i, j);
    }
  }
  for (int i = 0; i < dim_; i++) {
    B(i, i) -= q;
  }
  for (int j = 0; j < dim_; j++) {
    for (int i = 0; i < dim_; i++) {
      B(i, j) *= (1.0 / max(p, sml));
    }
  }
  detB = B(0, 0) * (B(1, 1) * B(2, 2) - B(2, 1) * B(1, 2)) - B(0, 1) * (B(1, 0) * B(2, 2) - B(2, 0) * B(1, 2)) +
         B(0, 2) * (B(1, 0) * B(2, 1) - B(2, 0) * B(1, 1));
  r = 0.5 * detB;
  // cout << "r: " << r << endl; fflush(stdout);

  if (r <= -1.0) {
    phi = onethird * pi;
  } else if (r >= 1.0) {
    phi = 0.0;
  } else {
    phi = onethird * acos(r);
  }

  // eigenvalues satisfy eig3 <= eig2 <= eig1 (L^4/T^2)
  ev[0] = q + 2.0 * p * cos(phi);
  ev[2] = q + 2.0 * p * cos(phi + (2.0 * onethird * pi));
  ev[1] = 3.0 * q - ev[0] - ev[2];

  // actual sigma (L^2/T)
  sigma[0] = sqrt(max(ev[0], sml));
  sigma[1] = sqrt(max(ev[1], sml));
  sigma[2] = sqrt(max(ev[2], sml));
  // cout << "sigma: " << sigma[0] << " " << sigma[1] << " " << sigma[2] << endl; fflush(stdout);

  // eddy viscosity
  nu = sigma[2] * (sigma[0] - sigma[1]) * (sigma[1] - sigma[2]);
  nu = max(nu, 0.0);
  nu /= (sigma[0] * sigma[0]);
  nu *= (Cd * Cd);
  // cout << "mu: " << mu << endl; fflush(stdout);

  // shouldnt be necessary
  if (nu != nu) nu = 0.0;
}
