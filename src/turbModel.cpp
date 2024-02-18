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

#include "turbModel.hpp"

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

TurbModel::TurbModel(mfem::ParMesh *pmesh, RunConfiguration *config, LoMachOptions *loMach_opts,
                     temporalSchemeCoefficients &timeCoeff)
    : pmesh_(pmesh), loMach_opts_(loMach_opts), config_(config), timeCoeff_(timeCoeff) {}

void TurbModel::initializeSelf() {
  // groupsMPI = pmesh->GetComm();
  rank = pmesh_->GetMyRank();
  // rank0 = pmesh->isWorldRoot();
  rank0 = false;
  if (rank == 0) {
    rank0 = true;
  }
  dim = pmesh_->Dimension();
  nvel = dim;

  bool verbose = rank0;
  if (verbose) grvy_printf(ginfo, "Initializing TurbModel solver.\n");

  if (loMach_opts_->uOrder == -1) {
    order = std::max(config_->solOrder, 1);
    double no;
    no = ceil(((double)order * 1.5));
  } else {
    order = loMach_opts_->uOrder;
  }

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec = new H1_FECollection(order);
  sfes = new ParFiniteElementSpace(pmesh_, sfec);

  // vector
  vfec = new H1_FECollection(order, dim);
  vfes = new ParFiniteElementSpace(pmesh_, vfec, dim);

  int vfes_truevsize = vfes->GetTrueVSize();
  int sfes_truevsize = sfes->GetTrueVSize();

  subgridVisc.SetSize(sfes_truevsize);
  subgridVisc_gf.SetSpace(sfes);
  delta.SetSize(sfes_truevsize);

  gradU.SetSize(vfes_truevsize);
  gradV.SetSize(vfes_truevsize);
  gradW.SetSize(vfes_truevsize);
  rn.SetSize(sfes_truevsize);

  if (verbose) grvy_printf(ginfo, "TurbModel vectors and gf initialized...\n");

  // exports
  toFlow_interface_.eddy_viscosity = &subgridVisc_gf;
  toThermoChem_interface_.eddy_viscosity = &subgridVisc_gf;
}

void TurbModel::setup(double dt) {
  // empty for now, will be used for transport models
}

void TurbModel::step() {
  // gather necessary information from other classes
  (flow_interface_->gradU)->GetTrueDofs(gradU);
  (flow_interface_->gradV)->GetTrueDofs(gradV);
  (flow_interface_->gradW)->GetTrueDofs(gradW);
  (thermoChem_interface_->density)->GetTrueDofs(rn);
  // (grid_interface_->delta).GetTrueDofs(delta); // doesnt exist yet
  delta = 0.0;  // temporary

  subgridVisc = 0.0;
  if (config_->sgsModelType > 0) {
    double *dGradU = gradU.HostReadWrite();
    double *dGradV = gradV.HostReadWrite();
    double *dGradW = gradW.HostReadWrite();
    double *del = delta.HostReadWrite();
    double *rho = rn.HostReadWrite();
    double *data = subgridVisc.HostReadWrite();

    if (config_->sgsModelType == 1) {
      for (int i = 0; i < SdofInt; i++) {
        double nu_sgs = 0.;
        DenseMatrix gradUp;
        gradUp.SetSize(nvel, dim);
        for (int dir = 0; dir < dim; dir++) {
          gradUp(0, dir) = dGradU[i + dir * SdofInt];
        }
        for (int dir = 0; dir < dim; dir++) {
          gradUp(1, dir) = dGradV[i + dir * SdofInt];
        }
        for (int dir = 0; dir < dim; dir++) {
          gradUp(2, dir) = dGradW[i + dir * SdofInt];
        }
        sgsSmag(gradUp, del[i], nu_sgs);
        data[i] = rho[i] * nu_sgs;
      }

    } else if (config_->sgsModelType == 2) {
      for (int i = 0; i < SdofInt; i++) {
        double nu_sgs = 0.;
        DenseMatrix gradUp;
        gradUp.SetSize(nvel, dim);
        for (int dir = 0; dir < dim; dir++) {
          gradUp(0, dir) = dGradU[i + dir * SdofInt];
        }
        for (int dir = 0; dir < dim; dir++) {
          gradUp(1, dir) = dGradV[i + dir * SdofInt];
        }
        for (int dir = 0; dir < dim; dir++) {
          gradUp(2, dir) = dGradW[i + dir * SdofInt];
        }
        sgsSigma(gradUp, del[i], nu_sgs);
        data[i] = rho[i] * nu_sgs;
      }

    } else {
      if (rank0) std::cout << " => turbulence model not supported!!!" << endl;
      exit(1);
    }
  }

  subgridVisc_gf.SetFromTrueDofs(subgridVisc);
}

/**
Basic Smagorinksy subgrid model with user-specified coefficient
*/
void TurbModel::sgsSmag(const DenseMatrix &gradUp, double delta, double &nu) {
  Vector Sij(6);
  double Smag = 0.;
  double Cd;
  double l_floor;
  double d_model;
  Cd = config_->sgs_model_const;

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
void TurbModel::sgsSigma(const DenseMatrix &gradUp, double delta, double &nu) {
  DenseMatrix Qij(dim, dim);
  DenseMatrix du(dim, dim);
  DenseMatrix B(dim, dim);
  Vector ev(dim);
  Vector sigma(dim);
  double Cd;
  double sml = 1.0e-12;
  double pi = 3.14159265359;
  double onethird = 1. / 3.;
  double l_floor, d_model, d4;
  double p1, p2, p, q, detB, r, phi;
  Cd = config_->sgs_model_const;

  // Qij = u_{k,i}*u_{k,j}
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      Qij(i, j) = 0.;
    }
  }
  for (int k = 0; k < dim; k++) {
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
        Qij(i, j) += gradUp(k, i) * gradUp(k, j);
      }
    }
  }

  // shifted grid scale, d should really be sqrt of J^T*J
  // l_floor = config->GetSgsFloor();
  // d_model = max((delta-l_floor), sml);
  d_model = delta;
  d4 = pow(d_model, 4);
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
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

  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
      B(i, j) = Qij(i, j);
    }
  }
  for (int i = 0; i < dim; i++) {
    B(i, i) -= q;
  }
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {
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
