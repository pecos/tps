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

#include "gaussianInterpExtData.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "externalData_base.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

GaussianInterpExtData::GaussianInterpExtData(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps)
    : tpsP_(tps), loMach_opts_(loMach_opts), pmesh_(pmesh) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  // only allows one inlet now
  isInterpInlet_ = false;

  int numInlets;
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);

  for (int i = 1; i <= numInlets; i++) {
    std::string type;
    std::string basepath("boundaryConditions/inlet" + std::to_string(i));
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
    if (type == "interpolate") {
      if (isInterpInlet_) {
        // If we've already set isInterpInlet_, then user requested
        // multiple interpolation BCs, which is not supported
        if (rank0_) std::cout << "Only one interpolated inlet currently supported." << endl;
      }
      isInterpInlet_ = true;
      tpsP_->getInput((basepath + "/name").c_str(), fname_, std::string("inletPlane.csv"));
    }
  }
}

GaussianInterpExtData::~GaussianInterpExtData() {
  delete sfec_;
  delete sfes_;
  delete vfec_;
  delete vfes_;
}

void GaussianInterpExtData::initializeSelf() {
  if (!isInterpInlet_) {
    return;
  }

  rank_ = pmesh_->GetMyRank();
  rank0_ = false;
  if (rank_ == 0) {
    rank0_ = true;
  }
  dim_ = pmesh_->Dimension();

  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing Gaussian interpolation inlet.\n");

  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  vfec_ = new H1_FECollection(order_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  Sdof_ = sfes_->GetNDofs();

  temperature_gf_.SetSpace(sfes_);
  temperature_gf_ = 0.0;
  velocity_gf_.SetSpace(vfes_);
  velocity_gf_ = 0.0;

  // exports
  toThermoChem_interface_.Tdata = &temperature_gf_;
  toFlow_interface_.Udata = &velocity_gf_;
}

void GaussianInterpExtData::initializeViz(ParaViewDataCollection &pvdc) {
  if (!isInterpInlet_) {
    return;
  }
  pvdc.RegisterField("externalTemp", &temperature_gf_);
  pvdc.RegisterField("externalU", &velocity_gf_);
}

// TODO(swh): add a translation and rotation for external data plane
void GaussianInterpExtData::setup() {
  if (rank0_) {
    std::cout << "in interp setup..." << endl;
  }

  if (!isInterpInlet_) {
    return;
  }
  ParGridFunction coordsDof(vfes_);
  pmesh_->GetNodes(coordsDof);

  double *Tdata = temperature_gf_.HostReadWrite();
  double *Udata = velocity_gf_.HostReadWrite();
  double *hcoords = coordsDof.HostReadWrite();

  struct inlet_profile {
    double x, y, z, rho, temp, u, v, w;
  };

  string fnameBase;
  std::string basepath("./inputs/");
  fnameBase = (basepath + fname_);
  const char *fnameRead = fnameBase.c_str();
  int nCount = 0;

  // find size
  if (rank0_) {
    std::cout << " Attempting to open inlet file for counting... " << fnameRead << " ";

    FILE *inlet_file;
    if ((inlet_file = fopen(fnameRead, "r"))) {
      std::cout << " ...and open" << endl;
      fflush(stdout);
    } else {
      std::cout << " ...CANNOT OPEN FILE" << endl;
      fflush(stdout);
    }

    int ch = 0;
    for (ch = getc(inlet_file); ch != EOF; ch = getc(inlet_file)) {
      if (ch == '\n') {
        nCount++;
      }
    }
    fclose(inlet_file);
    std::cout << " final external data line count: " << nCount << endl;
    fflush(stdout);
  }

  // broadcast size
  MPI_Bcast(&nCount, 1, MPI_INT, 0, tpsP_->getTPSCommWorld());
  struct inlet_profile inlet[nCount];

  // mean output from plane interpolation
  // 0) no, 1) x, 2) y, 3) z, 4) rho, 5) u, 6) v, 7) w
  // Change format so temp is given instead of rho!

  // open, read data
  if (rank0_) {
    ifstream file;
    file.open(fnameRead);

    string line;
    getline(file, line);
    int nLines = 0;
    while (getline(file, line)) {
      stringstream sline(line);
      int entry = 0;
      while (sline.good()) {
        string substr;
        getline(sline, substr, ',');  // csv delimited
        double buffer = std::stod(substr);
        entry++;

        // using code plane interp format
        if (entry == 2) {
          inlet[nLines].x = buffer;
        } else if (entry == 3) {
          inlet[nLines].y = buffer;
        } else if (entry == 4) {
          inlet[nLines].z = buffer;
        } else if (entry == 5) {
          // inlet[nLines].rho = buffer;
          // to prevent nans in temp in out-of-domain plane regions
          // inlet[nLines].temp = thermoPressure / (Rgas * std::max(buffer,1.0));

          // just change format so 5 slot has temp
          inlet[nLines].temp = buffer;

        } else if (entry == 6) {
          inlet[nLines].u = buffer;
        } else if (entry == 7) {
          inlet[nLines].v = buffer;
        } else if (entry == 8) {
          inlet[nLines].w = buffer;
        }
      }
      nLines++;
    }
    file.close();
  }

  // broadcast data
  MPI_Bcast(&inlet, nCount * 8, MPI_DOUBLE, 0, tpsP_->getTPSCommWorld());
  if (rank0_) {
    std::cout << " communicated inlet data" << endl;
    fflush(stdout);
  }

  /*
  // width of interpolation stencil
  // double Lx, Ly, Lz, Lmax;
  // Lx = xmax - xmin;
  // Ly = ymax - ymin;
  // Lz = zmax - zmin;
  // Lmax = std::max(Lx,Ly);
  // Lmax = std::max(Lmax,Lz);
  //radius = 0.25 * Lmax;
  //radius = 0.01 * Lmax;

  // double torch_radius = 28.062/1000.0;
  double torch_radius = 0.2;
  double radius = 2.0 * torch_radius / nSamples; // 300 from nxn interp plane
  */
  double radius = 1.0;

  // TODO(swh): dont need to fill entire gf, just appropriate face
  // NB: be careful with GetNBE (see comments in mfem/mesh/mesh.hpp)
  for (int be = 0; be < pmesh_->GetNBE(); be++) {
    Array<int> vdofs;
    sfes_->GetBdrElementVDofs(be, vdofs);
    for (int i = 0; i < vdofs.Size(); i++) {
      // index in gf of bndry element
      int n = vdofs[i];

      double xp[3];
      for (int d = 0; d < dim_; d++) {
        xp[d] = hcoords[n + d * Sdof_];
      }

      // int iCount = 0;
      double dist = 0.0;
      double wt = 0.0;
      double wt_tot = 0.0;
      double val_u = 0.0;
      double val_v = 0.0;
      double val_w = 0.0;
      double val_T = 0.0;

      // minimum distance in interpolant field
      double distMin = 1.0e12;
      for (int j = 0; j < nCount; j++) {
        // exclude points outside the domain
        if (inlet[j].temp < 0.0) {
          continue;
        }

        dist = sqrt((xp[0] - inlet[j].x) * (xp[0] - inlet[j].x) + (xp[1] - inlet[j].y) * (xp[1] - inlet[j].y) +
                    (xp[2] - inlet[j].z) * (xp[2] - inlet[j].z));
        distMin = std::min(distMin, dist);
      }

      // find second closest data pt
      double distMinSecond = 1.0e12;
      for (int j = 0; j < nCount; j++) {
        // exclude points outside the domain
        if (inlet[j].temp < 0.0) {
          continue;
        }

        dist = sqrt((xp[0] - inlet[j].x) * (xp[0] - inlet[j].x) + (xp[1] - inlet[j].y) * (xp[1] - inlet[j].y) +
                    (xp[2] - inlet[j].z) * (xp[2] - inlet[j].z));
        if (dist > distMin) {
          distMinSecond = std::min(distMinSecond, dist);
        }
      }

      // radius for Gaussian interpolation
      radius = distMinSecond;

      for (int j = 0; j < nCount; j++) {
        // exclude points outside the domain
        if (inlet[j].temp < 0.0) {
          continue;
        }

        dist = sqrt((xp[0] - inlet[j].x) * (xp[0] - inlet[j].x) + (xp[1] - inlet[j].y) * (xp[1] - inlet[j].y) +
                    (xp[2] - inlet[j].z) * (xp[2] - inlet[j].z));

        // gaussian interpolation
        if (dist <= 1.5 * radius) {
          wt = exp(-(dist * dist) / (radius * radius));
          wt_tot += wt;
          val_u = val_u + wt * inlet[j].u;
          val_v = val_v + wt * inlet[j].v;
          val_w = val_w + wt * inlet[j].w;
          val_T = val_T + wt * inlet[j].temp;
        }

        // nearest, just for testing
        // if(dist <= distMin) {
        //  wt_tot = 1.0;
        //   val_rho = inlet[j].rho;
        //   val_u = inlet[j].u;
        //   val_v = inlet[j].v;
        //   val_w = inlet[j].w;
        //   val_T = inlet[j].temp;
        //   iCount = 1;
        // }
      }

      if (wt_tot > 0.0) {
        Udata[n + 0 * Sdof_] = val_u / wt_tot;
        Udata[n + 1 * Sdof_] = val_v / wt_tot;
        Udata[n + 2 * Sdof_] = val_w / wt_tot;
        Tdata[n] = val_T / wt_tot;
      } else {
        Udata[n + 0 * Sdof_] = 0.0;
        Udata[n + 1 * Sdof_] = 0.0;
        Udata[n + 2 * Sdof_] = 0.0;
        Tdata[n] = 0.0;
      }
    }
  }
}

void GaussianInterpExtData::step() {
  // empty for now, use for any updates/modifications
  // during simulation e.g. ramping up with time
}
