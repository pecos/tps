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

#include "gslib_interpolator.hpp"

using namespace std;
using namespace mfem;

InterpolatorBase::InterpolatorBase() {
  dim_ = 0;
  xyz_.SetSize(0);
  soln_.SetSize(0);
#ifdef HAVE_GSLIB
  finder_ = NULL;
#endif
}

InterpolatorBase::~InterpolatorBase() {
#ifdef HAVE_GSLIB
  delete finder_;
#endif
}

void InterpolatorBase::initializeFinder(ParMesh *mesh) {
  dim_ = mesh->Dimension();
#ifdef HAVE_GSLIB
  finder_ = new FindPointsGSLIB(mesh->GetComm());
  finder_->Setup(*mesh);
  finder_->SetDefaultInterpolationValue(0.0); // NOTE(malamast): Do I need that?
#else
  mfem_error("InterpolatorBase requires gslib support!");
#endif
}

void InterpolatorBase::setInterpolationPoints(mfem::Vector xyz) { xyz_ = xyz; }

void InterpolatorBase::setInterpolationPoints() {
  mfem_error("InterpolatorBase::setInterpolationPoints() is not implemented");
}

void InterpolatorBase::interpolate(ParGridFunction *u) {
  assert(dim_ == 2 || dim_ == 3);

  const int totalPts = xyz_.Size(); // NOTE(malamast): Do I need to divide by dim_?
  assert(totalPts > 0);

  assert(u != NULL);
  const int interpNum = u->VectorDim();

  soln_.SetSize(totalPts * interpNum);
#ifdef HAVE_GSLIB
  finder_->Interpolate(xyz_, *u, soln_);
#else
  mfem_error("InterpolatorBase requires gslib support!");
#endif
}

void InterpolatorBase::writeAscii(std::string oname, bool rank0) const {
  if (rank0) {
    assert(dim_ == 2 || dim_ == 3);
    assert(xyz_.Size() > 0);
    assert(soln_.Size() > 0);

    const int totalPts = xyz_.Size() / dim_;
    const int interpNum = soln_.Size() / totalPts;

    std::ofstream outfile;
    outfile.open(oname, std::ios_base::app);
    for (int n = 0; n < totalPts; n++) {
      outfile << n << " ";
      for (int d = 0; d < dim_; d++) {
        outfile << xyz_[n + d * totalPts] << " ";
      }
      for (int eq = 0; eq < interpNum; eq++) {
        outfile << soln_[n + eq * totalPts] << " ";
      }
      outfile << endl;
    }
    outfile.close();
  }
}

PlaneInterpolator::PlaneInterpolator() : InterpolatorBase() {
  point_.SetSize(0);
  normal_.SetSize(0);
}

PlaneInterpolator::PlaneInterpolator(mfem::Vector point, mfem::Vector normal, mfem::Vector bb0, mfem::Vector bb1, int n)
    : InterpolatorBase(), point_(point), normal_(normal), bb0_(bb0), bb1_(bb1), n_(n) {}

PlaneInterpolator::~PlaneInterpolator() {}

void PlaneInterpolator::setInterpolationPoints() {
  assert(dim_ == 3);
  const int totalPts = n_ * n_;

  double ndotp = 0.0;
  double majorD;
  for (int i = 0; i < dim_; i++) {
    ndotp += normal_[i] * point_[i];
  }
  majorD = std::max(std::abs(normal_[0]), std::abs(normal_[1]));
  if (dim_ == 3) {
    majorD = std::max(majorD, std::abs(normal_[2]));
  }

  // plane points
  xyz_.SetSize(totalPts * 3);

  int iCnt = 0;
  const double Lx = bb1_[0] - bb0_[0];
  const double Ly = bb1_[1] - bb0_[1];
  const double Lz = bb1_[2] - bb0_[2];

  const double ncell = (double)(n_ - 1);

  // TODO(trevilo): This only works when the plane we want is aligned
  // with a coordinate direction.  Extend to generic plane.
  if (majorD == std::abs(normal_[0])) {
    const double dy = Ly / ncell;
    const double dz = Lz / ncell;
    for (int j = 0; j < n_; j++) {
      for (int i = 0; i < n_; i++) {
        const double yp = dy * (double)i + bb0_[1];
        const double zp = dz * (double)j + bb0_[2];
        const double xp = (ndotp - (normal_[1] * yp) - (normal_[2] * zp)) / normal_[0];
        xyz_[iCnt + 0 * totalPts] = xp;
        xyz_[iCnt + 1 * totalPts] = yp;
        xyz_[iCnt + 2 * totalPts] = zp;
        iCnt++;
      }
    }
  } else if (majorD == std::abs(normal_[1])) {
    const double dx = Lx / ncell;
    const double dz = Lz / ncell;
    for (int j = 0; j < n_; j++) {
      for (int i = 0; i < n_; i++) {
        const double xp = dx * (double)i + bb0_[0];
        const double zp = dz * (double)j + bb0_[2];
        const double yp = (ndotp - (normal_[0] * xp) - (normal_[2] * zp)) / normal_[1];
        xyz_[iCnt + 0 * totalPts] = xp;
        xyz_[iCnt + 1 * totalPts] = yp;
        xyz_[iCnt + 2 * totalPts] = zp;
        iCnt++;
      }
    }
  } else {
    const double dx = Lx / ncell;
    const double dy = Ly / ncell;
    for (int j = 0; j < n_; j++) {
      for (int i = 0; i < n_; i++) {
        const double xp = dx * (double)i + bb0_[0];
        const double yp = dy * (double)j + bb0_[1];
        const double zp = (ndotp - (normal_[0] * xp) - (normal_[1] * yp)) / normal_[2];
        xyz_[iCnt + (0 * totalPts)] = xp;
        xyz_[iCnt + (1 * totalPts)] = yp;
        xyz_[iCnt + (2 * totalPts)] = zp;
        iCnt++;
      }
    }
  }
}

void PlaneInterpolator::writeAscii(std::string oname, bool rank0) const {
  if (rank0) {
    std::cout << " Writing plane data to " << oname << endl;

    // First write a little info about the plane
    assert(dim_ == 3);
    std::ofstream outfile;
    outfile.open(oname, std::ios_base::app);
    outfile << "#plane point " << point_[0] << " " << point_[1] << " " << point_[2] << endl;
    outfile << "#plane normal " << normal_[0] << " " << normal_[1] << " " << normal_[2] << endl;
    outfile.close();
  }
  // Rest of the write is handled by the base class
  InterpolatorBase::writeAscii(oname, rank0);
}


// ---------------------------------------------
// Routines to perform interpolation of a given grid function 
// to a grid function of a different finite element space
// ---------------------------------------------

FieldInterpolator::FieldInterpolator(
                ParMesh *_mesh_1, FiniteElementCollection *_fec_1, ParFiniteElementSpace *_fes_1,
                ParMesh *_mesh_2, FiniteElementCollection *_fec_2, ParFiniteElementSpace *_fes_2)
    :InterpolatorBase(),
     src_mesh(_mesh_1),
     src_fec(_fec_1),
     src_fes(_fes_1),
     tar_mesh(_mesh_2),
     tar_fec(_fec_2),
     tar_fes(_fes_2)
    {
  // 1 -> refers to the source mesh.
  // 2 -> refers to the target mesh.

  nprocs_ = Mpi::WorldSize();
  rank_ = Mpi::WorldRank();
  rank0_ = Mpi::Root();

  // Set mesh  Curvature 
  if (src_mesh->GetNodes() == NULL) src_mesh->SetCurvature(1);
  if (tar_mesh->GetNodes() == NULL) tar_mesh->SetCurvature(1);
  // if (tar_mesh->GetNodes() == NULL) tar_mesh->SetCurvature(1, false, -1, 0);
  // if (rank0_) cout << "Source mesh curvature: " << src_mesh->GetNodes()->OwnFEC()->Name() << endl;
  // if (rank0_) cout << "Target mesh curvature: " << tar_mesh->GetNodes()->OwnFEC()->Name() << endl;

  InterpolatorBase::initializeFinder(src_mesh);

  // GSLIB only works in 2 and 3 dimensions
  assert(dim_ > 1);
  // Input meshes must have same dimension
  assert(tar_mesh->Dimension() == src_mesh->Dimension());

  // Some checks
  const Geometry::Type gt = tar_mesh->GetNodalFESpace()->GetFE(0)->GetGeomType();
  MFEM_VERIFY(gt != Geometry::PRISM,"Wedge elements are not currently supported.");
  MFEM_VERIFY(tar_mesh->GetNumGeometries(tar_mesh->Dimension()) == 1,
              "Mixed meshes are not currently supported.");

  if (rank0_) std::cout << "Source FE collection: " << src_fec->Name() << std::endl;
  if (rank0_) std::cout << "Target FE collection: " << tar_fec->Name() << std::endl;

  // Auxiliary variables
  NE = tar_mesh->GetNE();
  nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints();
  n_interp_nodes = 0;
  for (int i = 0; i < tar_mesh->GetNE(); i++) {
    n_interp_nodes += tar_fes->GetFE(i)->GetNodes().GetNPoints();
  }

  assert(n_interp_nodes == NE * nsp);
  setInterpolationPoints();
}

FieldInterpolator::~FieldInterpolator() {}

void FieldInterpolator::setInterpolationPoints() {

  // NOTE(malamast): Where do we use pos, rowx,rowy,rowz??

  // The method for getting the necessary points here works for both
  // H1 and L2 fields.  However, for H1, it will contain duplicate
  // points, which means we must take care when setting the
  // corresponding ParGridFunction below.

  xyz_.SetSize(n_interp_nodes * dim_);

  for (int i = 0; i < NE; i++) {
    const FiniteElement *fe = tar_fes->GetFE(i);
    const IntegrationRule ir = fe->GetNodes();
    ElementTransformation *et = tar_fes->GetElementTransformation(i);
    // const int nsp = ir.GetNPoints();
    DenseMatrix pos;
    et->Transform(ir, pos);
    Vector rowx(xyz_.GetData() + i * nsp, nsp);
    Vector rowy(xyz_.GetData() + i * nsp + n_interp_nodes, nsp);
    Vector rowz;
    if (dim_ == 3) {
      rowz.SetDataAndSize(xyz_.GetData() + i * nsp + 2 * n_interp_nodes, nsp);
    }
    pos.GetRow(0, rowx);
    pos.GetRow(1, rowy);
    if (dim_ == 3) {
      pos.GetRow(2, rowz);
    }
  }
}

void FieldInterpolator::PerformInterpolation(ParGridFunction *func_source, ParGridFunction *func_target) {
  assert(func_source->ParFESpace() == src_fes);
  assert(func_target->ParFESpace() == tar_fes);

  const int nodes_cnt = xyz_.Size() / dim_;
  const int tar_ncomp = func_target->VectorDim();
  // if (rank0_) cout << "tar_ncomp = " << tar_ncomp << std::endl;

  // Interpolate source grid function.
  Vector interp_vals(nodes_cnt * tar_ncomp);
  finder_->Interpolate(xyz_, *func_source, interp_vals);

  // Set grid function
  // if (FromDGToH1Space) {
    Array<int> vdofs;
    Vector elem_dof_vals(nsp * tar_ncomp);

    for (int i = 0; i < NE; i++) {
      tar_fes->GetElementVDofs(i, vdofs);
      for (int j = 0; j < nsp; j++) {
        for (int d = 0; d < tar_ncomp; d++) {
          // Arrange values byNodes
          int idx = d * nsp * NE + i * nsp + j;
          elem_dof_vals(j + d * nsp) = interp_vals(idx);
        }
      }
      func_target->SetSubVector(vdofs, elem_dof_vals);
    }
    func_target->SetTrueVector(); // Do I need that?
  // }

  func_target->SetFromTrueVector();
}

