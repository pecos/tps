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

/** @file
 * @copydoc gpu_constructor.hpp
 */

#include "utils.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "M2ulPhyS.hpp"
#include "tps_mfem_wrap.hpp"

// check if the amount of time left in SLURM job is less than desired threshold.
// Subcommunicator safe version
// Forward declaration
bool slurm_job_almost_done_mpi(MPI_Comm comm, int threshold);

// check if the amount of time left in SLURM job is less than desired threshold
[[deprecated("Use slurm_job_almost_done(MPI_Comm comm, int threshold) instead")]] bool slurm_job_almost_done(
    int threshold, int rank) {
  return slurm_job_almost_done_mpi(MPI_COMM_WORLD, threshold);
}

#ifdef HAVE_SLURM
#include <slurm/slurm.h>

// check if the amount of time left in SLURM job is less than desired threshold.
// Subcommunicator safe version
bool slurm_job_almost_done_mpi(MPI_Comm comm, int threshold) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  char *SLURM_JOB_ID = getenv("SLURM_JOB_ID");
  if (SLURM_JOB_ID == NULL) {
    printf("[ERROR]: SLURM_JOB_ID env variable not set. Unable to query how much time remaining\n");
    return false;
  }

  int64_t secsRemaining;
  if (rank == 0) secsRemaining = slurm_get_rem_time(atoi(SLURM_JOB_ID));

  MPI_Bcast(&secsRemaining, 1, MPI_LONG, 0, comm);

  if (secsRemaining < 0) {
    printf("[WARN]: Unable to query how much time left for current SLURM job\n");
    return false;
  }

  if (rank == 0) printf("  --> Seconds remaining for current job = %li (threshold = %i)\n", secsRemaining, threshold);

  if (secsRemaining <= threshold) {
    if (rank == 0) {
      string delim(80, '=');
      cout << endl;
      cout << delim << endl;
      cout << "Remaining job time is less than " << threshold << " secs." << endl;
      cout << "--> Dumping final restart and terminating with exit_code = " << JOB_RESTART << endl;
      cout << "--> Use this restart exit code in your job scripts to auto-trigger a resubmit. " << endl;
      cout << delim << endl;
    }
    return true;
  } else {
    return false;
  }
}

// submit a new resource manager job with provided job script
int rm_restart(std::string jobFile, std::string mode) {
  // grvy_printf(GRVY_ginfo,"[RMS] Submitting fresh %s job using script -> %s\n",mode.c_str(),jobFile.c_str());
  std::cout << "[RMS] Submitting fresh" << mode << " job using script -> " << jobFile << std::endl;

  if (mode == "SLURM") {
    std::string command = "sbatch " + jobFile;
    std::string result = systemCmd(command.c_str());
    std::cout << result << std::endl;
  }

  return 0;
}

#else
bool slurm_job_almost_done_mpi(MPI_Comm comm, int threshold) {
  std::cout << "SLURM functionality not enabled in this build" << std::endl;
  exit(1);
}

int rm_restart(std::string jobFile, std::string mode) {
  std::cout << "SLURM functionality not enabled in this build" << std::endl;
  exit(1);
}

#endif

bool M2ulPhyS::Check_JobResubmit() {
  if (config.isAutoRestart()) {
    if (slurm_job_almost_done_mpi(this->tpsP->getTPSCommWorld(), config.rm_threshold()))
      return true;
    else
      return false;
  } else {
    return false;
  }
}

// Check for existence of DIE file on rank 0
bool M2ulPhyS::Check_ExitEarly(int iter) {
  if ((iter % config.exit_checkFreq()) == 0) {
    int status;
    if (rank0_) {
      status = static_cast<int>(file_exists("DIE"));
      if (status == 1) grvy_printf(ginfo, "Detected DIE file, terminating early...\n");
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, this->tpsP->getTPSCommWorld());
    return (static_cast<bool>(status));
  } else {
    return (false);
  }
}

// check if file exists
bool file_exists(const std::string &name) { return (access(name.c_str(), F_OK) != -1); }

// Look for existence of output.pvd files in vis output directory and
// keep a sequentially numbered copy. Useful when doing restart,
// otherwise, MFEM ParaviewDataCollection output will delete old
// timesteps from the file on restarts.
void M2ulPhyS::Cache_Paraview_Timesteps() {
  const int MAX_FILES = 9999;
  char suffix[5];

  std::string fileName = config.GetOutputName() + "/output.pvd";
  if (file_exists(fileName)) {
    for (int i = 1; i <= MAX_FILES; i++) {
      snprintf(suffix, sizeof(suffix), "%04i", i);
      std::string testFile = fileName + "." + suffix;
      if (file_exists(testFile)) {
        continue;
      } else {
        std::cout << "--> caching output.pvd file -> " << testFile << std::endl;
        std::string command = "cp " + fileName + " " + testFile;
        systemCmd(command.c_str());
        // std::filesystem::copy_file(fileName,testFile);
        return;
      }
    }
  }

  return;
}

// Run system command and capture output
std::string systemCmd(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

void LocalProjectDiscCoefficient(GridFunction &gf, VectorCoefficient &coeff, Array<int> &dof_attr) {
  Array<int> vdofs;
  Vector vals;

  gf.HostWrite();
  FiniteElementSpace *fes = gf.FESpace();

  // maximal element attribute for each dof
  dof_attr.SetSize(fes->GetVSize());
  dof_attr = -1;

  // NB: DofTransformation does not exist until after mfem 4.3, so
  // don't use here yet
  // DofTransformation * doftrans = NULL;

  // local projection
  for (int i = 0; i < fes->GetNE(); i++) {
    // doftrans = fes->GetElementVDofs(i, vdofs);
    fes->GetElementVDofs(i, vdofs);
    vals.SetSize(vdofs.Size());
    fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

    // if (doftrans) {
    //   doftrans->TransformPrimal(vals);
    // }

    // the values in shared dofs are determined from the element with maximal
    // attribute
    int attr = fes->GetAttribute(i);
    for (int j = 0; j < vdofs.Size(); j++) {
      // Mimic how negative indices are handled in Vector::SetSubVector
      int ind = vdofs[j];
      if (ind < 0) {
        ind = -1 - ind;
        vals[j] = -vals[j];
      }

      if (attr > dof_attr[ind]) {
        gf(ind) = vals[j];
        dof_attr[ind] = attr;
      }
    }
  }
}

void GlobalProjectDiscCoefficient(ParGridFunction &gf, VectorCoefficient &coeff) {
  // local maximal element attribute for each dof
  Array<int> ldof_attr;

  // local projection
  LocalProjectDiscCoefficient(gf, coeff, ldof_attr);

  // global maximal element attribute for each dof
  Array<int> gdof_attr;
  ldof_attr.Copy(gdof_attr);

  ParFiniteElementSpace *pfes = gf.ParFESpace();

  GroupCommunicator &gcomm = pfes->GroupComm();
  gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Max);
  gcomm.Bcast(gdof_attr);

  // set local value to zero if global maximal element attribute is larger than
  // the local one, and mark (in gdof_attr) if we have the correct value
  for (int i = 0; i < pfes->GetVSize(); i++) {
    if (gdof_attr[i] > ldof_attr[i]) {
      gf(i) = 0.0;
      gdof_attr[i] = 0;
    } else {
      gdof_attr[i] = 1;
    }
  }

  // parallel averaging plus interpolation to determine final values
  HypreParVector *tv = pfes->NewTrueDofVector();
  gcomm.Reduce<int>(gdof_attr, GroupCommunicator::Sum);
  gcomm.Bcast(gdof_attr);

  FiniteElementSpace *fes = gf.FESpace();
  for (int i = 0; i < fes->GetVSize(); i++) {
    gf(i) /= gdof_attr[i];
  }
  gf.ParallelAssemble(*tv);
  gf.Distribute(tv);
  delete tv;
}

bool h5ReadTable(const std::string &fileName, const std::string &datasetName, mfem::DenseMatrix &output,
                 mfem::Array<int> &shape) {
  bool success = false;
  hid_t file = -1;
  if (file_exists(fileName)) {
    file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } else {
    grvy_printf(GRVY_ERROR, "[ERROR]: Unable to open file -> %s\n", fileName.c_str());
    return success;
  }
  if (file < 0) return success;

  hid_t datasetID, dataspace;
  datasetID = H5Dopen2(file, datasetName.c_str(), H5P_DEFAULT);
  if (datasetID < 0) return success;
  dataspace = H5Dget_space(datasetID);
  // const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims[gpudata::MAXTABLEDIM];
  // int dummy = H5Sget_simple_extent_dims(dataspace,dims,NULL);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  // DenseMatrix memory is column-major, while HDF5 follows row-major.
  output.SetSize(dims[1], dims[0]);

  herr_t status;
  status = H5Dread(datasetID, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.HostReadWrite());
  if (status < 0) return success;
  H5Dclose(datasetID);
  H5Fclose(file);

  // DenseMatrix memory is column-major, while HDF5 follows row-major.
  output.Transpose();

  shape.SetSize(2);
  shape[0] = dims[0];
  shape[1] = dims[1];

  success = true;
  return success;
}

bool h5ReadBcastMultiColumnTable(const std::string &fileName, const std::string &datasetName, MPI_Comm TPSCommWorld,
                                 mfem::DenseMatrix &output, std::vector<TableInput> &tables) {
  int myrank;
  MPI_Comm_rank(TPSCommWorld, &myrank);
  const bool rank0 = (myrank == 0);

  int nrow = 0, ncol = 0;
  bool success = false;
  int suc_int;
  if (rank0) {
    Array<int> dims(2);
    success = h5ReadTable(fileName.c_str(), datasetName.c_str(), output, dims);
    nrow = dims[0];
    ncol = dims[1];
    suc_int = (int)success;
  }
  // MPI_Bcast(&success, 1, MPI_CXX_BOOL, 0, TPSCommWorld);
  MPI_Bcast(&suc_int, 1, MPI_INT, 0, TPSCommWorld);
  success = (suc_int != 0);
  if (!success) exit(ERROR);

  MPI_Bcast(&nrow, 1, MPI_INT, 0, TPSCommWorld);
  MPI_Bcast(&ncol, 1, MPI_INT, 0, TPSCommWorld);
  assert(nrow > 0);
  assert(ncol == (int)tables.size() + 1);

  if (!rank0) output.SetSize(nrow, ncol);
  double *h_table = output.HostReadWrite();
  MPI_Bcast(h_table, nrow * ncol, MPI_DOUBLE, 0, TPSCommWorld);

  for (size_t icol = 0; icol < tables.size(); icol++) {
    tables[icol].Ndata = nrow;
    tables[icol].xdata = output.HostRead();
    tables[icol].fdata = output.HostRead() + (icol + 1) * nrow;
  }
  return success;
}

void evaluateDistanceSerial(mfem::Mesh &mesh, const mfem::Array<int> &wall_patches, const mfem::GridFunction &coords,
                            mfem::GridFunction &distance) {
  distance = -1.0;  // Initialize with invalid data

  // Tolerance for solve
  const int iter_max = 20;
  const double rel_tol = 1e-10;
  const double abs_tol = 1e-16;

  Geometry geometry;
  double phys[3], dphys[3], ref[3], dref[3];
  double xloc[3];
  double residual[3];

  const int dim = mesh.Dimension();
  const int dof = distance.Size();

  auto h_dist = distance.HostWrite();
  auto h_coords = coords.HostRead();

  // For every node
  for (int i = 0; i < distance.Size(); i++) {
    // Get current point in physical space
    Vector xp(xloc, dim);
    for (int d = 0; d < dim; d++) {
      xp[d] = h_coords[d * dof + i];
    }

    // Evaluate distance to the wall...
    double dist_to_wall = 1e30;

    // Loop through mesh boundary elements
    for (int bel = 0; bel < mesh.GetNBE(); bel++) {
      const int attr = mesh.GetBdrAttribute(bel);

      bool is_wall = false;
      for (int ip = 0; ip < wall_patches.Size(); ip++) {
        if (attr == wall_patches[ip]) {
          is_wall = true;
          break;
        }
      }

      // If this boundary element is a wall, compute distance to it
      if (is_wall) {
        // For wall boundary faces, we solve for the position in the
        // reference element (of the face of course) that minimizes
        // the distance to the current point (xp).  If this point is
        // outside the face, we ignore it.  If it is inside the face,
        // the distance from xp to the point we found is computed.  If
        // that is less that the current minimum distance to the wall,
        // the wall distance is updated.

        // Get information about the current face
        FaceElementTransformations *Tr = mesh.GetBdrFaceTransformations(bel);
        const Geometry::Type gt = Tr->GetGeometryType();
        const int rdim = Tr->GetDimension();  // dim of reference space (should be dim-1)
        assert(rdim == dim - 1);

        // Below we do an approximate Newton solve to find the point
        // in the reference space that minimizes the distance to xp.
        // It is approximate because we neglect the Hessian of the map
        // from reference to physical space.  For a linear map, this
        // Hessian is zero and there is no approximation.  For a
        // nonlinear map, this method should generally still converge
        // as long as the curvature of the element is not too extreme.

        // Initial guess for the approximate Newton solve is the center of reference element
        IntegrationPoint ip(geometry.GetCenter(gt));
        Tr->SetIntPoint(&ip);

        // xx is location in physical space corresponding to ip
        Vector xx(phys, dim);
        Tr->Transform(ip, xx);

        // dx = vector from xx to xp
        Vector dx(dphys, dim);
        subtract(xp, xx, dx);

        // res = - J^T dx
        Vector res(residual, dim);
        res = 0.0;
        Tr->Jacobian().AddMultTranspose_a(-1.0, dx, res);

        const double r0 = res.Norml2();
        double rnorm = r0;

        int iter = 0;
        Vector dxi(dref, rdim);
        while (rnorm > abs_tol && rnorm / r0 > rel_tol && iter < iter_max) {
          // Solve for update to reference space coords.  Note that J
          // is (dim x rdim) so it is singular.  In this case,
          // InverseJacobian returns [J^t.J]^{-1}.J^t, which is
          // exactly what we need.
          Tr->InverseJacobian().Mult(dx, dxi);

          // Update reference coordinates
          Vector xi(ref, rdim);
          ip.Get(xi.HostWrite(), rdim);
          xi += dxi;
          ip.Set(xi.HostRead(), rdim);

          // Update dx and residual
          Tr->SetIntPoint(&ip);
          Tr->Transform(ip, xx);
          subtract(xp, xx, dx);

          res = 0.0;
          Tr->Jacobian().AddMultTranspose_a(-1.0, dx, res);

          rnorm = res.Norml2();

          iter++;
        }

        if (rnorm > abs_tol && rnorm / r0 > rel_tol) {
          std::cerr << "WARNING: Distance fcn Newton solve did not converge.  Results may be inaccurate." << std::endl;
          std::cerr << "  rnorm = " << rnorm << ", rnorm / r0 = " << rnorm / r0 << std::endl;
        }

        // If converged outside reference element, move back to closest point inside
        bool inside_ref_elem = Geometry::CheckPoint(gt, ip);
        if (!inside_ref_elem) {
          inside_ref_elem = Geometry::ProjectPoint(gt, ip);
        }

        double dist_to_face = 1e30;
        Tr->SetIntPoint(&ip);
        Tr->Transform(ip, xx);
        subtract(xp, xx, dx);
        dist_to_face = dx.Norml2();

        if (dist_to_face < 0.0) {
          std::cerr << "WARNING: The distance cannot be negative!" << std::endl;
        }

        // If this face has a closer point than see so far, update distance to wall
        if (dist_to_face < dist_to_wall) dist_to_wall = dist_to_face;
      }
    }
    // After looping over all wall faces, we've found the closest point
    h_dist[i] = dist_to_wall;
  }  // end loop over points in GridFunction for distance
}

void multConstScalar(double A, Vector B, Vector *C) {
  {
    double *dataB = B.HostReadWrite();
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, B.Size(), { dataC[i] = A * dataB[i]; });
  }
}

void multConstScalarInv(double A, Vector B, Vector *C) {
  {
    double *dataB = B.HostReadWrite();
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, B.Size(), { dataC[i] = A / dataB[i]; });
  }
}

void multConstVector(double A, Vector B, Vector *C) {
  {
    double *dataB = B.HostReadWrite();
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, B.Size(), { dataC[i] = A * dataB[i]; });
  }
}

void multConstScalarIP(double A, Vector *C) {
  {
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, C->Size(), { dataC[i] = A * dataC[i]; });
  }
}

void multConstScalarInvIP(double A, Vector *C) {
  {
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, C->Size(), { dataC[i] = A / dataC[i]; });
  }
}

// not necessary
void multConstVectorIP(double A, Vector *C) {
  {
    double *dataC = C->HostReadWrite();
    MFEM_FORALL(i, C->Size(), { dataC[i] = A * dataC[i]; });
  }
}

void multScalarScalar(Vector A, Vector B, Vector *C) {
  double *dataA = A.HostReadWrite();
  double *dataB = B.HostReadWrite();
  double *dataC = C->HostReadWrite();
  MFEM_FORALL(i, A.Size(), { dataC[i] = dataA[i] * dataB[i]; });
}

void multScalarVector(Vector A, Vector B, Vector *C, int dim) {
  int Ndof = A.Size();
  double *dataA = A.HostReadWrite();
  double *dataB = B.HostReadWrite();
  double *dataC = C->HostReadWrite();
  for (int eq = 0; eq < dim; eq++) {
    for (int i = 0; i < Ndof; i++) {
      dataC[i + eq * Ndof] = dataA[i] * dataB[i + eq * Ndof];
    }
  }
}

void multScalarInvVector(Vector A, Vector B, Vector *C, int dim) {
  int Ndof = A.Size();
  double *dataA = A.HostReadWrite();
  double *dataB = B.HostReadWrite();
  double *dataC = C->HostReadWrite();
  for (int eq = 0; eq < dim; eq++) {
    for (int i = 0; i < Ndof; i++) {
      dataC[i + eq * Ndof] = (1.0 / dataA[i]) * dataB[i + eq * Ndof];
    }
  }
}

void multScalarInvVectorIP(Vector A, Vector *C, int dim) {
  int Ndof = A.Size();
  double *dataA = A.HostReadWrite();
  double *dataC = C->HostReadWrite();
  for (int eq = 0; eq < dim; eq++) {
    for (int i = 0; i < Ndof; i++) {
      dataC[i + eq * Ndof] = (1.0 / dataA[i]) * dataC[i + eq * Ndof];
    }
  }
}

void multVectorVector(Vector A, Vector B, Vector *C1, Vector *C2, Vector *C3, int dim) {
  {
    int Ndof = A.Size() / dim;
    double *dataA = A.HostReadWrite();
    double *dataB = B.HostReadWrite();
    double *dataC = C1->HostReadWrite();
    for (int eq = 0; eq < dim; eq++) {
      for (int i = 0; i < Ndof; i++) {
        dataC[i + eq * Ndof] = dataA[i + 0 * Ndof] * dataB[i + eq * Ndof];
      }
    }
  }

  {
    int Ndof = A.Size() / dim;
    double *dataA = A.HostReadWrite();
    double *dataB = B.HostReadWrite();
    double *dataC = C2->HostReadWrite();
    for (int eq = 0; eq < dim; eq++) {
      for (int i = 0; i < Ndof; i++) {
        dataC[i + eq * Ndof] = dataA[i + 1 * Ndof] * dataB[i + eq * Ndof];
      }
    }
  }

  {
    int Ndof = A.Size() / dim;
    double *dataA = A.HostReadWrite();
    double *dataB = B.HostReadWrite();
    double *dataC = C3->HostReadWrite();
    for (int eq = 0; eq < dim; eq++) {
      for (int i = 0; i < Ndof; i++) {
        dataC[i + eq * Ndof] = dataA[i + 2 * Ndof] * dataB[i + eq * Ndof];
      }
    }
  }
}

void dotVector(Vector A, Vector B, Vector *C, int dim) {
  int Ndof = A.Size() / dim;
  double *dataA = A.HostReadWrite();
  double *dataB = B.HostReadWrite();
  double *data = C->HostReadWrite();
  for (int i = 0; i < Ndof; i++) {
    data[i] = 0.0;
  }
  for (int eq = 0; eq < dim; eq++) {
    for (int i = 0; i < Ndof; i++) {
      data[i] += dataA[i + eq * Ndof] * dataB[i + eq * Ndof];
    }
  }
}

void multScalarScalarIP(Vector A, Vector *C) {
  double *dataA = A.HostReadWrite();
  double *dataB = C->HostReadWrite();
  MFEM_FORALL(i, A.Size(), { dataB[i] = dataA[i] * dataB[i]; });
}

void multScalarInvScalarIP(Vector A, Vector *C) {
  double *dataA = A.HostReadWrite();
  double *dataB = C->HostReadWrite();
  MFEM_FORALL(i, A.Size(), { dataB[i] = 1.0 / dataA[i] * dataB[i]; });
}

void multScalarVectorIP(Vector A, Vector *C, int dim) {
  int Ndof = A.Size();
  double *dataA = A.HostReadWrite();
  double *dataB = C->HostReadWrite();
  for (int eq = 0; eq < dim; eq++) {
    for (int i = 0; i < Ndof; i++) {
      dataB[i + eq * Ndof] = dataA[i] * dataB[i + eq * Ndof];
    }
  }
}

void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu) {
  FiniteElementSpace *fes = u.FESpace();

  // AccumulateAndCountZones.
  Array<int> zones_per_vdof;
  zones_per_vdof.SetSize(fes->GetVSize());
  zones_per_vdof = 0;

  cu = 0.0;

  // Local interpolation.
  int elndofs;
  Array<int> vdofs;
  Vector vals;
  Vector loc_data;
  int vdim = fes->GetVDim();
  DenseMatrix grad_hat;
  DenseMatrix dshape;
  DenseMatrix grad;
  Vector curl;

  for (int e = 0; e < fes->GetNE(); ++e) {
    fes->GetElementVDofs(e, vdofs);
    u.GetSubVector(vdofs, loc_data);
    vals.SetSize(vdofs.Size());
    ElementTransformation *tr = fes->GetElementTransformation(e);
    const FiniteElement *el = fes->GetFE(e);
    elndofs = el->GetDof();
    int dim = el->GetDim();
    dshape.SetSize(elndofs, dim);

    for (int dof = 0; dof < elndofs; ++dof) {
      // Project.
      const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
      tr->SetIntPoint(&ip);

      // Eval and GetVectorGradientHat.
      el->CalcDShape(tr->GetIntPoint(), dshape);
      grad_hat.SetSize(vdim, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
      MultAtB(loc_data_mat, dshape, grad_hat);

      const DenseMatrix &Jinv = tr->InverseJacobian();
      grad.SetSize(grad_hat.Height(), Jinv.Width());
      Mult(grad_hat, Jinv, grad);

      curl.SetSize(3);
      curl(0) = grad(2, 1) - grad(1, 2);
      curl(1) = grad(0, 2) - grad(2, 0);
      curl(2) = grad(1, 0) - grad(0, 1);

      for (int j = 0; j < curl.Size(); ++j) {
        vals(elndofs * j + dof) = curl(j);
      }
    }

    // Accumulate values in all dofs, count the zones.
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      cu(ldof) += vals[j];
      zones_per_vdof[ldof]++;
    }
  }

  // Communication

  // Count the zones globally.
  GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
  gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
  gcomm.Bcast(zones_per_vdof);

  // Accumulate for all vdofs.
  gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
  gcomm.Bcast<double>(cu.GetData());

  // Compute means.
  for (int i = 0; i < cu.Size(); i++) {
    const int nz = zones_per_vdof[i];
    if (nz) {
      cu(i) /= nz;
    }
  }
}

void vectorGrad3D(ParGridFunction &u, ParGridFunction &gu, ParGridFunction &gv, ParGridFunction &gw) {
  ParGridFunction uSub;
  FiniteElementSpace *sfes = u.FESpace();
  uSub.SetSpace(sfes);
  int nSize = uSub.Size();

  {
    double *dataSub = uSub.HostReadWrite();
    double *data = u.HostReadWrite();
    for (int i = 0; i < nSize; i++) {
      dataSub[i] = data[i + 0 * nSize];
    }
  }
  scalarGrad3D(uSub, gu);

  {
    double *dataSub = uSub.HostReadWrite();
    double *data = u.HostReadWrite();
    for (int i = 0; i < nSize; i++) {
      dataSub[i] = data[i + 1 * nSize];
    }
  }
  scalarGrad3D(uSub, gv);

  {
    double *dataSub = uSub.HostReadWrite();
    double *data = u.HostReadWrite();
    for (int i = 0; i < nSize; i++) {
      dataSub[i] = data[i + 2 * nSize];
    }
  }
  scalarGrad3D(uSub, gw);
}

void scalarGrad3D(ParGridFunction &u, ParGridFunction &gu) {
  FiniteElementSpace *fes = u.FESpace();

  // AccumulateAndCountZones.
  Array<int> zones_per_vdof;
  zones_per_vdof.SetSize(3 * (fes->GetVSize()));
  zones_per_vdof = 0;

  gu = 0.0;
  int nSize = u.Size();

  // Local interpolation.
  int elndofs;
  Array<int> vdofs;
  Vector vals1, vals2, vals3;
  Vector loc_data;
  int vdim = fes->GetVDim();
  DenseMatrix grad_hat;
  DenseMatrix dshape;
  DenseMatrix grad;
  int dim_;

  // element loop
  for (int e = 0; e < fes->GetNE(); ++e) {
    fes->GetElementVDofs(e, vdofs);
    u.GetSubVector(vdofs, loc_data);
    vals1.SetSize(vdofs.Size());
    vals2.SetSize(vdofs.Size());
    vals3.SetSize(vdofs.Size());
    ElementTransformation *tr = fes->GetElementTransformation(e);
    const FiniteElement *el = fes->GetFE(e);
    elndofs = el->GetDof();
    int dim = el->GetDim();
    dim_ = dim;
    dshape.SetSize(elndofs, dim);

    // element dof
    for (int dof = 0; dof < elndofs; ++dof) {
      // Project.
      const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
      tr->SetIntPoint(&ip);

      // Eval and GetVectorGradientHat.
      el->CalcDShape(tr->GetIntPoint(), dshape);
      grad_hat.SetSize(vdim, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, 1);
      MultAtB(loc_data_mat, dshape, grad_hat);

      const DenseMatrix &Jinv = tr->InverseJacobian();
      grad.SetSize(grad_hat.Height(), Jinv.Width());
      Mult(grad_hat, Jinv, grad);

      vals1(dof) = grad(0, 0);
      vals2(dof) = grad(0, 1);
      vals3(dof) = grad(0, 2);
    }

    // Accumulate values in all dofs, count the zones.
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      gu(ldof + 0 * nSize) += vals1[j];
    }
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      gu(ldof + 1 * nSize) += vals2[j];
    }
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      gu(ldof + 2 * nSize) += vals3[j];
    }

    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      zones_per_vdof[ldof]++;
    }
  }

  // Count the zones globally.
  GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
  gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
  gcomm.Bcast(zones_per_vdof);

  // Accumulate for all vdofs.
  gcomm.Reduce<double>(gu.GetData(), GroupCommunicator::Sum);
  gcomm.Bcast<double>(gu.GetData());

  // Compute means.
  for (int dir = 0; dir < dim_; dir++) {
    for (int i = 0; i < u.Size(); i++) {
      const int nz = zones_per_vdof[i];
      if (nz) {
        gu(i + dir * nSize) /= nz;
      }
    }
  }
}

void ComputeCurl2D(ParGridFunction &u, ParGridFunction &cu, bool assume_scalar) {
  FiniteElementSpace *fes = u.FESpace();

  // AccumulateAndCountZones.
  Array<int> zones_per_vdof;
  zones_per_vdof.SetSize(fes->GetVSize());
  zones_per_vdof = 0;

  cu = 0.0;

  // Local interpolation.
  int elndofs;
  Array<int> vdofs;
  Vector vals;
  Vector loc_data;
  int vdim = fes->GetVDim();
  DenseMatrix grad_hat;
  DenseMatrix dshape;
  DenseMatrix grad;
  Vector curl;

  for (int e = 0; e < fes->GetNE(); ++e) {
    fes->GetElementVDofs(e, vdofs);
    u.GetSubVector(vdofs, loc_data);
    vals.SetSize(vdofs.Size());
    ElementTransformation *tr = fes->GetElementTransformation(e);
    const FiniteElement *el = fes->GetFE(e);
    elndofs = el->GetDof();
    int dim = el->GetDim();
    dshape.SetSize(elndofs, dim);

    for (int dof = 0; dof < elndofs; ++dof) {
      // Project.
      const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
      tr->SetIntPoint(&ip);

      // Eval and GetVectorGradientHat.
      el->CalcDShape(tr->GetIntPoint(), dshape);
      grad_hat.SetSize(vdim, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
      MultAtB(loc_data_mat, dshape, grad_hat);

      const DenseMatrix &Jinv = tr->InverseJacobian();
      grad.SetSize(grad_hat.Height(), Jinv.Width());
      Mult(grad_hat, Jinv, grad);

      if (assume_scalar) {
        curl.SetSize(2);
        curl(0) = grad(0, 1);
        curl(1) = -grad(0, 0);
      } else {
        curl.SetSize(2);
        curl(0) = grad(1, 0) - grad(0, 1);
        curl(1) = 0.0;
      }

      for (int j = 0; j < curl.Size(); ++j) {
        vals(elndofs * j + dof) = curl(j);
      }
    }

    // Accumulate values in all dofs, count the zones.
    for (int j = 0; j < vdofs.Size(); j++) {
      int ldof = vdofs[j];
      cu(ldof) += vals[j];
      zones_per_vdof[ldof]++;
    }
  }

  // Communication.

  // Count the zones globally.
  GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
  gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
  gcomm.Bcast(zones_per_vdof);

  // Accumulate for all vdofs.
  gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
  gcomm.Bcast<double>(cu.GetData());

  // Compute means.
  for (int i = 0; i < cu.Size(); i++) {
    const int nz = zones_per_vdof[i];
    if (nz) {
      cu(i) /= nz;
    }
  }
}

void scalarGrad3DV(FiniteElementSpace *fes, FiniteElementSpace *vfes, Vector u, Vector *gu) {
  ParGridFunction R0_gf;
  R0_gf.SetSpace(fes);

  ParGridFunction R1_gf;
  R1_gf.SetSpace(vfes);

  R0_gf.SetFromTrueDofs(u);
  scalarGrad3D(R0_gf, R1_gf);
  R1_gf.GetTrueDofs(*gu);
}

void vectorGrad3DV(FiniteElementSpace *fes, Vector u, Vector *gu, Vector *gv, Vector *gw) {
  ParGridFunction R1_gf;
  R1_gf.SetSpace(fes);

  ParGridFunction R1a_gf, R1b_gf, R1c_gf;
  R1a_gf.SetSpace(fes);
  R1b_gf.SetSpace(fes);
  R1c_gf.SetSpace(fes);

  R1_gf.SetFromTrueDofs(u);
  vectorGrad3D(R1_gf, R1a_gf, R1b_gf, R1c_gf);
  R1a_gf.GetTrueDofs(*gu);
  R1b_gf.GetTrueDofs(*gv);
  R1c_gf.GetTrueDofs(*gw);
}

void EliminateRHS(Operator &A, ConstrainedOperator &constrainedA, const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                  Vector &X, Vector &B, int copy_interior) {
  const Operator *Po = A.GetOutputProlongation();
  const Operator *Pi = A.GetProlongation();
  const Operator *Ri = A.GetRestriction();
  A.InitTVectors(Po, Ri, Pi, x, b, X, B);
  if (!copy_interior) {
    X.SetSubVectorComplement(ess_tdof_list, 0.0);
  }
  constrainedA.EliminateRHS(X, B);
}

void Orthogonalize(Vector &v, MPI_Comm comm) {
  double loc_sum = v.Sum();
  double global_sum = 0.0;
  int loc_size = v.Size();
  int global_size = 0;

  MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

  v -= global_sum / static_cast<double>(global_size);
}
