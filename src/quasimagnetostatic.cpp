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
#include "quasimagnetostatic.hpp"

#include <hdf5.h>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "logger.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

void JFun(const Vector &x, Vector &f);

QuasiMagnetostaticSolver::QuasiMagnetostaticSolver(MPI_Session &mpi, ElectromagneticOptions em_opts, TPS::Tps *tps)
    : mpi_(mpi), em_opts_(em_opts) {
  tpsP_ = tps;

  // verify running on cpu
  if (tpsP_->getDeviceConfig() != "cpu") {
    if (mpi.Root()) {
      grvy_printf(GRVY_ERROR, "[ERROR] EM simulation currently only supported on cpu.\n");
    }
    exit(1);
  }

  pmesh_ = NULL;
  hcurl_ = NULL;
  h1_ = NULL;
  hdiv_ = NULL;
  Aspace_ = NULL;
  pspace_ = NULL;
  Bspace_ = NULL;
  K_ = NULL;
  r_ = NULL;
  A_ = NULL;
  B_ = NULL;

  operator_initialized_ = false;
  current_initialized_ = false;
}

QuasiMagnetostaticSolver::~QuasiMagnetostaticSolver() {
  delete B_;
  delete A_;
  delete r_;
  delete K_;
  delete Bspace_;
  delete pspace_;
  delete Aspace_;
  delete hdiv_;
  delete h1_;
  delete hcurl_;
  delete pmesh_;
}

void QuasiMagnetostaticSolver::initialize() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing quasimagnetostatic solver.\n");

  // Prepare for the quasi-magnetostatic solve in four steps...

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // 1a) Read the serial mesh (on each mpi rank)
  Mesh *mesh = new Mesh(em_opts_.mesh_file.c_str(), 1, 1);
  dim_ = mesh->Dimension();
  if (dim_ != 3) {
    if (verbose) {
      grvy_printf(gerror, "[ERROR] Quasi-magnetostatic solver on supported for 3D.");
    }
    exit(1);
  }

  // 1b) Refine the serial mesh, if requested
  if (verbose && (em_opts_.ref_levels > 0)) {
    grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", em_opts_.ref_levels);
  }
  for (int l = 0; l < em_opts_.ref_levels; l++) {
    mesh->UniformRefinement();
  }

  // 1c) Partition the mesh
  pmesh_ = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;  // no longer need the serial mesh
  // pmesh_->ReorientTetMesh();

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------
  hcurl_ = new ND_FECollection(em_opts_.order, dim_);
  h1_ = new H1_FECollection(em_opts_.order, dim_);
  hdiv_ = new RT_FECollection(em_opts_.order - 1, dim_);

  Aspace_ = new ParFiniteElementSpace(pmesh_, hcurl_);
  pspace_ = new ParFiniteElementSpace(pmesh_, h1_);
  Bspace_ = new ParFiniteElementSpace(pmesh_, hdiv_);

  //-----------------------------------------------------
  // 3) Get BC dofs (everything is essential---i.e., PEC)
  //-----------------------------------------------------
  Array<int> ess_bdr(pmesh_->bdr_attributes.Max());
  if (pmesh_->bdr_attributes.Size()) {
    ess_bdr = 1;
  }

  Aspace_->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs_);

  //-----------------------------------------------------
  // 4) Form curl-curl bilinear form
  //-----------------------------------------------------
  K_ = new ParBilinearForm(Aspace_);
  K_->AddDomainIntegrator(new CurlCurlIntegrator);
  K_->Assemble();
  K_->Finalize();

  // All done
  operator_initialized_ = true;

  // initialize current
  InitializeCurrent();
}

void QuasiMagnetostaticSolver::InitializeCurrent() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing rhs using source current.\n");

  // ensure we've initialized the operators
  // b/c we need the mesh, spaces, etc here
  assert(operator_initialized_);

  // ensure the mesh volume attributes conform to our expectations
  assert(pmesh_->attributes.Max() == 5);

  // Compute the right hand side in 3 steps...

  // 1) Build the current function For now, this is hardcoded to be
  //    uniformly distributed current density in rings defined by
  //    volume attributes in the mesh.  We assume 4 rings, plus a zero
  //    source current domain for 5 total attributes.
  Vector J0(pmesh_->attributes.Max());
  J0 = 0.0;
  if (em_opts_.bot_only) {
    // only bottom rings
    J0(1) = J0(2) = 1.0;
    J0(3) = J0(4) = 0.0;
  } else if (em_opts_.top_only) {
    // only top rings
    J0(1) = J0(2) = 0.0;
    J0(3) = J0(4) = 1.0;
  } else {
    // all rings
    J0(1) = J0(2) = 1.0;
    J0(3) = J0(4) = 1.0;
  }

  PWConstCoefficient J0coef(J0);
  VectorFunctionCoefficient current(dim_, JFun, &J0coef);

  // 2) Build a discretely divergence-free approximation of the source
  // current that lives in the Nedelec FE space defined in
  // Initialize()
  ParGridFunction *Jorig = new ParGridFunction(Aspace_);
  ParGridFunction *Jproj = new ParGridFunction(Aspace_);

  r_ = new ParLinearForm(Aspace_);

  int irOrder = pspace_->GetElementTransformation(0)->OrderW() + 2 * em_opts_.order;
  ParDiscreteGradOperator *grad = new ParDiscreteGradOperator(pspace_, Aspace_);
  grad->Assemble();
  grad->Finalize();

  DivergenceFreeProjector *div_free = new DivergenceFreeProjector(*pspace_, *Aspace_, irOrder, NULL, NULL, grad);

  // This call (i.e., GlobalProjectDiscCoefficient) replaces the
  // functionality of Jorig->ProjectCoefficient(current) in a way that
  // gives the same results in parallel for discontinuous functions.
  // Specifically, it handles dofs that are shared between multiple
  // elements by using the value from the element with the largest
  // attribute.  This reproduces the functionality of
  // mfem::ParGridFunction::ProjectDiscCoefficient but in a way that
  // works for Nedelec elements.  Specifically, it deals with the case
  // where FiniteElementSpace::GetElementVDofs has negative dof
  // indices.  In mfem v4.3 and earlier, this situation leads to
  // failed asserts when accessing elements of the array dof_attr in
  // GridFunction::ProjectDiscCoefficient (or worse, silently runs
  // incorrectly).
  GlobalProjectDiscCoefficient(*Jorig, current);
  div_free->Mult(*Jorig, *Jproj);

  delete div_free;
  delete grad;
  delete Jorig;

  // 3) Multiply by the mass matrix to get the RHS vector
  ParBilinearForm *mass = new ParBilinearForm(Aspace_);
  mass->AddDomainIntegrator(new VectorFEMassIntegrator);
  mass->Assemble();
  mass->Finalize();

  mass->Mult(*Jproj, *r_);

  delete mass;
  delete Jproj;

  current_initialized_ = true;
}

// query solver-specific runtime controls
void QuasiMagnetostaticSolver::parseSolverOptions() {
  tpsP_->getRequiredInput("em/mesh", em_opts_.mesh_file);

  tpsP_->getInput("em/order", em_opts_.order, 1);
  tpsP_->getInput("em/ref_levels", em_opts_.ref_levels, 0);
  tpsP_->getInput("em/max_iter", em_opts_.max_iter, 100);
  tpsP_->getInput("em/rtol", em_opts_.rtol, 1.0e-6);
  tpsP_->getInput("em/atol", em_opts_.atol, 1.0e-10);
  tpsP_->getInput("em/nBy", em_opts_.nBy, 0);
  tpsP_->getInput("em/yinterp_min", em_opts_.yinterp_min, 0.0);
  tpsP_->getInput("em/yinterp_max", em_opts_.yinterp_max, 1.0);
  tpsP_->getInput("em/By_file", em_opts_.By_file, std::string("By.h5"));
  tpsP_->getInput("em/top_only", em_opts_.top_only, false);
  tpsP_->getInput("em/bot_only", em_opts_.bot_only, false);

  // dump options to screen for user inspection
  if (mpi_.Root()) {
    em_opts_.print(std::cout);
  }
}

void QuasiMagnetostaticSolver::solve() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Solving the quasi-magnetostatic system for A (magnetic vector potential).\n");

  assert(operator_initialized_ && current_initialized_);

  // Solve for the magnetic vector potential and magnetic field in 3 steps

  // 1) Solve the curl(curl(A)) = J for magnetic vector potential
  //    using operators set up by Initialize() and InitializeCurrent()
  //    fcns
  A_ = new ParGridFunction(Aspace_);
  *A_ = 0;

  HypreParMatrix K;
  HypreParVector x(Aspace_);
  HypreParVector b(Aspace_);

  K_->FormLinearSystem(ess_bdr_tdofs_, *A_, *r_, K, x, b);

  // Set up the preconditioner
  HypreAMS *iK;
  iK = new HypreAMS(K, Aspace_);
  iK->SetSingularProblem();
  iK->iterative_mode = false;

  MINRESSolver solver(MPI_COMM_WORLD);
  solver.SetAbsTol(em_opts_.atol);
  solver.SetRelTol(em_opts_.rtol);
  solver.SetMaxIter(em_opts_.max_iter);
  solver.SetOperator(K);
  solver.SetPreconditioner(*iK);
  solver.SetPrintLevel(1);
  solver.Mult(b, x);
  delete iK;

  K_->RecoverFEMSolution(x, *r_, *A_);

  // 2) Determine B from A according to B = curl(A).  Here we use an
  //    interpolator rather than a projection.
  if (verbose) grvy_printf(ginfo, "Evaluating curl(A) to get the magnetic field.\n");
  B_ = new ParGridFunction(Bspace_);
  *B_ = 0;

  ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(Aspace_, Bspace_);
  curl->AddDomainInterpolator(new CurlInterpolator);
  curl->Assemble();
  curl->Finalize();
  curl->Mult(*A_, *B_);
  delete curl;

  // 3) Output A and B fields for visualization using paraview
  if (verbose) grvy_printf(ginfo, "Writing solution to paraview output.\n");
  ParaViewDataCollection paraview_dc("magnetostatic", pmesh_);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(em_opts_.order);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("magvecpot", A_);
  paraview_dc.RegisterField("magnfield", B_);
  paraview_dc.Save();

  // Compute and dump the magnetic field on the axis
  InterpolateToYAxis();

  if (mpi_.Root()) {
    std::cout << "EM simulation complete" << std::endl;
  }
}

void QuasiMagnetostaticSolver::InterpolateToYAxis() const {
  const bool root = mpi_.Root();

  // quick return if there are no interpolation points
  if (em_opts_.nBy < 1) return;

  // Set up array of points (on all ranks)
  DenseMatrix phys_points(dim_, em_opts_.nBy);

  const double dy = (em_opts_.yinterp_max - em_opts_.yinterp_min) / (em_opts_.nBy - 1);

  for (int ipt = 0; ipt < em_opts_.nBy; ipt++) {
    phys_points(0, ipt) = 0.0;
    phys_points(1, ipt) = em_opts_.yinterp_min + ipt * dy;
    phys_points(2, ipt) = 0.0;
  }

  Array<int> eid;
  Array<IntegrationPoint> ips;
  double *Byloc = new double[em_opts_.nBy];
  double *By = new double[em_opts_.nBy];
  Vector Bpoint(dim_);

  // Get element numbers and integration points
  pmesh_->FindPoints(phys_points, eid, ips);

  // And interpolate
  for (int ipt = 0; ipt < em_opts_.nBy; ipt++) {
    if (eid[ipt] >= 0) {
      B_->GetVectorValue(eid[ipt], ips[ipt], Bpoint);
      Byloc[ipt] = Bpoint[1];
    } else {
      Byloc[ipt] = 0;
    }
    MPI_Reduce(Byloc, By, em_opts_.nBy, MPI_DOUBLE, MPI_SUM, 0, pmesh_->GetComm());
  }

  // Finally, write the result to an hdf5 file
  hid_t file = -1;
  hid_t data_soln;
  herr_t status;
  hsize_t dims[1];
  hid_t group = -1;
  hid_t dataspace = -1;

  if (root) {
    file = H5Fcreate(em_opts_.By_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0);

    h5_save_attribute(file, "nBy", em_opts_.nBy);

    dims[0] = em_opts_.nBy;

    // write y locations of points
    dataspace = H5Screate_simple(1, dims, NULL);
    assert(dataspace >= 0);
    group = H5Gcreate(file, "Points", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(group >= 0);

    data_soln = H5Dcreate2(group, "y", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(data_soln >= 0);

    Vector yval;
    phys_points.GetRow(1, yval);

    status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, yval.GetData());
    assert(status >= 0);
    H5Dclose(data_soln);

    H5Gclose(group);
    H5Sclose(dataspace);

    // write y component of B field
    dataspace = H5Screate_simple(1, dims, NULL);
    assert(dataspace >= 0);
    group = H5Gcreate(file, "Magnetic-field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(group >= 0);

    data_soln = H5Dcreate2(group, "y", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(data_soln >= 0);

    status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, By);
    assert(status >= 0);
    H5Dclose(data_soln);

    H5Gclose(group);
    H5Sclose(dataspace);

    H5Fclose(file);
  }

  delete Byloc;
  delete By;
}

void JFun(const Vector &x, Vector &J) {
  Vector a(3);
  a(0) = 0.0;
  a(1) = 1.0;
  a(2) = 0.0;

  Vector axx(3);

  axx(0) = a(1) * x(2) - a(2) * x(1);
  axx(1) = a(2) * x(0) - a(0) * x(2);
  axx(2) = a(0) * x(1) - a(1) * x(0);
  axx /= axx.Norml2();

  J = axx;
}
