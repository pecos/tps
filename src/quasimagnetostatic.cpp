// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
    : _mpi(mpi), _em_opts(em_opts) {
  tpsP = tps;

  // verify running on cpu
  if (tpsP->getDeviceConfig() != "cpu") {
    if (mpi.Root()) {
      grvy_printf(GRVY_ERROR, "[ERROR] EM simulation currently only supported on cpu.\n");
    }
    exit(1);
  }

  _pmesh = NULL;
  _hcurl = NULL;
  _h1 = NULL;
  _hdiv = NULL;
  _Aspace = NULL;
  _pspace = NULL;
  _Bspace = NULL;
  _K = NULL;
  _A = NULL;
  _B = NULL;

  _operator_initialized = false;
  _current_initialized = false;
}

QuasiMagnetostaticSolver::~QuasiMagnetostaticSolver() {
  delete _B;
  delete _A;
  delete _K;
  delete _Bspace;
  delete _pspace;
  delete _Aspace;
  delete _hdiv;
  delete _h1;
  delete _hcurl;
  delete _pmesh;
}

void QuasiMagnetostaticSolver::initialize() {
  bool verbose = _mpi.Root();
  if (verbose) grvy_printf(ginfo, "Initializing quasimagnetostatic solver.\n");

  // Prepare for the quasi-magnetostatic solve in four steps...

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // 1a) Read the serial mesh (on each mpi rank)
  Mesh *mesh = new Mesh(_em_opts.mesh_file.c_str(), 1, 1);
  _dim = mesh->Dimension();
  if (_dim != 3) {
    if (verbose) {
      grvy_printf(gerror, "[ERROR] Quasi-magnetostatic solver on supported for 3D.");
    }
    exit(1);
  }

  // 1b) Refine the serial mesh, if requested
  if (verbose && (_em_opts.ref_levels > 0)) {
    grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", _em_opts.ref_levels);
  }
  for (int l = 0; l < _em_opts.ref_levels; l++) {
    mesh->UniformRefinement();
  }

  // 1c) Partition the mesh
  _pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;  // no longer need the serial mesh
  _pmesh->ReorientTetMesh();

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------
  _hcurl = new ND_FECollection(_em_opts.order, _dim);
  _h1 = new H1_FECollection(_em_opts.order, _dim);
  _hdiv = new RT_FECollection(_em_opts.order - 1, _dim);

  _Aspace = new ParFiniteElementSpace(_pmesh, _hcurl);
  _pspace = new ParFiniteElementSpace(_pmesh, _h1);
  _Bspace = new ParFiniteElementSpace(_pmesh, _hdiv);

  //-----------------------------------------------------
  // 3) Get BC dofs (everything is essential---i.e., PEC)
  //-----------------------------------------------------
  Array<int> ess_bdr(_pmesh->bdr_attributes.Max());
  if (_pmesh->bdr_attributes.Size()) {
    ess_bdr = 1;
  }

  _Aspace->GetEssentialTrueDofs(ess_bdr, _ess_bdr_tdofs);

  //-----------------------------------------------------
  // 4) Form curl-curl bilinear form
  //-----------------------------------------------------
  _K = new ParBilinearForm(_Aspace);
  _K->AddDomainIntegrator(new CurlCurlIntegrator);
  _K->Assemble();
  _K->Finalize();

  // All done
  _operator_initialized = true;

  // initialize current
  InitializeCurrent();
}

void QuasiMagnetostaticSolver::InitializeCurrent() {
  bool verbose = _mpi.Root();
  if (verbose) grvy_printf(ginfo, "Initializing rhs using source current.\n");

  // ensure we've initialized the operators
  // b/c we need the mesh, spaces, etc here
  assert(_operator_initialized);

  // ensure the mesh volume attributes conform to our expectations
  assert(_pmesh->attributes.Max() == 5);

  // Compute the right hand side in 3 steps...

  // 1) Build the current function For now, this is hardcoded to be
  //    uniformly distributed current density in rings defined by
  //    volume attributes in the mesh.  We assume 4 rings, plus a zero
  //    source current domain for 5 total attributes.
  Vector J0(_pmesh->attributes.Max());
  J0 = 0.0;
  if (_em_opts.bot_only) {
    // only bottom rings
    J0(1) = J0(2) = 1.0;
    J0(3) = J0(4) = 0.0;
  } else if (_em_opts.top_only) {
    // only top rings
    J0(1) = J0(2) = 0.0;
    J0(3) = J0(4) = 1.0;
  } else {
    // all rings
    J0(1) = J0(2) = 1.0;
    J0(3) = J0(4) = 1.0;
  }

  PWConstCoefficient J0coef(J0);
  VectorFunctionCoefficient current(_dim, JFun, &J0coef);

  // 2) Build a discretely divergence-free approximation of the source
  // current that lives in the Nedelec FE space defined in
  // Initialize()
  ParGridFunction *Jorig = new ParGridFunction(_Aspace);
  ParGridFunction *Jproj = new ParGridFunction(_Aspace);

  _r = new ParLinearForm(_Aspace);

  int irOrder = _pspace->GetElementTransformation(0)->OrderW() + 2 * _em_opts.order;
  ParDiscreteGradOperator *grad = new ParDiscreteGradOperator(_pspace, _Aspace);
  grad->Assemble();
  grad->Finalize();

  DivergenceFreeProjector *div_free = new DivergenceFreeProjector(*_pspace, *_Aspace, irOrder, NULL, NULL, grad);

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
  ParBilinearForm *mass = new ParBilinearForm(_Aspace);
  mass->AddDomainIntegrator(new VectorFEMassIntegrator);
  mass->Assemble();
  mass->Finalize();

  mass->Mult(*Jproj, *_r);

  delete mass;
  delete Jproj;

  _current_initialized = true;
}

// query solver-specific runtime controls
void QuasiMagnetostaticSolver::parseSolverOptions() {
  tpsP->getRequiredInput("em/mesh", _em_opts.mesh_file);

  tpsP->getInput("em/order", _em_opts.order, 1);
  tpsP->getInput("em/ref_levels", _em_opts.ref_levels, 0);
  tpsP->getInput("em/max_iter", _em_opts.max_iter, 100);
  tpsP->getInput("em/rtol", _em_opts.rtol, 1.0e-6);
  tpsP->getInput("em/atol", _em_opts.atol, 1.0e-10);
  tpsP->getInput("em/nBy", _em_opts.nBy, 0);
  tpsP->getInput("em/yinterp_min", _em_opts.yinterp_min, 0.0);
  tpsP->getInput("em/yinterp_max", _em_opts.yinterp_max, 1.0);
  tpsP->getInput("em/By_file", _em_opts.By_file, std::string("By.h5"));
  tpsP->getInput("em/top_only", _em_opts.top_only, false);
  tpsP->getInput("em/bot_only", _em_opts.bot_only, false);

  // dump options to screen for user inspection
  if (_mpi.Root()) {
    _em_opts.print(std::cout);
  }
}

void QuasiMagnetostaticSolver::solve() {
  bool verbose = _mpi.Root();
  if (verbose) grvy_printf(ginfo, "Solving the quasi-magnetostatic system for A (magnetic vector potential).\n");

  assert(_operator_initialized && _current_initialized);

  // Solve for the magnetic vector potential and magnetic field in 3 steps

  // 1) Solve the curl(curl(A)) = J for magnetic vector potential
  //    using operators set up by Initialize() and InitializeCurrent()
  //    fcns
  _A = new ParGridFunction(_Aspace);
  *_A = 0;

  HypreParMatrix K;
  HypreParVector x(_Aspace);
  HypreParVector b(_Aspace);

  _K->FormLinearSystem(_ess_bdr_tdofs, *_A, *_r, K, x, b);

  // Set up the preconditioner
  HypreAMS *iK;
  iK = new HypreAMS(K, _Aspace);
  iK->SetSingularProblem();
  iK->iterative_mode = false;

  MINRESSolver solver(MPI_COMM_WORLD);
  solver.SetAbsTol(_em_opts.atol);
  solver.SetRelTol(_em_opts.rtol);
  solver.SetMaxIter(_em_opts.max_iter);
  solver.SetOperator(K);
  solver.SetPreconditioner(*iK);
  solver.SetPrintLevel(1);
  solver.Mult(b, x);
  delete iK;

  _K->RecoverFEMSolution(x, *_r, *_A);

  // 2) Determine B from A according to B = curl(A).  Here we use an
  //    interpolator rather than a projection.
  if (verbose) grvy_printf(ginfo, "Evaluating curl(A) to get the magnetic field.\n");
  _B = new ParGridFunction(_Bspace);
  *_B = 0;

  ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(_Aspace, _Bspace);
  curl->AddDomainInterpolator(new CurlInterpolator);
  curl->Assemble();
  curl->Finalize();
  curl->Mult(*_A, *_B);
  delete curl;

  // 3) Output A and B fields for visualization using paraview
  if (verbose) grvy_printf(ginfo, "Writing solution to paraview output.\n");
  ParaViewDataCollection paraview_dc("magnetostatic", _pmesh);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(_em_opts.order);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("magvecpot", _A);
  paraview_dc.RegisterField("magnfield", _B);
  paraview_dc.Save();

  // Compute and dump the magnetic field on the axis
  InterpolateToYAxis();

  if (_mpi.Root()) {
    std::cout << "EM simulation complete" << std::endl;
  }
}

void QuasiMagnetostaticSolver::InterpolateToYAxis() const {
  const bool root = _mpi.Root();

  // quick return if there are no interpolation points
  if (_em_opts.nBy < 1) return;

  // Set up array of points (on all ranks)
  DenseMatrix phys_points(_dim, _em_opts.nBy);

  const double dy = (_em_opts.yinterp_max - _em_opts.yinterp_min) / (_em_opts.nBy - 1);

  for (int ipt = 0; ipt < _em_opts.nBy; ipt++) {
    phys_points(0, ipt) = 0.0;
    phys_points(1, ipt) = _em_opts.yinterp_min + ipt * dy;
    phys_points(2, ipt) = 0.0;
  }

  Array<int> eid;
  Array<IntegrationPoint> ips;
  double *Byloc = new double[_em_opts.nBy];
  double *By = new double[_em_opts.nBy];
  Vector Bpoint(_dim);

  // Get element numbers and integration points
  _pmesh->FindPoints(phys_points, eid, ips);

  // And interpolate
  for (int ipt = 0; ipt < _em_opts.nBy; ipt++) {
    if (eid[ipt] >= 0) {
      _B->GetVectorValue(eid[ipt], ips[ipt], Bpoint);
      Byloc[ipt] = Bpoint[1];
    } else {
      Byloc[ipt] = 0;
    }
    MPI_Reduce(Byloc, By, _em_opts.nBy, MPI_DOUBLE, MPI_SUM, 0, _pmesh->GetComm());
  }

  // Finally, write the result to an hdf5 file
  hid_t file = -1;
  hid_t data_soln;
  herr_t status;
  hsize_t dims[1];
  hid_t group = -1;
  hid_t dataspace = -1;

  if (root) {
    file = H5Fcreate(_em_opts.By_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0);

    h5_save_attribute(file, "nBy", _em_opts.nBy);

    dims[0] = _em_opts.nBy;

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
