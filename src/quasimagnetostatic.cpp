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

#include <memory>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "logger.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

static Vector axis(3);
void JFun(const Vector &x, Vector &f);

QuasiMagnetostaticSolverBase::QuasiMagnetostaticSolverBase(ElectromagneticOptions em_opts, TPS::Tps *tps)
    : em_opts_(em_opts), offsets_(3), storeE_(false), Ereal_(NULL), Eimag_(NULL) {
  MPI_Comm_size(tps->getTPSCommWorld(), &nprocs_);
  MPI_Comm_rank(tps->getTPSCommWorld(), &rank_);
  if (rank_ == 0)
    rank0_ = true;
  else
    rank0_ = false;

  tpsP_ = tps;
  pmesh_ = NULL;

  // verify running on cpu
  if (tpsP_->getDeviceConfig() != "cpu") {
    if (rank0_) {
      grvy_printf(GRVY_INFO, "[INFO] Even in gpu run, EM simulation only uses the cpu.\n");
    }
  }

  plasma_conductivity_ = NULL;
  plasma_conductivity_coef_ = NULL;
  joule_heating_ = NULL;
}

double JouleHeatingCoefficient3D::Eval(ElementTransformation &T, const IntegrationPoint &ip) {
  Vector Er, Ei;
  double sig;
  Ereal_.GetVectorValue(T, ip, Er);
  Eimag_.GetVectorValue(T, ip, Ei);
  sig = sigma_.Eval(T, ip);
  return sig * (Er * Er + Ei * Ei);
}

QuasiMagnetostaticSolver3D::QuasiMagnetostaticSolver3D(ElectromagneticOptions em_opts, TPS::Tps *tps)
    : QuasiMagnetostaticSolverBase(em_opts, tps) {
  hcurl_ = NULL;
  h1_ = NULL;
  hdiv_ = NULL;
  L2_ = NULL;
  Aspace_ = NULL;
  pspace_ = NULL;
  Bspace_ = NULL;
  jh_space_ = NULL;
  K_ = NULL;
  r_ = NULL;
  grad_ = NULL;
  div_free_ = NULL;
  Areal_ = NULL;
  Aimag_ = NULL;
  Breal_ = NULL;
  Bimag_ = NULL;

  true_size_ = 0;
  offsets_ = 0;

  operator_initialized_ = false;
  current_initialized_ = false;
}

QuasiMagnetostaticSolver3D::~QuasiMagnetostaticSolver3D() {
  if (this->storeE_) {
    delete Ereal_;
    delete Eimag_;
  }
  delete plasma_conductivity_coef_;
  delete plasma_conductivity_;
  delete joule_heating_;
  delete Bimag_;
  delete Breal_;
  delete Aimag_;
  delete Areal_;
  delete div_free_;
  delete grad_;
  delete r_;
  delete K_;
  delete jh_space_;
  delete Bspace_;
  delete pspace_;
  delete Aspace_;
  delete L2_;
  delete hdiv_;
  delete h1_;
  delete hcurl_;
  delete pmesh_;
}

void QuasiMagnetostaticSolver3D::initialize() {
  bool verbose = rank0_;
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
  pmesh_ = new ParMesh(tpsP_->getTPSCommWorld(), *mesh);
  delete mesh;  // no longer need the serial mesh
  // pmesh_->ReorientTetMesh();

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------
  hcurl_ = new ND_FECollection(em_opts_.order, dim_);
  h1_ = new H1_FECollection(em_opts_.order, dim_);
  hdiv_ = new RT_FECollection(em_opts_.order - 1, dim_);
  L2_ = new L2_FECollection(em_opts_.order, dim_);

  Aspace_ = new ParFiniteElementSpace(pmesh_, hcurl_);
  pspace_ = new ParFiniteElementSpace(pmesh_, h1_);
  Bspace_ = new ParFiniteElementSpace(pmesh_, hdiv_);
  jh_space_ = new ParFiniteElementSpace(pmesh_, L2_);

  true_size_ = Aspace_->TrueVSize();
  offsets_[0] = 0;
  offsets_[1] = true_size_;
  offsets_[2] = true_size_;
  offsets_.PartialSum();

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

  grad_ = new ParDiscreteGradOperator(pspace_, Aspace_);
  grad_->Assemble();
  grad_->Finalize();

  int irOrder = pspace_->GetElementTransformation(0)->OrderW() + 2 * em_opts_.order;
  div_free_ = new DivergenceFreeProjector(*pspace_, *Aspace_, irOrder, NULL, NULL, grad_);

  // All done
  operator_initialized_ = true;

  // initialize current
  InitializeCurrent();

  // initialize conductivity and Joule heating
  plasma_conductivity_ = new ParGridFunction(pspace_);
  *plasma_conductivity_ = 0.0;

  plasma_conductivity_coef_ = new GridFunctionCoefficient(plasma_conductivity_);

  joule_heating_ = new ParGridFunction(jh_space_);
  *joule_heating_ = 0.0;
}

void QuasiMagnetostaticSolver3D::InitializeCurrent() {
  bool verbose = rank0_;
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
  axis[0] = em_opts_.current_axis[0];
  axis[1] = em_opts_.current_axis[1];
  axis[2] = em_opts_.current_axis[2];

  Vector J0(pmesh_->attributes.Max());
  J0 = 0.0;

  const double mu0J = em_opts_.mu0 * em_opts_.current_amplitude;

  if (em_opts_.bot_only) {
    // only bottom rings
    J0(1) = J0(2) = mu0J;
    J0(3) = J0(4) = 0.0;
  } else if (em_opts_.top_only) {
    // only top rings
    J0(1) = J0(2) = 0.0;
    J0(3) = J0(4) = mu0J;
  } else {
    // all rings
    J0(1) = J0(2) = mu0J;
    J0(3) = J0(4) = mu0J;
  }

  if (em_opts_.variable_current) {
    J0(1) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(1) * 0.5;
    J0(2) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(2) * 0.5;
    J0(3) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(3) * 0.5;
    J0(4) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(4) * 0.5;
  }

  if (rank0_) {
    std::cout << "J0 = " << J0(0) << ", " << J0(1)
              << ", " << J0(2) << ", " << J0(3)
              << ", " << J0(4) << endl;
  }

  PWConstCoefficient J0coef(J0);
  VectorFunctionCoefficient current(dim_, JFun, &J0coef);

  // 2) Build a discretely divergence-free approximation of the source
  // current that lives in the Nedelec FE space defined in
  // Initialize()
  ParGridFunction *Jorig = new ParGridFunction(Aspace_);
  ParGridFunction *Jproj = new ParGridFunction(Aspace_);

  r_ = new ParLinearForm(Aspace_);

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
  div_free_->Mult(*Jorig, *Jproj);

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
void QuasiMagnetostaticSolver3D::parseSolverOptions() {
  tpsP_->getRequiredInput("em/mesh", em_opts_.mesh_file);

  tpsP_->getInput("em/order", em_opts_.order, 1);
  tpsP_->getInput("em/ref_levels", em_opts_.ref_levels, 0);
  tpsP_->getInput("em/max_iter", em_opts_.max_iter, 100);
  tpsP_->getInput("em/rtol", em_opts_.rtol, 1.0e-6);
  tpsP_->getInput("em/atol", em_opts_.atol, 1.0e-10);
  tpsP_->getInput("em/preconditioner_background_sigma", em_opts_.preconditioner_background_sigma, -1.0);
  tpsP_->getInput("em/evaluate_magnetic_field", em_opts_.evaluate_magnetic_field, true);
  tpsP_->getInput("em/nBy", em_opts_.nBy, 0);
  tpsP_->getInput("em/yinterp_min", em_opts_.yinterp_min, 0.0);
  tpsP_->getInput("em/yinterp_max", em_opts_.yinterp_max, 1.0);
  tpsP_->getInput("em/By_file", em_opts_.By_file, std::string("By.h5"));
  tpsP_->getInput("em/top_only", em_opts_.top_only, false);
  tpsP_->getInput("em/bot_only", em_opts_.bot_only, false);

  // FLAG TO SET IF THE COILS WILL HAVE DIFFERENT CURRENT
  tpsP_->getInput("em/variable_current", em_opts_.variable_current, false);

  // EACH RING HAS DIFFERENT CURRENT AMPLITUDE
  Vector default_current(5);
  default_current = 0.0;
  tpsP_->getVec("em/varcurrent_amplitude", em_opts_.varcurrent_amplitude, 5, default_current);

  tpsP_->getInput("em/current_amplitude", em_opts_.current_amplitude, 1.0);
  tpsP_->getInput("em/current_frequency", em_opts_.current_frequency, 1.0);
  tpsP_->getInput("em/permeability", em_opts_.mu0, 1.0);

  Vector default_axis(3);
  default_axis[0] = 0;
  default_axis[1] = 1;
  default_axis[2] = 0;
  tpsP_->getVec("em/current_axis", em_opts_.current_axis, 3, default_axis);

  tpsP_->getInput("em/eval_Rplasma", em_opts_.eval_Rplasma, false);
  tpsP_->getInput("em/print_level", em_opts_.print_level, 0);

  // dump options to screen for user inspection
  if (rank0_) {
    em_opts_.print(std::cout);
  }
}

void QuasiMagnetostaticSolver3D::solve() { this->solveStep(); }

void QuasiMagnetostaticSolver3D::solveBegin() { return; }

void QuasiMagnetostaticSolver3D::solveEnd() { return; }

void QuasiMagnetostaticSolver3D::solveStep() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Solving the quasi-magnetostatic system for A (magnetic vector potential).\n");

  assert(operator_initialized_ && current_initialized_);

  // Solve for the magnetic vector potential and magnetic field in 4 steps

  // 0) Set up block operators for the QMS system
  //
  //  [ K                -\omega M(\sigma) ] [real(A)] = [real(J)]
  //  [ \omega M(\sigma)  K                ] [imag(A)] = [imag(J)]
  //
  // where K corresponds to the discrete curl(curl()) operator and
  // \omega M(\sigma) comes from \sigma dA/dt.

  BlockVector Avec(offsets_), rhs(offsets_);

  Areal_ = new ParGridFunction(Aspace_);
  *Areal_ = 0.0;

  Aimag_ = new ParGridFunction(Aspace_);
  *Aimag_ = 0.0;

  Areal_->ParallelAssemble(Avec.GetBlock(0));
  Aimag_->ParallelAssemble(Avec.GetBlock(1));

  // NB: Driving current assumed to have *only* real content
  r_->ParallelAssemble(rhs.GetBlock(0));
  rhs.GetBlock(1) = 0.0;

  // Set up block operator
  OperatorPtr Kdiag;
  K_->FormSystemMatrix(ess_bdr_tdofs_, Kdiag);
  K_->EliminateVDofsInRHS(ess_bdr_tdofs_, Avec.GetBlock(0), rhs.GetBlock(0));
  K_->EliminateVDofsInRHS(ess_bdr_tdofs_, Avec.GetBlock(1), rhs.GetBlock(1));

  plasma_conductivity_coef_->SetGridFunction(plasma_conductivity_);

  const double mu0_omega = em_opts_.mu0 * em_opts_.current_frequency * 2 * M_PI;
  ProductCoefficient mu_sigma_omega(mu0_omega, *plasma_conductivity_coef_);

  std::unique_ptr<ParBilinearForm> Kconductivity(new ParBilinearForm(Aspace_));
  Kconductivity->AddDomainIntegrator(new VectorFEMassIntegrator(mu_sigma_omega));
  Kconductivity->Assemble();

  OperatorPtr Koffd;
  Kconductivity->FormSystemMatrix(ess_bdr_tdofs_, Koffd);

  HypreParMatrix *Kdiag_mat = Kdiag.As<HypreParMatrix>();
  HypreParMatrix *Koffd_mat = Koffd.As<HypreParMatrix>();

  Koffd_mat->EliminateRows(ess_bdr_tdofs_);

  std::unique_ptr<BlockOperator> qms(new BlockOperator(offsets_));

  qms->SetBlock(0, 0, Kdiag_mat);
  qms->SetBlock(0, 1, Koffd_mat, -1);
  qms->SetBlock(1, 0, Koffd_mat);
  qms->SetBlock(1, 1, Kdiag_mat);

  // preconditioner
  std::unique_ptr<ParBilinearForm> Kpre(new ParBilinearForm(Aspace_));
  Kpre->AddDomainIntegrator(new CurlCurlIntegrator);
  Kpre->AddDomainIntegrator(new VectorFEMassIntegrator(mu_sigma_omega));

  bool preconditioner_spd = false;
  if (em_opts_.preconditioner_background_sigma > 0) {
    ConstantCoefficient reg(mu0_omega * em_opts_.preconditioner_background_sigma);
    Kpre->AddDomainIntegrator(new VectorFEMassIntegrator(reg));
    preconditioner_spd = true;
  }
  Kpre->Assemble();

  OperatorPtr Kpre_op;
  Kpre->FormSystemMatrix(ess_bdr_tdofs_, Kpre_op);

  std::unique_ptr<HypreAMS> Paar(new HypreAMS(*Kpre_op.As<HypreParMatrix>(), Aspace_));
  Paar->iterative_mode = false;
  if (!preconditioner_spd) {
    Paar->SetSingularProblem();
  }

  std::unique_ptr<Operator> Paai(new ScaledOperator(Paar.get(), -1.0));

  BlockDiagonalPreconditioner BDP(offsets_);
  BDP.SetDiagonalBlock(0, Paar.get());
  BDP.SetDiagonalBlock(1, Paai.get());

  // 1) Solve QMS (i.e., i \omega A + curl(curl(A)) = J) for magnetic
  //    vector potential using operators set up by Initialize() and
  //    InitializeCurrent() fcns
  FGMRESSolver solver(tpsP_->getTPSCommWorld());

  solver.SetPreconditioner(BDP);
  solver.SetOperator(*qms);

  solver.SetRelTol(em_opts_.rtol);
  solver.SetAbsTol(em_opts_.atol);
  solver.SetMaxIter(em_opts_.max_iter);
  solver.SetPrintLevel(em_opts_.print_level);

  solver.Mult(rhs, Avec);

  Areal_->Distribute(&(Avec.GetBlock(0)));
  Aimag_->Distribute(&(Avec.GetBlock(1)));

  if (em_opts_.evaluate_magnetic_field) {
    // 2) Determine B from A according to B = curl(A).  Here we use an
    //    interpolator rather than a projection.
    if (verbose) grvy_printf(ginfo, "Evaluating curl(A) to get the magnetic field.\n");
    if (Breal_ == NULL) Breal_ = new ParGridFunction(Bspace_);
    *Breal_ = 0;

    if (Bimag_ == NULL) Bimag_ = new ParGridFunction(Bspace_);
    *Breal_ = 0;

    ParDiscreteLinearOperator *curl = new ParDiscreteLinearOperator(Aspace_, Bspace_);
    curl->AddDomainInterpolator(new CurlInterpolator);
    curl->Assemble();
    curl->Finalize();
    curl->Mult(*Areal_, *Breal_);
    curl->Mult(*Aimag_, *Bimag_);
    delete curl;

    // Compute and dump the magnetic field on the axis
    InterpolateToYAxis();
  }

  // Divergence free projection of A (so that we can evaluate E and Joule heating correctly)
  if (!this->storeE_) {
    this->Ereal_ = new ParGridFunction(Aspace_);
    this->Eimag_ = new ParGridFunction(Aspace_);
  }

  // E = \grad phi + i omega A. Multiplication by i switches the real and imaginary part
  div_free_->Mult(*Areal_, *(this->Eimag_));
  div_free_->Mult(*Aimag_, *(this->Ereal_));

  // Compute Joule heating
  const double omega = (2 * M_PI * em_opts_.current_frequency);
  *(this->Ereal_) *= (-omega);
  *(this->Eimag_) *= omega;

  JouleHeatingCoefficient3D jh_coeff(*plasma_conductivity_coef_, *(this->Ereal_), *(this->Eimag_));
  joule_heating_->ProjectCoefficient(jh_coeff);

  { /*open scope so that ParaViewDataCollection is destroyed before Ereal_, Eimag_ if needed*/
    // 3) Output A and B fields for visualization using paraview
    if (verbose && em_opts_.print_level > 0) grvy_printf(ginfo, "Writing solution to paraview output.\n");
    ParaViewDataCollection paraview_dc("magnetostatic", pmesh_);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(em_opts_.order);
    paraview_dc.SetCycle(0);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("magvecpot_real", Areal_);
    paraview_dc.RegisterField("magvecpot_imag", Aimag_);
    paraview_dc.RegisterField("magEfield_real", this->Ereal_);
    paraview_dc.RegisterField("magEfield_imag", this->Eimag_);
    paraview_dc.RegisterField("joule_heating", joule_heating_);

    if (em_opts_.evaluate_magnetic_field) {
      paraview_dc.RegisterField("magnfield_real", Breal_);
      paraview_dc.RegisterField("magnfield_imag", Bimag_);
    }

    paraview_dc.Save();
  } /*Close scope*/

  if (!this->storeE_) {
    delete Ereal_;
    Ereal_ = NULL;
    delete Eimag_;
    Eimag_ = NULL;
  }

  if (verbose) {
    std::cout << "EM simulation complete" << std::endl;
  }
}

void QuasiMagnetostaticSolver3D::InterpolateToYAxis() const {
  const bool root = rank0_;

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
      // B_->GetVectorValue(eid[ipt], ips[ipt], Bpoint);
      Breal_->GetVectorValue(eid[ipt], ips[ipt], Bpoint);
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

  delete[] Byloc;
  delete[] By;
}

void QuasiMagnetostaticSolver3D::setStoreE(bool storeE) {
  if (storeE == this->storeE_) return; /*Nothing to be done*/
  this->storeE_ = storeE;
  if (storeE) {
    this->Ereal_ = new ParGridFunction(Aspace_);
    this->Eimag_ = new ParGridFunction(Aspace_);
  } else {
    delete this->Ereal_;
    delete this->Eimag_;
  }
}

double QuasiMagnetostaticSolver3D::elementJouleHeating(const FiniteElement &el, ElementTransformation &Tr,
                                                       const Vector &elfun) {
  // Get size info
  const int dof = el.GetDof();
  const int nvar = 1;

  // Storage for basis fcns, derivatives, fluxes
  Vector shape;
  shape.SetSize(dof);
  Vector soln;
  soln.SetSize(nvar);

  // View elfun and elvec as matrices (make life easy below)
  DenseMatrix elfun_mat(elfun.GetData(), dof, nvar);

  // Get quadrature rule
  const int order = el.GetOrder();
  const int intorder = order + 1;
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

  double elem_jh = 0;

  // for every quadrature point...
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    // evaluate basis functions
    el.CalcShape(ip, shape);

    // Interpolate solution
    elfun_mat.MultTranspose(shape, soln);

    // Add quad point contribution to integral
    double qpcontrib = soln[0];

    const double wt = ip.weight * Tr.Weight();
    elem_jh += qpcontrib * wt;
  }

  return elem_jh;
}

double QuasiMagnetostaticSolver3D::totalJouleHeating() {
  const int NE = pmesh_->GetNE();
  const ParFiniteElementSpace *fes = jh_space_;

  double int_jh = 0;

  Array<int> vdofs;
  Vector el_x;

  // loop over elements on this mpi rank and integrate joule heating
  for (int ielem = 0; ielem < NE; ielem++) {
    const FiniteElement *fe = fes->GetFE(ielem);
    ElementTransformation *T = fes->GetElementTransformation(ielem);
#if MFEM_VERSION >= 40400
    DofTransformation *doftrans = fes->GetElementVDofs(ielem, vdofs);
    joule_heating_->GetSubVector(vdofs, el_x);
    if (doftrans) {
      doftrans->InvTransformPrimal(el_x);
    }
#else
    joule_heating_->GetSubVector(vdofs, el_x);
#endif

    int_jh += elementJouleHeating(*fe, *T, el_x);
  }

  // sum over mpi ranks
  double total_int_jh = 0;
  MPI_Allreduce(&int_jh, &total_int_jh, 1, MPI_DOUBLE, MPI_SUM, fes->GetComm());

  return total_int_jh;
}

void JFun(const Vector &x, Vector &J) {
  Vector axx(3);

  axx(0) = axis(1) * x(2) - axis(2) * x(1);
  axx(1) = axis(2) * x(0) - axis(0) * x(2);
  axx(2) = axis(0) * x(1) - axis(1) * x(0);
  axx /= axx.Norml2();

  J = axx;
}

static double radius(const Vector &x) { return x[0]; }

static double oneOverRadius(const Vector &x) { return 1.0 / x[0]; }

QuasiMagnetostaticSolverAxiSym::QuasiMagnetostaticSolverAxiSym(ElectromagneticOptions em_opts, TPS::Tps *tps)
    : QuasiMagnetostaticSolverBase(em_opts, tps) {
  h1_ = NULL;
  Atheta_space_ = NULL;
  K_ = NULL;
  r_ = NULL;
  Atheta_real_ = NULL;
  Atheta_imag_ = NULL;
  plasma_conductivity_ = NULL;
  plasma_conductivity_coef_ = NULL;

  true_size_ = 0;
  offsets_ = 0;

  operator_initialized_ = false;
  current_initialized_ = false;
}

QuasiMagnetostaticSolverAxiSym::~QuasiMagnetostaticSolverAxiSym() {
  if (this->storeE_) {
    delete Ereal_;
    delete Eimag_;
  }
  delete joule_heating_;
  delete plasma_conductivity_coef_;
  delete plasma_conductivity_;
  delete Atheta_imag_;
  delete Atheta_real_;
  delete r_;
  delete K_;
  delete Atheta_space_;
  delete h1_;
  delete pmesh_;
}

void QuasiMagnetostaticSolverAxiSym::initialize() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing axisymmetric quasimagnetostatic solver.\n");

  // Prepare for the quasi-magnetostatic solve in four steps...

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // 1a) Read the serial mesh (on each mpi rank)
  if (verbose) grvy_printf(ginfo, "Reading EM mesh file: %s\n", em_opts_.mesh_file.c_str());
  Mesh *mesh = new Mesh(em_opts_.mesh_file.c_str(), 1, 1);
  dim_ = mesh->Dimension();
  if (dim_ != 2) {
    if (verbose) {
      grvy_printf(gerror, "[ERROR] Axisymmetric quasi-magnetostatic solver on supported for 2D meshes.");
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
  pmesh_ = new ParMesh(tpsP_->getTPSCommWorld(), *mesh);
  delete mesh;  // no longer need the serial mesh
  // pmesh_->ReorientTetMesh();

  //-----------------------------------------------------
  // 2) Prepare the required finite element and sizes
  //-----------------------------------------------------
  h1_ = new H1_FECollection(em_opts_.order, dim_);

  Atheta_space_ = new ParFiniteElementSpace(pmesh_, h1_);

  true_size_ = Atheta_space_->TrueVSize();

  offsets_[0] = 0;
  offsets_[1] = true_size_;
  offsets_[2] = true_size_;
  offsets_.PartialSum();

  //-----------------------------------------------------
  // 3) Get BC dofs (everything is essential)
  //-----------------------------------------------------
  Array<int> ess_bdr(pmesh_->bdr_attributes.Max());
  if (pmesh_->bdr_attributes.Size()) {
    ess_bdr = 1;
  }

  Atheta_space_->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs_);

  //-----------------------------------------------------
  // 4) Form axisymmetric QMS bilinear form
  //-----------------------------------------------------
  FunctionCoefficient radius_coeff(radius);
  FunctionCoefficient one_over_radius_coeff(oneOverRadius);

  K_ = new ParBilinearForm(Atheta_space_);
  K_->AddDomainIntegrator(new DiffusionIntegrator(radius_coeff));
  K_->AddDomainIntegrator(new MassIntegrator(one_over_radius_coeff));
  K_->Assemble();

  operator_initialized_ = true;

  // initialize current
  InitializeCurrent();

  // initialize conductivity
  plasma_conductivity_ = new ParGridFunction(Atheta_space_);
  *plasma_conductivity_ = 0.0;

  plasma_conductivity_coef_ = new GridFunctionCoefficient(plasma_conductivity_);

  // initialize joule heating
  joule_heating_ = new ParGridFunction(Atheta_space_);
  *joule_heating_ = 0.0;
}

void QuasiMagnetostaticSolverAxiSym::InitializeCurrent() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing source current.\n");

  // ensure we've initialized the operators
  // b/c we need the mesh, spaces, etc here
  assert(operator_initialized_);

  // ensure the mesh volume attributes conform to our expectations
  // assert(pmesh_->attributes.Max() == 5);
  if (pmesh_->attributes.Max() != 5) {
    printf("WARNING: Could not initialize current source.\n");
    printf("         This probably means you should fix your mesh,\n");
    printf("         but I will continue anyway with no source.\n");
    current_initialized_ = false;
    return;
  }

  // Compute the right hand side in 3 steps...

  // 1) Build the current function For now, this is hardcoded to be
  //    uniformly distributed current density in rings defined by
  //    volume attributes in the mesh.  We assume 4 rings, plus a zero
  //    source current domain for 5 total attributes.
  Vector J0(pmesh_->attributes.Max());
  J0 = 0.0;

  const double mu0J = em_opts_.mu0 * em_opts_.current_amplitude * 0.5;

  if (em_opts_.bot_only) {
    // only bottom rings
    J0(1) = J0(2) = mu0J;
    J0(3) = J0(4) = 0.0;
  } else if (em_opts_.top_only) {
    // only top rings
    J0(1) = J0(2) = 0.0;
    J0(3) = J0(4) = mu0J;
  } else {
    // all rings
    J0(1) = J0(2) = mu0J;
    J0(3) = J0(4) = mu0J;
  }

  if (em_opts_.variable_current) {
    J0(1) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(1) * 0.5;
    J0(2) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(2) * 0.5;
    J0(3) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(3) * 0.5;
    J0(4) = em_opts_.mu0 * em_opts_.varcurrent_amplitude(4) * 0.5;
  }

  if (rank0_) {
    std::cout << "J0 = " << J0(0) << ", " << J0(1)
              << ", " << J0(2) << ", " << J0(3)
              << ", " << J0(4) << endl;
  }

  FunctionCoefficient radius_coeff(radius);

  PWConstCoefficient J0coef(J0);
  ProductCoefficient current(J0coef, radius_coeff);

  r_ = new ParLinearForm(Atheta_space_);
  r_->AddDomainIntegrator(new DomainLFIntegrator(current));
  r_->Assemble();

  current_initialized_ = true;
}

void QuasiMagnetostaticSolverAxiSym::parseSolverOptions() {
  tpsP_->getRequiredInput("em/mesh", em_opts_.mesh_file);

  tpsP_->getInput("em/order", em_opts_.order, 1);
  tpsP_->getInput("em/ref_levels", em_opts_.ref_levels, 0);
  tpsP_->getInput("em/max_iter", em_opts_.max_iter, 100);
  tpsP_->getInput("em/rtol", em_opts_.rtol, 1.0e-6);
  tpsP_->getInput("em/atol", em_opts_.atol, 1.0e-10);
  tpsP_->getInput("em/evaluate_magnetic_field", em_opts_.evaluate_magnetic_field, false);
  tpsP_->getInput("em/nBy", em_opts_.nBy, 0);
  tpsP_->getInput("em/yinterp_min", em_opts_.yinterp_min, 0.0);
  tpsP_->getInput("em/yinterp_max", em_opts_.yinterp_max, 1.0);
  tpsP_->getInput("em/By_file", em_opts_.By_file, std::string("By.h5"));
  tpsP_->getInput("em/top_only", em_opts_.top_only, false);
  tpsP_->getInput("em/bot_only", em_opts_.bot_only, false);

  tpsP_->getInput("em/variable_current", em_opts_.variable_current, false);

  // EACH RING HAS DIFFERENT CURRENT AMPLITUDE
  Vector default_current(5);
  default_current = 0.0;
  tpsP_->getVec("em/varcurrent_amplitude", em_opts_.varcurrent_amplitude, 5, default_current);

  // CONSTANT CURRENT AMPLITUDE FOR ALL RINGS OF THE COIL
  tpsP_->getInput("em/current_amplitude", em_opts_.current_amplitude, 1.0);
  tpsP_->getInput("em/current_frequency", em_opts_.current_frequency, 1.0);
  tpsP_->getInput("em/permeability", em_opts_.mu0, 1.0);

  tpsP_->getInput("em/eval_Rplasma", em_opts_.eval_Rplasma, false);

  // dump options to screen for user inspection
  if (rank0_) {
    em_opts_.print(std::cout);
  }
}

void QuasiMagnetostaticSolverAxiSym::solve() { this->solveStep(); }

void QuasiMagnetostaticSolverAxiSym::solveBegin() {}

void QuasiMagnetostaticSolverAxiSym::solveEnd() {}

void QuasiMagnetostaticSolverAxiSym::solveStep() {
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Solving the axisymmetric quasi-magnetostatic system.\n");

  assert(operator_initialized_ && current_initialized_);

  // Set up block soln and rhs
  BlockVector Atheta_vec(offsets_), rhs_vec(offsets_);

  if (Atheta_real_ == NULL) Atheta_real_ = new ParGridFunction(Atheta_space_);
  *Atheta_real_ = 0.0;

  if (Atheta_imag_ == NULL) Atheta_imag_ = new ParGridFunction(Atheta_space_);
  *Atheta_imag_ = 0.0;

  Atheta_real_->ParallelAssemble(Atheta_vec.GetBlock(0));
  Atheta_imag_->ParallelAssemble(Atheta_vec.GetBlock(1));

  // NB: Have assumed that driving current has only REAL content
  r_->ParallelAssemble(rhs_vec.GetBlock(0));
  rhs_vec.GetBlock(1) = 0.0;

  // Set up block operator
  OperatorPtr Kdiag;
  K_->FormSystemMatrix(ess_bdr_tdofs_, Kdiag);
  K_->EliminateVDofsInRHS(ess_bdr_tdofs_, Atheta_vec.GetBlock(0), rhs_vec.GetBlock(0));
  K_->EliminateVDofsInRHS(ess_bdr_tdofs_, Atheta_vec.GetBlock(1), rhs_vec.GetBlock(1));

  plasma_conductivity_coef_->SetGridFunction(plasma_conductivity_);

  FunctionCoefficient radius_coeff(radius);
  const double mu0_omega = em_opts_.mu0 * em_opts_.current_frequency * 2 * M_PI;
  ProductCoefficient temp_coef(mu0_omega, *plasma_conductivity_coef_);
  ProductCoefficient mu_sigma_omega(temp_coef, radius_coeff);

  std::unique_ptr<ParBilinearForm> Kconductivity(new ParBilinearForm(Atheta_space_));
  Kconductivity->AddDomainIntegrator(new MassIntegrator(mu_sigma_omega));
  Kconductivity->Assemble();

  OperatorPtr Koffd;
  Kconductivity->FormSystemMatrix(ess_bdr_tdofs_, Koffd);

  HypreParMatrix *Kdiag_mat = Kdiag.As<HypreParMatrix>();
  HypreParMatrix *Koffd_mat = Koffd.As<HypreParMatrix>();

  Koffd_mat->EliminateRows(ess_bdr_tdofs_);

  std::unique_ptr<BlockOperator> qms(new BlockOperator(offsets_));

  qms->SetBlock(0, 0, Kdiag_mat);
  qms->SetBlock(0, 1, Koffd_mat, -1);
  qms->SetBlock(1, 0, Koffd_mat);
  qms->SetBlock(1, 1, Kdiag_mat);

  // Set up block preconditioner
  FunctionCoefficient one_over_radius_coeff(oneOverRadius);

  std::unique_ptr<ParBilinearForm> Kpre(new ParBilinearForm(Atheta_space_));
  Kpre->AddDomainIntegrator(new MassIntegrator(mu_sigma_omega));
  Kpre->AddDomainIntegrator(new DiffusionIntegrator(radius_coeff));
  Kpre->AddDomainIntegrator(new MassIntegrator(one_over_radius_coeff));
  Kpre->Assemble();

  OperatorPtr KpreOp;
  Kpre->FormSystemMatrix(ess_bdr_tdofs_, KpreOp);

  std::unique_ptr<HypreBoomerAMG> prec(new HypreBoomerAMG(*KpreOp.As<HypreParMatrix>()));
  prec->SetPrintLevel(em_opts_.print_level);

  BlockDiagonalPreconditioner BDP(offsets_);
  BDP.SetDiagonalBlock(0, prec.get());
  BDP.SetDiagonalBlock(1, prec.get());

  FGMRESSolver solver(tpsP_->getTPSCommWorld());

  solver.SetPreconditioner(BDP);
  solver.SetOperator(*qms);

  solver.SetRelTol(em_opts_.rtol);
  solver.SetAbsTol(em_opts_.atol);
  solver.SetMaxIter(em_opts_.max_iter);
  solver.SetPrintLevel(em_opts_.print_level);

  solver.Mult(rhs_vec, Atheta_vec);

  Atheta_real_->Distribute(&(Atheta_vec.GetBlock(0)));
  Atheta_imag_->Distribute(&(Atheta_vec.GetBlock(1)));

  // TODO(trevilo): Compute B field (maybe... we only need it for validation comparisons)

  // Compute Joule heating (on em mesh obviously)
  const double omega = (2 * M_PI * em_opts_.current_frequency);
  const double omega2 = omega * omega;

  Vector tmp1 = (*Atheta_real_);
  tmp1 *= (*Atheta_real_);
  Vector tmp2 = (*Atheta_imag_);
  tmp2 *= (*Atheta_imag_);

  tmp2 += tmp1;
  tmp2 *= (*plasma_conductivity_);
  tmp2 *= 2.0 * omega2;

  *joule_heating_ = tmp2;

  if (this->storeE_) {
    *(this->Ereal_) = (*Atheta_imag_);
    (*(this->Ereal_)) *= (-omega);
    *(this->Eimag_) = (*Atheta_real_);
    (*(this->Eimag_)) *= omega;
  }

  // 3) Output A and B fields for visualization using paraview
  if (verbose && em_opts_.print_level > 0) grvy_printf(ginfo, "Writing solution to paraview output.\n");
  ParaViewDataCollection paraview_dc("magnetostatic", pmesh_);
  paraview_dc.SetPrefixPath("ParaView");
  paraview_dc.SetLevelsOfDetail(em_opts_.order);
  paraview_dc.SetCycle(0);
  paraview_dc.SetDataFormat(VTKFormat::BINARY);
  paraview_dc.SetHighOrderOutput(true);
  paraview_dc.SetTime(0.0);
  paraview_dc.RegisterField("magvecpot_real", Atheta_real_);
  paraview_dc.RegisterField("magvecpot_imag", Atheta_imag_);
  paraview_dc.RegisterField("plasma_conductivity", plasma_conductivity_);
  paraview_dc.RegisterField("joule_heating", joule_heating_);
  // paraview_dc.RegisterField("magnfield", _B);
  paraview_dc.Save();

  // Compute and dump the magnetic field on the axis
  // InterpolateToYAxis();

  if (verbose) {
    std::cout << "EM simulation complete" << std::endl;
  }
}

void QuasiMagnetostaticSolverAxiSym::setStoreE(bool storeE) {
  if (storeE == this->storeE_) return; /*Nothing to be done*/
  this->storeE_ = storeE;
  if (storeE) {
    this->Ereal_ = new ParGridFunction(Atheta_space_);
    this->Eimag_ = new ParGridFunction(Atheta_space_);
  } else {
    delete this->Ereal_;
    this->Ereal_ = NULL;
    delete this->Eimag_;
    this->Eimag_ = NULL;
  }
}

double QuasiMagnetostaticSolverAxiSym::elementJouleHeating(const FiniteElement &el, ElementTransformation &Tr,
                                                           const Vector &elfun) {
  // Get size info
  const int dof = el.GetDof();
  const int dim = el.GetDim();
  const int nvar = 1;

  assert(dim == 2);

  // Storage for basis fcns, derivatives, fluxes
  Vector shape;
  shape.SetSize(dof);
  Vector soln;
  soln.SetSize(nvar);

  // View elfun and elvec as matrices (make life easy below)
  DenseMatrix elfun_mat(elfun.GetData(), dof, nvar);

  // Get quadrature rule
  const int order = el.GetOrder();
  const int intorder = order + 1;
  const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

  double elem_jh = 0;

  // for every quadrature point...
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    // evaluate radius
    double x[3];
    Vector transip(x, 3);
    Tr.Transform(ip, transip);
    const double radius = transip[0];

    // evaluate basis functions
    el.CalcShape(ip, shape);

    // Interpolate solution
    elfun_mat.MultTranspose(shape, soln);

    // Add quad point contribution to integral
    double qpcontrib = soln[0] * radius;

    const double wt = ip.weight * Tr.Weight();
    elem_jh += qpcontrib * wt;
  }

  return elem_jh;
}

double QuasiMagnetostaticSolverAxiSym::totalJouleHeating() {
  const int NE = pmesh_->GetNE();
  const ParFiniteElementSpace *fes = this->getFESpace();

  double int_jh = 0;

  Array<int> vdofs;
  Vector el_x;

  // loop over elements on this mpi rank and integrate joule heating
  for (int ielem = 0; ielem < NE; ielem++) {
    const FiniteElement *fe = fes->GetFE(ielem);
    ElementTransformation *T = fes->GetElementTransformation(ielem);
#if MFEM_VERSION >= 40400
    DofTransformation *doftrans = fes->GetElementVDofs(ielem, vdofs);
    joule_heating_->GetSubVector(vdofs, el_x);
    if (doftrans) {
      doftrans->InvTransformPrimal(el_x);
    }
#else
    joule_heating_->GetSubVector(vdofs, el_x);
#endif

    int_jh += elementJouleHeating(*fe, *T, el_x);
  }

  // sum over mpi ranks
  double total_int_jh = 0;
  MPI_Allreduce(&int_jh, &total_int_jh, 1, MPI_DOUBLE, MPI_SUM, fes->GetComm());

  // Factor of 2*pi from azimuthal direction integral
  return 2 * M_PI * total_int_jh;
}

double QuasiMagnetostaticSolverAxiSym::coilCurrent() const {
  assert(current_initialized_);

  // We re-build the current coefficient here, but only using 1 ring
  // so that when we integrate over area, we obtain the correct
  // current and not the current times number of rings
  Vector J0(pmesh_->attributes.Max());
  J0 = 0.0;
  J0(1) = em_opts_.current_amplitude * 0.5;

  // if(em_opts_.variable_current){
  //   J0(1) = em_opts_.varcurrent_amplitude(1) * 0.5;
  //   J0(2) = em_opts_.varcurrent_amplitude(2) * 0.5;
  //   J0(3) = em_opts_.varcurrent_amplitude(3) * 0.5;
  //   J0(4) = em_opts_.varcurrent_amplitude(4) * 0.5;
  // }

  PWConstCoefficient J0coef(J0);

  // Integrate the current by looping over the elements.  Note that
  // this is an area integral, not a volume integral, so there is no
  // factor of 2 \pi radius
  double current = 0;
  const int NE = pmesh_->GetNE();
  const ParFiniteElementSpace *fes = Atheta_space_;

  for (int ielem = 0; ielem < NE; ielem++) {
    const FiniteElement *fe = fes->GetFE(ielem);
    ElementTransformation *T = fes->GetElementTransformation(ielem);

    // B/c the current is piecewise constant, first order rule is sufficient
    const int intorder = 1;
    const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T->SetIntPoint(&ip);
      current += ip.weight * T->Weight() * J0coef.Eval(*T, ip);
    }
  }

  // sum over mpi ranks
  double total_current = 0;
  MPI_Allreduce(&current, &total_current, 1, MPI_DOUBLE, MPI_SUM, fes->GetComm());

  return total_current;
}

double QuasiMagnetostaticSolverAxiSym::magneticEnergy() const {
  double magnetic_energy = 0;

  const double mu0 = em_opts_.mu0;
  const int NE = pmesh_->GetNE();
  const ParFiniteElementSpace *fes = Atheta_space_;

  for (int ielem = 0; ielem < NE; ielem++) {
    const FiniteElement *fe = fes->GetFE(ielem);
    ElementTransformation *T = fes->GetElementTransformation(ielem);

    // Get size info
    const int dof = fe->GetDof();
    const int dim = fe->GetDim();
    const int nvar = 1;

    assert(dim == 2);

    // Storage for basis fcns, derivatives, fluxes
    Vector shape;
    shape.SetSize(dof);

    DenseMatrix dshape;
    dshape.SetSize(dof, dim);

    Vector Areal, Aimag;
    Areal.SetSize(nvar);
    Aimag.SetSize(nvar);

    DenseMatrix Areal_x, Aimag_x;
    Areal_x.SetSize(nvar, dim);
    Aimag_x.SetSize(nvar, dim);

    Vector Breal, Bimag;
    Breal.SetSize(dim);
    Bimag.SetSize(dim);

    // Get solution (Atheta) dofs for this element
    Array<int> vdofs;
    Vector el_Areal;
    Vector el_Aimag;

    DofTransformation *doftrans = fes->GetElementVDofs(ielem, vdofs);
    Atheta_real_->GetSubVector(vdofs, el_Areal);
    if (doftrans) {
      doftrans->InvTransformPrimal(el_Areal);
    }
    Atheta_imag_->GetSubVector(vdofs, el_Aimag);
    if (doftrans) {
      doftrans->InvTransformPrimal(el_Aimag);
    }

    // View element dofs as matrices (make life easy below)
    const DenseMatrix el_Areal_mat(el_Areal.HostReadWrite(), dof, nvar);
    const DenseMatrix el_Aimag_mat(el_Aimag.HostReadWrite(), dof, nvar);

    // Get quadrature rule
    const int order = fe->GetOrder();
    const int intorder = 2 * order + 1;
    const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

    // for every quadrature point...
    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T->SetIntPoint(&ip);

      // evaluate radius
      double x[3];
      Vector transip(x, 3);
      T->Transform(ip, transip);
      const double radius = transip[0];

      // evaluate basis functions
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*T, dshape);

      // Interpolate solution and gradient
      el_Areal_mat.MultTranspose(shape, Areal);
      el_Aimag_mat.MultTranspose(shape, Aimag);

      // el_Areal_mat.MultTranspose(dshape, Areal_x);
      // el_Aimag_mat.MultTranspose(dshape, Aimag_x);

      MultAtB(el_Areal_mat, dshape, Areal_x);
      MultAtB(el_Aimag_mat, dshape, Aimag_x);

      const double *hAr = Areal.HostRead();
      const double *hAi = Aimag.HostRead();
      const double *hAr_x = Areal_x.HostRead();
      const double *hAi_x = Aimag_x.HostRead();

      double *hBr = Breal.HostWrite();
      double *hBi = Bimag.HostWrite();

      // B = curl(A)
      hBr[0] = -hAr_x[1];
      hBr[1] = hAr_x[0] + hAr[0] / radius;

      hBi[0] = -hAi_x[1];
      hBi[1] = hAi_x[0] + hAi[0] / radius;

      const double BdotConjB = (hBr[0] * hBr[0] + hBi[0] * hBi[0] + hBr[1] * hBr[1] + hBi[1] * hBi[1]);

      // Add quad point contribution to integral
      double qpcontrib = BdotConjB * radius;

      const double wt = ip.weight * T->Weight();
      magnetic_energy += qpcontrib * wt;
    }
  }
  magnetic_energy *= 2 * M_PI / mu0;

  // sum over mpi ranks
  double total_magnetic_energy = 0;
  MPI_Allreduce(&magnetic_energy, &total_magnetic_energy, 1, MPI_DOUBLE, MPI_SUM, fes->GetComm());

  return total_magnetic_energy;
}
