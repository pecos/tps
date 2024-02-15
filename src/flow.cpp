

// #include "turbModel.hpp"
// #include "thermoChem.hpp"
#include "flow.hpp"
// #include "loMach.hpp"
#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

// temporary
void accel(const Vector &x, double t, Vector &f);
void vel_ic(const Vector &coords, double t, Vector &u);
void vel_icBox(const Vector &coords, double t, Vector &u);
void vel_wall(const Vector &x, double t, Vector &u);
void vel_inlet(const Vector &x, double t, Vector &u);
double pres_inlet(const Vector &x, double p);
void vel_channelTest(const Vector &coords, double t, Vector &u);

Flow::Flow(mfem::ParMesh *pmesh_, RunConfiguration *config_, LoMachOptions *loMach_opts_)
    : pmesh(pmesh_), loMach_opts(loMach_opts_), config(config_) {}

void Flow::initialize() {
  // groupsMPI = pmesh->GetComm();
  rank = pmesh->GetMyRank();
  // rank0 = pmesh->isWorldRoot();
  rank0 = false;
  if (rank == 0) {
    rank0 = true;
  }
  dim = pmesh->Dimension();
  nvel = dim;

  bool verbose = rank0;
  if (verbose) grvy_printf(ginfo, "Initializing Flow solver.\n");

  if (loMach_opts->uOrder == -1) {
    order = std::max(config->solOrder, 1);
    porder = order;
    double no;
    no = ceil(((double)order * 1.5));
    norder = int(no);
  } else {
    order = loMach_opts->uOrder;
    porder = loMach_opts->pOrder;
    norder = loMach_opts->nOrder;
  }

  MaxIters = config->GetNumIters();
  // num_equation = 5; // hard code for now, fix with species

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------

  // scalar
  sfec = new H1_FECollection(order);
  sfes = new ParFiniteElementSpace(pmesh, sfec);

  // vector
  vfec = new H1_FECollection(order, dim);
  vfes = new ParFiniteElementSpace(pmesh, vfec, dim);

  // pressure
  pfec = new H1_FECollection(porder);
  // pfec = new L2_FECollection(porder,dim);
  pfes = new ParFiniteElementSpace(pmesh, pfec);

  // dealias nonlinear term
  nfec = new H1_FECollection(norder, dim);
  nfes = new ParFiniteElementSpace(pmesh, nfec, dim);
  nfecR0 = new H1_FECollection(norder);
  nfesR0 = new ParFiniteElementSpace(pmesh, nfecR0);

  // Check if fully periodic mesh
  if (!(pmesh->bdr_attributes.Size() == 0)) {
    vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
    vel_ess_attr = 0;

    pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
    pres_ess_attr = 0;
  }
  if (verbose) grvy_printf(ginfo, "Flow spaces constructed...\n");

  int vfes_truevsize = vfes->GetTrueVSize();
  int pfes_truevsize = pfes->GetTrueVSize();
  int sfes_truevsize = sfes->GetTrueVSize();
  int nfes_truevsize = nfes->GetTrueVSize();
  int nfesR0_truevsize = nfesR0->GetTrueVSize();

  gravity.SetSize(3);
  gravity = 0.0;

  un_next.SetSize(vfes_truevsize);
  un.SetSize(vfes_truevsize);
  unm1.SetSize(vfes_truevsize);
  unm2.SetSize(vfes_truevsize);
  un_next = 0.0;
  un = 0.0;
  unm1 = 0.0;
  unm2 = 0.0;

  u_star.SetSize(vfes_truevsize);
  u_star = 0.0;

  u_half.SetSize(vfes_truevsize);
  u_half = 0.0;

  fn.SetSize(vfes_truevsize);
  fn = 0.0;

  Nun.SetSize(vfes_truevsize);
  Nunm1.SetSize(vfes_truevsize);
  Nunm2.SetSize(vfes_truevsize);
  Nun = 0.0;
  Nunm1 = 0.0;
  Nunm2 = 0.0;

  uBn.SetSize(nfes_truevsize);
  uBn = 0.0;

  Fext.SetSize(vfes_truevsize);
  FText.SetSize(vfes_truevsize);
  Lext.SetSize(vfes_truevsize);
  Uext.SetSize(vfes_truevsize);
  Ldiv.SetSize(vfes_truevsize);
  LdivImp.SetSize(vfes_truevsize);
  resu.SetSize(vfes_truevsize);
  resu = 0.0;
  FText = 0.0;

  gradMu.SetSize(vfes_truevsize);
  gradRho.SetSize(vfes_truevsize);
  gradDivU.SetSize(vfes_truevsize);
  gradU.SetSize(vfes_truevsize);
  gradV.SetSize(vfes_truevsize);
  gradW.SetSize(vfes_truevsize);

  gradX.SetSize(vfes_truevsize);
  gradY.SetSize(vfes_truevsize);
  gradZ.SetSize(vfes_truevsize);

  gradU_gf.SetSpace(vfes);
  gradV_gf.SetSpace(vfes);
  gradW_gf.SetSpace(vfes);

  uns.SetSize(vfes_truevsize);
  uns = 0.0;

  boussinesqField.SetSize(vfes_truevsize);
  boussinesqField = 0.0;
  divU.SetSize(sfes_truevsize);
  divU = 0.0;

  pn.SetSize(pfes_truevsize);
  pnm1.SetSize(pfes_truevsize);
  deltaP.SetSize(pfes_truevsize);
  pn = 0.0;
  pnm1 = 0.0;
  deltaP = 0.0;

  resp.SetSize(pfes_truevsize);
  resp = 0.0;
  FText_bdr.SetSize(sfes_truevsize);
  FText_bdr = 0.0;
  g_bdr.SetSize(sfes_truevsize);
  g_bdr = 0.0;

  // pnBig.SetSize(sfes_truevsize);
  // pnBig = 0.0;

  u_star_gf.SetSpace(vfes);
  u_star_gf = 0.0;

  un_next_gf.SetSpace(vfes);
  un_gf.SetSpace(vfes);
  unm1_gf.SetSpace(vfes);
  unm2_gf.SetSpace(vfes);
  un_next_gf = 0.0;
  un_gf = 0.0;
  unm1_gf = 0.0;
  unm2_gf = 0.0;

  Lext_gf.SetSpace(vfes);
  curlu_gf.SetSpace(vfes);
  curlcurlu_gf.SetSpace(vfes);
  FText_gf.SetSpace(vfes);
  FText_gf = 0.0;
  resu_gf.SetSpace(vfes);
  resu_gf = 0.0;

  pn_gf.SetSpace(pfes);
  pn_gf = 0.0;
  resp_gf.SetSpace(pfes);
  resp_gf = 0.0;
  // tmpR0PM1.SetSize(pfes_truevsize);

  Pext_bdr.SetSize(pfes_truevsize);
  Pext_gf.SetSpace(pfes);
  p_bdr.SetSize(pfes_truevsize);

  rn.SetSize(sfes_truevsize);
  visc.SetSize(sfes_truevsize);
  Qt.SetSize(sfes_truevsize);

  tmpR0.SetSize(sfes_truevsize);
  tmpR0a.SetSize(sfes_truevsize);
  tmpR0b.SetSize(sfes_truevsize);
  tmpR0c.SetSize(sfes_truevsize);
  tmpR0 = 0.0;
  tmpR0a = 0.0;
  tmpR0b = 0.0;
  tmpR0c = 0.0;

  R0PM0_gf.SetSpace(sfes);
  R0PM0_gf = 0.0;

  // pressure space
  R0PM1_gf.SetSpace(pfes);
  R0PM1_gf = 0.0;

  R1PM0_gf.SetSpace(vfes);
  R1PM0_gf = 0.0;

  R1PX2_gf.SetSpace(nfes);
  R1PX2_gf = 0.0;

  tmpR1.SetSize(vfes_truevsize);
  tmpR1a.SetSize(vfes_truevsize);
  tmpR1b.SetSize(vfes_truevsize);
  tmpR1c.SetSize(vfes_truevsize);
  tmpR1 = 0.0;
  tmpR1a = 0.0;
  tmpR1b = 0.0;
  tmpR1c = 0.0;

  r0pm0.SetSize(sfes_truevsize);
  r1pm0.SetSize(vfes_truevsize);
  r0px2a.SetSize(nfesR0_truevsize);
  r0px2b.SetSize(nfesR0_truevsize);
  r0px2c.SetSize(nfesR0_truevsize);
  r1px2a.SetSize(nfes_truevsize);
  r1px2b.SetSize(nfes_truevsize);
  r1px2c.SetSize(nfes_truevsize);

  if (verbose) grvy_printf(ginfo, "Flow vectors and gf initialized...\n");
}

void Flow::initializeExternal(ParGridFunction *visc_gf_, ParGridFunction *rn_gf_, ParGridFunction *Qt_gf_, int nWalls,
                              int nInlets, int nOutlets) {
  visc_gf = visc_gf_;
  rn_gf = rn_gf_;
  Qt_gf = Qt_gf_;

  numWalls = nWalls;
  numInlets = nInlets;
  numOutlets = nOutlets;
}

void Flow::setup(double dt) {
  // HARD CODE
  partial_assembly = false;
  partial_assembly_pressure = false;
  numerical_integ = true;
  // if (verbose) grvy_printf(ginfo, "in Setup...\n");

  Vdof = vfes->GetNDofs();
  Pdof = pfes->GetNDofs();
  Sdof = sfes->GetNDofs();

  VdofInt = vfes->GetTrueVSize();
  PdofInt = pfes->GetTrueVSize();
  SdofInt = sfes->GetTrueVSize();

  // initial conditions
  // ParGridFunction *u_gf = GetCurrentVelocity();
  // ParGridFunction *p_gf = GetCurrentPressure();

  if (config->useICFunction != true) {
    Vector uic_vec;
    uic_vec.SetSize(3);
    double rho = config->initRhoRhoVp[0];
    uic_vec[0] = config->initRhoRhoVp[1] / rho;
    uic_vec[1] = config->initRhoRhoVp[2] / rho;
    uic_vec[2] = config->initRhoRhoVp[3] / rho;
    if (rank0) {
      std::cout << " Initial velocity conditions: " << uic_vec[0] << " " << uic_vec[1] << " " << uic_vec[2] << endl;
    }
    VectorConstantCoefficient u_ic_coef(uic_vec);
    // if (!config.restart) u_gf->ProjectCoefficient(u_ic_coef);
    if (!config->restart) un_gf.ProjectCoefficient(u_ic_coef);
  } else {
    VectorFunctionCoefficient u_ic_coef(dim, vel_ic);
    // if (!config.restart) u_gf->ProjectCoefficient(u_ic_coef);
    if (!config->restart) un_gf.ProjectCoefficient(u_ic_coef);
  }
  un_gf.GetTrueDofs(un);
  // std::cout << "Check 1..." << std::endl;

  Array<int> domain_attr(pmesh->attributes);
  domain_attr = 1;

  // add mean pressure gradient term
  Vector accel_vec(3);
  if (config->isForcing) {
    for (int i = 0; i < nvel; i++) {
      accel_vec[i] = config->gradPress[i];
    }
  } else {
    for (int i = 0; i < nvel; i++) {
      accel_vec[i] = 0.0;
    }
  }
  buffer_accel = new VectorConstantCoefficient(accel_vec);
  AddAccelTerm(buffer_accel, domain_attr);
  if (rank0) {
    std::cout << " Acceleration: " << accel_vec[0] << " " << accel_vec[1] << " " << accel_vec[2] << endl;
  }

  // Boundary conditions
  Array<int> attr(pmesh->bdr_attributes.Max());
  Array<int> Pattr(pmesh->bdr_attributes.Max());

  // inlet bc
  attr = 0.0;
  Pattr = 0.0;
  for (int i = 0; i < numInlets; i++) {
    for (int iFace = 1; iFace < pmesh->bdr_attributes.Max() + 1; iFace++) {
      if (iFace == config->inletPatchType[i].first) {
        attr[iFace - 1] = 1;
        // Pattr[iFace-1] = 1;
      }
    }
  }

  for (int i = 0; i < numInlets; i++) {
    if (config->inletPatchType[i].second == InletType::UNI_DENS_VEL) {
      if (rank0) std::cout << "Caught uniform inlet conditions " << endl;

      buffer_uInletInf = new ParGridFunction(vfes);
      buffer_uInlet = new ParGridFunction(vfes);
      uniformInlet();
      uInletField = new VectorGridFunctionCoefficient(buffer_uInlet);
      AddVelDirichletBC(uInletField, attr);

    } else if (config->inletPatchType[i].second == InletType::INTERPOLATE) {
      if (rank0) std::cout << "Caught interpolated inlet conditions " << endl;
      // interpolate from external file
      buffer_uInletInf = new ParGridFunction(vfes);
      buffer_uInlet = new ParGridFunction(vfes);
      interpolateInlet();
      if (rank0) std::cout << "Interpolation complete... " << endl;
      uInletField = new VectorGridFunctionCoefficient(buffer_uInlet);
      AddVelDirichletBC(uInletField, attr);

      if (rank0) std::cout << "Interpolated velocity conditions applied " << endl;
    }
  }
  if (rank0) std::cout << "Inlet bc's completed: " << numInlets << endl;

  // outlet bc
  attr = 0.0;
  Pattr = 0.0;
  for (int i = 0; i < numOutlets; i++) {
    for (int iFace = 1; iFace < pmesh->bdr_attributes.Max() + 1; iFace++) {
      if (iFace == config->outletPatchType[i].first) {
        Pattr[iFace - 1] = 1;
      }
    }
  }

  for (int i = 0; i < numOutlets; i++) {
    if (config->outletPatchType[i].second == OutletType::SUB_P) {
      if (rank0) {
        std::cout << "Caught pressure outlet bc" << endl;
      }

      // set directly from input
      double outlet_pres;
      outlet_pres = config->outletPressure;
      bufferOutlet_pbc = new ConstantCoefficient(outlet_pres);
      AddPresDirichletBC(bufferOutlet_pbc, Pattr);
    }
  }
  if (rank0) std::cout << "Outlet bc's completed: " << numOutlets << endl;

  // velocity wall bc
  Vector zero_vec(3);
  zero_vec = 0.0;
  attr = 0.0;

  buffer_ubc = new VectorConstantCoefficient(zero_vec);
  // for (int i = 0; i < config.wallPatchType.size(); i++) {
  for (int i = 0; i < numWalls; i++) {
    for (int iFace = 1; iFace < pmesh->bdr_attributes.Max() + 1; iFace++) {
      // if (config.wallPatchType[i].second != WallType::INV) {
      if (config->wallPatchType[i].second == WallType::VISC_ISOTH) {
        // std::cout << " wall check " << i << " " << iFace << " " << config->wallPatchType[i].first << endl;
        if (iFace == config->wallPatchType[i].first) {
          attr[iFace - 1] = 1;
        }
      } else if (config->wallPatchType[i].second == WallType::VISC_ADIAB) {
        // std::cout << " wall check " << i << " " << iFace << " " << config->wallPatchType[i].first << endl;
        if (iFace == config->wallPatchType[i].first) {
          attr[iFace - 1] = 1;
        }
      }
    }
  }
  AddVelDirichletBC(vel_wall, attr);
  if (rank0) std::cout << "Velocity wall bc completed: " << numWalls << endl;

  vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
  pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);
  if (rank0) std::cout << "Flow essential true dof step" << endl;

  Array<int> empty;

  // unsteady: p+p [+p] = 2p [3p]
  // convection: p+p+(p-1) [+p] = 3p-1 [4p-1]
  // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]

  // GLL integration rule (Numerical Integration)
  const IntegrationRule &ir_i = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2 * order + 1);    // 3 5
  const IntegrationRule &ir_pi = gll_rules.Get(pfes->GetFE(0)->GetGeomType(), 3 * porder - 1);  // 2 5
  const IntegrationRule &ir_nli = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 4 * norder);     // 4 8
  const IntegrationRule &ir_di = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 3 * order - 1);   // 2 5
  if (rank0) std::cout << "Integration rules set" << endl;

  // gravitational term
  Vector gravity(3);
  if (config->isGravity) {
    for (int i = 0; i < nvel; i++) {
      gravity[i] = config->gravity[i];
    }
    if (rank0) std::cout << "Adding Boussinesq gravity term" << endl;
  } else {
    for (int i = 0; i < nvel; i++) {
      gravity[i] = 0.0;
    }
    if (rank0) std::cout << "No gravity" << endl;
  }

  // setup coefficients, all updated before uses so dont have to set values here
  /*
  bufferRho = new ParGridFunction(sfes);
  {
    double *data = bufferRho->HostReadWrite();
    for (int i = 0; i < Sdof; i++) { data[i] = 1.0; }
  }
  */

  // bufferInvRho = new ParGridFunction(pfes);
  bufferInvRho.SetSpace(pfes);
  {
    double *data = bufferInvRho.HostReadWrite();
    for (int i = 0; i < Pdof; i++) {
      data[i] = 1.0;
    }
  }

  /*
  bufferVisc = new ParGridFunction(sfes);
  {
    double *data = bufferVisc->HostReadWrite();
    double *dataTmp = tmpR0.HostReadWrite();
    for (int i = 0; i < Sdof; i++) { data[i] = dataTmp[i]; }
  }
  */

  // bufferRhoDt = new ParGridFunction(sfes);
  bufferRhoDt.SetSpace(sfes);
  {
    double *data = bufferRhoDt.HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = 1.0;
    }
  }

  /*
  bufferRhoDtR1 = new ParGridFunction(vfes);
  {
    double *data = bufferRhoDtR1->HostReadWrite();
    for (int eq = 0; eq < dim; eq++) {
      for (int i = 0; i < Sdof; i++) { data[i+eq*Sdof] = 1.0; }
    }
  }
  */

  // Rho = new GridFunctionCoefficient(bufferRho);
  Rho = new GridFunctionCoefficient(rn_gf);
  invRho = new GridFunctionCoefficient(&bufferInvRho);
  // viscField = new GridFunctionCoefficient(bufferVisc);
  viscField = new GridFunctionCoefficient(visc_gf);
  rhoDtField = new GridFunctionCoefficient(&bufferRhoDt);
  // rhoDtFieldR1 = new VectorGridFunctionCoefficient(bufferRhoDtR1);

  // convection section, extrapolation
  ///// nlcoeff.constant = -1.0; // starts with negative to move to rhs
  nlcoeff.constant = +1.0;
  // N = new ParNonlinearForm(vfes);
  N = new ParNonlinearForm(nfes);
  // auto *nlc_nlfi = new SkewSymmetricVectorConvectionNLFIntegrator(nlcoeff);
  // auto *nlc_nlfi = new ConvectiveVectorConvectionNLFIntegrator(nlcoeff);
  auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff);
  // auto *nlc_nlfi = new VectorConvectionNLFIntegrator(*Rho); // make this n-size, only valid equal order
  if (numerical_integ) {
    nlc_nlfi->SetIntRule(&ir_nli);
  }
  N->AddDomainIntegrator(nlc_nlfi);
  if (partial_assembly) {
    N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    N->Setup();
  }
  if (rank0) std::cout << "Convection (NL) set" << endl;

  // mass matrix
  Mv_form = new ParBilinearForm(vfes);
  auto *mv_blfi = new VectorMassIntegrator;
  if (numerical_integ) {
    mv_blfi->SetIntRule(&ir_i);
  }
  Mv_form->AddDomainIntegrator(mv_blfi);
  if (partial_assembly) {
    Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Mv_form->Assemble();
  Mv_form->FormSystemMatrix(empty, Mv);

  // mass matrix with rho
  MvRho_form = new ParBilinearForm(vfes);
  // auto *mvrho_blfi = new VectorMassIntegrator;
  auto *mvrho_blfi = new VectorMassIntegrator(*Rho);
  if (numerical_integ) {
    mvrho_blfi->SetIntRule(&ir_i);
  }
  MvRho_form->AddDomainIntegrator(mvrho_blfi);
  if (partial_assembly) {
    MvRho_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MvRho_form->Assemble();
  MvRho_form->FormSystemMatrix(empty, MvRho);

  // mass matrix
  Ms_form = new ParBilinearForm(sfes);
  auto *ms_blfi = new MassIntegrator;
  if (numerical_integ) {
    ms_blfi->SetIntRule(&ir_i);
  }
  Ms_form->AddDomainIntegrator(ms_blfi);
  if (partial_assembly) {
    Ms_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Ms_form->Assemble();
  Ms_form->FormSystemMatrix(empty, Ms);
  // std::cout << "Check 20..." << std::endl;

  // mass matrix with rho
  MsRho_form = new ParBilinearForm(sfes);
  // auto *ms_blfi = new MassIntegrator;
  auto *msrho_blfi = new MassIntegrator(*Rho);
  if (numerical_integ) {
    msrho_blfi->SetIntRule(&ir_i);
  }
  MsRho_form->AddDomainIntegrator(msrho_blfi);
  if (partial_assembly) {
    MsRho_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  MsRho_form->Assemble();
  MsRho_form->FormSystemMatrix(empty, MsRho);

  // looks like this is the Laplacian for press eq
  Sp_form = new ParBilinearForm(pfes);
  // auto *sp_blfi = new DiffusionIntegrator;
  auto *sp_blfi = new DiffusionIntegrator(*invRho);  // HERE breaks with amg
  if (numerical_integ) {
    sp_blfi->SetIntRule(&ir_pi);
  }
  Sp_form->AddDomainIntegrator(sp_blfi);
  if (partial_assembly_pressure) {
    Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Sp_form->Assemble();
  //? Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);
  Sp_form->FormSystemMatrix(empty, Sp);
  if (rank0) std::cout << "Pressure-poisson operator set" << endl;

  // pressure boundary terms => patterning off existing stuff
  /*
  Pext_gfcoeff = new GridFunctionCoefficient(&Pext_gf);
  Pext_bdr_form = new ParLinearForm(pfes);
  auto *Pext_blfi = new BoundaryLFIntegrator(*Pext_gfcoeff);
  if (numerical_integ) { Pext_blfi->SetIntRule(&ir_i); }
  Pext_bdr_form->AddBoundaryIntegrator(Pext_blfi, pres_ess_attr);
  p_bdr_form = new ParLinearForm(pfes);
  for (auto &pres_dbc : pres_dbcs)
  {
     auto *Pbdr_blfi = new BoundaryLFIntegrator(*pres_dbc.coeff);
     if (numerical_integ) { Pbdr_blfi->SetIntRule(&ir_i); }
     p_bdr_form->AddBoundaryIntegrator(Pbdr_blfi, pres_dbc.attr);
  }
  */

  // div(u)
  D_form = new ParMixedBilinearForm(vfes, sfes);
  auto *vd_mblfi = new VectorDivergenceIntegrator();
  if (numerical_integ) {
    vd_mblfi->SetIntRule(&ir_i);
  }
  D_form->AddDomainIntegrator(vd_mblfi);
  if (partial_assembly) {
    D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  D_form->Assemble();
  D_form->FormRectangularSystemMatrix(empty, empty, D);
  if (rank0) std::cout << "Divergence operator set" << endl;

  DRho_form = new ParMixedBilinearForm(vfes, sfes);
  // auto *vdr_mblfi = new VectorDivergenceIntegrator();
  auto *vdr_mblfi = new VectorDivergenceIntegrator(*Rho);
  if (numerical_integ) {
    vdr_mblfi->SetIntRule(&ir_i);
  }
  DRho_form->AddDomainIntegrator(vdr_mblfi);
  if (partial_assembly) {
    DRho_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  DRho_form->Assemble();
  DRho_form->FormRectangularSystemMatrix(empty, empty, DRho);

  // gradient of scalar
  G_form = new ParMixedBilinearForm(sfes, vfes);
  auto *g_mblfi = new GradientIntegrator();
  if (numerical_integ) {
    g_mblfi->SetIntRule(&ir_i);
  }
  G_form->AddDomainIntegrator(g_mblfi);
  if (partial_assembly) {
    G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  G_form->Assemble();
  G_form->FormRectangularSystemMatrix(empty, empty, G);
  if (rank0) std::cout << "Gradient operator set" << endl;

  // gradient of pressure
  Gp_form = new ParMixedBilinearForm(pfes, vfes);
  auto *gp_mblfi = new GradientIntegrator();
  if (numerical_integ) {
    gp_mblfi->SetIntRule(&ir_i);
  }
  Gp_form->AddDomainIntegrator(gp_mblfi);
  if (partial_assembly) {
    Gp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  Gp_form->Assemble();
  Gp_form->FormRectangularSystemMatrix(empty, empty, Gp);
  if (rank0) std::cout << "Gradient of pressure operator set" << endl;

  // helmholtz for velocity
  // H_lincoeff.constant = config->const_visc;;
  H_bdfcoeff.constant = 1.0 / dt;
  H_form = new ParBilinearForm(vfes);
  hmv_blfi = new VectorMassIntegrator(*rhoDtField);
  // hmv_blfi = new VectorMassIntegrator(*rhoDtFieldR1);
  if (config->timeIntegratorType == 1) {
    hdv_blfi = new VectorDiffusionIntegrator(*viscField);
    // hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);
  } else {
    hev_blfi = new ElasticityIntegrator(*bulkViscField, *viscField);
  }

  if (numerical_integ) {
    hmv_blfi->SetIntRule(&ir_di);
    if (config->timeIntegratorType == 1) {
      hdv_blfi->SetIntRule(&ir_di);
    } else {
      hev_blfi->SetIntRule(&ir_di);
    }
  }
  H_form->AddDomainIntegrator(hmv_blfi);
  if (config->timeIntegratorType == 1) {
    H_form->AddDomainIntegrator(hdv_blfi);
  } else {
    H_form->AddDomainIntegrator(hev_blfi);
  }
  if (partial_assembly) {
    H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
  }
  H_form->Assemble();
  H_form->FormSystemMatrix(vel_ess_tdof, H);
  if (rank0) std::cout << "Velocity Helmholtz operator set" << endl;

  // boundary terms
  /*
  bufferFText = new ParGridFunction(vfes);
  {
    double *data = bufferFText->HostReadWrite();
    for (int i = 0; i < Vdof; i++) { data[i] = 0.0; }
  }
  FText_gfcoeff = new VectorGridFunctionCoefficient(bufferFText);
  */
  FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
  FText_bdr_form = new ParLinearForm(sfes);
  auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
  if (numerical_integ) {
    ftext_bnlfi->SetIntRule(&ir_i);
  }
  FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);

  g_bdr_form = new ParLinearForm(sfes);
  for (auto &vel_dbc : vel_dbcs) {
    auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
    if (numerical_integ) {
      gbdr_bnlfi->SetIntRule(&ir_i);
    }
    g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
  }
  if (rank0) std::cout << "Pressure-poisson rhs bc terms set" << endl;

  f_form = new ParLinearForm(vfes);
  for (auto &accel_term : accel_terms) {
    auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
    // @TODO: This order should always be the same as the nonlinear forms one!
    // const IntegrationRule &ir = IntRules.Get(vfes->GetFE(0)->GetGeomType(),
    //                                          4 * order);
    // vdlfi->SetIntRule(&ir);
    if (numerical_integ) {
      vdlfi->SetIntRule(&ir_i);
    }
    f_form->AddDomainIntegrator(vdlfi);
  }
  if (rank0) std::cout << "Acceleration terms set" << endl;

  if (partial_assembly) {
    Vector diag_pa(vfes->GetTrueVSize());
    Mv_form->AssembleDiagonal(diag_pa);
    MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MvInv = new CGSolver(vfes->GetComm());
  MvInv->iterative_mode = false;
  MvInv->SetOperator(*Mv);
  MvInv->SetPreconditioner(*MvInvPC);
  MvInv->SetPrintLevel(pl_mvsolve);
  MvInv->SetRelTol(config->solver_tol);
  MvInv->SetMaxIter(config->solver_iter);

  if (partial_assembly) {
    Vector diag_pa(sfes->GetTrueVSize());
    Ms_form->AssembleDiagonal(diag_pa);
    MsInvPC = new OperatorJacobiSmoother(diag_pa, empty);
  } else {
    MsInvPC = new HypreSmoother(*Ms.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(MsInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  MsInv = new CGSolver(sfes->GetComm());
  MsInv->iterative_mode = false;
  MsInv->SetOperator(*Ms);
  MsInv->SetPreconditioner(*MsInvPC);
  MsInv->SetPrintLevel(pl_mtsolve);
  MsInv->SetRelTol(config->solver_tol);
  MsInv->SetMaxIter(config->solver_iter);
  // std::cout << "Check 26..." << std::endl;

  /*
  if (partial_assembly_pressure) {
     lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
     SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
     SpInvPC->SetPrintLevel(pl_amg);
     SpInvPC->Mult(resp, pn);
     SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
     SpInvOrthoPC->SetSolver(*SpInvPC);
  } else {
     SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
     SpInvPC->SetPrintLevel(0);
     SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
     SpInvOrthoPC->SetSolver(*SpInvPC);
  }
  SpInv = new CGSolver(pfes->GetComm());
  SpInv->iterative_mode = true;
  SpInv->SetOperator(*Sp);
  if (pres_dbcs.empty())
  {
     SpInv->SetPreconditioner(*SpInvOrthoPC);
  }
  else
  {
     SpInv->SetPreconditioner(*SpInvPC);
  }
  SpInv->SetPrintLevel(pl_spsolve);
  SpInv->SetRelTol(config.solver_tol);
  SpInv->SetMaxIter(config.solver_iter);
  */

  // this wont be as efficient but AMG merhod (above) wasnt allowing for updates to variable coeff
  /*
  if (partial_assembly_pressure)
  {
     Vector diag_pa(pfes->GetTrueVSize());
     Sp_form->AssembleDiagonal(diag_pa);
     SpInvPC = new OperatorJacobiSmoother(diag_pa, pres_ess_tdof);
  }
  else
  {
     SpInvPC = new HypreSmoother(*Sp.As<HypreParMatrix>());
     dynamic_cast<HypreSmoother *>(SpInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  SpInv = new CGSolver(pfes->GetComm());
  SpInv->iterative_mode = true;
  SpInv->SetOperator(*Sp);
  SpInv->SetPreconditioner(*SpInvPC);
  SpInv->SetPrintLevel(pl_spsolve);
  SpInv->SetRelTol(rtol_spsolve);
  SpInv->SetMaxIter(1000);
  */

  if (partial_assembly_pressure) {
    lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
    SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
    // SpInvPC->SetPrintLevel(pl_amg);
    SpInvPC->Mult(resp, pn);
    SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
    SpInvOrthoPC->SetSolver(*SpInvPC);
  } else {
    // SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
    // SpInvPC->SetPrintLevel(0);
    SpInvPC = new HypreSmoother(*Sp.As<HypreParMatrix>());
    // dynamic_cast<HypreSmoother *>(SpInvPC)->SetType(HypreSmoother::Jacobi, 1);
    dynamic_cast<HypreSmoother *>(SpInvPC)->SetType(HypreSmoother::GS, 1);
    SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
    SpInvOrthoPC->SetSolver(*SpInvPC);
    // Jacobi, l1Jacobi, l1GS, l1GStr, lumpedJacobi, GS, OPFS, Chebyshev, Taubin, FIR
  }
  SpInv = new CGSolver(pfes->GetComm());
  // SpInv = new GMRESSolver(vfes->GetComm());
  // SpInv = new BiCGSTABSolver(vfes->GetComm());
  SpInv->iterative_mode = true;
  SpInv->SetOperator(*Sp);
  if (pres_dbcs.empty()) {
    SpInv->SetPreconditioner(*SpInvOrthoPC);
  } else {
    SpInv->SetPreconditioner(*SpInvPC);
  }
  SpInv->SetPrintLevel(pl_spsolve);
  SpInv->SetRelTol(config->solver_tol);
  SpInv->SetMaxIter(2000);

  if (partial_assembly) {
    Vector diag_pa(vfes->GetTrueVSize());
    H_form->AssembleDiagonal(diag_pa);
    HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
  } else {
    HInvPC = new HypreSmoother(*H.As<HypreParMatrix>());
    dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
  }
  HInv = new CGSolver(vfes->GetComm());
  // HInv = new GMRESSolver(vfes->GetComm());
  // HInv = new BiCGSTABSolver(vfes->GetComm());
  HInv->iterative_mode = true;
  HInv->SetOperator(*H);
  HInv->SetPreconditioner(*HInvPC);
  HInv->SetPrintLevel(pl_hsolve);
  HInv->SetRelTol(config->solver_tol);
  HInv->SetMaxIter(config->solver_iter);

  if (rank0) std::cout << "Inverse operators set" << endl;

  // If the initial condition was set, it has to be aligned with dependent
  // Vectors and GridFunctions

  un_gf.GetTrueDofs(un);  // ~ un = un_gf
  // un_next = un; // invalid read
  {
    double *data = un_next.HostReadWrite();
    double *Udata = un.HostReadWrite();
    for (int i = 0; i < VdofInt; i++) {
      data[i] = Udata[i];
    }
  }
  un_next_gf.SetFromTrueDofs(un_next);  // ~ un_next_gf = un_next

  // Velocity filter
  filter_alpha = loMach_opts->filterWeight;
  filter_cutoff_modes = loMach_opts->nFilter;
  if (loMach_opts->filterVel == true) {
    vfec_filter = new H1_FECollection(order - filter_cutoff_modes, pmesh->Dimension());
    vfes_filter = new ParFiniteElementSpace(pmesh, vfec_filter, pmesh->Dimension());

    un_NM1_gf.SetSpace(vfes_filter);
    un_NM1_gf = 0.0;

    un_filtered_gf.SetSpace(vfes);
    un_filtered_gf = 0.0;
  }
}

void Flow::UpdateTimestepHistory(double dt) {
  // Shift values in nonlinear extrapolation history
  Nunm2 = Nunm1;
  Nunm1 = Nun;

  // Shift values in solution history
  unm2 = unm1;
  unm1 = un;

  // Update the current solution and corresponding GridFunction
  un_next_gf.GetTrueDofs(un_next);
  un = un_next;
  un_gf.SetFromTrueDofs(un);
}

void Flow::flowStep(double &time, double dt, const int current_step, const int start_step,
                    std::vector<double> bdf)  //, bool provisional)
{
  // if(rank0) {std::cout << "wtf?" << endl;}
  Array<int> empty;

  // update bdf coeffs
  bd0 = bdf[0];
  bd1 = bdf[1];
  bd2 = bdf[2];
  bd3 = bdf[3];
  // if(rank0) {std::cout << "time: " << time << " dt: " << dt << endl;}
  // if(rank0) {std::cout << "bdfs: " << bd0 << " " << bd1 << " " << bd2 << " " << bd3 << endl;}

  // testing...
  // Qt = 0.0;
  // Qt_gf->SetFromTrueDofs(Qt);

  // update vectors from external data and gradients
  rn_gf->GetTrueDofs(rn);
  Qt_gf->GetTrueDofs(Qt);
  visc_gf->GetTrueDofs(visc);

  // gradients of external
  // G->Mult(rn, tmpR1);
  // MvInv->Mult(tmpR1, gradRho);
  G->Mult(visc, tmpR1);
  MvInv->Mult(tmpR1, gradMu);
  divU.Set(1.0, Qt);
  G->Mult(divU, tmpR1);
  MvInv->Mult(tmpR1, gradDivU);
  // if(rank0) {std::cout << "update block okay" << endl;}

  // update coefficients of operators
  /*
  {
    double *data = bufferRho->HostReadWrite();
    double *dataR = rn_gf->HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = dataR[i];
      //if(dataR[i]!=config->const_dens) {std::cout << "rho in flow: " << dataR[i] << " at dof: " << i << endl;}
    }
  }
  */
  {
    R0PM0_gf.SetFromTrueDofs(rn);
    R0PM1_gf.ProjectGridFunction(R0PM0_gf);
    // double *data = bufferInvRho->HostReadWrite();
    double *data = bufferInvRho.HostReadWrite();
    double *dataR = R0PM1_gf.HostReadWrite();
    for (int i = 0; i < Pdof; i++) {
      data[i] = 1.0 / dataR[i];
    }
  }
  /*
  {
    double *data = bufferVisc->HostReadWrite();
    double *dataMu = visc_gf->HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = dataMu[i];
      //if(dataMu[i]!=config->const_visc) {std::cout << "mu in flow: " << dataMu[i] << " at dof: " << i << endl; }
      //std::cout << "mu in flow: " << dataMu[i] << " at dof: " << i << endl;
    }
  }
  */
  {
    // double *data = bufferRhoDt->HostReadWrite();
    double *data = bufferRhoDt.HostReadWrite();
    double *dataR = rn_gf->HostReadWrite();
    for (int i = 0; i < Sdof; i++) {
      data[i] = dataR[i] / dt;
    }
  }
  /*
  {
    double *data = bufferRhoDtR1->HostReadWrite();
    double *dataR = rn_gf->HostReadWrite();
    for (int eq = 0; eq < dim; eq++) {
      for (int i = 0; i < Sdof; i++) { data[i+eq*Sdof] = dataR[i] / dt; }
    }
  }
  */
  // if(rank0) {std::cout << "update coeff okay" << endl;}

  // Set current time for velocity Dirichlet boundary conditions.
  for (auto &vel_dbc : vel_dbcs) {
    vel_dbc.coeff->SetTime(time + dt);
  }
  for (auto &pres_dbc : pres_dbcs) {
    pres_dbc.coeff->SetTime(time + dt);
  }

  // zero rhs vectors <warp>
  fn = 0.0;
  Fext = 0.0;
  FText = 0.0;
  uns = 0.0;
  LdivImp = 0.0;
  Ldiv = 0.0;

  // total forcing
  computeExplicitForcing();  // ->fn

  // convection in cont form
  // computeExplicitConvection(1.0); // ->Fext
  // computeExplicitConvectionOP(1.0,false); // use extrap u
  computeExplicitConvectionOP(0.0, true);  // uses extrap nl div product

  // total unsteady contribution
  computeExplicitUnsteadyBDF(dt);  // ->uns

  // implicit part of diff term (calculated explicitly) using elasticity op
  computeImplicitDiffusion();  // ->LdivImp (has rho via mu)

  // explicit part of diff term using elasticity op
  computeExplicitDiffusion();  // ->Ldiv (has rho via mu)

  // if(rank0) {std::cout << "rhs parts okay?" << endl;}

  // sum contributions to rhs of p-p, these do not have density
  FText.Set(1.0, fn);
  FText.Add(1.0, uns);
  FText.Add(1.0, Fext);  // if rho not added to the integrator

  // tmpR1.Set(1.0,FText);
  // D->Mult(tmpR1,tmpR0); // with 1/rho in p-p op
  // if(rank0) {std::cout << "div 1 okay" << endl;}

  // tmpR1 = 0.0;
  tmpR1.Set(1.0, Ldiv);
  tmpR1.Add(1.0, LdivImp);
  multScalarInvVectorIP(rn, &tmpR1);  // with 1/rho in p-p op because viscous terms use mu
  FText.Add(1.0, tmpR1);
  D->Mult(FText, tmpR0);
  // if(rank0) {std::cout << "div 2 okay" << endl;}

  // neg necessary because of the negative from IBP on the Sp Laplacian operator
  // tmpR1.Set(-1.0,FText);
  tmpR0.Neg();
  // if(rank0) {std::cout << "okay 2" << endl;}

  // unsteady div term, (-) on rhs rho * gamma * Qt/dt (after div term)
  multConstScalar((bd0 / dt), Qt, &tmpR0b);
  Ms->Mult(tmpR0b, tmpR0c);  // with 1/rho in p-p op
  tmpR0.Add(+1.0, tmpR0c);   // because of the tmpR0.Neg()...
  // if(rank0) {std::cout << "okay 3" << endl;}

  // store old p
  pnm1.Set(1.0, pn);
  // MPI_Barrier(MPI_COMM_WORLD);
  // if(rank0) {std::cout << "okay 4" << endl;}

  // Add boundary terms.
  /*
  FText_gf.SetFromTrueDofs(FText);
  {
    double *dFT = FText_gf.HostReadWrite();
    double *data = bufferFText->HostReadWrite();
    for (int i = 0; i < Vdof; i++) { data[i] = dFT[i]; }
  }
  FText_bdr_form->Update();
  */

  FText_bdr_form->Assemble();
  FText_bdr_form->ParallelAssemble(FText_bdr);
  tmpR0.Add(1.0, FText_bdr);

  g_bdr_form->Assemble();
  g_bdr_form->ParallelAssemble(g_bdr);
  tmpR0.Add((-bd0 / dt), g_bdr);

  // testing!!!
  // tmpR0 = 0.0;

  // project rhs to p-space
  R0PM0_gf.SetFromTrueDofs(tmpR0);
  R0PM1_gf.ProjectGridFunction(R0PM0_gf);
  R0PM1_gf.GetTrueDofs(resp);
  // if(rank0) {std::cout << "okay 6" << endl;}

  // testing!!!
  // resp = 0.0;

  Sp_form->Update();
  Sp_form->Assemble();
  Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);
  SpInv->SetOperator(*Sp);

  if (pres_dbcs.empty()) {
    Orthogonalize(resp, sfes->GetComm());
  }
  for (auto &pres_dbc : pres_dbcs) {
    pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
  }
  pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

  Vector X1, B1;
  if (partial_assembly_pressure) {
    auto *SpC = Sp.As<ConstrainedOperator>();
    EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
  } else {
    Sp_form->FormLinearSystem(pres_ess_tdof, pn_gf, resp_gf, Sp, X1, B1, 1);
  }

  // actual implicit solve for p(n+1)
  sw_spsolve.Start();
  SpInv->Mult(B1, X1);  // uninitialised? how?
  sw_spsolve.Stop();
  iter_spsolve = SpInv->GetNumIterations();
  res_spsolve = SpInv->GetFinalNorm();
  Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

  // If the boundary conditions on the pressure are pure Neumann remove the
  // nullspace by removing the mean of the pressure solution. This is also
  // ensured by the OrthoSolver wrapper for the preconditioner which removes
  // the nullspace after every application.
  if (pres_dbcs.empty()) {
    MeanZero(pn_gf);
  }
  MPI_Barrier(vfes->GetComm());
  // if(rank0) {std::cout << "Pressure solve complete in " << iter_spsolve << " iters" << endl;}

  pn_gf.GetTrueDofs(pn);

  // using mixed orders op directly
  Gp->Mult(pn, resu);
  resu.Neg();

  // assemble rhs for u solve
  tmpR1.Set(1.0, uns);
  tmpR1.Add(1.0, fn);
  tmpR1.Add(1.0, Fext);  // if rho inc in coeff
  MvRho_form->Update();
  MvRho_form->Assemble();
  MvRho_form->FormSystemMatrix(empty, MvRho);
  MvRho->Mult(tmpR1, tmpR1b);
  resu.Add(1.0, tmpR1b);
  Mv->Mult(Ldiv, tmpR1);
  resu.Add(1.0, tmpR1);
  // if(rank0) {std::cout << "okay 7" << endl;}

  // for c-n
  // Mv->Mult(LdivImp,tmpR1);
  // resu.Add(0.5,tmpR1);

  H_bdfcoeff.constant = 1.0 / dt;
  H_form->Update();
  H_form->Assemble();
  H_form->FormSystemMatrix(vel_ess_tdof, H);

  HInv->SetOperator(*H);
  if (partial_assembly) {
    delete HInvPC;
    Vector diag_pa(vfes->GetTrueVSize());
    H_form->AssembleDiagonal(diag_pa);
    HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
    HInv->SetPreconditioner(*HInvPC);
  }

  for (auto &vel_dbc : vel_dbcs) {
    un_next_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
  }
  vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

  Vector X2, B2;
  if (partial_assembly) {
    auto *HC = H.As<ConstrainedOperator>();
    EliminateRHS(*H_form, *HC, vel_ess_tdof, un_next_gf, resu_gf, X2, B2, 1);
  } else {
    H_form->FormLinearSystem(vel_ess_tdof, un_next_gf, resu_gf, H, X2, B2, 1);
  }
  sw_hsolve.Start();
  HInv->Mult(B2, X2);
  sw_hsolve.Stop();
  iter_hsolve = HInv->GetNumIterations();
  res_hsolve = HInv->GetFinalNorm();
  H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);
  un_next_gf.GetTrueDofs(un_next);

  MPI_Barrier(vfes->GetComm());
  // if(rank0) {std::cout << "Helmholtz solve complete in " << iter_hsolve << " iters" << endl;}

  // explicit filter
  if (loMach_opts->filterVel == true) {
    un_NM1_gf.ProjectGridFunction(un_next_gf);
    un_filtered_gf.ProjectGridFunction(un_NM1_gf);
    const auto d_un_filtered_gf = un_filtered_gf.Read();
    auto d_un_gf = un_next_gf.ReadWrite();
    const auto filter_alpha_ = filter_alpha;
    MFEM_FORALL(i, un_next_gf.Size(),
                { d_un_gf[i] = (1.0 - filter_alpha_) * d_un_gf[i] + filter_alpha_ * d_un_filtered_gf[i]; });
    un_next_gf.GetTrueDofs(un_next);
  }

  // If the current time step is not provisional, accept the computed solution
  // and update the time step history by default.
  // if (!provisional)
  //{
  UpdateTimestepHistory(dt);
  // time += dt; leave this for the loMach UTH
  //}

  int iflag = 0;
  if (iter_spsolve == config->solver_iter || iter_hsolve == config->solver_iter) {
    iflag = 1;
  }

  if (rank0 && iflag == 1) {
    mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8) << "Resid" << std::setw(12) << "Reltol"
              << "\n";
    // If numerical integration is active, there is no solve (thus no
    // iterations), on the inverse velocity mass application.
    if (!numerical_integ) {
      mfem::out << std::setw(5) << "MVIN " << std::setw(5) << std::fixed << iter_mvsolve << "   " << std::setw(3)
                << std::setprecision(2) << std::scientific << res_mvsolve << "   " << 1e-12 << "\n";
    }
    mfem::out << std::setw(5) << "PRES " << std::setw(5) << std::fixed << iter_spsolve << "   " << std::setw(3)
              << std::setprecision(2) << std::scientific << res_spsolve << "   " << rtol_spsolve << "\n";
    mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed << iter_hsolve << "   " << std::setw(3)
              << std::setprecision(2) << std::scientific << res_hsolve << "   " << rtol_hsolve << "\n";
    mfem::out << std::setprecision(8);
    mfem::out << std::fixed;
  }

  if (iflag == 1) {
    if (rank0) {
      grvy_printf(GRVY_ERROR, "[ERROR] Solution not converging... \n");
    }
    exit(1);
  }
}

void Flow::MeanZero(ParGridFunction &v) {
  // Make sure not to recompute the inner product linear form every
  // application.
  if (mass_lf == nullptr) {
    onecoeff.constant = 1.0;
    mass_lf = new ParLinearForm(v.ParFESpace());
    auto *dlfi = new DomainLFIntegrator(onecoeff);
    if (numerical_integ) {
      const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order - 1);
      dlfi->SetIntRule(&ir_ni);
    }
    mass_lf->AddDomainIntegrator(dlfi);
    mass_lf->Assemble();

    ParGridFunction one_gf(v.ParFESpace());
    one_gf.ProjectCoefficient(onecoeff);

    volume = mass_lf->operator()(one_gf);
  }

  double integ = mass_lf->operator()(v);

  v -= integ / volume;
}

/*
void Flow::EliminateRHS(Operator &A,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A.GetOutputProlongation();
   const Operator *Pi = A.GetProlongation();
   const Operator *Ri = A.GetRestriction();
   A.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   constrainedA.EliminateRHS(X, B);
}


void Flow::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pfes->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}
*/

/*
void Flow::ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
{

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

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      //int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
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

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
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
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}
*/

void Flow::computeExplicitDiffusion() {
  // TOTAL:
  // d_j[mu*(d_ju_i + d_iu_j - 2/3(d_ku_k)delta_ij)]
  // mu*d_j[(d_ju_i + d_iu_j - 2/3(d_ku_k)delta_ij)] + d_j(mu)*[(d_ju_i + d_iu_j - 2/3(d_ku_k)delta_ij)]
  // mu*d_j(d_ju_i) + d_j(mu)*(d_ju_i) + mu*d_j[(d_iu_j - 2/3(d_ku_k)delta_ij)] + d_j(mu)*[(d_iu_j -
  // 2/3(d_ku_k)delta_ij)] mu*d_j(d_ju_i) + d_j(mu)*(d_ju_i) + mu*1/3*d_i(d_ku_k) + d_j(mu)*[(d_iu_j -
  // 2/3(d_ku_k)delta_ij)]
  // ---------------------------------   -------------------------------------------------------------
  //             IMPLICIT                                          EXPLICIT
  //
  //
  // PARTS..............................................
  // Included in integrated weak form on lhs:
  // mu*nabla(u_i)
  // d_j(mu)d_j(u_i)
  //
  // explicit on rhs:
  // 1. d_j(mu)*d_i(u_j)
  // 2. 1/3*(mu)*d_i(divU)
  // 3. (d_i(mu))*(-2/3*divU)

  // PART 1: d_j(mu)*d_i(u_j)
  {
    dotVector(gradMu, gradX, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 0 * SdofInt] = dataB[i];
    }
  }
  {
    dotVector(gradMu, gradY, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 1 * SdofInt] = dataB[i];
    }
  }
  {
    dotVector(gradMu, gradZ, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 2 * SdofInt] = dataB[i];
    }
  }
  Ldiv.Set(1.0, tmpR1);

  // PART 2: 1/3*(rho*nu)*d_i(divU)
  multScalarVector(visc, gradDivU, &tmpR1b);
  Ldiv.Add((1.0 / 3.0), tmpR1b);

  // PART 3: (d_i(mu))*(-2/3*divU)
  multScalarVector(divU, gradMu, &tmpR1a);
  Ldiv.Add((-2.0 / 3.0), tmpR1a);
}

void Flow::computeImplicitDiffusion() {
  // PARTS:
  // 1. d_j(rho*nu)d_j(u_i)
  // 2. (rho*nu)*nabla(u_i) =
  //    a) (rho*nu)*d_i(divU) -
  //    b) (rho*nu)*curl(curl(u_i))

  // PART 1: d_j(rho*nu)d_j(u_i)
  {
    dotVector(gradMu, gradU, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 0 * SdofInt] = dataB[i];
    }
  }
  {
    dotVector(gradMu, gradV, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 1 * SdofInt] = dataB[i];
    }
  }
  {
    dotVector(gradMu, gradW, &tmpR0);
    double *dataB = tmpR0.HostReadWrite();
    double *data = tmpR1.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 2 * SdofInt] = dataB[i];
    }
  }
  LdivImp.Set(1.0, tmpR1);

  // PART 2a: (rho*nu)*d_i(divU)
  multScalarVector(visc, gradDivU, &tmpR1b);
  LdivImp.Add(1.0, tmpR1b);

  // PART 2b: -(rho*nu)*curl(curl(u_i))
  Lext_gf.SetFromTrueDofs(Uext);
  if (dim == 2) {
    ComputeCurl2D(Lext_gf, curlu_gf);
    ComputeCurl2D(curlu_gf, curlcurlu_gf, true);
  } else {
    ComputeCurl3D(Lext_gf, curlu_gf);

    /*
    {
      double *dU = gradU.HostReadWrite();
      double *dV = gradV.HostReadWrite();
      double *dW = gradW.HostReadWrite();
      double *data = tmpR1.HostReadWrite();

      DenseMatrix gradU;
      gradU.SetSize(nvel, dim);

      for (int eq = 0; eq < dim; eq++) {
        for (int i = 0; i < SdofInt; i++) {
          data[i + eq*SdofInt] = 0.0;
        }
      }
      for (int i = 0; i < SdofInt; i++) {
        for (int dir = 0; dir < dim; dir++) { gradU(0,dir) = dU[i + dir * SdofInt]; }
        for (int dir = 0; dir < dim; dir++) { gradU(1,dir) = dV[i + dir * SdofInt]; }
        for (int dir = 0; dir < dim; dir++) { gradU(2,dir) = dW[i + dir * SdofInt]; }
        data[i + 0*SdofInt] = gradU(2,1) - gradU(1,2);
        data[i + 1*SdofInt] = gradU(0,2) - gradU(2,0);
        data[i + 2*SdofInt] = gradU(1,0) - gradU(0,1);
      }
    }
    curlu_gf.SetFromTrueDofs(tmpR1);
    */

    ComputeCurl3D(curlu_gf, curlcurlu_gf);

    // gradient of curl
    /*
    {
      double *dBuffer = tmpR0.HostReadWrite();
      double *dCurl = tmpR1.HostReadWrite();
      for (int i = 0; i < SdofInt; i++) { dBuffer[i] = dCurl[i + 0*SdofInt]; }
      G->Mult(tmpR0, tmpR1b);
      MvInv->Mult(tmpR1b, tmpR1a);
      for (int i = 0; i < SdofInt; i++) { dBuffer[i] = dCurl[i + 1*SdofInt]; }
      G->Mult(tmpR0, tmpR1c);
      MvInv->Mult(tmpR1c, tmpR1b);
      for (int i = 0; i < SdofInt; i++) { dBuffer[i] = dCurl[i + 2*SdofInt]; }
      G->Mult(tmpR0, tmpR1); // can overwrite tmpR1 now
      MvInv->Mult(tmpR1, tmpR1c);
   }
   {
      double *dU = tmpR1a.HostReadWrite();
      double *dV = tmpR1b.HostReadWrite();
      double *dW = tmpR1c.HostReadWrite();
      double *data = tmpR1.HostReadWrite();
      DenseMatrix gradU;
      gradU.SetSize(nvel, dim);

      for (int eq = 0; eq < dim; eq++) {
        for (int i = 0; i < SdofInt; i++) {
          data[i + eq*SdofInt] = 0.0;
        }
      }
      for (int i = 0; i < SdofInt; i++) {
        for (int dir = 0; dir < dim; dir++) { gradU(0,dir) = dU[i + dir * SdofInt]; }
        for (int dir = 0; dir < dim; dir++) { gradU(1,dir) = dV[i + dir * SdofInt]; }
        for (int dir = 0; dir < dim; dir++) { gradU(2,dir) = dW[i + dir * SdofInt]; }
        data[i + 0*SdofInt] = gradU(2,1) - gradU(1,2);
        data[i + 1*SdofInt] = gradU(0,2) - gradU(2,0);
        data[i + 2*SdofInt] = gradU(1,0) - gradU(0,1);
      }
    }
    curlcurlu_gf.SetFromTrueDofs(tmpR1);
    */
  }

  curlcurlu_gf.GetTrueDofs(Lext);
  multScalarVector(visc, Lext, &tmpR1b);
  LdivImp.Add(-1.0, tmpR1b);
}

void Flow::computeExplicitForcing() {
  // Extrapolated f^{n+1} (source term: f(n+1))
  /*
  for (auto &accel_term : accel_terms) {
    accel_term.coeff->SetTime(time + dt);
  }
  f_form->Assemble();
  f_form->ParallelAssemble(fn);
  MvInv->Mult(fn, tmpR1c);
  multScalarVector(rn,tmpR1c,&fn);
  */

  {
    double *data = fn.HostReadWrite();
    // double *Rdata = rn.HostReadWrite();
    double tol = 1.0e-8;
    for (int eq = 0; eq < nvel; eq++) {
      for (int i = 0; i < SdofInt; i++) {
        data[i + eq * SdofInt] = config->gradPress[eq];
        // data[i + eq*SdofInt] = Rdata[i] * config.gradPress[eq];
        /*
        if ( (data[i + eq*SdofInt]+tol) <= Rdata[i] * config.gradPress[eq] ||
             (data[i + eq*SdofInt]-tol) >= Rdata[i] * config.gradPress[eq] ) {
        std:cout << " FORCING FUNKY: " << data[i + eq*SdofInt] << " vs. " << Rdata[i] * config.gradPress[eq] << endl;
        }
        */
      }
    }
  }

  // gravity term
  if (config->isGravity) {
    double *data = boussinesqField.HostReadWrite();
    // double *Rdata = rn.HostReadWrite();
    for (int eq = 0; eq < nvel; eq++) {
      for (int i = 0; i < SdofInt; i++) {
        // data[i + eq*SdofInt] = Rdata[i] * gravity[eq];
        // data[i + eq*SdofInt] = gravity[eq];
        data[i + eq * SdofInt] = config->gravity[eq];
      }
    }
    // Mv->Mult(tmpR1,boussinesqField);
  }

  // add together
  fn.Add(1.0, boussinesqField);
}

void Flow::computeExplicitUnsteady(double dt) {
  {
    const auto d_un = un.Read();
    auto data = uns.ReadWrite();
    // postive on rhs
    MFEM_FORALL(i, VdofInt, { data[i] = d_un[i] / dt; });
  }
  multScalarVectorIP(rn, &uns);
}

void Flow::computeExplicitUnsteadyBDF(double dt) {
  // negative on bd's to move to rhs
  {
    const double bd1idt = -bd1 / dt;
    const double bd2idt = -bd2 / dt;
    const double bd3idt = -bd3 / dt;
    const auto d_un = un.Read();
    const auto d_unm1 = unm1.Read();
    const auto d_unm2 = unm2.Read();
    auto data = uns.ReadWrite();
    MFEM_FORALL(i, uns.Size(), {
      data[i] = bd1idt * d_un[i] + bd2idt * d_unm1[i] + bd3idt * d_unm2[i];
      if (data[i] != data[i]) {
        std::cout << "dt: " << dt << endl;
        std::cout << "bds: " << bd1idt << " " << bd2idt << " " << bd3idt << endl;
        std::cout << "uns: " << d_un[i] << " " << d_unm1[i] << " " << d_unm2[i] << endl;
      }
    });
  }
}

void Flow::computeExplicitConvectionOP(double uStep, bool extrap) {
  if (uStep == 0.0) {
    un_next_gf.SetFromTrueDofs(un);
  } else if (uStep == 0.5) {
    un_next_gf.SetFromTrueDofs(u_half);
  } else {
    un_next_gf.SetFromTrueDofs(un_next);
  }
  R1PX2_gf.ProjectGridFunction(un_next_gf);
  R1PX2_gf.GetTrueDofs(uBn);

  N->Mult(uBn, r1px2a);

  R1PX2_gf.SetFromTrueDofs(r1px2a);
  R1PM0_gf.ProjectGridFunction(R1PX2_gf);
  R1PM0_gf.GetTrueDofs(tmpR1);

  MvInv->Mult(tmpR1, Nun);
  // multScalarVector(rn,tmpR1,&Nun);

  // ab predictor
  if (extrap == true) {
    const auto d_Nun = Nun.Read();
    const auto d_Nunm1 = Nunm1.Read();
    const auto d_Nunm2 = Nunm2.Read();
    auto d_Fext = Fext.Write();
    const auto ab1_ = ab1;
    const auto ab2_ = ab2;
    const auto ab3_ = ab3;
    MFEM_FORALL(i, Fext.Size(), { d_Fext[i] = ab1_ * d_Nun[i] + ab2_ * d_Nunm1[i] + ab3_ * d_Nunm2[i]; });
    Fext.Neg();  /// changed
  } else {
    // Fext.Set(1.0,Nun); /// sign positive because N has a negative (change this stupid shit)
    Fext.Set(-1.0, Nun);  /// changed
  }
}

void Flow::computeExplicitConvection(double uStep) {
  // PART 1: rho * u_j * d_j (u_i)
  if (uStep == 0.0) {
    dotVector(un, gradU, &tmpR0a);
    dotVector(un, gradV, &tmpR0b);
    dotVector(un, gradW, &tmpR0c);
  } else if (uStep == 0.5) {
    dotVector(u_half, gradU, &tmpR0a);
    dotVector(u_half, gradV, &tmpR0b);
    dotVector(u_half, gradW, &tmpR0c);
  } else {
    dotVector(un_next, gradU, &tmpR0a);
    dotVector(un_next, gradV, &tmpR0b);
    dotVector(un_next, gradW, &tmpR0c);
  }

  {
    double *dataA = tmpR0a.HostReadWrite();
    double *dataB = tmpR0b.HostReadWrite();
    double *dataC = tmpR0c.HostReadWrite();
    double *data = Fext.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i + 0 * SdofInt] = dataA[i];
    }
    for (int i = 0; i < SdofInt; i++) {
      data[i + 1 * SdofInt] = dataB[i];
    }
    for (int i = 0; i < SdofInt; i++) {
      data[i + 2 * SdofInt] = dataC[i];
    }
  }
  tmpR1c.Set(-1.0, Fext);
  multScalarVector(rn, tmpR1c, &Fext);  // (rho*u)*nabla(u)
}

void Flow::updateGradients(double uStep) {
  // select desired u step
  if (uStep == 0) {
    un_next_gf.SetFromTrueDofs(un);
  } else if (uStep == 0.5) {
    un_next_gf.SetFromTrueDofs(u_half);
  } else {
    un_next_gf.SetFromTrueDofs(Uext);
  }

  // gradient of velocity
  vectorGrad3D(un_next_gf, R1PM0a_gf, R1PM0b_gf, R1PM0c_gf);
  R1PM0a_gf.GetTrueDofs(gradU);
  R1PM0b_gf.GetTrueDofs(gradV);
  R1PM0c_gf.GetTrueDofs(gradW);

  // shifted storage for efficient access later
  {
    double *dataX = gradX.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 0 * SdofInt] = dataU[i + 0 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 1 * SdofInt] = dataV[i + 0 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 2 * SdofInt] = dataW[i + 0 * SdofInt];
    }
  }
  {
    double *dataY = gradY.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 0 * SdofInt] = dataU[i + 1 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 1 * SdofInt] = dataV[i + 1 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 2 * SdofInt] = dataW[i + 1 * SdofInt];
    }
  }
  {
    double *dataZ = gradZ.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 0 * SdofInt] = dataU[i + 2 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 1 * SdofInt] = dataV[i + 2 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 2 * SdofInt] = dataW[i + 2 * SdofInt];
    }
  }

  // divergence of velocity
  {
    double *dataGradU = gradU.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] = dataGradU[i + 0 * SdofInt];
    }
  }
  {
    double *dataGradU = gradV.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] += dataGradU[i + 1 * SdofInt];
    }
  }
  {
    double *dataGradU = gradW.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] += dataGradU[i + 2 * SdofInt];
    }
  }

  // additional gradients
  // scalarGrad3DV(viscSml, &gradMu);
  // scalarGrad3DV(rn, &gradRho);
  scalarGrad3DV(sfes, vfes, divU, &gradDivU);
}

void Flow::updateGradientsOP(double uStep) {
  // gradient of velocity
  {
    double *dataU;
    if (uStep == 0.0) {
      dataU = un.HostReadWrite();
    } else if (uStep == 0.5) {
      dataU = u_half.HostReadWrite();
    } else {
      dataU = un_next.HostReadWrite();
    }

    double *dBuffer = tmpR0.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dBuffer[i] = dataU[i + 0 * SdofInt];
    }
    G->Mult(tmpR0, tmpR1);
    MvInv->Mult(tmpR1, gradU);

    for (int i = 0; i < SdofInt; i++) {
      dBuffer[i] = dataU[i + 1 * SdofInt];
    }
    G->Mult(tmpR0, tmpR1);
    MvInv->Mult(tmpR1, gradV);

    for (int i = 0; i < SdofInt; i++) {
      dBuffer[i] = dataU[i + 2 * SdofInt];
    }
    G->Mult(tmpR0, tmpR1);
    MvInv->Mult(tmpR1, gradW);
  }

  // shifted storage for efficient access later
  {
    double *dataX = gradX.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 0 * SdofInt] = dataU[i + 0 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 1 * SdofInt] = dataV[i + 0 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataX[i + 2 * SdofInt] = dataW[i + 0 * SdofInt];
    }
  }
  {
    double *dataY = gradY.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 0 * SdofInt] = dataU[i + 1 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 1 * SdofInt] = dataV[i + 1 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataY[i + 2 * SdofInt] = dataW[i + 1 * SdofInt];
    }
  }
  {
    double *dataZ = gradZ.HostReadWrite();
    double *dataU = gradU.HostReadWrite();
    double *dataV = gradV.HostReadWrite();
    double *dataW = gradW.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 0 * SdofInt] = dataU[i + 2 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 1 * SdofInt] = dataV[i + 2 * SdofInt];
    }
    for (int i = 0; i < SdofInt; i++) {
      dataZ[i + 2 * SdofInt] = dataW[i + 2 * SdofInt];
    }
  }

  // divergence of velocity
  {
    double *dataGradU = gradU.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] = dataGradU[i + 0 * SdofInt];
    }
  }
  {
    double *dataGradU = gradV.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] += dataGradU[i + 1 * SdofInt];
    }
  }
  {
    double *dataGradU = gradW.HostReadWrite();
    double *data = divU.HostReadWrite();
    for (int i = 0; i < SdofInt; i++) {
      data[i] += dataGradU[i + 2 * SdofInt];
    }
  }

  // divergence of velocity gradient
  G->Mult(divU, tmpR1);
  MvInv->Mult(tmpR1, gradDivU);

  // copy to gf for exporting
  gradU_gf.SetFromTrueDofs(gradU);
  gradV_gf.SetFromTrueDofs(gradV);
  gradW_gf.SetFromTrueDofs(gradW);
}

void Flow::updateBC(int current_step) {
  if (numInlets > 0) {
    double *duInf = buffer_uInletInf->HostReadWrite();
    double *du = buffer_uInlet->HostReadWrite();
    int nRamp = config->rampStepsInlet;
    // double wt = std::min(time/tRamp, 1.0);
    double wt = std::pow(std::min((double)current_step / (double)nRamp, 1.0), 1.0);
    // double wt = 0.5 * (tanh((time-tRamp)/tRamp) + 1.0);
    // if(rank0) {std::cout << "Inlet ramp weight: " << wt << " using ramp steps " << nRamp << endl;}
    for (int i = 0; i < Sdof; i++) {
      for (int eq = 0; eq < nvel; eq++) {
        du[i + eq * Sdof] =
            wt * duInf[i + eq * Sdof] + (1.0 - wt) * config->initRhoRhoVp[eq + 1] / config->initRhoRhoVp[0];
      }
    }
  }
}

void Flow::extrapolateState(int current_step, std::vector<double> ab) {
  // update ab coeffs
  ab1 = ab[0];
  ab2 = ab[1];
  ab3 = ab[2];

  // extrapolated velocity at {n+1}
  {
    const auto d_un = un.Read();
    const auto d_unm1 = unm1.Read();
    const auto d_unm2 = unm2.Read();
    auto d_Uext = Uext.Write();
    const auto ab1_ = ab1;
    const auto ab2_ = ab2;
    const auto ab3_ = ab3;
    MFEM_FORALL(i, Uext.Size(), { d_Uext[i] = ab1_ * d_un[i] + ab2_ * d_unm1[i] + ab3_ * d_unm2[i]; });
  }

  // u{n+1/2}
  {
    double *data = u_half.HostReadWrite();
    double *dataUp0 = un.HostReadWrite();
    double *dataUp1 = Uext.HostReadWrite();
    for (int eq = 0; eq < nvel; eq++) {
      for (int i = 0; i < SdofInt; i++) {
        data[i + eq * SdofInt] = 0.5 * (dataUp1[i + eq * SdofInt] + dataUp0[i + eq * SdofInt]);
      }
    }
  }

  // store ext in _next containers, remove Text, Uext completely later
  un_next_gf.SetFromTrueDofs(Uext);
  un_next_gf.GetTrueDofs(un_next);
}

void Flow::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr) {
  vel_dbcs.emplace_back(attr, coeff);

  if (verbose && pmesh->GetMyRank() == 0) {
    mfem::out << "Adding Velocity Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((vel_ess_attr[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      vel_ess_attr[i] = 1;
    }
  }
}

void Flow::AddVelDirichletBC(VecFuncT *f, Array<int> &attr) {
  AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void Flow::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr) {
  pres_dbcs.emplace_back(attr, coeff);

  if (verbose && pmesh->GetMyRank() == 0) {
    mfem::out << "Adding Pressure Dirichlet BC to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }

  for (int i = 0; i < attr.Size(); ++i) {
    MFEM_ASSERT((pres_ess_attr[i] && attr[i]) == 0, "Duplicate boundary definition deteceted.");
    if (attr[i] == 1) {
      pres_ess_attr[i] = 1;
    }
  }
}

void Flow::AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr) {
  AddPresDirichletBC(new FunctionCoefficient(f), attr);
}

void Flow::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr) {
  accel_terms.emplace_back(attr, coeff);

  if (verbose && pmesh->GetMyRank() == 0) {
    mfem::out << "Adding Acceleration term to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }
}

void Flow::AddAccelTerm(VecFuncT *f, Array<int> &attr) {
  AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void Flow::AddGravityTerm(VectorCoefficient *coeff, Array<int> &attr) {
  gravity_terms.emplace_back(attr, coeff);

  if (verbose && pmesh->GetMyRank() == 0) {
    mfem::out << "Adding Gravity term to attributes ";
    for (int i = 0; i < attr.Size(); ++i) {
      if (attr[i] == 1) {
        mfem::out << i << " ";
      }
    }
    mfem::out << std::endl;
  }
}

void Flow::AddGravityTerm(VecFuncT *g, Array<int> &attr) {
  AddGravityTerm(new VectorFunctionCoefficient(pmesh->Dimension(), g), attr);
}

void Flow::interpolateInlet() {
  // MPI_Comm bcomm = groupsMPI->getComm(patchNumber);
  // int gSize = groupsMPI->groupSize(bcomm);
  int myRank;
  // MPI_Comm_rank(bcomm, &myRank);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  int Sdof = sfes->GetNDofs();

  string fname;
  fname = "./inputs/inletPlane.csv";
  // fname = "./inputs/inletPlane.txt";
  const int fname_length = fname.length();
  char *char_array = new char[fname_length + 1];
  strcpy(char_array, fname.c_str());
  int nCount = 0;

  // open, find size
  // if (groupsMPI->isGroupRoot(bcomm)) {
  if (rank0) {
    std::cout << " Attempting to open inlet file for counting... " << fname << endl;
    fflush(stdout);

    FILE *inlet_file;
    // if ( inlet_file = fopen(char_array,"r") ) {
    if (inlet_file = fopen("./inputs/inletPlane.csv", "r")) {
      // if ( inlet_file = fopen("./inputs/inletPlane.txt","r") ) {
      std::cout << " ...and open" << endl;
      fflush(stdout);
    } else {
      std::cout << " ...CANNOT OPEN FILE" << endl;
      fflush(stdout);
    }

    char *line = NULL;
    int ch = 0;
    size_t len = 0;
    ssize_t read;

    std::cout << " ...starting count" << endl;
    fflush(stdout);
    for (ch = getc(inlet_file); ch != EOF; ch = getc(inlet_file)) {
      if (ch == '\n') {
        nCount++;
      }
    }
    fclose(inlet_file);
    std::cout << " final count: " << nCount << endl;
    fflush(stdout);
  }

  // broadcast size
  // MPI_Bcast(&nCount, 1, MPI_INT, 0, bcomm);
  MPI_Bcast(&nCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // set size
  struct inlet_profile {
    double x, y, z, rho, temp, u, v, w;
  };
  struct inlet_profile inlet[nCount];
  double junk;

  // paraview slice output from mean
  // 0)"dens",1)"rms:0",2)"rms:1",3)"rms:2",4)"rms:3",5)"rms:4",6)"rms:5",7)"temp",8)"vel:0",9)"vel:1",10)"vel:2",11)"Points:0",12)"Points:1",13)"Points:2"

  // mean output from plane interpolation
  // 0) no, 1) x, 2) y, 3) z, 4) rho, 5) u, 6) v, 7) w

  // open, read data
  // if (groupsMPI->isGroupRoot(bcomm)) {
  if (rank0) {
    std::cout << " Attempting to read line-by-line..." << endl;
    fflush(stdout);

    /**/
    ifstream file;
    file.open("./inputs/inletPlane.csv");
    // file.open("./inputs/inletPlane.txt");
    string line;
    getline(file, line);
    int nLines = 0;
    while (getline(file, line)) {
      // cout << line << endl;
      stringstream sline(line);
      int entry = 0;
      while (sline.good()) {
        string substr;
        getline(sline, substr, ',');  // csv delimited
        // getline(sline, substr, ' ');  // space delimited
        double buffer = std::stod(substr);
        entry++;

        // using paraview dump
        /*
        if (entry == 1) {
          inlet[nLines].rho = buffer;
        } else if (entry == 8) {
          inlet[nLines].temp = buffer;
        } else if (entry == 9) {
          inlet[nLines].u = buffer;
        } else if (entry == 10) {
          inlet[nLines].v = buffer;
        } else if (entry == 11) {
          inlet[nLines].w = buffer;
        } else if (entry == 12) {
          inlet[nLines].x = buffer;
        } else if (entry == 13) {
          inlet[nLines].y = buffer;
        } else if (entry == 14) {
          inlet[nLines].z = buffer;
        }
        */

        // using code plane interp
        /**/
        if (entry == 2) {
          inlet[nLines].x = buffer;
        } else if (entry == 3) {
          inlet[nLines].y = buffer;
        } else if (entry == 4) {
          inlet[nLines].z = buffer;
        } else if (entry == 5) {
          inlet[nLines].rho = buffer;
          // to prevent nans in temp in out-of-domain plane regions
          inlet[nLines].temp = 0.0;  // thermoPressure / (Rgas * std::max(buffer,1.0));
        } else if (entry == 6) {
          inlet[nLines].u = buffer;
        } else if (entry == 7) {
          inlet[nLines].v = buffer;
        } else if (entry == 8) {
          inlet[nLines].w = buffer;
        }
        /**/
      }
      // std::cout << inlet[nLines].rho << " " << inlet[nLines].temp << " " << inlet[nLines].u << " " << inlet[nLines].x
      // << " " << inlet[nLines].y << " " << inlet[nLines].z << endl; fflush(stdout);
      nLines++;
    }
    file.close();

    // std::cout << " stream method ^" << endl; fflush(stdout);
    std::cout << " " << endl;
    fflush(stdout);
    /**/

    /*
    FILE *inlet_file;
    //inlet_file = fopen(char_array,"r");
    inlet_file = fopen("./inputs/inletPlane.csv","r");
    //std::cout << " ...opened" << endl; fflush(stdout);
    std::cout << " " << endl; fflush(stdout);
    for (int i = 0; i < nCount; i++) {
      int got;
      //got = fscanf(inlet_file, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
      got = fscanf(inlet_file, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
                   &inlet[i].rho, &junk, &junk, &junk, &junk, &junk, &junk, &inlet[i].temp, &inlet[i].u,
                   &inlet[i].v, &inlet[i].w, &inlet[i].x, &inlet[i].y, &inlet[i].z);
      std::cout << inlet[i].rho << " " << inlet[i].temp << " " << inlet[i].u << " " << inlet[i].v << " " << inlet[i].w
    << " " << inlet[i].x << endl; fflush(stdout);
    }
    std::cout << " ...done" << endl; fflush(stdout);
    std::cout << " " << endl; fflush(stdout);
    fclose(inlet_file);
    */
  }

  // broadcast data
  // MPI_Bcast(&inlet, nCount*8, MPI_DOUBLE, 0, bcomm);
  MPI_Bcast(&inlet, nCount * 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank0) {
    std::cout << " communicated inlet data" << endl;
    fflush(stdout);
  }

  // width of interpolation stencil
  // double Lx, Ly, Lz, Lmax, radius;
  // Lx = xmax - xmin;
  // Ly = ymax - ymin;
  // Lz = zmax - zmin;
  // Lmax = std::max(Lx,Ly);
  // Lmax = std::max(Lmax,Lz);
  // radius = 0.25 * Lmax;
  // radius = 0.01 * Lmax;

  double torch_radius = 28.062 / 1000.0;
  double radius = 2.0 * torch_radius / 300.0;  // 300 from nxn interp plane

  ParGridFunction coordsDof(vfes);
  pmesh->GetNodes(coordsDof);
  double *dataVel = buffer_uInlet->HostWrite();
  // double *dataTemp = buffer_tInlet->HostWrite();
  double *dataVelInf = buffer_uInletInf->HostWrite();
  // double *dataTempInf = buffer_tInletInf->HostWrite();
  for (int n = 0; n < sfes->GetNDofs(); n++) {
    auto hcoords = coordsDof.HostRead();  // get coords
    double xp[3];
    for (int d = 0; d < dim; d++) {
      xp[d] = hcoords[n + d * vfes->GetNDofs()];
    }

    int iCount = 0;
    double dist, wt;
    double wt_tot = 0.0;
    double val_rho = 0.0;
    double val_u = 0.0;
    double val_v = 0.0;
    double val_w = 0.0;
    double val_T = 0.0;
    double dmin = 1.0e15;

    // find minimum distance in interpolant field to use as interpolation radius
    /*
    double distMin = 1.0e12;
    for (int j = 0; j < nCount; j++) {
      dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x) + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);
      dist = sqrt(dist);
      distMin = std::min(distMin,dist);
    }
    // find penulitmate minimum distance;
    double distMinSecond = 1.0e12;
    for (int j = 0; j < nCount; j++) {
      dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x) + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);
      dist = sqrt(dist);
      if (dist != distMin) {
        distMinSecond = std::min(distMinSecond,dist);
      }
    }

    //radius = 0.5 * (radius + distMin);
    radius = distMinSecond;
    */

    // std::cout << " radius for interpoaltion: " << 4.0*radius << endl; fflush(stdout);
    for (int j = 0; j < nCount; j++) {
      dist = (xp[0] - inlet[j].x) * (xp[0] - inlet[j].x) + (xp[1] - inlet[j].y) * (xp[1] - inlet[j].y) +
             (xp[2] - inlet[j].z) * (xp[2] - inlet[j].z);
      dist = sqrt(dist);
      // std::cout << " Gaussian interpolation, point " << n << " with distance " << dist << endl; fflush(stdout);

      // exclude points outside the domain
      if (inlet[j].rho < 1.0e-8) {
        continue;
      }

      // gaussian
      if (dist <= 1.0 * radius) {
        // std::cout << " Caught an interp point " << dist << endl; fflush(stdout);
        wt = exp(-(dist * dist) / (radius * radius));
        wt_tot = wt_tot + wt;
        val_rho = val_rho + wt * inlet[j].rho;
        val_u = val_u + wt * inlet[j].u;
        val_v = val_v + wt * inlet[j].v;
        val_w = val_w + wt * inlet[j].w;
        val_T = val_T + wt * inlet[j].temp;
        iCount++;
        // std::cout << "* " << n << " "  << iCount << " " << wt_tot << " " << val_T << endl; fflush(stdout);
      }
      // nearest, just for testing
      // if(dist <= dmin) {
      //  dmin = dist;
      //  wt_tot = 1.0;
      //   val_rho = inlet[j].rho;
      //   val_u = inlet[j].u;
      //   val_v = inlet[j].v;
      //   val_w = inlet[j].w;
      //   iCount = 1;
      // }
    }

    // if(rank0_) { std::cout << " attempting to record interpolated values in buffer" << endl; fflush(stdout); }
    if (wt_tot > 0.0) {
      dataVel[n + 0 * Sdof] = val_u / wt_tot;
      dataVel[n + 1 * Sdof] = val_v / wt_tot;
      dataVel[n + 2 * Sdof] = val_w / wt_tot;
      // dataTemp[n] = val_T / wt_tot;
      dataVelInf[n + 0 * Sdof] = val_u / wt_tot;
      dataVelInf[n + 1 * Sdof] = val_v / wt_tot;
      dataVelInf[n + 2 * Sdof] = val_w / wt_tot;
      // dataTempInf[n] = val_T / wt_tot;
      // std::cout << n << " point set to: " << dataVel[n + 0*Sdof] << " " << dataTemp[n] << endl; fflush(stdout);
    } else {
      dataVel[n + 0 * Sdof] = 0.0;
      dataVel[n + 1 * Sdof] = 0.0;
      dataVel[n + 2 * Sdof] = 0.0;
      // dataTemp[n] = 0.0;
      dataVelInf[n + 0 * Sdof] = 0.0;
      dataVelInf[n + 1 * Sdof] = 0.0;
      dataVelInf[n + 2 * Sdof] = 0.0;
      // dataTempInf[n] = 0.0;
      // std::cout << " iCount of zero..." << endl; fflush(stdout);
    }
  }
}

void Flow::uniformInlet() {
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  int Sdof = sfes->GetNDofs();

  ParGridFunction coordsDof(vfes);
  pmesh->GetNodes(coordsDof);
  double *dataVel = buffer_uInlet->HostWrite();
  // double *dataTemp = buffer_tInlet->HostWrite();
  double *dataVelInf = buffer_uInletInf->HostWrite();
  // double *dataTempInf = buffer_tInletInf->HostWrite();

  Vector inlet_vec(3);
  double inlet_temp, inlet_pres;
  for (int i = 0; i < dim; i++) {
    inlet_vec[i] = config->velInlet[i];
  }
  // inlet_temp = ambientPressure / (Rgas*config.densInlet);
  // inlet_pres = ambientPressure;

  for (int n = 0; n < sfes->GetNDofs(); n++) {
    auto hcoords = coordsDof.HostRead();
    double xp[3];
    for (int d = 0; d < dim; d++) {
      xp[d] = hcoords[n + d * vfes->GetNDofs()];
    }
    for (int d = 0; d < dim; d++) {
      dataVel[n + d * Sdof] = inlet_vec[d];
    }
    for (int d = 0; d < dim; d++) {
      dataVelInf[n + d * Sdof] = inlet_vec[d];
    }
    // dataTemp[n] = inlet_temp;
    // dataTempInf[n] = inlet_temp;
  }
}

///~ BC/IC hard-codes ~///

void accel(const Vector &x, double t, Vector &f) {
  f(0) = 0.1;
  f(1) = 0.0;
  f(2) = 0.0;
}

void vel_ic(const Vector &coords, double t, Vector &u) {
  double yp;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);

  double pi = 3.14159265359;
  double A = -1.0;
  double B = -1.0;
  double C = +2.0;
  double aL = 4.0;
  double bL = 2.0 * pi;
  double cL = 4.0;
  double M = 1.0;
  double L = 0.1;
  double scl;

  /**/
  u(0) = M;
  u(1) = 0.0;
  u(2) = 0.0;

  u(0) += L * M * A * cos(aL * x) * sin(bL * y) * sin(cL * z);
  u(1) += L * M * B * sin(aL * x) * cos(bL * y) * sin(cL * z);
  u(2) += L * M * C * sin(aL * x) * sin(bL * y) * cos(cL * z);

  u(0) -= 0.5 * L * M * A * cos(2.0 * aL * x) * sin(2.0 * bL * y) * sin(2.0 * cL * z);
  u(1) -= 0.5 * L * M * B * sin(2.0 * aL * x) * cos(2.0 * bL * y) * sin(2.0 * cL * z);
  u(2) -= 0.5 * L * M * C * sin(2.0 * aL * x) * sin(2.0 * bL * y) * cos(2.0 * cL * z);

  u(0) += 0.25 * L * M * A * cos(4.0 * aL * x) * sin(4.0 * bL * y) * sin(4.0 * cL * z);
  u(1) += 0.25 * L * M * B * sin(4.0 * aL * x) * cos(4.0 * bL * y) * sin(4.0 * cL * z);
  u(2) += 0.25 * L * M * C * sin(4.0 * aL * x) * sin(4.0 * bL * y) * cos(4.0 * cL * z);

  /*
  scl = std::max(1.0-std::pow((y-0.0)/0.2,8),0.0);
  if (y > 0) {
    scl = scl + 0.1 * std::pow(y/0.2,4.0);
  }
  */

  scl = std::max(1.0 - std::pow((y - 0.0) / 1.0, 4), 0.0);
  // if (y > 0) {
  //   scl = scl + 0.001 * std::pow((y-1.0)/1.0,4.0);
  // }
  // scl = 1.0;

  u(0) = u(0) * scl;
  u(1) = u(1) * scl;
  u(2) = u(2) * scl;
  /**/

  // scl = std::max(1.0-std::pow((y-0.0)/0.2,4),0.0);
  // u(0) = scl;
  // u(1) = 0.0;
  // u(2) = 0.0;

  /*
  // FOR INLET TEST ONLY
  u(0) = 1.0; // - std::pow((y-0.0)/0.5,2);
  u(1) = 0.0;
  u(2) = 0.0;
  */
}

void vel_icBox(const Vector &coords, double t, Vector &u) {
  double yp;
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);

  double pi = 3.14159265359;
  double A = -1.0 * pi;
  double B = -1.0 * pi;
  double C = +2.0 * pi;
  // double aL = 2.0 / (2.0) * pi;
  // double bL = 2.0 * pi;
  // double cL = 2.0 * pi;
  double aL = 4.0;
  double bL = 2.0 * pi;
  double cL = 2.0;
  double M = 1.0;
  double L = 0.1;
  double scl;

  u(0) = 0.0;
  u(1) = 0.0;
  u(2) = 0.0;
}

void vel_channelTest(const Vector &coords, double t, Vector &u) {
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);

  u(0) = 1.0 - std::pow((y - 0.0) / 0.2, 2);
  u(1) = 0.0;
  u(2) = 0.0;
}

void vel_wall(const Vector &x, double t, Vector &u) {
  u(0) = 0.0;
  u(1) = 0.0;
  u(2) = 0.0;
}

void vel_inlet(const Vector &coords, double t, Vector &u) {
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);

  u(0) = 1.0 - std::pow((y - 0.0) / 0.5, 2);
  u(1) = 0.0;
  u(2) = 0.0;
}

double pres_inlet(const Vector &coords, double p) {
  double x = coords(0);
  double y = coords(1);
  double z = coords(2);
  double pres;
  pres = 101325.0;
  return pres;
}
