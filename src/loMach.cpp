
///*  Add description later //

#include "loMach.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include <hdf5.h>
#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace mfem::common;

// temporary
double mesh_stretching_func(const double y);
void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst);
void accel(const Vector &x, double t, Vector &f);
void vel_ic(const Vector &coords, double t, Vector &u);
void vel_wall(const Vector &x, double t, Vector &u);
double temp_ic(const Vector &coords, double t);
double temp_wall(const Vector &x, double t);

LoMachSolver::LoMachSolver(MPI_Session &mpi, LoMachOptions loMach_opts, TPS::Tps *tps)
    : mpi_(mpi), loMach_opts_(loMach_opts) {
  tpsP_ = tps;
  pmesh = NULL;
  nprocs_ = mpi_.WorldSize();
  rank_ = mpi_.WorldRank();
  if (rank_ == 0) { rank0_ = true; }
  else { rank0_ = false; }
  groupsMPI = new MPI_Groups(&mpi_);
  groupsMPI->init();

  parseSolverOptions();
  parseSolverOptions2();

  // set default solver state
  exit_status_ = NORMAL;

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }
}

void LoMachSolver::initialize() {
  bool verbose = mpi_.Root();
  if (verbose) grvy_printf(ginfo, "Initializing loMach solver.\n");

  order = config.solOrder; //loMach_opts_.order;
  porder = config.solOrder; //loMach_opts_.order;
  norder = config.solOrder; //loMach_opts_.order;

  // temporary hard-coding
  Re_tau = 182.0;
  kin_vis = 1.0/Re_tau;
  ambientPressure = 101325.0;
  Rgas = 287.0;
  Pr = 1.2;

  //-----------------------------------------------------
  // 1) Prepare the mesh
  //-----------------------------------------------------

  // Read the serial mesh (on each mpi rank)
  Mesh mesh = Mesh(loMach_opts_.mesh_file.c_str());

  Vector x_translation({config.GetXTrans(), 0.0, 0.0});
  Vector y_translation({0.0, config.GetYTrans(), 0.0});
  Vector z_translation({0.0, 0.0, config.GetZTrans()});
  std::vector<Vector> translations = {x_translation, y_translation, z_translation};

  if (mpi_.Root()) {
    std::cout << " Making the mesh periodic using the following offsets:" << std::endl;
    std::cout << "   xTrans: " << config.GetXTrans() << std::endl;
    std::cout << "   yTrans: " << config.GetYTrans() << std::endl;
    std::cout << "   zTrans: " << config.GetZTrans() << std::endl;
  }

  // NB: be careful here... periodic mesh is created but mesh is used some places below.
  // It looks ok except in case where mesh refinement is requested.

  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  Mesh periodic_mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(translations));
  dim_ = mesh.Dimension();

  // for now number of velocity components is equal to dim
  nvel = dim_;

  eqSystem = config.GetEquationSystem();
  mixture = NULL;

  // HARD CODE
  config.dryAirInput.specific_heat_ratio = 1.4;
  config.dryAirInput.gas_constant = 287.058;
  config.dryAirInput.f = config.workFluid;
  config.dryAirInput.eq_sys = config.eqSystem;
  mixture = new DryAir(config, dim_, nvel);
  transportPtr = new DryAirTransport(mixture, config);

  MaxIters = config.GetNumIters();
  max_speed = 0.;
  num_equation = nvel + 2;  // no species support yet

  // check if a simulation is being restarted
  if (config.GetRestartCycle() > 0) {
    if (config.GetUniformRefLevels() > 0) {
      if (mpi_.Root()) {
        std::cerr << "ERROR: Uniform mesh refinement not supported upon restart." << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    nelemGlobal_ = mesh.GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      partitioning_file_hdf5("read");
    }

  } else {
    // remove previous solution
    if (mpi_.Root()) {
      string command = "rm -r ";
      command.append(config.GetOutputName());
      int err = system(command.c_str());
      if (err != 0) {
        cout << "Error deleting previous data in " << config.GetOutputName() << endl;
      }
    }

    // uniform refinement, user-specified number of times
    for (int l = 0; l < config.GetUniformRefLevels(); l++) {
      if (mpi_.Root()) {
        std::cout << "Uniform refinement number " << l << std::endl;
      }
      mesh.UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = mesh.GetNE();
    if (nprocs_ > 1) {
      assert(mesh.Conforming());
      partitioning_ = Array<int>(mesh.GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write");
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // make sure these are actually hooked up!
    time = 0.;
    iter = 0;
  }
  // 1c) Partition the mesh (see partitioning_ in M2ulPhyS)
  pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
  if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");

  //-----------------------------------------------------
  // 2) Prepare the required finite elements
  //-----------------------------------------------------
  fec = new H1_FECollection(order, dim_);

  // space for dim_ dimensional vector (i.e., velocity)
  vfes = new ParFiniteElementSpace(pmesh, fec, dim_);

  // space for scalar variables (e.g., pressure, temperature, etc)
  sfes = new ParFiniteElementSpace(pmesh, fec, 1);

  // space for entire solution concatenated together
  fvfes = new ParFiniteElementSpace(pmesh, fec, num_equation); //, Ordering::byNODES);

  // Check if fully periodic mesh
  if (!(pmesh->bdr_attributes.Size() == 0))
    {
      vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr = 0;

      pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      pres_ess_attr = 0;

      temp_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      temp_ess_attr = 0;
    }
  if (verbose) grvy_printf(ginfo, "Spaces constructed...\n");

  int vfes_truevsize = vfes->GetTrueVSize();
  int sfes_truevsize = sfes->GetTrueVSize();
  if (verbose) {
    grvy_printf(ginfo, "Got sizes...\n");
    printf("vfes_truevsize = %d\n", vfes_truevsize);
    printf("sfes_truevsize = %d\n", sfes_truevsize);
    fflush(stdout);
  }

  //*************************************************
  // Allocate necessary fields and vectors
  //*************************************************

  // First, dim_ dimensional quantities at each node
  un.SetSize(vfes_truevsize);
  un = 0.0;
  un_next.SetSize(vfes_truevsize);
  un_next = 0.0;

  unm1.SetSize(vfes_truevsize);
  unm1 = 0.0;
  unm2.SetSize(vfes_truevsize);
  unm2 = 0.0;

  fn.SetSize(vfes_truevsize);

  // if dealiasing, use nfes
  Nun.SetSize(vfes_truevsize);
  Nun = 0.0;
  Nunm1.SetSize(vfes_truevsize);
  Nunm1 = 0.0;
  Nunm2.SetSize(vfes_truevsize);
  Nunm2 = 0.0;

  uBn.SetSize(vfes_truevsize);
  uBn = 0.0;
  uBnm1.SetSize(vfes_truevsize);
  uBnm1 = 0.0;
  uBnm2.SetSize(vfes_truevsize);
  uBnm2 = 0.0;

  FBext.SetSize(vfes_truevsize);

  Fext.SetSize(vfes_truevsize);
  FText.SetSize(vfes_truevsize);
  Lext.SetSize(vfes_truevsize);
  resu.SetSize(vfes_truevsize);

  tmpR1.SetSize(vfes_truevsize);

  un_gf.SetSpace(vfes);
  un_gf = 0.0;
  un_next_gf.SetSpace(vfes);
  un_next_gf = 0.0;

  Lext_gf.SetSpace(vfes);
  curlu_gf.SetSpace(vfes);
  curlcurlu_gf.SetSpace(vfes);
  FText_gf.SetSpace(vfes);
  resu_gf.SetSpace(vfes);

  R1PM0_gf.SetSpace(vfes);
  R1PM0_gf = 0.0;
  //R1PX2_gf.SetSpace(nfes);
  R1PX2_gf.SetSpace(vfes);
  R1PX2_gf = 0.0;

  // And second, scalar fields/vectors
  pn.SetSize(sfes_truevsize);
  pn = 0.0;
  resp.SetSize(sfes_truevsize);
  resp = 0.0;
  FText_bdr.SetSize(sfes_truevsize);
  g_bdr.SetSize(sfes_truevsize);

  pnBig.SetSize(sfes_truevsize);
  pnBig = 0.0;

  sml_gf.SetSpace(sfes);
  big_gf.SetSpace(sfes);

  pn_gf.SetSpace(sfes);
  pn_gf = 0.0;
  resp_gf.SetSpace(sfes);
  tmpR0PM1.SetSize(sfes_truevsize);

  // adding temperature
  Tn.SetSize(sfes_truevsize);
  Tn = 0.0;
  Tn_next.SetSize(sfes_truevsize);
  Tn_next = 0.0;

  Tnm1.SetSize(sfes_truevsize);
  Tnm1 = 298.0; // fix hardcode
  Tnm2.SetSize(sfes_truevsize);
  Tnm2 = 298.0;

  fTn.SetSize(sfes_truevsize); // forcing term
  NTn.SetSize(sfes_truevsize); // advection terms
  NTn = 0.0;
  NTnm1.SetSize(sfes_truevsize);
  NTnm1 = 0.0;
  NTnm2.SetSize(sfes_truevsize);
  NTnm2 = 0.0;

  Text.SetSize(sfes_truevsize);
  Text_bdr.SetSize(sfes_truevsize);
  Text_gf.SetSpace(sfes);
  t_bdr.SetSize(sfes_truevsize);

  resT.SetSize(sfes_truevsize);
  tmpR0.SetSize(sfes_truevsize);

  Tn_gf.SetSpace(sfes); // bc?
  Tn_gf = 298.0; // fix hardcode
  Tn_next_gf.SetSpace(sfes);
  Tn_next_gf = 298.0;

  resT_gf.SetSpace(sfes);

  // density, not actually solved for directly
  rn.SetSize(sfes_truevsize);
  rn = 1.0;
  rn_gf.SetSpace(sfes);
  rn_gf = 1.0;

  R0PM0_gf.SetSpace(sfes);
  R0PM0_gf = 0.0;
  R0PM1_gf.SetSpace(sfes);
  R0PM1_gf = 0.0;
  if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");

  cur_step = 0;

  initSolutionAndVisualizationVectors();
  if (verbose) grvy_printf(ginfo, "init Sol and Vis okay...\n");

  ioData.initializeSerial(mpi_.Root(), (config.RestartSerial() != "no"), &mesh);
  if (verbose) grvy_printf(ginfo, " ioData.init thingy...\n");

  CFL = config.GetCFLNumber();

  // Determine the minimum element size.
  {
    double local_hmin = 1.0e18;
    for (int i = 0; i < pmesh->GetNE(); i++) {
      local_hmin = min(pmesh->GetElementSize(i, 1), local_hmin);
    }
    MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
  }

  // maximum size
  {
    double local_hmax = 1.0e-15;
    for (int i = 0; i < pmesh->GetNE(); i++) {
      local_hmax = max(pmesh->GetElementSize(i, 1), local_hmax);
    }
    MPI_Allreduce(&local_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
  }
  if (verbose) grvy_printf(ginfo, " element size found...\n");      

  if (mpi_.Root()) cout << "Maximum element size: " << hmax << "m" << endl;
  if (mpi_.Root()) cout << "Minimum element size: " << hmin << "m" << endl;
}


void LoMachSolver::Setup(double dt) {
   partial_assembly = true; // not working?

   ParGridFunction *u_gf = GetCurrentVelocity();
   ParGridFunction *p_gf = GetCurrentPressure();
   ParGridFunction *t_gf = GetCurrentTemperature();

   VectorFunctionCoefficient u_ic_coef(dim_, vel_ic);
   u_gf->ProjectCoefficient(u_ic_coef);

   FunctionCoefficient t_ic_coef(temp_ic);
   t_gf->ProjectCoefficient(t_ic_coef);

   Array<int> domain_attr(pmesh->attributes);
   domain_attr = 1;
   AddAccelTerm(accel, domain_attr); // HERE wire to input file

   Vector zero_vec(3); zero_vec = 0.0;
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 0.0;

   buffer_ubc = new VectorConstantCoefficient(zero_vec);
   for (int i = 0; i < numWalls; i++) {
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {
       if (config.wallPatchType[i].second != WallType::INV) {
         if (iFace == config.wallPatchType[i].first) {
            attr[iFace-1] = 1;
         }
       }
     }
   }
   AddVelDirichletBC(vel_wall, attr);

   ConstantCoefficient t_bc_coef;
   t_bc_coef.constant = config.wallBC[0].Th;
   Array<int> Tattr(pmesh->bdr_attributes.Max());
   Tattr = 0;
   buffer_tbc = new ConstantCoefficient(t_bc_coef.constant);
   for (int i = 0; i < numWalls; i++) {
     t_bc_coef.constant = config.wallBC[i].Th;
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {
       if (config.wallPatchType[i].second != WallType::INV) { // should only be isothermal
         if (iFace == config.wallPatchType[i].first) {
            Tattr[iFace-1] = 1;
         }
       }
     }
   }
   AddTempDirichletBC(buffer_tbc, Tattr);
   std::cout << "Check 4..." << std::endl;

   // returning to regular setup
   sw_setup.Start();

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   sfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);
   sfes->GetEssentialTrueDofs(temp_ess_attr, temp_ess_tdof);

   int Vdof = vfes->GetTrueVSize();
   int Pdof = sfes->GetTrueVSize();
   int Tdof = sfes->GetTrueVSize();

   Array<int> empty;

   // unsteady: p+p = 2p
   // convection: p+p+(p-1) = 3p-1
   // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]

   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order);
   const IntegrationRule &ir_nli = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 3 * norder - 1);
   const IntegrationRule &ir_pi = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2 * porder);
   const IntegrationRule &ir_i  = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2 * order);
   std::cout << "Check 6..." << std::endl;

   // convection section, extrapolation
   nlcoeff.constant = -1.0;
   N = new ParNonlinearForm(vfes);
   //N = new ParNonlinearForm(nfes);
   auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff);
   if (numerical_integ)
   {
      nlc_nlfi->SetIntRule(&ir_nli);
   }
   N->AddDomainIntegrator(nlc_nlfi);
   if (partial_assembly)
   {
      N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      N->Setup();
   }
   std::cout << "Check 7..." << std::endl;

   // mass matrix
   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   if (numerical_integ) { mv_blfi->SetIntRule(&ir_ni); }
   Mv_form->AddDomainIntegrator(mv_blfi);
   if (partial_assembly) { Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);

   // need to build another with q=1/rho
   // DiffusionIntegrator(MatrixCoefficient &q, const IntegrationRule *ir = nullptr)
   // BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma); with sigma some type of Coefficient class
   //ParGridFunction buffer1(sfes);
   bufferInvRho = new ParGridFunction(sfes);
   {
     double *data = bufferInvRho->HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       //data[i] = (Rgas * Tdata[i]) / ambientPressure;
       data[i] = 1.0;
     }
   }
   invRho = new GridFunctionCoefficient(bufferInvRho);

   // looks like this is the Laplacian for press eq
   Sp_form = new ParBilinearForm(sfes);
   auto *sp_blfi = new DiffusionIntegrator;
   //auto *sp_blfi = new DiffusionIntegrator(*invRho); // HERE breaks with amg
   if (numerical_integ)
   {
      sp_blfi->SetIntRule(&ir_pi);
   }
   Sp_form->AddDomainIntegrator(sp_blfi);
   if (partial_assembly)
   {
      Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);
   std::cout << "Check 10..." << std::endl;

   // div(u)
   D_form = new ParMixedBilinearForm(vfes, sfes);
   auto *vd_mblfi = new VectorDivergenceIntegrator();
   if (numerical_integ)
   {
      vd_mblfi->SetIntRule(&ir_ni);
   }
   D_form->AddDomainIntegrator(vd_mblfi);
   if (partial_assembly)
   {
      D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, D);
   std::cout << "Check 11..." << std::endl;

   // for grad(div(u))?
   G_form = new ParMixedBilinearForm(sfes, vfes);
   auto *g_mblfi = new GradientIntegrator();
   if (numerical_integ)
   {
      g_mblfi->SetIntRule(&ir_ni);
   }
   G_form->AddDomainIntegrator(g_mblfi);
   if (partial_assembly)
   {
      G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, G);
   std::cout << "Check 12..." << std::endl;

   // viscosity field
   //ParGridFunction buffer2(vfes); // or sfes here?
   bufferVisc = new ParGridFunction(vfes);
   {
     double *data = bufferVisc->HostReadWrite();
     double *Tdata = Tn.HostReadWrite();
     double visc[2];
     double prim[nvel+2];
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
       for (int eq = 0; eq < nvel; eq++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         data[i + eq * Tdof] = visc[0];
       }
     }
   }
   //GridFunctionCoefficient viscField(&buffer2);
   viscField = new GridFunctionCoefficient(bufferVisc);
   std::cout << "Check 13..." << std::endl;

   // this is used in the last velocity implicit solve, Step kin_vis is
   // only for the pressure-poisson
   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);
   //H_form = new ParBilinearForm(sfes); // maybe?
   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff); // diagonal from unsteady term
   auto *hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);
   //auto *hdv_blfi = new VectorDiffusionIntegrator(*viscField); // Laplacian
   if (numerical_integ)
   {
      hmv_blfi->SetIntRule(&ir_ni);
      hdv_blfi->SetIntRule(&ir_ni);
   }
   H_form->AddDomainIntegrator(hmv_blfi);
   H_form->AddDomainIntegrator(hdv_blfi);
   if (partial_assembly)
   {
      H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);
   std::cout << "Check 14..." << std::endl;

   // boundary terms
   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   // Vector vec_one(3);
   // vec_one = 1.0;
   // FText_gfcoeff = new VectorConstantCoefficient(vec_one);
   FText_bdr_form = new ParLinearForm(sfes);
   //FText_bdr_form = new ParLinearForm(vfes); // maybe?
   auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
   if (numerical_integ) { ftext_bnlfi->SetIntRule(&ir_ni); }
   FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);

   //g_bdr_form = new ParLinearForm(vfes); //? was sfes
   g_bdr_form = new ParLinearForm(sfes);
   for (auto &vel_dbc : vel_dbcs)
   {
      auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
      if (numerical_integ) { gbdr_bnlfi->SetIntRule(&ir_ni); }
      g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
   }

   f_form = new ParLinearForm(vfes);
   for (auto &accel_term : accel_terms)
   {
      auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
      // @TODO: This order should always be the same as the nonlinear forms one!
      // const IntegrationRule &ir = IntRules.Get(vfes->GetFE(0)->GetGeomType(),
      //                                          4 * order);
      // vdlfi->SetIntRule(&ir);
      if (numerical_integ)
      {
         vdlfi->SetIntRule(&ir_ni);
      }
      f_form->AddDomainIntegrator(vdlfi);
   }

   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      Mv_form->AssembleDiagonal(diag_pa);
      MvInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MvInvPC = new HypreSmoother(*Mv.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MvInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MvInv = new CGSolver(vfes->GetComm());
   MvInv->iterative_mode = false;
   MvInv->SetOperator(*Mv);
   MvInv->SetPreconditioner(*MvInvPC);
   MvInv->SetPrintLevel(pl_mvsolve);
   MvInv->SetRelTol(1e-12);
   MvInv->SetMaxIter(500);
   std::cout << "Check 17..." << std::endl;

   /**/
   if (partial_assembly)
   {
      lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
      SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      SpInvPC->SetPrintLevel(pl_amg);
      SpInvPC->Mult(resp, pn);
      //SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC = new OrthoSolver(sfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   else
   {
      SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(0);
      //SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC = new OrthoSolver(sfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   //SpInv = new CGSolver(vfes->GetComm());
   SpInv = new CGSolver(sfes->GetComm());
   SpInv->iterative_mode = true;
   //SpInv->iterative_mode = false;
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
   SpInv->SetRelTol(rtol_spsolve);
   SpInv->SetMaxIter(500);
   /**/
   std::cout << "Check 18..." << std::endl;

   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      //Vector diag_pa(sfes->GetTrueVSize()); // maybe?
      std::cout << "Check 18a..." << std::endl;
      H_form->AssembleDiagonal(diag_pa); // invalid read
      std::cout << "Check 18b..." << std::endl;
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
      std::cout << "Check 18c..." << std::endl;
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>()); // conditional jump
      std::cout << "Check 18d..." << std::endl;
      dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
      std::cout << "Check 18e..." << std::endl;
   }
   HInv = new CGSolver(vfes->GetComm());
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(500);
   std::cout << "Check 19..." << std::endl;

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   un_gf.GetTrueDofs(un);
   //un_next = un; // invalid read
   {
     double *data = un_next.HostReadWrite();
     double *Udata = un.HostReadWrite();
     for (int i = 0; i < Vdof; i++) {
       data[i] = Udata[i];
     }
   }
   un_next_gf.SetFromTrueDofs(un_next);

   // temperature.....................................
   Mt_form = new ParBilinearForm(sfes);
   auto *mt_blfi = new MassIntegrator;
   if (numerical_integ) { mt_blfi->SetIntRule(&ir_i); }
   Mt_form->AddDomainIntegrator(mt_blfi);
   if (partial_assembly) { Mt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Mt_form->Assemble();
   Mt_form->FormSystemMatrix(empty, Mt);
   std::cout << "Check 20..." << std::endl;

   // div(uT) for temperature
   Dt_form = new ParMixedBilinearForm(vfes, sfes);
   auto *vtd_mblfi = new VectorDivergenceIntegrator();
   if (numerical_integ)
   {
      vtd_mblfi->SetIntRule(&ir_nli);
   }
   Dt_form->AddDomainIntegrator(vtd_mblfi);
   if (partial_assembly)
   {
      Dt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Dt_form->Assemble();
   Dt_form->FormRectangularSystemMatrix(empty, empty, Dt);
   std::cout << "Check 21..." << std::endl;

   // thermal diffusivity field
   //ParGridFunction buffer3(sfes);
   bufferAlpha = new ParGridFunction(sfes);
   {
     double *data = bufferAlpha->HostReadWrite();
     double *Tdata = Tn.HostReadWrite();
     double visc[2];
     double prim[nvel+2];
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         data[i] = visc[0] / Pr;
         //data[i] = kin_vis; // static value
     }
   }
   //GridFunctionCoefficient alphaField(&buffer3);
   alphaField = new GridFunctionCoefficient(bufferAlpha);

   Ht_lincoeff.constant = kin_vis / Pr;
   Ht_bdfcoeff.constant = 1.0 / dt;
   Ht_form = new ParBilinearForm(sfes);
   auto *hmt_blfi = new MassIntegrator(Ht_bdfcoeff); // unsteady bit
   auto *hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);

   if (numerical_integ)
   {
     hmt_blfi->SetIntRule(&ir_i);
     hdt_blfi->SetIntRule(&ir_i);
   }
   Ht_form->AddDomainIntegrator(hmt_blfi);
   Ht_form->AddDomainIntegrator(hdt_blfi);
   if (partial_assembly)
   {
      Ht_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Ht_form->Assemble();
   Ht_form->FormSystemMatrix(temp_ess_tdof, Ht);

   // temp boundary terms
   Text_gfcoeff = new GridFunctionCoefficient(&Text_gf);
   Text_bdr_form = new ParLinearForm(sfes);
   auto *text_blfi = new BoundaryLFIntegrator(*Text_gfcoeff);
   if (numerical_integ) { text_blfi->SetIntRule(&ir_i); }
   Text_bdr_form->AddBoundaryIntegrator(text_blfi, temp_ess_attr);

   t_bdr_form = new ParLinearForm(sfes);
   for (auto &temp_dbc : temp_dbcs)
   {
      auto *tbdr_blfi = new BoundaryLFIntegrator(*temp_dbc.coeff);
      if (numerical_integ) { tbdr_blfi->SetIntRule(&ir_i); }
      t_bdr_form->AddBoundaryIntegrator(tbdr_blfi, temp_dbc.attr);
   }

   if (partial_assembly)
   {
      Vector diag_pa(sfes->GetTrueVSize());
      Mt_form->AssembleDiagonal(diag_pa);
      MtInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MtInvPC = new HypreSmoother(*Mt.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MtInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MtInv = new CGSolver(sfes->GetComm());
   MtInv->iterative_mode = false;
   MtInv->SetOperator(*Mt);
   MtInv->SetPreconditioner(*MtInvPC);
   MtInv->SetPrintLevel(pl_mtsolve);
   MtInv->SetRelTol(1e-12);
   MtInv->SetMaxIter(500);
   std::cout << "Check 26..." << std::endl;

   if (partial_assembly)
   {
      Vector diag_pa(sfes->GetTrueVSize());
      Ht_form->AssembleDiagonal(diag_pa);
      HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
   }
   else
   {
      HtInvPC = new HypreSmoother(*Ht.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(HtInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HtInv = new CGSolver(sfes->GetComm());
   HtInv->iterative_mode = true;
   HtInv->SetOperator(*Ht);
   HtInv->SetPreconditioner(*HtInvPC);
   HtInv->SetPrintLevel(pl_hsolve);
   HtInv->SetRelTol(rtol_hsolve);
   HtInv->SetMaxIter(500);

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   Tn_gf.GetTrueDofs(Tn);
   //Tn_next = Tn;
   {
     double *data = Tn_next.HostReadWrite();
     double *Tdata = Tn.HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       data[i] = Tdata[i];
     }
   }
   //Tn_next_gf.SetFromTrueDofs(Tn_next); // invalid read
   {
     double *data = Tn_next_gf.HostReadWrite();
     double *Tdata = Tn_next.HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       data[i] = Tdata[i];
     }
   }
   dthist[0] = dt;

   // Velocity filter
   if (filter_alpha != 0.0)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes, pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh, vfec_filter, pmesh->Dimension());

      un_NM1_gf.SetSpace(vfes_filter);
      un_NM1_gf = 0.0;

      un_filtered_gf.SetSpace(vfes);
      un_filtered_gf = 0.0;
   }

   // add filter to temperature field later...
   if (filter_alpha != 0.0)
   {
      tfec_filter = new H1_FECollection(order - filter_cutoff_modes);
      tfes_filter = new ParFiniteElementSpace(pmesh, tfec_filter);

      Tn_NM1_gf.SetSpace(tfes_filter);
      Tn_NM1_gf = 0.0;

      Tn_filtered_gf.SetSpace(sfes);
      Tn_filtered_gf = 0.0;
   }

   sw_setup.Stop();
   std::cout << "Check 29..." << std::endl;
}


void LoMachSolver::UpdateTimestepHistory(double dt)
{
   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;

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

   // temperature
   NTnm2 = NTnm1;
   NTnm1 = NTn;
   Tnm2 = Tnm1;
   Tnm1 = Tn;
   Tn_next_gf.GetTrueDofs(Tn_next);
   Tn = Tn_next;
   Tn_gf.SetFromTrueDofs(Tn);
}


void LoMachSolver::projectInitialSolution() {
  // Initialize the state.
}


void LoMachSolver::initialTimeStep() {
  auto dataU = U->HostReadWrite();
  int dof = vfes->GetNDofs();

  for (int n = 0; n < dof; n++) {
    Vector state(num_equation);
    for (int eq = 0; eq < num_equation; eq++) state[eq] = dataU[n + eq * dof];
  }

  double partition_C = max_speed;
  MPI_Allreduce(&partition_C, &max_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  dt = CFL * hmin / max_speed / static_cast<double>(dim_);

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }
}


void LoMachSolver::solve() {
  // just restart here
  if (config.restart) {
     restart_files_hdf5("read");
     copyU();
  }

   if (iter==0) { dt = 1.0e-6; } // HARD CODE

   double Umax_lcl = 1.0e-12;
   max_speed = Umax_lcl;
   double Umag;
   int dof = vfes->GetNDofs();

   // temporary hardcodes
   //double t = 0.0;
   double CFL_actual;
   double t_final = 1.0;
   double dtFactor = 0.1;
   //CFL = config.cflNum;
   CFL = config.GetCFLNumber();

   auto dataU = un_gf.HostRead();

   ParGridFunction *r_gf = GetCurrentDensity();
   ParGridFunction *u_gf = GetCurrentVelocity();
   ParGridFunction *p_gf = GetCurrentPressure();
   ParGridFunction *t_gf = GetCurrentTemperature();

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }

   Setup(dt);
   updateU();

   /**/
   ParaViewDataCollection pvdc("turbchan", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(order);
   pvdc.SetCycle(0);
   pvdc.SetTime(time);
   //pvdc.RegisterField("density", r_gf);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("temperature", t_gf);
   pvdc.Save(); // invalid read
   if( rank0_ == true ) std::cout << " Saving final step to paraview: " << iter << endl;
   /**/

   int iter_start = iter;
   if( rank0_ == true ) std::cout << " Starting main loop, from " << iter_start << " to " << MaxIters << endl;

   for (int step = iter_start; step <= MaxIters; step++)
   {

     iter = step;

     /// make a seperate function ///
       auto dataU = un_gf.HostRead();
       for (int n = 0; n < dof; n++) {
         Umag = 0.0;
         for (int eq = 0; eq < dim_; eq++) {
           Umag += dataU[n + eq * dof]*dataU[n + eq * dof];
         }
         Umag = sqrt(Umag);
         Umax_lcl = std::max(Umag,Umax_lcl);
       }
       MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
       dt = (1.0-dtFactor) * dt + dtFactor * CFL*hmin/(max_speed*(double)order) ; // this is conservative
     ////////////////////////////////

      if (rank0_ == true) {
        std::cout << "  N    Time    dt   "<< endl;
        std::cout << " " << step << "    " << time << " " << dt << endl; // conditional jump
      }

      if (step > MaxIters) { break; }
      Step(time, dt, step, iter_start);

      // restart files
      if ( iter%config.itersOut == 0) {
        updateU();
        restart_files_hdf5("write");
      }
   }

   pvdc.SetCycle(iter);
   pvdc.SetTime(time);
   pvdc.Save(); // invalid read and conditional jump
   if( rank0_ == true ) std::cout << " Saving final step to paraview: " << iter << endl;
}

// all the variable coefficient operators musc be re-build every step...
void LoMachSolver::Step(double &time, double dt, const int current_step, const int start_step, bool provisional) {
   sw_step.Start();

   int Vdof = vfes->GetTrueVSize();
   int Pdof = sfes->GetTrueVSize();
   int Tdof = sfes->GetTrueVSize();

   SetTimeIntegrationCoefficients(current_step-start_step);

   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs) {vel_dbc.coeff->SetTime(time + dt);}

   // Set current time for pressure Dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs) {pres_dbc.coeff->SetTime(time + dt);}

   {
     double *dataVisc = bufferVisc->HostWrite();
     double *Tdata = Tn.HostReadWrite();
     double visc[2];
     double prim[nvel+2];
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
       for (int eq = 0; eq < nvel; eq++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         dataVisc[i + eq * Tdof] = visc[0];
         //dataVisc[i + eq * Tdof] = kin_vis; // static value
       }
     }
   }

   H_bdfcoeff.constant = bd0 / dt;
   H_form->Update();
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   HInv->SetOperator(*H);
   if (partial_assembly)
   {
      delete HInvPC;
      Vector diag_pa(vfes->GetTrueVSize());
      H_form->AssembleDiagonal(diag_pa); // invalid read
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
      HInv->SetPreconditioner(*HInvPC);
   }

   // Extrapolated f^{n+1} (source term: f(n+1))
   for (auto &accel_term : accel_terms)
   {
      accel_term.coeff->SetTime(time + dt);
   }

   f_form->Assemble();
   f_form->ParallelAssemble(fn);

   // Nonlinear extrapolated terms (convection: N*(n+1))
   sw_extrap.Start();

   // Nonlinear term at previous steps
   //N->Mult(un, Nun);
   //N->Mult(unm1, Nunm1);
   //N->Mult(unm2, Nunm2);

   // project u to "padded" order space
   {
   double *data = R1PM0_gf.HostReadWrite();
   double *Udata = un.HostReadWrite();
   for (int i = 0; i < Vdof; i++) {
     data[i] = Udata[i];
   }
   }
   R1PX2_gf.ProjectGridFunction(R1PM0_gf);
   {
   double *data = R1PX2_gf.HostReadWrite();
   double *Udata = uBn.HostReadWrite();
   //   for (int i = 0; i < Ndof; i++) {
      for (int i = 0; i < Vdof; i++) {
     Udata[i] = data[i];
   }
   }

   {
   double *data = R1PM0_gf.HostReadWrite();
   double *Udata = unm1.HostReadWrite();
   for (int i = 0; i < Vdof; i++) {
     data[i] = Udata[i];
   }
   }
   R1PX2_gf.ProjectGridFunction(R1PM0_gf);
   {
   double *data = R1PX2_gf.HostReadWrite();
   double *Udata = uBnm1.HostReadWrite();
   //for (int i = 0; i < Ndof; i++) {
   for (int i = 0; i < Vdof; i++) {
     Udata[i] = data[i];
   }
   }

   {
   double *data = R1PM0_gf.HostReadWrite();
   double *Udata = unm2.HostReadWrite();
   for (int i = 0; i < Vdof; i++) {
     data[i] = Udata[i];
   }
   }
   R1PX2_gf.ProjectGridFunction(R1PM0_gf);
   {
   double *data = R1PX2_gf.HostReadWrite();
   double *Udata = uBnm2.HostReadWrite();
   //for (int i = 0; i < Ndof; i++) {
   for (int i = 0; i < Vdof; i++) {
     Udata[i] = data[i];
   }
   }

   N->Mult(uBn, Nun); // invalid read
   N->Mult(uBnm1, Nunm1);
   N->Mult(uBnm2, Nunm2);

   // ab-predictor of nonliner term at {n+1}
   {
      const auto d_Nun = Nun.Read();
      const auto d_Nunm1 = Nunm1.Read();
      const auto d_Nunm2 = Nunm2.Read();
      auto d_Fext = FBext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      //mfem::forall(FBext.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, FBext.Size(),
      {
         d_Fext[i] = ab1_ * d_Nun[i] +
                     ab2_ * d_Nunm1[i] +
                     ab3_ * d_Nunm2[i];
      });
   }

   // project NL product back to v-space
   {
   double *data = R1PX2_gf.HostReadWrite();
   double *Fdata = FBext.HostReadWrite();
   //for (int i = 0; i < Ndof; i++) {
   for (int i = 0; i < Vdof; i++) {
     data[i] = Fdata[i];
   }
   }
   R1PM0_gf.ProjectGridFunction(R1PX2_gf);
   {
   double *data = R1PM0_gf.HostReadWrite();
   double *Fdata = Fext.HostReadWrite();
   for (int i = 0; i < Vdof; i++) {
     Fdata[i] = data[i];
   }
   }

   // add forcing/accel term to Fext
   Fext.Add(1.0, fn);

   // Fext = M^{-1} (F(u^{n}) + f^{n+1}) (F* w/o known part of BDF)
   MvInv->Mult(Fext, tmpR1); // conditional jump
   iter_mvsolve = MvInv->GetNumIterations();
   res_mvsolve = MvInv->GetFinalNorm();
   Fext.Set(1.0, tmpR1);

   // for unsteady term, compute BDF terms (known part of BDF u for full F*)
   {
      const double bd1idt = -bd1 / dt;
      const double bd2idt = -bd2 / dt;
      const double bd3idt = -bd3 / dt;
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Fext = Fext.ReadWrite();
      //mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, Fext.Size(),
      {
         d_Fext[i] += bd1idt * d_un[i] +
                      bd2idt * d_unm1[i] +
                      bd3idt * d_unm2[i];
      });
   }
   sw_extrap.Stop();

   // Pressure Poisson (extrapolated curl(curl(u)) L*(n+1))
   sw_curlcurl.Start();
   {
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Lext = Lext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      //mfem::forall(Lext.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, Lext.Size(),
      {
         d_Lext[i] = ab1_ * d_un[i] +
                     ab2_ * d_unm1[i] +
                     ab3_ * d_unm2[i];
      });
   }

   Lext_gf.SetFromTrueDofs(Lext);
   if (dim_ == 2)
   {
      ComputeCurl2D(Lext_gf, curlu_gf);
      ComputeCurl2D(curlu_gf, curlcurlu_gf, true);
   }
   else
   {
      ComputeCurl3D(Lext_gf, curlu_gf);
      ComputeCurl3D(curlu_gf, curlcurlu_gf);
   }

   curlcurlu_gf.GetTrueDofs(Lext);

   // (!) with a field viscosity, this likely breaks
   // (!) must change to dynamic/rho
   //Lext *= kin_vis;

   {
     const double *dataVisc = bufferVisc->HostRead();
     for (int eq = 0; eq < dim_; eq++) {
       for (int i = 0; i < Tdof; i++) {
         //double rho = 1.0;
         // get nu here
         //transport->ComputeViscosities(rho, double* kin_vis, double* bulk_visc, double* therm_cond)
         //Lext[i + eq * dof] = Lext[i + eq * dof] * kin_vis * 1.0;

         // what if we don't do this?  Then all is well
         //Lext[i + eq * Tdof] = Lext[i + eq * Tdof] * dataVisc[i]; // data2 points to kin_vis
         Lext[i + eq * Tdof] *= dataVisc[i]; // data2 points to kin_vis
       }
     }
   }
   sw_curlcurl.Stop();

   // \tilde{F} = F - \nu CurlCurl(u), (F* + L*)
   FText.Set(-1.0, Lext);
   FText.Add(1.0, Fext);

   // add some if for porder != vorder
   // project FText to p-1
   D->Mult(FText, tmpR0);  // tmp3 c p, tmp2 and resp c (p-1) // possible invalid read
   tmpR0.Neg();

   // Add boundary terms.
   FText_gf.SetFromTrueDofs(FText);
   FText_bdr_form->Assemble();  // invalid read and conditional jump
   FText_bdr_form->ParallelAssemble(FText_bdr);
   g_bdr_form->Assemble();
   g_bdr_form->ParallelAssemble(g_bdr);
   tmpR0.Add(1.0, FText_bdr);
   tmpR0.Add(-bd0 / dt, g_bdr);

   // project rhs to p-space
   {
     double *data = R0PM0_gf.HostReadWrite();
     double *d_buf = tmpR0.HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       data[i] = d_buf[i];
     }
   }
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);
   {
     double *data = resp.HostReadWrite();
     double *d_buf = R0PM1_gf.HostReadWrite();
     for (int i = 0; i < Pdof; i++) {
       data[i] = d_buf[i];
     }
   }

   if (pres_dbcs.empty()) { Orthogonalize(resp); }
   for (auto &pres_dbc : pres_dbcs) { pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr); }

   // ?
   sfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

   {
     double *data = bufferInvRho->HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       //data[i] = (Rgas * Tdata[i]) / ambientPressure;
       data[i] = 1.0;
     }
   }


   Vector X1, B1;
   if (partial_assembly)
   {
      auto *SpC = Sp.As<ConstrainedOperator>();
      EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
   }
   else
   {
      Sp_form->FormLinearSystem(pres_ess_tdof, pn_gf, resp_gf, Sp, X1, B1, 1);
   }

   // actual implicit solve for p(n+1)
   sw_spsolve.Start();
   SpInv->Mult(B1, X1); // conditional jump
   sw_spsolve.Stop();
   iter_spsolve = SpInv->GetNumIterations();
   res_spsolve = SpInv->GetFinalNorm();
   Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

   // If the boundary conditions on the pressure are pure Neumann remove the
   // nullspace by removing the mean of the pressure solution. This is also
   // ensured by the OrthoSolver wrapper for the preconditioner which removes
   // the nullspace after every application.
   if (pres_dbcs.empty()) { MeanZero(pn_gf); }

   pn_gf.GetTrueDofs(pn);

   // Project velocity (not really "projecting" here...)
   //G->Mult(pn, resu); // grad(P) in resu

   // project p to v-space
   {
   double *data = R0PM1_gf.HostReadWrite();
   double *d_buf = pn.HostReadWrite();
   for (int i = 0; i < Pdof; i++) {
     data[i] = d_buf[i];
   }
   }
   R0PM0_gf.ProjectGridFunction(R0PM1_gf);
   {
   double *d_buf = R0PM0_gf.HostReadWrite();
   double *data = pnBig.HostReadWrite();
   for (int i = 0; i < Tdof; i++) {
     data[i] = d_buf[i];
   }
   }

   // grad(P)
   G->Mult(pnBig, resu);

   resu.Neg(); // -gradP
   Mv->Mult(Fext, tmpR1); //Mv{F*}
   resu.Add(1.0, tmpR1); // add to resu

   for (auto &vel_dbc : vel_dbcs) { un_next_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);}

   // (?)
   vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

   Vector X2, B2;
   if (partial_assembly)
   {
      auto *HC = H.As<ConstrainedOperator>();
      EliminateRHS(*H_form, *HC, vel_ess_tdof, un_next_gf, resu_gf, X2, B2, 1);
   }
   else
   {
      H_form->FormLinearSystem(vel_ess_tdof, un_next_gf, resu_gf, H, X2, B2, 1);
   }
   sw_hsolve.Start();
   HInv->Mult(B2, X2); // conditional jump
   sw_hsolve.Stop();
   iter_hsolve = HInv->GetNumIterations();
   res_hsolve = HInv->GetFinalNorm();
   H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);

   un_next_gf.GetTrueDofs(un_next);


   // begin temperature......................................

   // Set current time for temperature Dirichlet boundary conditions.
   for (auto &temp_dbc : temp_dbcs) {temp_dbc.coeff->SetTime(time + dt);}   
   {
     double *data = bufferAlpha->HostReadWrite();
     double *Tdata = Tn.HostReadWrite();
     double visc[2];
     double prim[nvel+2];
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         data[i] = visc[0] / Pr;
     }
   }

   Ht_bdfcoeff.constant = bd0 / dt;
   // update visc here
   Ht_form->Update();
   Ht_form->Assemble();
   Ht_form->FormSystemMatrix(temp_ess_tdof, Ht);

   HtInv->SetOperator(*Ht);
   if (partial_assembly)
   {
      delete HtInvPC;
      Vector diag_pa(sfes->GetTrueVSize());
      Ht_form->AssembleDiagonal(diag_pa);
      HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
      HtInv->SetPreconditioner(*HtInvPC);
   }

   // add source terms to Text here (e.g. viscous heating)

   // for unsteady term, compute and add known part of BDF unsteady term
   {
      const double bd1idt = -bd1 / dt;
      const double bd2idt = -bd2 / dt;
      const double bd3idt = -bd3 / dt;
      const double *d_tn = Tn.Read();

      const double *d_tnm1 = Tnm1.Read();
      const double *d_tnm2 = Tnm2.Read();
      double *d_Text = Text.HostReadWrite();
      //mfem::forall(Text.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, Text.Size(),
      {
        d_Text[i] = bd1idt * d_tn[i] +
                    bd2idt * d_tnm1[i] +
                    bd3idt * d_tnm2[i];
      });
   }

   // Add boundary terms.
   Text_gf.SetFromTrueDofs(Text);
   Text_bdr_form->Assemble();
   Text_bdr_form->ParallelAssemble(Text_bdr);
   t_bdr_form->Assemble();
   t_bdr_form->ParallelAssemble(t_bdr);

   Mt->Mult(Text, tmpR0);
   resT.Set(1.0, tmpR0);

   // advection => will break is t&u arent in the same space
   bufferTemp = new ParGridFunction(vfes);
   double *dataTemp = bufferTemp->HostReadWrite();
   double *Udnm0 = un.HostReadWrite();
   double *Tdnm0 = Tn.HostReadWrite();
   double *Udnm1 = unm1.HostReadWrite();
   double *Tdnm1 = Tnm1.HostReadWrite();
   double *Udnm2 = unm2.HostReadWrite();
   double *Tdnm2 = Tnm2.HostReadWrite();
   const auto ab1_ = ab1;
   const auto ab2_ = ab2;
   const auto ab3_ = ab3;
   for (int eq = 0; eq < dim_; eq++) {
     for (int i = 0; i < Tdof; i++) {
       dataTemp[i + eq * Tdof] = ab1_ * Udnm0[i + eq * Tdof] * Tdnm0[i] +
        	                 ab2_ * Udnm1[i + eq * Tdof] * Tdnm1[i] +
	                         ab3_ * Udnm2[i + eq * Tdof] * Tdnm2[i];
       }
   }

   Dt->Mult(*bufferTemp, tmpR0); // explicit div(uT) at {n}
   resT.Add(-1.0, tmpR0);

   // what is this?
   for (auto &temp_dbc : temp_dbcs) { Tn_next_gf.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr); }

   // (?)
   sfes->GetRestrictionMatrix()->MultTranspose(resT, resT_gf);

   Vector Xt2, Bt2;
   if (partial_assembly)
   {
      auto *HC = Ht.As<ConstrainedOperator>();
      EliminateRHS(*Ht_form, *HC, temp_ess_tdof, Tn_next_gf, resT_gf, Xt2, Bt2, 1);
   }
   else
   {
      Ht_form->FormLinearSystem(temp_ess_tdof, Tn_next_gf, resT_gf, Ht, Xt2, Bt2, 1);
   }

   // solve helmholtz eq for temp
   HtInv->Mult(Bt2, Xt2);
   iter_htsolve = HtInv->GetNumIterations();
   res_htsolve = HtInv->GetFinalNorm();
   Ht_form->RecoverFEMSolution(Xt2, resT_gf, Tn_next_gf);
   //Ht_form->RecoverFEMSolution(Xt2, resT, Tn_next_gf);
   Tn_next_gf.GetTrueDofs(Tn_next);

   // end temperature....................................

   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      // If numerical integration is active, there is no solve (thus no
      // iterations), on the inverse velocity mass application.
      if (!numerical_integ)
      {
         mfem::out << std::setw(5) << "MVIN " << std::setw(5) << std::fixed
                   << iter_mvsolve << "   " << std::setw(3)
                   << std::setprecision(2) << std::scientific << res_mvsolve
                   << "   " << 1e-12 << "\n";
      }
      mfem::out << std::setw(5) << "PRES " << std::setw(5) << std::fixed
                << iter_spsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_spsolve << "   " << rtol_spsolve // conditional jump
                << "\n";
      mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed
                << iter_hsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_hsolve << "   " << rtol_hsolve // conditional jump
                << "\n";
      mfem::out << std::setw(5) << "TEMP " << std::setw(5) << std::fixed
                << iter_htsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_htsolve << "   " << rtol_htsolve
                << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }
}


 // copies for compatability => is conserved (U) even necessary with this method?
void LoMachSolver::updateU() {

  std::cout << " updating big U vector..." << endl;
  // primitive state
  {
    double *dataUp = Up->HostReadWrite();

    const auto d_rn_gf = rn_gf.Read();
    //mfem::forall(rn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, rn_gf.Size(),
    {
      dataUp[i] = d_rn_gf[i];
    });

    int vstart = sfes->GetNDofs();
    const auto d_un_gf = un_gf.Read();
    //mfem::forall(un_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, un_gf.Size(),
    {
      dataUp[i + vstart] = d_un_gf[i];
    });

    int tstart = (1+nvel)*(sfes->GetNDofs());
    const auto d_tn_gf = Tn_gf.Read();
    //mfem::forall(Tn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, Tn_gf.Size(),
    {
      dataUp[i + tstart] = d_tn_gf[i];
    });
  }
}

void LoMachSolver::copyU() {
  std::cout << " copying from U vector..." << endl;
  // primitive state
  {

    double *dataUp = Up->HostReadWrite();
    double *d_rn_gf = rn_gf.ReadWrite();
    //mfem::forall(rn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, rn_gf.Size(),
    {
      d_rn_gf[i] = dataUp[i];
    });

    int vstart = sfes->GetNDofs();
    double *d_un_gf = un_gf.ReadWrite();
    //mfem::forall(un_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, un_gf.Size(),
    {
       d_un_gf[i] = dataUp[i + vstart];
    });

    int tstart = (1+nvel)*(sfes->GetNDofs());
    double *d_tn_gf = Tn_gf.ReadWrite();
    //mfem::forall(Tn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, Tn_gf.Size(),
    {
       d_tn_gf[i] = dataUp[i + tstart];
    });
  }
}



void LoMachSolver::MeanZero(ParGridFunction &v)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(v.ParFESpace());
      auto *dlfi = new DomainLFIntegrator(onecoeff);
      if (numerical_integ)
      {
         const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2*order - 1);
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


void LoMachSolver::EliminateRHS(Operator &A,
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


void LoMachSolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, sfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, sfes->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}


void LoMachSolver::ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
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
      dshape.SetSize(elndofs, dim_);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim_);
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


void LoMachSolver::ComputeCurl2D(ParGridFunction &u,
                                 ParGridFunction &cu,
                                 bool assume_scalar)
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
      dshape.SetSize(elndofs, dim_);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim_);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         if (assume_scalar)
         {
            curl.SetSize(2);
            curl(0) = grad(0, 1);
            curl(1) = -grad(0, 0);
         }
         else
         {
            curl.SetSize(2);
            curl(0) = grad(1, 0) - grad(0, 1);
            curl(1) = 0.0;
         }

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

   // Communication.

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

double LoMachSolver::ComputeCFL(ParGridFunction &u, double dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);
         if (vdim == 3)
         {
            cflz = fabs(dt * uz(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}


void LoMachSolver::EnforceCFL(double maxCFL, ParGridFunction &u, double &dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   double cflx = 0.0;
   double cfly = 0.0;
   double cflz = 0.0;
   double cflm = 0.0;
   double cflmax = 0.0;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                               fe->GetOrder());
      ElementTransformation *tr = fes->GetElementTransformation(e);

      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      double hmin = pmesh_u->GetElementSize(e, 1) /
                    (double) fes->GetElementOrder(0);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         const DenseMatrix &invJ = tr->InverseJacobian();
         const double detJinv = 1.0 / tr->Jacobian().Det();

         if (vdim == 2)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
         }
         else if (vdim == 3)
         {
            ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                     + uz(i) * invJ(2, 0))
                    * detJinv;
            us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                     + uz(i) * invJ(2, 1))
                    * detJinv;
            ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                     + uz(i) * invJ(2, 2))
                    * detJinv;
         }

	 // here
	 double dtx, dty, dtz;
         //cflx = fabs(dt * ux(i) / hmin);
         //cfly = fabs(dt * uy(i) / hmin);
	 dtx = maxCFL * hmin / fabs(ux(i));
	 dty = maxCFL * hmin / fabs(uy(i));
         if (vdim == 3)
         {
	   //cflz = fabs(dt * uz(i) / hmin);
	   dtz = maxCFL * hmin / fabs(uz(i));
         }
         //cflm = cflx + cfly + cflz;
         //cflmax = fmax(cflmax, cflm);
	 dt = fmin(dtx,dty);
	 dt = fmin(dt,dtz);
      }
   }

   double cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MPI_DOUBLE,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return;
}


void LoMachSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i) {
	if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((vel_ess_attr[i] && attr[i]) == 0,"Duplicate boundary definition deteceted.");
      if (attr[i] == 1) { vel_ess_attr[i] = 1; }
   }
}

void LoMachSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}


void LoMachSolver::AddPresDirichletBC(Coefficient *coeff, Array<int> &attr)
{
   pres_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Pressure Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((pres_ess_attr[i] && attr[i]) == 0,"Duplicate boundary definition deteceted.");
      if (attr[i] == 1) { pres_ess_attr[i] = 1; }
   }
}

void LoMachSolver::AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(f), attr);
}


void LoMachSolver::AddTempDirichletBC(Coefficient *coeff, Array<int> &attr)
{

   temp_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Temperature Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i) {
         if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((temp_ess_attr[i] && attr[i]) == 0,"Duplicate boundary definition deteceted.");
      if (attr[i] == 1) { temp_ess_attr[i] = 1; }
   }
}

void LoMachSolver::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void LoMachSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }
}

void LoMachSolver::AddAccelTerm(VecFuncT *f, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}


void LoMachSolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step + 1, max_bdf_order);

   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   double rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   double rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];

   if (bdf_order == 3)
   {
      rho2 = dthist[1] / dthist[2];
   }

   if (step == 0 && bdf_order == 1)
   {
      bd0 = 1.0;
      bd1 = -1.0;
      bd2 = 0.0;
      bd3 = 0.0;
      ab1 = 1.0;
      ab2 = 0.0;
      ab3 = 0.0;
   }
   else if (step >= 1 && bdf_order == 2)
   {
      bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
      bd1 = -(1.0 + rho1);
      bd2 = pow(rho1, 2.0) / (1.0 + rho1);
      bd3 = 0.0;
      ab1 = 1.0 + rho1;
      ab2 = -rho1;
      ab3 = 0.0;
   }
   else if (step >= 2 && bdf_order == 3)
   {
      bd0 = 1.0 + rho1 / (1.0 + rho1)
            + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
      bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
      bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
      ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
      ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
      ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
   }
}


void LoMachSolver::PrintTimingData()
{
   double my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_step.RealTime();
   my_rt[2] = sw_extrap.RealTime();
   my_rt[3] = sw_curlcurl.RealTime();
   my_rt[4] = sw_spsolve.RealTime();
   my_rt[5] = sw_hsolve.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MPI_DOUBLE, MPI_MAX, 0, pmesh->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP"
                << std::setw(10) << "EXTRAP" << std::setw(10) << "CURLCURL"
                << std::setw(10) << "PSOLVE" << std::setw(10) << "HSOLVE"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4]
                << std::setw(10) << my_rt[5] << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10)
                << my_rt[1] / my_rt[1] << std::setw(10) << my_rt[2] / my_rt[1]
                << std::setw(10) << my_rt[3] / my_rt[1] << std::setw(10)
                << my_rt[4] / my_rt[1] << std::setw(10) << my_rt[5] / my_rt[1]
                << "\n";

      mfem::out << std::setprecision(8);
   }
}


void LoMachSolver::PrintInfo()
{
   int fes_size0 = vfes->GlobalVSize();
   int fes_size1 = sfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
     //      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
     mfem::out  << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}


LoMachSolver::~LoMachSolver() {

   delete FText_gfcoeff;
   delete g_bdr_form;
   delete FText_bdr_form;
   delete Text_gfcoeff;
   delete t_bdr_form;
   delete Text_bdr_form;
   delete f_form;

   delete mass_lf;
   delete N;
   delete Mv_form;
   delete Mt_form;
   delete Sp_form;
   delete D_form;
   delete Dt_form;
   delete G_form;
   delete H_form;
   delete Ht_form;

   delete MvInv;
   delete MtInv;
   delete HInv;
   delete HtInv;
   delete HInvPC;
   delete HtInvPC;
   delete MvInvPC;
   delete MtInvPC;

   delete SpInv;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete lor;

   delete fec;

   delete vfes;
   delete sfes;
   delete fvfes;

   delete vfec_filter;
   delete vfes_filter;
   delete tfec_filter;
   delete tfes_filter;
}

// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {

  //if (verbose) grvy_printf(ginfo, "parsing solver options...\n");
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
  tpsP_->getInput("loMach/order", loMach_opts_.order, 1);
  tpsP_->getInput("loMach/ref_levels", loMach_opts_.ref_levels, 0);
  tpsP_->getInput("loMach/max_iter", loMach_opts_.max_iter, 100);
  tpsP_->getInput("loMach/rtol", loMach_opts_.rtol, 1.0e-6);

  // dump options to screen for user inspection
  if (mpi_.Root()) {
    loMach_opts_.print(std::cout);
  }
}


///* move these to separate support class *//
void LoMachSolver::parseSolverOptions2() {
  //tpsP->getRequiredInput("flow/mesh", config.meshFile);

  // flow/numerics
  parseFlowOptions();

  // time integration controls
  parseTimeIntegrationOptions();

  // statistics
  parseStatOptions();

  // I/O settings
  parseIOSettings();

  // initial conditions
  //if (!config.use_mms_) { parseICOptions(); }
  parseICOptions();

  // fluid presets
  parseFluidPreset();

  // changed the order of parsing in order to use species list.
  // boundary conditions. The number of boundaries of each supported type if
  // parsed first. Then, a section corresponding to each defined boundary is parsed afterwards
  parseBCInputs();

  // periodicity
  parsePeriodicInputs();

  // post-process visualization inputs
  parsePostProcessVisualizationInputs();

  return;
}

void LoMachSolver::parseFlowOptions() {
  std::map<std::string, int> sgsModel;
  sgsModel["none"] = 0;
  sgsModel["smagorinsky"] = 1;
  sgsModel["sigma"] = 2;
  tpsP_->getInput("loMach/order", config.solOrder, 2);
  tpsP_->getInput("loMach/integrationRule", config.integrationRule, 1);
  tpsP_->getInput("loMach/basisType", config.basisType, 1);
  tpsP_->getInput("loMach/maxIters", config.numIters, 10);
  tpsP_->getInput("loMach/outputFreq", config.itersOut, 50);
  tpsP_->getInput("loMach/timingFreq", config.timingFreq, 100);
  tpsP_->getInput("loMach/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP_->getInput("loMach/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  tpsP_->getInput("loMach/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("loMach/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("loMach/SutherlandPr", config.sutherland_.Pr, 0.71);

  tpsP_->getInput("loMach/enablePressureForcing", config.isForcing, false);
  if (config.isForcing) {
    for (int d = 0; d < 3; d++) tpsP_->getRequiredVecElem("loMach/pressureGrad", config.gradPress[d], d);
  }
  tpsP_->getInput("loMach/refinement_levels", config.ref_levels, 0);
  tpsP_->getInput("loMach/computeDistance", config.compute_distance, false);

  std::string type;
  tpsP_->getInput("loMach/sgsModel", type, std::string("none"));
  config.sgsModelType = sgsModel[type];

  double sgs_const = 0.;
  if (config.sgsModelType == 1) {
    sgs_const = 0.12;
  } else if (config.sgsModelType == 2) {
    sgs_const = 0.135;
  }
  tpsP_->getInput("loMach/sgsModelConstant", config.sgs_model_const, sgs_const);

  assert(config.solOrder > 0);
  assert(config.numIters >= 0);
  assert(config.itersOut > 0);
  assert(config.refLength > 0);
}

void LoMachSolver::parseTimeIntegrationOptions() {
  std::string type;
  tpsP_->getInput("time/cfl", config.cflNum, 0.2);
  tpsP_->getInput("time/enableConstantTimestep", config.constantTimeStep, false);
  tpsP_->getInput("time/dt_fixed", config.dt_fixed, -1.);
}

void LoMachSolver::parseStatOptions() {
  tpsP_->getInput("averaging/saveMeanHist", config.meanHistEnable, false);
  tpsP_->getInput("averaging/startIter", config.startIter, 0);
  tpsP_->getInput("averaging/sampleFreq", config.sampleInterval, 0);
  std::cout << " sampleInterval: " << config.sampleInterval << endl;
  tpsP_->getInput("averaging/enableContinuation", config.restartMean, false);
}

void LoMachSolver::parseIOSettings() {
  tpsP_->getInput("io/outdirBase", config.outputFile, std::string("output-default"));
  tpsP_->getInput("io/enableRestart", config.restart, false);
  tpsP_->getInput("io/exitCheckFreq", config.exit_checkFrequency_, 500);

  std::string restartMode;
  tpsP_->getInput("io/restartMode", restartMode, std::string("standard"));
  if (restartMode == "variableP") {
    config.restartFromAux = true;
  } else if (restartMode == "singleFileWrite") {
    config.restart_serial = "write";
  } else if (restartMode == "singleFileRead") {
    config.restart_serial = "read";
  } else if (restartMode == "singleFileReadWrite") {
    config.restart_serial = "readwrite";
  } else if (restartMode != "standard") {
    grvy_printf(GRVY_ERROR, "\nUnknown restart mode -> %s\n", restartMode.c_str());
    exit(ERROR);
  }
}

void LoMachSolver::parseRMSJobOptions() {
  tpsP_->getInput("jobManagement/enableAutoRestart", config.rm_enableMonitor_, false);
  tpsP_->getInput("jobManagement/timeThreshold", config.rm_threshold_, 15 * 60);  // 15 minutes
  tpsP_->getInput("jobManagement/checkFreq", config.rm_checkFrequency_, 500);     // 500 iterations
}

void LoMachSolver::parseICOptions() {
  tpsP_->getRequiredInput("initialConditions/rho", config.initRhoRhoVp[0]);
  tpsP_->getRequiredInput("initialConditions/rhoU", config.initRhoRhoVp[1]);
  tpsP_->getRequiredInput("initialConditions/rhoV", config.initRhoRhoVp[2]);
  tpsP_->getRequiredInput("initialConditions/rhoW", config.initRhoRhoVp[3]);
  tpsP_->getRequiredInput("initialConditions/pressure", config.initRhoRhoVp[4]);
}

void LoMachSolver::parseFluidPreset() {
  std::string fluidTypeStr;
  tpsP_->getInput("loMach/fluid", fluidTypeStr, std::string("dry_air"));
  if (fluidTypeStr == "dry_air") {
    config.workFluid = DRY_AIR;
    tpsP_->getInput("loMach/specific_heat_ratio", config.dryAirInput.specific_heat_ratio, 1.4);
    tpsP_->getInput("loMach/gas_constant", config.dryAirInput.gas_constant, 287.058);
  } else if (fluidTypeStr == "user_defined") {
    config.workFluid = USER_DEFINED;
  } else if (fluidTypeStr == "lte_table") {
    config.workFluid = LTE_FLUID;
    std::string thermo_file;
    tpsP_->getRequiredInput("loMach/lte/thermo_table", thermo_file);
    config.lteMixtureInput.thermo_file_name = thermo_file;
    std::string trans_file;
    tpsP_->getRequiredInput("loMach/lte/transport_table", trans_file);
    config.lteMixtureInput.trans_file_name = trans_file;
    std::string e_rev_file;
    tpsP_->getRequiredInput("loMach/lte/e_rev_table", e_rev_file);
    config.lteMixtureInput.e_rev_file_name = e_rev_file;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown fluid preset supplied at runtime -> %s", fluidTypeStr.c_str());
    exit(ERROR);
  }

  if ((rank0_ == true) && (config.workFluid != USER_DEFINED))
    cout << "Fluid is set to the preset '" << fluidTypeStr << "'. Input options in [plasma_models] will not be used."
         << endl;
}

void LoMachSolver::parseBCInputs() {
  // number of BC regions defined
  // int numWalls, numInlets, numOutlets;
  tpsP_->getInput("boundaryConditions/numWalls", numWalls, 0);
  tpsP_->getInput("boundaryConditions/numInlets", numInlets, 0);
  tpsP_->getInput("boundaryConditions/numOutlets", numOutlets, 0);

  // Wall.....................................
  std::map<std::string, WallType> wallMapping;
  wallMapping["inviscid"] = INV;
  wallMapping["slip"] = SLIP;
  wallMapping["viscous_adiabatic"] = VISC_ADIAB;
  wallMapping["viscous_isothermal"] = VISC_ISOTH;
  wallMapping["viscous_general"] = VISC_GNRL;

  config.wallBC.resize(numWalls);
  for (int i = 1; i <= numWalls; i++) {
    int patch;
    double temperature;
    std::string type;
    std::string basepath("boundaryConditions/wall" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
    if (type == "viscous_isothermal") {
      //std::cout << "*Caught a wall type! " <<  i << " of " << numWalls << endl;
      tpsP_->getRequiredInput((basepath + "/temperature").c_str(), temperature);
      config.wallBC[i - 1].hvyThermalCond = ISOTH;
      config.wallBC[i - 1].elecThermalCond = ISOTH;
      config.wallBC[i - 1].Th = temperature;
      config.wallBC[i - 1].Te = temperature;
    } else if (type == "viscous_adiabatic") {
      config.wallBC[i - 1].hvyThermalCond = ADIAB;
      config.wallBC[i - 1].elecThermalCond = ADIAB;
      config.wallBC[i - 1].Th = -1.0;
      config.wallBC[i - 1].Te = -1.0;
    } else if (type == "viscous_general") {
      std::map<std::string, ThermalCondition> thmCondMap;
      thmCondMap["adiabatic"] = ADIAB;
      thmCondMap["isothermal"] = ISOTH;
      thmCondMap["sheath"] = SHTH;
      thmCondMap["none"] = NONE_THMCND;

      std::string hvyType, elecType;
      tpsP_->getRequiredInput((basepath + "/heavy_thermal_condition").c_str(), hvyType);
      config.wallBC[i - 1].hvyThermalCond = thmCondMap[hvyType];

      config.wallBC[i - 1].Th = -1.0;
      switch (config.wallBC[i - 1].hvyThermalCond) {
        case ISOTH: {
          double Th;
          tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Th);
          config.wallBC[i - 1].Th = Th;
        } break;
        case SHTH: {
          grvy_printf(GRVY_ERROR, "Wall%d: sheath condition is only supported for electron!\n", i);
          exit(-1);
        } break;
        case NONE_THMCND: {
          grvy_printf(GRVY_ERROR, "Wall%d: must specify heavies' thermal condition!\n", i);
          exit(-1);
        }
        default:
          break;
      }

      // NOTE(kevin): sheath can exist even for single temperature.
      if (config.twoTemperature) {
        tpsP_->getRequiredInput((basepath + "/electron_thermal_condition").c_str(), elecType);
        if (thmCondMap[elecType] == NONE_THMCND) {
          grvy_printf(GRVY_ERROR, "Wall%d: electron thermal condition must be specified for two-temperature plasma!\n",
                      i);
          exit(-1);
        }
        config.wallBC[i - 1].elecThermalCond = thmCondMap[elecType];
      } else {
        tpsP_->getInput((basepath + "/electron_thermal_condition").c_str(), elecType, std::string("none"));
        config.wallBC[i - 1].elecThermalCond = (thmCondMap[elecType] == SHTH) ? SHTH : NONE_THMCND;
      }
      config.wallBC[i - 1].Te = -1.0;
      switch (config.wallBC[i - 1].elecThermalCond) {
        case ISOTH: {
          double Te;
          tpsP_->getInput((basepath + "/electron_temperature").c_str(), Te, -1.0);
          if (Te < 0.0) {
            grvy_printf(GRVY_INFO, "Wall%d: no input for electron_temperature. Using temperature instead.\n", i);
            tpsP_->getRequiredInput((basepath + "/temperature").c_str(), Te);
          }
          if (Te < 0.0) {
            grvy_printf(GRVY_ERROR, "Wall%d: invalid electron temperature: %.8E!\n", i, Te);
            exit(-1);
          }
          config.wallBC[i - 1].Te = Te;
        } break;
        case SHTH: {
          if (!config.ambipolar) {
            grvy_printf(GRVY_ERROR, "Wall%d: plasma must be ambipolar for sheath condition!\n", i);
            exit(-1);
          }
        }
        default:
          break;
      }
    }

    // NOTE(kevin): maybe redundant at this point. Kept this just in case.
    std::pair<int, WallType> patchType;
    patchType.first = patch;
    patchType.second = wallMapping[type];
    config.wallPatchType.push_back(patchType);
  }

  // Inlet......................................
  std::map<std::string, InletType> inletMapping;
  inletMapping["subsonic"] = SUB_DENS_VEL;
  inletMapping["nonreflecting"] = SUB_DENS_VEL_NR;
  inletMapping["nonreflectingConstEntropy"] = SUB_VEL_CONST_ENT;

  for (int i = 1; i <= numInlets; i++) {
    int patch;
    double density;
    std::string type;
    std::string basepath("boundaryConditions/inlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
    // all inlet BCs require 4 inputs (density + vel(3))
    {
      Array<double> uvw;
      tpsP_->getRequiredInput((basepath + "/density").c_str(), density);
      tpsP_->getRequiredVec((basepath + "/uvw").c_str(), uvw, 3);
      config.inletBC.Append(density);
      config.inletBC.Append(uvw, 3);
    }
    // For multi-component gas, require (numActiveSpecies)-more inputs.
    if ((config.workFluid != DRY_AIR) && (config.numSpecies > 1)) {
      grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
      if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

      for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index
        double Ysp;
        // read mass fraction of species as listed in the input file.
        int inputSp = config.mixtureToInputMap[sp];
        std::string speciesBasePath(basepath + "/mass_fraction/species" + std::to_string(inputSp + 1));
        tpsP_->getRequiredInput(speciesBasePath.c_str(), Ysp);
        config.inletBC.Append(Ysp);
      }
    }
    std::pair<int, InletType> patchType;
    patchType.first = patch;
    patchType.second = inletMapping[type];
    config.inletPatchType.push_back(patchType);
  }

  // Outlet.......................................
  std::map<std::string, OutletType> outletMapping;
  outletMapping["subsonicPressure"] = SUB_P;
  outletMapping["nonReflectingPressure"] = SUB_P_NR;
  outletMapping["nonReflectingMassFlow"] = SUB_MF_NR;
  outletMapping["nonReflectingPointBasedMassFlow"] = SUB_MF_NR_PW;

  for (int i = 1; i <= numOutlets; i++) {
    int patch;
    double pressure, massFlow;
    std::string type;
    std::string basepath("boundaryConditions/outlet" + std::to_string(i));

    tpsP_->getRequiredInput((basepath + "/patch").c_str(), patch);
    tpsP_->getRequiredInput((basepath + "/type").c_str(), type);

    if ((type == "subsonicPressure") || (type == "nonReflectingPressure")) {
      tpsP_->getRequiredInput((basepath + "/pressure").c_str(), pressure);
      config.outletBC.Append(pressure);
    } else if ((type == "nonReflectingMassFlow") || (type == "nonReflectingPointBasedMassFlow")) {
      tpsP_->getRequiredInput((basepath + "/massFlow").c_str(), massFlow);
      config.outletBC.Append(massFlow);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown outlet BC supplied at runtime -> %s", type.c_str());
      exit(ERROR);
    }

    std::pair<int, OutletType> patchType;
    patchType.first = patch;
    patchType.second = outletMapping[type];
    config.outletPatchType.push_back(patchType);
  }
}

void LoMachSolver::parsePostProcessVisualizationInputs() {
  if (tpsP_->isVisualizationMode()) {
    tpsP_->getRequiredInput("post-process/visualization/prefix", config.postprocessInput.prefix);
    tpsP_->getRequiredInput("post-process/visualization/start-iter", config.postprocessInput.startIter);
    tpsP_->getRequiredInput("post-process/visualization/end-iter", config.postprocessInput.endIter);
    tpsP_->getRequiredInput("post-process/visualization/frequency", config.postprocessInput.freq);
  }
}

void LoMachSolver::parsePeriodicInputs() {
  tpsP_->getInput("periodicity/enablePeriodic", config.periodic, false);
  tpsP_->getInput("periodicity/xTrans", config.xTrans, 1.0e12);
  tpsP_->getInput("periodicity/yTrans", config.yTrans, 1.0e12);
  tpsP_->getInput("periodicity/zTrans", config.zTrans, 1.0e12);
}

/// HERE HERE HERE
void LoMachSolver::initSolutionAndVisualizationVectors() {
  visualizationVariables_.clear();
  visualizationNames_.clear();

  offsets = new Array<int>(num_equation + 1);
  for (int k = 0; k <= num_equation; k++) {
    (*offsets)[k] = k * fvfes->GetNDofs();
  }
  u_block = new BlockVector(*offsets);
  up_block = new BlockVector(*offsets);

  U = new ParGridFunction(fvfes, u_block->HostReadWrite());
  Up = new ParGridFunction(fvfes, up_block->HostReadWrite());

  ioData.registerIOFamily("Solution state variables", "/solution", Up);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "u", 1);
  ioData.registerIOVar("/solution", "v", 2);
  if (nvel == 3) {
    ioData.registerIOVar("/solution", "w", 3);
    ioData.registerIOVar("/solution", "temperature", 4);
  } else {
    ioData.registerIOVar("/solution", "temperature", 3);
  }
}

///* temporary, remove when hooked up *//
double mesh_stretching_func(const double y)
{
   double C = 1.8;
   double delta = 1.0;
   return delta * tanh(C * (2.0 * y - 1.0)) / tanh(C);
}

void CopyDBFIntegrators(ParBilinearForm *src, ParBilinearForm *dst)
{
   Array<BilinearFormIntegrator *> *bffis = src->GetDBFI();
   for (int i = 0; i < bffis->Size(); ++i)
   { dst->AddDomainIntegrator((*bffis)[i]); }
}


void accel(const Vector &x, double t, Vector &f)
{
   f(0) = 0.1;
   f(1) = 0.0;
   f(2) = 0.0;
}

void vel_ic(const Vector &coords, double t, Vector &u)
{
  
   double yp;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   double pi = 3.14159265359;
   double A = -1.0 * pi;
   double B = -1.0 * pi;
   double C = +2.0 * pi;
   double aL = 2.0 / (2.0) * pi;
   double bL = 2.0 * pi;
   double cL = 2.0 * pi;   
   double M = 1.0;
   double scl;

   
   u(0) = M;
   u(1) = 0.0;
   u(2) = 0.0;

   u(0) += 0.2 * M * A * cos(aL*x) * sin(bL*y) * sin(cL*z);
   u(1) += 0.2 * M * B * sin(aL*x) * cos(bL*y) * sin(cL*z);
   u(2) += 0.2 * M * C * sin(aL*x) * sin(bL*y) * cos(cL*z);

   u(0) -= 0.1 * M * A * cos(2.0*aL*x) * sin(2.0*bL*y) * sin(2.0*cL*z);
   u(1) -= 0.1 * M * B * sin(2.0*aL*x) * cos(2.0*bL*y) * sin(2.0*cL*z);
   u(2) -= 0.1 * M * C * sin(2.0*aL*x) * sin(2.0*bL*y) * cos(2.0*cL*z);

   u(0) += 0.05 * M * A * cos(4.0*aL*x) * sin(4.0*bL*y) * sin(4.0*cL*z);
   u(1) += 0.05 * M * B * sin(4.0*aL*x) * cos(4.0*bL*y) * sin(4.0*cL*z);
   u(2) += 0.05 * M * C * sin(4.0*aL*x) * sin(4.0*bL*y) * cos(4.0*cL*z);

   scl = std::max(1.0-std::pow((y-0.0)/0.2,4),0.0);
   u(0) = u(0) * scl;
   u(1) = u(1) * scl;
   u(2) = u(2) * scl;
   
}


void vel_wall(const Vector &x, double t, Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0;
   u(2) = 0.0;   
}

double temp_ic(const Vector &coords, double t)
{
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp = 300.0;
   return temp;
}

double temp_wall(const Vector &x, double t)  
{
   double temp = 298.15;
   return temp;
}
