
///*  Add description later //

#include "loMach.hpp"
//#include "../../general/forall.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
//#include "forall.hpp"
//#include "solvers.hpp"

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
    : mpi_(mpi), loMach_opts_(loMach_opts), offsets_(3) {
  tpsP_ = tps;
  pmesh = NULL;

  // verify running on cpu
  /*
  if (tpsP_->getDeviceConfig() != "cpu") {
    if (mpi.Root()) {
      grvy_printf(GRVY_ERROR, "[ERROR] Low Mach simulation currently only supported on cpu.\n");
    }
    exit(1);
  }
  */

  //grvy_printf(GRVY_INFO, "Basic LoMachSolver empty \n");  
  //plasma_conductivity_ = NULL;
  //plasma_conductivity_coef_ = NULL;
  //joule_heating_ = NULL;
}




/*
NavierSolver::NavierSolver(ParMesh *mesh, int order, int porder, int norder, double kin_vis, double Po, double Rgas)
   : pmesh(mesh), order(order), porder(porder), norder(norder), kin_vis(kin_vis), Po(Po), Rgas(Rgas),
     gll_rules(0, Quadrature1D::GaussLobatto)
     //gll_rules(0, Quadrature1D::GaussLegendre)          
{
*/

void LoMachSolver::initialize() {
  
   bool verbose = mpi_.Root();
   if (verbose) grvy_printf(ginfo, "Initializing loMach solver.\n");

   // temporary hard-coding
   order = 3;
   porder = 3;
   norder = 3;
   Re_tau = 182.0;   
   kin_vis = 1.0/Re_tau;
   Po = 101325.0;
   Rgas = 287.0;


   
   //-----------------------------------------------------
   // 1) Prepare the mesh
   //-----------------------------------------------------

   // 1a) Read the serial mesh (on each mpi rank)
   //Mesh *mesh = new Mesh(lomach_opts_.mesh_file.c_str(), 1, 1);

   
   // temporary hard-coded mesh

   // domain size
   double Lx = 2.0 * M_PI;
   double Ly = 1.0;
   double Lz = M_PI;

   // mesh size
   int N = order + 1;
   int NL = static_cast<int>(std::round(64.0 / N)); // Coarse
   double LC = M_PI / NL;
   int NX = 2 * NL;
   int NY = 2 * static_cast<int>(std::round(48.0 / N));
   int NZ = NL;

   Mesh mesh = Mesh::MakeCartesian3D(NX, NY, NZ, Element::HEXAHEDRON, Lx, Ly, Lz);
   Mesh *mesh_ptr = &mesh;
   //if (verbose) grvy_printf(ginfo, "Constructed mesh...\n");   
   
   for (int i = 0; i < mesh.GetNV(); ++i)
   {
      double *v = mesh.GetVertex(i);
      v[1] = mesh_stretching_func(v[1]);
   }
   //if (verbose) grvy_printf(ginfo, "Stretched mesh...\n");      

   // Create translation vectors defining the periodicity
   Vector x_translation({Lx, 0.0, 0.0});
   Vector z_translation({0.0, 0.0, Lz});
   std::vector<Vector> translations = {x_translation, z_translation};

   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(translations));
   //if (verbose) grvy_printf(ginfo, "Mesh made periodic...\n");         
   
   
   dim_ = mesh_ptr->Dimension();
   //if (verbose) grvy_printf(ginfo, "Got mesh dim...\n");            

   // 1b) Refine the serial mesh, if requested
   if (verbose && (loMach_opts_.ref_levels > 0)) {
     grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", loMach_opts_.ref_levels);
   }
   for (int l = 0; l < loMach_opts_.ref_levels; l++) {
     mesh_ptr->UniformRefinement();
   }


   /*
   if (Mpi::Root())
   {
      printf("NL=%i NX=%i NY=%i NZ=%i dx+=%f\n", NL, NX, NY, NZ, LC * Re_tau);
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }
   */

   

   // 1c) Partition the mesh
   //pmesh_ = new ParMesh(MPI_COMM_WORLD, *mesh);
   pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
   //auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
   //delete mesh_ptr;
   //delete mesh;  // no longer need the serial mesh
   // pmesh_->ReorientTetMesh();
   //if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");      


   //-----------------------------------------------------
   // 2) Prepare the required finite elements
   //-----------------------------------------------------

   int dim = pmesh->Dimension();
   
   // velocity
   vfec = new H1_FECollection(order, dim);
   vfes = new ParFiniteElementSpace(pmesh, vfec, dim);

   // dealias nonlinear term 
   nfec = new H1_FECollection(norder, dim);   
   nfes = new ParFiniteElementSpace(pmesh, vfec, dim);   

   // pressure
   pfec = new H1_FECollection(porder);
   pfes = new ParFiniteElementSpace(pmesh, pfec);

   // temperature
   tfec = new H1_FECollection(order);
   tfes = new ParFiniteElementSpace(pmesh, tfec);   

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
   //if (verbose) grvy_printf(ginfo, "Spaces constructed...\n");   
   
   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();
   int tfes_truevsize = tfes->GetTrueVSize();
   int nfes_truevsize = nfes->GetTrueVSize();
   //if (verbose) grvy_printf(ginfo, "Got sizes...\n");   
   
   un.SetSize(vfes_truevsize);
   un = 0.0;
   un_next.SetSize(vfes_truevsize);
   un_next = 0.0;
   
   unm1.SetSize(vfes_truevsize);
   unm1 = 0.0;
   unm2.SetSize(vfes_truevsize);
   unm2 = 0.0;
   
   fn.SetSize(vfes_truevsize);
   
   Nun.SetSize(nfes_truevsize);
   Nun = 0.0;
   Nunm1.SetSize(nfes_truevsize);
   Nunm1 = 0.0;
   Nunm2.SetSize(nfes_truevsize);
   Nunm2 = 0.0;

   uBn.SetSize(nfes_truevsize);
   uBn = 0.0;   
   uBnm1.SetSize(nfes_truevsize);
   uBnm1 = 0.0;
   uBnm2.SetSize(nfes_truevsize);
   uBnm2 = 0.0;

   FBext.SetSize(vfes_truevsize);
   
   Fext.SetSize(vfes_truevsize);
   FText.SetSize(vfes_truevsize);
   Lext.SetSize(vfes_truevsize);
   resu.SetSize(vfes_truevsize);

   tmpR1.SetSize(vfes_truevsize);

   pn.SetSize(pfes_truevsize);
   pn = 0.0;
   resp.SetSize(pfes_truevsize);
   resp = 0.0;
   FText_bdr.SetSize(pfes_truevsize);
   g_bdr.SetSize(pfes_truevsize);

   pnBig.SetSize(tfes_truevsize);
   pnBig = 0.0;   
   
   un_gf.SetSpace(vfes);
   un_gf = 0.0;
   un_next_gf.SetSpace(vfes);
   un_next_gf = 0.0;

   sml_gf.SetSpace(pfes);      
   big_gf.SetSpace(tfes);   

   Lext_gf.SetSpace(vfes);
   curlu_gf.SetSpace(vfes);
   curlcurlu_gf.SetSpace(vfes);
   FText_gf.SetSpace(vfes);
   resu_gf.SetSpace(vfes);

   pn_gf.SetSpace(pfes);
   pn_gf = 0.0;
   resp_gf.SetSpace(pfes);
   tmpR0PM1.SetSize(pfes_truevsize);   
   
   cur_step = 0;

   // adding temperature
   Tn.SetSize(tfes_truevsize);
   Tn = 0.0;
   Tn_next.SetSize(tfes_truevsize);
   Tn_next = 0.0;

   Tnm1.SetSize(tfes_truevsize);
   Tnm1 = 298.0; // fix hardcode
   Tnm2.SetSize(tfes_truevsize);
   Tnm2 = 298.0;
   
   fTn.SetSize(tfes_truevsize); // forcing term
   NTn.SetSize(tfes_truevsize); // advection terms
   NTn = 0.0; 
   NTnm1.SetSize(tfes_truevsize);
   NTnm1 = 0.0;
   NTnm2.SetSize(tfes_truevsize);
   NTnm2 = 0.0;

   Text.SetSize(tfes_truevsize);   
   Text_bdr.SetSize(tfes_truevsize);   
   Text_gf.SetSpace(tfes);
   t_bdr.SetSize(tfes_truevsize);   
   
   resT.SetSize(tfes_truevsize);
   tmpR0.SetSize(tfes_truevsize);
   
   Tn_gf.SetSpace(tfes); // bc?
   Tn_gf = 298.0; // fix hardcode
   Tn_next_gf.SetSpace(tfes);
   Tn_next_gf = 298.0;

   resT_gf.SetSpace(tfes);

   R0PM0_gf.SetSpace(tfes);
   R0PM0_gf = 0.0;
   R0PM1_gf.SetSpace(pfes);
   R0PM1_gf = 0.0;
   
   R1PM0_gf.SetSpace(vfes);
   R1PM0_gf = 0.0;   
   R1PX2_gf.SetSpace(nfes);
   R1PX2_gf = 0.0;
   //if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");      
   
   //PrintInfo();
}


void LoMachSolver::Setup(double dt)
{
  
   //partial_assembly = false; // HACK
   //if (verbose) grvy_printf(ginfo, "in Setup...\n");
   /*
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
      if (partial_assembly) { mfem::out << "Using Partial Assembly" << std::endl; }
      else { mfem::out << "Using Full Assembly" << std::endl; }
   }
   */

   int dim = pmesh->Dimension();
   
   // adding bits from old main here
   // Set the initial condition.
   ParGridFunction *u_gf = GetCurrentVelocity();
   //std::cout << "Check 0..." << std::endl;   
   ParGridFunction *p_gf = GetCurrentPressure();
   ParGridFunction *t_gf = GetCurrentTemperature();   
   //std::cout << "Check 1..." << std::endl;
   
   VectorFunctionCoefficient u_ic_coef(dim, vel_ic);
   //std::cout << "Check 1a..." << std::endl;      
   u_gf->ProjectCoefficient(u_ic_coef);
   //std::cout << "Check 1b..." << std::endl;   

   FunctionCoefficient t_ic_coef(temp_ic);   
   t_gf->ProjectCoefficient(t_ic_coef);   
   //std::cout << "Check 2..." << std::endl;
   
   Array<int> domain_attr(pmesh->attributes);
   domain_attr = 1;
   AddAccelTerm(accel, domain_attr);

   Array<int> attr(pmesh->bdr_attributes.Max());
   attr[1] = 1;
   attr[3] = 1;
   AddVelDirichletBC(vel_wall, attr);

   Array<int> Tattr(pmesh->bdr_attributes.Max());
   Tattr[1] = 1;
   Tattr[3] = 1;
   AddTempDirichletBC(temp_wall, Tattr);   
   //std::cout << "Check 3..." << std::endl;  

   // returning to regular setup   
   sw_setup.Start();

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);
   tfes->GetEssentialTrueDofs(temp_ess_attr, temp_ess_tdof);
   nfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);    //  this may break?
   //std::cout << "Check 4..." << std::endl;     

   int Vdof = vfes->GetTrueVSize();   
   int Pdof = pfes->GetTrueVSize();
   int Tdof = tfes->GetTrueVSize();
   int Ndof = nfes->GetTrueVSize();   

   //bufferPM0 = new ParGridFunction(tfes);
   //bufferPM1 = new ParGridFunction(pfes);      
   
   Array<int> empty;

   // unsteady: p+p = 2p
   // convection: p+p+(p-1) = 3p-1
   // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
   
   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order);
   const IntegrationRule &ir_nli = gll_rules.Get(nfes->GetFE(0)->GetGeomType(), 3 * norder - 1);
   const IntegrationRule &ir_pi = gll_rules.Get(pfes->GetFE(0)->GetGeomType(), 2 * porder);   
   const IntegrationRule &ir_i  = gll_rules.Get(tfes->GetFE(0)->GetGeomType(), 2 * order);
   //std::cout << "Check 5..." << std::endl;     
   
   // convection section, extrapolation
   nlcoeff.constant = -1.0;
   //N = new ParNonlinearForm(vfes);
   N = new ParNonlinearForm(nfes);
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
   //std::cout << "Check 6..." << std::endl;     

   // mass matrix
   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   if (numerical_integ)
   {
      mv_blfi->SetIntRule(&ir_ni);
   }
   Mv_form->AddDomainIntegrator(mv_blfi);
   if (partial_assembly)
   {
      Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);
   //std::cout << "Check 7..." << std::endl;     

   // need to build another with q=1/rho
   // DiffusionIntegrator(MatrixCoefficient &q, const IntegrationRule *ir = nullptr)
   // BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma); with sigma some type of Coefficient class
   //ParGridFunction buffer1(pfes);
   bufferInvRho = new ParGridFunction(pfes);
   {
   double *data = bufferInvRho->HostReadWrite();
   double *Tdata = Tn.HostReadWrite();
   double To = 298.0;
   double rho_o = Po / (Rgas * To); // P = rho*RT
   //for (int eq = 0; eq < pmesh->Dimension(); eq++) {
     for (int i = 0; i < Pdof; i++) {     
       // get rho here
       //data1[i] = (Rgas * Tdata[i]) / Po * rho_o;
       data[i] = 1.0;
     }
   }
   //GridFunctionCoefficient invRho(&buffer1);    
   invRho = new GridFunctionCoefficient(bufferInvRho);
   //std::cout << "Check 8..." << std::endl;     
   
   // looks like this is the Laplacian for press eq
   Sp_form = new ParBilinearForm(pfes);
   auto *sp_blfi = new DiffusionIntegrator;
   //auto *sp_blfi = new DiffusionIntegrator(*invRho);
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
   //std::cout << "Check 9..." << std::endl;  
   
   // div(u)
   D_form = new ParMixedBilinearForm(vfes, tfes);
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
   //std::cout << "Check 10..." << std::endl;  
   
   // for grad(div(u))?
   G_form = new ParMixedBilinearForm(tfes, vfes);
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
   //std::cout << "Check 11..." << std::endl;  
   
   // viscosity field
   //ParGridFunction buffer2(vfes); // or pfes here?
   bufferVisc = new ParGridFunction(vfes);
   {
   double *data = bufferVisc->HostReadWrite();   
   for (int i = 0; i < Vdof; i++) {
     // double rho = 1.0;
     // get nu here
     //transport->ComputeViscosities(rho, double* kin_vis, double* bulk_visc, double* therm_cond)
     data[i] = kin_vis;
   }
   }
   //GridFunctionCoefficient viscField(&buffer2);
   viscField = new GridFunctionCoefficient(bufferVisc);
   //std::cout << "Check 12..." << std::endl;     
   
   // why is this even needed, kin_vis is applied manually in "Step"
   // this is used in the last velocity implicit solve, Step kin_vis is
   // only for the pressure-poisson
   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);
   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff); // diagonal from unsteady term
   //auto *hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);
   auto *hdv_blfi = new VectorDiffusionIntegrator(*viscField); // Laplacian
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
   //std::cout << "Check 13..." << std::endl;     
   
   // boundary terms   
   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(pfes);
   auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
   if (numerical_integ) { ftext_bnlfi->SetIntRule(&ir_ni); }
   FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);
   // std::cout << "Check 14..." << std::endl;     

   g_bdr_form = new ParLinearForm(vfes); //? was pfes 
   for (auto &vel_dbc : vel_dbcs)
   {
      auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
      if (numerical_integ) { gbdr_bnlfi->SetIntRule(&ir_ni); }
      g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
   }
   //std::cout << "Check 15..." << std::endl;     
   
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
   //std::cout << "Check 16..." << std::endl;     
   
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
   //std::cout << "Check 17..." << std::endl;     

   
   /**/
   if (partial_assembly)
   {
      lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
      SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      SpInvPC->SetPrintLevel(pl_amg);
      SpInvPC->Mult(resp, pn);
      SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   else
   {
      SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(0);
      SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   SpInv = new CGSolver(vfes->GetComm());
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
   SpInv->SetRelTol(rtol_spsolve);
   SpInv->SetMaxIter(500);
   /**/
   //std::cout << "Check 18..." << std::endl;     

   // this wont be as efficient but AMG merhod (above) wasnt allowing for updates to variable coeff
   /*
   if (partial_assembly)
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
   SpInv->SetMaxIter(500);
   */

   
   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      H_form->AssembleDiagonal(diag_pa);
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HInv = new CGSolver(vfes->GetComm());
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(500);
   //std::cout << "Check 19..." << std::endl;     

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   un_gf.GetTrueDofs(un);
   un_next = un;
   un_next_gf.SetFromTrueDofs(un_next);

   
   // temperature.....................................

   Mt_form = new ParBilinearForm(tfes);
   auto *mt_blfi = new MassIntegrator;
   if (numerical_integ) { mt_blfi->SetIntRule(&ir_i); }
   Mt_form->AddDomainIntegrator(mt_blfi);
   if (partial_assembly) { Mt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Mt_form->Assemble();
   Mt_form->FormSystemMatrix(empty, Mt);
   //std::cout << "Check 20..." << std::endl;        
   
   // div(uT) for temperature
   Dt_form = new ParMixedBilinearForm(vfes, tfes);
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
   //std::cout << "Check 21..." << std::endl;        
   
   // thermal diffusivity field
   //ParGridFunction buffer3(tfes);
   double Pr = 1.2;
   bufferAlpha = new ParGridFunction(tfes);
   {
   double *data = bufferAlpha->HostReadWrite();
   for (int i = 0; i < Tdof; i++) {
     // get alpha here
     //transport->ComputeViscosities(rho, double* kin_vis, double* bulk_visc, double* therm_cond)
     data[i] = kin_vis / Pr;
   }
   }
   //GridFunctionCoefficient alphaField(&buffer3);
   alphaField = new GridFunctionCoefficient(bufferAlpha);
   //std::cout << "Check 22..." << std::endl;        
 
   Ht_lincoeff.constant = kin_vis / Pr; 
   Ht_bdfcoeff.constant = 1.0 / dt;
   Ht_form = new ParBilinearForm(tfes);
   auto *hmt_blfi = new MassIntegrator(Ht_bdfcoeff); // unsteady bit
   //auto *hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);
   auto *hdt_blfi = new DiffusionIntegrator(*alphaField); // Laplacian bit
   //std::cout << "Check 23..." << std::endl;        
   
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
   //std::cout << "Check 24..." << std::endl;        
   
   // boundary terms   
   Text_gfcoeff = new GridFunctionCoefficient(&Text_gf);
   Text_bdr_form = new ParLinearForm(tfes);
   auto *text_blfi = new BoundaryLFIntegrator(*Text_gfcoeff);
   if (numerical_integ) { text_blfi->SetIntRule(&ir_i); }
   Text_bdr_form->AddBoundaryIntegrator(text_blfi, temp_ess_attr);
   //std::cout << "Check 25..." << std::endl;        

   t_bdr_form = new ParLinearForm(tfes);
   for (auto &temp_dbc : temp_dbcs)
   {
      auto *tbdr_blfi = new BoundaryLFIntegrator(*temp_dbc.coeff);
      if (numerical_integ) { tbdr_blfi->SetIntRule(&ir_i); }
      t_bdr_form->AddBoundaryIntegrator(tbdr_blfi, temp_dbc.attr);
   }

   if (partial_assembly)
   {
      Vector diag_pa(tfes->GetTrueVSize());
      Mt_form->AssembleDiagonal(diag_pa);
      MtInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MtInvPC = new HypreSmoother(*Mt.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MtInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MtInv = new CGSolver(tfes->GetComm());
   MtInv->iterative_mode = false;
   MtInv->SetOperator(*Mt);
   MtInv->SetPreconditioner(*MtInvPC);
   MtInv->SetPrintLevel(pl_mtsolve);
   MtInv->SetRelTol(1e-12);
   MtInv->SetMaxIter(500);
   //std::cout << "Check 26..." << std::endl;        
   
   if (partial_assembly)
   {
      Vector diag_pa(tfes->GetTrueVSize());
      Ht_form->AssembleDiagonal(diag_pa);
      HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
   }
   else
   {
      HtInvPC = new HypreSmoother(*Ht.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(HtInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   HtInv = new CGSolver(tfes->GetComm());
   HtInv->iterative_mode = true;
   HtInv->SetOperator(*Ht);
   HtInv->SetPreconditioner(*HtInvPC);
   HtInv->SetPrintLevel(pl_hsolve);
   HtInv->SetRelTol(rtol_hsolve);
   HtInv->SetMaxIter(500);
   //std::cout << "Check 27..." << std::endl;        

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   Tn_gf.GetTrueDofs(Tn);
   Tn_next = Tn;
   Tn_next_gf.SetFromTrueDofs(Tn_next);
   //std::cout << "Check 28..." << std::endl;     

   
   // Set initial time step in the history array
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

      Tn_filtered_gf.SetSpace(tfes);
      Tn_filtered_gf = 0.0;
   }
   
   sw_setup.Stop();
   //std::cout << "Check 29..." << std::endl;     
   
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


// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {

  //if (verbose) grvy_printf(ginfo, "parsing solver options...\n");        
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
  tpsP_->getInput("loMach/order", loMach_opts_.order, 1);
  tpsP_->getInput("loMach/ref_levels", loMach_opts_.ref_levels, 0);
  tpsP_->getInput("loMach/max_iter", loMach_opts_.max_iter, 100);
  tpsP_->getInput("loMach/rtol", loMach_opts_.rtol, 1.0e-6);
  
  //tpsP_->getInput("em/atol", em_opts_.atol, 1.0e-10);
  //tpsP_->getInput("em/preconditioner_background_sigma", em_opts_.preconditioner_background_sigma, -1.0);
  //tpsP_->getInput("em/evaluate_magnetic_field", em_opts_.evaluate_magnetic_field, true);
  //tpsP_->getInput("em/nBy", em_opts_.nBy, 0);
  //tpsP_->getInput("em/yinterp_min", em_opts_.yinterp_min, 0.0);
  //tpsP_->getInput("em/yinterp_max", em_opts_.yinterp_max, 1.0);
  //tpsP_->getInput("em/By_file", em_opts_.By_file, std::string("By.h5"));
  //tpsP_->getInput("em/top_only", em_opts_.top_only, false);
  //tpsP_->getInput("em/bot_only", em_opts_.bot_only, false);

  //tpsP_->getInput("em/current_amplitude", em_opts_.current_amplitude, 1.0);
  //tpsP_->getInput("em/current_frequency", em_opts_.current_frequency, 1.0);
  //tpsP_->getInput("em/permeability", em_opts_.mu0, 1.0);

  //Vector default_axis(3);
  //default_axis[0] = 0;
  //default_axis[1] = 1;
  //default_axis[2] = 0;
  //tpsP_->getVec("em/current_axis", em_opts_.current_axis, 3, default_axis);

  // dump options to screen for user inspection
  if (mpi_.Root()) {
    loMach_opts_.print(std::cout);
  }
}


void LoMachSolver::solve()
{    
   //Mpi::Init();
   //Hypre::Init();
  
   //if (verbose) grvy_printf(ginfo, "in solve...\n");      

   /*
   // Create translation vectors defining the periodicity
   Vector x_translation({Lx, 0.0, 0.0});
   Vector z_translation({0.0, 0.0, Lz});
   std::vector<Vector> translations = {x_translation, z_translation};

   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(translations));
   */

   //std::cout << "Check 2..." << std::endl;

  /*
   if (Mpi::Root())
   {
      printf("NL=%d NX=%d NY=%d NZ=%d dx+=%f\n", NL, NX, NY, NZ, LC * Re_tau);
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }
  */

   //std::cout << "Check 6..." << std::endl;

   // temporary hard-codes
   double t = 0.0;
   double dt = 1.0e-4;
   double t_final = 1.0;
   bool last_step = false;

   Setup(dt);

   /*
   ParaViewDataCollection pvdc("turbchan", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("temperature", t_gf);   
   pvdc.Save();
   */

   //std::cout << "Check 7..." << std::endl;
   
   for (int step = 0; !last_step; ++step)
   {

     /*
      if (step % 100 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }
     */
      
      if (t + dt >= t_final - dt / 2) { last_step = true; }

      if (Mpi::Root())
      {
	printf("%1s\n", " ");	
	printf("%2s %11s %11s\n", "N", "Time", "dt");
	printf("%o %4s %.5E %.5E\n", step, "    ", t, dt);
        fflush(stdout);
      }
      
      Step(t, dt, step);
      
      if (t > 5.0) { dt = 1e-2; }

   }

   //flowsolver.PrintTimingData();

   //return 0;
}



// all the variable coefficient operators musc be re-build every step...
void LoMachSolver::Step(double &time, double dt, int current_step,
                        bool provisional)
{
   sw_step.Start();

   int Vdof = vfes->GetTrueVSize();   
   int Pdof = pfes->GetTrueVSize();
   int Tdof = tfes->GetTrueVSize();
   int Ndof = nfes->GetTrueVSize();      
   
   //std::cout << "Check a..." << std::endl;
   SetTimeIntegrationCoefficients(current_step);

   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs) {vel_dbc.coeff->SetTime(time + dt);}

   // Set current time for pressure Dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs) {pres_dbc.coeff->SetTime(time + dt);}
   //std::cout << "Check b..." << std::endl;

   double Pr = 1.2;
   double *dataVisc = bufferVisc->HostReadWrite();
   for (int i = 0; i < Vdof; i++) { // should this be dof of Vdof size?
     dataVisc[i] = kin_vis;
   }
   //std::cout << "Check c..." << std::endl;   
   
   H_bdfcoeff.constant = bd0 / dt;
   H_form->Update();
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);
   //std::cout << "Check d..." << std::endl;   
   
   HInv->SetOperator(*H);
   if (partial_assembly)
   {
      delete HInvPC;
      Vector diag_pa(vfes->GetTrueVSize());
      H_form->AssembleDiagonal(diag_pa);
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
      HInv->SetPreconditioner(*HInvPC);
   }
   //std::cout << "Check e..." << std::endl;       
   
   // Extrapolated f^{n+1} (source term: f(n+1))
   for (auto &accel_term : accel_terms)
   {
      accel_term.coeff->SetTime(time + dt);
   }

   f_form->Assemble();
   f_form->ParallelAssemble(fn);
   //std::cout << "Check f..." << std::endl;      
   
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
   for (int i = 0; i < Ndof; i++) {     
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
   for (int i = 0; i < Ndof; i++) {     
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
   for (int i = 0; i < Ndof; i++) {     
     Udata[i] = data[i];
   }
   }
   //std::cout << "Check g..." << std::endl;         
   
   N->Mult(uBn, Nun);
   N->Mult(uBnm1, Nunm1);
   N->Mult(uBnm2, Nunm2);
   //std::cout << "Check h..." << std::endl;         
   
   // ab-predictor of nonliner term at {n+1}
   {
      const auto d_Nun = Nun.Read();
      const auto d_Nunm1 = Nunm1.Read();
      const auto d_Nunm2 = Nunm2.Read();
      auto d_Fext = FBext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      mfem::forall(FBext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Fext[i] = ab1_ * d_Nun[i] +
                     ab2_ * d_Nunm1[i] +
                     ab3_ * d_Nunm2[i];
      });
   }
   //std::cout << "Check i..." << std::endl;         

   // project NL product back to v-space
   {
   double *data = R1PX2_gf.HostReadWrite();
   double *Fdata = FBext.HostReadWrite();   
   for (int i = 0; i < Ndof; i++) {     
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
   //std::cout << "Check j..." << std::endl;            
      
   // add forcing/accel term to Fext   
   Fext.Add(1.0, fn); 

   // Fext = M^{-1} (F(u^{n}) + f^{n+1}) (F* w/o known part of BDF)
   MvInv->Mult(Fext, tmpR1);
   iter_mvsolve = MvInv->GetNumIterations();
   res_mvsolve = MvInv->GetFinalNorm();
   Fext.Set(1.0, tmpR1);
   //std::cout << "Check k..." << std::endl;   

   // for unsteady term, compute BDF terms (known part of BDF u for full F*)
   {
      const double bd1idt = -bd1 / dt;
      const double bd2idt = -bd2 / dt;
      const double bd3idt = -bd3 / dt;
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Fext = Fext.ReadWrite();
      mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Fext[i] += bd1idt * d_un[i] +
                      bd2idt * d_unm1[i] +
                      bd3idt * d_unm2[i];
      });
   }
   sw_extrap.Stop();
   //std::cout << "Check l..." << std::endl;   

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
      mfem::forall(Lext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Lext[i] = ab1_ * d_un[i] +
                     ab2_ * d_unm1[i] +
                     ab3_ * d_unm2[i];
      });
   }
   //std::cout << "Check m..." << std::endl;   

   Lext_gf.SetFromTrueDofs(Lext);
   int dim = pmesh->Dimension();
   if (dim == 2)
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
   //std::cout << "Check n..." << std::endl;   

   // (!) with a field viscosity, this likely breaks
   // (!) must change to dynamic/rho
   //Lext *= kin_vis;

   for (int eq = 0; eq < dim; eq++) {
     for (int i = 0; i < Tdof; i++) {
       //double rho = 1.0;
       // get nu here
       //transport->ComputeViscosities(rho, double* kin_vis, double* bulk_visc, double* therm_cond)
       //Lext[i + eq * dof] = Lext[i + eq * dof] * kin_vis * 1.0;
       Lext[i + eq * Tdof] = Lext[i + eq * Tdof] * dataVisc[i]; // data2 points to kin_vis
     }
   }
   //std::cout << "Check o..." << std::endl;   
   
   sw_curlcurl.Stop();

   // \tilde{F} = F - \nu CurlCurl(u), (F* + L*)
   FText.Set(-1.0, Lext);
   FText.Add(1.0, Fext);

   // p_r = \nabla \cdot FText ( i think "p_r" means rhs of pressure-poisson eq, so div(\tilde{F}))
   // applied divergence to full FText vector
   //D->Mult(FText, resp);
   //resp.Neg();   

   // add some if for porder != vorder
   // project FText to p-1
   D->Mult(FText, tmpR0);  // tmp3 c p, tmp2 and resp c (p-1)
   tmpR0.Neg();
   //std::cout << "Check p..." << std::endl;   

   // Add boundary terms.
   FText_gf.SetFromTrueDofs(FText);
   FText_bdr_form->Assemble();
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
   
   //resp.Add(1.0, FText_bdr);
   //resp.Add(-bd0 / dt, g_bdr);
   //std::cout << "Check q..." << std::endl;   

   if (pres_dbcs.empty()) { Orthogonalize(resp); }   
   for (auto &pres_dbc : pres_dbcs) { pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr); }

   // ?
   pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);
   //std::cout << "Check r..." << std::endl;   
   
   // update variable coeff pressure laplacian
   /*
   {
   double *data = R0PM0_gf.HostReadWrite();
   double *Tdata = Tn.HostReadWrite();      
   for (int i = 0; i < Tdof; i++) {     
     data[i] = Tdata[i];
   }
   }
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);   
   {
   double To = 298.0;
   //double invRho_o = (Rgas * To) / Po; // P = rho*RT
   double *data = bufferInvRho->HostReadWrite(); // invRho
   double *Tdata = R0PM1_gf.HostReadWrite();   
   for (int i = 0; i < Pdof; i++) {     
     data[i] = 1.0; // Tdata[i] / To;
   }
   }

   Sp_form->Update();
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);
   
   SpInv->SetOperator(*Sp);
   if (partial_assembly)
   {
      delete SpInvPC;
      Vector diag_pa(pfes->GetTrueVSize());
      Sp_form->AssembleDiagonal(diag_pa);
      SpInvPC = new OperatorJacobiSmoother(diag_pa, pres_ess_tdof);
      SpInv->SetPreconditioner(*SpInvPC);
   }
   */

   /* // broken for variable coeff
   Sp_form->Update();
   Sp_form->Assemble();
   Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);

   SpInv->SetOperator(*Sp);
   if (partial_assembly)
   {
      delete lor;   	
      delete SpInvPC;
      lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
      SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      if (pres_dbcs.empty()) {
        delete SpInvOrthoPC;           	
        SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
        SpInvOrthoPC->SetSolver(*SpInvPC);      	
	SpInv->SetPreconditioner(*SpInvOrthoPC);
      }
      else {
	SpInv->SetPreconditioner(*SpInvPC);
      }
   }
   */   
   
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
   SpInv->Mult(B1, X1);
   sw_spsolve.Stop();
   iter_spsolve = SpInv->GetNumIterations();
   res_spsolve = SpInv->GetFinalNorm();
   Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);
   //std::cout << "Check s..." << std::endl;   

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
   //std::cout << "Check t..." << std::endl;
   
   // un_next_gf = un_gf;

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
   HInv->Mult(B2, X2);
   sw_hsolve.Stop();
   iter_hsolve = HInv->GetNumIterations();
   res_hsolve = HInv->GetFinalNorm();
   H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);
   //std::cout << "Check 7a15..." << std::endl;   

   un_next_gf.GetTrueDofs(un_next);


   // begin temperature......................................
   //std::cout << "Check u..." << std::endl;

   // Set current time for temperature Dirichlet boundary conditions.
   for (auto &temp_dbc : temp_dbcs) {temp_dbc.coeff->SetTime(time + dt);}   
   //std::cout << "Check v..." << std::endl;
   
   // helmholtz (alphaField)
   //double Pr = 1.2;
   {
   double *data = bufferAlpha->HostReadWrite();
   for (int i = 0; i < Tdof; i++) {
     data[i] = kin_vis / Pr;
   }
   }
   
   //Ht_lincoeff.constant = 1000.0 * kin_vis / Pr;   
   Ht_bdfcoeff.constant = bd0 / dt;
   // update visc here
   Ht_form->Update();
   Ht_form->Assemble();
   Ht_form->FormSystemMatrix(temp_ess_tdof, Ht);
   //std::cout << "Check w..." << std::endl;

   HtInv->SetOperator(*Ht);
   if (partial_assembly)
   {
      delete HtInvPC;
      Vector diag_pa(tfes->GetTrueVSize());
      Ht_form->AssembleDiagonal(diag_pa);
      HtInvPC = new OperatorJacobiSmoother(diag_pa, temp_ess_tdof);
      HtInv->SetPreconditioner(*HtInvPC);
   }
   //std::cout << "Check x..." << std::endl;   
   
   // HACK
   //Text.Set(0.0, tmp2);   
   
   //std::cout << "Check 7g..." << std::endl;
   //std::cout << "bds: " << bd0 << " " << bd1 << " " << bd2 << " " << bd3 << std::endl;   
   
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
      mfem::forall(Text.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
        //d_Text[i] = -bd0/dt * 298.0; //d_tn[i];	
        //d_Text[i] = 0.0;
        d_Text[i] = bd1idt * d_tn[i] +	
                    bd2idt * d_tnm1[i] +
                    bd3idt * d_tnm2[i];
      });
   }
   //std::cout << "Check y..." << std::endl;
   
   // Add boundary terms.
   Text_gf.SetFromTrueDofs(Text);
   //std::cout << "Check y0..." << std::endl;      
   Text_bdr_form->Assemble();
   //std::cout << "Check y1..." << std::endl;      
   Text_bdr_form->ParallelAssemble(Text_bdr);
   //std::cout << "Check y2..." << std::endl;      
   t_bdr_form->Assemble();
   //std::cout << "Check y3..." << std::endl;      
   t_bdr_form->ParallelAssemble(t_bdr);
   //std::cout << "Check y4..." << std::endl;   

   Mt->Mult(Text, tmpR0);
   //std::cout << "Check y5..." << std::endl;   
   //resT.Add(1.0, tmp2); // part of unsteady and full extrapolated advection on lhs
   resT.Set(1.0, tmpR0);      
   //resT.Neg(); // move to rhs
   //std::cout << "Check y6..." << std::endl;   


   // advection => will break is t&u arent in the same space
   bufferTemp = new ParGridFunction(vfes);      
   double *dataTemp = bufferTemp->HostReadWrite();
   double *Udnm0 = un.HostReadWrite();
   double *Tdnm0 = Tn.HostReadWrite();
   double *Udnm1 = unm1.HostReadWrite();
   double *Tdnm1 = Tnm1.HostReadWrite();
   double *Udnm2 = unm2.HostReadWrite();
   double *Tdnm2 = Tnm2.HostReadWrite();
   //std::cout << "Check y4..." << std::endl;   
   const auto ab1_ = ab1;
   const auto ab2_ = ab2;
   const auto ab3_ = ab3;   
   for (int eq = 0; eq < dim; eq++) {   
     for (int i = 0; i < Tdof; i++) {
       dataTemp[i + eq * Tdof] = ab1_ * Udnm0[i + eq * Tdof] * Tdnm0[i] +
        	                 ab2_ * Udnm1[i + eq * Tdof] * Tdnm1[i] +
	                         ab3_ * Udnm2[i + eq * Tdof] * Tdnm2[i];
       }
   }
   //std::cout << "Check z..." << std::endl;      
   
   Dt->Mult(*bufferTemp, tmpR0); // explicit div(uT) at {n}
   //Text.Add(-1.0, tmp2);
   resT.Add(-1.0, tmpR0);


   
   // M^-1 * advection -> not sure about this part...
   /*
   MtInv->Mult(Text, tmp2);
   iter_mtsolve = MtInv->GetNumIterations();
   res_mtsolve = MtInv->GetFinalNorm();
   Text.Set(1.0, tmp2);
   */
   
   //std::cout << "Check 7i..." << std::endl;   

   // what is this?   
   for (auto &temp_dbc : temp_dbcs) { Tn_next_gf.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr); }
   //std::cout << "Check 7j..." << std::endl;   

   // (?)
   tfes->GetRestrictionMatrix()->MultTranspose(resT, resT_gf);

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
   //std::cout << "Check 7k..." << std::endl;   

   // solve helmholtz eq for temp   
   HtInv->Mult(Bt2, Xt2);
   iter_htsolve = HtInv->GetNumIterations();
   res_htsolve = HtInv->GetFinalNorm();
   Ht_form->RecoverFEMSolution(Xt2, resT_gf, Tn_next_gf);
   //Ht_form->RecoverFEMSolution(Xt2, resT, Tn_next_gf);
   Tn_next_gf.GetTrueDofs(Tn_next);
   //std::cout << "Check 7l..." << std::endl;   

   // end temperature....................................
   
   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   // explicit filter
   if (filter_alpha != 0.0)
   {
     
      un_NM1_gf.ProjectGridFunction(un_gf);
      un_filtered_gf.ProjectGridFunction(un_NM1_gf);
      const auto d_un_filtered_gf = un_filtered_gf.Read();
      auto d_un_gf = un_gf.ReadWrite();
      const auto filter_alpha_ = filter_alpha;
      mfem::forall(un_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_un_gf[i] = (1.0 - filter_alpha_) * d_un_gf[i]
                      + filter_alpha_ * d_un_filtered_gf[i];
      });

      Tn_NM1_gf.ProjectGridFunction(Tn_gf);
      Tn_filtered_gf.ProjectGridFunction(Tn_NM1_gf);
      const auto d_Tn_filtered_gf = Tn_filtered_gf.Read();
      auto d_Tn_gf = Tn_gf.ReadWrite();
      mfem::forall(Tn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Tn_gf[i] = (1.0 - filter_alpha_) * d_Tn_gf[i]
                      + filter_alpha_ * d_Tn_filtered_gf[i];
      });
      
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
                << std::scientific << res_spsolve << "   " << rtol_spsolve
                << "\n";
      mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed
                << iter_hsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_hsolve << "   " << rtol_hsolve
                << "\n";
      mfem::out << std::setw(5) << "TEMP " << std::setw(5) << std::fixed
                << iter_htsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_htsolve << "   " << rtol_htsolve
                << "\n";      
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
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

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pfes->GetComm());

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
      int dim = el->GetDim();
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
      int dim = el->GetDim();
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


void LoMachSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
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
      for (int i = 0; i < attr.Size(); ++i)
      {
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
   int fes_size1 = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
     //      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
     mfem::out  << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}


/*
LoMachSolver::~LoMachSolver()
{
   delete FText_gfcoeff;
   delete g_bdr_form;
   delete FText_bdr_form;
   delete mass_lf;
   delete Mv_form;
   delete N;
   delete Sp_form;
   delete D_form;
   delete G_form;
   delete HInvPC;
   delete HInv;
   delete H_form;
   delete SpInv;
   delete MvInvPC;
   
   delete SpInvOrthoPC;
   delete SpInvPC;
   
   delete lor;
   delete f_form;
   delete MvInv;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete vfec_filter;
   delete vfes_filter;
   delete tfec_filter;
   delete tfes_filter;
}
*/


// temporary, remove when hooked up
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
   {
      dst->AddDomainIntegrator((*bffis)[i]);
   }
}


void accel(const Vector &x, double t, Vector &f)
{
   f(0) = 1.0;
   f(1) = 0.0;
   f(2) = 0.0;
}

void vel_ic(const Vector &coords, double t, Vector &u)
{
  
   double yp;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   double A = -1.0;
   double B = -1.0;
   double C = +2.0;
   double aL = 2.0 / (2.0);
   double bL = 2.0;
   double cL = 2.0;   
   double M = 4.0;

   
   u(0) = 22.0; // - std::pow(y/0.5,4);
   u(1) = 0.0;
   u(2) = 0.0;

   u(0) += M * A * cos(aL*x) * sin(bL*y) * sin(cL*z);
   u(1) += M * B * sin(aL*x) * cos(bL*y) * sin(cL*z);
   u(2) += M * C * sin(aL*x) * sin(bL*y) * cos(cL*z);
   
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
   double temp = 270.0;
   return temp;
}
