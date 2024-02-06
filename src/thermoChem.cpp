
#include "thermoChem.hpp"
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
double temp_ic(const Vector &coords, double t);
double temp_wall(const Vector &x, double t);
double temp_wallBox(const Vector &x, double t);
double temp_inlet(const Vector &x, double t);


ThermoChem::ThermoChem(TPS::Tps *tps, loMachSolver *loMach)
  : groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
    nprocs_(groupsMPI->getTPSWorldSize()),
    rank_(groupsMPI->getTPSWorldRank()),
    rank0_(groupsMPI->isWorldRoot()),
    tpsP_(tps),
    loMach_(loMach) {
}

void ThermoChem::initialize() {
  
   bool verbose = rank0_;
   if (verbose) grvy_printf(ginfo, "Initializing ThermoChem solver.\n");

   if ( loMach_.loMach_opts_.uOrder == -1) {
     sorder = std::max(config.solOrder,1);
     double no;
     no = ceil( ((double)order * 1.5) );   
     norder = int(no);
   } else {
     sorder = loMach_opts_.uOrder;
     norder = loMach_opts_.nOrder;
   }
   
   static_rho = config.const_dens;
   kin_vis = config.const_visc;
   ambientPressure = config.amb_pres;
   thermoPressure = config.amb_pres;
   tPm1 = thermoPressure;
   tPm2 = thermoPressure;
   dtP = 0.0;

   // HARD CODE HARDCODE get for dry air
   Rgas = 287.0;
   Pr = 0.76;
   Cp = 1000.5;
   gamma = 1.4;

   mixture = NULL;

   // HARD CODE HARD CODE HARD CODE
   config.dryAirInput.specific_heat_ratio = 1.4;
   config.dryAirInput.gas_constant = 287.058;
   config.dryAirInput.f = config.workFluid;
   config.dryAirInput.eq_sys = config.eqSystem;  
   mixture = new DryAir(config, dim, nvel);  // conditional jump, must be using something in config that wasnt parsed?
   transportPtr = new DryAirTransport(mixture, config);

   
   MaxIters = loMach_.config.GetNumIters();
   num_equation = 1; // hard code for now, fix with species
   /*
   numSpecies = mixture->GetNumSpecies();
   numActiveSpecies = mixture->GetNumActiveSpecies();
   ambipolar = mixture->IsAmbipolar();
   twoTemperature_ = mixture->IsTwoTemperature();
   num_equation = mixture->GetNumEquations();
   */

   //-----------------------------------------------------
   // 2) Prepare the required finite elements
   //-----------------------------------------------------

   // scalar
   sfec = new H1_FECollection(order);
   sfes = new ParFiniteElementSpace(pmesh, sfec);   
   
   // dealias nonlinear term 
   //nfec = new H1_FECollection(norder, dim);   
   //nfes = new ParFiniteElementSpace(pmesh, nfec, dim);   
   //nfecR0 = new H1_FECollection(norder);   
   //nfesR0 = new ParFiniteElementSpace(pmesh, nfecR0);         
   
   // Check if fully periodic mesh
   if (!(pmesh->bdr_attributes.Size() == 0))
   {
      temp_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      temp_ess_attr = 0;

      Qt_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      Qt_ess_attr = 0;
   }
   if (verbose) grvy_printf(ginfo, "ThermoChem paces constructed...\n");   
   
   int sfes_truevsize = sfes->GetTrueVSize();
   //int nfes_truevsize = nfes->GetTrueVSize();
   //int nfesR0_truevsize = nfesR0->GetTrueVSize();
   
   gradT.SetSize(vfes_truevsize);
   gradT = 0.0;
   
   Qt.SetSize(sfes_truevsize);
   Qt = 0.0;   

   gradT_gf.SetSpace(vfes);
   
   Tn.SetSize(sfes_truevsize);
   Tn = 298.0; // fix hardcode, shouldnt matter
   Tn_next.SetSize(sfes_truevsize);
   Tn_next = 298.0;

   Tnm1.SetSize(sfes_truevsize);
   Tnm1 = 298.0; // fix hardcode
   Tnm2.SetSize(sfes_truevsize);
   Tnm2 = 298.0;

   // forcing term
   fTn.SetSize(sfes_truevsize); 

   // temp convection terms
   NTn.SetSize(sfes_truevsize);
   NTnm1.SetSize(sfes_truevsize);
   NTnm2.SetSize(sfes_truevsize);   
   NTn = 0.0; 
   NTnm1 = 0.0;
   NTnm2 = 0.0;

   Text.SetSize(sfes_truevsize);   
   Text_bdr.SetSize(sfes_truevsize);   
   Text_gf.SetSpace(sfes);
   t_bdr.SetSize(sfes_truevsize);

   resT_gf.SetSpace(sfes);   
   resT.SetSize(sfes_truevsize);
   resT = 0.0;
   
   Tn_next_gf.SetSpace(sfes);
   Tn_gf.SetSpace(sfes);
   Tnm1_gf.SetSpace(sfes);
   Tnm2_gf.SetSpace(sfes);
   Tn_next_gf = 298.0;
   Tn_gf = 298.0; // fix hardcode   
   Tnm1_gf = 298.0;
   Tnm2_gf = 298.0;  

   rn.SetSize(sfes_truevsize);
   rn = 1.0;
   rn_gf.SetSpace(sfes);
   rn_gf = 1.0;

   alphaSml.SetSize(sfes_truevsize);
   alphaSml = 1.0e-12;      
   alphaTotal_gf.SetSpace(sfes);
   alphaTotal_gf = 0.0;   
   
   if (verbose) grvy_printf(ginfo, "ThermpChem vectors and gf initialized...\n");     

   // setup pointers to loMach owned temp data blocks
   
}

void ThermoChem::Setup(double dt)
{

   // HARD CODE
   //partial_assembly = true;
   partial_assembly = false;
   //if (verbose) grvy_printf(ginfo, "in Setup...\n");

   Sdof = sfes->GetNDofs();
   //Ndof = nfes->GetNDofs();
   //NdofR0 = nfesR0->GetNDofs();         
   
   SdofInt = sfes->GetTrueVSize();
   //NdofInt = nfes->GetTrueVSize();
   //NdofR0Int = nfesR0->GetTrueVSize();

   // initial conditions
   //ParGridFunction *t_gf = GetCurrentTemperature();
   
   if (config.useICFunction == true) {   
     FunctionCoefficient t_ic_coef(temp_ic);
     //if (!config.restart) t_gf->ProjectCoefficient(t_ic_coef);
     if (!config.restart) Tn_gf.ProjectCoefficient(t_ic_coef);          
     if(rank0_) { std::cout << "Using initial condition function" << endl;}     
   } else if (config.useICBoxFunction == true) {
     FunctionCoefficient t_ic_coef(temp_wallBox);
     //if (!config.restart) t_gf->ProjectCoefficient(t_ic_coef);
     if (!config.restart) Tn_gf.ProjectCoefficient(t_ic_coef);     
     if(rank0_) { std::cout << "Using initial condition box function" << endl;}          
   } else {
     ConstantCoefficient t_ic_coef;
     //t_ic_coef.constant = config.initRhoRhoVp[4] / (Rgas * config.initRhoRhoVp[0]);
     t_ic_coef.constant = config.initRhoRhoVp[4];
     //if (!config.restart) t_gf->ProjectCoefficient(t_ic_coef);
     if (!config.restart) Tn_gf.ProjectCoefficient(t_ic_coef);          
     if(rank0_) { std::cout << "Initial temperature set from input file: " << config.initRhoRhoVp[4] << endl;}
   }
   Tn_gf.GetTrueDofs(Tn);
   //std::cout << "Check 2..." << std::endl;
   

   // Boundary conditions
   //ConstantCoefficient t_bc_coef;
   ConstantCoefficient Qt_bc_coef;   
   Array<int> Tattr(pmesh->bdr_attributes.Max());
   Array<int> Qattr(pmesh->bdr_attributes.Max());         
   
   // inlet bc
   Tattr = 0.0;
   for (int i = 0; i < numInlets; i++) {
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (iFace == config.inletPatchType[i].first) {
         Tattr[iFace-1] = 1;
       }
     }
   }

   for (int i = 0; i < numInlets; i++) {   
     if (config.inletPatchType[i].second == InletType::UNI_DENS_VEL) {
       if (rank0_) std::cout << "Caught uniform inlet conditions " << endl;   
       buffer_tInletInf = new ParGridFunction(sfes);             
       buffer_tInlet = new ParGridFunction(sfes);             
       tInletField = new GridFunctionCoefficient(buffer_tInlet);   
       AddTempDirichletBC(tInletField, Tattr);       
       
     } else if (config.inletPatchType[i].second == InletType::INTERPOLATE) {
       if (rank0_) std::cout << "Caught interpolated inlet conditions " << endl;          
       // interpolate from external file
       buffer_tInletInf = new ParGridFunction(sfes);             
       buffer_tInlet = new ParGridFunction(sfes);      
       interpolateInlet();
       if (rank0_) std::cout << "ThermoChem interpolation complete... " << endl;
       
       tInletField = new GridFunctionCoefficient(buffer_tInlet);   
       AddTempDirichletBC(tInletField, Tattr);
       if (rank0_) std::cout << "Interpolated temperature conditions applied " << endl;       

     }     
   }
   if (rank0_) std::cout << "ThermoChem inlet bc's completed: " << numInlets << endl;   
   /**/


   // outlet bc
   Tattr = 0.0;
   /*
   for (int i = 0; i < numOutlets; i++) {
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (iFace == config.outletPatchType[i].first) {
         Tattr[iFace-1] = 1;	 	 
       }
     }
   }
   */


   /*   
   for (int i = 0; i < numOutlets; i++) {
     
     if (config.outletPatchType[i].second == OutletType::SUB_P) {
       if (rank0_) { std::cout << "Caught pressure outlet bc" << endl; }
       
       // set directly from input
       double outlet_pres;            
       outlet_pres = config.outletPressure;       
       bufferOutlet_pbc = new ConstantCoefficient(outlet_pres);       
       AddPresDirichletBC(bufferOutlet_pbc, Pattr);
       
     }     
   }
   if (rank0_) std::cout << "Outlet bc's completed: " << numOutlets << endl;   
   */

   
   // temperature wall bc dirichlet
   Vector wallBCs;
   wallBCs.SetSize(numWalls);
   for (int i = 0; i < numWalls; i++) { wallBCs[i] = config.wallBC[i].Th; }

   // this is absolutely terrible
   //ConstantCoefficient t_bc_coef(wallBCs);
   if (numWalls > 0) { ConstantCoefficient t_bc_coef0(wallBCs[0]); }   
   if (numWalls > 1) { ConstantCoefficient t_bc_coef1(wallBCs[1]); }
   if (numWalls > 2) { ConstantCoefficient t_bc_coef2(wallBCs[2]); }
   if (numWalls > 3) { ConstantCoefficient t_bc_coef2(wallBCs[3]); }   
   
   for (int i = 0; i < numWalls; i++) {

     Tattr = 0;     
     //t_bc_coef.constant = config.wallBC[i].Th;
     
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (config.wallPatchType[i].second == WallType::VISC_ISOTH) {	   	 
         if (iFace == config.wallPatchType[i].first) {
            Tattr[iFace-1] = 1;
            if (rank0_) std::cout << "Dirichlet wall temperature BC: " << iFace << endl;

     if (config.useWallFunction == true) {
       AddTempDirichletBC(temp_wall, Tattr);
       if (rank0_) std::cout << "Using wall temp function bc for wall number: " << i+1 << endl;	               
     } else if (config.useWallBox == true) {
       AddTempDirichletBC(temp_wallBox, Tattr);
       if (rank0_) std::cout << "Using box temp bc for wall number: " << i+1 << endl;	        
     } else {
       if(i==0) {buffer_tbc = new ConstantCoefficient(t_bc_coef0); }
       if(i==1) {buffer_tbc = new ConstantCoefficient(t_bc_coef1); }
       if(i==2) {buffer_tbc = new ConstantCoefficient(t_bc_coef2); }
       if(i==3) {buffer_tbc = new ConstantCoefficient(t_bc_coef3); }       
       AddTempDirichletBC(buffer_tbc, Tattr);
       if (rank0_) std::cout << i+1 << ") wall temperature BC: " << config.wallBC[i].Th << endl;       
     }      
	    
	 }
       } else if (config.wallPatchType[i].second == WallType::VISC_ADIAB) {
         if (iFace == config.wallPatchType[i].first) {
	   Tattr[iFace-1] = 0; // no essential
           if (rank0_) std::cout << "Insulated wall temperature BC: " << iFace << endl;	   
         }	 
       }
     }
       
   }

     
   // temperature wall bc nuemann
   /*
   Tattr = 0;
   for (int i = 0; i < numWalls; i++) {
     t_bc_coef.constant = config.wallBC[i].Th;
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (config.wallPatchType[i].second == WallType::VISC_ADIAB) {
         if (iFace == config.wallPatchType[i].first) {
	   Tattr[iFace-1] = 0; // no essential
           if (rank0_) std::cout << "Insulated wall temperature BC: " << iFace << endl;	   
         }	 
       }
     }
   }
   */
   if (rank0_) std::cout << "Temp wall bc completed: " << numWalls << endl;


   // Qt wall bc dirichlet (divU = 0)
   Qattr = 0;
   for (int i = 0; i < numWalls; i++) {
     Qt_bc_coef.constant = 0.0;     
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (config.wallPatchType[i].second == WallType::VISC_ISOTH
       || config.wallPatchType[i].second == WallType::VISC_ADIAB) {	   	 
         if (iFace == config.wallPatchType[i].first) {
            Qattr[iFace-1] = 1;
            if (rank0_) std::cout << "Dirichlet wall Qt BC: " << iFace << endl;
         }
       }
     }
   }
   buffer_qbc = new ConstantCoefficient(Qt_bc_coef);
   AddQtDirichletBC(buffer_qbc, Qattr);


   sfes->GetEssentialTrueDofs(temp_ess_attr, temp_ess_tdof);
   sfes->GetEssentialTrueDofs(Qt_ess_attr, Qt_ess_tdof);   
   if (rank0_) std::cout << "ThermoChem Essential true dof step" << endl;

   
   Array<int> empty;

   // unsteady: p+p [+p] = 2p [3p]
   // convection: p+p+(p-1) [+p] = 3p-1 [4p-1]
   // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
   
   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_i  = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2*sorder + 1);  //3 5
   const IntegrationRule &ir_nli = gll_rules.Get(nfes->GetFE(0)->GetGeomType(), 4*sorder);    //4 8
   const IntegrationRule &ir_di  = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 3*sorder - 1); //2 5  
   if (rank0_) std::cout << "Integration rules set" << endl;   


   // density coefficient
   bufferRho = new ParGridFunction(sfes);
   {
     double *data = bufferRho->HostReadWrite();
     //double *Tdata = Tn.HostReadWrite();
     for (int i = 0; i < Sdof; i++) {     
       data[i] = 1.0; // gets updated anyway... ambientPressure / (Rgas * Tdata[i]);
     }
   }    
   Rho = new GridFunctionCoefficient(bufferRho);

   bufferInvRho = new ParGridFunction(pfes);
   {
     double *data = bufferInvRho->HostReadWrite();
     for (int i = 0; i < Pdof; i++) { data[i] = 1.0; }
   }
   invRho = new GridFunctionCoefficient(bufferInvRho);

   // viscosity field 
   //ParGridFunction buffer2(vfes); // or pfes here?
   //bufferVisc = new ParGridFunction(vfes);
   //bufferViscMult = new ParGridFunction(sfes);           
   bufferVisc = new ParGridFunction(sfes);
   {
     double *data = bufferVisc->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     double visc[2];
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Sdof; i++) {
       prim[1+nvel] = Tdata[i];
       transportPtr->GetViscosities(prim, prim, visc);
       data[i] = visc[0];       
     }
   }
   viscField = new GridFunctionCoefficient(bufferVisc);

   bufferBulkVisc = new ParGridFunction(sfes);
   {
     double *data = bufferBulkVisc->HostReadWrite();
     double *dataMu = bufferVisc->HostReadWrite();
     double twothird = 2.0/3.0;
     for (int i = 0; i < Sdof; i++) { data[i] = twothird * dataMu[i]; }
   }
   bulkViscField = new GridFunctionCoefficient(bufferBulkVisc);
   
   // thermal diffusivity field
   bufferAlpha = new ParGridFunction(sfes);
   {
     double *data = bufferAlpha->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     double visc[2];
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Sdof; i++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         data[i] = visc[0] / Pr;
	 //data[i] = kin_vis / Pr;
     }
   }   
   alphaField = new GridFunctionCoefficient(bufferAlpha);
   //std::cout << "Check 22..." << std::endl;        

   //thermal_diff_coeff.constant = thermal_diff;
   thermal_diff_coeff = new GridFunctionCoefficient(&alphaTotal_gf);   
   gradT_coeff = new GradientGridFunctionCoefficient(&Tn_next_gf);
   kap_gradT_coeff = new ScalarVectorProductCoefficient(*thermal_diff_coeff, *gradT_coeff);
   
   
   // Convection: Atemperature(i,j) = \int_{\Omega} \phi_i \rho u \cdot \nabla \phi_j
   un_next_coeff = new VectorGridFunctionCoefficient(&un_next_gf);
   rhon_next_coeff = new GridFunctionCoefficient(&rn_gf);
   rhou_coeff = new ScalarVectorProductCoefficient(*rhon_next_coeff, *un_next_coeff);   
   At_form = new ParBilinearForm(sfes);
   auto *at_blfi = new ConvectionIntegrator(*rhou_coeff);
   if (numerical_integ) {
     at_blfi->SetIntRule(&ir_nli);
   }
   At_form->AddDomainIntegrator(at_blfi);
   if (partial_assembly) {
     At_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   At_form->Assemble();
   At_form->FormSystemMatrix(empty, At);
      
   // mass matrix
   Ms_form = new ParBilinearForm(sfes);
   auto *ms_blfi = new MassIntegrator;
   if (numerical_integ) { ms_blfi->SetIntRule(&ir_i); }
   Ms_form->AddDomainIntegrator(ms_blfi);
   if (partial_assembly) { Ms_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Ms_form->Assemble();
   Ms_form->FormSystemMatrix(empty, Ms);
   //std::cout << "Check 20..." << std::endl;        

   // mas smatrix with rho
   MsRho_form = new ParBilinearForm(sfes);
   //auto *ms_blfi = new MassIntegrator;
   auto *msrho_blfi = new MassIntegrator(*Rho);   
   if (numerical_integ) { msrho_blfi->SetIntRule(&ir_i); }
   MsRho_form->AddDomainIntegrator(msrho_blfi);
   if (partial_assembly) { MsRho_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   MsRho_form->Assemble();
   MsRho_form->FormSystemMatrix(empty, MsRho);
   
   // temperature laplacian   
   Lt_form = new ParBilinearForm(sfes);
   //auto *lt_blfi = new DiffusionIntegrator;
   //Lt_coeff.constant = -1.0;   
   auto *lt_blfi = new DiffusionIntegrator(Lt_coeff);
   if (numerical_integ)
   {
      lt_blfi->SetIntRule(&ir_di);
   }
   Lt_form->AddDomainIntegrator(lt_blfi);
   if (partial_assembly)
   {
      Lt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Lt_form->Assemble();
   Lt_form->FormSystemMatrix(empty, Lt);
   
   // boundary terms
   /*
   bufferGradT = new ParGridFunction(vfes);
   gradTField = new VectorGridFunctionCoefficient(bufferGradT);
   t_bnlfi = new BoundaryNormalLFIntegrator(*gradTField);
   if (numerical_integ) { t_bnlfi->SetIntRule(&ir_ni); }
   T_bdr_form->AddBoundaryIntegrator(t_bnlfi);   
   */

   Ht_lincoeff.constant = kin_vis / Pr; 
   Ht_bdfcoeff.constant = 1.0 / dt;   
   Ht_form = new ParBilinearForm(sfes);
   //auto *hmt_blfi = new MassIntegrator(Ht_bdfcoeff); // unsteady bit
   hmt_blfi = new MassIntegrator(*rhoDtField);   
   /*
   if (constantViscosity == true) {   
     //auto *hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);
     hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);     
   } else {
   */
     //auto *hdt_blfi = new DiffusionIntegrator(*alphaField);
     hdt_blfi = new DiffusionIntegrator(*alphaField);     
     //}
   //std::cout << "Check 23..." << std::endl;        
   
   if (numerical_integ)
   {
     hmt_blfi->SetIntRule(&ir_di);
     hdt_blfi->SetIntRule(&ir_di);
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
   
   // temp boundary terms   
   Text_gfcoeff = new GridFunctionCoefficient(&Text_gf);
   Text_bdr_form = new ParLinearForm(sfes);
   auto *text_blfi = new BoundaryLFIntegrator(*Text_gfcoeff);
   if (numerical_integ) { text_blfi->SetIntRule(&ir_i); }
   Text_bdr_form->AddBoundaryIntegrator(text_blfi, temp_ess_attr);
   //std::cout << "Check 25..." << std::endl;        

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
      Ms_form->AssembleDiagonal(diag_pa);
      MsInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MsInvPC = new HypreSmoother(*Ms.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MsInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MsInv = new CGSolver(sfes->GetComm());
   MsInv->iterative_mode = false;
   MsInv->SetOperator(*Ms);
   MsInv->SetPreconditioner(*MsInvPC);
   MsInv->SetPrintLevel(pl_mtsolve);
   MsInv->SetRelTol(config.solver_tol);
   MsInv->SetMaxIter(config.solver_iter);
   //std::cout << "Check 26..." << std::endl;        
   
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
   //HtInv = new GMRESSolver(sfes->GetComm());
   //HtInv = new BiCGSTABSolver(sfes->GetComm());   
   HtInv->iterative_mode = true;
   HtInv->SetOperator(*Ht);
   HtInv->SetPreconditioner(*HtInvPC);
   HtInv->SetPrintLevel(pl_hsolve);
   HtInv->SetRelTol(config.solver_tol);
   HtInv->SetMaxIter(config.solver_iter);
   if (rank0_) std::cout << "Temperature operators set" << endl;   
   
     
   // Qt .....................................
   Mq_form = new ParBilinearForm(sfes);
   auto *mq_blfi = new MassIntegrator;
   if (numerical_integ) { mq_blfi->SetIntRule(&ir_i); }
   Mq_form->AddDomainIntegrator(mq_blfi);
   if (partial_assembly) { Mq_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Mq_form->Assemble();
   Mq_form->FormSystemMatrix(Qt_ess_tdof, Mq);

   if (partial_assembly)
   {
      Vector diag_pa(sfes->GetTrueVSize());
      Mq_form->AssembleDiagonal(diag_pa);
      MqInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      MqInvPC = new HypreSmoother(*Mq.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(MqInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }
   MqInv = new CGSolver(sfes->GetComm());
   MqInv->iterative_mode = false;
   MqInv->SetOperator(*Mq);
   MqInv->SetPreconditioner(*MqInvPC);
   MqInv->SetPrintLevel(pl_mtsolve);
   MqInv->SetRelTol(config.solver_tol);
   MqInv->SetMaxIter(config.solver_iter);
   
   LQ_form = new ParBilinearForm(sfes);
   auto *lqd_blfi = new DiffusionIntegrator(*alphaField);     
   if (numerical_integ) {
     lqd_blfi->SetIntRule(&ir_di);
   }
   LQ_form->AddDomainIntegrator(lqd_blfi);
   if (partial_assembly) {
     LQ_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   LQ_form->Assemble();
   LQ_form->FormSystemMatrix(empty, LQ);

   LQ_bdry = new ParLinearForm(sfes);
   auto *lq_bdry_lfi = new BoundaryNormalLFIntegrator(*kap_gradT_coeff, 2, -1);
   if (numerical_integ) {
     lq_bdry_lfi->SetIntRule(&ir_di);
   }
   LQ_bdry->AddBoundaryIntegrator(lq_bdry_lfi, temp_ess_attr);

   
   // remaining initialization
   if (loMach_.config.resetTemp == true) {
     double *data = Tn_gf.HostReadWrite();     
     for (int i = 0; i < Sdof; i++) {
       data[i] = config.initRhoRhoVp[4];
     }
     if (rank0_) std::cout << "Reset temperature to IC" << endl;        
   }     
   
   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   Tn_gf.GetTrueDofs(Tn);
   {
     double *data = Tn_next.HostReadWrite();
     double *Tdata = Tn.HostReadWrite();   
     for (int i = 0; i < SdofInt; i++) { data[i] = Tdata[i]; }
   }      
   Tn_next_gf.SetFromTrueDofs(Tn_next);
   //std::cout << "Check 28..." << std::endl;     

   // Temp filter
   filter_alpha = loMach_opts_.filterWeight;
   filter_cutoff_modes = loMach_opts_.nFilter;

   if (loMach_.loMach_opts_.filterTemp == true)     
   {
      sfec_filter = new H1_FECollection(order - filter_cutoff_modes);
      sfes_filter = new ParFiniteElementSpace(pmesh, sfec_filter);

      Tn_NM1_gf.SetSpace(sfes_filter);
      Tn_NM1_gf = 0.0;

      Tn_filtered_gf.SetSpace(sfes);
      Tn_filtered_gf = 0.0;
   }
   
}


void LoMachSolver::UpdateTimestepHistory(double dt)
{

   // temperature
   NTnm2 = NTnm1;
   NTnm1 = NTn;
   Tnm2 = Tnm1;
   Tnm1 = Tn;
   Tn_next_gf.GetTrueDofs(Tn_next);
   Tn = Tn_next;
   Tn_gf.SetFromTrueDofs(Tn);
   
}


void ThermoChem::thermoChemStep(double &time, double dt, const int current_step, const int start_step, bool provisional)
{
   
   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &temp_dbc : temp_dbcs) {temp_dbc.coeff->SetTime(time + dt);}   
   
   if (loMach_opts_.solveTemp == true) {
   //if(rank0_) {std::cout<< "in temp solve..." << endl;}     
   
   resT = 0.0;

   // convection
   computeExplicitTempConvectionOP(true); // ->tmpR0
   resT.Set(-1.0, tmpR0);                
   
   // for unsteady term, compute and add known part of BDF unsteady term
   /*
   {
     const double bd1idt = bd1 / dt;
     const double bd2idt = bd2 / dt;
     const double bd3idt = bd3 / dt;
     double *d_tn = Tn.HostReadWrite();     
     double *d_tnm1 = Tnm1.HostReadWrite();     
     double *d_tnm2 = Tnm2.HostReadWrite();     
     double *data = tmpR0.HostReadWrite();     
     MFEM_FORALL(i, tmpR0.Size(),	
     {
       data[i] = bd1idt*d_tn[i] + bd2idt*d_tnm1[i] + bd3idt*d_tnm2[i];
     });
   }
   */
   tmpR0.Set(bd1/dt,Tn);
   tmpR0.Add(bd2/dt,Tnm1);
   tmpR0.Add(bd3/dt,Tnm2);
   
   //multScalarScalarIP(rn,&tmpR0);
   MsRho->Mult(tmpR0,tmpR0b);
   resT.Add(-1.0,tmpR0b);
   //multConstScalarIP(Cp,&resT);      
   
   //resT.Add(-1.0, tmpR0); // move to rhs
   //Ms->Mult(resT,tmpR0);
   //resT.Set(1.0,tmpR0);

   // dPo/dt
   //tmpR0 = dtP;
   tmpR0 = (dtP/Cp);   
   Ms->Mult(tmpR0,tmpR0b);
   resT.Add(1.0, tmpR0b);

   // Add natural boundary terms here later

   //Ht_bdfcoeff.constant = bd0 / dt;
   { 
     double *data = bufferRhoDt->HostReadWrite();
     double *Rdata = rn_gf.HostReadWrite();
     //double coeff = Cp*bd0/dt;
     double coeff = bd0/dt;     
     double *d_imp = R0PM0_gf.HostReadWrite();
     for (int i = 0; i < Sdof; i++) { data[i] = coeff * Rdata[i]; }
     //for (int i = 0; i < Sdof; i++) {data[i] = coeff*Rdata[i] + d_imp[i]; }     
   }         
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
  
   for (auto &temp_dbc : temp_dbcs) { Tn_next_gf.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr); }
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
   Tn_next_gf.GetTrueDofs(Tn_next);

   // explicit filter
   if (loMach_opts_.filterTemp == true)
   {
      const auto filter_alpha_ = filter_alpha;
      Tn_NM1_gf.ProjectGridFunction(Tn_next_gf);
      Tn_filtered_gf.ProjectGridFunction(Tn_NM1_gf);
      const auto d_Tn_filtered_gf = Tn_filtered_gf.Read();
      auto d_Tn_gf = Tn_next_gf.ReadWrite();
      MFEM_FORALL(i, Tn_next_gf.Size(),			
      {
         d_Tn_gf[i] = (1.0 - filter_alpha_) * d_Tn_gf[i]
                    + filter_alpha_ * d_Tn_filtered_gf[i];
      });
      Tn_next_gf.GetTrueDofs(Tn_next);      
   }

   }
   
}


void ThermoChem::computeExplicitTempConvectionOP(bool extrap) {

   Array<int> empty;      
   At_form->Update();
   At_form->Assemble();
   At_form->FormSystemMatrix(empty, At);
   if (extrap == true) {
     At->Mult(Tn,NTn);
   } else {
     At->Mult(Text,NTn);     
   }

   // ab predictor
   if (extrap == true) {
      const auto d_Ntn = NTn.Read();
      const auto d_Ntnm1 = NTnm1.Read();
      const auto d_Ntnm2 = NTnm2.Read();
      auto data = tmpR0.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;      
      MFEM_FORALL(i, tmpR0.Size(),	
      {
         data[i] = ab1_*d_Ntn[i] + ab2_*d_Ntnm1[i] + ab3_*d_Ntnm2[i];
      });
    } else {
      tmpR0.Set(1.0,NTn);
    }    
   
}


void LoMachSolver::extrapolateState(int current_step) {
   
   // extrapolated temp at {n+1}
   {
      const auto d_Tn = Tn.Read();
      const auto d_Tnm1 = Tnm1.Read();
      const auto d_Tnm2 = Tnm2.Read();
      auto d_Text = Text.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;      
      MFEM_FORALL(i, Text.Size(),	
      {
	d_Text[i] = ab1_*d_Tn[i] + ab2_*d_Tnm1[i] + ab3_*d_Tnm2[i];
      });
   }
   
   // store ext in _next containers, remove Text, Uext completely later
   Tn_next_gf.SetFromTrueDofs(Text);
   Tn_next_gf.GetTrueDofs(Tn_next);         

}

// update thermodynamic pressure
void LoMachSolver::updateThermoP() {
  
  if (config.isOpen != true) {

    double allMass, PNM1;
    double myMass = 0.0;
    double *Tdata = Tn.HostReadWrite();
    double *data = tmpR0.HostReadWrite();    
    for (int i = 0; i < SdofInt; i++) {
       data[i] = 1.0 / Tdata[i];
    }
    multConstScalarIP((thermoPressure/Rgas),&tmpR0);
    Ms->Mult(tmpR0,tmpR0b);
    for (int i = 0; i < SdofInt; i++) {
      myMass += tmpR0b[i];
    }     
    MPI_Allreduce(&myMass, &allMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    PNM1 = thermoPressure;
    thermoPressure = systemMass/allMass * PNM1;
    dtP = (thermoPressure - PNM1) / dt;
    //if( rank0_ == true ) std::cout << " Original closed system mass: " << systemMass << " [kg]" << endl;
    //if( rank0_ == true ) std::cout << " New (closed) system mass: " << allMass << " [kg]" << endl;
    //if( rank0_ == true ) std::cout << " ThermoP: " << thermoPressure << " dtP: " << dtP << endl;             
    
  }

}

void LoMachSolver::updateDiffusivity() {

   // viscosity
   if (constantViscosity != true) {
     double *dataVisc = bufferVisc->HostReadWrite();        
     double *Tdata = Tn_gf.HostReadWrite();
     double *Rdata = rn_gf.HostReadWrite();     
     double visc[2];
     double prim[nvel+2];
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Sdof; i++) {
       prim[1+nvel] = Tdata[i];
       transportPtr->GetViscosities(prim, prim, visc); // returns dynamic
       dataVisc[i] = visc[0];       
     }     
   } else {
     double *dataVisc = bufferVisc->HostReadWrite();        
     for (int i = 0; i < Sdof; i++) { dataVisc[i] = dyn_vis; }
   }

   if (config.sgsModelType > 0) {
     double *dataSubgrid = bufferSubgridVisc->HostReadWrite();
     double *dataVisc = bufferVisc->HostReadWrite();
     double *Rdata = rn_gf.HostReadWrite();          
     for (int i = 0; i < Sdof; i++) { dataVisc[i] += Rdata[i] * dataSubgrid[i]; }     
   }
   if (config.linViscData.isEnabled) {
     double *viscMult = bufferViscMult->HostReadWrite();
     double *dataVisc = bufferVisc->HostReadWrite();
     for (int i = 0; i < Sdof; i++) { dataVisc[i] *= viscMult[i]; }
   }      

   // interior-sized visc needed for p-p rhs
   bufferVisc->GetTrueDofs(viscSml);   

   // for elasticity operator
   /*
   {
     double *data = bufferBulkVisc->HostReadWrite();
     double *dataMu = bufferVisc->HostReadWrite();
     double twothird = 2.0/3.0;
     if (bulkViscFlag == true) {
       for (int i = 0; i < Sdof; i++) { data[i] = twothird * dataMu[i]; }
     } else {
       for (int i = 0; i < Sdof; i++) { data[i] = 0.0; }       
     }
   }
   */
  
   // thermal diffusivity, this is really k
   {
     double *data = bufferAlpha->HostReadWrite();
     double *dataVisc = bufferVisc->HostReadWrite();
     //double Ctmp = Cp / Pr;
     double Ctmp = 1.0 / Pr;
     for (int i = 0; i < Sdof; i++) { data[i] = Ctmp * dataVisc[i]; }
   }
   bufferAlpha->GetTrueDofs(alphaSml);

   viscTotal_gf.SetFromTrueDofs(viscSml);      
   alphaTotal_gf.SetFromTrueDofs(alphaSml);

   // viscosity gradient
   G->Mult(viscSml, tmpR1);     
   MvInv->Mult(tmpR1, gradMu);
   
}


void LoMachSolver::updateDensity(double tStep) {

   Array<int> empty;
  
   // set rn
   if (constantDensity != true) {
     if(tStep == 1) {
       multConstScalarInv((thermoPressure/Rgas), Tn_next, &rn);
     } else if(tStep == 0.5) {
       multConstScalarInv((thermoPressure/Rgas), Tn_next, &tmpR0a);
       multConstScalarInv((thermoPressure/Rgas), Tn, &tmpR0b);
       {
         double *data = rn.HostWrite();
         double *data1 = tmpR0a.HostWrite();
         double *data2 = tmpR0b.HostWrite();        	 
         for (int i = 0; i < SdofInt; i++) { data[i] = 0.5 * (data1[i] + data2[i]); }
       }
     } else {
       multConstScalarInv((thermoPressure/Rgas), Tn, &rn);       
     }
   }
   else {
     double *data = rn.HostWrite();        
     for (int i = 0; i < SdofInt; i++) { data[i] = static_rho; }
   }
   rn_gf.SetFromTrueDofs(rn);

   {
     double *data = bufferRho->HostReadWrite();
     double *rho = rn_gf.HostReadWrite();     
     for (int i = 0; i < Sdof; i++) { data[i] = rho[i]; }
   }    
   MsRho_form->Update();
   MsRho_form->Assemble();
   MsRho_form->FormSystemMatrix(empty, MsRho);   

   R0PM0_gf.SetFromTrueDofs(rn);   
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);
   {
     double *data = bufferInvRho->HostReadWrite();
     double *rho = R0PM1_gf.HostReadWrite();     
     for (int i = 0; i < Pdof; i++) { data[i] = 1.0/rho[i]; }
   }    
   
   // density gradient
   G->Mult(rn, tmpR1);     
   MvInv->Mult(tmpR1, gradRho);   
   
}


void LoMachSolver::updateGradientsOP(double tStep) {

   // gradient of extrapolated temperature
   if (tStep == 0.0) {
     G->Mult(Tn, tmpR1);
   } else {
     G->Mult(Tn_next, tmpR1);     
   }
   MvInv->Mult(tmpR1, gradT);

}



///~ BC hard-codes ~///

double temp_ic(const Vector &coords, double t)
{

   double Thi = 400.0;  
   double Tlo = 200.0;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp;
   temp = Tlo + 0.5 * (y + 1.0) * (Thi - Tlo);
   //temp = 0.5 * (Thi + Tlo);
   
   return temp;      
   
}

double temp_wall(const Vector &coords, double t)  
{
   double Thi = 400.0;  
   double Tlo = 200.0;
   double Tmean, tRamp, wt;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp;

   Tmean = 0.5 * (Thi + Tlo);
   tRamp = 100.0;
   wt = min(t/tRamp,1.0);
   //wt = 0.0; // HACK   
   wt = 1.0; // HACK
   
   if(y > 0.0) {temp = Tmean + wt * (Thi-Tmean);}
   if(y < 0.0) {temp = Tmean + wt * (Tlo-Tmean);}

   return temp;   
}

double temp_wallBox(const Vector &coords, double t)  
{
   double Thi = 480.0;  
   double Tlo = 120.0;
   double Tmean, tRamp, wt;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp;

   Tmean = 0.5 * (Thi + Tlo);
   //tRamp = 10.0;
   //wt = min(t/tRamp,1.0);
   //wt = 1.0;
   
   temp = Tmean + x * (Thi-Tlo);   

   // hack
   //temp = 310.0;

   //std::cout << " *** temp_wallBox: " << temp << " at " << x << endl;
   
   return temp;   
}


double temp_inlet(const Vector &coords, double t)  
{
   double Thi = 400.0;  
   double Tlo = 200.0;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp;
   temp = Tlo + (y + 0.5) * (Thi - Tlo);
   return temp;   
}
