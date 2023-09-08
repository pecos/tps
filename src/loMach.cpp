

// look at elasticity integrator

///*  Add description later //

//#include "../../general/forall.hpp"
//#include "forall.hpp"
//#include "solvers.hpp"

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
void vel_channelTest(const Vector &coords, double t, Vector &u);


LoMachSolver::LoMachSolver(MPI_Session &mpi, LoMachOptions loMach_opts, TPS::Tps *tps)
    : mpi_(mpi), loMach_opts_(loMach_opts) {

  tpsP_ = tps;
  pmesh = NULL;
  nprocs_ = mpi_.WorldSize();
  rank_ = mpi_.WorldRank();
  if (rank_ == 0) { rank0_ = true; }
  else { rank0_ = false; }
  groupsMPI = new MPI_Groups(&mpi_);

  // is this needed?
  groupsMPI->init();

  // ini file options
  parseSolverOptions();  
  parseSolverOptions2(); 

  // incomp version
  if (config.const_visc > 0.0) { constantViscosity = true; }
  if (config.const_dens > 0.0) { constantDensity = true; }
  if ( constantViscosity == true && constantDensity == true ) {
    incompressibleSolve = true;
  }
  
  // set default solver state
  exit_status_ = NORMAL;

  // remove DIE file if present
  if (rank0_) {
    if (file_exists("DIE")) {
      grvy_printf(gdebug, "Removing DIE file on startup\n");
      remove("DIE");
    }
  }
  
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

   if ( loMach_opts_.uOrder == -1) {
     //LoMachOptions loMach_opts_ = GetOptions();   
     order = std::max(config.solOrder,1);
     //porder = std::max(order-1,1);
     porder = order;
     //norder = config.solOrder;
     //norder = 2*order;
     double no;
     no = ceil( ((double)order * 1.5) );   
     norder = int(no);
   } else {
     order = loMach_opts_.uOrder;
     porder = loMach_opts_.pOrder;
     norder = loMach_opts_.nOrder;
   }
   
   if (mpi_.Root()) {
     std::cout << "  " << endl;
     std::cout << " Order of solution-spaces " << endl;     
     std::cout << " +velocity : " << order << endl;
     std::cout << " +pressure : " << porder << endl;          
     std::cout << " +non-linear : " << norder << endl;
     std::cout << "  " << endl;          
   }

   // temporary hard-coding   
   //Re_tau = 1.0/1.48e-5; //182.0;   
   //kin_vis = 1.0/Re_tau;
   //ambientPressure = 101325.0;

   static_rho = config.const_dens;
   kin_vis = config.const_visc;
   ambientPressure = config.amb_pres;
   thermoPressure = config.amb_pres;
   tPm1 = thermoPressure;
   tPm2 = thermoPressure;
   dtP = 0.0;

   // get for dry air
   Rgas = 287.0;
   Pr = 1.2;
   Cp = 1.0;
   gamma = 1.4;

   
   //-----------------------------------------------------
   // 1) Prepare the mesh
   //-----------------------------------------------------
   
   // temporary hard-coded mesh
   /*
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
   if (verbose) grvy_printf(ginfo, "Constructed mesh...\n");   

   for (int i = 0; i < mesh.GetNV(); ++i)
   {
      double *v = mesh.GetVertex(i);
      v[1] = mesh_stretching_func(v[1]);
   }

   // Create translation vectors defining the periodicity
   Vector x_translation({Lx, 0.0, 0.0});
   Vector z_translation({0.0, 0.0, Lz});
   std::vector<Vector> translations = {x_translation, z_translation};   
   */

   /**/
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
    /**/
    

   // 1a) Read the serial mesh (on each mpi rank)
   //Mesh *mesh_ptr = new Mesh(lomach_opts_.mesh_file.c_str(), 1, 1);

   /**/
   //std::cout << " Reading mesh from: " << loMach_opts_.mesh_file.c_str() << endl;
   Mesh mesh = Mesh(loMach_opts_.mesh_file.c_str());
   Mesh *mesh_ptr = &mesh;
   Mesh *serial_mesh = &mesh;   
   //if (verbose) grvy_printf(ginfo, "Initial mesh read...\n");
   /**/
   
   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(translations));
   //if (verbose) grvy_printf(ginfo, "Mesh made periodic...\n");         

   // HACK HACK HACK HARD CODE
   nvel = 3;
   
   // underscore stuff is terrible
   dim_ = mesh_ptr->Dimension();
   //int dim = pmesh->Dimension();
   dim = dim_;
   //if (verbose) grvy_printf(ginfo, "Got mesh dim...\n");            

   // 1b) Refine the serial mesh, if requested
   /*
   if (verbose && (loMach_opts_.ref_levels > 0)) {
     grvy_printf(ginfo, "Refining mesh: ref_levels %d\n", loMach_opts_.ref_levels);
   }
   for (int l = 0; l < loMach_opts_.ref_levels; l++) {
     mesh_ptr->UniformRefinement();
   }
   */


   /*
   if (Mpi::Root())
   {
      printf("NL=%i NX=%i NY=%i NZ=%i dx+=%f\n", NL, NX, NY, NZ, LC * Re_tau);
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }
   */

  eqSystem = config.GetEquationSystem();
  mixture = NULL;

  // HARD CODE
  config.dryAirInput.specific_heat_ratio = 1.4;
  config.dryAirInput.gas_constant = 287.058;
  config.dryAirInput.f = config.workFluid;
  config.dryAirInput.eq_sys = config.eqSystem;  
  mixture = new DryAir(config, dim, nvel);  // conditional jump, must be using something in config that wasnt parsed?
  transportPtr = new DryAirTransport(mixture, config);
  
  /*
  switch (config.GetWorkingFluid()) {
    case WorkingFluid::DRY_AIR:
      mixture = new DryAir(config, dim, nvel);

#if defined(_CUDA_) || defined(_HIP_)
      tpsGpuMalloc((void **)(&d_mixture), sizeof(DryAir));
      gpu::instantiateDeviceDryAir<<<1, 1>>>(config.dryAirInput, dim, nvel, d_mixture);

      tpsGpuMalloc((void **)&transportPtr, sizeof(DryAirTransport));
      gpu::instantiateDeviceDryAirTransport<<<1, 1>>>(d_mixture, config.GetViscMult(), config.GetBulkViscMult(),
                                                      config.sutherland_.C1, config.sutherland_.S0,
                                                      config.sutherland_.Pr, transportPtr);
#else
      transportPtr = new DryAirTransport(mixture, config);
#endif
      break;
    case WorkingFluid::USER_DEFINED:
      switch (config.GetGasModel()) {
        case GasModel::PERFECT_MIXTURE:
          mixture = new PerfectMixture(config, dim, nvel);
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)(&d_mixture), sizeof(PerfectMixture));
          gpu::instantiateDevicePerfectMixture<<<1, 1>>>(config.perfectMixtureInput, dim, nvel, d_mixture);
#endif
          break;
        default:
          mfem_error("GasModel not recognized.");
          break;
      }
      switch (config.GetTranportModel()) {
        case ARGON_MINIMAL:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ArgonMinimalTransport));
          gpu::instantiateDeviceArgonMinimalTransport<<<1, 1>>>(d_mixture, config.argonTransportInput, transportPtr);
#else
          transportPtr = new ArgonMinimalTransport(mixture, config);
#endif
          break;
        case ARGON_MIXTURE:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ArgonMixtureTransport));
          gpu::instantiateDeviceArgonMixtureTransport<<<1, 1>>>(d_mixture, config.argonTransportInput, transportPtr);
#else
          transportPtr = new ArgonMixtureTransport(mixture, config);
#endif
          break;
        case CONSTANT:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&transportPtr, sizeof(ConstantTransport));
          gpu::instantiateDeviceConstantTransport<<<1, 1>>>(d_mixture, config.constantTransport, transportPtr);
#else
          transportPtr = new ConstantTransport(mixture, config);
#endif
          break;
        default:
          mfem_error("TransportModel not recognized.");
          break;
      }
      switch (config.GetChemistryModel()) {
        default:
#if defined(_CUDA_) || defined(_HIP_)
          tpsGpuMalloc((void **)&chemistry_, sizeof(Chemistry));
          gpu::instantiateDeviceChemistry<<<1, 1>>>(d_mixture, config.chemistryInput, chemistry_);
#else
          chemistry_ = new Chemistry(mixture, config.chemistryInput);
#endif
          break;
      }
      break;
      
    case WorkingFluid::LTE_FLUID:
#if defined(_GPU_)
      mfem_error("LTE_FLUID not supported for GPU.");
#else
      mixture = new LteMixture(config, dim, nvel);
      transportPtr = new LteTransport(mixture, config);
#endif
      break;
    default:
      mfem_error("WorkingFluid not recognized.");
      break;
  }
  */


  /*
  switch (config.radiationInput.model) {
    case NET_EMISSION:
#if defined(_CUDA_) || defined(_HIP_)
      tpsGpuMalloc((void **)(&radiation_), sizeof(NetEmission));
      gpu::instantiateDeviceNetEmission<<<1, 1>>>(config.radiationInput, radiation_);
#else
      radiation_ = new NetEmission(config.radiationInput);
#endif
      break;
    case NONE_RAD:
      break;
    default:
      mfem_error("RadiationModel not recognized.");
      break;      
  }
  */

  /*
  assert(mixture != NULL);
#if defined(_CUDA_) || defined(_HIP_)
#else
  d_mixture = mixture;
#endif
  */
  //std::cout << " okay 1..." << endl;


   
  MaxIters = config.GetNumIters();
  max_speed = 0.;
  num_equation = 5; // HARD CODE
  /*
  numSpecies = mixture->GetNumSpecies();
  numActiveSpecies = mixture->GetNumActiveSpecies();
  ambipolar = mixture->IsAmbipolar();
  twoTemperature_ = mixture->IsTwoTemperature();
  num_equation = mixture->GetNumEquations();
  */
  //std::cout << " okay 2..." << endl;  

  
  // check if a simulation is being restarted
  //std::cout << "RestartCycle: " << config.GetRestartCycle() << std::endl;  
  if (config.GetRestartCycle() > 0) {
    
    if (config.GetUniformRefLevels() > 0) {
      if (mpi_.Root()) {
        std::cerr << "ERROR: Uniform mesh refinement not supported upon restart." << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // read partitioning info from original decomposition (unless restarting from serial soln)
    nelemGlobal_ = serial_mesh->GetNE();
    //nelemGlobal_ = mesh->GetNE();
    if (rank0_) grvy_printf(ginfo, "Total # of mesh elements = %i\n", nelemGlobal_);

    if (nprocs_ > 1) {
      /*
      if (config.isRestartSerialized("read")) {
        assert(serial_mesh->Conforming());
        //assert(mesh->Conforming());	
        partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
	//partitioning_ = Array<int>(mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
        partitioning_file_hdf5("write");
      } else {
      */
        partitioning_file_hdf5("read");
      //}
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
      serial_mesh->UniformRefinement();
      //mesh->UniformRefinement();
    }

    // generate partitioning file (we assume conforming meshes)
    nelemGlobal_ = serial_mesh->GetNE();
    //nelemGlobal_ = mesh->GetNE();    
    if (nprocs_ > 1) {
      assert(serial_mesh->Conforming());
      //assert(mesh->Conforming());
      partitioning_ = Array<int>(serial_mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      //partitioning_ = Array<int>(mesh->GeneratePartitioning(nprocs_, defaultPartMethod), nelemGlobal_);
      if (rank0_) partitioning_file_hdf5("write");
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // make sure these are actually hooked up!
    time = 0.;
    iter = 0;
    
  }
  //std::cout << " okay 3..." << endl;
  
  // If requested, evaluate the distance function (i.e., the distance to the nearest no-slip wall)
  /*
  distance_ = NULL;
  GridFunction *serial_distance = NULL;
  if (config.compute_distance) {
    order = config.GetSolutionOrder();
    dim = serial_mesh->Dimension();
    basisType = config.GetBasisType();
    DG_FECollection *tmp_fec = NULL;
    if (basisType == 0) {
      tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLegendre);
    } else if (basisType == 1) {
      tmp_fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    }

    // Serial FE space for scalar variable
    FiniteElementSpace *serial_fes = new FiniteElementSpace(serial_mesh, tmp_fec);

    // Serial grid function for scalar variable
    serial_distance = new GridFunction(serial_fes);

    // Build a list of wall patches
    Array<int> wall_patch_list;
    for (int i = 0; i < config.wallPatchType.size(); i++) {
      if (config.wallPatchType[i].second != WallType::INV) {
        wall_patch_list.Append(config.wallPatchType[i].first);
      }
    }

    if (serial_mesh->GetNodes() == NULL) {
      serial_mesh->SetCurvature(1);
    }

    FiniteElementSpace *tmp_dfes = new FiniteElementSpace(serial_mesh, tmp_fec, dim, Ordering::byNODES);
    GridFunction coordinates(tmp_dfes);
    serial_mesh->GetNodes(coordinates);

    // Evaluate the distance function
    evaluateDistanceSerial(*serial_mesh, wall_patch_list, coordinates, *serial_distance);
  }
  */
   


  // ADD VISCOUS SPONGE
   

   // 1c) Partition the mesh (see partitioning_ in M2ulPhyS)
   //pmesh_ = new ParMesh(MPI_COMM_WORLD, *mesh);
   pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
   //auto *pmesh = new ParMesh(MPI_COMM_WORLD, periodic_mesh);
   //delete mesh_ptr;
   //delete mesh;  // no longer need the serial mesh
   // pmesh_->ReorientTetMesh();
   if (verbose) grvy_printf(ginfo, "Mesh partitioned...\n");      


   //-----------------------------------------------------
   // 2) Prepare the required finite elements
   //-----------------------------------------------------
   
   // velocity
   vfec = new H1_FECollection(order, dim);
   vfes = new ParFiniteElementSpace(pmesh, vfec, dim);

   // dealias nonlinear term 
   nfec = new H1_FECollection(norder, dim);   
   nfes = new ParFiniteElementSpace(pmesh, nfec, dim);   
   nfecR0 = new H1_FECollection(norder);   
   nfesR0 = new ParFiniteElementSpace(pmesh, nfecR0);   
   
   // pressure
   pfec = new H1_FECollection(porder);
   pfes = new ParFiniteElementSpace(pmesh, pfec);

   // temperature
   tfec = new H1_FECollection(order);
   tfes = new ParFiniteElementSpace(pmesh, tfec);   

   // density
   rfec = new H1_FECollection(order);
   rfes = new ParFiniteElementSpace(pmesh, rfec);   
   
   // full vector for compatability
   fvfes = new ParFiniteElementSpace(pmesh, vfec, num_equation); //, Ordering::byNODES);

   
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
   int pfes_truevsize = pfes->GetTrueVSize();
   int tfes_truevsize = tfes->GetTrueVSize();
   int nfes_truevsize = nfes->GetTrueVSize();
   int nfesR0_truevsize = nfesR0->GetTrueVSize();   
   int rfes_truevsize = rfes->GetTrueVSize();   
   if (verbose) grvy_printf(ginfo, "Got sizes...\n");   

   gravity.SetSize(3);
   gravity = 0.0;
   
   un.SetSize(vfes_truevsize);
   un = 0.0;
   un_next.SetSize(vfes_truevsize);
   un_next = 0.0;
   
   unm1.SetSize(vfes_truevsize);
   unm1 = 0.0;
   unm2.SetSize(vfes_truevsize);
   unm2 = 0.0;
   
   fn.SetSize(vfes_truevsize);
   
   bufferR0sml.SetSize(tfes_truevsize);
   bufferR1sml.SetSize(vfes_truevsize);   
   bufferR0sml = 0.0;
   bufferR1sml = 0.0;   
   
   // if dealiasing, use nfes 
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

   FBext.SetSize(nfes_truevsize);
   
   Fext.SetSize(vfes_truevsize);
   FText.SetSize(vfes_truevsize);
   Lext.SetSize(vfes_truevsize);
   Uext.SetSize(vfes_truevsize);   
   Ldiv.SetSize(vfes_truevsize);   
   resu.SetSize(vfes_truevsize); 

   tmpR1.SetSize(vfes_truevsize);
   tmpR1b.SetSize(vfes_truevsize);
   tmpR1c.SetSize(vfes_truevsize);

   gradMu.SetSize(vfes_truevsize);   
   gradU.SetSize(vfes_truevsize);
   gradV.SetSize(vfes_truevsize);
   gradW.SetSize(vfes_truevsize);      

   boussinesqField.SetSize(vfes_truevsize);
   boussinesqField = 0.0;   
   divU.SetSize(tfes_truevsize);
   divU = 0.0;
   Qt.SetSize(tfes_truevsize);
   Qt = 0.0;   

   pn.SetSize(pfes_truevsize);
   pn = 0.0;
   resp.SetSize(pfes_truevsize);
   resp = 0.0;
   FText_bdr.SetSize(tfes_truevsize);
   g_bdr.SetSize(tfes_truevsize);

   pnBig.SetSize(tfes_truevsize);
   pnBig = 0.0;   
   
   un_gf.SetSpace(vfes);
   un_gf = 0.0;
   un_next_gf.SetSpace(vfes);
   un_next_gf = 0.0;

   unm1_gf.SetSpace(vfes);
   unm1_gf = 0.0;
   unm2_gf.SetSpace(vfes);
   unm2_gf = 0.0;

   
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

   TBn.SetSize(nfesR0_truevsize);
   TBn = 298.0;   
   TBnm1.SetSize(nfesR0_truevsize);
   TBnm1 = 298.0;
   TBnm2.SetSize(nfesR0_truevsize);
   TBnm2 = 298.0;

   
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
   tmpR0b.SetSize(tfes_truevsize);
   tmpR0c.SetSize(tfes_truevsize);      
   
   Tn_gf.SetSpace(tfes);
   Tn_gf = 298.0; // fix hardcode
   Tn_next_gf.SetSpace(tfes);
   Tn_next_gf = 298.0;

   Tnm1_gf.SetSpace(tfes);
   Tnm1_gf = 0.0;
   Tnm2_gf.SetSpace(tfes);
   Tnm2_gf = 0.0;
   
   resT_gf.SetSpace(tfes);

   viscSml.SetSize(tfes_truevsize);
   viscSml = 1.0e-12;
   
   // density, not actually solved for directly
   rn.SetSize(rfes_truevsize);
   rn = 1.0;
   rn_gf.SetSpace(rfes);
   rn_gf = 1.0;   
   
   R0PM0_gf.SetSpace(tfes);
   R0PM0_gf = 0.0;
   R0PM1_gf.SetSpace(pfes);
   R0PM1_gf = 0.0;
   
   R1PM0_gf.SetSpace(vfes);
   R1PM0_gf = 0.0;   
   R1PX2_gf.SetSpace(nfes); // padded space
   R1PX2_gf = 0.0;
   R0PX2_gf.SetSpace(nfesR0); // padded space
   R0PX2_gf = 0.0;
   
   if (verbose) grvy_printf(ginfo, "vectors and gf initialized...\n");      
   
   //PrintInfo();

   /*
   // Paraview setup, not sure where this should go...
   //std::cout << "GetOutputName: " << config.GetOutputName() << endl;
   paraviewColl = new ParaViewDataCollection(config.GetOutputName(), pmesh);
   paraviewColl->SetLevelsOfDetail(config.GetSolutionOrder());
   paraviewColl->SetHighOrderOutput(true);
   paraviewColl->SetPrecision(8);
   if (verbose) grvy_printf(ginfo, "paraview collection initialized...\n");
   */

   
   initSolutionAndVisualizationVectors();
   if (verbose) grvy_printf(ginfo, "init Sol and Vis okay...\n");
   


   /*
  average = new Averaging(Up, pmesh, tfec, tfes, vfes, fvfes, eqSystem, d_mixture, num_equation, dim, config, groupsMPI);
  //average->read_meanANDrms_restart_files(); this guy is empty
  if (verbose) grvy_printf(ginfo, "average initialized...\n");        

  // register rms and mean sol into ioData
  if (average->ComputeMean()) {
    if (verbose) grvy_printf(ginfo, "setting up mean stuff...\n");              
    
    // meanUp
    ioData.registerIOFamily("Time-averaged primitive vars", "/meanSolution", average->GetMeanUp(), false,
                            config.GetRestartMean());
    ioData.registerIOVar("/meanSolution", "meanDens", 0);
    ioData.registerIOVar("/meanSolution", "mean-u", 1);
    ioData.registerIOVar("/meanSolution", "mean-v", 2);
    if (nvel == 3) {
      ioData.registerIOVar("/meanSolution", "mean-w", 3);
      ioData.registerIOVar("/meanSolution", "mean-E", 4);
    } else {
      ioData.registerIOVar("/meanSolution", "mean-p", dim + 1);
    }
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      // Only for NS_PASSIVE.
      if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

      // int inputSpeciesIndex = mixture->getInputIndexOf(sp);
      std::string speciesName = config.speciesNames[sp];
      ioData.registerIOVar("/meanSolution", "mean-Y" + speciesName, sp + nvel + 2);
    }

    // rms
    ioData.registerIOFamily("RMS velocity fluctuation", "/rmsData", average->GetRMS(), false, config.GetRestartMean());
    ioData.registerIOVar("/rmsData", "uu", 0);
    ioData.registerIOVar("/rmsData", "vv", 1);
    ioData.registerIOVar("/rmsData", "ww", 2);
    ioData.registerIOVar("/rmsData", "uv", 3);
    ioData.registerIOVar("/rmsData", "uw", 4);
    ioData.registerIOVar("/rmsData", "vw", 5);
  }
   */

  
  ioData.initializeSerial(mpi_.Root(), (config.RestartSerial() != "no"), serial_mesh);
  if (verbose) grvy_printf(ginfo, " ioData.init thingy...\n");

  /*
  projectInitialSolution();
  if (verbose) grvy_printf(ginfo, "initial sol projected...\n");
  */

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

  
  // ADD Up and U FILLER
  
  // estimate initial dt => how the hell does this work?
  //Up->ExchangeFaceNbrData();

  // 
  //if (config.GetRestartCycle() == 0) initialTimeStep();
  if (mpi_.Root()) cout << "Maximum element size: " << hmax << "m" << endl;
  if (mpi_.Root()) cout << "Minimum element size: " << hmin << "m" << endl;
  //if (mpi_.Root()) cout << "Initial time-step: " << dt << "s" << endl;
  
  
}


void LoMachSolver::Setup(double dt)
{

  //partial_assembly = true; // not working?
   //partial_assembly = false;
   //if (verbose) grvy_printf(ginfo, "in Setup...\n");
   /*
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
      if (partial_assembly) { mfem::out << "Using Partial Assembly" << std::endl; }
      else { mfem::out << "Using Full Assembly" << std::endl; }
   }
   */

   //int dim = pmesh->Dimension();      
   
   // adding bits from old main here
   // Set the initial condition.
   //ParGridFunction *r_gf = GetCurrentDensity();   
   ParGridFunction *u_gf = GetCurrentVelocity();
   ParGridFunction *p_gf = GetCurrentPressure();
   ParGridFunction *t_gf = GetCurrentTemperature();
   //std::cout << "Check 0..." << std::endl;   

   //ConstantCoefficient r_ic_coef;
   //r_ic_coef.constant = config.initRhoRhoVp[0];
   //r_gf->ProjectCoefficient(r_ic_coef);   
   
   VectorFunctionCoefficient u_ic_coef(dim, vel_ic);
   /*
   Vector uic_vec(3);
   uic_vec(0) = config.initRhoRhoVp[1];
   uic_vec(1) = config.initRhoRhoVp[2];
   uic_vec(2) = config.initRhoRhoVp[3];      
   VectorConstantCoefficient u_ic_coef(uic_vec);
   */
   u_gf->ProjectCoefficient(u_ic_coef);
   //std::cout << "Check 1..." << std::endl;
   
   FunctionCoefficient t_ic_coef(temp_ic);
   /*
   ConstantCoefficient t_ic_coef;
   t_ic_coef.constant = config.initRhoRhoVp[4] / (Rgas * config.initRhoRhoVp[0]);
   */
   t_gf->ProjectCoefficient(t_ic_coef);   
   //std::cout << "Check 2..." << std::endl;
   
   Array<int> domain_attr(pmesh->attributes);
   domain_attr = 1;
   //AddAccelTerm(accel, domain_attr); // HERE wire to input file config.gradPress[3]

   // add mean pressure gradient term
   Vector accel_vec(3);
   if ( config.isForcing ) {
     for (int i = 0; i < nvel; i++) { accel_vec[i] = config.gradPress[i]; }
   } else {
     for (int i = 0; i < nvel; i++) { accel_vec[i] = 0.0; }     
   }
   buffer_accel = new VectorConstantCoefficient(accel_vec);
   AddAccelTerm(buffer_accel, domain_attr);
   if (rank0_ ) {
     std::cout << " Acceleration: " << accel_vec[0] << " " << accel_vec[1] << " " << accel_vec[2] << endl;
   }
   
   // velocity bc
   Vector zero_vec(3); zero_vec = 0.0;
   Array<int> attr(pmesh->bdr_attributes.Max());
   attr = 0.0;
   //std::cout << "attr bdr size: " << pmesh->bdr_attributes.Max() << endl;   

   buffer_ubc = new VectorConstantCoefficient(zero_vec);         
   /*
   attr[1] = 1;
   attr[3] = 1;
   AddVelDirichletBC(vel_wall, attr);
   //AddVelDirichletBC(&u_bc_coef, attr);
   */
   /**/
   //for (int i = 0; i < config.wallPatchType.size(); i++) {
   for (int i = 0; i < numWalls; i++) {
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       //if (config.wallPatchType[i].second != WallType::INV) { 
       if (config.wallPatchType[i].second == WallType::VISC_ISOTH) {	   
	 //std::cout << " wall check " << i << " " << iFace << " " << config.wallPatchType[i].first << endl;
         if (iFace == config.wallPatchType[i].first) {
            attr[iFace-1] = 1;	 
         }
       }
     }
   }
   AddVelDirichletBC(vel_wall, attr);
   //AddVelDirichletBC(buffer_ubc, attr);
   /**/
   //std::cout << "Check 3..." << std::endl;    

   
   Vector wallTemp(numWalls);
   ConstantCoefficient t_bc_coef;
   //ConstantCoefficient t_bc_coef(numWalls);
   //VectorConstantCoefficient t_bc_coef(numWalls) ;
   for (int i = 0; i < numWalls; i++) {
     wallTemp[i] = config.wallBC[i].Th;
   }
   Array<int> Tattr(pmesh->bdr_attributes.Max());
   Tattr = 0;
   //ConstantCoefficient t_bc_coef(wallTemp);   
   //buffer_tbc = new VectorConstantCoefficient(wallTemp);      
   /*
   Tattr[1] = 1;
   Tattr[3] = 1;
   AddTempDirichletBC(temp_wall, Tattr);
   //AddTempDirichletBC(&t_bc_coef, Tattr);
   */
   /**/
   //for (int i = 0; i < config.wallPatchType.size(); i++) {
   for (int i = 0; i < numWalls; i++) {
     t_bc_coef.constant = config.wallBC[i].Th;
     for (int iFace = 1; iFace < pmesh->bdr_attributes.Max()+1; iFace++) {     
       if (config.wallPatchType[i].second != WallType::INV) { // should only be isothermal
         if (iFace == config.wallPatchType[i].first) {
            Tattr[iFace-1] = 1;	 
            //AddTempDirichletBC(&t_bc_coef, Tattr);
            //AddTempDirichletBC(wallTemp, Tattr); 	    
         }
       }
     }
   }
   //buffer_tbc = new ConstantCoefficient(t_bc_coef.constant);
   buffer_tbc = new ConstantCoefficient(t_bc_coef);   
   AddTempDirichletBC(temp_wall, Tattr); // FIX FIX FIX
   //AddTempDirichletBC(buffer_tbc, Tattr);
   //AddTempDirichletBC(t_bc_coef, Tattr);
   if (rank0_) std::cout << "***NUMBER OF WALLS: " << numWalls << endl;
   /**/
   //std::cout << "Check 4..." << std::endl;  

   // returning to regular setup   
   sw_setup.Start();

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);
   tfes->GetEssentialTrueDofs(temp_ess_attr, temp_ess_tdof);
   //nfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);    //  this may break?
   //std::cout << "Check 5..." << std::endl;     

   int Vdof = vfes->GetNDofs(); //vfes->GetVSize();   
   int Pdof = pfes->GetNDofs(); //pfes->GetVSize();
   int Tdof = tfes->GetNDofs(); //tfes->GetVSize();
   int Ndof = nfes->GetNDofs();   

   //int Vdof = vfes->GetTrueVSize();   
   //int Pdof = pfes->GetTrueVSize();
   //int Tdof = tfes->GetTrueVSize();
   
   int VdofInt = vfes->GetTrueVSize();   
   int PdofInt = pfes->GetTrueVSize();
   int TdofInt = tfes->GetTrueVSize();
   int NdofInt = nfes->GetTrueVSize();   

   //bufferPM0 = new ParGridFunction(tfes);
   //bufferPM1 = new ParGridFunction(pfes);      
   
   Array<int> empty;

   // unsteady: p+p = 2p
   // convection: p+p+(p-1) = 3p-1
   // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
   
   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 2 * order);
   const IntegrationRule &ir_nli = gll_rules.Get(nfes->GetFE(0)->GetGeomType(), 3 * norder - 1);
   //const IntegrationRule &ir_nli = gll_rules.Get(vfes->GetFE(0)->GetGeomType(), 3 * norder - 1);
   const IntegrationRule &ir_pi = gll_rules.Get(pfes->GetFE(0)->GetGeomType(), 2 * porder);   
   const IntegrationRule &ir_i  = gll_rules.Get(tfes->GetFE(0)->GetGeomType(), 2 * order);
   //std::cout << "Check 6..." << std::endl;     


   // add gravitational term term
   Vector gravity(3);
   if ( config.isGravity ) {
     for (int i = 0; i < nvel; i++) { gravity[i] = config.gravity[i]; }
   } else {
     for (int i = 0; i < nvel; i++) { gravity[i] = 0.0; }     
   }
   /*
   bufferGravity = new ParGridFunction(vfes);   
   {
     double *data = bufferGravity->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     for (int eq = 0; eq < nvel; eq++) {     
       for (int i = 0; i < Tdof; i++) {
  	 data[i] = ambientPressure / (Rgas * Tdata[i]) * gravity[eq];
       }       
     }
   }   
   bousField = new GridFunctionCoefficient(bufferGravity);
   AddGravityTerm(bousField, domain_attr);
   if (rank0_ ) {
     std::cout << " Gravity: " << gravity_vec[0] << " " << gravity_vec[1] << " " << gravity_vec[2] << endl;
   }   
   gravity_form = new ParLinearForm(vfes);
   for (auto &gravity_term : gravity_terms)
   {
      auto *grvdlfi = new VectorDomainLFIntegrator(*bousField);
      if (numerical_integ)
      {
         grvdlfi->SetIntRule(&ir_ni);
      }
      gravity_form->AddDomainIntegrator(grvdlfi);
   }
   */

   
   // convection section, extrapolation
   nlcoeff.constant = -1.0; // starts with negative
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
   //std::cout << "Check 7..." << std::endl;     

   // mass matrix
   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   if (numerical_integ) { mv_blfi->SetIntRule(&ir_ni); }
   Mv_form->AddDomainIntegrator(mv_blfi);
   if (partial_assembly) { Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);
   //std::cout << "Check 8..." << std::endl;     

   // need to build another with q=1/rho
   // DiffusionIntegrator(MatrixCoefficient &q, const IntegrationRule *ir = nullptr)
   // BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma); with sigma some type of Coefficient class
   //ParGridFunction buffer1(pfes);
   bufferInvRho = new ParGridFunction(tfes);
   {
     double *data = bufferInvRho->HostReadWrite();
     //double *dataRho = rn_gf.HostReadWrite();   
     //double *Tdata = Tn.HostReadWrite();
     //for (int eq = 0; eq < pmesh->Dimension(); eq++) {
     for (int i = 0; i < Tdof; i++) {     
       //dataRho[i] = ambientPressure / (Rgas * Tdata[i]);       
       //data[i] = (Rgas * Tdata[i]) / ambientPressure;
       data[i] = 1.0;
     }
   }
   //GridFunctionCoefficient invRho(&buffer1);    
   invRho = new GridFunctionCoefficient(bufferInvRho);
   //std::cout << "Check 9..." << std::endl;     
   
   // looks like this is the Laplacian for press eq
   Sp_form = new ParBilinearForm(pfes);
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
   //std::cout << "Check 10..." << std::endl;  

   Lt_form = new ParBilinearForm(tfes);
   auto *lt_blfi = new DiffusionIntegrator;
   if (numerical_integ)
   {
      lt_blfi->SetIntRule(&ir_nli);
   }
   Lt_form->AddDomainIntegrator(lt_blfi);
   if (partial_assembly)
   {
      Lt_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   Lt_form->Assemble();
   Lt_form->FormSystemMatrix(temp_ess_tdof, Lt);

   
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
   //std::cout << "Check 11..." << std::endl;  
   
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
   //std::cout << "Check 12..." << std::endl;  
   
   // viscosity field
   //ParGridFunction buffer2(vfes); // or pfes here?
   //bufferVisc = new ParGridFunction(vfes);
   bufferVisc = new ParGridFunction(tfes);
   {
     double *data = bufferVisc->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     //double *Tdata = Tn.HostReadWrite();     // TO version
     //Vector visc(2);
     //Vector prim(nvel+2);
     double visc[2];  // TO version
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
       //for (int eq = 0; eq < nvel; eq++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
	 //transportPtr->GetViscBasic(Tdata[i], visc);
	 //std::cout << " visc: " << visc[0] << endl;
	 
	 data[i] = visc[0];
	 //data[i] = kin_vis;
	 
         //data[i + eq * Tdof] = visc[0];	   
         //data[i + eq * Tdof] = kin_vis; // static value
	 //}
     }
   }
   //GridFunctionCoefficient viscField(&buffer2);
   viscField = new GridFunctionCoefficient(bufferVisc);
   //std::cout << "Check 13..." << std::endl;     
   
   // this is used in the last velocity implicit solve, Step kin_vis is
   // only for the pressure-poisson
   H_lincoeff.constant = kin_vis;
   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);
   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff); // diagonal from unsteady term   
   if (constantViscosity == true) {
     //auto *hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);
     hdv_blfi = new VectorDiffusionIntegrator(H_lincoeff);     
   } else {
     //auto *hdv_blfi = new VectorDiffusionIntegrator(*viscField);
     hdv_blfi = new VectorDiffusionIntegrator(*viscField);
   }
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
   //std::cout << "Check 14..." << std::endl;     
   
   // boundary terms   
   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(tfes);
   //FText_bdr_form = new ParLinearForm(vfes); // maybe?
   auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
   if (numerical_integ) { ftext_bnlfi->SetIntRule(&ir_ni); }
   FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);
   // std::cout << "Check 14..." << std::endl;     

   //g_bdr_form = new ParLinearForm(vfes); //? was pfes
   g_bdr_form = new ParLinearForm(tfes);
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
   MvInv->SetRelTol(1e-8);
   MvInv->SetMaxIter(500);
   //std::cout << "Check 17..." << std::endl;
   
   /**/
   if (partial_assembly)
   {
      lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
      SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      SpInvPC->SetPrintLevel(pl_amg);
      SpInvPC->Mult(resp, pn);
      //SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   else
   {
      SpInvPC = new HypreBoomerAMG(*Sp.As<HypreParMatrix>());
      SpInvPC->SetPrintLevel(0);
      //SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
      SpInvOrthoPC = new OrthoSolver(pfes->GetComm());
      SpInvOrthoPC->SetSolver(*SpInvPC);
   }
   //SpInv = new CGSolver(vfes->GetComm());
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
      //Vector diag_pa(pfes->GetTrueVSize()); // maybe?
      //std::cout << "Check 18a..." << std::endl;
      H_form->AssembleDiagonal(diag_pa); // invalid read
      //std::cout << "Check 18b..." << std::endl;                 
      HInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
      //std::cout << "Check 18c..." << std::endl;                 
   }
   else
   {
      HInvPC = new HypreSmoother(*H.As<HypreParMatrix>()); // conditional jump
      //std::cout << "Check 18d..." << std::endl;                 
      dynamic_cast<HypreSmoother *>(HInvPC)->SetType(HypreSmoother::Jacobi, 1);
      //std::cout << "Check 18e..." << std::endl;                 
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
   
   un_gf.GetTrueDofs(un); // ~ un = un_gf
   //un_next = un; // invalid read
   {
     double *data = un_next.HostReadWrite();
     double *Udata = un.HostReadWrite();   
     for (int i = 0; i < VdofInt; i++) {     
       data[i] = Udata[i];
     }
   }   
   un_next_gf.SetFromTrueDofs(un_next); // ~ un_next_gf = un_next

   
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
   bufferAlpha = new ParGridFunction(tfes);
   {
     double *data = bufferAlpha->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     //double *Tdata = Tn.HostReadWrite();     // TO version
     //Vector visc(2);
     //Vector prim(nvel+2);
     double visc[2]; // TO version
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         data[i] = visc[0] / Pr;
	 //data[i] = kin_vis / Pr;
     }
   }   
   //GridFunctionCoefficient alphaField(&buffer3);
   alphaField = new GridFunctionCoefficient(bufferAlpha);
   //std::cout << "Check 22..." << std::endl;        
   
   Ht_lincoeff.constant = kin_vis / Pr; 
   Ht_bdfcoeff.constant = 1.0 / dt;
   Ht_form = new ParBilinearForm(tfes);
   auto *hmt_blfi = new MassIntegrator(Ht_bdfcoeff); // unsteady bit
   if (constantViscosity == true) {   
     //auto *hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);
     hdt_blfi = new DiffusionIntegrator(Ht_lincoeff);     
   } else {
     //auto *hdt_blfi = new DiffusionIntegrator(*alphaField);
     hdt_blfi = new DiffusionIntegrator(*alphaField);     
   }
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
   
   // temp boundary terms   
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
   {
     double *data = Tn_next.HostReadWrite();
     double *Tdata = Tn.HostReadWrite();   
     for (int i = 0; i < TdofInt; i++) { data[i] = Tdata[i]; }
   }      
   Tn_next_gf.SetFromTrueDofs(Tn_next);
   //std::cout << "Check 28..." << std::endl;     

   
   // Set initial time step in the history array
   dthist[0] = dt;

   // Velocity filter
   filter_alpha = loMach_opts_.filterWeight;
   filter_cutoff_modes = loMach_opts_.nFilter;
   if (filter_cutoff_modes > 0) { pFilter = true; }
   
   if (pFilter == true)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes, pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh, vfec_filter, pmesh->Dimension());

      un_NM1_gf.SetSpace(vfes_filter);
      un_NM1_gf = 0.0;

      un_filtered_gf.SetSpace(vfes);
      un_filtered_gf = 0.0;
   }

   // add filter to temperature field later...
   if (pFilter == true)     
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


void LoMachSolver::projectInitialSolution() {
  
  // Initialize the state.

  // particular case: Euler vortex
  //   {
  //     void (*initialConditionFunction)(const Vector&, Vector&);
  //     t_final = 5.*  2./17.46;
  //     initialConditionFunction = &(this->InitialConditionEulerVortex);
  // initialConditionFunction = &(this->testInitialCondition);

  //     VectorFunctionCoefficient u0(num_equation, initialConditionFunction);
  //     U->ProjectCoefficient(u0);
  //   }
  //if (mpi_.Root()) std::cout << "restart: " << config.GetRestartCycle() << std::endl;

  /*  
#ifdef HAVE_MASA
  if (config.use_mms_) {
    initMasaHandler();
  }
#endif

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    if (config.use_mms_) {
#ifdef HAVE_MASA
      projectExactSolution(0.0, U);
      if (config.mmsSaveDetails_) projectExactSolution(0.0, masaU_);
#else
      mfem_error("Require MASA support to use MMS.");
#endif
    } else {
      uniformInitialConditions();
    }
  } else {
#ifdef HAVE_MASA
    if (config.use_mms_ && config.mmsSaveDetails_) projectExactSolution(0.0, masaU_);
#endif

    restart_files_hdf5("read");

    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->UseRestartMode(true);
  }
  */  

  /*
  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    // why would this be called if restart==0?  restart_files_hdf5("read");
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->UseRestartMode(true);
  }
  */
  //if (verbose) grvy_printf(ginfo, " PIS 1...\n");            
  
  //initGradUp();

  //updatePrimitives();

  /*
  // update pressure grid function
  mixture->UpdatePressureGridFunction(press, Up);
  //if (verbose) grvy_printf(ginfo, " PIS 2...\n");  

  // update plasma electrical conductivity
  if (tpsP_->isFlowEMCoupled()) {
    mixture->SetConstantPlasmaConductivity(plasma_conductivity_, Up);
  }
  //if (verbose) grvy_printf(ginfo, " PIS 3...\n");
  */

  if (config.GetRestartCycle() == 0 && !loadFromAuxSol) {
    // Only save IC from fresh start.  On restart, will save viz at
    // next requested iter.  This avoids possibility of trying to
    // overwrite existing paraview data for the current iteration.
    
    /////// CAUSING SEG FAULT     // HERE HERE HERE
    // if (!(tpsP_->isVisualizationMode())) paraviewColl->Save();
    
  }
  //if (verbose) grvy_printf(ginfo, " PIS 4...\n");            
  
}


void LoMachSolver::initialTimeStep() {
  
  auto dataU = U->HostReadWrite();
  int dof = vfes->GetNDofs();

  for (int n = 0; n < dof; n++) {
    Vector state(num_equation);
    for (int eq = 0; eq < num_equation; eq++) state[eq] = dataU[n + eq * dof];

    // REMOVE SOS
    //double iC = mixture->ComputeMaxCharSpeed(state);
    //if (iC > max_speed) max_speed = iC;
    
  }

  double partition_C = max_speed;
  MPI_Allreduce(&partition_C, &max_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  dt = CFL * hmin / max_speed / static_cast<double>(dim);

  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
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

  
  // just restart here
  if (config.restart) {
     restart_files_hdf5("read");
     copyU();
  }
  
  
   //std::cout << "Check 6..." << std::endl;

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
   
     /// make a seperate function ///
   /*
   if(iter == 0) {    
       //std::cout << "okay 1a " << dof << " " << dim << endl;
       for (int n = 0; n < dof; n++) {
         Umag = 0.0;
         for (int eq = 0; eq < dim; eq++) {
           Umag += dataU[n + eq * dof]*dataU[n + eq * dof];
         }
         Umag = sqrt(Umag);
         Umax_lcl = std::max(Umag,Umax_lcl);
       }
       //std::cout << "okay 1c " << endl;
       std::cout << "values: " << Umax_lcl << " " << max_speed << endl;       
       MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
       //std::cout << "okay 1d " << endl;       
       dt = CFL * hmin / max_speed; // this is conservative  ADD ORDER
       if( rank0_ == true ) { std::cout << " Starting timestep dt: " << dt << endl; }
       if( rank0_ == true ) { std::cout << " CFL / hmin / max_speed " << CFL << " " << hmin << " " << max_speed << endl; }              
    }
   */
  

   ParGridFunction *r_gf = GetCurrentDensity();   
   ParGridFunction *u_gf = GetCurrentVelocity();
   ParGridFunction *p_gf = GetCurrentPressure();
   ParGridFunction *t_gf = GetCurrentTemperature();

   int TdofInt = tfes->GetTrueVSize();
   
  // dt_fixed is initialized to -1, so if it is positive, then the
  // user requested a fixed dt run
  const double dt_fixed = config.GetFixedDT();
  if (dt_fixed > 0) {
    dt = dt_fixed;
  }   
  
   
   //dt = config.dt_fixed;
   //CFL_actual = ComputeCFL(*u_gf, dt);   
   //EnforceCFL(config.cflNum, u_gf, &dt);
   //dt = dt * max(cflMax_set/cflmax_actual,1.0);
   //if (verbose) grvy_printf(ginfo, "got timestep...\n");         

   /*
   if (!config.isTimeStepConstant()) {
     double dt_local = CFL * hmin / max_speed / static_cast<double>(dim);
     MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
     //dt = dt * max(CFL/CFL_actual,1.0);
   }
   */
   

   //updateU();
   //if (verbose) grvy_printf(ginfo, "updateU 1 good...\n");      
   
   Setup(dt);
   //if (verbose) grvy_printf(ginfo, "setup good...\n");   

   updateU();
   //copyU(); // testing   
   //if (verbose) grvy_printf(ginfo, "updateU 2 good...\n");         
   
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
   if( rank0_ == true ) std::cout << " Saving first step to paraview: " << iter << endl;
   /**/

   // if closed, mass is constant
   if (config.isOpen != true) {
     systemMass = 0.0;
     double myMass = 0.0;
     double *Tdata = Tn.HostReadWrite();
     double *Rdata = rn.HostReadWrite();          
     for (int i = 0; i < TdofInt; i++) {
       Rdata[i] = thermoPressure / (Rgas * Tdata[i]);
     }
     Mt->Mult(rn,tmpR0);
     for (int i = 0; i < TdofInt; i++) {
       myMass += tmpR0[i];
     }     
     MPI_Allreduce(&myMass, &systemMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     if( rank0_ == true ) std::cout << " Closed system mass: " << systemMass << " [kg]" << endl;     
   }      
   
   int iter_start = iter;
   if( rank0_ == true ) std::cout << " Starting main loop, from " << iter_start << " to " << MaxIters << endl;
   
   for (int step = iter_start; step <= MaxIters; step++)
   {
     
     iter = step;
     //if(rank0_) { std::cout << " Starting step " << iter << endl;}
     //std::cout << pmesh->GetMyRank() << ") Starting step " << iter << endl;
     //if (time + dt >= t_final - dt / 2) { break; }
     //if (step + 1 == MaxIters) { last_step = true; }
     //if (step+1 == 1) { last_step = true; }      ///HACK


     /// make a seperate function ///
     {
       auto dataU = un_gf.HostRead();
       //std::cout << "okay 1a " << dof << " " << dim << endl;
       for (int n = 0; n < dof; n++) {
         Umag = 0.0;
         for (int eq = 0; eq < dim; eq++) {
           Umag += dataU[n + eq * dof] * dataU[n + eq * dof];
         }
         Umag = std::sqrt(Umag);
         Umax_lcl = std::max(Umag,Umax_lcl);
       }
       //std::cout << "okay 1c " << endl;
       //std::cout << "values: " << Umax_lcl << " " << max_speed << endl;       
       MPI_Allreduce(&Umax_lcl, &max_speed, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
       //std::cout << "okay 1d " << endl;       
       dt = (1.0-dtFactor) * dt + dtFactor * CFL*hmin/(max_speed*(double)order) ; // this is conservative
       //std::cout << "okay 1e " << endl;
       //if( rank0_ == true ) { std::cout << "dt: " << dt << endl; }
       //if( rank0_ == true ) { std::cout << "Umax: " << max_speed << endl; }
     } 
     ////////////////////////////////

     
     /*
      if (step % 10 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }
     */

       //if (Mpi::Root())
      if (rank0_ == true) {
	//printf("%1s\n", " ");	
	//printf("%2s %11s %11s\n", "N", "Time", "dt");
	//printf("%o %4s %.5E %.5E\n", step, "    ", time, dt);
        //fflush(stdout);
        std::cout << "  N    Time    dt   "<< endl;
        std::cout << " " << step << "    " << time << " " << dt << endl; // conditional jump
      }

      //std::cout << " step/Max: " << step << " " << MaxIters << endl;
      if (step > MaxIters) { break; }

      //if (verbose) grvy_printf(ginfo, "calling step...\n");
      //if(rank0_) { std::cout << " ...entering main section" << endl;}      
      Step(time, dt, step, iter_start);
      //if(rank0_) { std::cout << " ...complete" << endl;}      


      
      // NECESSARY ADDITIONS:

      // update dt periodically
      /*
      if (!config.isTimeStepConstant()) {
        //double dt_local = CFL * hmin / max_speed / static_cast<double>(dim);
        //MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, mesh->GetComm());
        dt = dt * max(CFL/CFL_actual,1.0);	
      }
      */

      // restart files
      if ( iter%config.itersOut == 0 && iter != 0) {
	//if(rank0_) { std::cout << " Caught restart dump..." << endl;}
        updateU();
        restart_files_hdf5("write");
      }
      
      // paraview
      /*
      if (iter == MaxIters) {
        // auto hUp = Up->HostRead();
        Up->HostRead();
        mixture->UpdatePressureGridFunction(press, Up);
	
        paraviewColl->SetCycle(iter);
        paraviewColl->SetTime(time);
        paraviewColl->Save();

        average->write_meanANDrms_restart_files(iter, time);
	
      }
      */

      //std::cout << pmesh->GetMyRank() << ") Done with step " << iter << endl;      
      
   }

   //ParaViewDataCollection pvdc("turbchan", pmesh);
   //pvdc.SetDataFormat(VTKFormat::BINARY32);
   //pvdc.SetHighOrderOutput(true);
   //pvdc.SetLevelsOfDetail(order);
   pvdc.SetCycle(iter);
   pvdc.SetTime(time);
   //pvdc.RegisterField("velocity", u_gf);
   //pvdc.RegisterField("pressure", p_gf);
   //pvdc.RegisterField("temperature", t_gf);   
   pvdc.Save(); // invalid read and conditional jump
   if( rank0_ == true ) std::cout << " Saving final step to paraview: " << iter << endl;      
   
   /*
   //Up->HostRead();
   //mixture->UpdatePressureGridFunction(press, Up);
   restart_files_hdf5("write");
   paraviewColl->SetCycle(iter);
   paraviewColl->SetTime(time);
   paraviewColl->Save();
   */
   
   //std::cout << " Finished main loop"  << endl;   
   //flowsolver.PrintTimingData();

}



// all the variable coefficient operators musc be re-build every step...
void LoMachSolver::Step(double &time, double dt, const int current_step, const int start_step, bool provisional)
{
  //if (verbose) grvy_printf(ginfo, "in step...\n");           
   sw_step.Start();

   int Vdof = vfes->GetNDofs(); //vfes->GetVSize();   
   int Pdof = pfes->GetNDofs(); //pfes->GetVSize();
   int Tdof = tfes->GetNDofs(); //tfes->GetVSize();
   int Ndof = nfes->GetNDofs();
   int NdofR0 = nfesR0->GetNDofs();      

   //int Vdof = vfes->GetTrueVSize();   
   //int Pdof = pfes->GetTrueVSize();
   //int Tdof = tfes->GetTrueVSize();
   
   int VdofInt = vfes->GetTrueVSize();   
   int PdofInt = pfes->GetTrueVSize();
   int TdofInt = tfes->GetTrueVSize();
   int NdofInt = nfes->GetTrueVSize();
   int NdofR0Int = nfesR0->GetTrueVSize();   
   
   //std::cout << "Check a..." << std::endl;
   SetTimeIntegrationCoefficients(current_step-start_step);
   
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

   // extrapolated temp at {n+1}   
   {
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Uext = Uext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;      
      MFEM_FORALL(i, Uext.Size(),	
      {
         d_Uext[i] = ab1_*d_un[i] + ab2_*d_unm1[i] + ab3_*d_unm2[i];
      });
   }

   // set rn from extrapolated Text state
   if (constantDensity != true) {
     double *data = rn.HostWrite();        
     double *Tdata = Text.HostReadWrite();
     for (int i = 0; i < TdofInt; i++) {
         data[i] = thermoPressure / (Rgas * Tdata[i]);
     }
   }   

   // thermodynamic pressure for closed systems
   if (config.isOpen != true) {     
   }
   
   // Set current time for velocity Dirichlet boundary conditions.
   for (auto &vel_dbc : vel_dbcs) {vel_dbc.coeff->SetTime(time + dt);}

   // Set current time for pressure Dirichlet boundary conditions.
   for (auto &pres_dbc : pres_dbcs) {pres_dbc.coeff->SetTime(time + dt);}
   //std::cout << "Check b..." << std::endl;   

   /*
   double *dataVisc = bufferVisc->HostReadWrite();
   for (int i = 0; i < Vdof; i++) { // should this be dof of Vdof size?
     dataVisc[i] = kin_vis;
   }
   */
   //std::cout << "Check c..." << std::endl;

   //Tn_gf.SetFromTrueDofs(Tn);
   if (constantViscosity != true) {
     double *dataVisc = bufferVisc->HostWrite();        
     double *Tdata = Tn_gf.HostReadWrite();
     //double *Tdata = Tn.HostReadWrite();     // TO version
     //Vector visc(2); 
     //Vector prim(nvel+2);
     double visc[2];  // TO version
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
       //for (int eq = 0; eq < nvel; eq++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc); // returns dynamic

	 // kinematic needed for lhs of momentum
         dataVisc[i] = visc[0] * (Rgas * Tdata[i]) / thermoPressure;
	 //dataVisc[i] = kin_vis;

     }
   }
   
   
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
      H_form->AssembleDiagonal(diag_pa); // invalid read
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

   // gravity term
   {
     double *data = tmpR1.HostReadWrite();
     double *Rdata = rn.HostReadWrite();
     for (int eq = 0; eq < nvel; eq++) {     
       for (int i = 0; i < TdofInt; i++) {
  	 data[i + eq*TdofInt] = Rdata[i] * gravity[eq];
       }       
     }
   }
   Mv->Mult(tmpR1,boussinesqField);
   
   // Nonlinear extrapolated terms (convection: N*(n+1))
   sw_extrap.Start();

   // Nonlinear term at previous steps
   /*
   N->Mult(un, Nun);
   N->Mult(unm1, Nunm1);
   N->Mult(unm2, Nunm2);
   */
   
   // project u to "padded" order space
   /*
   un_gf.SetFromTrueDofs(un);         
   R1PX2_gf.ProjectGridFunction(un_gf);
   R1PX2_gf.GetTrueDofs(uBn);

   unm1_gf.SetFromTrueDofs(unm1);      
   R1PX2_gf.ProjectGridFunction(unm1_gf);
   R1PX2_gf.GetTrueDofs(uBnm1);

   unm2_gf.SetFromTrueDofs(unm2);         
   R1PX2_gf.ProjectGridFunction(unm2_gf);
   R1PX2_gf.GetTrueDofs(uBnm2);
   */

   un_gf.SetFromTrueDofs(Uext);         
   R1PX2_gf.ProjectGridFunction(un_gf);
   R1PX2_gf.GetTrueDofs(uBn);   
   //std::cout << "Check g..." << std::endl;         

   // this is wastefull, just get Uext and apply N once
   /*
   N->Mult(uBn, Nun); 
   N->Mult(uBnm1, Nunm1);
   N->Mult(uBnm2, Nunm2);
   */

   N->Mult(uBn, Nun); 
   //std::cout << "Check h..." << std::endl;
   
   // ab-predictor of nonliner term at {n+1}
   {
      const auto d_Nun = Nun.Read();
      //const auto d_Nunm1 = Nunm1.Read();
      //const auto d_Nunm2 = Nunm2.Read();
      auto d_Fext = FBext.Write();
      //auto d_Fext = Fext.Write();      
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      //mfem::forall(FBext.Size(), [=] MFEM_HOST_DEVICE (int i)
      //mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      //MFEM_FORALL(i, Fext.Size(),
      MFEM_FORALL(i, FBext.Size(),
      {
	//d_Fext[i] = ab1_ * d_Nun[i] +
        //             ab2_ * d_Nunm1[i] +
        //             ab3_ * d_Nunm2[i];
         d_Fext[i] = d_Nun[i];	 
      });
   }
   //std::cout << "Check i..." << std::endl;         

   // project NL product back to v-space
   R1PX2_gf.SetFromTrueDofs(FBext);
   R1PM0_gf.ProjectGridFunction(R1PX2_gf);  
   R1PM0_gf.GetTrueDofs(Fext);
   //std::cout << "Check j..." << std::endl;
      
   // add forcing/accel term to Fext   
   Fext.Add(1.0, fn);
   Fext.Add(1.0, boussinesqField);

   // Fext = M^{-1} (F(u^{n}) + f^{n+1}) (F* w/o known part of BDF)
   MvInv->Mult(Fext, tmpR1); // conditional jump
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
      //mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, Fext.Size(),      
      {
         d_Fext[i] += bd1idt * d_un[i] +
                      bd2idt * d_unm1[i] +
                      bd3idt * d_unm2[i];
      });
   }
   // Fext now has v*/dt from eq 2.3 in orszag
   sw_extrap.Stop();
   //std::cout << "Check l..." << std::endl;   

   // Pressure Poisson (extrapolated curl(curl(u)) L*(n+1))
   sw_curlcurl.Start();

   /*
   {
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      //auto d_Lext = Lext.Write();
      auto d_Lext = Uext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;      
      //mfem::forall(Lext.Size(), [=] MFEM_HOST_DEVICE (int i)
      MFEM_FORALL(i, Lext.Size(),	
      {
         d_Lext[i] = ab1_*d_un[i] + ab2_*d_unm1[i] + ab3_*d_unm2[i];
      });
   }
   */
   //std::cout << "Check m..." << std::endl;   

   //Lext_gf.SetFromTrueDofs(Lext);
   Lext_gf.SetFromTrueDofs(Uext);
   
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


   // interior-sized visc needed for p-p rhs
   if (constantViscosity != true) {
     double *dataViscSml = viscSml.HostReadWrite();                
     double *Tdata = Tn.HostReadWrite();     
     //Vector visc(2);
     //Vector prim(nvel+2);
     double visc[2];  // TO version
     double prim[nvel+2];          
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < TdofInt; i++) {
         prim[1+nvel] = Tdata[i];
         transportPtr->GetViscosities(prim, prim, visc);
         dataViscSml[i] = visc[0]; // dynamic
     }
   }

   
   // dataVisc is full size, Lext is only TrueSize
   if (constantViscosity != true) {
     //const double *dataVisc = bufferVisc->HostRead();     // TO version
     double *dataVisc = viscSml.HostReadWrite();
     //double *data = Lext.HostReadWrite();     
     for (int eq = 0; eq < dim; eq++) {
       for (int i = 0; i < TdofInt; i++) {
         Lext[i + eq*TdofInt] *= dataVisc[i];
         //Lext[i + eq * Tdof] = Lext[i + eq * Tdof] * dataVisc[i];
         //data[i + eq * TdofInt] = data[i + eq * TdofInt] * dataViscSml[i]; // invalid read
       }
     }
   } else {
     //double *data = Lext.HostReadWrite();     
     for (int eq = 0; eq < dim; eq++) {
       for (int i = 0; i < TdofInt; i++) {
         Lext[i + eq*TdofInt] *= (static_rho * kin_vis);
       }
     }     
   }
   //std::cout << "Check o..." << std::endl;      
   sw_curlcurl.Stop();


   // Qt ~= divU, thermal divergence
   if (incompressibleSolve != true) {
     
     // laplace(T) => should be T{n+1}, move T-solve before u?
     Lt->Mult(Text,tmpR0);
     MtInv->Mult(tmpR0, Qt);

     // multiple lapl(T) to get correct units
     // nabla(T) -> div(u): k/P = a*rho*Cp/P =  Cp*mu/Pr * 1/P
     {
       double *Tdata = Text.HostReadWrite();
       double *Mdata = viscSml.HostReadWrite();       
       double *data = Qt.HostReadWrite();     
       for (int i = 0; i < TdofInt; i++) {
         data[i] *= Cp * Mdata[i] / (Pr*thermoPressure);
       }
     }

     // add closed-domain pressure term
     if (config.isOpen != true) {
       double *data = Qt.HostReadWrite();     
       for (int i = 0; i < TdofInt; i++) {
         data[i] += ((gamma - 1.0)/gamma - Cp) * 1.0 / (Cp*thermoPressure) * dtP;
       }       
     }     
     
   }
   
   // dj(mu)(\tau/mu), have to store bits separately due to weak form helmholtz in momentum
   // could use elasticity integrator to prevent need to include part of this in momentum
   if (incompressibleSolve != true) {

     // viscosity gradient
     G->Mult(viscSml, tmpR1);
     MvInv->Mult(tmpR1, gradMu);

     // gradient of u_i
     {
       int eq = 0;
       double *dataU = Uext.HostReadWrite();
       double *data = tmpR0.HostReadWrite();              
       for (int i = 0; i < TdofInt; i++) {
         data[i] = dataU[i + eq*TdofInt];
       }
     }     
     G->Mult(tmpR0, tmpR1);
     MvInv->Mult(tmpR1, gradU);
     {
       int eq = 1;
       double *dataU = Uext.HostReadWrite();
       double *data = tmpR0.HostReadWrite();              
       for (int i = 0; i < TdofInt; i++) {
         data[i] = dataU[i + eq*TdofInt];
       }
     }     
     G->Mult(tmpR0, tmpR1);
     MvInv->Mult(tmpR1, gradV);
     {
       int eq = 2;
       double *dataU = Uext.HostReadWrite();
       double *data = tmpR0.HostReadWrite();              
       for (int i = 0; i < TdofInt; i++) {
         data[i] = dataU[i + eq*TdofInt];
       }
     }     
     G->Mult(tmpR0, tmpR1);
     MvInv->Mult(tmpR1, gradW);
     
     
     // divergence of u_i
     {
       int eq = 0;
       double *dataGradU = gradU.HostReadWrite();      
       double *data = divU.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i] = dataGradU[i + eq*TdofInt];
       }
     }
     {
       int eq = 1;
       double *dataGradU = gradV.HostReadWrite();      
       double *data = divU.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i] += dataGradU[i + eq*TdofInt];
       }
     }
     {
       int eq = 2;
       double *dataGradU = gradW.HostReadWrite();      
       double *data = divU.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i] += dataGradU[i + eq*TdofInt];
       }
     }


     // momentum only portion, TODO: break-up loops, this will thrash
     {
       int eq = 0;
       double *data1 = gradMu.HostReadWrite(); 
       double *dU = gradU.HostReadWrite();
       double *dV = gradV.HostReadWrite();
       double *dW = gradW.HostReadWrite();             
       double *data = Ldiv.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i + eq*TdofInt] = data1[i + 0*TdofInt] * dU[i + eq*TdofInt]
        	              + data1[i + 1*TdofInt] * dV[i + eq*TdofInt]
	                      + data1[i + 2*TdofInt] * dW[i + eq*TdofInt];
       }
     }    
     {
       int eq = 1;
       double *data1 = gradMu.HostReadWrite(); 
       double *dU = gradU.HostReadWrite();
       double *dV = gradV.HostReadWrite();
       double *dW = gradW.HostReadWrite();             
       double *data = Ldiv.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i + eq*TdofInt] = data1[i + 0*TdofInt] * dU[i + eq*TdofInt]
        	              + data1[i + 1*TdofInt] * dV[i + eq*TdofInt]
	                      + data1[i + 2*TdofInt] * dW[i + eq*TdofInt];
       }
     }
     {
       int eq = 2;
       double *data1 = gradMu.HostReadWrite(); 
       double *dU = gradU.HostReadWrite();
       double *dV = gradV.HostReadWrite();
       double *dW = gradW.HostReadWrite();             
       double *data = Ldiv.HostReadWrite();
       for (int i = 0; i < TdofInt; i++) {
         data[i + eq*TdofInt] = data1[i + 0*TdofInt] * dU[i + eq*TdofInt]
        	              + data1[i + 1*TdofInt] * dV[i + eq*TdofInt]
	                      + data1[i + 2*TdofInt] * dW[i + eq*TdofInt];
       }
     }         
     {
       double *data1 = gradMu.HostReadWrite(); 
       //double *divU = divU.HostReadWrite();
       double *divU = Qt.HostReadWrite(); // use only thermal-divergence
       double *data = Ldiv.HostReadWrite();
       double onethird = 1.0/3.0;
       for (int eq = 0; eq < nvel; eq++) {       
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] -= onethird * data1[i + eq*TdofInt] * divU[i];
         }
       }
     }         
         
     
     // full contribution to pressure-possion
     {
       double *data1 = Ldiv.HostReadWrite(); 
       double *data = tmpR1c.HostReadWrite();
       for (int eq = 0; eq < dim; eq++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] = data1[i + eq*TdofInt];
         }
       }
     }
     {
       int eq = 0;
       double *data1 = gradMu.HostReadWrite(); 
       double *data2 = gradU.HostReadWrite();      
       double *data = tmpR1c.HostReadWrite();
       for (int j = 0; j < dim; j++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] += data1[i + j*TdofInt]*data2[i + j*TdofInt];
         }
       }
     }    
     {
       int eq = 1;
       double *data1 = gradMu.HostReadWrite(); 
       double *data2 = gradV.HostReadWrite();      
       double *data = tmpR1c.HostReadWrite();
       for (int j = 0; j < dim; j++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] += data1[i + j*TdofInt]*data2[i + j*TdofInt];
         }
       }
     }
     {
       int eq = 2;
       double *data1 = gradMu.HostReadWrite(); 
       double *data2 = gradW.HostReadWrite();      
       double *data = tmpR1c.HostReadWrite();
       for (int j = 0; j < dim; j++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] += data1[i + j*TdofInt]*data2[i + j*TdofInt];
         }
       }
     }
     
   }
   /**/
   // Ldiv => to rhs of momentum
   // tmpR1c => to rhs of pressure-poisson
   

   // add Qt source here 
   if (incompressibleSolve != true) {     

     // gradient of Qt
     G->Mult(Qt, tmpR1);
     MvInv->Mult(tmpR1, tmpR1b);

     // mult by 4/3*mu
     {
       double factor = 4.0/3.0;
       double *dataVisc = viscSml.HostReadWrite();
       double *data = tmpR1b.HostReadWrite();     
       for (int eq = 0; eq < dim; eq++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] *= (factor * dataVisc[i]);
         }
       }
     }

     // add Qt part to Lext
     {
       double *dataDiv = tmpR1b.HostReadWrite();
       double *data = Lext.HostReadWrite();
       for (int eq = 0; eq < dim; eq++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] -= dataDiv[i + eq*TdofInt];
         }
       }
     }     

     // add grad(mu) part to Lext
     {
       double *data1 = tmpR1c.HostReadWrite();
       double *data = Lext.HostReadWrite();
       for (int eq = 0; eq < dim; eq++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] += data1[i + eq*TdofInt];
         }
       }
     }     
          
     // add Qt part to Ldiv (or substitute) for momentum
     {
       double *dataQterm = tmpR1b.HostReadWrite(); 
       double *data = Ldiv.HostReadWrite();
       for (int eq = 0; eq < dim; eq++) {
         for (int i = 0; i < TdofInt; i++) {
           data[i + eq*TdofInt] += dataQterm[i + eq*TdofInt];
           //data[i + eq*TdofInt] = dataQterm[i + eq*TdofInt]; // if not adding
         }
       }
     }
     
   }
   
   // can multiply rhs by rho prior to taking div instead of keeping in lhs,
   // Lext already has rho in it
   if (constantDensity != true) {   
     double *data = FText.HostReadWrite();     
     double *Tdata = Tn.HostReadWrite();
     for (int i = 0; i < TdofInt; i++) {     
       data[i] *= thermoPressure / (Rgas * Tdata[i]);
     }
   } else {
     double *data = FText.HostReadWrite();     
     for (int i = 0; i < TdofInt; i++) {     
       data[i] *= static_rho;
     }     
   }   
   
   // \tilde{F} = F - \nu CurlCurl(u), (F* + L*)
   FText.Set(-1.0, Lext); // negative from curl-curl
   FText.Add(1.0, Fext); // unsteady also neg on rhs, see HIGH-ORDER MATRIX-FREE INCOMPRESSIBLE FLOW SOLVERS eq 42

   // p_r = \nabla \cdot FText ( i think "p_r" means rhs of pressure-poisson eq, so div(\tilde{F}))
   // applied divergence to full FText vector
   //D->Mult(FText, resp);
   //resp.Neg();
   
   // add some if for porder != vorder
   // project FText to p-1
   D->Mult(FText, tmpR0);  // tmp3 c p, tmp2 and resp c (p-1) // possible invalid read
   tmpR0.Neg();
   //std::cout << "Check p..." << std::endl;   

   // unsteady div term, (-) on rhs rho * gamma * Qt/dt (after div term)
   /**/
   if (incompressibleSolve != true) {
     //double *dataDivU = divU.HostReadWrite();
     double *dataDivU = Qt.HostReadWrite();     
     double *Tdata = Tn.HostReadWrite();
     double *Rdata = rn.HostReadWrite();     
     double *data = tmpR0b.HostReadWrite();     
     for (int i = 0; i < TdofInt; i++) {     
       //data[i] = ambientPressure / (Rgas * Tdata[i]) * (gamma / dt) * dataDivU[i];
       data[i] = Rdata[i] * (bd0 / dt) * dataDivU[i];       
     }          
   }
   Mt->Mult(tmpR0b, tmpR0c);
   tmpR0.Add(+1.0,tmpR0c); // NOT SURE ABOUT SIGN!!!
   /**/
   
   // Add boundary terms.
   FText_gf.SetFromTrueDofs(FText);
   FText_bdr_form->Assemble();
   FText_bdr_form->ParallelAssemble(FText_bdr);
   g_bdr_form->Assemble();
   g_bdr_form->ParallelAssemble(g_bdr);
   tmpR0.Add(1.0, FText_bdr);
   tmpR0.Add(-bd0 / dt, g_bdr);

   //dont get why this is not correct
   //tmpR0.Add(-1.0, FText_bdr); // because tmpR0 is already negged
   //tmpR0.Add(+bd0 / dt, g_bdr);
   
   
   // project rhs to p-space
   R0PM0_gf.SetFromTrueDofs(tmpR0);   
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);
   R0PM1_gf.GetTrueDofs(resp);

   resp.Set(1.0, tmpR0);
   
   //resp.Add(1.0, FText_bdr);
   //resp.Add(-bd0/dt, g_bdr);
   
   //std::cout << "Check q..." << std::endl;   

   if (pres_dbcs.empty()) { Orthogonalize(resp); }   
   for (auto &pres_dbc : pres_dbcs) { pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr); }

   // ?
   pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);
   //std::cout << "Check r..." << std::endl;   
   
   // update variable coeff pressure laplacian
   /*
   R0PM0_gf.SetFromTrueDofs(Tn);   
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);
   //R0PM1_gf.GetTrueDofs(resp);
   */
   /*
   {
   double *data = R0PM0_gf.HostReadWrite();
   double *Tdata = Tn.HostReadWrite();      
   for (int i = 0; i < Tdof; i++) {     
     data[i] = Tdata[i];
   }
   }
   R0PM1_gf.ProjectGridFunction(R0PM0_gf);
   */
   {
     double *data = bufferInvRho->HostReadWrite();
     //double *dataRho = rn_gf.HostReadWrite();   
     //     double *Tdata = Tn.HostReadWrite();          
     for (int i = 0; i < Tdof; i++) {     
       //dataRho[i] = ambientPressure / (Rgas * Tdata[i]);
       //dataRho[i] = Tdata[i];              
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
   R0PM1_gf.SetFromTrueDofs(pn);   
   R0PM0_gf.ProjectGridFunction(R0PM1_gf);
   R0PM0_gf.GetTrueDofs(pnBig);

   // grad(P)
   G->Mult(pnBig, resu);

   // add Ldiv to rhs (has rho in it)
   if (incompressibleSolve != true) {          
     Mv->Mult(Ldiv,tmpR1);
     resu.Add(1.0, tmpR1);
   }
   
   // multiply by 1/rho
   if (constantDensity != true) {   
     double *data = resu.HostReadWrite();     
     double *Tdata = Tn.HostReadWrite();
     for (int i = 0; i < TdofInt; i++) {     
       data[i] *=  (Rgas * Tdata[i]) / thermoPressure;       
     }
   } else {
     double *data = resu.HostReadWrite();     
     for (int i = 0; i < TdofInt; i++) {     
       data[i] *= 1.0 / static_rho;
     }     
   }
   
   resu.Neg(); // -gradP => so res ARE on rhs
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
   un_next_gf.GetTrueDofs(un_next);


   // begin temperature......................................
   //std::cout << "Check u..." << std::endl;

   // Set current time for temperature Dirichlet boundary conditions.
   for (auto &temp_dbc : temp_dbcs) {temp_dbc.coeff->SetTime(time + dt);}   
   //std::cout << "Check v..." << std::endl;
   
   // helmholtz (alphaField)
   /*
   {
     double *data = bufferAlpha->HostReadWrite();
     for (int i = 0; i < Tdof; i++) {
       data[i] = kin_vis / Pr;
     }
   }
   */

   if (constantViscosity != true) {
     double *data = bufferAlpha->HostReadWrite();
     double *Tdata = Tn_gf.HostReadWrite();
     //double *Tdata = Tn.HostReadWrite();     // TO version
     //Vector visc(2);
     //Vector prim(nvel+2);
     double visc[2]; // TO version
     double prim[nvel+2];     
     for (int i = 0; i < nvel+2; i++) { prim[i] = 0.0; }
     for (int i = 0; i < Tdof; i++) {
         prim[1 + nvel] = Tdata[i]; 
         transportPtr->GetViscosities(prim, prim, visc);
	 data[i] = visc[0] / Pr;
     }
   }   
     
   Ht_bdfcoeff.constant = bd0 / dt;
   //Ht_bdfcoeff.constant = +1.0 / dt;
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


   /*
   //Dt->Mult(bufferR1sml, tmpR0); // explicit div(uT) at {n}
   //std::cout << "Check 7..." << std::endl;            
   //Dt->Mult(*bufferTemp, tmpR0);
   //Text.Set(-1.0, tmpR0);
   Text.Set(+1.0, tmpR0); // not sure here!   
   //resT.Add(-1.0, tmpR0);
   //std::cout << "Check 8..." << std::endl;            

   // M^-1 * advection => SOMETHING IS WRONG HERE!!! should be necessary but it makes the temp go crazy
   MtInv->Mult(Text, tmpR0);
   iter_mtsolve = MtInv->GetNumIterations();
   res_mtsolve = MtInv->GetFinalNorm();
   Text.Set(1.0, tmpR0);
   */
   
   // for unsteady term, compute and add known part of BDF unsteady term
   // moving this bit before advection to avoid the extra Minv
   // bd3 is unstable!!!
   {
     //double *d_Text = Text.HostReadWrite();
     const double bd1idt = bd1 / dt;
     const double bd2idt = bd2 / dt;
     const double bd3idt = bd3 / dt;
     const double *d_tn = Tn.Read();
     const double *d_tnm1 = Tnm1.Read();
     const double *d_tnm2 = Tnm2.Read();
     MFEM_FORALL(i, Text.Size(),	
     {
       Text[i] = bd1idt*d_tn[i] + bd2idt*d_tnm1[i] + bd3idt*d_tnm2[i];
       //Text[i] = -1.0/dt * d_tn[i];
     });
   }
   //std::cout << "Check y..." << std::endl;

   //std::cout << "Check y4..." << std::endl;   
      
   Mt->Mult(Text, tmpR0);
   //resT.Add(1.0, tmpR0); 
   resT.Set(-1.0, tmpR0); // move to rhs
   
   //resT.Neg(); // move to rhs
   //std::cout << "Check y6..." << std::endl;   

   // Add boundary terms.
   /*
   Text_gf.SetFromTrueDofs(Text);
   Text_bdr_form->Assemble();
   Text_bdr_form->ParallelAssemble(Text_bdr);
   t_bdr_form->Assemble();
   t_bdr_form->ParallelAssemble(t_bdr);
   resT.Add(1.0, Text_bdr);
   resT.Add(-bd0/dt, t_bdr);
   */   

   // advection
   Tn_gf.SetFromTrueDofs(Tn);         
   R0PX2_gf.ProjectGridFunction(Tn_gf);
   R0PX2_gf.GetTrueDofs(TBn);
   //std::cout << "Check 1..." << std::endl;      

   Tnm1_gf.SetFromTrueDofs(Tnm1);      
   R0PX2_gf.ProjectGridFunction(Tnm1_gf);
   R0PX2_gf.GetTrueDofs(TBnm1);
   //std::cout << "Check 2..." << std::endl;         

   Tnm2_gf.SetFromTrueDofs(Tnm2);         
   R0PX2_gf.ProjectGridFunction(Tnm2_gf);
   R0PX2_gf.GetTrueDofs(TBnm2);
   //std::cout << "Check 3..." << std::endl;
   
   //bufferTemp = new ParGridFunction(vfes);      
   //double *dataTemp = bufferTemp->HostReadWrite();
   {
     // TO version uses bufferTemp and then Dt->Mult(*bufferTemp, tmpR0) 
     //double *data = bufferR1sml.HostReadWrite();
     //double *data = bufferTemp->HostReadWrite();
     //double *data = FBext.HostReadWrite(); // container
     const double *Udnm0 = uBn.Read();
     const double *Udnm1 = uBnm1.Read();
     const double *Udnm2 = uBnm2.Read();     
     const double *Tdnm0 = TBn.Read();
     const double *Tdnm1 = TBnm1.Read();
     const double *Tdnm2 = TBnm2.Read();
     /*
     //double *data = Fext.HostReadWrite(); // container
     const double *Udnm0 = un.Read();
     const double *Udnm1 = unm1.Read();
     const double *Udnm2 = unm2.Read();     
     const double *Tdnm0 = Tn.Read();
     const double *Tdnm1 = Tnm1.Read();
     const double *Tdnm2 = Tnm2.Read();
     */
     const auto ab1_ = ab1;
     const auto ab2_ = ab2;
     const auto ab3_ = ab3;
     for (int eq = 0; eq < dim; eq++) {
       
       for (int i = 0; i < NdofR0Int; i++) {
         //FBext[i + eq * NdofR0Int] = Udnm0[i + eq*NdofR0Int] * Tdnm0[i];
         FBext[i + eq * NdofR0Int] = ab1_ * Udnm0[i + eq*NdofR0Int] * Tdnm0[i] +
	 	                     ab2_ * Udnm1[i + eq*NdofR0Int] * Tdnm1[i] +
	                             ab3_ * Udnm2[i + eq*NdofR0Int] * Tdnm2[i];	 
       }
       
       /*
       for (int i = 0; i < TdofInt; i++) {
         Fext[i + eq * TdofInt] = ab1_ * Udnm0[i + eq*TdofInt] * Tdnm0[i] +
       	                          ab2_ * Udnm1[i + eq*TdofInt] * Tdnm1[i] +
	                          ab3_ * Udnm2[i + eq*TdofInt] * Tdnm2[i];
       }
       */
       
     }
   }
   //std::cout << "Check 5..." << std::endl;      

   // project back to p-space
   R1PX2_gf.SetFromTrueDofs(FBext);
   R1PM0_gf.ProjectGridFunction(R1PX2_gf);
   R1PM0_gf.GetTrueDofs(Fext);

   Dt->Mult(Fext, tmpR0); // explicit div(uT) at extrapolated {n+1}
   
   // div(uT) should already be in integrated weak form and can be added directly
   resT.Add(-1.0, tmpR0); // minus to move to rhs

   // subtract T*divU from lhs to prevent divu source?
   //D->Mult(un, tmpR0);   
   {
     //double *data = tmpR0.HostReadWrite();
     double *data = divU.HostReadWrite();     
     double *Tdata = Tn.HostReadWrite();     
     MFEM_FORALL(i, Tn.Size(),	
     {
       data[i] *= Tdata[i];
     });
   }
   resT.Add(+1.0, tmpR0); // minus on lhs and minus to move to rhs

   
   //std::cout << "Check 7i..." << std::endl;   

   // what is this?   
   for (auto &temp_dbc : temp_dbcs) { Tn_next_gf.ProjectBdrCoefficient(*temp_dbc.coeff, temp_dbc.attr); }
   //std::cout << "Check 7j..." << std::endl;   

   // (?)
   tfes->GetRestrictionMatrix()->MultTranspose(resT, resT_gf);
   //std::cout << "Check 7jA..." << std::endl;      

   Vector Xt2, Bt2;
   if (partial_assembly)
   {
      auto *HC = Ht.As<ConstrainedOperator>();
      //std::cout << "Check 7jB..." << std::endl;                  
      EliminateRHS(*Ht_form, *HC, temp_ess_tdof, Tn_next_gf, resT_gf, Xt2, Bt2, 1);
      //std::cout << "Check 7jC..." << std::endl;            
   }
   else
   {
      Ht_form->FormLinearSystem(temp_ess_tdof, Tn_next_gf, resT_gf, Ht, Xt2, Bt2, 1);
      //std::cout << "Check 7jD..." << std::endl;                  
   }
   //std::cout << "Check 7k..." << std::endl;   

   // solve helmholtz eq for temp   
   HtInv->Mult(Bt2, Xt2);
   iter_htsolve = HtInv->GetNumIterations();
   res_htsolve = HtInv->GetFinalNorm();
   Ht_form->RecoverFEMSolution(Xt2, resT_gf, Tn_next_gf);
   Tn_next_gf.GetTrueDofs(Tn_next);

   // end temperature....................................
   
   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   // explicit filter
   /**/
   if (pFilter == true)
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
   /**/
   
   sw_step.Stop();


   /*
   // causing hangs
   if (channelTest == true) {
     VectorFunctionCoefficient u_test(dim, vel_channelTest);     
     un_gf.ComputeL2Error(u_test);
   }
   */

   // update thermodynamic pressure
   if (config.isOpen != true) {     
     double myInvT = 0.0;
     double systemInvT = 0.0;     
     double *Tdata = Tn.HostReadWrite();
     double *data = tmpR0.HostReadWrite();     
     for (int i = 0; i < TdofInt; i++) {
       data[i] = 1.0 / (Tdata[i]);
     }
     Mt->Mult(tmpR0,tmpR0b);
     for (int i = 0; i < TdofInt; i++) {
       myInvT += tmpR0b[i];
     }     
     MPI_Allreduce(&myInvT, &systemInvT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     tPm2 = tPm1;     
     tPm1 = thermoPressure;
     thermoPressure = Rgas * systemMass / systemInvT;
     dtP = 1.5*thermoPressure - 2.0*tPm1 + 0.5*tPm2;
     dtP = dtP / dt;
   }      

   
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

  //std::cout << " updating big U vector..." << endl;
  //int dof = tfes->GetNDofs();
  //std::cout << " check 0" << endl;      

  // primitive state
  {

    //double *dataU = U->HostReadWrite();
    double *dataUp = Up->HostReadWrite();
    //std::cout << " check 1" << endl;    
    
    const auto d_rn_gf = rn_gf.Read();    
    //mfem::forall(rn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, rn_gf.Size(),      
    {
      dataUp[i] = d_rn_gf[i];
    });
    //std::cout << " check 2" << endl;        
    
    int vstart = rfes->GetNDofs();
    const auto d_un_gf = un_gf.Read();    
    //mfem::forall(un_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, un_gf.Size(),      
    {
      dataUp[i + vstart] = d_un_gf[i];
    });
    //std::cout << " check 3" << endl;        

    int tstart = (1+nvel)*(tfes->GetNDofs());    
    const auto d_tn_gf = Tn_gf.Read();    
    //mfem::forall(Tn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, Tn_gf.Size(),      
    {
      dataUp[i + tstart] = d_tn_gf[i];
    });
    //std::cout << " check 4" << endl;
    
  }

  // conserved state
  /*
  {

    double *dataU = U->HostReadWrite();
    double *dataUp = Up->HostReadWrite();
  
    Vector Upi;
    Upi.UseDevice(false);
    Upi.SetSize(num_equation);
    Vector Ui;
    Ui.UseDevice(false);
    Ui.SetSize(num_equation);

    for (int i = 0; i < dof; i++) {
      for (int eq = 0; eq < num_equation; eq++) {
         Upi(eq) = dataUp[i + eq * dof];
      }    
      mixture->GetConservativesFromPrimitives(Upi, Ui);
      for (int eq = 0; eq < num_equation; eq++) {
        dataU[i + eq * dof] = Ui(eq);
      }    
    }
  }
*/

}


void LoMachSolver::copyU() {

  std::cout << " copying from U vector..." << endl;
  //int dof = tfes->GetNDofs();
  //std::cout << " check 0" << endl;      

  // primitive state
  {

    double *dataUp = Up->HostReadWrite();
    //std::cout << " check 1" << endl;    
    
    double *d_rn_gf = rn_gf.ReadWrite();    
    //mfem::forall(rn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, rn_gf.Size(),      
    {
      d_rn_gf[i] = dataUp[i];
    });
    //std::cout << " check 2" << endl;        
    
    int vstart = rfes->GetNDofs();
    double *d_un_gf = un_gf.ReadWrite();    
    //mfem::forall(un_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, un_gf.Size(),      
    {
       d_un_gf[i] = dataUp[i + vstart];
    });
    //std::cout << " check 3" << endl;        

    int tstart = (1+nvel)*(tfes->GetNDofs());    
    double *d_tn_gf = Tn_gf.ReadWrite();    
    //mfem::forall(Tn_gf.Size(), [=] MFEM_HOST_DEVICE (int i)
    MFEM_FORALL(i, Tn_gf.Size(),      
    {
       d_tn_gf[i] = dataUp[i + tstart];
    });
    //std::cout << " check 4" << endl;
    
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
//void LoMachSolver::AddTempDirichletBC(Array<int> &coeff, Array<int> &attr)
{

   temp_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Temperature Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i) {
         if (attr[i] == 1) {
	   mfem::out << i << " ";
	 }
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

void LoMachSolver::AddGravityTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   gravity_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Gravity term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1) { mfem::out << i << " "; }
      }
      mfem::out << std::endl;
   }
}

void LoMachSolver::AddGravityTerm(VecFuncT *g, Array<int> &attr)
{
   AddGravityTerm(new VectorFunctionCoefficient(pmesh->Dimension(), g), attr);
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
   
   delete vfec;
   delete pfec;
   delete rfec;
   delete tfec;
   
   delete vfes;
   delete pfes;
   delete rfes;
   delete tfes;
   delete fvfes;      
   
   delete vfec_filter;
   delete vfes_filter;
   delete tfec_filter;
   delete tfes_filter;
   
}
*/

// query solver-specific runtime controls
void LoMachSolver::parseSolverOptions() {

  //if (verbose) grvy_printf(ginfo, "parsing solver options...\n");        
  tpsP_->getRequiredInput("loMach/mesh", loMach_opts_.mesh_file);
  tpsP_->getInput("loMach/order", loMach_opts_.order, 1);  
  tpsP_->getInput("loMach/uOrder", loMach_opts_.uOrder, -1);
  tpsP_->getInput("loMach/pOrder", loMach_opts_.pOrder, -1);
  tpsP_->getInput("loMach/nOrder", loMach_opts_.nOrder, -1);    
  tpsP_->getInput("loMach/ref_levels", loMach_opts_.ref_levels, 0);
  tpsP_->getInput("loMach/max_iter", loMach_opts_.max_iter, 100);
  tpsP_->getInput("loMach/rtol", loMach_opts_.rtol, 1.0e-6);
  tpsP_->getInput("loMach/nFilter", loMach_opts_.nFilter, 0);
  tpsP_->getInput("loMach/filterWeight", loMach_opts_.filterWeight, 1.0);

  tpsP_->getInput("loMach/channelTest", loMach_opts_.channelTest, false);      
  
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

  // RMS job management
  //parseRMSJobOptions();

  // heat source
  //parseHeatSrcOptions();

  // viscosity multiplier function
  parseViscosityOptions();

  // MMS (check before IC b/c if MMS, IC not required)
  //parseMMSOptions();

  // initial conditions
  //if (!config.use_mms_) { parseICOptions(); }
  parseICOptions();

  // add passive scalar forcings
  //parsePassiveScalarOptions();

  // fluid presets
  parseFluidPreset();

  //parseSystemType();

  // plasma conditions.
  /*
  config.gasModel = NUM_GASMODEL;
  config.transportModel = NUM_TRANSPORTMODEL;
  config.chemistryModel_ = NUM_CHEMISTRYMODEL;
  if (config.workFluid == USER_DEFINED) {
    parsePlasmaModels();
  } else {
    // parse options for other plasma presets.
  }
  */

  // species list.
  //parseSpeciesInputs();

  //if (config.workFluid == USER_DEFINED) {  // Transport model
  //  parseTransportInputs();
  //}

  // Reaction list
  //parseReactionInputs();

  // changed the order of parsing in order to use species list.
  // boundary conditions. The number of boundaries of each supported type if
  // parsed first. Then, a section corresponding to each defined boundary is parsed afterwards
  parseBCInputs();

  // sponge zone
  //parseSpongeZoneInputs();

  // Pack up input parameters for device objects.
  //packUpGasMixtureInput();

  // Radiation model
  //parseRadiationInputs();

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
  //tpsP_->getInput("flow/useRoe", config.useRoe, false);
  //tpsP_->getInput("flow/refLength", config.refLength, 1.0);
  tpsP_->getInput("loMach/viscosityMultiplier", config.visc_mult, 1.0);
  tpsP_->getInput("loMach/bulkViscosityMultiplier", config.bulk_visc, 0.0);
  //tpsP_->getInput("flow/axisymmetric", config.axisymmetric_, false);
  tpsP_->getInput("loMach/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("loMach/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("loMach/SutherlandPr", config.sutherland_.Pr, 0.71);

  tpsP_->getInput("loMach/constantViscosity", config.const_visc, -1.0);
  tpsP_->getInput("loMach/constantDensity", config.const_dens, -1.0);
  tpsP_->getInput("loMach/ambientPressure", config.amb_pres, 101325.0);  
  tpsP_->getInput("loMach/openSystem", config.isOpen, true);
  
  tpsP_->getInput("loMach/enablePressureForcing", config.isForcing, false);
  if (config.isForcing) {
    for (int d = 0; d < 3; d++) tpsP_->getRequiredVecElem("loMach/pressureGrad", config.gradPress[d], d);
  }

  tpsP_->getInput("loMach/enableGravity", config.isGravity, false);
  if (config.isGravity) {
    for (int d = 0; d < 3; d++) tpsP_->getRequiredVecElem("loMach/gravity", config.gravity[d], d);
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

  // Sutherland inputs default to dry air values
  /*
  tpsP_->getInput("flow/SutherlandC1", config.sutherland_.C1, 1.458e-6);
  tpsP_->getInput("flow/SutherlandS0", config.sutherland_.S0, 110.4);
  tpsP_->getInput("flow/SutherlandPr", config.sutherland_.Pr, 0.71);
  */
}

void LoMachSolver::parseTimeIntegrationOptions() {
  //std::map<std::string, int> integrators;
  //integrators["forwardEuler"] = 1;
  //integrators["rk2"] = 2;
  //integrators["rk3"] = 3;
  //integrators["rk4"] = 4;
  //integrators["rk6"] = 6;

  //std::cout << "getting time options" << endl;
  std::string type;
  tpsP_->getInput("time/cfl", config.cflNum, 0.2);
  //std::cout << "got cflNum" << config.cflNum << endl;  
  //tpsP_->getInput("time/integrator", type, std::string("rk4"));
  tpsP_->getInput("time/enableConstantTimestep", config.constantTimeStep, false);
  tpsP_->getInput("time/dt_fixed", config.dt_fixed, -1.);
  //if (integrators.count(type) == 1) {
  //  config.timeIntegratorType = integrators[type];
  //} else {
  //  grvy_printf(GRVY_ERROR, "Unknown time integrator > %s\n", type.c_str());
  //  exit(ERROR);
  //}
}

void LoMachSolver::parseStatOptions() {
  tpsP_->getInput("averaging/saveMeanHist", config.meanHistEnable, false);
  tpsP_->getInput("averaging/startIter", config.startIter, 0);
  tpsP_->getInput("averaging/sampleFreq", config.sampleInterval, 0);
  //std::cout << " sampleInterval: " << config.sampleInterval << endl;
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

/*
void M2ulPhyS::parseHeatSrcOptions() {
  int numHeatSources;
  tpsP->getInput("heatSource/numHeatSources", numHeatSources, 0);
  config.numHeatSources = numHeatSources;

  if (numHeatSources > 0) {
    config.numHeatSources = numHeatSources;
    config.heatSource = new heatSourceData[numHeatSources];

    for (int s = 0; s < numHeatSources; s++) {
      std::string base("heatSource" + std::to_string(s + 1));

      tpsP->getInput((base + "/isEnabled").c_str(), config.heatSource[s].isEnabled, false);

      if (config.heatSource[s].isEnabled) {
        tpsP->getRequiredInput((base + "/value").c_str(), config.heatSource[s].value);

        std::string type;
        tpsP->getRequiredInput((base + "/distribution").c_str(), type);
        if (type == "cylinder") {
          config.heatSource[s].type = type;
        } else {
          grvy_printf(GRVY_ERROR, "\nUnknown heat source distribution -> %s\n", type.c_str());
          exit(ERROR);
        }

        tpsP->getRequiredInput((base + "/radius").c_str(), config.heatSource[s].radius);
        config.heatSource[s].point1.SetSize(3);
        config.heatSource[s].point2.SetSize(3);

        tpsP->getRequiredVec((base + "/point1").c_str(), config.heatSource[s].point1, 3);
        tpsP->getRequiredVec((base + "/point2").c_str(), config.heatSource[s].point2, 3);
      }
    }
  }
}
*/

void LoMachSolver::parseViscosityOptions() {
  
  tpsP_->getInput("viscosityMultiplierFunction/isEnabled", config.linViscData.isEnabled, false);
  
  if (config.linViscData.isEnabled) {
    auto normal = config.linViscData.normal.HostWrite();
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[0], 0);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[1], 1);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/norm", normal[2], 2);

    auto point0 = config.linViscData.point0.HostWrite();
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[0], 0);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[1], 1);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/p0", point0[2], 2);

    auto pointInit = config.linViscData.pointInit.HostWrite();
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[0], 0);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[1], 1);
    tpsP_->getRequiredVecElem("viscosityMultiplierFunction/pInit", pointInit[2], 2);
    tpsP_->getRequiredInput("viscosityMultiplierFunction/width", config.linViscData.width);
    tpsP_->getRequiredInput("viscosityMultiplierFunction/viscosityRatio", config.linViscData.viscRatio);
  }
  
}

/*
void M2ulPhyS::parseMMSOptions() {
  tpsP->getInput("mms/isEnabled", config.use_mms_, false);

  if (config.use_mms_) {
    tpsP->getRequiredInput("mms/name", config.mms_name_);
    tpsP->getInput("mms/compare_rhs", config.mmsCompareRhs_, false);
    tpsP->getInput("mms/save_details", config.mmsSaveDetails_, false);
  }
}
*/

void LoMachSolver::parseICOptions() {
  tpsP_->getRequiredInput("initialConditions/rho", config.initRhoRhoVp[0]);
  tpsP_->getRequiredInput("initialConditions/rhoU", config.initRhoRhoVp[1]);
  tpsP_->getRequiredInput("initialConditions/rhoV", config.initRhoRhoVp[2]);
  tpsP_->getRequiredInput("initialConditions/rhoW", config.initRhoRhoVp[3]);
  tpsP_->getRequiredInput("initialConditions/pressure", config.initRhoRhoVp[4]);
}

/*
void M2ulPhyS::parsePassiveScalarOptions() {
  int numForcings;
  tpsP->getInput("passiveScalars/numScalars", numForcings, 0);
  for (int i = 1; i <= numForcings; i++) {
    // add new entry in arrayPassiveScalar
    config.arrayPassiveScalar.Append(new passiveScalarData);
    config.arrayPassiveScalar[i - 1]->coords.SetSize(3);

    std::string basepath("passiveScalar" + std::to_string(i));

    Array<double> xyz;
    tpsP->getRequiredVec((basepath + "/xyz").c_str(), xyz, 3);
    for (int d = 0; d < 3; d++) config.arrayPassiveScalar[i - 1]->coords[d] = xyz[d];

    double radius;
    tpsP->getRequiredInput((basepath + "/radius").c_str(), radius);
    config.arrayPassiveScalar[i - 1]->radius = radius;

    double value;
    tpsP->getRequiredInput((basepath + "/value").c_str(), value);
    config.arrayPassiveScalar[i - 1]->value = value;
  }
}
*/

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

/*
void M2ulPhyS::parseSystemType() {
  std::string systemType;
  tpsP->getInput("flow/equation_system", systemType, std::string("navier-stokes"));
  if (systemType == "euler") {
    config.eqSystem = EULER;
  } else if (systemType == "navier-stokes") {
    config.eqSystem = NS;
  } else if (systemType == "navier-stokes-passive") {
    config.eqSystem = NS_PASSIVE;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown equation_system -> %s", systemType.c_str());
    exit(ERROR);
  }
}

void M2ulPhyS::parsePlasmaModels() {
  tpsP->getInput("plasma_models/ambipolar", config.ambipolar, false);
  tpsP->getInput("plasma_models/two_temperature", config.twoTemperature, false);

  if (config.twoTemperature) {
    tpsP->getInput("plasma_models/electron_temp_ic", config.initialElectronTemperature, 300.0);
  } else {
    config.initialElectronTemperature = -1;
  }

  std::string gasModelStr;
  tpsP->getInput("plasma_models/gas_model", gasModelStr, std::string("perfect_mixture"));
  if (gasModelStr == "perfect_mixture") {
    config.gasModel = PERFECT_MIXTURE;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown gas_model -> %s", gasModelStr.c_str());
    exit(ERROR);
  }

  std::string transportModelStr;
  tpsP->getRequiredInput("plasma_models/transport_model", transportModelStr);
  if (transportModelStr == "argon_minimal") {
    config.transportModel = ARGON_MINIMAL;
  } else if (transportModelStr == "argon_mixture") {
    config.transportModel = ARGON_MIXTURE;
  } else if (transportModelStr == "constant") {
    config.transportModel = CONSTANT;
  }
  // } else {
  //   grvy_printf(GRVY_ERROR, "\nUnknown transport_model -> %s", transportModelStr.c_str());
  //   exit(ERROR);
  // }

  std::string chemistryModelStr;
  tpsP->getInput("plasma_models/chemistry_model", chemistryModelStr, std::string(""));

  tpsP->getInput("plasma_models/const_plasma_conductivity", config.const_plasma_conductivity_, 0.0);

  // TODO(kevin): cantera wrapper
  // if (chemistryModelStr == "cantera") {
  //   config.chemistryModel_ = ChemistryModel::CANTERA;
  // }
}

void M2ulPhyS::parseSpeciesInputs() {
  // quick return if can't use these inputs
  if (config.workFluid == LTE_FLUID) {
    config.numSpecies = 1;
    return;
  }

  // Take parameters from input. These will be sorted into config attributes.
  DenseMatrix inputGasParams;
  Vector inputCV, inputCP;
  Vector inputInitialMassFraction;
  std::vector<std::string> inputSpeciesNames;
  DenseMatrix inputSpeciesComposition;

  // number of species defined
  if (config.eqSystem != NS_PASSIVE) {
    tpsP->getInput("species/numSpecies", config.numSpecies, 1);
    inputGasParams.SetSize(config.numSpecies, GasParams::NUM_GASPARAMS);
    inputInitialMassFraction.SetSize(config.numSpecies);
    inputSpeciesNames.resize(config.numSpecies);

    if (config.gasModel == PERFECT_MIXTURE) {
      inputCV.SetSize(config.numSpecies);
      // config.constantMolarCP.SetSize(config.numSpecies);
    }
  }

  
   // NOTE: for now, we force the user to set the background species as the last,
   // and the electron species as the second to last.
   
  config.numAtoms = 0;
  if ((config.numSpecies > 1) && (config.eqSystem != NS_PASSIVE)) {
    tpsP->getRequiredInput("species/background_index", config.backgroundIndex);

    tpsP->getRequiredInput("atoms/numAtoms", config.numAtoms);
    config.atomMW.SetSize(config.numAtoms);
    for (int a = 1; a <= config.numAtoms; a++) {
      std::string basepath("atoms/atom" + std::to_string(a));

      std::string atomName;
      tpsP->getRequiredInput((basepath + "/name").c_str(), atomName);
      tpsP->getRequiredInput((basepath + "/mass").c_str(), config.atomMW(a - 1));
      config.atomMap[atomName] = a - 1;
    }
  }

  // Gas Params
  if (config.workFluid != DRY_AIR) {
    assert(config.numAtoms > 0);
    inputSpeciesComposition.SetSize(config.numSpecies, config.numAtoms);
    inputSpeciesComposition = 0.0;
    for (int i = 1; i <= config.numSpecies; i++) {
      // double mw, charge;
      double formEnergy;
      std::string type, speciesName;
      std::vector<std::pair<std::string, std::string>> composition;
      std::string basepath("species/species" + std::to_string(i));

      tpsP->getRequiredInput((basepath + "/name").c_str(), speciesName);
      inputSpeciesNames[i - 1] = speciesName;

      tpsP->getRequiredPairs((basepath + "/composition").c_str(), composition);
      for (size_t c = 0; c < composition.size(); c++) {
        if (config.atomMap.count(composition[c].first)) {
          int atomIdx = config.atomMap[composition[c].first];
          inputSpeciesComposition(i - 1, atomIdx) = stoi(composition[c].second);
        } else {
          grvy_printf(GRVY_ERROR, "Requested atom %s for species %s is not available!\n", composition[c].first.c_str(),
                      speciesName.c_str());
        }
      }

      tpsP->getRequiredInput((basepath + "/formation_energy").c_str(), formEnergy);
      inputGasParams(i - 1, GasParams::FORMATION_ENERGY) = formEnergy;

      tpsP->getRequiredInput((basepath + "/initialMassFraction").c_str(), inputInitialMassFraction(i - 1));

      if (config.gasModel == PERFECT_MIXTURE) {
        tpsP->getRequiredInput((basepath + "/perfect_mixture/constant_molar_cv").c_str(), inputCV(i - 1));
        // NOTE: For perfect gas, CP will be automatically set from CV.
      }
    }

    Vector speciesMass(config.numSpecies), speciesCharge(config.numSpecies);
    inputSpeciesComposition.Mult(config.atomMW, speciesMass);
    inputSpeciesComposition.GetColumn(config.atomMap["E"], speciesCharge);
    speciesCharge *= -1.0;
    inputGasParams.SetCol(GasParams::SPECIES_MW, speciesMass);
    inputGasParams.SetCol(GasParams::SPECIES_CHARGES, speciesCharge);

    // Sort the gas params for mixture and save mapping.
    {
      config.gasParams.SetSize(config.numSpecies, GasParams::NUM_GASPARAMS);
      config.initialMassFractions.SetSize(config.numSpecies);
      config.speciesNames.resize(config.numSpecies);
      config.constantMolarCV.SetSize(config.numSpecies);

      config.speciesComposition.SetSize(config.numSpecies, config.numAtoms);
      config.speciesComposition = 0.0;

      // TODO(kevin): electron species is enforced to be included in the input file.
      bool isElectronIncluded = false;

      config.gasParams = 0.0;
      int paramIdx = 0;  // new species index.
      int targetIdx;
      for (int sp = 0; sp < config.numSpecies; sp++) {  // input file species index.
        if (sp == config.backgroundIndex - 1) {
          targetIdx = config.numSpecies - 1;
        } else if (inputSpeciesNames[sp] == "E") {
          targetIdx = config.numSpecies - 2;
          isElectronIncluded = true;
        } else {
          targetIdx = paramIdx;
          paramIdx++;
        }
        config.speciesMapping[inputSpeciesNames[sp]] = targetIdx;
        config.speciesNames[targetIdx] = inputSpeciesNames[sp];
        config.mixtureToInputMap[targetIdx] = sp;
        if (mpi.Root()) {
          std::cout << "name, input index, mixture index: " << config.speciesNames[targetIdx] << ", " << sp << ", "
                    << targetIdx << std::endl;
        }

        for (int param = 0; param < GasParams::NUM_GASPARAMS; param++)
          config.gasParams(targetIdx, param) = inputGasParams(sp, param);
      }
      assert(isElectronIncluded);

      for (int sp = 0; sp < config.numSpecies; sp++) {
        int inputIdx = config.mixtureToInputMap[sp];
        config.initialMassFractions(sp) = inputInitialMassFraction(inputIdx);
        config.constantMolarCV(sp) = inputCV(inputIdx);
        for (int a = 0; a < config.numAtoms; a++)
          config.speciesComposition(sp, a) = inputSpeciesComposition(inputIdx, a);
      }
    }
  }
}

void M2ulPhyS::parseTransportInputs() {
  switch (config.transportModel) {
    case ARGON_MINIMAL: {
      // Check if unsupported species are included.
      for (int sp = 0; sp < config.numSpecies; sp++) {
        if ((config.speciesNames[sp] != "Ar") && (config.speciesNames[sp] != "Ar.+1") &&
            (config.speciesNames[sp] != "E")) {
          grvy_printf(GRVY_ERROR, "\nArgon ternary mixture transport does not support the species: %s !",
                      config.speciesNames[sp].c_str());
          exit(ERROR);
        }
      }
      tpsP->getInput("plasma_models/transport_model/argon_minimal/third_order_thermal_conductivity",
                     config.thirdOrderkElectron, true);

      // pack up argon minimal transport input.
      {
        if (config.speciesMapping.count("Ar")) {
          config.argonTransportInput.neutralIndex = config.speciesMapping["Ar"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar' !\n");
          exit(ERROR);
        }
        if (config.speciesMapping.count("Ar.+1")) {
          config.argonTransportInput.ionIndex = config.speciesMapping["Ar.+1"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'Ar.+1' !\n");
          exit(ERROR);
        }
        if (config.speciesMapping.count("E")) {
          config.argonTransportInput.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
          exit(ERROR);
        }

        config.argonTransportInput.thirdOrderkElectron = config.thirdOrderkElectron;

        // inputs for artificial transport multipliers.
        {
          tpsP->getInput("plasma_models/transport_model/artificial_multiplier/enabled",
                         config.argonTransportInput.multiply, false);
          if (config.argonTransportInput.multiply) {
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency",
                           config.argonTransportInput.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/diffusivity",
                           config.argonTransportInput.diffMult, 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/mobility",
                           config.argonTransportInput.mobilMult, 1.0);
          }
        }
      }
    } break;
    case ARGON_MIXTURE: {
      tpsP->getInput("plasma_models/transport_model/argon_mixture/third_order_thermal_conductivity",
                     config.thirdOrderkElectron, true);

      // pack up argon transport input.
      {
        if (config.speciesMapping.count("E")) {
          config.argonTransportInput.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nArgon ternary transport requires the species 'E' !\n");
          exit(ERROR);
        }

        config.argonTransportInput.thirdOrderkElectron = config.thirdOrderkElectron;

        Array<ArgonSpcs> speciesType(config.numSpecies);
        identifySpeciesType(speciesType);
        identifyCollisionType(speciesType, config.argonTransportInput.collisionIndex);

        // inputs for artificial transport multipliers.
        {
          tpsP->getInput("plasma_models/transport_model/artificial_multiplier/enabled",
                         config.argonTransportInput.multiply, false);
          if (config.argonTransportInput.multiply) {
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/bulk_viscosity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::BULK_VISCOSITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/heavy_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/electron_thermal_conductivity",
                           config.argonTransportInput.fluxTrnsMultiplier[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/momentum_transfer_frequency",
                           config.argonTransportInput.spcsTrnsMultiplier[SpeciesTrns::MF_FREQUENCY], 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/diffusivity",
                           config.argonTransportInput.diffMult, 1.0);
            tpsP->getInput("plasma_models/transport_model/artificial_multiplier/mobility",
                           config.argonTransportInput.mobilMult, 1.0);
          }
        }
      }
    } break;
    case CONSTANT: {
      tpsP->getRequiredInput("plasma_models/transport_model/constant/viscosity", config.constantTransport.viscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/bulk_viscosity",
                             config.constantTransport.bulkViscosity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/thermal_conductivity",
                             config.constantTransport.thermalConductivity);
      tpsP->getRequiredInput("plasma_models/transport_model/constant/electron_thermal_conductivity",
                             config.constantTransport.electronThermalConductivity);
      std::string diffpath("plasma_models/transport_model/constant/diffusivity");
      // config.constantTransport.diffusivity.SetSize(config.numSpecies);
      std::string mtpath("plasma_models/transport_model/constant/momentum_transfer_frequency");
      // config.constantTransport.mtFreq.SetSize(config.numSpecies);
      for (int sp = 0; sp < config.numSpecies; sp++) {
        config.constantTransport.diffusivity[sp] = 0.0;
        config.constantTransport.mtFreq[sp] = 0.0;
      }
      for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index.
        int inputSp = config.mixtureToInputMap[sp];
        tpsP->getRequiredInput((diffpath + "/species" + std::to_string(inputSp + 1)).c_str(),
                               config.constantTransport.diffusivity[sp]);
        if (config.twoTemperature)
          tpsP->getRequiredInput((mtpath + "/species" + std::to_string(inputSp + 1)).c_str(),
                                 config.constantTransport.mtFreq[sp]);
      }
      config.constantTransport.electronIndex = -1;
      if (config.IsTwoTemperature()) {
        if (config.speciesMapping.count("E")) {
          config.constantTransport.electronIndex = config.speciesMapping["E"];
        } else {
          grvy_printf(GRVY_ERROR, "\nConstant transport: two-temperature plasma requires the species 'E' !\n");
          exit(ERROR);
        }
      }
    } break;
    default:
      break;
  }
}

void M2ulPhyS::parseReactionInputs() {
  tpsP->getInput("reactions/number_of_reactions", config.numReactions, 0);
  if (config.numReactions > 0) {
    assert((config.workFluid != DRY_AIR) && (config.numSpecies > 1));
    config.reactionEnergies.SetSize(config.numReactions);
    config.reactionEquations.resize(config.numReactions);
    config.reactionModels.SetSize(config.numReactions);
    config.reactantStoich.SetSize(config.numSpecies, config.numReactions);
    config.productStoich.SetSize(config.numSpecies, config.numReactions);

    // config.reactionModelParams.resize(config.numReactions);
    config.detailedBalance.SetSize(config.numReactions);
    // config.equilibriumConstantParams.resize(config.numReactions);
  }
  config.rxnModelParamsHost.clear();

  for (int r = 1; r <= config.numReactions; r++) {
    std::string basepath("reactions/reaction" + std::to_string(r));

    std::string equation, model;
    tpsP->getRequiredInput((basepath + "/equation").c_str(), equation);
    config.reactionEquations[r - 1] = equation;
    tpsP->getRequiredInput((basepath + "/model").c_str(), model);

    double energy;
    tpsP->getRequiredInput((basepath + "/reaction_energy").c_str(), energy);
    config.reactionEnergies[r - 1] = energy;

    // NOTE: reaction inputs are stored directly into ChemistryInput.
    // Initialize the pointers with null.
    config.chemistryInput.reactionInputs[r - 1].modelParams = NULL;

    if (model == "arrhenius") {
      config.reactionModels[r - 1] = ARRHENIUS;

      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.rxnModelParamsHost.push_back(Vector({A, b, E}));

    } else if (model == "hoffert_lien") {
      config.reactionModels[r - 1] = HOFFERTLIEN;
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/arrhenius/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/arrhenius/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/arrhenius/E").c_str(), E);
      config.rxnModelParamsHost.push_back(Vector({A, b, E}));

    } else if (model == "tabulated") {
      config.reactionModels[r - 1] = TABULATED_RXN;
      std::string inputPath(basepath + "/tabulated");
      readTable(inputPath, config.chemistryInput.reactionInputs[r - 1].tableInput);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown reaction_model -> %s", model.c_str());
      exit(ERROR);
    }

    bool detailedBalance;
    tpsP->getInput((basepath + "/detailed_balance").c_str(), detailedBalance, false);
    config.detailedBalance[r - 1] = detailedBalance;
    if (detailedBalance) {
      double A, b, E;
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/A").c_str(), A);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/b").c_str(), b);
      tpsP->getRequiredInput((basepath + "/equilibrium_constant/E").c_str(), E);

      // NOTE(kevin): this array keeps max param in the indexing, as reactions can have different number of params.
      config.equilibriumConstantParams[0 + (r - 1) * gpudata::MAXCHEMPARAMS] = A;
      config.equilibriumConstantParams[1 + (r - 1) * gpudata::MAXCHEMPARAMS] = b;
      config.equilibriumConstantParams[2 + (r - 1) * gpudata::MAXCHEMPARAMS] = E;
    }

    Array<double> stoich(config.numSpecies);
    tpsP->getRequiredVec((basepath + "/reactant_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) {
      int inputSp = config.mixtureToInputMap[sp];
      config.reactantStoich(sp, r - 1) = stoich[inputSp];
    }
    tpsP->getRequiredVec((basepath + "/product_stoichiometry").c_str(), stoich, config.numSpecies);
    for (int sp = 0; sp < config.numSpecies; sp++) {
      int inputSp = config.mixtureToInputMap[sp];
      config.productStoich(sp, r - 1) = stoich[inputSp];
    }
  }

  // check conservations.
  {
    const int numReactions_ = config.numReactions;
    const int numSpecies_ = config.numSpecies;
    for (int r = 0; r < numReactions_; r++) {
      Vector react(numSpecies_), product(numSpecies_);
      config.reactantStoich.GetColumn(r, react);
      config.productStoich.GetColumn(r, product);

      // atom conservation.
      DenseMatrix composition;
      composition.Transpose(config.speciesComposition);
      Vector reactAtom(config.numAtoms), prodAtom(config.numAtoms);
      composition.Mult(react, reactAtom);
      composition.Mult(product, prodAtom);
      for (int a = 0; a < config.numAtoms; a++) {
        if (reactAtom(a) != prodAtom(a)) {
          grvy_printf(GRVY_ERROR, "Reaction %d does not conserve atom %d.\n", r, a);
          exit(-1);
        }
      }

      // mass conservation. (already ensured with atom but checking again.)
      double reactMass = 0.0, prodMass = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        // int inputSp = (*mixtureToInputMap_)[sp];
        reactMass += react(sp) * config.gasParams(sp, SPECIES_MW);
        prodMass += product(sp) * config.gasParams(sp, SPECIES_MW);
      }
      // This may be too strict..
      if (abs(reactMass - prodMass) > 1.0e-15) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve mass.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E =/= %.8E\n", reactMass, prodMass);
        exit(-1);
      }

      // energy conservation.
      // TODO(kevin): this will need an adjustion when radiation comes into play.
      double reactEnergy = 0.0, prodEnergy = 0.0;
      for (int sp = 0; sp < numSpecies_; sp++) {
        // int inputSp = (*mixtureToInputMap_)[sp];
        reactEnergy += react(sp) * config.gasParams(sp, FORMATION_ENERGY);
        prodEnergy += product(sp) * config.gasParams(sp, FORMATION_ENERGY);
      }
      // This may be too strict..
      if (reactEnergy + config.reactionEnergies[r] != prodEnergy) {
        grvy_printf(GRVY_ERROR, "Reaction %d does not conserve energy.\n", r);
        grvy_printf(GRVY_ERROR, "%.8E + %.8E = %.8E =/= %.8E\n", reactEnergy, config.reactionEnergies[r],
                    reactEnergy + config.reactionEnergies[r], prodEnergy);
        exit(-1);
      }
    }
  }

  // Pack up chemistry input for instantiation.
  {
    config.chemistryInput.model = config.chemistryModel_;
    const int numSpecies = config.numSpecies;
    if (config.speciesMapping.count("E")) {
      config.chemistryInput.electronIndex = config.speciesMapping["E"];
    } else {
      config.chemistryInput.electronIndex = -1;
    }

    int rxn_param_idx = 0;

    config.chemistryInput.numReactions = config.numReactions;
    for (int r = 0; r < config.numReactions; r++) {
      config.chemistryInput.reactionEnergies[r] = config.reactionEnergies[r];
      config.chemistryInput.detailedBalance[r] = config.detailedBalance[r];
      for (int sp = 0; sp < config.numSpecies; sp++) {
        config.chemistryInput.reactantStoich[sp + r * numSpecies] = config.reactantStoich(sp, r);
        config.chemistryInput.productStoich[sp + r * numSpecies] = config.productStoich(sp, r);
      }

      config.chemistryInput.reactionModels[r] = config.reactionModels[r];
      for (int p = 0; p < gpudata::MAXCHEMPARAMS; p++) {
        config.chemistryInput.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS] =
            config.equilibriumConstantParams[p + r * gpudata::MAXCHEMPARAMS];
      }

      if (config.reactionModels[r] != TABULATED_RXN) {
        assert(rxn_param_idx < config.rxnModelParamsHost.size());
        config.chemistryInput.reactionInputs[r].modelParams = config.rxnModelParamsHost[rxn_param_idx].Read();
        rxn_param_idx += 1;
      }
    }
  }
}
*/

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

/*
void M2ulPhyS::parseSpongeZoneInputs() {
  tpsP->getInput("spongezone/numSpongeZones", config.numSpongeRegions_, 0);

  if (config.numSpongeRegions_ > 0) {
    config.spongeData_ = new SpongeZoneData[config.numSpongeRegions_];

    for (int sz = 0; sz < config.numSpongeRegions_; sz++) {
      std::string base("spongezone" + std::to_string(sz + 1));

      std::string type;
      tpsP->getInput((base + "/type").c_str(), type, std::string("none"));

      if (type == "none") {
        grvy_printf(GRVY_ERROR, "\nUnknown sponge zone type -> %s\n", type.c_str());
        exit(ERROR);
      } else if (type == "planar") {
        config.spongeData_[sz].szType = SpongeZoneType::PLANAR;
      } else if (type == "annulus") {
        config.spongeData_[sz].szType = SpongeZoneType::ANNULUS;

        tpsP->getInput((base + "/r1").c_str(), config.spongeData_[sz].r1, 0.);
        tpsP->getInput((base + "/r2").c_str(), config.spongeData_[sz].r2, 0.);
      }

      config.spongeData_[sz].normal.SetSize(3);
      config.spongeData_[sz].point0.SetSize(3);
      config.spongeData_[sz].pointInit.SetSize(3);
      config.spongeData_[sz].targetUp.SetSize(5 + config.numSpecies + 2);  // always large enough now

      tpsP->getRequiredVec((base + "/normal").c_str(), config.spongeData_[sz].normal, 3);
      tpsP->getRequiredVec((base + "/p0").c_str(), config.spongeData_[sz].point0, 3);
      tpsP->getRequiredVec((base + "/pInit").c_str(), config.spongeData_[sz].pointInit, 3);
      tpsP->getInput((base + "/tolerance").c_str(), config.spongeData_[sz].tol, 1e-5);
      tpsP->getInput((base + "/multiplier").c_str(), config.spongeData_[sz].multFactor, 1.0);

      tpsP->getRequiredInput((base + "/targetSolType").c_str(), type);
      if (type == "userDef") {
        config.spongeData_[sz].szSolType = SpongeZoneSolution::USERDEF;
        auto hup = config.spongeData_[sz].targetUp.HostWrite();
        tpsP->getRequiredInput((base + "/density").c_str(), hup[0]);   // rho
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[1], 0);  // u
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[2], 1);  // v
        tpsP->getRequiredVecElem((base + "/uvw").c_str(), hup[3], 2);  // w
        tpsP->getRequiredInput((base + "/pressure").c_str(), hup[4]);  // P

        // For multi-component gas, require (numActiveSpecies)-more inputs.
        if (config.workFluid != DRY_AIR) {
          if (config.numSpecies > 1) {
            grvy_printf(GRVY_INFO, "\nInlet mass fraction of background species will not be used. \n");
            if (config.ambipolar) grvy_printf(GRVY_INFO, "\nInlet mass fraction of electron will not be used. \n");

            for (int sp = 0; sp < config.numSpecies; sp++) {  // mixture species index.
              // read mass fraction of species as listed in the input file.
              int inputSp = config.mixtureToInputMap[sp];
              std::string speciesBasePath(base + "/mass_fraction/species" + std::to_string(inputSp + 1));
              tpsP->getRequiredInput(speciesBasePath.c_str(), hup[5 + sp]);
            }
          }

          if (config.twoTemperature) {
            tpsP->getInput((base + "/single_temperature").c_str(), config.spongeData_[sz].singleTemperature, false);
            if (!config.spongeData_[sz].singleTemperature) {
              tpsP->getRequiredInput((base + "/electron_temperature").c_str(), hup[5 + config.numSpecies]);  // P
            }
          }
        }

      } else if (type == "mixedOut") {
        config.spongeData_[sz].szSolType = SpongeZoneSolution::MIXEDOUT;
      } else {
        grvy_printf(GRVY_ERROR, "\nUnknown sponge zone type -> %s\n", type.c_str());
        exit(ERROR);
      }
    }
  }
}
*/

void LoMachSolver::parsePostProcessVisualizationInputs() {
  if (tpsP_->isVisualizationMode()) {
    tpsP_->getRequiredInput("post-process/visualization/prefix", config.postprocessInput.prefix);
    tpsP_->getRequiredInput("post-process/visualization/start-iter", config.postprocessInput.startIter);
    tpsP_->getRequiredInput("post-process/visualization/end-iter", config.postprocessInput.endIter);
    tpsP_->getRequiredInput("post-process/visualization/frequency", config.postprocessInput.freq);
  }
}

/*
void M2ulPhyS::parseRadiationInputs() {
  std::string type;
  std::string basepath("plasma_models/radiation_model");

  tpsP->getInput(basepath.c_str(), type, std::string("none"));
  std::string modelInputPath(basepath + "/" + type);

  if (type == "net_emission") {
    config.radiationInput.model = NET_EMISSION;

    std::string coefficientType;
    tpsP->getRequiredInput((modelInputPath + "/coefficient").c_str(), coefficientType);
    if (coefficientType == "tabulated") {
      config.radiationInput.necModel = TABULATED_NEC;
      std::string inputPath(modelInputPath + "/tabulated");
      readTable(inputPath, config.radiationInput.necTableInput);
    } else {
      grvy_printf(GRVY_ERROR, "\nUnknown net emission coefficient type -> %s\n", coefficientType.c_str());
      exit(ERROR);
    }
  } else if (type == "none") {
    config.radiationInput.model = NONE_RAD;
  } else {
    grvy_printf(GRVY_ERROR, "\nUnknown radiation model -> %s\n", type.c_str());
    exit(ERROR);
  }
}
*/

void LoMachSolver::parsePeriodicInputs() {
  tpsP_->getInput("periodicity/enablePeriodic", config.periodic, false);
  tpsP_->getInput("periodicity/xTrans", config.xTrans, 1.0e12);
  tpsP_->getInput("periodicity/yTrans", config.yTrans, 1.0e12);
  tpsP_->getInput("periodicity/zTrans", config.zTrans, 1.0e12);
}

/*
void M2ulPhyS::packUpGasMixtureInput() {
  if (config.workFluid == DRY_AIR) {
    config.dryAirInput.f = config.workFluid;
    config.dryAirInput.eq_sys = config.eqSystem;
#if defined(_HIP_)
    config.dryAirInput.visc_mult = config.visc_mult;
    config.dryAirInput.bulk_visc_mult = config.bulk_visc;
#endif
  } else if (config.workFluid == USER_DEFINED) {
    switch (config.gasModel) {
      case PERFECT_MIXTURE: {
        config.perfectMixtureInput.f = config.workFluid;
        config.perfectMixtureInput.numSpecies = config.numSpecies;
        if (config.speciesMapping.count("E")) {
          config.perfectMixtureInput.isElectronIncluded = true;
        } else {
          config.perfectMixtureInput.isElectronIncluded = false;
        }
        config.perfectMixtureInput.ambipolar = config.ambipolar;
        config.perfectMixtureInput.twoTemperature = config.twoTemperature;

        for (int sp = 0; sp < config.numSpecies; sp++) {
          for (int param = 0; param < (int)GasParams::NUM_GASPARAMS; param++) {
            config.perfectMixtureInput.gasParams[sp + param * config.numSpecies] = config.gasParams(sp, param);
          }
          config.perfectMixtureInput.molarCV[sp] = config.constantMolarCV(sp);
        }
      } break;
      default:
        grvy_printf(GRVY_ERROR, "Gas model is not specified!\n");
        exit(ERROR);
        break;
    }  // switch gasModel
  } else if (config.workFluid == LTE_FLUID) {
    config.lteMixtureInput.f = config.workFluid;
    // config.lteMixtureInput.thermo_file_name // already set in parseFluidPreset
    assert(config.numSpecies == 1);  // inconsistent to specify lte and carry species
    assert(!config.twoTemperature);  // inconsistent to specify lte and have two temperatures
  }
}
*/


/// HERE HERE HERE
void LoMachSolver::initSolutionAndVisualizationVectors() {

  //std::cout << " In initSol&Viz..." << endl;
  visualizationVariables_.clear();
  //std::cout << " check 1..." << endl;  
  visualizationNames_.clear();
  //std::cout << " check 2..." << endl;    

  offsets = new Array<int>(num_equation + 1);
  //std::cout << " check 2a..." << endl;      
  for (int k = 0; k <= num_equation; k++) {
    (*offsets)[k] = k * fvfes->GetNDofs();
  }
  //std::cout << " check 3..." << endl;    
  u_block = new BlockVector(*offsets);
  //std::cout << " check 4..." << endl;    
  up_block = new BlockVector(*offsets);
  //std::cout << " check 5..." << endl;    

  // gradUp.SetSize(num_equation*dim*vfes->GetNDofs());
  //gradUp = new ParGridFunction(gradUpfes);

  U = new ParGridFunction(fvfes, u_block->HostReadWrite());
  //std::cout << " check 6..." << endl;    
  Up = new ParGridFunction(fvfes, up_block->HostReadWrite());
  //std::cout << " check 7..." << endl;
  
  //dens = new ParGridFunction(rfes, Up->HostReadWrite());
  //vel = new ParGridFunction(vfes, Up->HostReadWrite() + rfes->GetNDofs());
  //std::cout << " check 8..." << endl;    
  //temperature = new ParGridFunction(tfes, Up->HostReadWrite() + (1 + nvel) * tfes->GetNDofs()); // this may break...
  //std::cout << " check 9..." << endl;    
  //press = new ParGridFunction(pfes);
  //std::cout << " check 10..." << endl;    


  //if (config.isAxisymmetric()) {
  //  vtheta = new ParGridFunction(fes, Up->HostReadWrite() + 3 * fes->GetNDofs());
  //} else {
  //  vtheta = NULL;
  //}

  //electron_temp_field = NULL;
  //if (config.twoTemperature) {
  //  electron_temp_field = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());
  //}

  /*
  passiveScalar = NULL;
  if (eqSystem == NS_PASSIVE) {
    passiveScalar = new ParGridFunction(fes, Up->HostReadWrite() + (num_equation - 1) * fes->GetNDofs());
  } else {
    // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
    // will need to add full list of species.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      std::string speciesName = config.speciesNames[sp];
      visualizationVariables_.push_back(
          new ParGridFunction(fes, U->HostReadWrite() + (sp + nvel + 2) * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("partial_density_" + speciesName));
    }
  }
  */

  // add visualization variables if tps is run on post-process visualization mode.
  // TODO(kevin): maybe enable users to specify what to visualize.
  /*
  if (tpsP->isVisualizationMode()) {
    if (config.workFluid != DRY_AIR) {
      // species primitives.
      visualizationIndexes_.Xsp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("X_" + speciesName));
      }
      visualizationIndexes_.Ysp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("Y_" + speciesName));
      }
      visualizationIndexes_.nsp = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("n_" + speciesName));
      }

      // transport properties.
      visualizationIndexes_.FluxTrns = visualizationVariables_.size();
      for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++) {
        std::string fieldName;
        switch (t) {
          case FluxTrns::VISCOSITY:
            fieldName = "viscosity";
            break;
          case FluxTrns::BULK_VISCOSITY:
            fieldName = "bulk_viscosity";
            break;
          case FluxTrns::HEAVY_THERMAL_CONDUCTIVITY:
            fieldName = "thermal_cond_heavy";
            break;
          case FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY:
            fieldName = "thermal_cond_elec";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown flux transport property!");
            exit(ERROR);
            break;
        }
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(fieldName);
      }
      visualizationIndexes_.diffVel = visualizationVariables_.size();
      for (int sp = 0; sp < numSpecies; sp++) {
        std::string speciesName = config.speciesNames[sp];
        visualizationVariables_.push_back(new ParGridFunction(nvelfes));
        visualizationNames_.push_back(std::string("diff_vel_" + speciesName));
      }
      visualizationIndexes_.SrcTrns = visualizationVariables_.size();
      for (int t = 0; t < SrcTrns::NUM_SRC_TRANS; t++) {
        std::string fieldName;
        switch (t) {
          case SrcTrns::ELECTRIC_CONDUCTIVITY:
            fieldName = "electric_cond";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown source transport property!");
            exit(ERROR);
            break;
        }
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(fieldName);
      }
      visualizationIndexes_.SpeciesTrns = visualizationVariables_.size();
      for (int t = 0; t < SpeciesTrns::NUM_SPECIES_COEFFS; t++) {
        std::string fieldName;
        switch (t) {
          case SpeciesTrns::MF_FREQUENCY:
            fieldName = "momentum_tranfer_freq";
            break;
          default:
            grvy_printf(GRVY_ERROR, "Error in initializing visualization: Unknown species transport property!");
            exit(ERROR);
            break;
        }
        for (int sp = 0; sp < numSpecies; sp++) {
          std::string speciesName = config.speciesNames[sp];
          visualizationVariables_.push_back(new ParGridFunction(fes));
          visualizationNames_.push_back(std::string(fieldName + "_" + speciesName));
        }
      }

      // chemistry reaction rates.
      visualizationIndexes_.rxn = visualizationVariables_.size();
      for (int r = 0; r < config.numReactions; r++) {
        visualizationVariables_.push_back(new ParGridFunction(fes));
        visualizationNames_.push_back(std::string("rxn_rate_" + std::to_string(r + 1)));
        // visualizationNames_.push_back(std::string("rxn_rate: " + config.reactionEquations[r]));
      }
    }  // if (config.workFluid != DRY_AIR)
  }    // if tpsP->isVisualizationMode()

  // If mms, add conserved and exact solution.
#ifdef HAVE_MASA
  if (config.use_mms_ && config.mmsSaveDetails_) {
    // for visualization.
    masaUBlock_ = new BlockVector(*offsets);
    masaU_ = new ParGridFunction(vfes, masaUBlock_->HostReadWrite());
    masaRhs_ = new ParGridFunction(*U);

    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, U->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("U" + std::to_string(eq)));
    }
    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, masaU_->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("mms_U" + std::to_string(eq)));
    }
    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables_.push_back(new ParGridFunction(fes, masaRhs_->HostReadWrite() + eq * fes->GetNDofs()));
      visualizationNames_.push_back(std::string("RHS" + std::to_string(eq)));
    }
  }
#endif

  plasma_conductivity_ = NULL;
  if (tpsP->isFlowEMCoupled()) {
    plasma_conductivity_ = new ParGridFunction(fes);
    *plasma_conductivity_ = 0.0;
  }

  joule_heating_ = NULL;
  if (tpsP->isFlowEMCoupled()) {
    joule_heating_ = new ParGridFunction(fes);
    *joule_heating_ = 0.0;
  }
  */

  // define solution parameters for i/o
  /*
  ioData.registerIOFamily("Solution state variables", "/solution", U);
  ioData.registerIOVar("/solution", "density", 0);
  ioData.registerIOVar("/solution", "rho-u", 1);
  ioData.registerIOVar("/solution", "rho-v", 2);
  if (nvel == 3) {
    ioData.registerIOVar("/solution", "rho-w", 3);
    ioData.registerIOVar("/solution", "rho-E", 4);
  } else {
    ioData.registerIOVar("/solution", "rho-E", 3);
  }
  */

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

  
  /*
  // TODO(kevin): for now, keep the number of primitive variables same as conserved variables.
  // will need to add full list of species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // Only for NS_PASSIVE.
    if ((eqSystem == NS_PASSIVE) && (sp == 1)) break;

    std::string speciesName = config.speciesNames[sp];
    ioData.registerIOVar("/solution", "rho-Y_" + speciesName, sp + nvel + 2);
  }

  if (config.twoTemperature) {
    ioData.registerIOVar("/solution", "rhoE_e", num_equation - 1);
  }
  */

  // compute factor to multiply viscosity when this option is active
  spaceVaryViscMult = NULL;
  ParGridFunction coordsDof(vfes);
  pmesh->GetNodes(coordsDof);
  if (config.linViscData.isEnabled) {
    spaceVaryViscMult = new ParGridFunction(tfes);
    double *viscMult = spaceVaryViscMult->HostWrite();
    double wgt = 0.;
    for (int n = 0; n < tfes->GetNDofs(); n++) {
      auto hcoords = coordsDof.HostRead();  // get coords
      double coords[3];
      for (int d = 0; d < dim; d++) {
        coords[d] = hcoords[n + d * vfes->GetNDofs()];
      }
      viscSpongePlanar(coords, wgt);
      viscMult[n] = wgt;
    }
  }
  

  /*
  paraviewColl->SetCycle(0);
  //std::cout << " check 15..." << endl;      
  paraviewColl->SetTime(0.);
  //std::cout << " check 16..." << endl;      

  paraviewColl->RegisterField("dens", dens);
  paraviewColl->RegisterField("vel", vel);
  //std::cout << " check 17..." << endl;      
  paraviewColl->RegisterField("temp", temperature);
  //std::cout << " check 18..." << endl;      
  paraviewColl->RegisterField("press", press);
  //std::cout << " check 19..." << endl;
  */
  
  //if (config.isAxisymmetric()) {
  //  paraviewColl->RegisterField("vtheta", vtheta);
  //}
  
  //if (eqSystem == NS_PASSIVE) {
  //  paraviewColl->RegisterField("passiveScalar", passiveScalar);
  //}

  //if (config.twoTemperature) {
  //  paraviewColl->RegisterField("Te", electron_temp_field);
  //}

  /*
  for (int var = 0; var < visualizationVariables_.size(); var++) {
    paraviewColl->RegisterField(visualizationNames_[var], visualizationVariables_[var]);
  }
  //std::cout << " check 20..." << endl;    

  //if (spaceVaryViscMult != NULL) paraviewColl->RegisterField("viscMult", spaceVaryViscMult);
  //if (distance_ != NULL) paraviewColl->RegisterField("distance", distance_);

  paraviewColl->SetOwnData(true);
  //paraviewColl->Save(); 
  //std::cout << " check 21..." << endl;
  */
  
}


/**
Simple planar viscous sponge layer with smooth tanh-transtion using user-specified width and
total amplification.  Note: duplicate in M2
*/
MFEM_HOST_DEVICE void LoMachSolver::viscSpongePlanar(double *x, double &wgt) {
  double normal[3];
  double point[3];
  double s[3];
  double factor, width, dist;

  // get settings
  factor = max(vsd_.ratio, 1.0);
  width = vsd_.width;

  // distance from plane
  dist = 0.;
  for (int d = 0; d < dim; d++) s[d] = (x[d] - vsd_.p[d]);
  for (int d = 0; d < dim; d++) dist += s[d] * vsd_.n[d];

  // weight
  wgt = 0.5 * (tanh(dist / width - 2.0) + 1.0);
  wgt *= (factor - 1.0);
  wgt += 1.0;
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

   /**/
   u(0) = M;
   u(1) = 0.0;
   u(2) = 0.0;

   u(0) += 0.1 * M * A * cos(aL*x) * sin(bL*y) * sin(cL*z);
   u(1) += 0.1 * M * B * sin(aL*x) * cos(bL*y) * sin(cL*z);
   u(2) += 0.1 * M * C * sin(aL*x) * sin(bL*y) * cos(cL*z);

   u(0) -= 0.05 * M * A * cos(2.0*aL*x) * sin(2.0*bL*y) * sin(2.0*cL*z);
   u(1) -= 0.05 * M * B * sin(2.0*aL*x) * cos(2.0*bL*y) * sin(2.0*cL*z);
   u(2) -= 0.05 * M * C * sin(2.0*aL*x) * sin(2.0*bL*y) * cos(2.0*cL*z);

   u(0) += 0.025 * M * A * cos(4.0*aL*x) * sin(4.0*bL*y) * sin(4.0*cL*z);
   u(1) += 0.025 * M * B * sin(4.0*aL*x) * cos(4.0*bL*y) * sin(4.0*cL*z);
   u(2) += 0.025 * M * C * sin(4.0*aL*x) * sin(4.0*bL*y) * cos(4.0*cL*z);

   scl = std::max(1.0-std::pow((y-0.0)/0.2,4),0.0);
   u(0) = u(0) * scl;
   u(1) = u(1) * scl;
   u(2) = u(2) * scl;
   /**/

   //scl = std::max(1.0-std::pow((y-0.0)/0.2,4),0.0);
   //u(0) = scl;
   //u(1) = 0.0;
   //u(2) = 0.0;
   
}

void vel_channelTest(const Vector &coords, double t, Vector &u)
{
  
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);

   u(0) = 1.0-std::pow((y-0.0)/0.2,2);
   u(1) = 0.0;
   u(2) = 0.0;
   
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

double temp_wall(const Vector &coords, double t)  
{
   double Thi = 400.0;  
   double Tlo = 200.0;
   double x = coords(0);
   double y = coords(1);
   double z = coords(2);
   double temp;
   if(y > 0.0) {temp = Thi;}
   if(y < 0.0) {temp = Tlo;}   
   return temp;   
}

