
#ifndef TURBMODEL_HPP
#define TURBMODEL_HPP

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <iostream>
//#include <hdf5.h>
#include <tps_config.h>

//#include "loMach_options.hpp"
#include "io.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "run_configuration.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "../utils/mfem_extras/pfem_extras.hpp"
#include "split_flow_base.hpp"
#include "thermo_chem_base.hpp"
#include "turb_model_base.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

class LoMachSolver;
class LoMachOptions;
struct temporalSchemeCoefficients;

/// Add some description here...
class TurbModel : public TurbModelBase {
  friend class LoMachSolver;
private:

   //TPS::Tps *tpsP_;
   //LoMachSolver *loMach_;
   LoMachOptions *loMach_opts_ = nullptr;

   //MPI_Groups *groupsMPI;
   int nprocs_;  // total number of MPI procs
   int rank;    // local MPI rank
   bool rank0;  // flag to indicate rank 0

   // Run options
   RunConfiguration *config_ = nullptr;
  
   // space sizes
   int Sdof, SdofInt;

   /// Enable/disable verbose output.
   bool verbose = true;
  
   ParMesh *pmesh_ = nullptr;
  
   // The order of the scalar spaces
   int order;

   int nvel, dim;

   // just keep these saved for ease
   int numWalls, numInlets, numOutlets;

   // loMach options to run as incompressible
   bool pFilter = false;  
  
   double dt;
   double time;
   int iter;

   // Coefficients necessary to take a time step (including dt).
   // Assumed to be externally managed and determined, so just get a
   // reference here.
   const temporalSchemeCoefficients &timeCoeff_;
  
   // Scalar \f$H^1\f$ finite element collection.
   FiniteElementCollection *sfec = nullptr;

   // Scalar \f$H^1\f$ finite element space.
   ParFiniteElementSpace *sfes = nullptr;  

   /// Velocity \f$H^1\f$ finite element collection.
   FiniteElementCollection *vfec = nullptr;

   /// Velocity \f$(H^1)^d\f$ finite element space.
   ParFiniteElementSpace *vfes = nullptr;

   ParGridFunction *gradU_gf = nullptr;
   ParGridFunction *gradV_gf = nullptr;
   ParGridFunction *gradW_gf = nullptr;      
   Vector gradU; 
   Vector gradV;
   Vector gradW;

   ParGridFunction *rn_gf = nullptr;        
   Vector rn;
  
   ParGridFunction *delta_gf = nullptr;
   Vector delta;
  
   Vector subgridVisc;
   ParGridFunction subgridVisc_gf;
  
public:
  TurbModel(mfem::ParMesh *pmesh, RunConfiguration *config, LoMachOptions *loMach_opts, temporalSchemeCoefficients &timeCoeff);  
  virtual ~TurbModel() {}

   void initializeSelf();
   void initializeExternal(ParGridFunction *gradU_gf_, ParGridFunction *gradV_gf_, ParGridFunction *gradW_gf_, ParGridFunction *delta_gf_);  
  
   void step();
  
   /// Initialize forms, solvers and preconditioners.
   void setup(double dt);  

   /// Return a pointer to the current temperature ParGridFunction.
   ParGridFunction *GetCurrentEddyViscosity() { return &subgridVisc_gf; }

   // subgrid scale models => move to turb model class
   void sgsSmag(const DenseMatrix &gradUp, double delta, double &nu_sgs);
   void sgsSigma(const DenseMatrix &gradUp, double delta, double &nu_sgs);  
  
  
};
#endif
