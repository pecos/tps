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

#ifndef M2ULPHYS_HPP_
#define M2ULPHYS_HPP_
/*
 *  MFEM MULTY PHYSICS SOLVER (M2ulPhyS)
 * This class contains all necessary functions and classes
 * reuired to run example 18 without the need of having global
 * variables and functions.
 */

// forward declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <fstream>
#include <mfem/general/forall.hpp>

#include "BCintegrator.hpp"
#include "argon_transport.hpp"
#include "averaging.hpp"
#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "dgNonlinearForm.hpp"
#include "domain_integrator.hpp"
#include "equation_of_state.hpp"
#include "faceGradientIntegration.hpp"
#include "face_integrator.hpp"
#include "fluxes.hpp"
#include "gpu_constructor.hpp"
#include "gradNonLinearForm.hpp"
#include "io.hpp"
#include "logger.hpp"
#include "lte_mixture.hpp"
#include "lte_transport_properties.hpp"
#include "mixing_length_transport.hpp"
#include "mpi_groups.hpp"
#include "radiation.hpp"
#include "rhs_operator.hpp"
#include "riemann_solver.hpp"
#include "run_configuration.hpp"
#include "solver.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"
#include "utils.hpp"

#ifdef HAVE_MASA
#include "masa_handler.hpp"
#endif

using namespace mfem;
using namespace std;

namespace TPS {
class Tps2Boltzmann;
}

class M2ulPhyS : public TPS::PlasmaSolver {
 private:
  MPI_Groups *groupsMPI;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // pointer to parent Tps class
  TPS::Tps *tpsP;

  // Run options
  RunConfiguration config;

  // History file
  std::ofstream histFile;

  // Number of dimensions
  int dim;
  int nvel;

  // Number of species
  int numSpecies;

  // Number of active species that are part of conserved variables.
  int numActiveSpecies;

  // Is it ambipolar?
  bool ambipolar = false;

  // Is it two-temperature?
  bool twoTemperature_ = false;

  // Number of equations
  int num_equation;

  // order of polynomials
  int order;
  // total number of mesh elements (serial)
  int nelemGlobal_;

  // order of polynomials for auxiliary solution
  bool loadFromAuxSol;

  // Equations solved
  Equations eqSystem;

  // Maximum characteristic speed (updated by integrators)
  double max_char_speed;

  // reference to mesh
  ParMesh *mesh;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;

  // time integrator
  ODESolver *timeIntegrator;

  // Pointers to the different classes
  GasMixture *mixture;    // valid on host
  GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture

  TransportProperties *transportPtr = NULL;  // valid on both host and device
  // TransportProperties *d_transport = NULL;  // valid on device, when available; otherwise = transportPtr

  Chemistry *chemistry_ = NULL;

  Radiation *radiation_ = NULL;

  // space varying viscosity multiplier
  ParGridFunction *spaceVaryViscMult;

  /// Distance to nearest no-slip wall
  ParGridFunction *distance_;

  Fluxes *fluxClass;    // valid on host
  Fluxes *d_fluxClass;  // valid on device, when available; otherwise = fluxClass

  RHSoperator *rhsOperator;

  // Integration rule
  int intRuleType;  // 0: GaussLegendre; 1: GaussLobatto
  IntegrationRules *intRules;

  // Interpolant function
  // 0: GaussLegendre; 1: GaussLobatto
  int basisType;

  // Finite element collection
  // DG_FECollection *fec;
  // H1_FECollection *fec;
  FiniteElementCollection *fec;

  // Finite element space for a scalar (thermodynamic quantity)
  ParFiniteElementSpace *fes;

  // Finite element space for a mesh-dim vector quantity (momentum)
  ParFiniteElementSpace *dfes;

  // Finite element space for a nvel vector quantity. only for visualization (diffusion velocity).
  ParFiniteElementSpace *nvelfes;

  // Finite element space for all variables together (total thermodynamic state)
  ParFiniteElementSpace *vfes;

  // nodes IDs and indirection array
  const int maxIntPoints = gpudata::MAXINTPOINTS;  // corresponding to HEX face with p=5
  const int maxDofs = gpudata::MAXDOFS;            // corresponding to HEX with p=5

  /// Data required for residual evaluation in _GPU_ path
  precomputedIntegrationData gpu_precomputed_data_;

  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets;
  BlockVector *u_block;
  BlockVector *up_block;

  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;
  // DataCollection *visitColl = NULL;

  // Riemann Solver
  RiemannSolver *rsolver;

  // RHS operators
  // ParNonlinearForm *A;
  DGNonLinearForm *A;

  FaceIntegrator *faceIntegrator;

  MixedBilinearForm *Aflux;
  DomainIntegrator *domainIntegrator;

  // Boundary condition non-linear integrator
  BCintegrator *bcIntegrator;

  // Conservative variables
  ParGridFunction *U;

  // Primitive variables
  ParGridFunction *Up;

  // Visualization functions (these are pointers to Up)
  ParGridFunction *temperature, *dens, *vel, *vtheta, *passiveScalar;
  ParGridFunction *electron_temp_field;
  ParGridFunction *press;
  std::vector<ParGridFunction *> visualizationVariables_;
  std::vector<std::string> visualizationNames_;
  AuxiliaryVisualizationIndexes visualizationIndexes_;
  ParGridFunction *plasma_conductivity_;
  ParGridFunction *joule_heating_;

  // gradient of primitive variables
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  // ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;

  // Auxiliary grid function to store external reaction rates
  std::unique_ptr<ParGridFunction> externalReactionRates;

  // Average handler
  Averaging *average;

  // time variable
  double time;

  // time step
  double dt;

  // iteration
  int iter;

  // time of end of simulation
  double t_final;
  int MaxIters;

  // Courant-Friedrich-Levy condition
  double CFL;

  // Reference length
  double refLength;

  // minimum element size
  double hmin;

  // maximum element size
  double hmax;

  // domain extent
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  // exit status code;
  int exit_status_;

  // mapping from local to global element index
  int *locToGlobElem;

  // a serial mesh, finite element space, and grid function
  // for use if we want to write a serial file
  Mesh *serial_mesh;

  // I/O organizer
  IODataOrganizer ioData;

  std::list<mfem::DenseMatrix> tableHost_;

#ifdef HAVE_MASA
  VectorFunctionCoefficient *DenMMS_, *VelMMS_, *PreMMS_;
  VectorFunctionCoefficient *stateMMS_;
  std::vector<VectorConstantCoefficient *> componentWindow_;

  ParGridFunction *zeroU_;    // to compute L2 norm of exact solution via ComputeLpError.
  ParGridFunction *masaU_;    // for visualization of the exact solution.
  ParGridFunction *masaRhs_;  // for visualization of the right-hand side.
  BlockVector *zeroUBlock_, *masaUBlock_;
#endif

#ifdef _GPU_
  Array<int> loc_print;
#endif

  void getAttributesInPartition(Array<int> &local_attr);

  /** @brief Fill precomputedIntegrationData struct
   *
   *  The _GPU_ path required some quantities (e.g., basis function at
   *  all face quadrature points) to be evaluated and stored at the
   *  beginning of a simulation.  These quantities are evaluated by
   *  this function.
   */
  void initIndirectionArrays();

  /** @brief Fill BC part of precomputedIntegrationData struct
   *
   *  The _GPU_ path required some BC information to be evaluated and
   *  stored at the beginning of a simulation.  These quantities are
   *  set by this function, which is separate from
   *  initIndirectionArrays() b/c it must be called after the boundary
   *  conditions are initialized.
   */
  void initIndirectionBC();

  void initMixtureAndTransportModels();
  void initVariables();
  void initSolutionAndVisualizationVectors();
  void initialTimeStep();

  static void InitialConditionEulerVortex(const Vector &x, Vector &y);
  static void testInitialCondition(const Vector &x, Vector &y);
  // void dryAirUniformInitialConditions();
  void uniformInitialConditions();
  void initGradUp();
  void initilizeSpeciesFromLTE();

  // i/o routines
  void restart_files_hdf5(string mode, string inputFileName = std::string());
  void write_restart_files_hdf5(hid_t file, bool serialized_write);
  void read_restart_files_hdf5(hid_t file, bool serialized_read);

  void Check_NAN();
  bool Check_JobResubmit();
  bool Check_ExitEarly(int iter);
  void Cache_Paraview_Timesteps();

#ifdef HAVE_MASA
  void initMasaHandler();
  void projectExactSolution(const double _time, ParGridFunction *prjU);
  void initMMSCoefficients();
  void checkSolutionError(const double _time, const bool final = false);
#endif

 public:
  M2ulPhyS(string &inputFileName, TPS::Tps *tps);
  M2ulPhyS(TPS::Tps *tps);
  ~M2ulPhyS();

  void parseSolverOptions() override;
  void parseSolverOptions2();
  // NOTE(kevin): had to split parseSolverOptions2 to enforce style..
  void parseFlowOptions();
  void parseTimeIntegrationOptions();
  void parseStatOptions();
  void parseIOSettings();
  void parseRMSJobOptions();
  void parseHeatSrcOptions();
  void parseViscosityOptions();
  void parseMMSOptions();
  void parseICOptions();
  void parsePassiveScalarOptions();
  void parseFluidPreset();
  void parseSystemType();
  void parsePlasmaModels();
  void parseSpeciesInputs();
  void parseTransportInputs();
  void parseReactionInputs();
  void parseBCInputs();
  void parseSpongeZoneInputs();
  void parsePeriodicInputs();
  void parsePostProcessVisualizationInputs();
  void parseRadiationInputs();
  void parsePlaneDump();
  void readTableWrapper(std::string inputPath, TableInput &result);

  void packUpGasMixtureInput();
  void identifySpeciesType(Array<ArgonSpcs> &speciesType);
  void identifyCollisionType(const Array<ArgonSpcs> &speciesType, ArgonColl *collisionIndex);

  void checkSolverOptions() const;
  void projectInitialSolution();
  void writeHDF5(string inputFileName = std::string()) { restart_files_hdf5("write", inputFileName); }
  void readHDF5(string inputFileName = std::string()) { restart_files_hdf5("read", inputFileName); }
  void writeParaview(int iter, double time) {
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
  }

  void solve() override;
  void solveStep() override;
  void solveBegin() override;
  void solveEnd() override;
  void visualization() override;
  ParMesh *getMesh() const override { return mesh; }
  ParFiniteElementSpace *getFESpace() const override { return vfes; }
  const FiniteElementCollection *getFEC() const override { return fec; }

  ParGridFunction *getPlasmaConductivityGF() override { return plasma_conductivity_; }
  ParGridFunction *getJouleHeatingGF() override { return joule_heating_; }
  void evaluatePlasmaConductivityGF() override;

  void updateVisualizationVariables();

  // Accessors
  RHSoperator *getRHSoperator() { return rhsOperator; }
  ParFiniteElementSpace *GetScalarFES() { return fes; }
  ParFiniteElementSpace *GetVectorFES() { return dfes; }
  ParaViewDataCollection *GetParaviewColl() { return paraviewColl; }
  ParGridFunction *GetSolutionGF() { return U; }
  ParGridFunction *getPrimitiveGF() { return Up; }
  ParGridFunction *getGradientGF() { return gradUp; }
  ParGridFunction *getPressureGF() { return press; }
  IntegrationRules *getIntegrationRules() { return intRules; }
  RunConfiguration &GetConfig() { return config; }
  GasMixture *getMixture() { return mixture; }
  Chemistry *getChemistry() { return chemistry_; }

  const ParGridFunction *getDistanceFcn() { return distance_; }

  void updatePrimitives();

  static int Check_NaN_GPU(ParGridFunction *U, int lengthU, Array<int> &loc_print);
  void Check_Undershoot();

  void setConstantPlasmaConductivityGF() {
    ParGridFunction *coordsDof = new ParGridFunction(dfes);
    mesh->GetNodes(*coordsDof);
    mixture->SetConstantPlasmaConductivity(plasma_conductivity_, Up, coordsDof);
    delete coordsDof;
  }

  // tps2Boltzmann interface (implemented in M2ulPhyS2Boltzmann.cpp)
  /// Push solver variables to interface
  void push(TPS::Tps2Boltzmann &interface) override;
  /// Fetch solver variables from interface
  void fetch(TPS::Tps2Boltzmann &interface) override;

  // Exit code access
  void SetStatus(int code) {
    exit_status_ = code;
    return;
  }
  int getStatus() override { return exit_status_; }

  int getMaximumIterations() const { return MaxIters; }
  int getCurrentIterations() const { return iter; }
  void setMaximumIterations(int value) { MaxIters = value; }
};

#endif  // M2ULPHYS_HPP_
