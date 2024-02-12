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

#ifndef LOMACH_HPP_
#define LOMACH_HPP_

/*
using namespace mfem;
struct viscositySpongeData {
  bool enabled;
  double n[3];
  double p[3];
  double ratio;
  double width;
};
*/

// forward-declaration for Tps support class
namespace TPS {
class Tps;
}

#include <hdf5.h>
#include <tps_config.h>

#include <iostream>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "argon_transport.hpp"
#include "averaging_and_rms.hpp"
#include "chemistry.hpp"
#include "flow.hpp"
#include "io.hpp"
#include "loMach_options.hpp"
#include "mfem.hpp"
#include "mfem/linalg/solvers.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "thermoChem.hpp"
#include "tps.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"
#include "turbModel.hpp"

using VecFuncT = void(const Vector &x, double t, Vector &u);
using ScalarFuncT = double(const Vector &x, double t);

/** Base class for lo-Mach solvers
 *
 */
class LoMachSolver : public TPS::Solver {
  // friend class TurbModel;
  // friend class ThermoChem;
  // friend class Flow;
 protected:
  LoMachOptions loMach_opts_;

  TurbModel *turbClass = nullptr;
  ThermoChem *tcClass = nullptr;
  Flow *flowClass = nullptr;

  MPI_Groups *groupsMPI = nullptr;
  int nprocs_;  // total number of MPI procs
  int rank_;    // local MPI rank
  bool rank0_;  // flag to indicate rank 0

  // Run options
  RunConfiguration config;

  // History file
  std::ofstream histFile;

  // Early terminatation flag
  int earlyExit = 0;

  // Number of dimensions
  int dim;
  int nvel;

  // Number of equations
  int num_equation;

  // Is it ambipolar?
  bool ambipolar = false;

  // Is it two-temperature?
  bool twoTemperature_ = false;

  // loMach options to run as incompressible
  bool constantViscosity = false;
  bool constantDensity = false;
  bool incompressibleSolve = false;
  bool pFilter = false;

  // Average handler
  Averaging *average;

  double dt;
  double time;
  int iter;

  // pointer to parent Tps class
  TPS::Tps *tpsP_ = nullptr;

  mfem::ParMesh *pmesh_ = nullptr;
  int dim_;
  int true_size_;
  Array<int> offsets_;

  // min/max element size
  double hmin, hmax;

  // domain extent
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  // space sizes;
  int Sdof, Pdof, Vdof, Ndof, NdofR0;
  int SdofInt, PdofInt, VdofInt, NdofInt, NdofR0Int;

  int MaxIters;
  double max_speed;
  double CFL;
  int numActiveSpecies, numSpecies;

  // exit status code;
  int exit_status_;

  // mapping from local to global element index
  int *locToGlobElem = nullptr;

  // just keep these saved for ease
  int numWalls, numInlets, numOutlets;

  /// Print information about the Navier version.
  void PrintInfo();

  /// Update the EXTk/BDF time integration coefficient.
  /**
   * Depending on which time step the computation is in, the EXTk/BDF time
   * integration coefficients have to be set accordingly. This allows
   * bootstrapping with a BDF scheme of order 1 and increasing the order each
   * following time step, up to order 3 (or whichever order is set in
   * SetMaxBDFOrder).
   */
  void SetTimeIntegrationCoefficients(int step);

  /// Eliminate essential BCs in an Operator and apply to RHS.
  // rename this to something sensible "ApplyEssentialBC" or something
  /*
   void EliminateRHS(Operator &A,
                     ConstrainedOperator &constrainedA,
                     const Array<int> &ess_tdof_list,
                     Vector &x,
                     Vector &b,
                     Vector &X,
                     Vector &B,
                     int copy_interior = 0);
  */

  /// Enable/disable debug output.
  bool debug = false;

  /// Enable/disable verbose output.
  bool verbose = true;

  /// Enable/disable partial assembly of forms.
  bool partial_assembly = false;
  bool partial_assembly_pressure = true;

  /// Enable/disable numerical integration rules of forms.
  // bool numerical_integ = false;
  bool numerical_integ = true;

  /// The parallel mesh.
  ParMesh *pmesh = nullptr;
  // Mesh *serial_mesh;

  /// The order of the velocity and pressure space.
  int order;
  int porder;
  int norder;

  // total number of mesh elements (serial)
  int nelemGlobal_;

  // original mesh partition info (stored on rank 0)
  Array<int> partitioning_;
  const int defaultPartMethod = 1;

  // temporary
  // double Re_tau, Pr, Cp, gamma;

  /// Kinematic viscosity (dimensionless).
  // double kin_vis, dyn_vis;

  // make everything dimensionless after generalizing form channel
  // double ambientPressure, thermoPressure, static_rho, Rgas, systemMass;
  // double tPm1, tPm2, dtP;

  IntegrationRules gll_rules;

  /// Velocity \f$H^1\f$ finite element collection.
  FiniteElementCollection *vfec = nullptr;

  /// Velocity \f$(H^1)^d\f$ finite element space.
  ParFiniteElementSpace *vfes = nullptr;

  //// total space for compatibility
  ParFiniteElementSpace *fvfes = nullptr;
  ParFiniteElementSpace *fvfes2 = nullptr;

  /// Pressure \f$H^1\f$ finite element collection.
  FiniteElementCollection *pfec = nullptr;

  /// Pressure \f$H^1\f$ finite element space.
  ParFiniteElementSpace *pfes = nullptr;

  /// Scalar \f$H^1\f$ finite element collection.
  FiniteElementCollection *sfec = nullptr;

  /// Scalar \f$H^1\f$ finite element space.
  ParFiniteElementSpace *sfes = nullptr;

  // nonlinear term
  FiniteElementCollection *nfec = nullptr;
  ParFiniteElementSpace *nfes = nullptr;
  FiniteElementCollection *nfecR0 = nullptr;
  ParFiniteElementSpace *nfesR0 = nullptr;

  ParFiniteElementSpace *HCurlFESpace = nullptr;

  /// Linear form to compute the mass matrix in various subroutines.
  ParLinearForm *mass_lf = nullptr;
  ConstantCoefficient onecoeff;
  double volume = 0.0;

  // TrueDof size
  // Vector viscSml;
  // Vector alphaSml;
  // Vector viscMultSml;
  // Vector subgridViscSml;
  Vector gridScaleSml;
  Vector gridScaleXSml;
  Vector gridScaleYSml;
  Vector gridScaleZSml;

  int max_bdf_order;  // input option now
  // int max_bdf_order = 3;
  // int max_bdf_order = 2;
  // int max_bdf_order = 1;
  int cur_step = 0;
  std::vector<double> dthist = {0.0, 0.0, 0.0};

  // BDFk/EXTk coefficients.
  double bd0 = 0.0;
  double bd1 = 0.0;
  double bd2 = 0.0;
  double bd3 = 0.0;
  double ab1 = 0.0;
  double ab2 = 0.0;
  double ab3 = 0.0;
  std::vector<double> abCoef = {ab1, ab2, ab3};
  std::vector<double> bdfCoef = {bd0, bd1, bd2, bd3};

  // Timers.
  StopWatch sw_setup, sw_step, sw_extrap, sw_curlcurl, sw_spsolve, sw_hsolve;

  // Equations solved
  Equations eqSystem;

  // order of polynomials for auxiliary solution
  bool loadFromAuxSol;

  // Pointers to the different classes
  GasMixture *mixture;    // valid on host
  GasMixture *d_mixture;  // valid on device, when available; otherwise = mixture

  // TransportProperties *transportPtr = NULL;  // valid on both host and device
  //  TransportProperties *d_transport = NULL;  // valid on device, when available; otherwise = transportPtr

  Chemistry *chemistry_ = NULL;
  Radiation *radiation_ = NULL;

  /// Distance to nearest no-slip wall
  ParGridFunction *distance_ = nullptr;

  // to interface with existing code
  ParGridFunction *U = nullptr;
  ParGridFunction *Up = nullptr;

  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets = nullptr;
  BlockVector *u_block = nullptr;
  BlockVector *up_block = nullptr;

  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;
  // DataCollection *visitColl = NULL;

  // Visualization functions (these are pointers to Up)
  ParGridFunction *temperature = nullptr;
  ParGridFunction *dens = nullptr;
  ParGridFunction *vel = nullptr;
  ParGridFunction *vtheta = nullptr;
  ParGridFunction *passiveScalar = nullptr;
  ParGridFunction *electron_temp_field = nullptr;
  ParGridFunction *press = nullptr;
  std::vector<ParGridFunction *> visualizationVariables_;
  std::vector<std::string> visualizationNames_;
  AuxiliaryVisualizationIndexes visualizationIndexes_;
  ParGridFunction *plasma_conductivity_ = nullptr;
  ParGridFunction *joule_heating_ = nullptr;

  // I/O organizer
  IODataOrganizer ioData;

  // space varying viscosity multiplier
  /*
   ParGridFunction *bufferViscMult;
   viscositySpongeData vsd_;
   ParGridFunction viscTotal_gf;
   ParGridFunction alphaTotal_gf;
  */

  // subgrid scale
  // ParGridFunction *bufferSubgridVisc;

  // for plotting
  // ParGridFunction viscTotal_gf;
  // ParGridFunction alphaTotal_gf;
  ParGridFunction resolution_gf;

  // grid information
  ParGridFunction *bufferGridScale = nullptr;
  ParGridFunction *bufferGridScaleX = nullptr;
  ParGridFunction *bufferGridScaleY = nullptr;
  ParGridFunction *bufferGridScaleZ = nullptr;

  ParGridFunction *bufferMeanUp = nullptr;

  bool channelTest;

 public:
  /// Initialize data structures, set FE space order and kinematic viscosity.
  /**
   * The ParMesh @a mesh can be a linear or curved parallel mesh. The @a order
   * of the finite element spaces is this algorithm is of equal order
   * \f$(P_N)^d P_N\f$ for velocity and pressure respectively. This means the
   * pressure is in discretized in the same space (just scalar instead of a
   * vector space) as the velocity.
   *
   * Kinematic viscosity (dimensionless) is set using @a kin_vis and
   * automatically converted to the Reynolds number. If you want to set the
   * Reynolds number directly, you can provide the inverse.
   */
  LoMachSolver(LoMachOptions loMach_opts, TPS::Tps *tps);
  virtual ~LoMachSolver() {}

  void initialize();
  void parseSolverOptions() override;
  void parseSolverOptions2();
  void parsePeriodicInputs();
  void parseFlowOptions();
  void parseTimeIntegrationOptions();
  void parseStatOptions();
  void parseIOSettings();
  void parseRMSJobOptions();
  void parseBCInputs();
  void parseICOptions();
  void parsePostProcessVisualizationInputs();
  void parseFluidPreset();
  void parseViscosityOptions();
  void initSolutionAndVisualizationVectors();
  void initialTimeStep();
  void solve();
  void updateU();
  void copyU();
  // void interpolateInlet();
  // void uniformInlet();
  void updateTimestep();
  void setTimestep();

  // i/o routines
  void restart_files_hdf5(string mode, string inputFileName = std::string());
  void write_restart_files_hdf5(hid_t file, bool serialized_write);
  void read_restart_files_hdf5(hid_t file, bool serialized_read);

  // subgrid scale models
  // void sgsSmag(const DenseMatrix &gradUp, double delta, double &nu_sgs);
  // void sgsSigma(const DenseMatrix &gradUp, double delta, double &nu_sgs);

  void projectInitialSolution();
  void writeHDF5(string inputFileName = std::string()) { restart_files_hdf5("write", inputFileName); }
  void readHDF5(string inputFileName = std::string()) { restart_files_hdf5("read", inputFileName); }
  void writeParaview(int iter, double time) {
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
  }

  // nonlinear product functions and helpers
  /*
   void multScalarScalar(Vector A, Vector B, Vector* C);
   void multScalarVector(Vector A, Vector B, Vector* C);
   void multVectorVector(Vector A, Vector B, Vector* C1, Vector* C2, Vector* C3);
   void multConstScalar(double A, Vector B, Vector* C);
   void multConstVector(double A, Vector B, Vector* C);
   void multScalarInvVector(Vector A, Vector B, Vector* C);
   void multScalarInvVectorIP(Vector A, Vector* C);
   void multConstScalarInv(double A, Vector B, Vector* C);
   void multScalarInvScalarIP(Vector A, Vector* C);
   void multScalarScalarIP(Vector A, Vector* C);
   void multScalarVectorIP(Vector A, Vector* C);
   void multConstScalarIP(double A, Vector* C);
   void multConstVectorIP(double A, Vector* C);
   void multConstScalarInvIP(double A, Vector* C);
   void dotVector(Vector A, Vector B, Vector* C);
  */

  // common routines used for multiple integration methods
  /*
   void extrapolateState(int current_step);
  //void updateDensity(double tStep);
   void updateGradients(double uStep);
   void updateGradientsOP(double uStep);
   void updateBC(int current_step);
  //void updateDiffusivity(bool bulkViscFlag);
  //void updateThermoP();
   void computeExplicitForcing();
   void computeExplicitUnsteady();
   void computeExplicitUnsteadyBDF();
   void computeExplicitConvection(double uStep);
   void computeExplicitConvectionOP(double uStep, bool extrap);
  //   void computeExplicitTempConvectionOP(bool extrap);
   void computeExplicitDiffusion();
   void computeImplicitDiffusion();
  //void computeQt();
  //void computeQtTO();
   void computeDtRho();
   void makeDivFree();
   void makeDivFreeOP();
   void makeDivFreeLessQt();
   void computeLaplace(Vector u, Vector* Lu);
  */

  /// Initialize forms, solvers and preconditioners.
  void Setup(double dt);

  /// Compute solution at the next time step t+dt.
  /**
   * This method can be called with the default value @a provisional which
   * always accepts the computed time step by automatically calling
   * UpdateTimestepHistory.
   *
   * If @a provisional is set to true, the solution at t+dt is not accepted
   * automatically and the application code has to call UpdateTimestepHistory
   * and update the @a time variable accordingly.
   *
   * The application code can check the provisional step by retrieving the
   * GridFunction with the method GetProvisionalVelocity. If the check fails,
   * it is possible to retry the step with a different time step by not
   * calling UpdateTimestepHistory and calling this method with the previous
   * @a time and @a cur_step.
   *
   * The method and parameter choices are based on [1].
   *
   * [1] D. Wang, S.J. Ruuth (2008) Variable step-size implicit-explicit
   * linear multistep methods for time-dependent partial differential
   * equations
   */
  // void Step(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);
  /*
   void curlcurlStep(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);
   void staggeredTimeStep(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);
   void deltaPStep(double &time, double dt, const int cur_step, const int start_step, bool provisional = false);
  */

  /// Return a pointer to the provisional velocity ParGridFunction.
  // ParGridFunction *GetProvisionalVelocity() { return &un_next_gf; }

  /// Return a pointer to the current velocity ParGridFunction.
  // ParGridFunction *GetCurrentVelocity() { return &un_gf; }

  /// Return a pointer to the current pressure ParGridFunction.
  // ParGridFunction *GetCurrentPressure() { return &pn_gf; }

  /// Return a pointer to the current temperature ParGridFunction.
  // ParGridFunction *GetCurrentTemperature() { return &Tn_gf; }

  /// Return a pointer to the current density ParGridFunction.
  // ParGridFunction *GetCurrentDensity() { return &rn_gf; }

  /// Return a pointer to the current total viscosity ParGridFunction.
  // ParGridFunction *GetCurrentTotalViscosity() { return &viscTotal_gf; }

  /// Return a pointer to the current total thermal diffusivity ParGridFunction.
  // ParGridFunction *GetCurrentTotalThermalDiffusivity() { return &alphaTotal_gf; }

  /// Return a pointer to the current total resolution ParGridFunction.
  ParGridFunction *GetCurrentResolution() { return &resolution_gf; }

  /// Return a pointer to the current temperature ParGridFunction.
  // ParGridFunction *GetCurrentDensity() { return &rn_gf; }

  LoMachOptions GetOptions() { return loMach_opts_; }

  /// Add a Dirichlet boundary condition to the velocity field.
  // void AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
  // void AddVelDirichletBC(VecFuncT *f, Array<int> &attr);
  // void AddVelDirichletBC(VectorGridFunctionCoefficient *coeff, Array<int> &attr);
  // void AddVel4PDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
  // void AddVel4PDirichletBC(VecFuncT *f, Array<int> &attr);

  /// Add a Dirichlet boundary condition to the pressure field.
  // void AddPresDirichletBC(Coefficient *coeff, Array<int> &attr);
  // void AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr);

  /// Add a Dirichlet boundary condition to the temperature field.
  /*
   void AddTempDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr);
   //void AddTempDirichletBC(VectorCoefficient *coeff, Array<int> &attr);
   //void AddTempDirichletBC(VecFuncT *f, Array<int> &attr);
   //void AddTempDirichletBC(Array<int> &coeff, Array<int> &attr);

   void AddQtDirichletBC(Coefficient *coeff, Array<int> &attr);
   void AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr);
  */

  /// Add an acceleration term to the RHS of the equation.
  /**
   * The VecFuncT @a f is evaluated at the current time t and extrapolated
   * together with the nonlinear parts of the Navier Stokes equation.
   */
  // void AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr);
  // void AddAccelTerm(VecFuncT *f, Array<int> &attr);

  /// Add an Boussinesq term to the RHS of the equation.
  // void AddGravityTerm(VectorCoefficient *coeff, Array<int> &attr);
  // void AddGravityTerm(VecFuncT *g, Array<int> &attr);

  /// Enable partial assembly for every operator.
  void EnablePA(bool pa) { partial_assembly = pa; }

  /// Enable numerical integration rules. This means collocated quadrature at
  /// the nodal points.
  void EnableNI(bool ni) { numerical_integ = ni; }

  /// Print timing summary of the solving routine.
  /**
   * The summary shows the timing in seconds in the first row of
   *
   * 1. SETUP: Time spent for the setup of all forms, solvers and
   *    preconditioners.
   * 2. STEP: Time spent computing a full time step. It includes all three
   *    solves.
   * 3. EXTRAP: Time spent for extrapolation of all forcing and nonlinear
   *    terms.
   * 4. CURLCURL: Time spent for computing the curl curl term in the pressure
   *    Poisson equation (see references for detailed explanation).
   * 5. PSOLVE: Time spent in the pressure Poisson solve.
   * 6. HSOLVE: Time spent in the Helmholtz solve.
   *
   * The second row shows a proportion of a column relative to the whole
   * time step.
   */
  void PrintTimingData();

  /*
   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^2\f$.
   void ComputeCurl2D(ParGridFunction &u,
                      ParGridFunction &cu,
                      bool assume_scalar = false);

   /// Compute \f$\nabla \times \nabla \times u\f$ for \f$u \in (H^1)^3\f$.
   void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu);

   void vectorGrad3D(ParGridFunction &u, ParGridFunction &gu, ParGridFunction &gv, ParGridFunction &gw);
   void scalarGrad3D(ParGridFunction &u, ParGridFunction &gu);

   void vectorGrad3DV(Vector u, Vector* gu, Vector* gv, Vector* gw);
   void scalarGrad3DV(Vector u, Vector* gu);
  */

  /// Remove mean from a Vector.
  /**
   * Modify the Vector @a v by subtracting its mean using
   * \f$v = v - \frac{\sum_i^N v_i}{N} \f$
   */
  // void Orthogonalize(Vector &v);

  /// Remove the mean from a ParGridFunction.
  /**
   * Modify the ParGridFunction @a v by subtracting its mean using
   * \f$ v = v - \int_\Omega \frac{v}{vol(\Omega)} dx \f$.
   */
  // void MeanZero(ParGridFunction &v);

  /// Rotate entries in the time step and solution history arrays.
  void UpdateTimestepHistory(double dt);

  /// Set the maximum order to use for the BDF method.
  void SetMaxBDFOrder(int maxbdforder) { max_bdf_order = maxbdforder; }

  /// Compute CFL
  // double ComputeCFL(ParGridFunction &u, double dt);
  double computeCFL(double dt);

  /// Enforce CFL
  // void EnforceCFL(double maxCFL, ParGridFunction &u, double &dt);

  /// Set the number of modes to cut off in the interpolation filter
  // void SetCutoffModes(int c) { filter_cutoff_modes = c; }

  // MFEM_HOST_DEVICE void viscSpongePlanar(double *x, double &wgt);

  /// Set the interpolation filter parameter @a a
  /**
   * If @a a is > 0, the filtering algorithm for the velocity field after every
   * time step from [1] is used. The parameter should be 0 > @a >= 1.
   *
   * [1] Paul Fischer, Julia Mullen (2001) Filter-based stabilization of
   * spectral element methods
   */
  // void SetFilterAlpha(double a) { filter_alpha = a; }
};

#endif  // LOMACH_HPP_

/// Transient incompressible Navier Stokes solver in a split scheme formulation.
/**
 * This implementation of a transient incompressible Navier Stokes solver uses
 * the non-dimensionalized formulation. The coupled momentum and
 * incompressibility equations are decoupled using the split scheme described in
 * [1]. This leads to three solving steps.
 *
 * 1. An extrapolation step for all nonlinear terms which are treated
 *    explicitly. This step avoids a fully coupled nonlinear solve and only
 *    requires a solve of the mass matrix in velocity space \f$M_v^{-1}\f$. On
 *    the other hand this introduces a CFL stability condition on the maximum
 *    timestep.
 *
 * 2. A Poisson solve \f$S_p^{-1}\f$.
 *
 * 3. A Helmholtz like solve \f$(M_v - \partial t K_v)^{-1}\f$.
 *
 * The numerical solver setup for each step are as follows.
 *
 * \f$M_v^{-1}\f$ is solved using CG with Jacobi as preconditioner.
 *
 * \f$S_p^{-1}\f$ is solved using CG with AMG applied to the low order refined
 * (LOR) assembled pressure Poisson matrix. To avoid assembling a matrix for
 * preconditioning, one can use p-MG as an alternative (NYI).
 *
 * \f$(M_v - \partial t K_v)^{-1}\f$ due to the CFL condition we expect the time
 * step to be small. Therefore this is solved using CG with Jacobi as
 * preconditioner. For large time steps a preconditioner like AMG or p-MG should
 * be used (NYI).
 *
 * Statements marked with NYI mean this feature is planned but Not Yet
 * Implemented.
 *
 * A detailed description is available in [1] section 4.2. The algorithm is
 * originated from [2].
 *
 * [1] Michael Franco, Jean-Sylvain Camier, Julian Andrej, Will Pazner (2020)
 * High-order matrix-free incompressible flow solvers with GPU acceleration and
 * low-order refined preconditioners (https://arxiv.org/abs/1910.03032)
 *
 * [2] A. G. Tomboulides, J. C. Y. Lee & S. A. Orszag (1997) Numerical
 * Simulation of Low Mach Number Reactive Flows
 */
