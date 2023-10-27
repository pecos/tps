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
#ifndef RADIATION_HPP_
#define RADIATION_HPP_

#include <tps_config.h>
#include "mpi_groups.hpp"

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "table.hpp"
#include "tps_mfem_wrap.hpp"

#include "gslib_interpolator.hpp"

#include <limits>
#include <hdf5.h>

class Radiation {
 protected:
  int iter;
  double time;

 public:
  const RadiationInput inputs;

  MFEM_HOST_DEVICE Radiation(const RadiationInput &_inputs);

  MFEM_HOST_DEVICE virtual ~Radiation() {}

  virtual void SetCycle(int value) { iter = value; }
  virtual void SetTime(double value) { time = value; }

  // Currently has the minimal format required for NEC and P1 models.
  MFEM_HOST_DEVICE virtual double computeEnergySink(const double &T_h) {
    printf("computeEnergySink not implemented");
    assert(false);
    return 0;
  }

  virtual void Solve(ParGridFunction *_Temperature, ParGridFunction *_energySinkRad){
    printf("Solve function in Radiation class is not implemented");
    assert(false);
  }
};

class NetEmission : public Radiation {
 private:
  const double PI_ = PI;

  // NOTE(kevin): currently only takes tabulated data.
  LinearTable necTable_;

 public:
  MFEM_HOST_DEVICE NetEmission(const RadiationInput &_inputs);

  MFEM_HOST_DEVICE ~NetEmission();

  MFEM_HOST_DEVICE double computeEnergySink(const double &T_h) override { return -4.0 * PI_ * necTable_.eval(T_h); }
};





/** 
 * P1 groups - Class used by the P1 Radiation Model
 */
class P1Groups {
private:

  const double big = 1.0e+50;
  const double small = 1.0e-50;
  const double eps = std::numeric_limits<double>::epsilon();
  // const double small = eps;    

  int nprocs_;
  int rank_;
  bool rank0_;

  int dim;
  int dof;

  int NumOfGroups;
  int GroupID;

  int MaxIters;
  double P1_Tolerance;
  int PrintLevels;

  // reference to mesh
  ParMesh *mesh;

  // Finite element collection
  FiniteElementCollection *fec_rad;

  // Finite element space for a scalar (thermodynamic quantity)
  ParFiniteElementSpace *fes_rad;

  // Finite element space for a mesh-dim vector quantity (momentum)
  ParFiniteElementSpace *dfes_rad;

  // Finite element space for all variables together (total thermodynamic state)
  ParFiniteElementSpace *vfes_rad;


  const RadiationInput inputs;
  double rbc_a_val;
  std::vector<mfem::DenseMatrix > &tableHost; 
  bool axisymmetric;



  // Variables

  ParGridFunction *g;
  ParGridFunction *Temperature; 

  ParGridFunction *AbsorptionCoef; 
  ParGridFunction *oneOver3Absorp; 
  ParGridFunction *emittedRad; 

  ParGridFunction *Radius;
  ParGridFunction *OneOverRadius;


  GridFunctionCoefficient *RadiusCoeff; 
  GridFunctionCoefficient *OneOverRadiusCoeff; 


  GridFunctionCoefficient *rbcBCoef; 

  Coefficient *rf_coeff; 
  Coefficient *rk_coeff; 
  Coefficient *rb_coeff; 
  
  Coefficient *rm_rbcBCoef; 
  Coefficient *r_rbcACoef; 

  // Linear tables
  TableInput TableInputAbsorption;
  TableInput TableInputEmittedRad;

  LinearTable *linearTableAbsorption;
  LinearTable *linearTableEmittedRad;


  // FEM operators
  HYPRE_BigInt total_num_dofs;
  Array<int> boundary_dofs;
  Array<int> ess_tdof_list; // this list remains empty for pure Robin b.c.
  Array<int>* dbc_bdr;
  Array<int>* nbc_bdr;
  Array<int>* rbc_bdr;

  double dbc_val = 0.0;
  double nbc_val = 0.0;

  // Solver variables
  ParLinearForm *b;
  ParBilinearForm *a;

  HyprePCG *pcg; 
  HypreSolver *amg; 

  CGSolver *G_solver;    // Krylov solver for inverting the mass matrix A
  HypreSmoother *G_prec; // Preconditioner for the mass matrix A

  HypreParMatrix A;
  Vector B, X;

  MPI_Groups *groupsMPI;

public:
  P1Groups(ParMesh *_mesh, FiniteElementCollection *_fec, ParFiniteElementSpace *_fes, ParFiniteElementSpace *_dfes,
          const RadiationInput &inputs, double _rbc_a_val, std::vector<mfem::DenseMatrix> &_tableHost, bool _axisymmetric,
          ParGridFunction *_G, ParGridFunction *_Radius, ParGridFunction *_OneOverRadius, MPI_Groups *_groupsMPI);

  int getNumOfGroups() const { return NumOfGroups; }
  int getGroupID() const { return GroupID; }

  bool isAxisymmetric() const { return axisymmetric; }

  void setNumOfGroups(int value) { NumOfGroups = value; }
  void setGroupID(int value) { GroupID = value; }

  void setMaxIters(int value) { MaxIters = value; }
  void setPrintLevels(int value) { PrintLevels = value; }
  void setTolerance(double value) { P1_Tolerance = value; }


  virtual ~P1Groups();

  /// Update parameters and Solve P1 equations for the groups.
  void Solve(ParGridFunction *_Temperature);

  void CalcTemperatureDependentParameters();

  void readDataAndSetLinearTables();

  void readTable(const std::string &inputPath, const std::string &groupName, const std::string &datasetName, TableInput &result);

  double getValueEnergySink(int i) {return -((*emittedRad)[i] -  (*AbsorptionCoef)[i] * (*g)[i]);}
};


class P1Model : public Radiation {
 private:

  const double big = 1.0e+50;
  const double small = 1.0e-50;
  const double eps = std::numeric_limits<double>::epsilon();
  // const double small = eps;    

  int rank_;
  int nprocs_;
  bool rank0_;

  //! Maximum number of iterations
  int max_iters_;
  //! We call the radiation solver only every solve_rad_every_n steps
  int solve_rad_every_n_;
  //! Report timings every timing_freq_ steps
  int timing_freq_;

  int dim;
  int dof;

  // const double PI_ = 3.14159265358979323846264338327950288419717;
  const double PI_ = PI;

  // order of polynomials
  int order;

  // reference to mesh
  ParMesh *mesh;

  // Flow Solver
  FiniteElementCollection *fec; // Finite element collection 
  ParFiniteElementSpace *fes;   // Finite element space for a scalar (thermodynamic quantity)
  ParFiniteElementSpace *dfes;  // Finite element space for a mesh-dim vector quantity (momentum)
  ParFiniteElementSpace *vfes;  // Finite element space for all variables together (total thermodynamic state)


  // Radiation Solver
  FiniteElementCollection *fec_rad; // Finite element collection
  ParFiniteElementSpace *fes_rad;   // Finite element space for a scalar
  ParFiniteElementSpace *dfes_rad;  // Finite element space for a mesh-dim vector quantity 
  ParFiniteElementSpace *vfes_rad;  // Finite element space for all variables together

  // Inputs
  int NumOfGroups;  // Number of equations 
  bool axisymmetric;

  // Variables
  ParGridFunction *U;
  ParGridFunction *G;
  ParGridFunction *Temperature; 

  // The solution u has components for all groups considered.
  // These are stored contiguously in the BlockVector u_block.
  Array<int> *offsets_g;
  BlockVector *g_block;

  // Auxiliary Variables
  ParGridFunction *coordsDof;
  ParGridFunction *Radius;
  ParGridFunction *OneOverRadius;

  // Visualization functions
  ParGridFunction *energySink_P1, *energySink_NEC, *energySink_NEC_tps;
  ParGridFunction *energySinkB2B, *energySinkOptThin, *energySinkContinuum, *energySinkOptIntermediate;
  ParGridFunction *G_integratedRadiance;

  // paraview collection pointer
  ParaViewDataCollection *paraviewColl = NULL;
  bool visualization;

  std::vector<ParGridFunction *> Variables_G;
  std::vector<std::string> varNames_G;

  // this vector of DenseMatrix stores all the tables and provides a proper pointer for both gpu and cpu.
  // note that this will store all types of tabulated data.
  // Indexes of this vector therefore do not correspond exactly to rxn index or other quantities.
  std::vector<mfem::DenseMatrix> tableHost; 

  // Linear Solver Parameters
  HYPRE_BigInt total_num_dofs;
  Array<int> boundary_dofs;

  int MaxIters;
  double P1_Tolerance;
  int PrintLevels;

  // Groups
  P1Groups **P1Group;

  // Boundary conditions
  double WallEmissivity;
  double rbc_a_val;

  // Interpolator 
  FieldInterpolator *interp_flow_to_rad_;
  FieldInterpolator *interp_rad_to_flow_;

  // Tables
  TableInput TableInputOpThin;
  LinearTable *linearTableOpThin;

  TableInput TableInputContinuum;
  LinearTable *linearTableContinuum;

  TableInput TableInputNEC;
  LinearTable *linearTableNEC;

  TableInput TableInputNEC_tps;
  LinearTable *linearTableNEC_tps;


  MPI_Groups *groupsMPI;

 public:

  P1Model(const RadiationInput &_inputs, ParMesh *_mesh, FiniteElementCollection *_fec, 
          ParFiniteElementSpace *_fes, bool _axisymmetric, MPI_Groups *_groupsMPI);
  virtual ~P1Model();


  // Accessors
  int getNumOfGroups() const { return NumOfGroups; }
  int getMaximumIterations() const { return MaxIters; }

  ParMesh *GetMesh() { return mesh; }
  FiniteElementCollection *GetFEC() { return fec_rad; }
  ParFiniteElementSpace *GetScalarFES() { return fes_rad; }
  ParFiniteElementSpace *GetDimFESpace() { return dfes_rad; }
  ParFiniteElementSpace *GetFESpace() { return vfes_rad; }
  ParaViewDataCollection *GetParaviewColl() { return paraviewColl; }
  ParGridFunction *GetSolutionGF() { return U; }
  ParGridFunction *getTemperatureGF() { return Temperature; }

  // Setters
  void setMaximumIterations(int value) { MaxIters = value; }
  void setNumOfGroups(int value) { NumOfGroups = value; }


  /// Update parameters and Solve P1 equations for the groups.
  void setGToZero();
  void Solve(ParGridFunction *src_Temperature, ParGridFunction *src_energySinkRad);

  void readTabulatedData();
  void readTable(const std::string &inputPath, const std::string &groupName, 
                 const std::string &datasetName, TableInput &result);

  void initializeInterpolationData();


  // Functions
  MFEM_HOST_DEVICE double computeEnergySink(const double &T_h) override {return 0.0;}

  double computeEnergySinkNEC(const double &T_h) { return -4.0 * PI_ * linearTableNEC->eval(T_h); }
  double computeEnergySinkNEC_tps(const double &T_h) { return -4.0 * PI_ * linearTableNEC_tps->eval(T_h); }

  double computeOptThinRadContribution(const double &T_h) { return -1.0 * linearTableOpThin->eval(T_h); }
  double computeContinuumRadContribution(const double &T_h) { return -1.0 * linearTableContinuum->eval(T_h); }

  void findValuesOnWall();
};





#endif  // RADIATION_HPP_
