#ifndef RUN_CONFIGURATION
#define RUN_CONFIGURATION

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "inletBC.hpp"
#include "outletBC.hpp"

using namespace mfem;

// Class to manage the run options as
// specified in the input file
// Lines begining with # are ignored
class RunConfiguration
{
private:
  // mesh file file
  char *meshFile;
  
  // output file name
  char *outputFile;
  
  // time integrator. Possible values
  //  1: ForwardEulerSolver
  //  2: RK2Solver(1.0)
  //  3: RK3SSPSolver
  //  4: RK4Solver
  //  6: RK6Solver
  int timeIntegratorType;
  
  // order of the solution. Defaults to 4
  int solOrder;
  
  // Integration rule (quadrature)
  // 0 -> Gauss-Legendre
  // 1 -> Gauss-Lobatto
  int integrationRule;
  
  // Basis type
  // 0 -> Gauss-Legendre
  // 1 -> Gauss-Lobatto
  int basisType;
  
  double cflNum;
  
  // num iterations. Defaults to 0
  int numIters;
  
  // output interval in num. of iters.
  int itersOut;
  
  // working fluid. Options thus far
  // DRY_AIR
  // Defaults to DRAY_AIR
  WorkingFluid workFluid;
  
  // equations to be solved. Options thus far
  // EULER, NS, MHD
  Equations eqSystem;
  
  bool SBP;
  
  // initial constant field
  double initRhoRhoVp[5];
  
  // Inlet BC data
  Array<Array<double> > inletBC;
  Array<std::pair<int,InletType> > inletPatchType;
  
  
  // Outlet BC data
  Array<Array<double> > outletBC;
  Array<std::pair<int,OutletType> > outletPatchType;
  
public:
  RunConfiguration();
  ~RunConfiguration();
  
  void readInputFile(std::string inpuFileName);
  
  char* GetMeshFileName(){return meshFile;}
  char* GetOutputName(){return outputFile;}
  
  int GetTimeIntegratorType(){return timeIntegratorType;}
  
  int GetSolutionOrder(){return solOrder;}
  int GetIntegrationRule(){return integrationRule;}
  int GetBasisType(){return basisType;}
  
  double GetCFLNumber(){return cflNum;}
  int GetNumIters(){return numIters;}
  int GetNumItersOutput(){return itersOut;}
  WorkingFluid GetWorkingFluid(){return workFluid;}
  Equations GetEquationSystem(){return eqSystem;}
  bool isSBP(){return SBP;}
  double* GetConstantInitialCondition(){return &initRhoRhoVp[0];}
  
  Array<std::pair<int,InletType> >* GetInletPatchType(){return &inletPatchType;}
  Array<double>* GetInletData(int in){return &inletBC[in];}
  Array<std::pair<int,OutletType> >* GetOutletPatchType(){return &outletPatchType;}
  Array<double>* GetOutletData(int out){return &outletBC[out];}
};

#endif // RUN_CONFIGURATION
