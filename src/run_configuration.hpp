#ifndef RUN_CONFIGURATION
#define RUN_CONFIGURATION

#include <utility>
#include <string>

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "inletBC.hpp"
#include "outletBC.hpp"
#include "wallBC.hpp"
#include <vector>

using namespace mfem;
using namespace std;

// Class to manage the run options as
// specified in the input file
// Lines begining with # are ignored
class RunConfiguration
{
private:
  // mesh file file
  string meshFile;
  int ref_levels;
  
  // output file name
  string outputFile;
  
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
  
  // variable/constant time-step
  bool constantTimeStep;
  double dt_fixed;
  
  // num iterations. Defaults to 0
  int numIters;
  
  // output interval in num. of iters.
  int itersOut;
  
  // Use Roe-type Riemann solver
  bool useRoe;
  
  // cycle for restart
  bool restart;
  bool restart_hdf5_conversion;   // read in older ascii format
  int restart_cycle;
  
  // Auxiliary order. A solution would be interpolated
  // from order POL_ORDER to AUX_ORDER and dumped
  int auxOrder;

  // Restart from solution of AUX_ORDER
  bool restartFromAux;
  
  // mean and RMS
  int sampleInterval;
  int startIter;
  bool restartMean;
  
  // working fluid. Options thus far
  // DRY_AIR
  // Defaults to DRAY_AIR
  WorkingFluid workFluid;
  
  // Factor by which the viscosity is multiplied.
  double visc_mult;
  
  // Reference length
  double refLength;
  
  // equations to be solved. Options thus far
  // EULER, NS, MHD
  Equations eqSystem;
  
  bool SBP;
  
  // initial constant field
  double initRhoRhoVp[5];
  
  bool isForcing;
  // Imposed pressure gradient
  double gradPress[3];
  
  // Inlet BC data
  Array<double> inletBC;
  std::vector<pair<int,InletType> > inletPatchType;
  
  
  // Outlet BC data
  Array<double> outletBC;
  std::vector< pair<int,OutletType> > outletPatchType;
  
  // Wall BC data
  Array<double> wallBC;
  std::vector<pair<int,WallType> > wallPatchType;
  
  // Resource management system - job monitoring
  bool   rm_enableMonitor_;      // flag to trigger RMS monitoring and job resubmissions
  int    rm_threshold_;          // remaining secs for current job to trigger re-submission
  int    rm_checkFrequency_;     // iteration frequency to check RMS status
  
public:
  RunConfiguration();
  ~RunConfiguration();
  
  void readInputFile(std::string inpuFileName);
  
  string GetMeshFileName(){return meshFile;}
  string GetOutputName(){return outputFile;}
  int GetUniformRefLevels(){return ref_levels;}

  int GetTimeIntegratorType(){return timeIntegratorType;}
  
  int GetSolutionOrder(){return solOrder;}
  int GetAuxSolOrder(){return auxOrder;}
  int GetIntegrationRule(){return integrationRule;}
  int GetBasisType(){return basisType;}
  
  double GetCFLNumber(){return cflNum;}
  bool isTimeStepConstant(){return constantTimeStep;}
  double GetFixedDT(){return dt_fixed;}
  int GetNumIters(){return numIters;}
  int GetNumItersOutput(){return itersOut;}
  bool RoeRiemannSolver(){return useRoe; }
  
  int GetMeanStartIter(){return startIter;}
  int GetMeanSampleInterval(){return sampleInterval;}
  bool GetRestartMean(){return restartMean;}
  
  WorkingFluid GetWorkingFluid(){return workFluid;}
  double GetViscMult(){return visc_mult;}
  double GetReferenceLength(){return refLength;}
  Equations GetEquationSystem(){return eqSystem;}
  bool isSBP(){return SBP;}
  double* GetConstantInitialCondition(){return &initRhoRhoVp[0];}

  // resource manager controls
  bool   isAutoRestart(){return rm_enableMonitor_;}
  int    rm_threshold() {return rm_threshold_;}
  int    rm_checkFreq() {return rm_checkFrequency_;}
  
  bool thereIsForcing(){return isForcing;}
  double* GetImposedPressureGradient(){return &gradPress[0];}
  
  int  GetRestartCycle(){return restart;}
  void SetRestartCycle(int iter){restart_cycle = iter; return;}
  bool RestartFromAux(){return restartFromAux;}
  bool RestartHDFConversion(){return restart_hdf5_conversion;}
  
  std::vector<pair<int,InletType> >* GetInletPatchType(){return &inletPatchType;}
  Array<double> GetInletData(int i);
  
  std::vector<pair<int,OutletType> >* GetOutletPatchType(){return &outletPatchType;}
  Array<double> GetOutletData(int out);
  
  std::vector<pair<int,WallType> >* GetWallPatchType(){return &wallPatchType;}
  Array<double> GetWallData(int w);
};

#endif // RUN_CONFIGURATION
