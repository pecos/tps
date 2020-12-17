#ifndef EQUATION_OF_STATE
#define EQUATION_OF_STATE

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include "mfem.hpp"

using namespace mfem;
using namespace std;

enum WorkingFluid {DRY_AIR };

class EquationOfState
{
private:
  const WorkingFluid fluid;
  double specific_heat_ratio;
  double gas_constant;
public:
  EquationOfState(WorkingFluid _fluid = DRY_AIR );
  
  double ComputePressure(const Vector &state, int dim);
  
  // Compute the maximum characteristic speed.
  double ComputeMaxCharSpeed(const Vector &state, const int dim);
  
  // Physicality check (at end)
  bool StateIsPhysical(const Vector &state, const int dim);
  
  double GetSpecificHeatRatio(){ return specific_heat_ratio; }
  double GetGasConstant(){return gas_constant;}
};

#endif // EQUATION_OF_STATE
