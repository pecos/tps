#ifndef EQUATION_OF_STATE
#define EQUATION_OF_STATE

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include "mfem.hpp"
#include "general/forall.hpp"

using namespace mfem;
using namespace std;

enum WorkingFluid {DRY_AIR };

class EquationOfState
{
private:
  WorkingFluid fluid;
  double specific_heat_ratio;
  double gas_constant;
  double visc_mult;
  double thermalConductivity;
  
  // Prandtl number
  double Pr;
public:
  EquationOfState();
  
  void setFluid(WorkingFluid _fluid);
  
  void setViscMult(double _visc_mult){visc_mult = _visc_mult;}
  
  double ComputePressure(const Vector &state, int dim);
  
  // Compute the maximum characteristic speed.
  double ComputeMaxCharSpeed(const Vector &state, const int dim);
  
  // Physicality check (at end)
  bool StateIsPhysical(const Vector &state, const int dim);
  
  double GetSpecificHeatRatio(){ return specific_heat_ratio; }
  double GetGasConstant(){return gas_constant;}
  double GetViscosity(const double &temp);
  double GetPrandtlNum(){return Pr;}
  double GetViscMultiplyer(){return visc_mult;}
  double GetThermalConductivity(const double &visc);
  
  // GPU functions
#ifdef _GPU_
  static MFEM_HOST_DEVICE double pressure(const double *state, 
                                          double *KE, 
                                          const double gamma, 
                                          const int dim, 
                                          const int num_equations)
  {
    double p = 0.;
    for(int k=0;k<dim;k++) p += KE[k];
    return (gamma-1.)*(state[num_equations-1] - p); 
  }
  
  static MFEM_HOST_DEVICE double GetViscosity_gpu(const double &temp)
  {
    // Sutherland's law
    return 1.458e-6 *pow(temp,1.5)/(temp+110.4);
  }
  
  static MFEM_HOST_DEVICE double GetThermalConductivity_gpu(const double& visc, 
                                                            const double &gamma,
                                                            const double &Rg,
                                                            const double &Pr
                                                           )
  {
    const double cp = gamma*Rg/(gamma-1.);
    return visc*cp/Pr;
  }
#endif

};

#endif // EQUATION_OF_STATE
