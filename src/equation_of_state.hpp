// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
#ifndef EQUATION_OF_STATE
#define EQUATION_OF_STATE

/*
 * Implementation of the equation of state and some
 * handy functions to deal with operations.
 */

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>

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
  
  double bulk_visc_mult;
  
  // Prandtl number
  double Pr;            // Prandtl number
  double cp_div_pr;     // cp divided by Pr (used in conductivity calculation)
public:
  EquationOfState();
  
  void setFluid(WorkingFluid _fluid);
  
  void setViscMult(double _visc_mult){visc_mult = _visc_mult;}
  void setBulkViscMult(double _bulk_mult){bulk_visc_mult = _bulk_mult;}
  
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
  double GetBulkViscMultiplyer(){return bulk_visc_mult;}
  double GetThermalConductivity(const double &visc);


  
  // GPU functions
#ifdef _GPU_
  static MFEM_HOST_DEVICE double pressure(const double *state, 
                                          double *KE, 
                                          const double &gamma, 
                                          const int &dim, 
                                          const int &num_equations)
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

// additional functions inlined for speed...
inline double EquationOfState::ComputePressure(const Vector& state, int dim)
{
  double den_vel2 = 0;
  for (int d=0;d<dim;d++)
    den_vel2 += state(d+1)*state(d+1);
  den_vel2 /= state[0];
  
  return (specific_heat_ratio - 1.0)*(state[1+dim] - 0.5*den_vel2);
}

inline double EquationOfState::GetViscosity(const double &temp)
{
  // Sutherland's law
  return(1.458e-6*visc_mult*pow(temp,1.5)/(temp+110.4));
}

inline double EquationOfState::GetThermalConductivity(const double &visc)
{
  return(visc*cp_div_pr);
}

#endif // EQUATION_OF_STATE
