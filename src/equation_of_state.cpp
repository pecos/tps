

#include "equation_of_state.hpp"


EquationOfState::EquationOfState(WorkingFluid _fluid):
  fluid(_fluid)
{
  switch(fluid)
  {
    case DRY_AIR:
      gas_constant = 287.058;
      //gas_constant = 1.; // for comparison against ex18
      specific_heat_ratio = 1.4;
      visc_mult = 1.;
      Pr = 0.71;
      break;
    default:
      break;
  }
}


// Pressure (EOS) computation
double EquationOfState::ComputePressure(const Vector& state, int dim)
{
   double den_vel2 = 0;
   for (int d=0;d<dim;d++) den_vel2 += state(d+1)*state(d+1);
   den_vel2 /= state[0];

   return (specific_heat_ratio - 1.0)*(state[1+dim] - 0.5*den_vel2);
}

bool EquationOfState::StateIsPhysical(const mfem::Vector& state, const int dim)
{
  const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) { den_vel2 += den_vel(i) * den_vel(i); }
   den_vel2 /= den;

   const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}

// Compute the maximum characteristic speed.
double EquationOfState::ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   const double pres = ComputePressure(state, dim);
   const double sound = sqrt(specific_heat_ratio * pres / den);
   const double vel = sqrt(den_vel2 / den);

   return vel + sound;
}

double EquationOfState::GetViscosity(const double &temp)
{
  // Sutherland's law
  double visc = 1.458e-6 *pow(temp,1.5)/(temp+110.4);
  return visc*visc_mult;
}

double EquationOfState::GetThermalConductivity(const double& visc)
{
  const double cp = specific_heat_ratio*gas_constant/(
                    specific_heat_ratio-1.);
  return visc*cp/Pr;
}

