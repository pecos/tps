
#include "fluxes.hpp"

Fluxes::Fluxes(EquationOfState *_eqState):
  eqState(_eqState)
{
}


void Fluxes::ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(eqState->StateIsPhysical(state, dim), "");

   const double pres =  eqState->ComputePressure(state, dim);
   //cout<<pres<<" "<<den<<" "<<den_energy<<endl;

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
      {
         flux(1+i, d) = den_vel(i) * den_vel(d) / den;
      }
      flux(1+d, d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim, d) = den_vel(d) * H;
   }
}

void Fluxes::ComputeSplitFlux(const mfem::Vector &state, 
                              mfem::DenseMatrix &a_mat, 
                              mfem::DenseMatrix &c_mat)
{
  const int num_equations = state.Size();
  const int dim = num_equations-2;
  
  a_mat.SetSize(num_equations, dim);
  c_mat.SetSize(num_equations, dim);
  
  const double rho  = state( 0 );
  const Vector rhoV(state.GetData()+1, dim);
  const double rhoE = state(num_equations-1);
  //const double p = eqState->ComputePressure(state, dim);
  //cout<<"*"<<p<<" "<<rho<<" "<<rhoE<<endl;
  
  for(int d=0; d<dim; d++)
  {
    for(int i=0; i<num_equations-1; i++) a_mat(i,d) = rhoV(d);
    a_mat(num_equations-1,d) = rhoV(d)/rho;
    
    c_mat(0,d) = 1.;
    for(int i=0; i<dim; i++)
    {
      c_mat(i+1,d) = rhoV(i)/rho;
    }
    c_mat(num_equations-1, d) = rhoE /*+p*/;
  }
  
}
