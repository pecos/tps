
#include "fluxes.hpp"

Fluxes::Fluxes(EquationOfState *_eqState,
               Equations &_eqSystem,
               const int &_num_equations,
               const int &_dim
              ):
eqState(_eqState),
eqSystem(_eqSystem),
dim(_dim),
num_equations(_num_equations)
{
}

void Fluxes::ComputeTotalFlux(const Vector& state, 
                              const DenseMatrix& gradUpi, 
                              DenseMatrix& flux)
{
  switch( eqSystem )
  {
    case EULER:
      ComputeConvectiveFluxes(state,flux);
      break;
    case NS:
    {
      DenseMatrix convF(num_equations,dim);
      ComputeConvectiveFluxes(state,convF);
      
      DenseMatrix viscF(num_equations,dim);
      ComputeViscousFluxes(state,gradUpi,viscF);
      for(int eq=0;eq<num_equations;eq++)
      {
        for(int d=0;d<dim;d++)flux(eq,d) = convF(eq,d) - viscF(eq,d);
      }
    }
      break;
    case MHD:
      break;
  }
}


void Fluxes::ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux)
{
   const double pres =  eqState->ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = state(d+1);
      for (int i=0;i<dim;i++)
      {
         flux(1+i,d) = state(i+1)*state(d+1)/state[0];
      }
      flux(1+d, d) += pres;
   }

   const double H = (state[1+dim] + pres)/state[0];
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim,d) = state(d+1)*H;
   }
}

void Fluxes::ComputeViscousFluxes(const Vector& state, 
                                  const DenseMatrix& gradUp, 
                                  DenseMatrix& flux)
{
  switch( eqSystem )
  {
    case NS:
    {
      const double Rg = eqState->GetGasConstant();
      const double p  = eqState->ComputePressure(state,dim);
      const double temp = p/state[0]/Rg;
      const double visc = eqState->GetViscosity(temp);
      const double k    = eqState->GetThermalConductivity(visc);
      
      // make sure density visc. flux is 0
      for(int d=0;d<dim;d++) flux(0,d) = 0.;
      
      double divV = 0.;
      DenseMatrix stress(dim,dim);
      for(int i=0;i<dim;i++)
      {
        for(int j=0;j<dim;j++)
        {
          stress(i,j) = gradUp(1+j,i) + gradUp(1+i,j);
        }
        divV += gradUp(1+i,i);
      }
      for(int i=0;i<dim;i++) stress(i,i) -= 2./3.*divV;
      stress *= visc;
      
      for(int i=0;i<dim;i++)
      {
        for(int j=0;j<dim;j++)
        {
          flux(1+i,j) = stress(i,j);
        }
      }
      
      // temperature gradient
      Vector gradT(dim);
      for(int d=0;d<dim;d++) gradT[d] = temp*(gradUp(1+dim,d)/p - gradUp(0,d)/state[0]);
      
      Vector vel(dim);
      for(int d=0;d<dim;d++) vel(d) = state[1+d]/state[0];
      for(int d=0;d<dim;d++)
      {
        for(int i=0;i<dim;i++) flux(1+dim,d) += vel[i]*stress(d,i);
        flux(1+dim,d) += k*gradT[d];
      }
    }
      break;
    default:
      flux = 0.;
      break;
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
