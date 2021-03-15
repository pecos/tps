
#include "riemann_solver.hpp"

using namespace mfem;

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(int &_num_equation,
                             EquationOfState *_eqState,
                             Fluxes *_fluxClass,
                             bool _useRoe ):
num_equation(_num_equation),
eqState(_eqState),
fluxClass(_fluxClass),
useRoe(_useRoe)
{
  flux1.SetSize(num_equation);
  flux2.SetSize(num_equation);
}
   
// Compute the scalar F(u).n
void RiemannSolver::ComputeFluxDotN(const Vector &state, 
                                    const Vector &nor,
                                    Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   //MFEM_ASSERT(eqState->StateIsPhysical(state, dim), "");

   const double pres = eqState->ComputePressure(state, dim);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) { den_velN += den_vel(d) * nor(d); }

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d);
   }

   const double H = (den_energy + pres) / den;
   fluxN(1 + dim) = den_velN * H;
}

void RiemannSolver::Eval(const Vector& state1, 
                         const Vector& state2, 
                         const Vector& nor, 
                         Vector& flux, 
                         bool LF)
{
  if( useRoe && !LF )
  {
    Eval_Roe(state1, state2, nor,flux);
  }else
  {
    Eval_LF(state1, state2, nor,flux);
  }
}


void RiemannSolver::Eval_LF(const Vector &state1, const Vector &state2,
                            const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   //MFEM_ASSERT(eqState->StateIsPhysical(state1, dim), "");
   //MFEM_ASSERT(eqState->StateIsPhysical(state2, dim), "");

   const double maxE1 = eqState->ComputeMaxCharSpeed(state1, dim);
   const double maxE2 = eqState->ComputeMaxCharSpeed(state2, dim);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1);
   ComputeFluxDotN(state2, nor, flux2);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }
}

void RiemannSolver::Eval_Roe(const Vector &state1, const Vector &state2,
                             const Vector &nor, Vector &flux)
{
  const int dim = nor.Size();
  
  double normag = 0;
  for (int i = 0; i < dim; i++) normag += nor(i)*nor(i);
  normag = sqrt(normag);
  
  Vector unitN(dim);
  for(int d=0;d<dim;d++) unitN[d] = nor[d]/normag;
  
  DenseMatrix fluxes1(num_equation,dim), fluxes2(num_equation,dim);
  fluxClass->ComputeConvectiveFluxes(state1,fluxes1);
  fluxClass->ComputeConvectiveFluxes(state2,fluxes2);
  Vector meanFlux(num_equation);
  for(int eq=0;eq<num_equation;eq++)
  {
    meanFlux[eq] = 0.;
    for(int d=0;d<dim;d++) meanFlux[eq] += (fluxes1(eq,d) +fluxes2(eq,d))*unitN[d];
  }

    // Roe (Lohner)
  double r = sqrt(state1[0]*state2[0]);
  Vector vel(dim);
  for(int i=0;i<dim;i++)
  {
    vel[i] = state1[i+1]/sqrt(state1[0]) + state2[i+1]/sqrt(state2[0]);
    vel[i] /= sqrt(state1[0]) + sqrt(state2[0]);
  }
  double qk = 0.;
  for(int d=0;d<dim;d++) qk += vel[d]*unitN[d];
  
  double p1 = eqState->ComputePressure(state1, dim);
  double p2 = eqState->ComputePressure(state2, dim);
  double H = (state1[num_equation-1]+p1)/sqrt(state1[0]) +
             (state2[num_equation-1]+p2)/sqrt(state2[0]);
  H /= sqrt(state1[0]) + sqrt(state2[0]);
  double a2 = 0.4*(H -0.5*(vel[0]*vel[0]+vel[1]*vel[1]));
  double a = sqrt(a2);
  double lamb[3];
  lamb[0] = qk;
  lamb[1] = qk + a;
  lamb[2] = qk - a;
  
  if( fabs(lamb[0])<1e-4 ) lamb[0] = 1e-4;
  
  double deltaP = p2 - p1;
  double deltaU = state2[1]/state2[0]-state1[1]/state1[0];
  double deltaV = state2[2]/state2[0]-state1[2]/state1[0];
  double deltaQk = deltaU*unitN[0] + deltaV*unitN[1];
  
  Vector alpha(num_equation);
  alpha[0] = 0.5*(deltaP - r*a*deltaU)/a2;
  alpha[1] = state2[0] - state1[0] - deltaP/a2;
  alpha[2] = r*deltaV;
  alpha[3] = 0.5*(deltaP + r*a*deltaU)/a2;
  
  Vector DF1(num_equation);
  DF1[0] = 1.;
  DF1[1] = vel[0];
  DF1[2] = vel[1];
  DF1[3] = 0.5*(vel[0]*vel[0]+vel[1]*vel[1]);
  DF1 *= state2[0] - state1[0] - deltaP/a2;
  DF1[1] += r*(deltaU - unitN[0]*deltaQk);
  DF1[2] += r*(deltaV - unitN[1]*deltaQk);
  DF1[3] += r*(vel[0]*deltaU + vel[1]*deltaV -qk*deltaQk);
  
  DF1 *= fabs(lamb[0]);
  
  Vector DF4(num_equation);
  DF4[0] = 1.;
  DF4[1] = vel[0] + unitN[0]*a;
  DF4[2] = vel[1] + unitN[1]*a;
  DF4[3] = H + qk*a;
  DF4 *= fabs(lamb[1])*(deltaP + r*a*deltaQk)*0.5/a2;
  
  Vector DF5(num_equation);
  DF5[0] = 1.;
  DF5[1] = vel[0] - unitN[0]*a;
  DF5[2] = vel[1] - unitN[1]*a;
  DF5[3] = H - qk*a;
  DF5 *= fabs(lamb[2])*(deltaP - r*a*deltaQk)*0.5/a2;

  for (int i=0;i<num_equation;i++)
  {
    flux(i) = (meanFlux[i] -(DF1[i] + DF4[i] + DF5[i]) )*0.5*normag;
  }
}
