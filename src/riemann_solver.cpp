
#include "riemann_solver.hpp"

using namespace mfem;

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(int &_num_equation,
                             EquationOfState *_eqState ) :
num_equation(_num_equation),
eqState(_eqState)
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

void RiemannSolver::Eval(const Vector &state1, const Vector &state2,
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
