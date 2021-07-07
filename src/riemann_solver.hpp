#ifndef RIEMANN_SOLVER
#define RIEMANN_SOLVER

#include "mfem.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "general/forall.hpp"

using namespace mfem;

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
  const int &num_equation;
  
  Vector flux1;
  Vector flux2;

  EquationOfState *eqState;
  Fluxes *fluxClass;
  
  bool useRoe;
  
  void Eval_LF(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux); 
  void Eval_Roe(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux); 

public:
   RiemannSolver(int &_num_equation,
                 EquationOfState *_eqState,
                 Fluxes *_fluxClass,
                 bool _useRoe );
   
   void Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux, bool LF=false);
   
   void ComputeFluxDotN(const Vector &state, 
                        const Vector &nor,
                        Vector &fluxN);
   
#ifdef _GPU_
    static MFEM_HOST_DEVICE void convFluxDotNorm_gpu( const double *state,
                                                      const double pres,
                                                      const double *nor,
                                                      const int dim,
                                                      double *fluxN )
    {
      double den_velN = 0;
      for(int d=0;d<dim;d++) den_velN += state[d+1]*nor[d]; 

      fluxN[0] = den_velN;
      for(int d=0;d<dim;d++)
      {
          fluxN[1+d] = den_velN*state[d+1]/state[0] +pres*nor[d];
      }

      const double H = (state[dim+1] + pres)/state[0];
      fluxN[1 + dim] = den_velN * H;
    };
    
    static MFEM_HOST_DEVICE void riemannLF_gpu( const double *U1, 
                                                const double *U2, 
                                                double *flux,
                                                const double *nor,
                                                const double gamma,
                                                const double Rg,
                                                const int dim, 
                                                const int num_equation)
    {
      double KE1[3], KE2[3];
      double vel1 = 0., vel2 = 0.;
      for(int d=0;d<dim;d++)
      {
        KE1[d] = 0.5*U1[1+d]*U1[1+d]/U1[0];
        KE2[d] = 0.5*U2[1+d]*U2[1+d]/U2[0];
        vel1 += 2*KE1[d]/U1[0];
        vel2 += 2*KE2[d]/U2[0];
      }
      vel1 = sqrt(vel1);
      vel2 = sqrt(vel2);
      
      const double p1 = EquationOfState::pressure(U1, 
                                                  &KE1[0], 
                                                  gamma, 
                                                  dim, 
                                                  num_equation);
      const double p2 = EquationOfState::pressure(U2, 
                                                  &KE2[0], 
                                                  gamma, 
                                                  dim, 
                                                  num_equation);
      
      const double maxE1 = vel1 + sqrt(gamma*p1/U1[0]);
      const double maxE2 = vel2 + sqrt(gamma*p2/U2[0]);

      const double maxE = max(maxE1, maxE2);

      double flux1[5],flux2[5];
      RiemannSolver::convFluxDotNorm_gpu(U1,p1, nor,dim, flux1);
      RiemannSolver::convFluxDotNorm_gpu(U2,p2, nor,dim, flux2);

      double normag = 0;
      for(int i=0;i<dim;i++)
      {
        normag += nor[i]*nor[i];
      }
      normag = sqrt(normag);

      for(int i=0;i<num_equation;i++)
      {
        flux[i] = 0.5*(flux1[i]+flux2[i])
                  -0.5*maxE*(U2[i] -U1[i])*normag;
      }
    };
#endif
};

#endif // RIEMANN_SOLVER
