#include "wallBC.hpp"

WallBC::WallBC(RiemannSolver* _rsolver, 
               EquationOfState* _eqState, 
               Fluxes *_fluxClass,
               ParFiniteElementSpace *_vfes,
               IntegrationRules *_intRules,
               double &_dt,
               const int _dim, 
               const int _num_equation, 
               int _patchNumber, 
               WallType _bcType):
BoundaryCondition(_rsolver, 
                  _eqState,
                  _vfes,
                  _intRules,
                  _dt,
                  _dim,
                  _num_equation,
                  _patchNumber,
                  1 ), // so far walls do not require ref. length. Left at 1
wallType(_bcType),
fluxClass(_fluxClass)
{
}

WallBC::~WallBC()
{
}

void WallBC::computeBdrFlux(Vector &normal,
                            Vector &stateIn, 
                            DenseMatrix &gradState,
                            Vector &bdrFlux)
{
  switch(wallType)
  {
    case INV:
      computeINVwallFlux(normal,stateIn,bdrFlux);
      break;
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal,stateIn,gradState,bdrFlux);
      break;
  }
}

void WallBC::computeINVwallFlux(Vector &normal,
                                Vector &stateIn,
                                Vector &bdrFlux)
{
  Vector vel(dim);
  for(int d=0;d<dim;d++) vel[d] = stateIn[1+d]/stateIn[0];
  
  double norm = 0.;
  for(int d=0;d<dim;d++) norm += normal[d]*normal[d];
  norm = sqrt(norm);
  
  Vector unitN(dim);
  for(int d=0;d<dim;d++) unitN[d] = normal[d]/norm;
  
  double vn = 0;
  for(int d=0;d<dim;d++) vn += vel[d]*unitN[d];
  
  Vector stateMirror(num_equation);
  
  stateMirror[0] = stateIn[0];
  stateMirror[1] = stateIn[0]*(vel[0] -2.*vn*unitN[0]);
  stateMirror[2] = stateIn[0]*(vel[1] -2.*vn*unitN[1]);
  if(dim==3) stateMirror[3] = stateIn[0]*(vel[2] -2.*vn*unitN[2]);
  stateMirror[num_equation-1] = stateIn[num_equation-1];
  
  rsolver->Eval(stateIn,stateMirror,normal,bdrFlux);
}


void WallBC::computeAdiabaticWallFlux(Vector &normal,
                                      Vector &stateIn, 
                                      DenseMatrix &gradState,
                                      Vector &bdrFlux)
{
  double p = eqState->ComputePressure(stateIn,dim);
  const double gamma = eqState->GetSpecificHeatRatio();
  
  Vector wallState = stateIn;
  for(int d=0;d<dim;d++) wallState[1+d] = 0.;
  wallState[num_equation-1] = p/(gamma-1.);
  
  // Normal convective flux
  rsolver->Eval(stateIn,wallState,normal,bdrFlux, true);
  
  // incoming visc flux
  DenseMatrix viscF(num_equation,dim);
  fluxClass->ComputeViscousFluxes(stateIn,gradState,viscF);
  
  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  {
    double normN = 0.;
    for(int d=0;d<dim;d++) normN += normal[d]*normal[d];
    unitNorm *= 1./sqrt(normN);
  }
  
  // modify gradient density so dT/dn=0 at the wall
  double DrhoDn = 0.;
  for(int d=0;d<dim;d++) DrhoDn += gradState(0,d)*unitNorm[d];
  
  double DrhoDnNew = 0.;
  for(int d=0;d<dim;d++) DrhoDnNew += gradState(1+dim,d)*unitNorm[d];
  DrhoDnNew *= stateIn[0]/p;
  
  for(int d=0;d<dim;d++) gradState(0,d) += -DrhoDn*unitNorm[d] + 
                                            DrhoDnNew*unitNorm[d];
  
  DenseMatrix viscFw(num_equation,dim);
  fluxClass->ComputeViscousFluxes(wallState,gradState,viscFw);
  
  // Add visc fluxes
  for(int eq=1;eq<num_equation;eq++) // we skip density eq.
  {
    for(int d=0;d<dim;d++) bdrFlux[eq] -= 0.5*(viscFw(eq,d)+viscF(eq,d))*normal[d];
  }
}
