#include "wallBC.hpp"

WallBC::WallBC(RiemannSolver* _rsolver, 
               EquationOfState* _eqState, 
               FiniteElementSpace *_vfes,
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
                  _patchNumber),
wallType(_bcType)
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
      computeINVwallFlux(normal,stateIn,gradState,bdrFlux);
      break;
    case VISC_ADIAB:
      break;
  }
}

void WallBC::computeINVwallFlux(Vector &normal,
                                Vector &stateIn, 
                                DenseMatrix &gradState,
                                Vector &bdrFlux)
{
  double u = stateIn[1]/stateIn[0];
  double v = stateIn[2]/stateIn[0];
  const double norm = sqrt(normal[0]*normal[0] + normal[1]*normal[1]);
  Vector unitN(dim);
  for(int d=0;d<dim;d++) unitN[d] = normal[d]/norm;
  double vn = u*unitN[0] + v*unitN[1];
  
  Vector stateMirror(num_equation);
  
  stateMirror[0] = stateIn[0];
  stateMirror[1] = stateIn[0]*(u -2.*vn*unitN[0]);
  stateMirror[2] = stateIn[0]*(v -2.*vn*unitN[1]);
  stateMirror[3] = stateIn[3];
  
  rsolver->Eval(stateIn,stateMirror,normal,bdrFlux);
}
