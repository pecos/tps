#include "wallBC.hpp"

WallBC::WallBC(RiemannSolver* _rsolver, 
               EquationOfState* _eqState, 
               double &_dt,
               const int _dim, 
               const int _num_equation, 
               int _patchNumber, 
               WallType _bcType):
BoundaryCondition(_rsolver, 
                  _eqState,
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

void WallBC::computeState(Vector& nor, Vector& stateIn, Vector& stateOut)
{
  double u = stateIn[1]/stateIn[0];
  double v = stateIn[2]/stateIn[0];
  const double norm = sqrt(nor[0]*nor[0] + nor[1]*nor[1]);
  double vn = u*nor[0] + v*nor[1];
  vn /= norm;
  
  stateOut[0] = stateIn[0];
  stateOut[1] = stateIn[0]*(u -2.*vn*nor[0]/norm);
  stateOut[2] = stateIn[0]*(v -2.*vn*nor[1]/norm);
  stateOut[3] = stateIn[3];
}
