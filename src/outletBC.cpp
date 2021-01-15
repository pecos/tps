#include "outletBC.hpp"

OutletBC::OutletBC(RiemannSolver *_rsolver, 
            EquationOfState *_eqState,
            double &_dt,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            OutletType _bcType,
            const Array<double> &_inputData ):
BoundaryCondition(_rsolver, 
                  _eqState,
                  _dt,
                  _dim,
                  _num_equation,
                  _patchNumber),
outletType(_bcType),
inputState(_inputData)
{
}

OutletBC::~OutletBC()
{
}

void OutletBC::computeState(Vector &nor,
                            Vector &stateIn, 
                            Vector &stateOut)
{
  switch(outletType)
  {
    case SUB_P:
      subsonicReflectingPressure(stateIn, stateOut);
      break;
    case SUB_P_NR:
      break;
  }
}

void OutletBC::subsonicReflectingPressure(Vector &stateIn, Vector &stateOut)
{
  double rho   = stateIn(0);
  double rhoVx = stateIn(1);
  double rhoVy = stateIn(2);
  //double rhoVz = stateIn(3);
  
  const double gamma = eqState->GetSpecificHeatRatio();
  double rhoE = inputState[0]/(gamma-1.) + 0.5*(rhoVx*rhoVx + 
                                    rhoVy*rhoVy /*+
                                    rhoVz*rhoVz*/ )/rho;
                                    
  stateOut(0) = rho;
  stateOut(1) = rhoVx;
  stateOut(2) = rhoVy;
  //stateOut(3) = rhoVz;
  stateOut(3) = rhoE;
}
