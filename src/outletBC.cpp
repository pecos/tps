#include "outletBC.hpp"

OutletBC::OutletBC(mfem::Mesh* _mesh, 
                 mfem::IntegrationRules* _intRules, 
                 RiemannSolver* rsolver_, 
                 EquationOfState *_eqState,
                 const int _dim, 
                 const int _num_equation, 
                 double& _max_char_speed,
                 int _patchNumber,
                 OutletType _bcType,
                 const Array<double> &_inputData ):
InOutBC(_mesh, _intRules, rsolver_,_eqState,_dim,
        _num_equation,_max_char_speed, _patchNumber),
outletType(_bcType)
{
  // only subsonic case considered
  inputState.Append( _inputData[0] );
}

OutletBC::~OutletBC()
{
}

void OutletBC::computeState(Vector &stateIn, Vector &stateOut)
{
  //cout<<"outlet "<<outletType<<endl;
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
