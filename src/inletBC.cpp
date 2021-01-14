#include "inletBC.hpp"

InletBC::InletBC(mfem::Mesh* _mesh, 
                 mfem::IntegrationRules* _intRules, 
                 RiemannSolver* rsolver_, 
                 EquationOfState *_eqState,
                 const int _dim, 
                 const int _num_equation, 
                 double& _max_char_speed,
                 int _patchNumber,
                 InletType _bcType,
                 const Array<double> &inputData ):
InOutBC(_mesh, _intRules, rsolver_,_eqState,_dim,
        _num_equation,_max_char_speed, _patchNumber),
inletType(_bcType)
{
  
  // only subsonic case considered
  for(int i=0; i<4; i++) {inputState.Append( inputData[i] ); cout<<inputState[i]<<endl;}
}

InletBC::~InletBC()
{
}

void InletBC::computeState(Vector &stateIn, Vector &stateOut)
{
  cout<<"inlet "<<inletType<<endl;
  switch(inletType)
  {
    case PV:
      subsonicReflectingPressureVelocity(stateIn, stateOut);
      break;
    case PV_NR:
      break;
  }
}

void InletBC::subsonicReflectingPressureVelocity(Vector &stateIn, Vector &stateOut)
{
  double rho   = stateIn(0);
  double rhoVx = inputState[1]*rho;
  double rhoVy = inputState[2]*rho;
  double rhoVz = inputState[3]*rho;
  
  const double gamma = eqState->GetSpecificHeatRatio();
  double rhoE = inputState[0]/(gamma-1.) + 0.5*(rhoVx*rhoVx + 
                                    rhoVy*rhoVy +
                                    rhoVz*rhoVz )/rho;
                                    
  stateOut(0) = rho;
  stateOut(1) = rhoVx;
  stateOut(2) = rhoVy;
  //stateOut(3) = rhoVz;
  stateOut(3) = rhoE;
}
