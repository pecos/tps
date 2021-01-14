#ifndef INLET_BC
#define INLET_BC

#include "mfem.hpp"
#include "BoundaryCondition.hpp"

using namespace mfem;

enum InletType {
  PV, // Subsonic inlet specified by the pressure and velocity components
  PV_NR // Non-reflecting subsonic inlet specified by the pressure and velocity components
};

class InletBC : public BoundaryCondition
{
private:
  const InletType inletType;
  
  // In/out conditions specified in the configuration file
  const Array<double> &inputState;
  
  // Mean boundary state
  Vector *meanState;
  
  // computes state that imposes the boundary
  void computeState(Vector &stateIn, Vector &stateOut);
  void subsonicReflectingPressureVelocity(Vector &stateIn, Vector &stateOut);

public:
   InletBC( RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            InletType _bcType,
            const Array<double> &_inputData );
   ~InletBC();
};

#endif // INLET_BC
