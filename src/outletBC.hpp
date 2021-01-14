#ifndef OUTLET_BC
#define OUTLET_BC

#include "BoundaryCondition.hpp"

using namespace mfem;

enum OutletType {
  SUB_P, // subsonic outlet specified with pressure
  SUB_P_NR // non-reflecting subsonic outlet specified with pressure
};

class OutletBC : public BoundaryCondition
{
private:
  const OutletType outletType;
  
  // In/out conditions specified in the configuration file
  const Array<double> &inputState;
  
  // Mean boundary state
  Vector *meanState;
  
  // computes state that imposes the boundary
  void computeState(Vector &stateIn, Vector &stateOut);
  void subsonicReflectingPressure(Vector &stateIn, Vector &stateOut);

public:
   OutletBC( RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            OutletType _bcType,
            const Array<double> &_inputData );
   ~OutletBC();
};

#endif // OUTLET_BC
