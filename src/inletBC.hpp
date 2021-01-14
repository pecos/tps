#ifndef INLET_BC
#define INLET_BC

#include "InOutBC.hpp"

using namespace mfem;
using namespace std;

enum InletType {
  PV, // Subsonic inlet specified by the pressure and velocity components
  PV_NR // Non-reflecting subsonic inlet specified by the pressure and velocity components
};

class InletBC : public InOutBC
{
private:
  
  const InletType inletType;
  
  // computes state that imposes the boundary
  void computeState(Vector &stateIn, Vector &stateOut);
  void subsonicReflectingPressureVelocity(Vector &stateIn, Vector &stateOut);

public:
   InletBC( Mesh *_mesh,
            IntegrationRules *_intRules,
            RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            const int _dim,
            const int _num_equation,
            double &_max_char_speed,
            int _patchNumber,
            InletType _bcType,
            const Array<double> &inputData );
   ~InletBC();
};

#endif // INLET_BC
