#ifndef OUTLET_BC
#define OUTLET_BC

#include "InOutBC.hpp"

using namespace mfem;
using namespace std;

enum OutletType {
  SUB_P, // subsonic outlet specified with pressure
  SUB_P_NR // non-reflecting subsonic outlet specified with pressure
};

class OutletBC : public InOutBC
{
private:
  
  OutletType outletType;
  
  // computes state that imposes the boundary
  void computeState(Vector &stateIn, Vector &stateOut);
  void subsonicReflectingPressure(Vector &stateIn, Vector &stateOut);

public:
   OutletBC( Mesh *_mesh,
            IntegrationRules *_intRules,
            RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            const int _dim,
            const int _num_equation,
            double &_max_char_speed,
            int _patchNumber,
            OutletType _bcTpe,
            const Array<double> &_inputData );
   ~OutletBC();
};

#endif // OUTLET_BC
