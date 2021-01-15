#ifndef WALL_BC
#define WALL_BC

#include "mfem.hpp"
#include "BoundaryCondition.hpp"

using namespace mfem;

enum WallType {
  INV,        // Inviscid wall
  VISC_ADIAB  // Viscous adiabatic wall
};

class WallBC : public BoundaryCondition
{
private:
  // The type of wall
  const WallType wallType;
  
public:
  WallBC( RiemannSolver *rsolver_, 
          EquationOfState *_eqState,
          double &_dt,
          const int _dim,
          const int _num_equation,
          int _patchNumber,
          WallType _bcType );
  ~WallBC();
  
  void computeState(Vector &nor,
                    Vector &stateIn, 
                    Vector &stateOut);
};

#endif // WALL_BC
