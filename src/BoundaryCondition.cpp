#include "BoundaryCondition.hpp"

BoundaryCondition::BoundaryCondition(RiemannSolver* _rsolver, 
                                     EquationOfState* _eqState, 
                                     const int _dim, 
                                     const int _num_equation, 
                                     const int _patchNumber ):
rsolver(_rsolver),
eqState(_eqState),
dim(_dim),
num_equation(_num_equation),
patchNumber(_patchNumber)
{
}

BoundaryCondition::~BoundaryCondition()
{
}
