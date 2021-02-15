#include "BoundaryCondition.hpp"

BoundaryCondition::BoundaryCondition(RiemannSolver* _rsolver, 
                                     EquationOfState* _eqState, 
                                     ParFiniteElementSpace *_vfes,
                                     IntegrationRules *_intRules,
                                     double &_dt,
                                     const int _dim, 
                                     const int _num_equation, 
                                     const int _patchNumber,
                                     const double _refLength ):
rsolver(_rsolver),
eqState(_eqState),
vfes(_vfes),
intRules(_intRules),
dt(_dt),
dim(_dim),
num_equation(_num_equation),
patchNumber(_patchNumber),
refLength(_refLength)
{
}

BoundaryCondition::~BoundaryCondition()
{
}
