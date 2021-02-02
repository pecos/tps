#ifndef WALL_BC
#define WALL_BC

#include "mfem.hpp"
#include "BoundaryCondition.hpp"
#include "fluxes.hpp"

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
  
  Fluxes *fluxClass;
  
  void computeINVwallFlux(Vector &normal,
                          Vector &stateIn,
                          Vector &bdrFlux);
  
  void computeAdiabaticWallFlux(Vector &normal,
                                Vector &stateIn, 
                                DenseMatrix &gradState,
                                Vector &bdrFlux);
  
public:
  WallBC( RiemannSolver *rsolver_, 
          EquationOfState *_eqState,
          Fluxes *_fluxClass,
          ParFiniteElementSpace *_vfes,
          IntegrationRules *_intRules,
          double &_dt,
          const int _dim,
          const int _num_equation,
          int _patchNumber,
          WallType _bcType );
  ~WallBC();
  
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);
  
  virtual void updateMean(IntegrationRules *intRules,
                          ParGridFunction *Up){};
};

#endif // WALL_BC
