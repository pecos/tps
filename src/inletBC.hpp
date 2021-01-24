#ifndef INLET_BC
#define INLET_BC

#include "mfem.hpp"
#include "BoundaryCondition.hpp"

using namespace mfem;

enum InletType {
  SUB_DENS_VEL, // Subsonic inlet specified by the density and velocity components
  SUB_DENS_VEL_NR // Non-reflecting subsonic inlet specified by the density and velocity components
};

class InletBC : public BoundaryCondition
{
private:
  const InletType inletType;
  
  // In/out conditions specified in the configuration file
  const Array<double> &inputState;
  
  // Mean boundary state
  Vector meanUp;
  
  DenseMatrix boundaryU;
  int bdrN;
  bool bdrUInit;
  
  void subsonicReflectingDensityVelocity( Vector &normal,
                                          Vector &stateIn, 
                                          Vector &bdrFlux);
  
  void subsonicNonReflectingDensityVelocity(Vector &normal,
                                            Vector &stateIn, 
                                            DenseMatrix &gradState,
                                            Vector &bdrFlux);
  
  virtual void updateMean(IntegrationRules *intRules,
                          GridFunction *Up);

public:
   InletBC( RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            FiniteElementSpace *_vfes,
            IntegrationRules *_intRules,
            double &_dt,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            InletType _bcType,
            const Array<double> &_inputData );
   ~InletBC();
   
   // computes state that imposes the boundary
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);
};

#endif // INLET_BC
