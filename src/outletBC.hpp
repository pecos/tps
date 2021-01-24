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
  Vector meanUp;
  
  DenseMatrix boundaryU;
  int bdrN;
  bool bdrUInit;
  
  void subsonicReflectingPressure(Vector &normal,
                                  Vector &stateIn, 
                                  Vector &bdrFlux);
  
  void subsonicNonReflectingPressure( Vector &normal,
                                      Vector &stateIn, 
                                      DenseMatrix &gradState,
                                      Vector &bdrFlux);
  
  virtual void updateMean(IntegrationRules *intRules,
                          GridFunction *Up);

public:
   OutletBC( RiemannSolver *rsolver_, 
             EquationOfState *_eqState,
             FiniteElementSpace *_vfes,
             IntegrationRules *_intRules,
             double &_dt,
             const int _dim,
             const int _num_equation,
             int _patchNumber,
             OutletType _bcType,
             const Array<double> &_inputData );
   ~OutletBC();
   
   // computes state that imposes the boundary
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);
};

#endif // OUTLET_BC
