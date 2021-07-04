#ifndef INLET_BC
#define INLET_BC

#include "mfem.hpp"
#include "mpi_groups.hpp"
#include "BoundaryCondition.hpp"
#include "logger.hpp"

using namespace mfem;

enum InletType {
  SUB_DENS_VEL, // Subsonic inlet specified by the density and velocity components
  SUB_DENS_VEL_NR, // Non-reflecting subsonic inlet specified by the density and velocity components
  SUB_VEL_CONST_ENT // Subsonic non-reflecting. Specified vel, keeps entropy constant
};

class InletBC : public BoundaryCondition
{
private:
  MPI_Groups *groupsMPI;
  
  const InletType inletType;
  
  // In/out conditions specified in the configuration file
  const Array<double> inputState;
  
  // Mean boundary state
  Vector meanUp;
  
  DenseMatrix boundaryU;
  int bdrN;
  bool bdrUInit;

  // Unit trangent vector 1 & 2
  Vector tangent1;
  Vector tangent2;
  
  void subsonicReflectingDensityVelocity( Vector &normal,
                                          Vector &stateIn, 
                                          Vector &bdrFlux);
  
  void subsonicNonReflectingDensityVelocity(Vector &normal,
                                            Vector &stateIn, 
                                            DenseMatrix &gradState,
                                            Vector &bdrFlux);
  
  virtual void updateMean(IntegrationRules *intRules,
                          ParGridFunction *Up);

public:
   InletBC( MPI_Groups *_groupsMPI,
            RiemannSolver *rsolver_, 
            EquationOfState *_eqState,
            ParFiniteElementSpace *_vfes,
            IntegrationRules *_intRules,
            double &_dt,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            double _refLength,
            InletType _bcType,
            const Array<double> &_inputData );
   ~InletBC();
   
   // computes state that imposes the boundary
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);

  void initState();
};

#endif // INLET_BC
