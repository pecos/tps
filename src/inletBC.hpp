#ifndef INLET_BC
#define INLET_BC

#include <mfem.hpp>
#include <mfem/general/forall.hpp>
#include <tps_config.h>
#include "mpi_groups.hpp"
#include "BoundaryCondition.hpp"
#include "equation_of_state.hpp"
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
  Vector inputState;
  
  // Mean boundary state
  Vector meanUp;
  
  Vector boundaryU;
  int bdrN;
  bool bdrUInit;
  
  // boundary mean calculation variables
  Vector bdrUp; // Up at 
  Array<int> bdrElemsQ; // element dofs and face num. of integration points
  Array<int> bdrDofs; // indexes of the D
  Vector bdrShape; // shape functions evaluated at the integration points
  const int &maxIntPoints;
  const int &maxDofs;

  // Unit trangent vector 1 & 2
  Vector tangent1;
  Vector tangent2;
  Vector inverseNorm2cartesian;
  
  void initBdrElemsShape();
  void initBoundaryU(ParGridFunction *Up);
  
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
            const Array<double> &_inputData,
            const int &_maxIntPoints,
            const int &maxDofs );
   ~InletBC();
   
   // computes state that imposes the boundary
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);
  
  virtual void initBCs();
  
  virtual void integrationBC( Vector &y, // output
                              const Vector &x, 
                              const Array<int> &nodesIDs,
                              const Array<int> &posDofIds,
                              ParGridFunction *Up,
                              ParGridFunction *gradUp,
                              Vector &shapesBC,
                              Vector &normalsWBC,
                              Array<int> &intPointsElIDBC,
                              const int &maxIntPoints,
                              const int &maxDofs );


  static void updateMean_gpu( ParGridFunction *Up,
			      Vector &localMeanUp,
			      const int _num_equation,
			      const int numBdrElems,
			      const int totalDofs,
			      Vector &bdrUp,
			      Array<int> &bdrElemsQ,
			      Array<int> &bdrDofs,
			      Vector &bdrShape,
			      const int &maxIntPoints,
			      const int &maxDofs );
  
  void initState();

  // functions for BC integration on GPU

  static void integrateInlets_gpu(const InletType type,
				  const Vector &inputState,
				  const double &dt,
				  Vector &y, // output
				  const Vector &x,
				  const Array<int> &nodesIDs,
				  const Array<int> &posDofIds,
				  ParGridFunction *Up,
				  ParGridFunction *gradUp,
				  Vector &shapesBC,
				  Vector &normalsWBC,
				  Array<int> &intPointsElIDBC,
				  Array<int> &listElems,
				  const int &maxIntPoints,
				  const int &maxDofs,
				  const int &dim,
				  const int &num_equation,
				  const double &gamma,
				  const double &Rg );

#ifdef _GPU_
  static MFEM_HOST_DEVICE void computeSubDenseVel( const int &thrd,
						   const double *u1,
						   double *u2,
						   const double *nor,
						   const double *inputState,
						   const double &gamma,
						   const int &dim,
						   const int &num_equation )
  {
    MFEM_SHARED double KE[3];
    MFEM_SHARED double p;

    if(thrd<dim) KE[thrd] = 0.5*u1[1+thrd]*u1[1+thrd]/u1[0];
    if( dim!=3 && thrd==num_equation-1 ) KE[2] = 0.;
    MFEM_SYNC_THREAD;

    if( thrd==0 ) p = EquationOfState::pressure(u1,&KE[0],gamma,dim,num_equation);
    MFEM_SYNC_THREAD;

    if(thrd==0) u2[0] = inputState[0];
    if(thrd==1) u2[1] = inputState[0]*inputState[1];
    if(thrd==2) u2[2] = inputState[0]*inputState[2];
    if(dim==3 && thrd==3) u2[3] = inputState[0]*inputState[3];
    if(thrd==num_equation-1)
      {
	double ke = 0.;
	for(int d=0;d<dim;d++) ke += inputState[1+d]*inputState[1+d];
	u2[thrd] = p/(gamma-1.) + 0.5*inputState[0]*ke;
      }
  };
#endif
};

#endif // INLET_BC
