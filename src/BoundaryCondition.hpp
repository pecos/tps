#ifndef BOUNDARY_CONDITION
#define BOUNDARY_CONDITION

#include <mfem.hpp>
#include <tps_config.h>
#include <mfem/general/forall.hpp>
#include "riemann_solver.hpp"
#include "equation_of_state.hpp"

using namespace mfem;

class BoundaryCondition
{
protected:
  RiemannSolver *rsolver;
  EquationOfState *eqState;
  ParFiniteElementSpace *vfes;
  IntegrationRules *intRules;
  double &dt;
  const int dim;
  const int num_equation;
  const int patchNumber;
  const double refLength;
  
  bool BCinit;
  
  Array<int> listElems; // list of boundary elements (position in the BC array)
  
public:
  BoundaryCondition(RiemannSolver *_rsolver, 
                    EquationOfState *_eqState,
                    ParFiniteElementSpace *_vfes,
                    IntegrationRules *_intRules,
                    double &dt,
                    const int _dim,
                    const int _num_equation,
                    const int _patchNumber,
                    const double _refLength );
  virtual ~BoundaryCondition();
  
  virtual void computeBdrFlux(Vector &normal,
                              Vector &stateIn, 
                              DenseMatrix &gradState,
                              Vector &bdrFlux) = 0;
  
  // function to initiate variables that require MPI to be already initialized
  virtual void initBCs() = 0;
                              
  virtual void updateMean(IntegrationRules *intRules,
                          ParGridFunction *Up) = 0;

  // aggregate boundary area
  double aggregateArea (int bndry_attr, MPI_Comm bc_comm);
  // aggregate boundary face count
  int aggregateBndryFaces (int bndry_attr, MPI_Comm bc_comm);

  // holding function for any miscellanous items needed to initialize BCs
  // prior to use
  virtual void initState() = 0;

  // integration of BC on GPU
  void setElementList(Array<int> &listElems);

  virtual void integrationBC( Vector &y, // output
			      const Vector &x, // conservative vars (input)
			      const Array<int> &nodesIDs,
			      const Array<int> &posDofIds,
			      ParGridFunction *Up,
			      ParGridFunction *gradUp,
			      Vector &shapesBC,
			      Vector &normalsWBC,
			      Array<int> &intPointsElIDBC,
			      const int &maxIntPoints,
			      const int &maxDofs ) = 0;

  static void copyValues(const Vector &orig, Vector &target, const double &mult);
};

#endif // BOUNDARY_CONDITION
