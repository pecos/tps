#ifndef WALL_BC
#define WALL_BC

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>
#include "BoundaryCondition.hpp"
#include "fluxes.hpp"

using namespace mfem;

enum WallType {
  INV,        // Inviscid wall
  VISC_ADIAB, // Viscous adiabatic wall
  VISC_ISOTH  // Viscous isothermal wall
};

class WallBC : public BoundaryCondition
{
private:
  // The type of wall
  const WallType wallType;
  
  Fluxes *fluxClass;
  
  double wallTemp;
  
  const Array<int> &intPointsElIDBC;
  
  Array<int> wallElems;
  void buildWallElemsArray(const Array<int> &intPointsElIDBC);
  
  void computeINVwallFlux(Vector &normal,
                          Vector &stateIn,
                          Vector &bdrFlux);
  
  void computeAdiabaticWallFlux(Vector &normal,
                                Vector &stateIn, 
                                DenseMatrix &gradState,
                                Vector &bdrFlux);
  void computeIsothermalWallFlux( Vector &normal,
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
          WallType _bcType,
          const Array<double> _inputData,
          const Array<int> &intPointsElIDBC );
  ~WallBC();
  
  void computeBdrFlux(Vector &normal,
                      Vector &stateIn, 
                      DenseMatrix &gradState,
                      Vector &bdrFlux);

  // koomie TODO: blend ths with initState()
  virtual void initBCs();
  
  virtual void updateMean(IntegrationRules *intRules,
                          ParGridFunction *Up){};


  // functions for BC integration on GPU

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

  static void integrateWalls_gpu( const WallType type,
                                  const double &wallTemp,
                                  Vector &y, // output
                                  const Vector &x, 
                                  const Array<int> &nodesIDs,
                                  const Array<int> &posDofIds,
                                  ParGridFunction *Up,
                                  ParGridFunction *gradUp,
                                  Vector &shapesBC,
                                  Vector &normalsWBC,
                                  Array<int> &intPointsElIDBC,
                                  Array<int> &wallElems,
                                  Array<int> &listElems,
                                  const int &maxIntPoints,
                                  const int &maxDofs,
                                  const int &dim,
                                  const int &num_equation,
                                  const double &gamma,
                                  const double &Rg );
  
#ifdef _GPU_
  static MFEM_HOST_DEVICE void computeInvWallState(const int &thrd,
                                                   const double *u1,
                                                   double *u2,
                                                   const double *nor,
                                                   const int &dim,
                                                   const int &num_equation )
  {
    double momNormal = 0.;
    double unitNor[3];
    double norm;
    norm = 0.;
    for(int d=0;d<dim;d++) norm += nor[d]*nor[d];
    norm = sqrt(norm);

    for(int d=0;d<dim;d++)
    {
      unitNor[d] = nor[d]/norm;
      momNormal += unitNor[d]*u1[d+1];
    }
    //if(dim==2) unitNor[2] = 0.;


    if(thrd==0 || thrd==num_equation-1)
    {
      u2[thrd] = u1[thrd];
    }else
    {
      u2[thrd] = u1[thrd] -2.*momNormal*unitNor[thrd-1];
    }
  };

  static MFEM_HOST_DEVICE void computeIsothermalState(const int &thrd,
                                                      const double *u1,
                                                      double *u2,
                                                      const double *nor,
                                                      const double &wallTemp,
                                                      const double &gamma,
                                                      const double &Rg,
                                                      const int &dim,
                                                      const int &num_equation )
  {
    if(thrd==num_equation-1) u2[thrd] = Rg/(gamma-1.)*u1[0]*wallTemp;
    if(thrd==0) u2[thrd] = u1[thrd];
    if(thrd>0 && thrd<dim+1) u2[thrd] = 0.;
  };
#endif

  virtual void initState(){};
};

#endif // WALL_BC
