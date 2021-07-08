#ifndef OUTLET_BC
#define OUTLET_BC

#include <mfem.hpp>
#include <mfem/general/forall.hpp>
#include "mpi_groups.hpp"
#include "BoundaryCondition.hpp"
#include "logger.hpp"

using namespace mfem;

enum OutletType {
  SUB_P, // subsonic outlet specified with pressure
  SUB_P_NR, // non-reflecting subsonic outlet specified with pressure
  SUB_MF_NR, // Mass-flow non-reflecting 
  SUB_MF_NR_PW // point-based non-reflecting massflow BC
};

class OutletBC : public BoundaryCondition
{
private:
  MPI_Groups *groupsMPI;
  
  const OutletType outletType;
  
  // In/out conditions specified in the configuration file
  const Array<double> inputState;
  
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
  
  // area of the outlet
  double area;
  bool parallelAreaComputed;
  
  // Unit trangent vector 1 & 2
  Vector tangent1;
  Vector tangent2;
  Vector inverseNorm2cartesian;
  
  void initBdrElemsShape();
  void initBoundaryU( ParGridFunction *Up );
  
  void subsonicReflectingPressure(Vector &normal,
                                  Vector &stateIn, 
                                  Vector &bdrFlux);
  
  void subsonicNonReflectingPressure( Vector &normal,
                                      Vector &stateIn, 
                                      DenseMatrix &gradState,
                                      Vector &bdrFlux);
  
  void subsonicNonRefMassFlow(Vector &normal,
                              Vector &stateIn, 
                              DenseMatrix &gradState,
                              Vector &bdrFlux);
                              
  void subsonicNonRefPWMassFlow(Vector &normal,
                                Vector &stateIn, 
                                DenseMatrix &gradState,
                                Vector &bdrFlux);
  
  virtual void updateMean(IntegrationRules *intRules,
                          ParGridFunction *Up);

  void computeParallelArea();

public:
   OutletBC( MPI_Groups *_groupsMPI,
             RiemannSolver *rsolver_, 
             EquationOfState *_eqState,
             ParFiniteElementSpace *_vfes,
             IntegrationRules *_intRules,
             double &_dt,
             const int _dim,
             const int _num_equation,
             int _patchNumber,
             double _refLength,
             OutletType _bcType,
             const Array<double> &_inputData,
             const int &_maxIntPoints,
             const int &maxDofs );
   ~OutletBC();
   
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
  
  // functions for BC integration on GPU

  static void integrateOutlets_gpu( const OutletType type,
                                    const Array<double> &inputState,
                                    const double &dt,
                                    Vector &y, // output
                                    const Vector &x, 
                                    const Array<int> &nodesIDs,
                                    const Array<int> &posDofIds,
                                    ParGridFunction *Up,
                                    ParGridFunction *gradUp,
                                    Vector &meanUp,
                                    Vector &boundaryU,
                                    Vector &tangent1,
                                    Vector &tangent2,
                                    Vector &inverseNorm2cartesian,
                                    Vector &shapesBC,
                                    Vector &normalsWBC,
                                    Array<int> &intPointsElIDBC,
                                    Array<int> &listElems,
                                    const int &maxIntPoints,
                                    const int &maxDofs,
                                    const int &dim,
                                    const int &num_equation,
                                    const double &gamma,
                                    const double &Rg,
                                    const double &refLength,
                                    const double &area );
  
#ifdef _GPU_ // GPU functions
  static MFEM_HOST_DEVICE void computeSubPressure( const int &thrd,
                                                   const double *u1,
                                                   double *u2,
                                                   const double *nor,
                                                   const double &press,
                                                   const double &gamma,
                                                   const int &dim,
                                                   const int &num_equation )
  {
    if(thrd==num_equation-1)
    {
      double k = 0.;
      for(int d=0;d<dim;d++) k += u1[1+d]*u1[1+d];
      u2[thrd] = press/(gamma-1.) + 0.5*k/u1[0];
    }else
    {
      u2[thrd] = u1[thrd];
    }
  };

  static MFEM_HOST_DEVICE void computeNRSubPress( const int &thrd,
                                                  const int &n,
                                                  const double *u1,
                                                  const double *gradUp,
                                                  const double *meanUp,
                                                  const double &dt,
                                                  double *u2,
                                                  double *boundaryU,
                                                  const double *inputState,
                                                  const double *nor,
                                                  const double *d_tang1,
                                                  const double *d_tang2,
                                                  const double *d_inv,
                                                  const double &refLength,
                                                  const double &gamma,
                                                  const int &elDof,
                                                  const int &dim,
                                                  const int &num_equation )
  {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[5];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK;
    MFEM_SHARED double L1,L2,L3,L4,L5;
    MFEM_SHARED double d1,d2,d3,d4,d5;
    MFEM_SHARED double bdrFlux[5], uN[5], newU[5];

    if(thrd<dim) unitNorm[thrd] = nor[thrd];
    if(thrd==num_equation-1)
    {
      mod = 0.;
      for(int d=0;d<dim;d++) mod += nor[d]*nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;

    if(thrd<dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;

    // mean vel. outlet tangent and normal
    if(thrd<3)
    {
      meanVel[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==0) meanVel[0] += unitNorm[d]*meanUp[d+1];
        if(thrd==1) meanVel[1] += d_tang1[d]*meanUp[d+1];
        if(dim==3 && thrd==dim-1)
        {
          for(int d=0;d<dim;d++) meanVel[2] += d_tang2[d]*meanUp[d+1];
        }
      }
    }
    MFEM_SYNC_THREAD;

    if(thrd<num_equation)
    {
      normGrad[thrd] = 0.;
      for(int d=0;d<dim;d++) normGrad[thrd] += unitNorm[d]*gradUp[thrd+d*num_equation];
    }
    MFEM_SYNC_THREAD;


    if(thrd==0) speedSound = sqrt(gamma*meanUp[num_equation-1]/meanUp[0]);
    if(thrd==elDof-1) 
    {
      meanK = 0.;
      for(int d=0;d<dim;d++) meanK += meanUp[1+d]*meanUp[1+d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;

    // compute outgoing characteristics
    if(thrd==1)
    {
      L2 = speedSound*speedSound*normGrad[0] - normGrad[num_equation-1];
      L2 *= meanVel[0];
    }
    
    if(thrd==2)
    {
      L3 = 0;
      for(int d=0;d<dim;d++) L3 += d_tang1[d]*normGrad[1+d];
      L3 *= meanVel[0];
    }
    
    if(thrd==3)
    {
      L4 = 0.;
      if(dim==3)
      {
        for(int d=0;d<dim;d++) L4 += d_tang2[d]*normGrad[1+d];
        L4 *= meanVel[0];
      }
    }
    
    if(thrd==0)
    {
      L5 = 0.;
      for(int d=0;d<dim;d++) L5 += unitNorm[d]*normGrad[1+d];
      L5 = normGrad[num_equation-1] +meanUp[0]*speedSound*L5;
      L5 *= meanVel[0] + speedSound;
    }
    
    // estimate ingoing characteristic
    if(thrd==elDof-1)
    {
      const double sigma = speedSound/refLength;
      L1 = sigma*(meanUp[num_equation-1] - inputState[0]);
    }
    MFEM_SYNC_THREAD;

    if(thrd==0) d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
    if(thrd==1) d2 = 0.5*(L5-L1)/meanUp[0]/speedSound;
    if(thrd==2) d3 = L3;
    if(thrd==4) d4 = L4;
    if(thrd==5) d5 = 0.5*(L5+L1);
    MFEM_SYNC_THREAD;

    if(thrd==0) bdrFlux[0] = d1;
    if(thrd==1) bdrFlux[1] = meanVel[0]*d1 + meanUp[0]*d2;
    if(thrd==2) bdrFlux[2] = meanVel[1]*d1 + meanUp[0]*d3;
    if(thrd==elDof-1 && dim==3) bdrFlux[3] = meanVel[2]*d1 + meanUp[0]*d4;
    if(thrd==num_equation-1)
    {
      bdrFlux[num_equation-1] =  meanUp[0]*meanVel[0]*d2;
      bdrFlux[num_equation-1] += meanUp[0]*meanVel[1]*d3;
      if(dim==3) bdrFlux[num_equation-1] += meanUp[0]*meanVel[2]*d4;
      bdrFlux[num_equation-1] += meanK*d1 + d5/(gamma-1.);
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) u2[thrd] = boundaryU[thrd+n*num_equation];
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) uN[thrd] = u2[thrd];
    if(thrd>0 && thrd<dim+1)
    {
      uN[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==1) uN[1] += u2[1+d]*unitNorm[d];
        if(thrd==2) uN[2] += u2[1+d]*d_tang1[d];
        if(dim==3 && thrd==3) uN[3] += u2[1+d]*d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) newU[thrd] = uN[thrd] -dt*bdrFlux[thrd];
    MFEM_SYNC_THREAD;
    
    double sum = 0.;
    if(thrd<dim)
    {
      for(int j=0;j<dim;j++) sum += d_inv[thrd+j*dim]*newU[1+j];
    }
    MFEM_SYNC_THREAD;
    if(thrd<dim) newU[thrd+1] = sum;
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) boundaryU[thrd+n*num_equation] = newU[thrd];
  }
  
  static MFEM_HOST_DEVICE void computeNRSubMassFlow(const int &thrd,
                                                    const int &n,
                                                    const double *u1,
                                                    const double *gradUp,
                                                    const double *meanUp,
                                                    const double &dt,
                                                    double *u2,
                                                    double *boundaryU,
                                                    const double *inputState,
                                                    const double *nor,
                                                    const double *d_tang1,
                                                    const double *d_tang2,
                                                    const double *d_inv,
                                                    const double &refLength,
                                                    const double &area,
                                                    const double &gamma,
                                                    const int &elDof,
                                                    const int &dim,
                                                    const int &num_equation )
  {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[5];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK;
    MFEM_SHARED double L1,L2,L3,L4,L5;
    MFEM_SHARED double d1,d2,d3,d4,d5;
    MFEM_SHARED double bdrFlux[5], uN[5], newU[5];
    
    if(thrd<dim) unitNorm[thrd] = nor[thrd];
    if(thrd==num_equation-1)
    {
      mod = 0.;
      for(int d=0;d<dim;d++) mod += nor[d]*nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;
    
    // mean vel. outlet tangent and normal
    if(thrd<3)
    {
      meanVel[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==0) meanVel[0] += unitNorm[d]*meanUp[d+1];
        if(thrd==1) meanVel[1] += d_tang1[d]*meanUp[d+1];
        if(dim==3 && thrd==dim-1)
        {
          for(int d=0;d<dim;d++) meanVel[2] += d_tang2[d]*meanUp[d+1];
        }
      }
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation)
    {
      normGrad[thrd] = 0.;
      for(int d=0;d<dim;d++) normGrad[thrd] += unitNorm[d]*gradUp[thrd+d*num_equation];
    }
    MFEM_SYNC_THREAD;
    
    
    if(thrd==0) speedSound = sqrt(gamma*meanUp[num_equation-1]/meanUp[0]);
    if(thrd==elDof) 
    {
      meanK = 0.;
      for(int d=0;d<dim;d++) meanK += meanUp[1+d]*meanUp[1+d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;
    
    // compute outgoing characteristics
    if(thrd==1)
    {
      L2 = speedSound*speedSound*normGrad[0] - normGrad[num_equation-1];
      L2 *= meanVel[0];
    }
    
    if(thrd==2)
    {
      L3 = 0;
      for(int d=0;d<dim;d++) L3 += d_tang1[d]*normGrad[1+d];
      L3 *= meanVel[0];
    }
    
    if(thrd==3)
    {
      L4 = 0.;
      if(dim==3)
      {
        for(int d=0;d<dim;d++) L4 += d_tang2[d]*normGrad[1+d];
        L4 *= meanVel[0];
      }
    }
    
    if(thrd==0)
    {
      L5 = 0.;
      for(int d=0;d<dim;d++) L5 += unitNorm[d]*normGrad[1+d];
      L5 = normGrad[num_equation-1] +meanUp[0]*speedSound*L5;
      L5 *= meanVel[0] + speedSound;
    }
    
    // estimate ingoing characteristic
    if(thrd==elDof-1)
    {
      const double sigma = speedSound/refLength;
      L1 = -sigma*(meanVel[0] - inputState[0]/meanUp[0]/area);
      L1 *= meanUp[0]*speedSound;
    }
    MFEM_SYNC_THREAD;
    
    if(thrd==0) d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
    if(thrd==1) d2 = 0.5*(L5-L1)/meanUp[0]/speedSound;
    if(thrd==2) d3 = L3;
    if(thrd==4) d4 = L4;
    if(thrd==5) d5 = 0.5*(L5+L1);
    MFEM_SYNC_THREAD;
    
    if(thrd==0) bdrFlux[0] = d1;
    if(thrd==1) bdrFlux[1] = meanVel[0]*d1 + meanUp[0]*d2;
    if(thrd==2) bdrFlux[2] = meanVel[1]*d1 + meanUp[0]*d3;
    if(thrd==elDof-1 && dim==3) bdrFlux[3] = meanVel[2]*d1 + meanUp[0]*d4;
    if(thrd==num_equation-1)
    {
      bdrFlux[num_equation-1] =  meanUp[0]*meanVel[0]*d2;
      bdrFlux[num_equation-1] += meanUp[0]*meanVel[1]*d3;
      if(dim==3) bdrFlux[num_equation-1] += meanUp[0]*meanVel[2]*d4;
      bdrFlux[num_equation-1] += meanK*d1 + d5/(gamma-1.);
    }
    MFEM_SYNC_THREAD;

    if(thrd<num_equation) u2[thrd] = boundaryU[thrd+n*num_equation];
    MFEM_SYNC_THREAD;

    if(thrd<num_equation) uN[thrd] = u2[thrd];
    if(thrd>0 && thrd<dim+1)
    {
      uN[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==1) uN[1] += u2[1+d]*unitNorm[d];
        if(thrd==2) uN[2] += u2[1+d]*d_tang1[d];
        if(dim==3 && thrd==3) uN[3] += u2[1+d]*d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;

    if(thrd<num_equation) newU[thrd] = uN[thrd] -dt*bdrFlux[thrd];
    MFEM_SYNC_THREAD;

    double sum = 0.;
    if(thrd<dim)
    {
      for(int j=0;j<dim;j++) sum += d_inv[thrd+j*dim]*newU[1+j];
    }
    MFEM_SYNC_THREAD;
    if(thrd<dim) newU[thrd+1] = sum;
    MFEM_SYNC_THREAD;

    if(thrd<num_equation) boundaryU[thrd+n*num_equation] = newU[thrd];
  }
  
  static MFEM_HOST_DEVICE void computeNR_PW_SubMF(const int &thrd,
                                                const int &n,
                                                const double *u1,
                                                const double *gradUp,
                                                const double *meanUp,
                                                const double &dt,
                                                double *u2,
                                                double *boundaryU,
                                                const double *inputState,
                                                const double *nor,
                                                const double *d_tang1,
                                                const double *d_tang2,
                                                const double *d_inv,
                                                const double &refLength,
                                                const double &area,
                                                const double &gamma,
                                                const int &elDof,
                                                const int &dim,
                                                const int &num_equation )
  {
    MFEM_SHARED double unitNorm[3], meanVel[3], normGrad[5];
    MFEM_SHARED double mod;
    MFEM_SHARED double speedSound, meanK;
    MFEM_SHARED double L1,L2,L3,L4,L5;
    MFEM_SHARED double d1,d2,d3,d4,d5;
    MFEM_SHARED double bdrFlux[5], uN[5], newU[5];
    
    if(thrd<dim) unitNorm[thrd] = nor[thrd];
    if(thrd==num_equation-1)
    {
      mod = 0.;
      for(int d=0;d<dim;d++) mod += nor[d]*nor[d];
      mod = sqrt(mod);
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<dim) unitNorm[thrd] /= mod;
    MFEM_SYNC_THREAD;
    
    // mean vel. outlet tangent and normal
    if(thrd<3)
    {
      meanVel[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==0) meanVel[0] += unitNorm[d]*meanUp[d+1];
        if(thrd==1) meanVel[1] += d_tang1[d]*meanUp[d+1];
        if(dim==3 && thrd==dim-1)
        {
          for(int d=0;d<dim;d++) meanVel[2] += d_tang2[d]*meanUp[d+1];
        }
      }
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation)
    {
      normGrad[thrd] = 0.;
      for(int d=0;d<dim;d++) normGrad[thrd] += unitNorm[d]*gradUp[thrd+d*num_equation];
    }
    MFEM_SYNC_THREAD;
    
    
    if(thrd==0) speedSound = sqrt(gamma*meanUp[num_equation-1]/meanUp[0]);
    if(thrd==elDof) 
    {
      meanK = 0.;
      for(int d=0;d<dim;d++) meanK += meanUp[1+d]*meanUp[1+d];
      meanK *= 0.5;
    }
    MFEM_SYNC_THREAD;
    
    // compute outgoing characteristics
    if(thrd==1)
    {
      L2 = speedSound*speedSound*normGrad[0] - normGrad[num_equation-1];
      L2 *= meanVel[0];
    }
    
    if(thrd==2)
    {
      L3 = 0;
      for(int d=0;d<dim;d++) L3 += d_tang1[d]*normGrad[1+d];
      L3 *= meanVel[0];
    }
    
    if(thrd==3)
    {
      L4 = 0.;
      if(dim==3)
      {
        for(int d=0;d<dim;d++) L4 += d_tang2[d]*normGrad[1+d];
        L4 *= meanVel[0];
      }
    }
    
    if(thrd==0)
    {
      L5 = 0.;
      for(int d=0;d<dim;d++) L5 += unitNorm[d]*normGrad[1+d];
      L5 = normGrad[num_equation-1] +meanUp[0]*speedSound*L5;
      L5 *= meanVel[0] + speedSound;
    }
    
    // estimate ingoing characteristic
    if(thrd==elDof-1)
    {
      double normVel = 0.;
      for(int d=0;d<dim;d++) normVel += u1[1+d]*unitNorm[d];
      normVel /= u1[0];
      const double sigma = speedSound/refLength;
      L1 = -sigma*(normVel - inputState[0]/meanUp[0]/area);
      L1 *= meanUp[0]*speedSound;
    }
    MFEM_SYNC_THREAD;
    
    if(thrd==0) d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
    if(thrd==1) d2 = 0.5*(L5-L1)/meanUp[0]/speedSound;
    if(thrd==2) d3 = L3;
    if(thrd==4) d4 = L4;
    if(thrd==5) d5 = 0.5*(L5+L1);
    MFEM_SYNC_THREAD;
    
    if(thrd==0) bdrFlux[0] = d1;
    if(thrd==1) bdrFlux[1] = meanVel[0]*d1 + meanUp[0]*d2;
    if(thrd==2) bdrFlux[2] = meanVel[1]*d1 + meanUp[0]*d3;
    if(thrd==elDof-1 && dim==3) bdrFlux[3] = meanVel[2]*d1 + meanUp[0]*d4;
    if(thrd==num_equation-1)
    {
      bdrFlux[num_equation-1] =  meanUp[0]*meanVel[0]*d2;
      bdrFlux[num_equation-1] += meanUp[0]*meanVel[1]*d3;
      if(dim==3) bdrFlux[num_equation-1] += meanUp[0]*meanVel[2]*d4;
      bdrFlux[num_equation-1] += meanK*d1 + d5/(gamma-1.);
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) u2[thrd] = boundaryU[thrd+n*num_equation];
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) uN[thrd] = u2[thrd];
    if(thrd>0 && thrd<dim+1)
    {
      uN[thrd] = 0.;
      for(int d=0;d<dim;d++)
      {
        if(thrd==1) uN[1] += u2[1+d]*unitNorm[d];
        if(thrd==2) uN[2] += u2[1+d]*d_tang1[d];
        if(dim==3 && thrd==3) uN[3] += u2[1+d]*d_tang2[d];
      }
    }
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) newU[thrd] = uN[thrd] -dt*bdrFlux[thrd];
    MFEM_SYNC_THREAD;
    
    double sum = 0.;
    if(thrd<dim)
    {
      for(int j=0;j<dim;j++) sum += d_inv[thrd+j*dim]*newU[1+j];
    }
    MFEM_SYNC_THREAD;
    if(thrd<dim) newU[thrd+1] = sum;
    MFEM_SYNC_THREAD;
    
    if(thrd<num_equation) boundaryU[thrd+n*num_equation] = newU[thrd];
  }

#endif  // __GPU__

  void initState();

};

#endif // OUTLET_BC
