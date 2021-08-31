#ifndef DGNONLINEARFORM
#define DGNONLINEARFORM

#include <mfem.hpp>
#include <tps_config.h>
#include <fem/nonlinearform.hpp>
#include "equation_of_state.hpp"
#include "BCintegrator.hpp"
#include "dataStructures.hpp"

using namespace mfem;
using namespace std;

// Derived non linear form that re-implements the Mult method
// so that we can use GPU 
class DGNonLinearForm : public ParNonlinearForm
{
private:
  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *gradFes;
  
  ParGridFunction *gradUp;
  
  BCintegrator *bcIntegrator;
  
  IntegrationRules *intRules;
  const int dim;
  const int num_equation;
  EquationOfState *eqState;
  
  const volumeFaceIntegrationArrays &gpuArrays;
  
  const int *h_numElems;
  const int *h_posDofIds;
  
  // Parallel shared faces integration
  parallelFacesIntegrationArrays *parallelData;
  mutable dataTransferArrays *transferU;
  mutable dataTransferArrays *transferGradUp;
  
  const int &maxIntPoints;
  const int &maxDofs;
  
  
  
public:
  DGNonLinearForm(ParFiniteElementSpace *f,
                  ParFiniteElementSpace *gradFes,
                  ParGridFunction *_gradUp,
                  BCintegrator *_bcIntegrator,
                  IntegrationRules *intRules,
                  const int dim,
                  const int num_equation,
                  EquationOfState *eqState,
                  const volumeFaceIntegrationArrays &_gpuArrays,
                  const int &maxIntPoints,
                  const int &maxDofs );
  
  void Mult(const Vector &x, Vector &y );
  
  void setParallelData(parallelFacesIntegrationArrays *_parallelData,
                       dataTransferArrays *_transferU,
                       dataTransferArrays *_transferGradUp ){ 
    parallelData = _parallelData;
    transferU = _transferU;
    transferGradUp = _transferGradUp;
  }
            
  static void faceIntegration_gpu(const Vector &x,
                                  Vector &y,
                                  const ParGridFunction *gradUp,
                                  const int &Ndofs,
                                  const int &Nf,
                                  const int &NumElemsType,
                                  const int &elemOffset,
                                  const int &elDof,
                                  const int &dim,
                                  const int &num_equation,
                                  const double &gamma,
                                  const double &Rg,
                                  const double &viscMult,
                                  const double &bulkViscMult,
                                  const double &Pr,
                                  const volumeFaceIntegrationArrays &gpuArrays,
                                  const int &maxIntPoints,
                                  const int &maxDofs);
  
  static void sharedFaceIntegration_gpu(const Vector &x,
                                        const Vector &faceU,
                                        const ParGridFunction *gradUp,
                                        const Vector &faceGradUp,
                                        Vector &y,
                                        const int &Ndofs,
                                        const int &dim,
                                        const int &num_equation,
                                        const double &gamma,
                                        const double &Rg,
                                        const double &viscMult,
                                        const double &bulkViscMult,
                                        const double &Pr,
                                        const volumeFaceIntegrationArrays &gpuArrays,
                                        const parallelFacesIntegrationArrays *parallelData,
                                        const int &maxIntPoints,
                                        const int &maxDofs );
  
#ifdef _GPU_
  void Mult_domain(const Vector &x, Vector &y);
  void Mult_bdr(const Vector &x, Vector &y);
  static void setToZero_gpu(Vector &x, const int size);
#endif
};

#endif // DGNONLINEARFORM
