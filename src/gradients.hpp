// -----------------------------------------------------------------------------------bl-
// -----------------------------------------------------------------------------------el-
#ifndef _GRADIENTS_
#define _GRADIENTS_

// Class to manage gradients of primitives

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>
#include "gradNonLinearForm.hpp"
#include "dataStructures.hpp"
#include "equation_of_state.hpp"

using namespace mfem;
using namespace std;

class Gradients : public ParNonlinearForm
{
private:
  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *gradUpfes;
  const int dim;
  const int num_equation;
  
  ParGridFunction *Up;
  ParGridFunction *gradUp;
  
  EquationOfState *eqState;
  
  //ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;
  
  IntegrationRules *intRules;
  const int intRuleType;
  
  const volumeFaceIntegrationArrays &gpuArrays;
  
  const int *h_numElems;
  const int *h_posDofIds;
  
  //DenseMatrix *Me_inv;
  Array<DenseMatrix*> &Me_inv;
  Array<DenseMatrix*> Ke;
  Vector &invMArray;
  Array<int> &posDofInvM;
  
  const int &maxIntPoints;
  const int &maxDofs;
  
  // gradients of shape functions for all nodes and weight multiplied by det(Jac)
  // at each integration point
//   Vector elemShapeDshapeWJ; // [...l_0(i),...,l_dof(i),l_0_x(i),...,l_dof_d(i), w_i*detJac_i ...]
//   Array<int> elemPosQ_shapeDshapeWJ; // position and num. of integration points for each element

  parallelFacesIntegrationArrays *parallelData;
  dataTransferArrays *transferUp;

public:
  Gradients(ParFiniteElementSpace *_vfes,
            ParFiniteElementSpace *_gradUpfes,
            int _dim,
            int _num_equation,
            ParGridFunction *_Up,
            ParGridFunction *_gradUp,
            EquationOfState *_eqState,
            GradNonLinearForm *_gradUp_A,
            IntegrationRules *_intRules,
            int _intRuleType,
            const volumeFaceIntegrationArrays &gpuArrays,
            Array<DenseMatrix*> &Me_inv,
            Vector &_invMArray,
            Array<int> &_posDofInvM,
            const int &_maxIntPoints,
            const int &_maxDofs );
  
  ~Gradients();
  
  void setParallelData(parallelFacesIntegrationArrays *_parData,
                      dataTransferArrays *_transferUp ){
    parallelData = _parData;
    transferUp = _transferUp; }
  
  void computeGradients();
  
#ifdef _GPU_
  void computeGradients_domain();
  void computeGradients_bdr();
  
  static void computeGradients_gpu(const int numElems,
                                   const int offsetElems,
                                   const int elDof,
                                   const int totalDofs,
                                   const Vector &Up,
                                   Vector &gradUp,
                                   const int num_equation,
                                   const int dim,
                                   const volumeFaceIntegrationArrays &gpuArrays,
//                                    const Vector &elemShapeDshapeWJ,
//                                    const Array<int> &elemPosQ_shapeDshapeWJ,
                                   const int &maxDofs,
                                   const int &maxIntPoints );
  
  static void integrationGradSharedFace_gpu(const Vector *Up,
                                            const Vector &faceUp,
                                            ParGridFunction *gradUp,
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
                                            const int &maxDofs);
  
  static void multInverse_gpu(const int numElems,
                              const int offsetElems,
                              const int elDof,
                              const int totalDofs,
                              Vector &gradUp,
                              const int num_equation,
                              const int dim,
                              const volumeFaceIntegrationArrays &gpuArrays,
                              const Vector &invMArray,
                              const Array<int> &posDofInvM );
#endif
};

#endif // _GRADIENTS_
