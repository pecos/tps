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
  
  Array<int> &nodesIDs;
  Array<int> &posDofIds;
  Array<int> &numElems;
  
  int *h_numElems;
  int *h_posDofIds;
  
  //DenseMatrix *Me_inv;
  Array<DenseMatrix*> &Me_inv;
  Array<DenseMatrix*> Ke;
  Vector &invMArray;
  Array<int> &posDofInvM;
  
  // for face integration
  Vector &shapeWnor1;
  Vector &shape2;
  Array<int> &elemFaces; // number and faces IDs of each element
  Array<int> &elems12Q; // elements connecting a face
  const int &maxIntPoints;
  const int &maxDofs;
  
  // gradients of shape functions for all nodes and weight multiplied by det(Jac)
  // at each integration point
  Vector elemShapeDshapeWJ; // [...l_0(i),...,l_dof(i),l_0_x(i),...,l_dof_d(i), w_i*detJac_i ...]
  Array<int> elemPosQ_shapeDshapeWJ; // position and num. of integration points for each element

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
            Array<int> &nodesIDs,
            Array<int> &posDofIds,
            Array<int> &numElems,
            Array<DenseMatrix*> &Me_inv,
            Vector &_invMArray,
            Array<int> &_posDofInvM,
            Vector &_shapeWnor1,
            Vector &_shape2,
            Array<int> &_elemFaces,
            Array<int> &_elems12Q,
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
                                   const Array<int> &posDofIds,
                                   const Array<int> &nodesIDs,
                                   const Vector &elemShapeDshapeWJ,
                                   const Array<int> &elemPosQ_shapeDshapeWJ,
                                   const Array<int> &elemFaces,
                                   const Vector &shapeWnor1,
                                   const Vector &shape2,
                                   const int &maxDofs,
                                   const int &maxIntPoints,
                                   const Array<int> &elems12Q );
  
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
                                            const Array<int> &nodesIDs,
                                            const Array<int> &posDofIds,
                                            const parallelFacesIntegrationArrays *parallelData,
                                            const int &maxIntPoints,
                                            const int &maxDofs);
  
  static void multInversr_gpu(const int numElems,
                              const int offsetElems,
                              const int elDof,
                              const int totalDofs,
                              Vector &gradUp,
                              const int num_equation,
                              const int dim,
                              const Array<int> &posDofIds,
                              const Array<int> &nodesIDs,
                              const Vector &invMArray,
                              const Array<int> &posDofInvM );
#endif
};

#endif // _GRADIENTS_
