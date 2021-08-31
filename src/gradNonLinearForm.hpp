#ifndef GRAD_NONLINEARFORM
#define GRAD_NONLINEARFORM

#include <mfem.hpp>
#include <general/forall.hpp>
#include <tps_config.h>
#include "dataStructures.hpp"

using namespace mfem;
using namespace std;

class GradNonLinearForm : public ParNonlinearForm
{
private:
  ParFiniteElementSpace *vfes;
  IntegrationRules *intRules;
  const int dim;
  const int num_equation;
  
  const volumeFaceIntegrationArrays &gpuArrays;
  
//   Array<int> &numElems;
//   Array<int> &nodesIDs;
//   Array<int> &posDofIds;
  
//   int *h_numElems;
//   int *h_posDofIds;
/*  
  Vector &shapeWnor1;
  Vector &shape2;*/
  const int &maxIntPoints;
  const int &maxDofs;
  
//   Array<int> &elemFaces; // number and faces IDs of each element
//   Array<int> &elems12Q; // elements connecting a face
  
public:
  GradNonLinearForm(ParFiniteElementSpace *f,
                    IntegrationRules *intRules,
                    const int dim,
                    const int num_equation,
                    const volumeFaceIntegrationArrays &gpuArrays,
//                     Array<int> &_numElems,
//                     Array<int> &_nodesIDs,
//                     Array<int> &_posDofIds,
//                     Vector &_shapeWnor1,
//                     Vector &_shape2,
                    const int &maxIntPoints,
                    const int &maxDofs/*,
                    Array<int> &_elemFaces,
                    Array<int> &_elems12Q */ );
  
  void Mult(const ParGridFunction *Up, Vector &y );
  
#ifdef _GPU_
  void Mult_gpu(const Vector &x, Vector &y);
  static void IndexedAddToGlobalMemory(const Array<int> &vdofs,
                                       Vector &y,
                                       const Vector &el_y,
                                       const int &dof,
                                       const int &elDof,
                                       const int &num_equation,
                                       const int &dim );
#endif
};

#endif // GRAD_NONLINEARFORM
