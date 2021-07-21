#ifndef DGNONLINEARFORM
#define DGNONLINEARFORM

#include <mfem.hpp>
#include <tps_config.h>
#include <fem/nonlinearform.hpp>
#include "equation_of_state.hpp"
#include "BCintegrator.hpp"

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
  
  Array<int> &numElems;
  Array<int> &nodesIDs;
  Array<int> &posDofIds;
  
  const int *h_numElems;
  const int *h_posDofIds;
  
  Vector &shapeWnor1;
  Vector &shape2;
  
  Array<int> &elemFaces; // number and faces IDs of each element
  Array<int> &elems12Q; // elements connecting a face
  
  // Parallel shared faces integration
  Vector sharedShapeWnor1;
  Vector sharedShape2;
  Array<int> sharedElem1Dof12Q;
  Array<int> sharedVdofs;
  Array<int> sharedVdofsGradUp;
  Array<int> sharedElemsFaces;
  
  const int &maxIntPoints;
  const int &maxDofs;
  
  bool sharedDataFilled;
  void fillSharedData();
  
  Vector face_nbr_data;
  Vector send_data;
  Vector face_nbr_dataGrad;
  Vector send_dataGrad;
  
public:
  DGNonLinearForm(ParFiniteElementSpace *f,
                  ParFiniteElementSpace *gradFes,
                  ParGridFunction *_gradUp,
                  BCintegrator *_bcIntegrator,
                  IntegrationRules *intRules,
                  const int dim,
                  const int num_equation,
                  EquationOfState *eqState,
                  Array<int> &_numElems,
                  Array<int> &_nodesIDs,
                  Array<int> &_posDofIds,
                  Vector &_shapeWnor1,
                  Vector &_shape2,
                  Array<int> &_elemFaces,
                  Array<int> &_elems12Q,
                  const int &maxIntPoints,
                  const int &maxDofs );
  
  void Mult(const Vector &x, Vector &y );
            
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
                                  const Array<int> &elemFaces,
                                  const Array<int> &nodesIDs,
                                  const Array<int> &posDofIds,
                                  const Vector &shapeWnor1,
                                  const Vector &shape2,
                                  const int &maxIntPoints,
                                  const int &maxDofs,
                                  const Array<int> &elems12Q );
  
  static void sharedFaceIntegration_gpu(const Vector &x,
                                        const ParGridFunction *gradUp,
                                        const Vector &faceData,
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
                                        const Array<int> &nodesIDs,
                                        const Array<int> &posDofIds,
                                        const Vector &sharedShapeWnor1,
                                        const Vector &sharedShape2,
                                        const Array<int> &sharedElem1Dof12Q,
                                        const Array<int> &sharedVdofs,
                                        const Array<int> &sharedVdofsGradUp,
                                        const Array<int> &sharedElemFaces,
                                        const int &maxIntPoints,
                                        const int &maxDofs );
  
  static void exchangeBdrData(const Vector &x,
                              ParFiniteElementSpace *pfes,
                              Vector &face_nbr_data,
                              Vector &send_data);
#ifdef _GPU_
  void Mult_gpu(const Vector &x, Vector &y);
  static void setToZero_gpu(Vector &x, const int size);
#endif
};

#endif // DGNONLINEARFORM
