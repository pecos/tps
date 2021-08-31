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
  
  const int &maxIntPoints;
  const int &maxDofs;
  
public:
  GradNonLinearForm(ParFiniteElementSpace *f,
                    IntegrationRules *intRules,
                    const int dim,
                    const int num_equation,
                    const volumeFaceIntegrationArrays &gpuArrays,
                    const int &maxIntPoints,
                    const int &maxDofs );
  
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
