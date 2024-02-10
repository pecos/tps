#include "tps_mfem_wrap.hpp"

/// Container for a Dirichlet boundary conditions
template <class Tcoeff>
class DirichletBC_T {
 public:
  DirichletBC_T(mfem::Array<int> attr, Tcoeff *coeff) : attr(attr), coeff(coeff) {}

  // Move constructor (required for emplace_back)
  DirichletBC_T(DirichletBC_T &&obj) {
    // Deep copy the attribute array
    this->attr = obj.attr;

    // Move the coefficient pointer
    this->coeff = obj.coeff;
    obj.coeff = nullptr;
  }

  ~DirichletBC_T() { delete coeff; }

  mfem::Array<int> attr;
  Tcoeff *coeff;
};
