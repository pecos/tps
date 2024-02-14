#include "turb_model_base.hpp"

using namespace mfem;

ZeroTurbModel::ZeroTurbModel(ParMesh *pmesh, int sorder)
    : pmesh_(pmesh), sorder_(sorder) {}

ZeroTurbModel::~ZeroTurbModel() {
  delete eddy_viscosity_;
  delete fes_;
  delete fec_;
}

void ZeroTurbModel::initializeSelf() {
  fec_ = new H1_FECollection(sorder_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_);

  eddy_viscosity_ = new ParGridFunction(fes_);

  *eddy_viscosity_ = 0.0;

  toFlow_interface_.eddy_viscosity = viscosity_;
  toThermChem_interface_.eddy_viscosity = density_;
  
}
