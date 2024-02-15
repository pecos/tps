#include "thermo_chem_base.hpp"

using namespace mfem;

ConstantPropertyThermoChem::ConstantPropertyThermoChem(ParMesh *pmesh, int sorder, double rho, double mu)
    : pmesh_(pmesh), sorder_(sorder), rho_(rho), mu_(mu) {}

ConstantPropertyThermoChem::~ConstantPropertyThermoChem() {
  delete thermal_divergence_;
  delete viscosity_;
  delete density_;
  delete fes_;
  delete fec_;
}

void ConstantPropertyThermoChem::initializeSelf() {
  fec_ = new H1_FECollection(sorder_);
  fes_ = new ParFiniteElementSpace(pmesh_, fec_);

  density_ = new ParGridFunction(fes_);
  viscosity_ = new ParGridFunction(fes_);
  thermal_divergence_ = new ParGridFunction(fes_);

  *density_ = rho_;
  *viscosity_ = mu_;
  *thermal_divergence_ = 0.0;

  toFlow_interface_.density = density_;
  toFlow_interface_.viscosity = viscosity_;
  toFlow_interface_.thermal_divergence = thermal_divergence_;

  toTurbModel_interface_.density = density_;
}
