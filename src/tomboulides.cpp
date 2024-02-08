
#include "tomboulides.hpp"

using namespace mfem;

Tomboulides::Tomboulides(mfem::ParMesh *pmesh, int vorder, int porder)
    : pmesh_(pmesh), vorder_(vorder), porder_(porder), dim_(pmesh->Dimension()) {}

Tomboulides::~Tomboulides() {
  delete p_gf_;
  delete pfes_;
  delete pfec_;
  delete u_next_gf_;
  delete u_curr_gf_;
  delete vfes_;
  delete vfec_;
}

void Tomboulides::allocateState() {
  vfec_ = new H1_FECollection(vorder_, dim_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);
  u_curr_gf_ = new ParGridFunction(vfes_);
  u_next_gf_ = new ParGridFunction(vfes_);

  pfec_ = new H1_FECollection(porder_);
  pfes_ = new ParFiniteElementSpace(pmesh_, pfec_);
  p_gf_ = new ParGridFunction(pfes_);

  *u_curr_gf_ = 0.0;
  *u_next_gf_ = 0.0;
  *p_gf_ = 0.0;

  interface.velocity = u_next_gf_;
}

void Tomboulides::initializeSelf() { allocateState(); }
