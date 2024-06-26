// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-
#include "forcing_terms.hpp"

#include <vector>

ForcingTerms::ForcingTerms(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                           IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
                           ParGridFunction *_Up, ParGridFunction *_gradUp,
                           const precomputedIntegrationData &gpu_precomputed_data, bool axisym)
    : dim(_dim),
      nvel(axisym ? 3 : _dim),
      num_equation(_num_equation),
      axisymmetric_(axisym),
      order(_order),
      intRuleType(_intRuleType),
      intRules(_intRules),
      vfes(_vfes),
      U_(U),
      Up_(_Up),
      gradUp_(_gradUp),
      gpu_precomputed_data_(gpu_precomputed_data) {
  const elementIndexingData &elem_data = gpu_precomputed_data_.element_indexing_data;
  h_num_elems_of_type = elem_data.num_elems_of_type.HostRead();

  //   b = new ParGridFunction(vfes);
  //
  //   // Initialize to zero
  //   int dof = vfes->GetNDofs();
  //   double *data = b->HostWrite();
  //   for(int ii=0;ii<dof*num_equation;ii++) data[ii] = 0.;
  //   auto d_b = b->ReadWrite();
}

ForcingTerms::~ForcingTerms() {
  //   delete b;
}

// void ForcingTerms::addForcingIntegrals(Vector &in)
// {
// #ifdef _GPU_
//   const double *data = b->Read();
//   double *d_in = in.Write();
//
//   MFEM_FORALL(n,in.Size(),
//   {
//     d_in[n] += data[n];
//   });
// #else
//   double *dataB = b->GetData();
//   for(int i=0; i<in.Size(); i++)
//   {
//     in[i] = in[i] + dataB[i];
//   }
// #endif
// }

// TODO(kevin): gpu capability.
ConstantPressureGradient::ConstantPressureGradient(const int &_dim, const int &_num_equation, const int &_order,
                                                   const int &_intRuleType, IntegrationRules *_intRules,
                                                   ParFiniteElementSpace *_vfes, ParGridFunction *U,
                                                   ParGridFunction *_Up, ParGridFunction *_gradUp,
                                                   const precomputedIntegrationData &gpu_precomputed_data,
                                                   RunConfiguration &_config, GasMixture *mixture)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      mixture_(mixture) {
#ifdef _GPU_
  grvy_printf(GRVY_ERROR, "ConstantPressureGradient with GPU computing is out-dated and not supported yet.");
  exit(-1);
#endif

  pressGrad.UseDevice(true);
  pressGrad.SetSize(3);
  pressGrad = 0.;
  double *h_pressGrad = pressGrad.HostWrite();
  {
    double *data = _config.GetImposedPressureGradient();
    for (int jj = 0; jj < 3; jj++) h_pressGrad[jj] = data[jj];
  }
  pressGrad.ReadWrite();
}

// ConstantPressureGradient::~ConstantPressureGradient() { delete mixture; }

void ConstantPressureGradient::updateTerms(Vector &in) {
#ifdef _GPU_
  const elementIndexingData &elem_data = gpu_precomputed_data_.element_indexing_data;

  auto h_elem_dof_number = elem_data.dof_number.HostRead();

  for (int elType = 0; elType < elem_data.num_elems_of_type.Size(); elType++) {
    int elemOffset = 0;
    if (elType != 0) {
      for (int i = 0; i < elType; i++) elemOffset += h_num_elems_of_type[i];
    }
    int dof_el = h_elem_dof_number[elemOffset];

    updateTerms_gpu(h_num_elems_of_type[elType], elemOffset, dof_el, vfes->GetNDofs(), pressGrad, in, *Up_, *gradUp_,
                    num_equation, dim, gpu_precomputed_data_);
  }
#else
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  double *data = in.GetData();
  const double *dataUp = Up_->GetData();
  const double *dataGradUp = gradUp_->GetData();

  Vector gradUpk;  // interpolated gradient
  gradUpk.SetSize(num_equation * dim);
  Vector upk;  // interpolated Up
  upk.SetSize(num_equation);

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    // ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);

    Vector vel(nvel), primi(num_equation);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for (int eq = 0; eq < num_equation; eq++) primi[eq] = dataUp[index + eq * dof];
      double p = mixture_->ComputePressureFromPrimitives(primi);
      double grad_pV = 0.;

      // stays dim, not nvel, b/c no pressure gradient in theta direction
      for (int d = 0; d < dim; d++) {
        vel[d] = dataUp[index + (d + 1) * dof];
        data[index + (d + 1) * dof] -= pressGrad[d];
        grad_pV -= vel[d] * pressGrad[d];
        grad_pV -= p * dataGradUp[index + (d + 1) * dof + d * dof * num_equation];
      }

      data[index + (1 + nvel) * dof] += grad_pV;
    }
  }
#endif
}

#ifdef _GPU_
void ConstantPressureGradient::updateTerms_gpu(const int numElems, const int offsetElems, const int elDof,
                                               const int totalDofs, Vector &pressGrad, Vector &in, const Vector &Up,
                                               Vector &gradUp, const int num_equation, const int dim,
                                               const precomputedIntegrationData &gpu_precomputed_data) {
  const double *d_pressGrad = pressGrad.Read();
  double *d_in = in.ReadWrite();

  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();

  const elementIndexingData &elem_data = gpu_precomputed_data.element_indexing_data;
  auto d_elem_dof_off = elem_data.dof_offset.Read();
  auto d_elem_dofs_list = elem_data.dofs_list.Read();

  // clang-format off
  MFEM_FORALL_2D(el, numElems, elDof, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, elDof) {
      MFEM_SHARED double Ui[216 * 5], gradUpi[216 * 5 * 3];
      MFEM_SHARED double pGrad[3];
//      MFEM_SHARED double Ui[gpudata::MAXDOFS * gpudata::MAXEQUATIONS], // MFEM_SHARED double Ui[216 * 5],
//                         gradUpi[gpudata::MAXDOFS * gpudata::MAXEQUATIONS * gpudata::MAXDIM]; // gradUpi[216 * 5 * 3];
//      MFEM_SHARED double pGrad[gpudata::MAXDIM]; // MFEM_SHARED double pGrad[3];

      const int eli = el + offsetElems;
      const int offsetIDs = d_elem_dof_off[eli];

      const int indexi = d_elem_dofs_list[offsetIDs + i];

      if (i < 3) pGrad[i] = d_pressGrad[i];

      // fill out Ui and init gradUpi
      // NOTE: density is not being loaded to save mem. accesses
      for (int eq = 1; eq < num_equation; eq++) {
        Ui[i + eq * elDof] = d_Up[indexi + eq * totalDofs];
      }

      // NOTE: gradients of density and pressure not loaded
      for (int eq = 1; eq < num_equation - 1; eq++) {
        for (int d = 0; d < dim; d++) {
          gradUpi[i + eq * elDof + d * num_equation * elDof] =
            d_gradUp[indexi + eq * totalDofs + d * num_equation * totalDofs];
        }
      }
      MFEM_SYNC_THREAD;

      double grad_pV = 0.;

      for (int d = 0; d < dim; d++) {
        grad_pV -= Ui[i + (1 + d) * elDof] * pGrad[d];
        grad_pV -= Ui[i + (1+dim) * elDof] * gradUpi[i + (1 + d) * elDof + d * num_equation * elDof];
      }

      d_in[indexi +  totalDofs] -= pGrad[0];
      d_in[indexi + 2 * totalDofs] -= pGrad[1];
      if (dim == 3) d_in[indexi + 3 * totalDofs] -= pGrad[2];
      d_in[indexi + (1+dim)*totalDofs] += grad_pV;
    }
  });
}
// clang-format on

#endif

AxisymmetricSource::AxisymmetricSource(const int &_dim, const int &_num_equation, const int &_order,
                                       GasMixture *_mixture, TransportProperties *_transport,
                                       const Equations &_eqSystem, const int &_intRuleType, IntegrationRules *_intRules,
                                       ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up,
                                       ParGridFunction *_gradUp, ParGridFunction *spaceVaryViscMult,
                                       const precomputedIntegrationData &gpu_precomputed_data,
                                       RunConfiguration &_config, ParGridFunction *distance)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      mixture(_mixture),
      transport_(_transport),
      eqSystem(_eqSystem),
      space_vary_viscosity_mult_(spaceVaryViscMult),
      distance_(distance) {
  // no-op
}

void AxisymmetricSource::updateTerms(Vector &in) {
  assert(dim == 2);
  assert(nvel == 3);

  double *d_y = in.ReadWrite();
  const double *d_U = U_->Read();
  const double *d_Up = Up_->Read();
  const double *d_gradUp = gradUp_->Read();

  int dof = vfes->GetNDofs();

  // get coords
  const FiniteElementCollection *fec = vfes->FEColl();
  ParMesh *mesh = vfes->GetParMesh();
  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParGridFunction coordsDof(&dfes);
  mesh->GetNodes(coordsDof);
  auto d_coords = coordsDof.Read();

  const double *alpha;
  if (space_vary_viscosity_mult_ != NULL) {
    alpha = space_vary_viscosity_mult_->Read();
  } else {
    alpha = NULL;
  }

  // and distance
  const double *d_dist;
  if (distance_ != NULL) {
    d_dist = distance_->Read();
  } else {
    d_dist = NULL;
  }

  const int neqn = num_equation;
  const int sdim = dim;

  const Equations eqSys = eqSystem;

  GasMixture *d_mix = mixture;
  TransportProperties *d_trans = transport_;

  MFEM_FORALL(n, dof, {
    double U[gpudata::MAXEQUATIONS];
    double Up[gpudata::MAXEQUATIONS];
    double gradUp[gpudata::MAXDIM * gpudata::MAXEQUATIONS];
    double x[gpudata::MAXDIM];

    // Extract state information
    for (int eq = 0; eq < neqn; eq++) {
      U[eq] = d_U[n + eq * dof];
      Up[eq] = d_Up[n + eq * dof];
      for (int d = 0; d < sdim; d++) {
        gradUp[eq + d * neqn] = d_gradUp[n + eq * dof + d * neqn * dof];
      }
    }

    const int numActiveSpecies = d_mix->GetNumActiveSpecies();
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      int eq = 3 + 2 + sp;
      U[eq] = max(U[eq], 0.0);
      Up[eq] = max(Up[eq], 0.0);
    }

    // Extract radius
    for (int d = 0; d < sdim; d++) x[d] = d_coords[n + d * dof];
    const double radius = x[0];

    // Evaluate quantities required for axisym source
    const double rho = Up[0];
    const double ur = Up[1];
    const double ut = Up[3];

    const double pressure = d_mix->ComputePressureFromPrimitives(Up);

    const double rurut = rho * ur * ut;
    const double rutut = rho * ut * ut;

    double tau_tt, tau_tr;
    if (eqSys == EULER) {
      tau_tt = tau_tr = 0.0;
    } else {
      const double ur_r = gradUp[1 + 0 * neqn];
      const double uz_z = gradUp[2 + 1 * neqn];
      const double ut_r = gradUp[3 + 0 * neqn];

      double visc, bulkVisc, visc_vec[2];

      double dist = 0;
      if (d_dist != NULL) dist = d_dist[n];

      d_trans->GetViscosities(U, Up, gradUp, radius, dist, visc_vec);
      visc = visc_vec[0];
      bulkVisc = visc_vec[1];
      bulkVisc -= 2. / 3. * visc;

      if (alpha != NULL) {
        visc *= alpha[n];
      }

      double divV = ur_r + uz_z;
      if (radius > 0) divV += ur / radius;

      tau_tt = (radius > 0) ? 2.0 * ur / radius * visc : 0.0;
      tau_tt += bulkVisc * divV;

      tau_tr = ut_r;
      if (radius > 0) tau_tr -= ut / radius;

      tau_tr *= visc;
    }

    // Put source into the right spot
    d_y[n + 1 * dof] += (pressure + rutut - tau_tt) / radius;
    d_y[n + 3 * dof] += (-rurut + tau_tr) / radius;

    // if (radius > 0) {
    //   d_y[n + 1 * dof] += (pressure + rutut - tau_tt) / radius;
    //   d_y[n + 3 * dof] += (-rurut + tau_tr) / radius;
    // } else {
    //   // TODO(trevilo): Fix axis
    //   double fake_radius = 1e-3;
    //   d_y[n + 1 * dof] += (pressure - tau_tt) / fake_radius;
    //   d_y[n + 3 * dof] += (tau_tr) / fake_radius;
    // }
  });

  // Commented out code below has the right structure if we want to
  // add axisym source terms to rhs of weak form (as opposed to
  // M^{-1}*rhs).  It is not entirely correct b/c the \tau_{\theta
  // \theta} contribution is neglected, but this could be added if we
  // switch to this formulation.  See comments at L348 of rhs_operator.cpp

  // int numElem = vfes->GetNE();
  // for (int el = 0; el < numElem; el++) {
  //   const FiniteElement *elem = vfes->GetFE(el);
  //   ElementTransformation *Tr = vfes->GetElementTransformation(el);
  //   const int dof_elem = elem->GetDof();

  //   // nodes of the element
  //   Array<int> nodes;
  //   vfes->GetElementVDofs(el, nodes);

  //   const int order = elem->GetOrder();
  //   //cout<<"order :"<<maxorder<<" dof: "<<dof_elem<<endl;
  //   int intorder = 2*order;
  //   const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
  //   for (int k= 0; k< ir->GetNPoints(); k++)
  //     {
  //       const IntegrationPoint &ip = ir->IntPoint(k);

  //       Vector shape(dof_elem);
  //       elem->CalcShape(ip, shape);

  //       double pressure = 0.0;
  //       for (int k = 0; k < dof_elem; k++) pressure += dataUp[nodes[k] + (1+dim)*dof] * shape[k];

  //       Tr->SetIntPoint(&ip);
  //       shape *= Tr->Jacobian().Det()*ip.weight;

  //       // get coordinates of integration point
  //       double x[3];
  //       Vector transip(x, 3);
  //       Tr->Transform(ip,transip);

  //       for (int j= 0;j<dof_elem;j++)
  //         {
  //           int i = nodes[j];
  //           data[i + 1*dof] += pressure*shape[j];
  //         }

  //     }
  // }
}

JouleHeating::JouleHeating(const int &_dim, const int &_num_equation, const int &_order, GasMixture *_mixture,
                           const Equations &_eqSystem, const int &_intRuleType, IntegrationRules *_intRules,
                           ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up,
                           ParGridFunction *_gradUp, const precomputedIntegrationData &gpu_precomputed_data,
                           RunConfiguration &_config, ParGridFunction *jh_)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      eqSystem(_eqSystem),
      joule_heating_(jh_),
      mixture_(_mixture) {
  // no-op
}

void JouleHeating::updateTerms(Vector &in) {
  assert(nvel == 3);

  const int dof = vfes->GetNDofs();
  const int nvel_ = nvel;
  const int neqn = num_equation;

  double *data = in.ReadWrite();
  const double *jh = joule_heating_->Read();

  // NB: This GasMixture is valid on the HOST!
  const bool twoT = mixture_->IsTwoTemperature();

  MFEM_FORALL(n, dof, {
    // Add Joule heating to total energy
    const double heating = jh[n];
    const int e_index = n + (nvel_ + 1) * dof;
    if (heating > 0.) {
      data[e_index] += heating;
    }

    // Add Joule heating to electron energy (assumes ion Joule heating is negligible)
    if (twoT) {
      const int ee_index = n + (neqn - 1) * dof;
      if (heating > 0.) {
        data[ee_index] += heating;
      }
    }
  });
}

// TODO(kevin): implment gpu
SpongeZone::SpongeZone(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       Fluxes *_fluxClass, GasMixture *_mixture, IntegrationRules *_intRules,
                       ParFiniteElementSpace *_vfes, ParGridFunction *U, ParGridFunction *_Up, ParGridFunction *_gradUp,
                       const precomputedIntegrationData &gpu_precomputed_data, RunConfiguration &_config, const int sz)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      fluxes(_fluxClass),
      mixture(_mixture),
      szData(_config.spongeData_[sz]) {
  targetU.SetSize(num_equation);

  if (szData.szSolType == SpongeZoneSolution::USERDEF) {
    Vector conserved(num_equation);
    conserved[0] = szData.targetUp[0];
    // for (int d = 0; d < dim; d++) conserved[1 + d] = szData.targetUp[0] * szData.targetUp[1 + d];
    for (int d = 0; d < nvel; d++) conserved[1 + d] = szData.targetUp[0] * szData.targetUp[1 + d];
    // Kevin: I don't think we can do this.. we should set target temperature.
    // Up[1 + dim] = mixture->Temperature(&Up[0], &szData.targetUp[4], 1);
    // mixture->GetConservativesFromPrimitives(Up, targetU);

    int numActiveSpecies_ = mixture->GetNumActiveSpecies();
    if (numActiveSpecies_ > 0) {  // read species input state for multi-component flow.
      // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
      // NOTE: input Up will always contain 3 velocity components.
      for (int sp = 0; sp < numActiveSpecies_; sp++) {
        // int inputIndex = (*mixtureToInputMap)[sp];
        // store species density into inputState in the order of mixture-sorted index.
        conserved[nvel + 2 + sp] = szData.targetUp[0] * szData.targetUp[5 + sp];
      }
    }

    singleTemperature_ = false;
    if (mixture->IsTwoTemperature()) {
      singleTemperature_ = szData.singleTemperature;
      if (!singleTemperature_) {
        double numSpecies = mixture->GetNumSpecies();
        Vector n_sp(numSpecies);
        mixture->computeNumberDensities(conserved, n_sp);
        conserved[num_equation - 1] =
            mixture->computeElectronEnergy(n_sp(numSpecies - 2), szData.targetUp(5 + numSpecies));
      }
    }
    mixture->modifyEnergyForPressure(conserved, targetU, szData.targetUp[4], singleTemperature_);
  }

  // if (szData.szType == SpongeZoneSolution::USERDEF) {
  //   Vector Up(num_equation);
  //   Up[0] = szData.targetUp[0];
  //   for (int d = 0; d < nvel; d++) Up[1 + d] = szData.targetUp[1 + d];
  //   //Up[1 + dim] = mixture->Temperature(&Up[0], &szData.targetUp[4], 1);
  //   Up[1 + nvel] = mixture->Temperature(&Up[0], &szData.targetUp[4], 1);
  //   mixture->GetConservativesFromPrimitives(Up, targetU);
  // }

  // make sure normal is unitary
  double mod = 0.;
  for (int d = 0; d < dim; d++) mod += szData.normal(d) * szData.normal(d);
  mod = sqrt(mod);
  szData.normal /= mod;

  meanNormalFluxes.SetSize(num_equation + 1);

  ParMesh *mesh = vfes->GetParMesh();
  const FiniteElementCollection *fec = vfes->FEColl();
  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParFiniteElementSpace fes(mesh, fec);

  sigma = new ParGridFunction(&fes);
  double *hSigma = sigma->HostWrite();

  ParGridFunction coords(&dfes);
  mesh->GetNodes(coords);
  int ndofs = vfes->GetNDofs();

  if (szData.szType == SpongeZoneType::ANNULUS) {
    nodesInAnnulus.SetSize(ndofs);
    nodesInAnnulus = -1;
  }

  vector<int> nodesVec;
  vector<double> radialNormalsVec;
  nodesVec.clear();
  radialNormalsVec.clear();
  for (int n = 0; n < ndofs; n++) {
    Vector Xn(dim);
    for (int d = 0; d < dim; d++) Xn[d] = coords[n + d * ndofs];

    hSigma[n] = 0.0;
    if (szData.szType == SpongeZoneType::PLANAR) {
      // distance to the mix-out plane
      double distInit = 0.;
      for (int d = 0; d < dim; d++) distInit -= szData.normal[d] * (Xn[d] - szData.pointInit[d]);

      if (fabs(distInit) < szData.tol) nodesVec.push_back(n);

      // dist end plane
      double distF = 0.;
      for (int d = 0; d < dim; d++) distF += szData.normal[d] * (Xn[d] - szData.point0[d]);

      if (distInit > 0. && distF > 0.) {
        double planeDistance = distF + distInit;
        hSigma[n] = distInit / planeDistance / planeDistance;
      }
    } else if (szData.szType == SpongeZoneType::ANNULUS) {
      double distInit = 0.;
      for (int d = 0; d < dim; d++) distInit -= szData.normal[d] * (Xn[d] - szData.pointInit[d]);

      // dadial distance to axis
      double R = 0.;
      Vector tmp(dim);
      {
        for (int d = 0; d < dim; d++) tmp(d) = Xn[d] - szData.pointInit[d] + distInit * szData.normal(d);
        for (int d = 0; d < dim; d++) R += tmp(d) * tmp(d);
        R = sqrt(R);
      }

      // dist end plane
      double distF = 0.;
      for (int d = 0; d < dim; d++) distF += szData.normal[d] * (Xn[d] - szData.point0[d]);

      // nodes for mixed out plane
      if (fabs(R - szData.r1) < szData.tol) nodesVec.push_back(n);

      if (distInit > 0. && distF > 0. && R - szData.r1 > 0.) {
        double planeDistance = szData.r2 - szData.r1;
        hSigma[n] = (R - szData.r1) / planeDistance / planeDistance;
        for (int d = 0; d < dim; d++) radialNormalsVec.push_back(tmp(d) / R);
        nodesInAnnulus[n] = radialNormalsVec.size() / dim - 1;
      }
    }
  }

  radialNormal.SetSize(radialNormalsVec.size());
  for (int n = 0; n < radialNormal.Size(); n++) radialNormal(n) = radialNormalsVec[n];

  radialNormal.SetSize(radialNormalsVec.size());
  for (int n = 0; n < radialNormal.Size(); n++) radialNormal(n) = radialNormalsVec[n];

  radialNormal.SetSize(radialNormalsVec.size());
  for (int n = 0; n < radialNormal.Size(); n++) radialNormal(n) = radialNormalsVec[n];

  // find plane nodes
  if (szData.szSolType == SpongeZoneSolution::MIXEDOUT) {
    nodesInMixedOutPlane.SetSize(nodesVec.size());
    for (int n = 0; n < nodesInMixedOutPlane.Size(); n++) nodesInMixedOutPlane[n] = nodesVec[n];

    cout << "__________________________________________________" << endl;
    cout << "Running with Mixed-Out Sponge Zone" << endl;
    cout << "       Nodes searched with tolerance: " << szData.tol << endl;
    cout << "       Num. of nodes in mix-out plane: " << nodesInMixedOutPlane.Size() << endl;
    cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
  }
}

SpongeZone::~SpongeZone() { delete sigma; }

void SpongeZone::updateTerms(Vector &in) {
  if (szData.szSolType == SpongeZoneSolution::MIXEDOUT) computeMixedOutValues();

  addSpongeZoneForcing(in);
}

void SpongeZone::addSpongeZoneForcing(Vector &in) {
  const double *ds = sigma->HostRead();
  double *dataIn = in.HostReadWrite();
  const double *dataUp = Up_->HostRead();

  int nnodes = vfes->GetNDofs();

  Vector Un(num_equation), Up(num_equation);
  Vector ur(dim), uth(dim), uz(dim);
  Vector targetCyl(num_equation);
  DenseMatrix MM(dim);

  targetCyl = targetU;

  // compute speed of sound
  // double gamma = mixture->GetSpecificHeatRatio();
  // double Rg = mixture->GetGasConstant();
  mixture->GetPrimitivesFromConservatives(targetU, Up);

  // double speedSound = sqrt(gamma * Rg * Up[1 + nvel]);
  double speedSound = mixture->ComputeSpeedOfSound(Up, true);

  // add forcing to RHS, i.e., @in
  for (int n = 0; n < nnodes; n++) {
    double s = ds[n];
    if (s > 0.) {
      s *= szData.multFactor;
      for (int eq = 0; eq < num_equation; eq++) Up[eq] = dataUp[n + eq * nnodes];
      mixture->GetConservativesFromPrimitives(Up, Un);

      // transform to cylindrical in annular
      if (szData.szType == SpongeZoneType::ANNULUS) {
        const int node = nodesInAnnulus[n];
        for (int d = 0; d < dim; d++) ur(d) = radialNormal(3 * node + d);
        double Vr = 0., Vt = 0., Vx = 0.;
        Vector Vtv(dim);
        for (int d = 0; d < dim; d++) {
          Vx += targetU(1 + d) * szData.normal(d);
          Vr += targetU(1 + d) * ur(d);
          Vtv(d) = targetU(1 + d) - Vx * szData.normal(d) - Vr * ur(d);
          Vt += Vtv(d) * Vtv(d);
        }
        Vt = sqrt(Vt);
        targetCyl(1) = Vr;
        targetCyl(2) = Vt;
        if (dim == 3) targetCyl(3) = Vx;
      }

      // transform to cylindrical in annular
      if (szData.szType == SpongeZoneType::ANNULUS) {
        const int node = nodesInAnnulus[n];
        if (node > -1) {
          for (int d = 0; d < dim; d++) {
            ur(d) = radialNormal(3 * node + d);
            uz(d) = szData.normal(d);
          }
          uth(0) = uz(1) * ur(2) - ur(1) * uz(2);
          uth(1) = uz(2) * ur(0) - uz(0) * ur(2);
          uth(2) = uz(0) * ur(1) - ur(0) * uz(1);

          for (int d = 0; d < dim; d++) {
            MM(0, d) = ur(d);
            MM(1, d) = uth(d);
            MM(2, d) = uz(d);
          }
          MM.Invert();
          MM.Mult(targetU.HostRead() + 1, targetCyl.HostReadWrite() + 1);
        }
      }

      for (int eq = 0; eq < num_equation; eq++) dataIn[n + eq * nnodes] -= speedSound * s * (Un[eq] - targetCyl[eq]);
    }
  }
}

// TODO(kevin): change to conserved variables.
void SpongeZone::computeMixedOutValues() {
  int nnodes = vfes->GetNDofs();
  const double *dataUp = Up_->HostRead();
  // double gamma = mixture->GetSpecificHeatRatio();

  // compute mean normal fluxes
  meanNormalFluxes = 0.;

  Vector Un(num_equation), Up(num_equation);
  DenseMatrix f(num_equation, dim);
  for (int n = 0; n < nodesInMixedOutPlane.Size(); n++) {
    int node = nodesInMixedOutPlane[n];

    for (int eq = 0; eq < num_equation; eq++) Up[eq] = dataUp[node + eq * nnodes];
    mixture->GetConservativesFromPrimitives(Up, Un);
    fluxes->ComputeConvectiveFluxes(Un, f);

    for (int eq = 0; eq < num_equation; eq++)
      for (int d = 0; d < dim; d++) meanNormalFluxes[eq] += szData.normal[d] * f(eq, d);
  }
  meanNormalFluxes[num_equation] = double(nodesInMixedOutPlane.Size());

  // communicate different partitions
  Vector sum(num_equation + 1);
  MPI_Allreduce(meanNormalFluxes.HostRead(), sum.HostWrite(), num_equation + 1, MPI_DOUBLE, MPI_SUM, vfes->GetComm());
  meanNormalFluxes = sum;

  // normalize
  for (int eq = 0; eq < num_equation; eq++) meanNormalFluxes[eq] /= double(meanNormalFluxes[num_equation]);

  mixture->computeConservedStateFromConvectiveFlux(meanNormalFluxes, szData.normal, targetU);

  // // Kevin: we need to find temperature value, rather than pressure.
  // // compute mixed-out variables considering the outlet pressure
  // double temp = 0.;
  // for (int d = 0; d < dim; d++) temp += meanNormalFluxes[1 + d] * szData.normal[d];
  // double A = 1. - 2. * gamma / (gamma - 1.);
  // double B = 2 * temp / (gamma - 1.);
  // double C = -2. * meanNormalFluxes[0] * meanNormalFluxes[1 + dim];
  // for (int d = 0; d < dim; d++) C += meanNormalFluxes[1 + d] * meanNormalFluxes[1 + d];
  // //   double p = (-B+sqrt(B*B-4.*A*C))/(2.*A);
  // double p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);  // real solution
  //
  // Up[0] = meanNormalFluxes[0] * meanNormalFluxes[0] / (temp - p);
  // Up[1 + dim] = mixture->Temperature(&Up[0], &p, 1);
  // //   Up[1+dim] = p;
  //
  // for (int d = 0; d < dim; d++) Up[1 + d] = (meanNormalFluxes[1 + d] - p * szData.normal[d]) / meanNormalFluxes[0];
  //
  // double v2 = 0.;
  // for (int d = 0; d < dim; d++) v2 += Up[1 + d] * Up[1 + d];
  //
  // mixture->GetConservativesFromPrimitives(Up, targetU);
}

PassiveScalar::PassiveScalar(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                             IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, GasMixture *_mixture,
                             ParGridFunction *U, ParGridFunction *_Up, ParGridFunction *_gradUp,
                             const precomputedIntegrationData &gpu_precomputed_data, RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      mixture(_mixture) {
  psData_.DeleteAll();
  for (int i = 0; i < _config.GetPassiveScalarData().Size(); i++) psData_.Append(new passiveScalarData);

  for (int i = 0; i < psData_.Size(); i++) {
    psData_[i]->coords.SetSize(3);
    for (int d = 0; d < 3; d++) psData_[i]->coords[d] = _config.GetPassiveScalarData(i)->coords[d];
    psData_[i]->radius = _config.GetPassiveScalarData(i)->radius;
    psData_[i]->value = _config.GetPassiveScalarData(i)->value;
    psData_[i]->nodes.DeleteAll();
  }

  // find nodes for each passive scalar location
  ParFiniteElementSpace dfes(vfes->GetParMesh(), vfes->FEColl(), dim,
                             Ordering::byNODES);  // Kevin: Had a seg fault from this line.
  ParGridFunction coordinates(&dfes);
  vfes->GetParMesh()->GetNodes(coordinates);

  int nnodes = vfes->GetNDofs();

  for (int i = 0; i < psData_.Size(); i++) {
    Vector x0(dim);
    for (int d = 0; d < dim; d++) x0[d] = psData_[i]->coords[d];

    std::vector<int> list;
    list.clear();

    // loop through nodes to find the ones within radius
    Vector y(dim);
    for (int n = 0; n < nnodes; n++) {
      for (int d = 0; d < dim; d++) y[d] = coordinates[n + d * nnodes];
      double dist = 0.;
      for (int d = 0; d < dim; d++) dist += (y[d] - x0[d]) * (y[d] - x0[d]);
      dist = sqrt(dist);
      if (dist < psData_[i]->radius) list.push_back(n);
    }

    psData_[i]->nodes.SetSize(list.size());
    for (size_t n = 0; n < list.size(); n++) psData_[i]->nodes[n] = list[n];
  }
}

PassiveScalar::~PassiveScalar() {
  for (int i = 0; i < psData_.Size(); i++) delete psData_[i];
}

void PassiveScalar::updateTerms(Vector &in) {
#ifdef _GPU_
  updateTerms_gpu(in, Up_, psData_, vfes->GetNDofs(), num_equation);
#else
  auto dataUp = Up_->HostRead();
  auto dataIn = in.HostReadWrite();
  int nnode = vfes->GetNDofs();

  double Z = 0.;

  for (int i = 0; i < psData_.Size(); i++) {
    Z = psData_[i]->value;
    for (int n = 0; n < psData_[i]->nodes.Size(); n++) {
      int node = psData_[i]->nodes[n];
      double vel = 0.;
      for (int d = 0; d < dim; d++) vel += dataUp[node + (1 + d) * nnode] * dataUp[node + (1 + d) * nnode];
      vel = sqrt(vel);
      dataIn[node + (num_equation - 1) * nnode] -=
          vel * (dataUp[node + (num_equation - 1) * nnode] - dataUp[node] * Z) / psData_[i]->radius;
    }
  }
#endif
}

void PassiveScalar::updateTerms_gpu(Vector &in, ParGridFunction *Up, Array<passiveScalarData *> &psData,
                                    const int nnode, const int num_equation) {
#ifdef _GPU_
  double *d_in = in.ReadWrite();
  const double *d_Up = Up_->Read();

  double Z = 0.;
  double radius = 1.;

  const int locDim = dim;  // GPU code does no take class variables

  for (int i = 0; i < psData.Size(); i++) {
    Z = psData[i]->value;
    radius = psData[i]->radius;
    const int *d_nodes = psData[i]->nodes.Read();
    const int size = psData[i]->nodes.Size();

    MFEM_FORALL(n, size, {
      int node = d_nodes[n];
      double vel = 0.;
      for (int d = 0; d < locDim; d++) vel += d_Up[node + (1 + d) * nnode] * d_Up[node + (1 + d) * nnode];
      vel = sqrt(vel);
      d_in[node + (num_equation - 1) * nnode] -=
          vel * (d_Up[node + (num_equation - 1) * nnode] - d_Up[node] * Z) / radius;
    });
  }
#endif
}

HeatSource::HeatSource(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       heatSourceData &_heatSource, GasMixture *_mixture, mfem::IntegrationRules *_intRules,
                       mfem::ParFiniteElementSpace *_vfes, ParGridFunction *U, mfem::ParGridFunction *_Up,
                       mfem::ParGridFunction *_gradUp, const precomputedIntegrationData &gpu_precomputed_data,
                       RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()),
      mixture_(_mixture),
      heatSource_(_heatSource) {
  // Initialize the nodes where the heat source will be applied
  ParFiniteElementSpace dfes(vfes->GetParMesh(), vfes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  vfes->GetParMesh()->GetNodes(coordinates);

  int nnodes = vfes->GetNDofs();

  std::vector<int> nodesVec;
  nodesVec.clear();

  if (heatSource_.type == "cylinder") {
    // create unit vector from p1 to p2
    Vector norm(dim);
    for (int d = 0; d < dim; d++) norm(d) = heatSource_.point2(d) - heatSource_.point1(d);
    double mod = 0.;
    for (int d = 0; d < dim; d++) mod += norm(d) * norm(d);
    mod = sqrt(mod);
    norm /= mod;

    for (int n = 0; n < nnodes; n++) {
      Vector X(dim);
      for (int d = 0; d < dim; d++) X(d) = coordinates(n + d * nnodes) - heatSource_.point1(d);

      double proj = 0;
      for (int d = 0; d < dim; d++) proj += X(d) * norm(d);

      Vector r(dim);
      for (int d = 0; d < dim; d++) r(d) = X(d) - proj * norm(d);

      double normR = 0.;
      for (int d = 0; d < dim; d++) normR += r(d) * r(d);
      normR = sqrt(normR);

      if (normR < heatSource_.radius && proj > 0 && proj < mod) nodesVec.push_back(n);
    }
  }

  nodeList_.SetSize(nodesVec.size());
  for (size_t n = 0; n < nodesVec.size(); n++) nodeList_[n] = nodesVec[n];
}

void HeatSource::updateTerms(mfem::Vector &in) {
#ifdef _GPU_
  updateTerms_gpu(in);
#else
  const int nnode = vfes->GetNDofs();
  const double heatValue = heatSource_.value;

  for (int n = 0; n < nodeList_.Size(); n++) {
    int node = nodeList_[n];
    int index = node + (dim + 1) * nnode;
    in(index) += heatValue;
  }
#endif
}

void HeatSource::updateTerms_gpu(mfem::Vector &in) {
#ifdef _GPU_
  double *d_in = in.ReadWrite();
  const int *d_nodeList = nodeList_.Read();

  const int dimGPU = dim;
  const int nnode = vfes->GetNDofs();

  const double heatVal = heatSource_.value;

  MFEM_FORALL(n, nodeList_.Size(), {
    const int node = d_nodeList[n];
    const int index = node + (1 + dimGPU) * nnode;
    d_in[index] += heatVal;
  });
#endif
}

#ifdef HAVE_MASA
MASA_forcings::MASA_forcings(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                             IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *U,
                             ParGridFunction *_Up, ParGridFunction *_gradUp,
                             const precomputedIntegrationData &gpu_precomputed_data, RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, U, _Up, _gradUp, gpu_precomputed_data,
                   _config.isAxisymmetric()) {
  // NOTE: This has been taken care of by M2ulPhyS.masaHandler_.
  // initMasaHandler("forcing", dim, _config.GetEquationSystem(), _config.GetViscMult(), _config.mms_name_);
  if (_config.workFluid == DRY_AIR || _config.workFluid == LTE_FLUID) {
    switch (dim) {
      case 2: {
        evaluateForcing_ = &(dryair2d::evaluateForcing);
      } break;
      case 3: {
        evaluateForcing_ = &(dryair3d::evaluateForcing);
      } break;
    }
  } else {
    evaluateForcing_ = &(mms::evaluateForcing);
  }
}

void MASA_forcings::updateTerms(Vector &in) {
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  double *data = in.HostReadWrite();
  // const double *dataUp = Up->GetData();

  // get coords
  const FiniteElementCollection *fec = vfes->FEColl();
  ParMesh *mesh = vfes->GetParMesh();
  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParGridFunction coordsDof(&dfes);
  mesh->GetNodes(coordsDof);

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    // ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);

    Array<double> ip_forcing(num_equation);
    Vector x(dim);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for (int d = 0; d < dim; d++) x[d] = coordsDof[index + d * dof];

      (*evaluateForcing_)(x, time, ip_forcing);

      for (int eq = 0; eq < num_equation; eq++) data[index + eq * dof] += ip_forcing[eq];
    }

    //     const int order = elem->GetOrder();
    //     //cout<<"order :"<<maxorder<<" dof: "<<dof_elem<<endl;
    //     int intorder = 2*order;
    //     const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
    //     for (int k= 0; k< ir->GetNPoints(); k++)
    //     {
    //       const IntegrationPoint &ip = ir->IntPoint(k);
    //
    //       Vector shape(dof_elem);
    //       elem->CalcShape(ip, shape);
    //
    //       Tr->SetIntPoint(&ip);
    //       shape *= Tr->Jacobian().Det()*ip.weight;
    //
    //       // get coordinates of integration point
    //       double x[3];
    //       Vector transip(x, 3);
    //       Tr->Transform(ip,transip);
    //
    // //       for(int i=0;i<3;i++) transip[i] += 1.;
    // //       cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<endl;
    //
    //       Array<double> ip_forcing(num_equation);
    //
    //       // MASA forcing
    //       // ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2]); // rho
    //       // ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2]); // rho*u
    //       // ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2]); // rho*v
    //       // ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2]); // rho*w
    //       // ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2]); // rhp*e
    //
    //       // ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2],time); // rho
    //       // ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2],time); // rho*u
    //       // ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2],time); // rho*v
    //       // ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2],time); // rho*w
    //       // ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2],time); // rhp*e
    //
    //       ip_forcing[0] = MASA::masa_eval_source_rho<double>(x[0],x[1],x[2],time); // rho
    //       ip_forcing[1] = MASA::masa_eval_source_u<double>  (x[0],x[1],x[2],time); // rho*u
    //       ip_forcing[2] = MASA::masa_eval_source_v<double>  (x[0],x[1],x[2],time); // rho*v
    //       ip_forcing[3] = MASA::masa_eval_source_w<double>  (x[0],x[1],x[2],time); // rho*w
    //       ip_forcing[4] = MASA::masa_eval_source_e<double>  (x[0],x[1],x[2],time); // rhp*e
    //
    //       for (int j= 0;j<dof_elem;j++)
    //       {
    //         int i = nodes[j];
    //         for(int eq=0;eq<num_equation;eq++) data[i +eq*dof] += ip_forcing[eq]*shape[j];
    //       }
    //
    //     }
  }
}
#endif  //  HAVE_MASA
