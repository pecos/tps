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
#include "rhs_operator.hpp"

double getRadius(const Vector &pos) { return pos[0]; }
FunctionCoefficient radiusFcn(getRadius);

// Implementation of class RHSoperator
RHSoperator::RHSoperator(int &_iter, const int _dim, const int &_num_equation, const int &_order,
                         const Equations &_eqSystem, double &_max_char_speed, IntegrationRules *_intRules,
                         int _intRuleType, Fluxes *_fluxClass, GasMixture *_mixture, GasMixture *d_mixture,
                         Chemistry *_chemistry, TransportProperties *_transport, Radiation *_radiation,
                         ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_fes,
                         const precomputedIntegrationData &gpu_precomputed_data, const int &_maxIntPoints,
                         const int &_maxDofs, DGNonLinearForm *_A, MixedBilinearForm *_Aflux, ParMesh *_mesh,
                         ParGridFunction *_spaceVaryViscMult, ParGridFunction *U, ParGridFunction *_Up,
                         ParGridFunction *_gradUp, ParFiniteElementSpace *_gradUpfes, GradNonLinearForm *_gradUp_A,
                         BCintegrator *_bcIntegrator, bool &_isSBP, double &_alpha, RunConfiguration &_config,
                         ParGridFunction *pc, ParGridFunction *jh)
    : TimeDependentOperator(_A->Height()),
      config_(_config),
      iter(_iter),
      dim_(_dim),
      nvel(_config.isAxisymmetric() ? 3 : _dim),
      eqSystem(_eqSystem),
      max_char_speed(_max_char_speed),
      num_equation_(_num_equation),
      intRules(_intRules),
      intRuleType(_intRuleType),
      fluxClass(_fluxClass),
      mixture(_mixture),
      d_mixture_(d_mixture),
      transport_(_transport),
      vfes(_vfes),
      fes(_fes),
      gpu_precomputed_data_(gpu_precomputed_data),
      maxIntPoints(_maxIntPoints),
      maxDofs(_maxDofs),
      A(_A),
      Aflux(_Aflux),
      mesh(_mesh),
      spaceVaryViscMult(_spaceVaryViscMult),
      linViscData(_config.GetLinearVaryingData()),
      isSBP(_isSBP),
      alpha(_alpha),
      U_(U),
      Up(_Up),
      plasma_conductivity_(pc),
      joule_heating_(jh),
      gradUp(_gradUp),
      gradUpfes(_gradUpfes),
      gradUp_A(_gradUp_A),
      bcIntegrator(_bcIntegrator) {
  flux.SetSize(vfes->GetNDofs(), dim_, num_equation_);
  z.UseDevice(true);
  z.SetSize(A->Height());
  z = 0.;

  zk.UseDevice(true);
  fk.UseDevice(true);
  fk.SetSize(dim_ * vfes->GetNDofs());
  zk.SetSize(vfes->GetNDofs());

  const elementIndexingData &elem_data = gpu_precomputed_data_.element_indexing_data;
  h_num_elems_of_type = elem_data.num_elems_of_type.HostRead();

  Me_inv.SetSize(vfes->GetNE());
  Me_inv_rad.SetSize(vfes->GetNE());

  posDofInvM.SetSize(2 * vfes->GetNE());
  auto hposDofInvM = posDofInvM.HostWrite();

  forcing.DeleteAll();

  if (_config.thereIsForcing()) {
    forcing.Append(new ConstantPressureGradient(dim_, num_equation_, _order, intRuleType, intRules, vfes, U_, Up,
                                                gradUp, gpu_precomputed_data_, _config, mixture));
  }
  if (_config.GetPassiveScalarData().Size() > 0)
    forcing.Append(new PassiveScalar(dim_, num_equation_, _order, intRuleType, intRules, vfes, mixture, U_, Up, gradUp,
                                     gpu_precomputed_data_, _config));
  if (_config.numSpongeRegions_ > 0) {
    for (int sz = 0; sz < _config.numSpongeRegions_; sz++) {
      forcing.Append(new SpongeZone(dim_, num_equation_, _order, intRuleType, fluxClass, mixture, intRules, vfes, U_,
                                    Up, gradUp, gpu_precomputed_data_, _config, sz));
    }
  }
  if (_config.numHeatSources > 0) {
    for (int s = 0; s < _config.numHeatSources; s++) {
      if (_config.heatSource[s].isEnabled) {
        forcing.Append(new HeatSource(dim_, num_equation_, _order, intRuleType, _config.heatSource[s], mixture,
                                      intRules, vfes, U_, Up, gradUp, gpu_precomputed_data_, _config));
      }
    }
  }

  if (_config.GetWorkingFluid() != WorkingFluid::DRY_AIR) {
    forcing.Append(new SourceTerm(dim_, num_equation_, _order, intRuleType, intRules, vfes, U_, Up, gradUp,
                                  gpu_precomputed_data_, _config, mixture, d_mixture_, _transport, _chemistry,
                                  _radiation, plasma_conductivity_));
  }
#ifdef HAVE_MASA
  if (config_.use_mms_) {
    masaForcingIndex_ = forcing.Size();
    forcing.Append(new MASA_forcings(dim_, num_equation_, _order, intRuleType, intRules, vfes, U_, Up, gradUp,
                                     gpu_precomputed_data_, _config));
  }
#endif

  // not just for axisymmetric
  const FiniteElementCollection *fec = vfes->FEColl();
  dfes = new ParFiniteElementSpace(mesh, fec, dim_, Ordering::byNODES);
  coordsDof = new ParGridFunction(dfes);
  mesh->GetNodes(*coordsDof);

  // element size by dof index
  elSize = new ParGridFunction(fes);
  Array<int> vdofs_here;
  for (int j = 0; j < mesh->GetNE(); j++) {
    fes->GetElementVDofs(j, vdofs_here);
    int Ndofs = vdofs_here.Size();
    for (int n = 0; n < Ndofs; n++) {
      int idx = vdofs_here[n];
      (*elSize)[idx] = mesh->GetElementSize(j, 1);
      // cout << "check 8 " << (*elSize)[idx] << endl; fflush(stdout);
    }
  }

  if (config_.isAxisymmetric()) {
    forcing.Append(new AxisymmetricSource(dim_, num_equation_, _order, d_mixture_, transport_, eqSystem, intRuleType,
                                          intRules, vfes, U_, Up, gradUp, spaceVaryViscMult, gpu_precomputed_data_,
                                          _config));
  }

  if (joule_heating_ != NULL) {
    forcing.Append(new JouleHeating(dim_, num_equation_, _order, mixture, eqSystem, intRuleType, intRules, vfes, U_, Up,
                                    gradUp, gpu_precomputed_data_, _config, joule_heating_));
  }

  std::vector<double> temp, temp_rad;
  temp.clear();
  temp_rad.clear();

  for (int i = 0; i < vfes->GetNE(); i++) {
    // Standard local assembly and inversion for energy mass matrices.
    const int dof = vfes->GetFE(i)->GetDof();
    DenseMatrix Me(dof);
    Me_inv[i] = new DenseMatrix(dof, dof);

    MassIntegrator mi;

    int integrationOrder = 2 * vfes->GetFE(i)->GetOrder();
    const IntegrationRule intRule = intRules->Get(vfes->GetFE(i)->GetGeomType(), integrationOrder);

    mi.SetIntRule(&intRule);
    mi.AssembleElementMatrix(*(vfes->GetFE(i)), *(vfes->GetElementTransformation(i)), Me);

    Me.Invert();
    for (int n = 0; n < dof; n++)
      for (int j = 0; j < dof; j++) (*Me_inv[i])(n, j) = Me(n, j);

    if (config_.isAxisymmetric()) {
      // For axisymmetric solves, we also require the inverse of
      // M_{ij} = \int r \phi_i \phi_j dr dz,
      // where r is the radius.  This is denoted Me_inv_rad here.
      DenseMatrix Me_rad(dof);
      Me_inv_rad[i] = new DenseMatrix(dof, dof);

      MassIntegrator mi_rad(radiusFcn);

      mi_rad.SetIntRule(&intRule);
      mi_rad.AssembleElementMatrix(*(vfes->GetFE(i)), *(vfes->GetElementTransformation(i)), Me_rad);

      Me_rad.Invert();
      for (int n = 0; n < dof; n++)
        for (int j = 0; j < dof; j++) (*Me_inv_rad[i])(n, j) = Me_rad(n, j);
    } else {
      Me_inv_rad[i] = NULL;
    }

    hposDofInvM[2 * i] = temp.size();
    hposDofInvM[2 * i + 1] = dof;
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dof; k++) {
        temp.push_back((*Me_inv[i])(j, k));
      }
    }
    if (config_.isAxisymmetric()) {
      for (int j = 0; j < dof; j++) {
        for (int k = 0; k < dof; k++) {
          temp_rad.push_back((*Me_inv_rad[i])(j, k));
        }
      }
    }
  }

  invMArray.UseDevice(true);
  invMArray.SetSize(temp.size());
  invMArray = 0.;
  auto hinvMArray = invMArray.HostWrite();
  for (int i = 0; i < static_cast<int>(temp.size()); i++) hinvMArray[i] = temp[i];

  if (config_.isAxisymmetric()) {
    invMArray_rad.UseDevice(true);
    invMArray_rad.SetSize(temp.size());
    invMArray_rad = 0.;
    auto hinvMArray_rad = invMArray_rad.HostWrite();
    for (int i = 0; i < static_cast<int>(temp_rad.size()); i++) hinvMArray_rad[i] = temp_rad[i];
  }

  allocateTransferData();

  A->setParallelData(&transferU, &transferGradUp);

  // create gradients object
  gradients = new Gradients(vfes, gradUpfes, dim_, num_equation_, Up, gradUp, mixture, gradUp_A, intRules, intRuleType,
                            gpu_precomputed_data_, Me_inv, invMArray, posDofInvM, maxIntPoints, maxDofs);
  gradients->setParallelData(&transferUp);

  local_timeDerivatives.UseDevice(true);
  local_timeDerivatives.SetSize(num_equation_);
  local_timeDerivatives = 0.;

#ifdef DEBUG
  {
    int elem = 0;
    const int dof = vfes->GetFE(elem)->GetDof();
    DenseMatrix Me(dof);

    // Derivation matrix
    DenseMatrix Dx(dof), Kx(dof);
    Dx = 0.;
    Kx = 0.;
    ElementTransformation *Tr = vfes->GetElementTransformation(elem);
    int integrationOrder = 2 * vfes->GetFE(elem)->GetOrder();
    const IntegrationRule intRule = intRules->Get(vfes->GetFE(elem)->GetGeomType(), integrationOrder);
    for (int k = 0; k < intRule.GetNPoints(); k++) {
      const IntegrationPoint &ip = intRule.IntPoint(k);
      double wk = ip.weight;
      Tr->SetIntPoint(&ip);

      DenseMatrix invJ = Tr->InverseJacobian();
      double detJac = Tr->Jacobian().Det();
      DenseMatrix dshape(dof, 2);
      Vector shape(dof);
      vfes->GetFE(elem)->CalcShape(Tr->GetIntPoint(), shape);
      vfes->GetFE(elem)->CalcDShape(Tr->GetIntPoint(), dshape);

      for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
          Kx(j, i) += shape(i) * detJac * (dshape(j, 0) * invJ(0, 0) + dshape(j, 1) * invJ(1, 0)) * wk;
          Kx(j, i) += shape(i) * detJac * (dshape(j, 0) * invJ(0, 1) + dshape(j, 1) * invJ(1, 1)) * wk;
        }
      }
    }
    mfem::Mult(Kx, *Me_inv[elem], Dx);
    Dx.Transpose();

    // Mass matrix
    MassIntegrator mi;
    mi.SetIntRule(&intRule);
    mi.AssembleElementMatrix(*(vfes->GetFE(elem)), *(vfes->GetElementTransformation(elem)), Me);
    // Me *= 1./detJac;
    cout << "Mass matrix" << endl;
    for (int i = 0; i < Me.Size(); i++) {
      for (int j = 0; j < Me.Size(); j++) {
        if (fabs(Me(i, j)) < 1e-15) {
          cout << "* ";
        } else {
          cout << Me(i, j) << " ";
        }
      }
      cout << endl;
    }

    // Matrix Q=MD
    DenseMatrix Q(Me.Size());
    mfem::Mult(Me, Dx, Q);

    cout.precision(3);
    for (int i = 0; i < Me.Size(); i++) {
      for (int j = 0; j < Me.Size(); j++) {
        if (fabs(Q(i, j) + Q(j, i)) < 1e-5) {
          cout << "* ";
        } else {
          cout << Q(i, j) + Q(j, i) << " ";
        }
      }
      cout << endl;
    }
  }
#endif
}

RHSoperator::~RHSoperator() {
  delete coordsDof;
  delete dfes;
  delete elSize;
  delete gradients;
  // delete[] Me_inv;
  for (int n = 0; n < Me_inv.Size(); n++) delete Me_inv[n];
  for (int n = 0; n < Me_inv_rad.Size(); n++) delete Me_inv_rad[n];
  for (int i = 0; i < forcing.Size(); i++) delete forcing[i];

  if (transferU.requests != NULL) delete[] transferU.requests;
  if (transferUp.requests != NULL) delete[] transferUp.requests;
  if (transferGradUp.requests != NULL) delete[] transferGradUp.requests;

  if (transferU.statuses != NULL) delete[] transferU.statuses;
  if (transferUp.statuses != NULL) delete[] transferUp.statuses;
  if (transferGradUp.statuses != NULL) delete[] transferGradUp.statuses;
}

void RHSoperator::Mult(const Vector &x, Vector &y) const {
  max_char_speed = 0.;

  // Update primite varibales
  updatePrimitives(x);

#ifdef _GPU_
  // GPU version requires the exchange of data before gradient computation
  initNBlockDataTransfer(*Up, vfes, transferUp);
  initNBlockDataTransfer(x, vfes, transferU);

  gradients->computeGradients_domain();
  waitAllDataTransfer(vfes, transferUp);

  gradients->computeGradients_bdr();
  initNBlockDataTransfer(*gradUp, gradUpfes, transferGradUp);
#else
  gradients->computeGradients();
#endif

  // update boundary conditions
  if (bcIntegrator != NULL) bcIntegrator->updateBCMean(Up);

#ifdef _GPU_
  z = 0.;
  A->Mult_domain(x, z);

  waitAllDataTransfer(vfes, transferU);
  waitAllDataTransfer(gradUpfes, transferGradUp);
  A->Mult_bdr(x, z);
#else
  A->Mult(x, z);
#endif

  GetFlux(x, flux);

  for (int eq = 0; eq < num_equation_; eq++) {
#ifdef _GPU_
    RHSoperator::copyDataForFluxIntegration_gpu(z, flux, fk, zk, eq, vfes->GetNDofs(), dim_);
#else
    Vector fk(flux(eq).GetData(), dim_ * vfes->GetNDofs());
    Vector zk(z.HostReadWrite() + eq * vfes->GetNDofs(), vfes->GetNDofs());
#endif

    Aflux->AddMult(fk, zk);
#ifdef _GPU_
    RHSoperator::copyZk2Z_gpu(z, zk, eq, vfes->GetNDofs());
#endif
  }

  // It would be nice for the axisymmetric to add forcing here (i.e.,
  // to add the contribution of the forcing to the rhs of the weak
  // form and then mult by the inverse mass matrix below).  In
  // particular, this avoids a 1/r in the calculation of the forcing,
  // which eliminates any possibility of problems on the axis.  But,
  // given the way all other forcing terms are evaluated currently, we
  // don't do it that way.

  // // add forcing terms
  // for (int i = 0; i < forcing.Size(); i++) {
  //   // NOTE: Do not use RHSoperator::time here b/c it is not correctly
  //   // updated for each RK substep.  Instead, get time from parent
  //   // class using TimeDependentOperator::GetTime().
  //   forcing[i]->setTime(this->GetTime());
  //   forcing[i]->updateTerms(z);
  // }

  // 3. Multiply element-wise by the inverse mass matrices.
#ifdef _GPU_
  const elementIndexingData &elem_data = gpu_precomputed_data_.element_indexing_data;
  auto h_elem_dof_num = elem_data.dof_number.HostRead();
  for (int eltype = 0; eltype < elem_data.num_elems_of_type.Size(); eltype++) {
    int elemOffset = 0;
    if (eltype != 0) {
      for (int i = 0; i < eltype; i++) elemOffset += h_num_elems_of_type[i];
    }
    int dof = h_elem_dof_num[elemOffset];
    const int totDofs = vfes->GetNDofs();

    if (config_.isAxisymmetric()) {
      RHSoperator::multiPlyInvers_gpu(y, z, gpu_precomputed_data_, invMArray_rad, posDofInvM, num_equation_, totDofs,
                                      h_num_elems_of_type[eltype], elemOffset, dof);
    } else {
      RHSoperator::multiPlyInvers_gpu(y, z, gpu_precomputed_data_, invMArray, posDofInvM, num_equation_, totDofs,
                                      h_num_elems_of_type[eltype], elemOffset, dof);
    }
  }

#else
  for (int el = 0; el < vfes->GetNE(); el++) {
    Vector zval;
    Array<int> vdofs;
    const int dof = vfes->GetFE(el)->GetDof();
    DenseMatrix zmat, ymat(dof, num_equation_);

    // Return the vdofs ordered byNODES
    vfes->GetElementVDofs(el, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), dof, num_equation_);
    if (config_.isAxisymmetric()) {
      mfem::Mult(*Me_inv_rad[el], zmat, ymat);
    } else {
      mfem::Mult(*Me_inv[el], zmat, ymat);
    }
    y.SetSubVector(vdofs, ymat.GetData());
  }
#endif

  // add forcing terms
  for (int i = 0; i < forcing.Size(); i++) {
#ifdef HAVE_MASA
    if ((i == masaForcingIndex_) && (config_.mmsCompareRhs_)) continue;
#endif
    // NOTE: Do not use RHSoperator::time here b/c it is not correctly
    // updated for each RK substep.  Instead, get time from parent
    // class using TimeDependentOperator::GetTime().
    forcing[i]->setTime(this->GetTime());
    forcing[i]->updateTerms(y);
  }

  computeMeanTimeDerivatives(y);
}

void RHSoperator::copyZk2Z_gpu(Vector &z, Vector &zk, const int eq, const int dof) {
#ifdef _GPU_
  const double *d_zk = zk.Read();
  double *d_z = z.ReadWrite();

  MFEM_FORALL(n, dof, { d_z[n + eq * dof] = d_zk[n]; });
#endif
}

void RHSoperator::copyDataForFluxIntegration_gpu(const Vector &z, DenseTensor &flux, Vector &fk, Vector &zk,
                                                 const int eq, const int dof, const int dim) {
#ifdef _GPU_
  const double *d_flux = flux.Read();
  const double *d_z = z.Read();
  double *d_fk = fk.Write();
  double *d_zk = zk.Write();

  MFEM_FORALL(n, dof, {
    d_zk[n] = d_z[n + eq * dof];
    for (int d = 0; d < dim; d++) {
      d_fk[n + d * dof] = d_flux[n + d * dof + eq * dof * dim];
    }
  });
#endif
}

// Compute the flux at solution nodes.
void RHSoperator::GetFlux(const Vector &x, DenseTensor &flux) const {
#ifdef _GPU_

  GetFlux_gpu(x, flux);
#else

  DenseMatrix xmat(x.GetData(), vfes->GetNDofs(), num_equation_);
  DenseMatrix f(num_equation_, dim_);

  double *dataGradUp = gradUp->GetData();

  const int dof = flux.SizeI();
  const int dim = flux.SizeJ();

  Vector state(num_equation_);

  for (int i = 0; i < dof; i++) {
    for (int k = 0; k < num_equation_; k++) (state)(k) = xmat(i, k);
    DenseMatrix gradUpi(num_equation_, dim);
    for (int eq = 0; eq < num_equation_; eq++) {
      for (int d = 0; d < dim; d++) gradUpi(eq, d) = dataGradUp[i + eq * dof + d * num_equation_ * dof];
    }

    double radius = 1;
    double delta;
    Vector xyz(dim_);
    if (config_.isAxisymmetric()) {
      radius = (*coordsDof)[i + 0 * dof];
    }
    for (int d = 0; d < dim_; d++) xyz[d] = (*coordsDof)[i + d * dof];
    delta = (*elSize)[i];

    fluxClass->ComputeConvectiveFluxes(state, f);

    // TODO(Kevin): - This needs to be incorporated in Fluxes::ComputeConvectiveFluxes.
    // Kevin: we need to think through this.
    if (isSBP) {
      // NOTE(kevin): SBP part was never implemented for gpu, and it is outdated for cpu.
      grvy_printf(GRVY_ERROR, "SBP capability is currently outdated and unsupported!\n");
      exit(-1);

      f *= alpha;  // *= alpha
      double p = mixture->ComputePressure(state);
      p *= 1. - alpha;
      for (int d = 0; d < dim; d++) {
        f(d + 1, d) += p;
        f(dim + 1, d) += p * (state)(1 + d) / (state)(0);
      }
    }

    if (eqSystem != EULER) {
      DenseMatrix fvisc(num_equation_, dim);
      fluxClass->ComputeViscousFluxes(state, gradUpi, radius, xyz, delta, fvisc);

      // TODO(kevin): This needs to be incorporated in Fluxes::ComputeViscousFluxes.
      if (spaceVaryViscMult != NULL) {
        auto *alpha = spaceVaryViscMult->GetData();
        for (int eq = 0; eq < num_equation_; eq++)
          for (int d = 0; d < dim; d++) fvisc(eq, d) *= alpha[i];
      }
      f -= fvisc;
    }

    for (int d = 0; d < dim; d++) {
      for (int k = 0; k < num_equation_; k++) {
        flux(i, d, k) = f(k, d);
      }
    }

    // Update max char speed
    const double mcs = mixture->ComputeMaxCharSpeed(state);
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }
  }
#endif  // _GPU_

  double partition_max_char = max_char_speed;
  MPI_Allreduce(&partition_max_char, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
}

void RHSoperator::GetFlux_gpu(const Vector &x, DenseTensor &flux) const {
  auto dataIn = x.Read();
  auto d_flux = flux.Write();

  const int dof = vfes->GetNDofs();
  const int dim = dim_;
  const int num_equation = num_equation_;

  const bool axisymmetric = config_.isAxisymmetric();
  const double *d_coord;
  if (axisymmetric) {
    d_coord = coordsDof->Read();
  } else {
    d_coord = NULL;
  }
  const double *d_gradUp = gradUp->Read();

  const double *d_spaceVaryViscMult;
  if (spaceVaryViscMult != NULL) {
    d_spaceVaryViscMult = spaceVaryViscMult->Read();
  } else {
    d_spaceVaryViscMult = NULL;
  }

  Fluxes *d_fluxClass = fluxClass;

  MFEM_FORALL(n, dof, {
    double Un[gpudata::MAXEQUATIONS];  // double Un[20];
    double fluxn[gpudata::MAXEQUATIONS * gpudata::MAXDIM], fvisc[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double gradUpn[gpudata::MAXEQUATIONS * gpudata::MAXDIM];
    double linVisc;

    for (int eq = 0; eq < num_equation; eq++) {
      Un[eq] = dataIn[n + eq * dof];
      for (int d = 0; d < dim; d++) {
        gradUpn[eq + d * num_equation] = d_gradUp[n + eq * dof + d * num_equation * dof];
      }
    }

    d_fluxClass->ComputeConvectiveFluxes(Un, fluxn);

    /*
    // TODO(kevin): implement radius.
    // Is this correct?
    double radius = 1.0;
    if (axisymmetric) {
      radius = d_coord[n + 0 * dof];
    }
    */

    double radius = 1.0;
    double delta = 0.0;
    double xyz[dim_];
    // xyz = 0.;
    if (axisymmetric) {
      // radius = (*coordsDof)[n + 0 * dof];
      radius = d_coord[n + 0 * dof];
    }
    // for (int d = 0; d < dim_; d++) xyz[d] = (*coordsDof)[n + d * dof];
    for (int d = 0; d < dim; d++) xyz[d] = d_coord[n + d * dof];

    d_fluxClass->ComputeViscousFluxes(Un, gradUpn, radius, xyz, delta, fvisc);

    if (d_spaceVaryViscMult != NULL) {
      linVisc = d_spaceVaryViscMult[n];

      for (int d = 0; d < dim; d++) {
        for (int eq = 0; eq < num_equation; eq++) {
          fvisc[eq + d * num_equation] *= linVisc;
        }
      }
    }

    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) {
        d_flux[n + d * dof + eq * dof * dim] = fluxn[eq + d * num_equation] - fvisc[eq + d * num_equation];
      }
    }
  });
}

void RHSoperator::updatePrimitives(const Vector &x_in) const {
#ifdef _GPU_
  auto dataUp = Up->Write();  // make sure data is available in GPU
  auto dataIn = x_in.Read();  // make sure data is available in GPU

  const int ndofs = vfes->GetNDofs();
  const int num_equation = num_equation_;

  GasMixture *d_mix = d_mixture_;

  MFEM_FORALL(n, ndofs, {
    double state[gpudata::MAXEQUATIONS],
        prim[gpudata::MAXEQUATIONS];  // double state[20];

    for (int eq = 0; eq < num_equation; eq++) state[eq] = dataIn[n + eq * ndofs];
    d_mix->GetPrimitivesFromConservatives(state, prim);
    for (int eq = 0; eq < num_equation; eq++) dataUp[n + eq * ndofs] = prim[eq];
  });
#else
  double *dataUp = Up->GetData();
  for (int i = 0; i < vfes->GetNDofs(); i++) {
    Vector iState(num_equation_);
    Vector primitiveState(num_equation_);
    for (int eq = 0; eq < num_equation_; eq++) iState[eq] = x_in[i + eq * vfes->GetNDofs()];
    mixture->GetPrimitivesFromConservatives(iState, primitiveState);
    for (int eq = 0; eq < num_equation_; eq++) dataUp[i + eq * vfes->GetNDofs()] = primitiveState[eq];
  }
#endif  // _GPU_
}

void RHSoperator::updateGradients(const Vector &x, const bool &primitiveUpdated) const {
  // Update primite varibales
  if (!primitiveUpdated) updatePrimitives(x);

#ifdef _GPU_
  // GPU version requires the exchange of data before gradient computation
  initNBlockDataTransfer(*Up, vfes, transferUp);
  initNBlockDataTransfer(x, vfes, transferU);

  gradients->computeGradients_domain();
  waitAllDataTransfer(vfes, transferUp);

  gradients->computeGradients_bdr();
  initNBlockDataTransfer(*gradUp, gradUpfes, transferGradUp);

  waitAllDataTransfer(vfes, transferU);
  waitAllDataTransfer(gradUpfes, transferGradUp);
#else
  gradients->computeGradients();
#endif
}

void RHSoperator::multiPlyInvers_gpu(Vector &y, Vector &z, const precomputedIntegrationData &gpu_precomputed_data,
                                     const Vector &invMArray, const Array<int> &posDofInvM, const int num_equation,
                                     const int totNumDof, const int NE, const int elemOffset, const int dof) {
#ifdef _GPU_
  double *d_y = y.ReadWrite();
  const double *d_z = z.Read();

  const elementIndexingData &elem_data = gpu_precomputed_data.element_indexing_data;
  auto d_elem_dofs_list = elem_data.dofs_list.Read();
  auto d_elem_dof_off = elem_data.dof_offset.Read();
  auto d_posDofInvM = posDofInvM.Read();
  const double *d_invM = invMArray.Read();

  MFEM_FORALL_2D(el, NE, dof, 1, 1, {
    MFEM_SHARED double data[gpudata::MAXDOFS * gpudata::MAXEQUATIONS];  // MFEM_SHARED double data[216 * 20];

    int eli = el + elemOffset;
    int offsetInv = d_posDofInvM[2 * eli];
    int offsetIds = d_elem_dof_off[eli];

    MFEM_FOREACH_THREAD(i, x, dof) {
      int index = d_elem_dofs_list[offsetIds + i];
      for (int eq = 0; eq < num_equation; eq++) {
        data[i + eq * dof] = d_z[index + eq * totNumDof];
      }
    }
    MFEM_SYNC_THREAD;

    MFEM_FOREACH_THREAD(i, x, dof) {
      int index = d_elem_dofs_list[offsetIds + i];
      for (int eq = 0; eq < num_equation; eq++) {
        double tmp = 0.;
        for (int k = 0; k < dof; k++) tmp += d_invM[offsetInv + i * dof + k] * data[k + eq * dof];
        d_y[index + eq * totNumDof] = tmp;
      }
    }
    MFEM_SYNC_THREAD;
  });
#endif
}

void RHSoperator::allocateTransferData() {
  ParFiniteElementSpace *pfes = Up->ParFESpace();
  ParMesh *mesh = pfes->GetParMesh();
  mesh->ExchangeFaceNbrNodes();
  mesh->ExchangeFaceNbrData();
  vfes->ExchangeFaceNbrData();
  ParFiniteElementSpace *gradFes = gradUp->ParFESpace();
  gradFes->ExchangeFaceNbrData();

  const int Nshared = mesh->GetNSharedFaces();
  if (Nshared > 0) {
    // data transfer arrays
    vfes->ExchangeFaceNbrData();
    gradUpfes->ExchangeFaceNbrData();
    transferU.face_nbr_data.UseDevice(true);
    transferU.send_data.UseDevice(true);
    transferU.face_nbr_data.SetSize(vfes->GetFaceNbrVSize());
    transferU.send_data.SetSize(vfes->send_face_nbr_ldof.Size_of_connections());
    transferU.face_nbr_data = 0.;
    transferU.send_data = 0.;
    transferU.num_face_nbrs = mesh->GetNFaceNeighbors();

    transferUp.face_nbr_data.UseDevice(true);
    transferUp.send_data.UseDevice(true);
    transferUp.face_nbr_data.SetSize(vfes->GetFaceNbrVSize());
    transferUp.send_data.SetSize(vfes->send_face_nbr_ldof.Size_of_connections());
    transferUp.face_nbr_data = 0.;
    transferUp.send_data = 0.;
    transferUp.num_face_nbrs = mesh->GetNFaceNeighbors();

    transferGradUp.face_nbr_data.UseDevice(true);
    transferGradUp.send_data.UseDevice(true);
    transferGradUp.face_nbr_data.SetSize(gradUpfes->GetFaceNbrVSize());
    transferGradUp.send_data.SetSize(gradUpfes->send_face_nbr_ldof.Size_of_connections());
    transferGradUp.face_nbr_data = 0.;
    transferGradUp.send_data = 0.;
    transferGradUp.num_face_nbrs = mesh->GetNFaceNeighbors();

    MPI_Request *requestsU = new MPI_Request[2 * mesh->GetNFaceNeighbors()];
    MPI_Request *requestsUp = new MPI_Request[2 * mesh->GetNFaceNeighbors()];
    MPI_Request *requGradUp = new MPI_Request[2 * mesh->GetNFaceNeighbors()];
    transferU.requests = requestsU;
    transferUp.requests = requestsUp;
    transferGradUp.requests = requGradUp;

    transferU.statuses = new MPI_Status[transferU.num_face_nbrs];
    transferUp.statuses = new MPI_Status[transferUp.num_face_nbrs];
    transferGradUp.statuses = new MPI_Status[transferGradUp.num_face_nbrs];
  } else {
    transferU.requests = NULL;
    transferUp.requests = NULL;
    transferGradUp.requests = NULL;

    transferU.statuses = NULL;
    transferUp.statuses = NULL;
    transferGradUp.statuses = NULL;
  }
}

void RHSoperator::initNBlockDataTransfer(const Vector &x, ParFiniteElementSpace *pfes,
                                         dataTransferArrays &dataTransfer) {
  if (pfes->GetFaceNbrVSize() <= 0) {
    return;
  }

  ParMesh *pmesh = pfes->GetParMesh();

  //    face_nbr_data.SetSize(pfes->GetFaceNbrVSize());
  //    send_data.SetSize(pfes->send_face_nbr_ldof.Size_of_connections());

  int *send_offset = pfes->send_face_nbr_ldof.GetI();
  const int *d_send_ldof = mfem::Read(pfes->send_face_nbr_ldof.GetJMemory(), dataTransfer.send_data.Size());
  int *recv_offset = pfes->face_nbr_ldof.GetI();
  MPI_Comm MyComm = pfes->GetComm();

  //    int num_face_nbrs = pmesh->GetNFaceNeighbors();
  //    MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
  //    MPI_Request *send_requests = requests;
  //    MPI_Request *recv_requests = requests + num_face_nbrs;
  //    MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

  //    auto d_data = this->Read();
  auto d_data = x.Read();
  auto d_send_data = dataTransfer.send_data.Write();
  MFEM_FORALL(i, dataTransfer.send_data.Size(), {
    const int ldof = d_send_ldof[i];
    d_send_data[i] = d_data[ldof >= 0 ? ldof : -1 - ldof];
  });

  const bool mpi_gpu_aware = Device::GetGPUAwareMPI();

  if (mpi_gpu_aware) MFEM_STREAM_SYNC;

  auto send_data_ptr = mpi_gpu_aware ? dataTransfer.send_data.Read() : dataTransfer.send_data.HostRead();
  auto face_nbr_data_ptr = mpi_gpu_aware ? dataTransfer.face_nbr_data.Write() : dataTransfer.face_nbr_data.HostWrite();

  for (int fn = 0; fn < dataTransfer.num_face_nbrs; fn++) {
    int nbr_rank = pmesh->GetFaceNbrRank(fn);
    int tag = 0;

    MPI_Isend(&send_data_ptr[send_offset[fn]], send_offset[fn + 1] - send_offset[fn], MPI_DOUBLE, nbr_rank, tag, MyComm,
              &dataTransfer.requests[fn]);

    MPI_Irecv(&face_nbr_data_ptr[recv_offset[fn]], recv_offset[fn + 1] - recv_offset[fn], MPI_DOUBLE, nbr_rank, tag,
              MyComm, &dataTransfer.requests[fn + dataTransfer.num_face_nbrs]);
  }
}

void RHSoperator::waitAllDataTransfer(ParFiniteElementSpace *pfes, dataTransferArrays &dataTransfer) {
  if (pfes->GetFaceNbrVSize() <= 0) {
    return;
  }

  MPI_Waitall(dataTransfer.num_face_nbrs, dataTransfer.requests, dataTransfer.statuses);
  MPI_Waitall(dataTransfer.num_face_nbrs, dataTransfer.requests + dataTransfer.num_face_nbrs, dataTransfer.statuses);
}

void RHSoperator::computeMeanTimeDerivatives(Vector &y) const {
  if (iter % 100 == 0) {
#ifdef _GPU_
    int Ndof = y.Size() / num_equation_;
    meanTimeDerivatives_gpu(y, local_timeDerivatives, z, Ndof, num_equation_, dim_);
#else
    local_timeDerivatives = 0.;
    int Ndof = y.Size() / num_equation_;

    for (int n = 0; n < Ndof; n++) {
      for (int eq = 0; eq < num_equation_; eq++) {
        local_timeDerivatives[eq] += fabs(y[n + eq * Ndof]) / (static_cast<double>(Ndof));
      }
    }
#endif
  }
}

void RHSoperator::meanTimeDerivatives_gpu(Vector &y, Vector &local_timeDerivatives, Vector &tmp_vec, const int &NDof,
                                          const int &num_equation, const int &dim) {
#ifdef _GPU_

  auto d_y = y.Read();
  auto d_loc = local_timeDerivatives.Write();
  auto d_tmp = tmp_vec.Write();

  // copy values to temp vector
  MFEM_FORALL(n, y.Size(), { d_tmp[n] = fabs(d_y[n]); });

  // sum up all values
  MFEM_FORALL(n, NDof, {
    int interval = 1;
    while (interval < NDof) {
      interval *= 2;
      if (n % interval == 0) {
        int n2 = n + interval / 2;
        for (int eq = 0; eq < num_equation; eq++) {
          if (n2 < NDof) d_tmp[n + eq * NDof] += d_tmp[n2 + eq * NDof];
        }
      }
      MFEM_SYNC_THREAD;
    }
  });

  // transfer to smaller vector
  MFEM_FORALL(eq, num_equation, {
    double ddof = (double)NDof;
    d_loc[eq] = d_tmp[eq * NDof] / ddof;
  });

  // rearrange
  if (dim == 2) {
    MFEM_FORALL(eq, num_equation, {
      if (eq == 0) {
        d_loc[4] = d_loc[3];
        d_loc[3] = 0.;
      }
    });
  }
#endif
}
