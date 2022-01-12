// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
                           IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                           ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &_gpuArrays)
    : dim(_dim),
      num_equation(_num_equation),
      order(_order),
      intRuleType(_intRuleType),
      intRules(_intRules),
      vfes(_vfes),
      Up(_Up),
      gradUp(_gradUp),
      gpuArrays(_gpuArrays) {
  h_numElems = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();

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

ConstantPressureGradient::ConstantPressureGradient(const int &_dim, const int &_num_equation, const int &_order,
                                                   const int &_intRuleType, IntegrationRules *_intRules,
                                                   ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                                                   ParGridFunction *_gradUp,
                                                   const volumeFaceIntegrationArrays &_gpuArrays,
                                                   RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, _Up, _gradUp, _gpuArrays) {
  
  mixture = new DryAir(_config,dim);
  
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

ConstantPressureGradient::~ConstantPressureGradient()
{
  delete mixture;
}


void ConstantPressureGradient::updateTerms(Vector &in) {
#ifdef _GPU_
  for (int elType = 0; elType < gpuArrays.numElems.Size(); elType++) {
    int elemOffset = 0;
    if (elType != 0) {
      for (int i = 0; i < elType; i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2 * elemOffset + 1];

    updateTerms_gpu(h_numElems[elType], elemOffset, dof_el, vfes->GetNDofs(), pressGrad, in, *Up, *gradUp, num_equation,
                    dim, gpuArrays);
  }
#else
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  double *data = in.GetData();
  const double *dataUp = Up->GetData();
  const double *dataGradUp = gradUp->GetData();

  Vector gradUpk;  // interpolated gradient
  gradUpk.SetSize(num_equation * dim);
  Vector upk;  // interpolated Up
  upk.SetSize(num_equation);
  double grad_pV;

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);

    Vector vel(dim), primi(num_equation);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for(int eq=0;eq<num_equation;eq++) primi[eq] = dataUp[index + eq * dof];
      double p = mixture->ComputePressureFromPrimitives(primi);
      double grad_pV = 0.;

      for (int d = 0; d < dim; d++) {
        vel[d] = dataUp[index + (d + 1) * dof];
        data[index + (d + 1) * dof] -= pressGrad[d];
        grad_pV -= vel[d] * pressGrad[d];
        grad_pV -= p * dataGradUp[index + (d + 1) * dof + d * dof * num_equation];
      }

      data[index + (1 + dim) * dof] += grad_pV;
    }
  }
#endif
}

#ifdef _GPU_
void ConstantPressureGradient::updateTerms_gpu(const int numElems, const int offsetElems, const int elDof,
                                               const int totalDofs, Vector &pressGrad, Vector &in, const Vector &Up,
                                               Vector &gradUp, const int num_equation, const int dim,
                                               const volumeFaceIntegrationArrays &gpuArrays) {
  const double *d_pressGrad = pressGrad.Read();
  double *d_in = in.ReadWrite();

  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();

  // clang-format off
  MFEM_FORALL_2D(el, numElems, elDof, 1, 1, {
    MFEM_FOREACH_THREAD(i, x, elDof) {
      MFEM_SHARED double Ui[216 * 5], gradUpi[216 * 5 * 3];
      MFEM_SHARED double pGrad[3];

      const int eli = el + offsetElems;
      const int offsetIDs    = d_posDofIds[2 * eli];

      const int indexi = d_nodesIDs[offsetIDs + i];

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

SpongeZone::SpongeZone(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                       Fluxes *_fluxClass, GasMixture *_mixture, IntegrationRules *_intRules,
                       ParFiniteElementSpace *_vfes, ParGridFunction *_Up, ParGridFunction *_gradUp,
                       const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, _Up, _gradUp, gpuArrays),
      fluxes(_fluxClass),
      mixture(_mixture),
      szData(_config.GetSpongeZoneData()) {
  targetU.SetSize(num_equation);
  if (szData.szType == SpongeZoneSolution::USERDEF) {
    Vector Up(num_equation);
    Up[0] = szData.targetUp[0];
    for (int d = 0; d < dim; d++) Up[1 + d] = szData.targetUp[1 + d];
    Up[1 + dim] =  mixture->Temperature(&Up[0],&szData.targetUp[4],1);
    mixture->GetConservativesFromPrimitives(Up, targetU);
  }

  meanNormalFluxes.SetSize(num_equation + 1);

  ParMesh *mesh = vfes->GetParMesh();
  const FiniteElementCollection *fec = vfes->FEColl();
  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParFiniteElementSpace fes(mesh, fec);

  sigma = new ParGridFunction(&fes);
  *sigma = 0.;
  double *hSigma = sigma->HostWrite();

  ParGridFunction coords(&dfes);
  mesh->GetNodes(coords);
  int ndofs = vfes->GetNDofs();

  vector<int> nodesVec;
  nodesVec.clear();
  for (int n = 0; n < ndofs; n++) {
    Vector Xn(dim);
    for (int d = 0; d < dim; d++) Xn[d] = coords[n + d * ndofs];

    // distance to the mix-out plane
    double dist = 0.;
    for (int d = 0; d < dim; d++) dist += szData.normal[d] * (Xn[d] - szData.pointInit[d]);

    if (fabs(dist) < szData.tol) nodesVec.push_back(n);

    // dist end plane
    double distF = 0.;
    for (int d = 0; d < dim; d++) distF += szData.normal[d] * (Xn[d] - szData.point0[d]);

    if (dist < 0. && distF > 0.) {
      double planeDistance = distF - dist;
      hSigma[n] = -dist / planeDistance / planeDistance;
    }
  }

  // find plane nodes
  if (szData.szType == SpongeZoneSolution::MIXEDOUT) {
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
  if (szData.szType == SpongeZoneSolution::MIXEDOUT) computeMixedOutValues();

  addSpongeZoneForcing(in);
}

void SpongeZone::addSpongeZoneForcing(Vector &in) {
  const double *ds = sigma->HostRead();
  double *dataIn = in.HostReadWrite();
  const double *dataUp = Up->HostRead();

  int nnodes = vfes->GetNDofs();

  Vector Un(num_equation), Up(num_equation);

  // compute speed of sound
  double gamma = mixture->GetSpecificHeatRatio();
  mixture->GetPrimitivesFromConservatives(targetU, Up);
  double speedSound = sqrt(gamma * Up[1+dim] / Up[0]);

  // add forcing to RHS, i.e., @in
  for (int n = 0; n < nnodes; n++) {
    double s = ds[n];
    if (s > 0.) {
      s *= szData.multFactor;
      for (int eq = 0; eq < num_equation; eq++) Up[eq] = dataUp[n + eq * nnodes];
      mixture->GetConservativesFromPrimitives(Up, Un);

      for (int eq = 0; eq < num_equation; eq++) dataIn[n + eq * nnodes] -= speedSound * s * (Un[eq] - targetU[eq]);
    }
  }
}

void SpongeZone::computeMixedOutValues() {
  int nnodes = vfes->GetNDofs();
  const double *dataUp = Up->HostRead();
  double gamma = mixture->GetSpecificHeatRatio();

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
  MPI_Allreduce(meanNormalFluxes.HostRead(), sum.HostWrite(), num_equation + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  meanNormalFluxes = sum;

  // normalize
  for (int eq = 0; eq < num_equation; eq++) meanNormalFluxes[eq] /= double(meanNormalFluxes[num_equation]);

  // compute mixed-out variables considering the outlet pressure
  double temp = 0.;
  for (int d = 0; d < dim; d++) temp += meanNormalFluxes[1 + d] * szData.normal[d];
  double A = 1. - 2. * gamma / (gamma - 1.);
  double B = 2 * temp / (gamma - 1.);
  double C = -2. * meanNormalFluxes[0] * meanNormalFluxes[1+dim];
  for (int d = 0; d < dim; d++) C += meanNormalFluxes[1 + d] * meanNormalFluxes[1 + d];
  //   double p = (-B+sqrt(B*B-4.*A*C))/(2.*A);
  double p = (-B - sqrt(B * B - 4. * A * C)) / (2. * A);  // real solution

  Up[0] = meanNormalFluxes[0] * meanNormalFluxes[0] / (temp - p);
  Up[1+dim] = mixture->Temperature(&Up[0],&p,1);
//   Up[1+dim] = p;

  for (int d = 0; d < dim; d++) Up[1 + d] = (meanNormalFluxes[1 + d] - p * szData.normal[d]) / meanNormalFluxes[0];

  double v2 = 0.;
  for (int d = 0; d < dim; d++) v2 += Up[1 + d] * Up[1 + d];

  mixture->GetConservativesFromPrimitives(Up, targetU);
}

PassiveScalar::PassiveScalar(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                             IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, GasMixture *_mixture,
                             ParGridFunction *_Up, ParGridFunction *_gradUp,
                             const volumeFaceIntegrationArrays &gpuArrays, RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, _Up, _gradUp, gpuArrays),
    mixture(_mixture){
  
  psData.DeleteAll();
  for(int i=0;i<_config.GetPassiveScalarData().Size();i++) psData.Append( new passiveScalarData );
  
  for (int i = 0; i < psData.Size(); i++) {
    psData[i]->coords.SetSize(3);
    for (int d = 0; d < 3; d++) psData[i]->coords[d] = _config.GetPassiveScalarData(i)->coords[d];
    psData[i]->radius = _config.GetPassiveScalarData(i)->radius;
    psData[i]->value = _config.GetPassiveScalarData(i)->value;
    psData[i]->nodes.DeleteAll();
  }

  // find nodes for each passive scalar location
  ParFiniteElementSpace dfes(vfes->GetParMesh(), vfes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  vfes->GetParMesh()->GetNodes(coordinates);

  int nnodes = vfes->GetNDofs();

  for (int i = 0; i < psData.Size(); i++) {
    Vector x0(dim);
    for (int d = 0; d < dim; d++) x0[d] = psData[i]->coords[d];

    std::vector<int> list;
    list.clear();

    // loop through nodes to find the ones within radius
    Vector y(dim);
    for (int n = 0; n < nnodes; n++) {
      for (int d = 0; d < dim; d++) y[d] = coordinates[n + d * nnodes];
      double dist = 0.;
      for (int d = 0; d < dim; d++) dist += (y[d] - x0[d]) * (y[d] - x0[d]);
      dist = sqrt(dist);
      if (dist < psData[i]->radius) list.push_back(n);
    }

    psData[i]->nodes.SetSize(list.size());
    for (int n = 0; n < list.size(); n++) psData[i]->nodes[n] = list[n];
  }
}


PassiveScalar::~PassiveScalar()
{
  for(int i=0; i<psData.Size();i++) delete psData[i];
}


void PassiveScalar::updateTerms(Vector &in) {
  auto dataUp = Up->HostRead();
  auto dataIn = in.HostReadWrite();
  int nnode = vfes->GetNDofs();

  double Z = 0.;

  for (int i = 0; i < psData.Size(); i++) {
    Z = psData[i]->value;
    for (int n = 0; n < psData[i]->nodes.Size(); n++) {
      int node = psData[i]->nodes[n];
      double vel = 0.;
      for (int d = 0; d < dim; d++) vel += dataUp[node + (1 + d) * nnode] * dataUp[node + (1 + d) * nnode];
      vel = sqrt(vel);
      dataIn[node + (num_equation-1) * nnode] -=
          vel * (dataUp[node + (num_equation-1) * nnode] - dataUp[node] * Z) / psData[i]->radius;
    }
  }
}

#ifdef _MASA_
MASA_forcings::MASA_forcings(const int &_dim, const int &_num_equation, const int &_order, const int &_intRuleType,
                             IntegrationRules *_intRules, ParFiniteElementSpace *_vfes, ParGridFunction *_Up,
                             ParGridFunction *_gradUp, const volumeFaceIntegrationArrays &gpuArrays,
                             RunConfiguration &_config)
    : ForcingTerms(_dim, _num_equation, _order, _intRuleType, _intRules, _vfes, _Up, _gradUp, gpuArrays) {
  initMasaHandler("forcing", dim, _config.GetEquationSystem(), _config.GetViscMult());
}

void MASA_forcings::updateTerms(Vector &in) {
  // make sure we are in a 3D case
  assert(dim == 3);

  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();

  double *data = in.GetData();
  // const double *dataUp = Up->GetData();

  // get coords
  const FiniteElementCollection *fec = vfes->FEColl();
  ParMesh *mesh = vfes->GetParMesh();
  ParFiniteElementSpace dfes(mesh, fec, dim, Ordering::byNODES);
  ParGridFunction coordsDof(&dfes);
  mesh->GetNodes(coordsDof);

  for (int el = 0; el < numElem; el++) {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();

    // nodes of the element
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);

    Array<double> ip_forcing(num_equation);
    Vector x(dim);
    for (int n = 0; n < dof_elem; n++) {
      int index = nodes[n];
      for (int d = 0; d < dim; d++) x[d] = coordsDof[index + d * dof];

      ip_forcing[0] = MASA::masa_eval_source_rho<double>(x[0], x[1], x[2], time);  // rho
      ip_forcing[1] = MASA::masa_eval_source_u<double>(x[0], x[1], x[2], time);    // rho*u
      ip_forcing[2] = MASA::masa_eval_source_v<double>(x[0], x[1], x[2], time);    // rho*v
      ip_forcing[3] = MASA::masa_eval_source_w<double>(x[0], x[1], x[2], time);    // rho*w
      ip_forcing[4] = MASA::masa_eval_source_e<double>(x[0], x[1], x[2], time);    // rhp*e

      for (int eq = 0; eq < num_equation; eq++) data[index + eq * dof] += ip_forcing[eq];
    }

    //     const int order = elem->GetOrder();
    //     //cout<<"order :"<<maxorder<<" dof: "<<dof_elem<<endl;
    //     int intorder = 2*order;
    //     //if(intRuleType==1 && trial_fe.GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
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
#endif  //  _MASA_
