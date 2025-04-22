
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
#include "face_integrator.hpp"

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(IntegrationRules *_intRules, RiemannSolver *rsolver_, Fluxes *_fluxClass,
                               ParFiniteElementSpace *_vfes, bool _useLinear, const int _dim, const int _num_equation,
                               ParGridFunction *_gradUp, ParFiniteElementSpace *_gradUpfes, double &_max_char_speed,
                               bool axisym, ParGridFunction *distance, int rank)
    : rsolver(rsolver_),
      fluxClass(_fluxClass),
      vfes(_vfes),
      dim(_dim),
      num_equation(_num_equation),
      max_char_speed(_max_char_speed),
      gradUp(_gradUp),
      gradUpfes(_gradUpfes),
      distance_(distance),
      intRules(_intRules),
      useLinear(_useLinear),
      axisymmetric_(axisym),
      rank_(rank){
  assert(!useLinear);
  totDofs = vfes->GetNDofs();
}

FaceIntegrator::~FaceIntegrator() {
  if (useLinear) {
    delete[] faceMassMatrix1;
    delete[] faceMassMatrix2;
  }
}

void FaceIntegrator::getElementsGrads_cpu(FaceElementTransformations &Tr, const FiniteElement &el1,
                                          const FiniteElement &el2, DenseTensor &gradUp1, DenseTensor &gradUp2) {
  double *dataGradUp = gradUp->GetData();

  vfes->GetElementVDofs(Tr.Elem1->ElementNo, vdofs1);
  int eldDof = el1.GetDof();
  for (int n = 0; n < eldDof; n++) {
    int index = vdofs1[n];
    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) gradUp1(n, eq, d) = dataGradUp[index + eq * totDofs + d * num_equation * totDofs];
    }
  }

  eldDof = el2.GetDof();
  int no2 = Tr.Elem2->ElementNo;
  int NE = vfes->GetNE();
  if (no2 >= NE) {
    int Elem2NbrNo = no2 - NE;
    gradUpfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

    Array<double> arrayGrad2(vdofs2.Size());

    gradUp->FaceNbrData().GetSubVector(vdofs2, arrayGrad2.GetData());
    for (int n = 0; n < eldDof; n++) {
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) gradUp2(n, eq, d) = arrayGrad2[n + eq * eldDof + d * num_equation * eldDof];
      }
    }

  } else {
    vfes->GetElementVDofs(no2, vdofs2);

    for (int n = 0; n < eldDof; n++) {
      int index = vdofs2[n];
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) gradUp2(n, eq, d) = dataGradUp[index + eq * totDofs + d * num_equation * totDofs];
      }
    }
  }
}

void FaceIntegrator::getElementsGrads_gpu(const ParGridFunction *gradUp, ParFiniteElementSpace *vfes,
                                          const ParFiniteElementSpace *gradUpfes, FaceElementTransformations &Tr,
                                          const FiniteElement &el1, const FiniteElement &el2, DenseTensor &gradUp1,
                                          DenseTensor &gradUp2, const int &num_equation, const int &totalDofs,
                                          const int &dim) {
  const double *d_gradUp = gradUp->Read();
  double *d_gradUp1 = gradUp1.ReadWrite();

  Array<int> vdofs1;
  vfes->GetElementVDofs(Tr.Elem1->ElementNo, vdofs1);
  auto d_vdofs1 = vdofs1.Read();
  int eldDof = el1.GetDof();
  MFEM_FORALL(n, eldDof, {
    int index = d_vdofs1[n];
    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++)
        d_gradUp1[n + eq * eldDof + d * eldDof * num_equation] =
            d_gradUp[index + eq * totalDofs + d * num_equation * totalDofs];
    }
  });
  gradUp1.HostRead();

  eldDof = el2.GetDof();
  Array<int> vdofs2;
  int no2 = Tr.Elem2->ElementNo;
  int NE = vfes->GetNE();
  if (no2 >= NE) {
    int Elem2NbrNo = no2 - NE;
    gradUpfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

    Vector grad2;
    grad2.UseDevice(false);
    grad2.SetSize(vdofs2.Size());
    gradUp->FaceNbrData().GetSubVector(vdofs2, grad2);

    for (int n = 0; n < eldDof; n++) {
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) gradUp2(n, eq, d) = grad2[n + eq * eldDof + d * num_equation * eldDof];
      }
    }

  } else {
    vfes->GetElementVDofs(no2, vdofs2);
    double *d_gradUp2 = gradUp2.Write();
    auto d_vdofs2 = vdofs2.Read();

    MFEM_FORALL(n, eldDof, {
      int index = d_vdofs2[n];
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++)
          d_gradUp2[n + eq * eldDof + d * eldDof * num_equation] =
              d_gradUp[index + eq * totalDofs + d * num_equation * totalDofs];
      }
    });
    gradUp2.Read();
  }
}

void FaceIntegrator::getDistanceDofs(FaceElementTransformations &Tr, const FiniteElement &el1, const FiniteElement &el2,
                                     Vector &dist1, Vector &dist2) {
  assert(distance_ != NULL);

  const ParFiniteElementSpace *pfes = distance_->ParFESpace();

  Array<int> dofs1;
  pfes->GetElementVDofs(Tr.Elem1->ElementNo, dofs1);

  dist1.SetSize(dofs1.Size());
  distance_->GetSubVector(dofs1, dist1);

  int no2 = Tr.Elem2->ElementNo;
  int NE = pfes->GetNE();
  Array<int> dofs2;
  if (no2 >= NE) {
    int Elem2NbrNo = no2 - NE;
    pfes->GetFaceNbrElementVDofs(Elem2NbrNo, dofs2);
    dist2.SetSize(dofs2.Size());
    distance_->FaceNbrData().GetSubVector(dofs2, dist2);
  } else {
    pfes->GetElementVDofs(no2, dofs2);
    dist2.SetSize(dofs2.Size());
    distance_->GetSubVector(dofs2, dist2);
  }
}

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                        FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  NonLinearFaceIntegration(el1, el2, Tr, elfun, elvect);
}

void FaceIntegrator::NonLinearFaceIntegration(const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  // Compute the term <F.n(u),[w]> on the interior faces.

  funval1.SetSize(num_equation);
  funval2.SetSize(num_equation);
  nor.SetSize(dim);
  fluxN.SetSize(num_equation);

  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elvect.SetSize((dof1 + dof2) * num_equation);
  elvect = 0.0;

  elfun1_mat.UseExternalData(elfun.GetData(), dof1, num_equation);
  elfun2_mat.UseExternalData(elfun.GetData() + dof1 * num_equation, dof2, num_equation);

  elvect1_mat.UseExternalData(elvect.GetData(), dof1, num_equation);
  elvect2_mat.UseExternalData(elvect.GetData() + dof1 * num_equation, dof2, num_equation);

  // Get gradients of elements
  gradUp1.SetSize(dof1, num_equation, dim);
  gradUp2.SetSize(dof2, num_equation, dim);
#ifdef _GPU_
  getElementsGrads_gpu(gradUp, vfes, gradUpfes, Tr, el1, el2, gradUp1, gradUp2, num_equation, vfes->GetNDofs(), dim);
#else
  getElementsGrads_cpu(Tr, el1, el2, gradUp1, gradUp2);
#endif

  Vector dist1, dist2;
  if (distance_ != NULL) {
    getDistanceDofs(Tr, el1, el2, dist1, dist2);
  }

  // Integration order calculation from DGTraceIntegrator
  int intorder;
  if (Tr.Elem2No >= 0) {
    intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) + 2 * max(el1.GetOrder(), el2.GetOrder()));
  } else {
    intorder = Tr.Elem1->OrderW() + 2 * el1.GetOrder();
  }
  if (el1.Space() == FunctionSpace::Pk) {
    intorder++;
  }
  // IntegrationRules IntRules2(0, Quadrature1D::GaussLobatto);
  const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);

  gradUp1i.SetSize(num_equation, dim);
  gradUp2i.SetSize(num_equation, dim);

  viscF1.SetSize(num_equation, dim);
  viscF2.SetSize(num_equation, dim);

  // element size
  Mesh *mesh = vfes->GetMesh();
  const double delta1 = mesh->GetElementSize(Tr.Elem1No, 1) / el1.GetOrder();

  double delta2 = delta1;
  const int no2 = Tr.Elem2->ElementNo;
  const int NE = vfes->GetNE();
  if (no2 >= NE) {
    // On shared face where element 2 is by different rank.  In this
    // case, mesh->GetElementSize fails.  Instead, compute spacing
    // directly using the ElementTransformation object.  The code
    // below is from the variant of Mesh::GetElementSize that takes an
    // ElementTransformation as input, rather than an element index.
    // We should simply call that function, but it is not public.
    ElementTransformation *T = Tr.Elem2;
    DenseMatrix J(dim, dim);

    Geometry::Type geom = T->GetGeometryType();
    T->SetIntPoint(&Geometries.GetCenter(geom));
    Geometries.JacToPerfJac(geom, T->Jacobian(), J);

    // (dim-1) singular value is h_min, consistent with type=1 in GetElementSize
    delta2 = J.CalcSingularvalue(dim - 1) / el2.GetOrder();
  } else {
    delta2 = mesh->GetElementSize(Tr.Elem2No, 1) / el2.GetOrder();
  }

  // NOTE(malamast): Is there a better way to retrieve these values?
  const int numActiveSpecies = fluxClass->GetNumActiveSpecies();
  const int nvel = fluxClass->GetNumVels();

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    // set face and element int. points    
    Tr.SetAllIntPoints(&ip);  

    // x-y-z coordinates of int pts    
    double x[3];
    Vector transip(x, 3);
    Tr.Transform(ip, transip);  
    
    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, funval1);
    elfun2_mat.MultTranspose(shape2, funval2);
    
    // TODO(malamast): We force negative (unphysical) values of species that occur due to interpolation error to be
    // zero.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      int eq = nvel + 2 + sp;
      funval1[eq] = max(funval1[eq], 0.0);
      funval2[eq] = max(funval2[eq], 0.0);
    }

    // Interpolate the distance function
    double d1 = 0;
    double d2 = 0;
    if (distance_ != NULL) {
      d1 = dist1 * shape1;
      d2 = dist2 * shape2;
    }

    // Interpolate gradients at int. point
    gradUp1i = 0.;
    gradUp2i = 0.;

    for (int eq = 0; eq < num_equation; eq++) {
      for (int d = 0; d < dim; d++) {
        for (int k = 0; k < dof1; k++) gradUp1i(eq, d) += gradUp1(k, eq, d) * shape1(k);
        for (int k = 0; k < dof2; k++) gradUp2i(eq, d) += gradUp2(k, eq, d) * shape2(k);
      }
    }

    // Get the normal vector and the convective flux on the face
    CalcOrtho(Tr.Jacobian(), nor);
    rsolver->Eval(funval1, funval2, nor, fluxN);

    //double x[3];
    //Vector transip(x, 3);
    //Tr.Transform(ip, transip);

    // compute viscous fluxes
    viscF1 = viscF2 = 0.;

    fluxClass->ComputeViscousFluxes(funval1, gradUp1i, transip, delta1, d1, viscF1);
    fluxClass->ComputeViscousFluxes(funval2, gradUp2i, transip, delta2, d2, viscF2);

    // compute mean flux
    viscF1 += viscF2;
    viscF1 *= -0.5;

    // add normal viscous flux to fluxN
    viscF1.AddMult(nor, fluxN);
    fluxN *= ip.weight;

    if (axisymmetric_) {
      fluxN *= transip[0];
    }

    // add to element vectors
    AddMultVWt(shape2, fluxN, elvect2_mat);
    AddMult_a_VWt(-1.0, shape1, fluxN, elvect1_mat);
  }
}
