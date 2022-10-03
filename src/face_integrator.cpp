
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
                               bool axisym)
    : rsolver(rsolver_),
      fluxClass(_fluxClass),
      vfes(_vfes),
      dim(_dim),
      num_equation(_num_equation),
      max_char_speed(_max_char_speed),
      gradUp(_gradUp),
      gradUpfes(_gradUpfes),
      intRules(_intRules),
      useLinear(_useLinear),
      axisymmetric_(axisym) {
  if (useLinear) {
    faceMassMatrix1 = new DenseMatrix[vfes->GetNF() - vfes->GetNBE()];
    faceMassMatrix2 = new DenseMatrix[vfes->GetNF() - vfes->GetNBE()];
    faceMassMatrixComputed = false;
    faceNum = 0;
  }

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

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                        FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  if (useLinear) {
    assert(!axisymmetric_);  // axisym not supported in useLinear path
    MassMatrixFaceIntegral(el1, el2, Tr, elfun, elvect);
  } else {
    NonLinearFaceIntegration(el1, el2, Tr, elfun, elvect);
  }
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
  double delta;
  Mesh *mesh = vfes->GetMesh();  
  delta = mesh->GetElementSize(Tr.Elem1No, 1);
  
  
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip);  // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, funval1);
    elfun2_mat.MultTranspose(shape2, funval2);

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

    double radius = 1;
    double x[3];
    Vector transip(x, 3);
    Tr.Transform(ip, transip);
    if (axisymmetric_) {      
      radius = transip[0];
    }

    // compute viscous fluxes
    viscF1 = viscF2 = 0.;

    fluxClass->ComputeViscousFluxes(funval1, gradUp1i, radius, transip, delta, viscF1);
    fluxClass->ComputeViscousFluxes(funval2, gradUp2i, radius, transip, delta, viscF2);

    // compute mean flux
    viscF1 += viscF2;
    viscF1 *= -0.5;

    // add normal viscous flux to fluxN
    viscF1.AddMult(nor, fluxN);
    fluxN *= ip.weight;

    if (axisymmetric_) {
      fluxN *= radius;
    }

    // add to element vectors
    AddMultVWt(shape2, fluxN, elvect2_mat);
    AddMult_a_VWt(-1.0, shape1, fluxN, elvect1_mat);
  }
}

void FaceIntegrator::MassMatrixFaceIntegral(const FiniteElement &el1, const FiniteElement &el2,
                                            FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  // Compute the term <F.n(u),[w]> on the interior faces.

  Vector shape1;
  Vector shape2;
  Vector nor(dim);

  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elvect.SetSize((dof1 + dof2) * num_equation);
  elvect = 0.0;

  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
  DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2, num_equation);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
  DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2, num_equation);

  // Get gradients of elements
  DenseTensor gradUp1(dof1, num_equation, dim);
  DenseTensor gradUp2(dof2, num_equation, dim);
  getElementsGrads_cpu(Tr, el1, el2, gradUp1, gradUp2);

  //    DenseMatrix faceMass1( dof1 );
  //    DenseMatrix faceMass2( dof2 );
  //    faceMass1 = 0.;
  //    faceMass2 = 0.;
  // generate face mass matrix
  if (!faceMassMatrixComputed) {
    // cout<<faceNum<<" "<<vfes->GetNF()<<" "<<vfes->GetNBE()<<endl;
    faceMassMatrix1[faceNum].SetSize(dof1);
    faceMassMatrix2[faceNum].SetSize(dof2);
    faceMassMatrix1[faceNum] = 0.;
    faceMassMatrix2[faceNum] = 0.;

    // Integration order calculation from DGTraceIntegrator
    int intorder = el1.GetOrder() + el2.GetOrder();
    const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);  // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      for (int j = 0; j < dof1; j++) {
        for (int ii = 0; ii < dof1; ii++) faceMassMatrix1[faceNum](ii, j) += shape1[ii] * shape1[j] * ip.weight;
      }
      for (int j = 0; j < dof2; j++) {
        for (int ii = 0; ii < dof2; ii++) faceMassMatrix2[faceNum](ii, j) += shape2[ii] * shape2[j] * ip.weight;
      }
    }
  }

  // face mass matrix
  DenseMatrix faceMass1(faceMassMatrix1[faceNum]);
  DenseMatrix faceMass2(faceMassMatrix2[faceNum]);

  // Riemann fluxes
  {
    // make sure we have as many integration points as face points
    DG_FECollection fec(el1.GetOrder(), dim - 1, Tr.GetGeometryType());
    int numDofFace = fec.DofForGeometry(Tr.GetGeometryType());
    bool getIntOrder = true;
    int intorder = el1.GetOrder();
    // intorder is calculated in a non-elegant way in order to keep going
    // This should be calculated in a proper way; REVISIT THIS!
    while (getIntOrder) {
      const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
      if (ir->GetNPoints() == numDofFace) {
        getIntOrder = false;
      } else {
        intorder++;
      }
    }

    // element size
    double delta;
    Mesh *mesh = vfes->GetMesh();  
    delta = mesh->GetElementSize(Tr.Elem1No, 1);
    
    
    const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      CalcOrtho(Tr.Jacobian(), nor);

      double radius = 1;
      double x[3];
      Vector transip(x, 3);
      Tr.Transform(ip, transip);
      if (axisymmetric_) {      
        radius = transip[0];
      }  
      
      int index1 = 0, index2 = 0;
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);
      for (int j = 0; j < dof1; j++)
        if (fabs(shape1[j]) > 1e-10) index1 = j;
      for (int j = 0; j < dof2; j++)
        if (fabs(shape2[j]) > 1e-10) index2 = j;

      Vector state1(num_equation), state2(num_equation);
      for (int eq = 0; eq < num_equation; eq++) {
        state1[eq] = elfun1_mat(index1, eq);
        state2[eq] = elfun2_mat(index2, eq);
      }

      Vector fluxN(num_equation);
      rsolver->Eval(state1, state2, nor, fluxN);

      DenseMatrix igradUp1(num_equation, dim), igradUp2(num_equation, dim);
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) {
          igradUp1(eq, d) = gradUp1(index1, eq, d);
          igradUp2(eq, d) = gradUp2(index2, eq, d);
        }
      }

      DenseMatrix viscF1(num_equation, dim), viscF2(num_equation, dim);
      fluxClass->ComputeViscousFluxes(state1, igradUp1, 1, transip, delta, viscF1);
      fluxClass->ComputeViscousFluxes(state2, igradUp2, 1, transip, delta, viscF2);
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) viscF1(eq, d) += viscF2(eq, d);
      }
      viscF1 *= 0.5;
      // add to convective fluxes
      for (int eq = 0; eq < num_equation; eq++) {
        for (int d = 0; d < dim; d++) fluxN[eq] -= viscF1(eq, d) * nor[d];
      }

      for (int j = 0; j < dof1; j++) {
        // cout<<faceMass1(index1,j)<<endl;
        for (int eq = 0; eq < num_equation; eq++) {
          elvect1_mat(j, eq) -= fluxN[eq] * faceMass1(index1, j);
        }
      }
      for (int j = 0; j < dof2; j++) {
        for (int eq = 0; eq < num_equation; eq++) {
          elvect2_mat(j, eq) += fluxN[eq] * faceMass2(index2, j);
        }
      }
    }
  }
  faceNum++;
  if (faceNum == vfes->GetNF() - vfes->GetNBE()) {
    faceNum = 0;
    faceMassMatrixComputed = true;
  }
}
