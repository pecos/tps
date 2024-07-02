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

/** @file
 * @copydoc coefficient.hpp
 */

#include "coefficient.hpp"

namespace mfem {
GradientVectorGridFunctionCoefficient::GradientVectorGridFunctionCoefficient(const GridFunction *gf)
    : MatrixCoefficient((gf) ? gf->VectorDim() : 0) {
  GridFunc = gf;
}

void GradientVectorGridFunctionCoefficient::SetGridFunction(const GridFunction *gf) {
  GridFunc = gf;
  const int dim = (gf) ? gf->VectorDim() : 0;
  height = width = dim;
}

void GradientVectorGridFunctionCoefficient::Eval(DenseMatrix &G, ElementTransformation &T, const IntegrationPoint &ip) {
  Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
  if (T.mesh->GetNE() == gf_mesh->GetNE()) {
    GridFunc->GetVectorGradient(T, G);
  } else {
    IntegrationPoint coarse_ip;

    // NB: In mfem/fem/coefficients.cpp, the function RefinedToCoarse is defined.  Here we reproduce it explicitly.
    //
    // ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
    Mesh &fine_mesh = *T.mesh;
    // Get the element transformation of the coarse element containing the
    // fine element.
    int fine_element = T.ElementNo;
    const CoarseFineTransformations &cf = fine_mesh.GetRefinementTransforms();
    int coarse_element = cf.embeddings[fine_element].parent;
    ElementTransformation *coarse_T = gf_mesh->GetElementTransformation(coarse_element);
    // Transform the integration point from fine element coordinates to coarse
    // element coordinates.
    Geometry::Type geom = T.GetGeometryType();
    IntegrationPointTransformation fine_to_coarse;
    IsoparametricTransformation &emb_tr = fine_to_coarse.Transf;
    emb_tr.SetIdentityTransformation(geom);
    emb_tr.SetPointMat(cf.point_matrices[geom](cf.embeddings[fine_element].matrix));
    fine_to_coarse.Transform(ip, coarse_ip);
    coarse_T->SetIntPoint(&coarse_ip);

    GridFunc->GetVectorGradient(*coarse_T, G);
  }
}

void TpsRatioCoefficient::SetTime(double t) {
  if (a) {
    a->SetTime(t);
  }
  if (b) {
    b->SetTime(t);
  }
  this->Coefficient::SetTime(t);
}

}  // namespace mfem
