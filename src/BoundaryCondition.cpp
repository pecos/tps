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
#include "BoundaryCondition.hpp"

BoundaryCondition::BoundaryCondition(RiemannSolver *_rsolver, GasMixture *_mixture, Equations _eqSystem,
                                     ParFiniteElementSpace *_vfes, IntegrationRules *_intRules, double &_dt,
                                     const int _dim, const int _num_equation, const int _patchNumber,
                                     const double _refLength, bool axisym)
    : rsolver(_rsolver),
      mixture(_mixture),
      eqSystem(_eqSystem),
      vfes(_vfes),
      intRules(_intRules),
      dt(_dt),
      dim_(_dim),
      nvel_(axisym ? 3 : _dim),
      num_equation_(_num_equation),
      patchNumber(_patchNumber),
      refLength(_refLength),
      axisymmetric_(axisym) {
  BCinit = false;
}

BoundaryCondition::~BoundaryCondition() {}

double BoundaryCondition::aggregateArea(int bndry_patchnum, MPI_Comm bc_comm) {
  double area = 0;

  // loop over boundary faces and accumulate area of matching patch
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == bndry_patchnum) {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
      const IntegrationRule &ir = IntRules.Get(Tr->GetGeometryType(), Tr->OrderJ());

      for (int p = 0; p < ir.GetNPoints(); p++) {
        const IntegrationPoint &ip = ir.IntPoint(p);
        area += Tr->Weight() * ip.weight;
      }
    }
  }

  double areaTotal;
  MPI_Allreduce(&area, &areaTotal, 1, MPI_DOUBLE, MPI_SUM, bc_comm);

  return (areaTotal);
}

int BoundaryCondition::aggregateBndryFaces(int bndry_patchnum, MPI_Comm bc_comm) {
  int nfaces = 0;
  for (int bel = 0; bel < vfes->GetNBE(); bel++) {
    int attr = vfes->GetBdrAttribute(bel);
    if (attr == bndry_patchnum) nfaces++;
  }
  int nfacesTotal;
  MPI_Allreduce(&nfaces, &nfacesTotal, 1, MPI_INT, MPI_SUM, bc_comm);
  return (nfacesTotal);
}

void BoundaryCondition::setElementList(Array<int> &_listElems) {
  listElems.SetSize(_listElems.Size());
  for (int i = 0; i < listElems.Size(); i++) listElems[i] = _listElems[i];
  listElems.ReadWrite();
}

void BoundaryCondition::copyValues(const Vector &orig, Vector &target, const double &mult) {
  const double *dOrig = orig.Read();
  double *dTarget = target.Write();

  MFEM_FORALL(i, target.Size(), { dTarget[i] = dOrig[i] * mult; });
}
