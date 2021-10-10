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
#ifndef GRADNONLINEARFORM_HPP_
#define GRADNONLINEARFORM_HPP_

#include <tps_config.h>

#include <general/forall.hpp>
#include <mfem.hpp>

using namespace mfem;
using namespace std;

class GradNonLinearForm : public ParNonlinearForm {
 private:
  ParFiniteElementSpace *vfes;
  IntegrationRules *intRules;
  const int dim;
  const int num_equation;

  Array<int> &numElems;
  Array<int> &nodesIDs;
  Array<int> &posDofIds;

  //   int *h_numElems;
  //   int *h_posDofIds;

  Vector &shapeWnor1;
  Vector &shape2;
  const int &maxIntPoints;
  const int &maxDofs;

  Array<int> &elemFaces;  // number and faces IDs of each element
  Array<int> &elems12Q;   // elements connecting a face

 public:
  GradNonLinearForm(ParFiniteElementSpace *f, IntegrationRules *intRules, const int dim, const int num_equation,
                    Array<int> &_numElems, Array<int> &_nodesIDs, Array<int> &_posDofIds, Vector &_shapeWnor1,
                    Vector &_shape2, const int &maxIntPoints, const int &maxDofs, Array<int> &_elemFaces,
                    Array<int> &_elems12Q);

  void Mult(const ParGridFunction *Up, Vector &y);

#ifdef _GPU_
  void Mult_gpu(const Vector &x, Vector &y);
  static void IndexedAddToGlobalMemory(const Array<int> &vdofs, Vector &y, const Vector &el_y, const int &dof,
                                       const int &elDof, const int &num_equation, const int &dim);
#endif
};

#endif  // GRADNONLINEARFORM_HPP_
