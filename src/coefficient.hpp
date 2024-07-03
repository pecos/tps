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
#ifndef COEFFICIENT_HPP_
#define COEFFICIENT_HPP_
/** @file
 *
 * @brief Implements Coefficient classes for use in tps
 *
 */

#include "tps_mfem_wrap.hpp"

// Adding to the mfem namespace
namespace mfem {

/// Matrix coefficient defined as the Gradient of a Vector GridFunction
class GradientVectorGridFunctionCoefficient : public MatrixCoefficient {
 protected:
  const GridFunction *GridFunc;

 public:
  /** @brief Construct the coefficient with a scalar grid function @a gf. The
      grid function is not owned by the coefficient. */
  GradientVectorGridFunctionCoefficient(const GridFunction *gf);

  /// Set the scalar grid function.
  void SetGridFunction(const GridFunction *gf);

  /// Get the scalar grid function.
  const GridFunction *GetGridFunction() const { return GridFunc; }

  /// Evaluate the gradient vector coefficient at @a ip.
  virtual void Eval(DenseMatrix &G, ElementTransformation &T, const IntegrationPoint &ip);

  virtual ~GradientVectorGridFunctionCoefficient() {}
};

/** @brief Scalar coefficient defined as the ratio of two scalars where one or
    both scalars are scalar coefficients. */
class TpsRatioCoefficient : public Coefficient {
 private:
  double aConst;
  double bConst;
  Coefficient *a;
  Coefficient *b;

 public:
  /** Initialize a coefficient which returns A / B where @a A is a
      constant and @a B is a scalar coefficient */
  TpsRatioCoefficient(double A, Coefficient &B) : aConst(A), bConst(1.0), a(NULL), b(&B) {}
  /** Initialize a coefficient which returns A / B where @a A and @a B are both
      scalar coefficients */
  TpsRatioCoefficient(Coefficient &A, Coefficient &B) : aConst(0.0), bConst(1.0), a(&A), b(&B) {}
  /** Initialize a coefficient which returns A / B where @a A is a
      scalar coefficient and @a B is a constant */
  TpsRatioCoefficient(Coefficient &A, double B) : aConst(0.0), bConst(B), a(&A), b(NULL) {}

  virtual ~TpsRatioCoefficient() { }

  /// Set the time for internally stored coefficients
  void SetTime(double t);

  /// Evaluate the coefficient
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip) {
    double den = (b == NULL) ? bConst : b->Eval(T, ip);
    MFEM_ASSERT(den != 0.0, "Division by zero in TpsRatioCoefficient");
    return ((a == NULL) ? aConst : a->Eval(T, ip)) / den;
  }

  virtual void Project(QuadratureFunction &qf);
};

class TpsVectorMassIntegrator : public VectorMassIntegrator {
 public:
  TpsVectorMassIntegrator(Coefficient &q, int qo = 0) : VectorMassIntegrator(q, qo) {}

  using BilinearFormIntegrator::AssemblePA;
  virtual void AssemblePA(const FiniteElementSpace &fes);
};

}  // namespace mfem
#endif  // COEFFICIENT_HPP_
