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
#ifndef RHS_OPERATOR_HPP_
#define RHS_OPERATOR_HPP_

#include <grvy.h>
#include <tps_config.h>

#include <mfem/general/forall.hpp>

#include "BCintegrator.hpp"
#include "chemistry.hpp"
#include "dataStructures.hpp"
#include "dgNonlinearForm.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "forcing_terms.hpp"
#include "gradNonLinearForm.hpp"
#include "gradients.hpp"
#include "radiation.hpp"
#include "run_configuration.hpp"
#include "source_term.hpp"
#include "tps_mfem_wrap.hpp"
#include "transport_properties.hpp"

using namespace mfem;

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class RHSoperator : public TimeDependentOperator {
 private:
  const RunConfiguration &config_;
  Gradients *gradients;

  int &iter;

  const int dim_;
  const int nvel;

  const Equations &eqSystem;

  double &max_char_speed;
  const int &num_equation_;

  IntegrationRules *intRules;
  const int intRuleType;

  Fluxes *fluxClass;
  GasMixture *mixture, *d_mixture_;
  TransportProperties *transport_;

  ParFiniteElementSpace *vfes;
  ParFiniteElementSpace *fes;

  const precomputedIntegrationData &gpu_precomputed_data_;

  const int *h_num_elems_of_type;

  const int &maxIntPoints;
  const int &maxDofs;

  DGNonLinearForm *A;

  MixedBilinearForm *Aflux;

  ParMesh *mesh;

  ParFiniteElementSpace *dfes;
  ParGridFunction *coordsDof;
  ParGridFunction *elSize;

  ParGridFunction *spaceVaryViscMult;
  linearlyVaryingVisc &linViscData;

  Array<DenseMatrix *> Me_inv;
  Array<DenseMatrix *> Me_inv_rad;

  Vector invMArray;
  Vector invMArray_rad;
  Array<int> posDofInvM;

  // reference to conserved varibales
  ParGridFunction *U_;

  // reference to primitive varibales
  ParGridFunction *Up;
  ParGridFunction *plasma_conductivity_;
  ParGridFunction *joule_heating_;

  // gradients of primitives and associated forms&FE space
  ParGridFunction *gradUp;
  ParFiniteElementSpace *gradUpfes;
  // ParNonlinearForm *gradUp_A;
  GradNonLinearForm *gradUp_A;

  BCintegrator *bcIntegrator;
  ParGridFunction *distance_;

  Array<ForcingTerms *> forcing;
  int masaForcingIndex_ = -1;

  mutable DenseTensor flux;
  mutable Vector z;
  mutable Vector fk, zk;  // temp vectors for flux volume integral

  mutable dataTransferArrays transferU;
  mutable dataTransferArrays transferUp;
  mutable dataTransferArrays transferGradUp;
  void allocateTransferData();

  // void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;
  void GetFlux(const Vector &state, DenseTensor &flux) const;

  mutable Vector local_timeDerivatives;
  void computeMeanTimeDerivatives(Vector &y) const;

 public:
  RHSoperator(int &_iter, const int _dim, const int &_num_equation, const int &_order, const Equations &_eqSystem,
              double &_max_char_speed, IntegrationRules *_intRules, int _intRuleType, Fluxes *_fluxClass,
              GasMixture *_mixture, GasMixture *d_mixture, Chemistry *_chemistry, TransportProperties *_transport,
              Radiation *_radiation, ParFiniteElementSpace *_vfes, ParFiniteElementSpace *_fes,
              const precomputedIntegrationData &gpu_precomputed_data, const int &_maxIntPoints, const int &_maxDofs,
              DGNonLinearForm *_A, MixedBilinearForm *_Aflux, ParMesh *_mesh, ParGridFunction *_spaceVaryViscMult,
              ParGridFunction *U, ParGridFunction *_Up, ParGridFunction *_gradUp, ParFiniteElementSpace *_gradUpfes,
              GradNonLinearForm *_gradUp_A, BCintegrator *_bcIntegrator, RunConfiguration &_config, ParGridFunction *pc,
              ParGridFunction *jh, ParGridFunction *distance);

  virtual void Mult(const Vector &x, Vector &y) const;
  void updatePrimitives(const Vector &x) const;
  void updateGradients(const Vector &x, const bool &primitiveUpdated) const;

  virtual ~RHSoperator();

  Gradients *getGradients() { return gradients; }
  DenseTensor *getFlux() { return &flux; }

  const double *getLocalTimeDerivatives() { return local_timeDerivatives.HostRead(); }
  ForcingTerms *getForcingTerm(const int index) { return forcing[index]; }
  int getMasaForcingIndex() { return masaForcingIndex_; }

  static void initNBlockDataTransfer(const Vector &x, ParFiniteElementSpace *pfes, dataTransferArrays &dataTransfer);
  static void waitAllDataTransfer(ParFiniteElementSpace *pfes, dataTransferArrays &dataTransfer);

  // GPU functions
  void GetFlux_gpu(const Vector &state, DenseTensor &flux) const;
  static void copyZk2Z_gpu(Vector &z, Vector &zk, const int eq, const int dof);
  static void copyDataForFluxIntegration_gpu(const Vector &z, DenseTensor &flux, Vector &fk, Vector &zk, const int eq,
                                             const int dof, const int dim);
  static void multiPlyInvers_gpu(Vector &y, Vector &z, const precomputedIntegrationData &gpu_precomputed_data,
                                 const Vector &invMArray, const Array<int> &posDofInvM, const int num_equation,
                                 const int totNumDof, const int NE, const int elemOffset, const int dof);

  static void meanTimeDerivatives_gpu(Vector &y, Vector &local_timeDerivatives, Vector &tmp_vec, const int &NDof,
                                      const int &num_equation, const int &dim);
};

#endif  // RHS_OPERATOR_HPP_
