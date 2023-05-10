// -----------------------------------------------------------------------------------bl
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
#ifndef FACEGRADIENTINTEGRATION_HPP_
#define FACEGRADIENTINTEGRATION_HPP_

#include <tps_config.h>
#include "tps_mfem_wrap.hpp"

#include "BoundaryCondition.hpp"
#include "equation_of_state.hpp"
#include "fluxes.hpp"
#include "mpi_groups.hpp"
#include "riemann_solver.hpp"
#include "transport_properties.hpp"
#include "run_configuration.hpp"
#include "unordered_map"


using namespace mfem;

// Interior face term: <F.n(u),[w]>
class GradFaceIntegrator : public NonlinearFormIntegrator {
 private:
  const int dim;
  const int num_equation;
  IntegrationRules *intRules;

  std::unordered_map<int, BoundaryCondition *> inletBCmap;
  std::unordered_map<int, BoundaryCondition *> outletBCmap;  
  std::unordered_map<int, BoundaryCondition *> wallBCmap;
  ParFiniteElementSpace *vfes;
  RunConfiguration &config;
  RiemannSolver *rsolver;
  GasMixture *mixture;
  Fluxes *fluxClass;
  Array<int> &intPointsElIDBC;
  const int &maxIntPoints;
  const int &maxDofs;
  MPI_Groups *groupsMPI;  

  int nbdrInlet, nbdrOutlet, nbdrWall;  
  
 public:
  GradFaceIntegrator(IntegrationRules *_intRules, const int _dim, const int _num_equation,
		     RunConfiguration &_runFile, Array<int> &local_attr, ParFiniteElementSpace *_vfes, RiemannSolver *rsolver_, GasMixture *_mixture, GasMixture *d_mixture, Fluxes *_fluxClass, double &_dt, Array<int> &_intPointsElIDBC, const int &_maxIntPoints, const int &_maxDofs, MPI_Groups *_groupsMPI);

  virtual void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect);

  void initNbrInlet() {
    //cout << " RESET: " << nbdrInlet << endl; fflush(stdout);	        
    nbdrInlet = 0;
    //cout << " after: " << nbdrInlet << endl; fflush(stdout);	            
  }
  void initNbrOutlet() {nbdrOutlet = 0;}
  void initNbrWall() {nbdrWall = 0;}

  int getNbrInlet() {return nbdrInlet;}
  int getNbrOutlet() {return nbdrOutlet;}
  int getNbrWall() {return nbdrWall;}  

  void incNbrInlet() {nbdrInlet++;}
  void incNbrOutlet() {nbdrOutlet++;}
  void incNbrWall() {nbdrWall++;}
  
};

#endif  // FACEGRADIENTINTEGRATION_HPP_
