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
#include "wallBC.hpp"

#include "riemann_solver.hpp"

WallBC::WallBC(RiemannSolver *_rsolver, EquationOfState *_eqState, Fluxes *_fluxClass, ParFiniteElementSpace *_vfes,
               IntegrationRules *_intRules, double &_dt, const int _dim, const int _num_equation, int _patchNumber,
               WallType _bcType, const Array<double> _inputData, const Array<int> &_intPointsElIDBC)
    : BoundaryCondition(_rsolver, _eqState, _vfes, _intRules, _dt, _dim, _num_equation, _patchNumber,
                        1),  // so far walls do not require ref. length. Left at 1
      wallType(_bcType),
      fluxClass(_fluxClass),
      intPointsElIDBC(_intPointsElIDBC) {
  if (wallType == VISC_ISOTH) {
    wallTemp = _inputData[0];
  }
}

WallBC::~WallBC() {}

void WallBC::initBCs() {
  if (!BCinit) {
    buildWallElemsArray(intPointsElIDBC);
  }
  BCinit = true;
}

void WallBC::buildWallElemsArray(const Array<int> &intPointsElIDBC) {
  auto hlistElems = listElems.HostRead();  // this is actually a list of faces
  auto hintPointsElIDBC = intPointsElIDBC.HostRead();

  std::vector<int> unicElems;
  unicElems.clear();

  for (int f = 0; f < listElems.Size(); f++) {
    int bcFace = hlistElems[f];
    int elID = hintPointsElIDBC[2 * bcFace + 1];

    bool elInList = false;
    for (int i = 0; i < unicElems.size(); i++) {
      if (unicElems[i] == elID) elInList = true;
    }
    if (!elInList) unicElems.push_back(elID);
  }

  wallElems.SetSize(7 * unicElems.size());
  wallElems = -1;
  auto hwallElems = wallElems.HostWrite();
  for (int el = 0; el < wallElems.Size() / 7; el++) {
    int elID = unicElems[el];
    for (int f = 0; f < listElems.Size(); f++) {
      int bcFace = hlistElems[f];
      int elID2 = hintPointsElIDBC[2 * bcFace + 1];
      if (elID2 == elID) {
        int nf = hwallElems[0 + 7 * el];
        if (nf == -1) nf = 0;
        nf++;
        hwallElems[nf + 7 * el] = f;
        hwallElems[0 + 7 * el] = nf;
      }
    }
  }
  auto dwallElems = wallElems.ReadWrite();
}

void WallBC::computeBdrFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  switch (wallType) {
    case INV:
      computeINVwallFlux(normal, stateIn, bdrFlux);
      break;
    case VISC_ADIAB:
      computeAdiabaticWallFlux(normal, stateIn, gradState, bdrFlux);
      break;
    case VISC_ISOTH:
      computeIsothermalWallFlux(normal, stateIn, gradState, bdrFlux);
      break;
  }
}

void WallBC::integrationBC(Vector &y, const Vector &x, const Array<int> &nodesIDs, const Array<int> &posDofIds,
                           ParGridFunction *Up, ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                           Array<int> &intPointsElIDBC, const int &maxIntPoints, const int &maxDofs) {
  integrateWalls_gpu(wallType, wallTemp,
                     y,  // output
                     x, nodesIDs, posDofIds, Up, gradUp, shapesBC, normalsWBC, intPointsElIDBC, wallElems, listElems,
                     maxIntPoints, maxDofs, dim, num_equation, eqState->GetSpecificHeatRatio(),
                     eqState->GetGasConstant());
}

void WallBC::computeINVwallFlux(Vector &normal, Vector &stateIn, Vector &bdrFlux) {
  Vector vel(dim);
  for (int d = 0; d < dim; d++) vel[d] = stateIn[1 + d] / stateIn[0];

  double norm = 0.;
  for (int d = 0; d < dim; d++) norm += normal[d] * normal[d];
  norm = sqrt(norm);

  Vector unitN(dim);
  for (int d = 0; d < dim; d++) unitN[d] = normal[d] / norm;

  double vn = 0;
  for (int d = 0; d < dim; d++) vn += vel[d] * unitN[d];

  Vector stateMirror(num_equation);

  stateMirror[0] = stateIn[0];
  stateMirror[1] = stateIn[0] * (vel[0] - 2. * vn * unitN[0]);
  stateMirror[2] = stateIn[0] * (vel[1] - 2. * vn * unitN[1]);
  if (dim == 3) stateMirror[3] = stateIn[0] * (vel[2] - 2. * vn * unitN[2]);
  stateMirror[num_equation - 1] = stateIn[num_equation - 1];

  rsolver->Eval(stateIn, stateMirror, normal, bdrFlux);
}

void WallBC::computeAdiabaticWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  double p = eqState->ComputePressure(stateIn, dim);
  const double gamma = eqState->GetSpecificHeatRatio();

  Vector wallState = stateIn;
  for (int d = 0; d < dim; d++) wallState[1 + d] = 0.;
  wallState[num_equation - 1] = p / (gamma - 1.);

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);

  // incoming visc flux
  DenseMatrix viscF(num_equation, dim);
  fluxClass->ComputeViscousFluxes(stateIn, gradState, viscF);

  // modify gradients so that wall is adibatic
  Vector unitNorm = normal;
  {
    double normN = 0.;
    for (int d = 0; d < dim; d++) normN += normal[d] * normal[d];
    unitNorm *= 1. / sqrt(normN);
  }

  // modify gradient density so dT/dn=0 at the wall
  double DrhoDn = 0.;
  for (int d = 0; d < dim; d++) DrhoDn += gradState(0, d) * unitNorm[d];

  double DrhoDnNew = 0.;
  for (int d = 0; d < dim; d++) DrhoDnNew += gradState(1 + dim, d) * unitNorm[d];
  DrhoDnNew *= stateIn[0] / p;

  for (int d = 0; d < dim; d++) gradState(0, d) += -DrhoDn * unitNorm[d] + DrhoDnNew * unitNorm[d];

  DenseMatrix viscFw(num_equation, dim);
  fluxClass->ComputeViscousFluxes(wallState, gradState, viscFw);

  // Add visc fluxes (we skip density eq.)
  for (int eq = 1; eq < num_equation; eq++) {
    for (int d = 0; d < dim; d++) bdrFlux[eq] -= 0.5 * (viscFw(eq, d) + viscF(eq, d)) * normal[d];
  }
}

void WallBC::computeIsothermalWallFlux(Vector &normal, Vector &stateIn, DenseMatrix &gradState, Vector &bdrFlux) {
  const double gamma = eqState->GetSpecificHeatRatio();

  Vector wallState(num_equation);
  wallState[0] = stateIn[0];
  for (int d = 0; d < dim; d++) wallState[1 + d] = 0.;

  wallState[num_equation - 1] = eqState->GetGasConstant() / (gamma - 1.);  // Cv
  wallState[num_equation - 1] *= stateIn[0] * wallTemp;

  // Normal convective flux
  rsolver->Eval(stateIn, wallState, normal, bdrFlux, true);
}

// void WallBC::integrateWalls_gpu(const WallType type,
//                                 const double &wallTemp,
//                                 Vector& y,
//                                 const Vector &x,
//                                 const Array<int> &nodesIDs,
//                                 const Array<int> &posDofIds,
//                                 ParGridFunction* Up,
//                                 ParGridFunction* gradUp,
//                                 Vector& shapesBC,
//                                 Vector& normalsWBC,
//                                 Array<int>& intPointsElIDBC,
//                                 Array<int> &elemFacesBC,
//                                 Array<int>& listElems,
//                                 const int& maxIntPoints,
//                                 const int& maxDofs,
//                                 const int& dim,
//                                 const int& num_equation,
//                                 const double &gamma,
//                                 const double &Rg )
// {
// #ifdef _GPU_
//   double *d_y = y.Write();
//   const double *d_U = x.Read();
//   const int *d_nodesIDs = nodesIDs.Read();
//   const int *d_posDofIds = posDofIds.Read();
//   //const double *d_Up = Up->Read();
//   //const double *d_gradUp = gradUp->Read();
//   const double *d_shapesBC = shapesBC.Read();
//   const double *d_normW = normalsWBC.Read();
//   const int *d_intPointsElIDBC = intPointsElIDBC.Read();
//   const int *d_listElems = listElems.Read();
//
//   const int totDofs = x.Size()/num_equation;
//   const int numBdrElem = listElems.Size();
//
//   MFEM_FORALL_2D(n,numBdrElem,maxDofs,1,1,
//   {
//     MFEM_FOREACH_THREAD(i,x,maxDofs)
//     {
//       MFEM_SHARED double Ui[216*5], Fcontrib[216*5];
//       MFEM_SHARED double shape[216];
//       MFEM_SHARED double Rflux[5], u1[5],u2[5], nor[3];
//       MFEM_SHARED double weight;
//
//       const int el = d_listElems[n];
//       const int Q    = d_intPointsElIDBC[2*el  ];
//       const int elID = d_intPointsElIDBC[2*el+1];
//
//       const int elOffset = d_posDofIds[2*elID  ];
//       const int elDof    = d_posDofIds[2*elID+1];
//       int indexi;
//       if(i<elDof) indexi = d_nodesIDs[elOffset+i];
//
//       // retreive data
//       for(int eq=0;eq<num_equation;eq++)
//       {
//         if(i<elDof)
//         {
//           Ui[i+eq*elDof] = d_U[indexi+eq*totDofs];
//           Fcontrib[i+eq*elDof] = 0.;
//         }
//       }
//
//       for(int q=0;q<Q;q++) // loop over int. points
//       {
//         if(i<elDof) shape[i] = d_shapesBC[i+q*maxDofs+el*maxIntPoints*maxDofs];
//         if(i<dim) nor[i] = d_normW[i+q*(dim+1)+el*maxIntPoints*(dim+1)];
//         if(dim==2 && i==maxDofs-2) nor[2] = 0.;
//         if(i==maxDofs-1) weight = d_normW[dim+q*(dim+1)+el*maxIntPoints*(dim+1)];
//         MFEM_SYNC_THREAD;
//
//         // interpolate to int. point
//         if(i<num_equation)
//         {
//           u1[i] = 0.;
//           for(int k=0;k<elDof;k++) u1[i] += Ui[k+i*elDof]*shape[k];
//         }
//         MFEM_SYNC_THREAD;
//
//         // compute mirror state
//         switch(type)
//         {
//           case WallType::INV:
//             if(i<num_equation) computeInvWallState(i,&u1[0],&u2[0],&nor[0],dim,num_equation );
//             break;
//           case WallType::VISC_ISOTH:
//             if(i<num_equation) computeIsothermalState(i,&u1[0],&u2[0],&nor[0],wallTemp,
//                                                       gamma,Rg,dim,num_equation );
//             break;
//           case WallType::VISC_ADIAB:
//             break;
//         }
//         MFEM_SYNC_THREAD;
//
//         // compute flux
//         if(i==0) RiemannSolver::riemannLF_gpu(&u1[0],&u2[0],&Rflux[0],&nor[0],
//                                               gamma,Rg,dim,num_equation);
//         MFEM_SYNC_THREAD;
//
//         // sum contributions to integral
//         if(i<elDof)
//         {
//           for(int eq=0;eq<num_equation;eq++) Fcontrib[i+eq*elDof] -= Rflux[eq]*shape[i]*weight;
//         }
//         MFEM_SYNC_THREAD;
//       }
//
//       // add to global data
//       if(i<elDof)
//       {
//         for(int eq=0;eq<num_equation;eq++) d_y[indexi+eq*totDofs] += Fcontrib[i+eq*elDof];
//       }
//     }
//   });
// #endif
// }

void WallBC::integrateWalls_gpu(const WallType type, const double &wallTemp, Vector &y, const Vector &x,
                                const Array<int> &nodesIDs, const Array<int> &posDofIds, ParGridFunction *Up,
                                ParGridFunction *gradUp, Vector &shapesBC, Vector &normalsWBC,
                                Array<int> &intPointsElIDBC, Array<int> &wallElems, Array<int> &listElems,
                                const int &maxIntPoints, const int &maxDofs, const int &dim, const int &num_equation,
                                const double &gamma, const double &Rg) {
#ifdef _GPU_
  double *d_y = y.Write();
  const double *d_U = x.Read();
  const int *d_nodesIDs = nodesIDs.Read();
  const int *d_posDofIds = posDofIds.Read();
  // const double *d_Up = Up->Read();
  // const double *d_gradUp = gradUp->Read();
  const double *d_shapesBC = shapesBC.Read();
  const double *d_normW = normalsWBC.Read();
  const int *d_intPointsElIDBC = intPointsElIDBC.Read();
  const int *d_wallElems = wallElems.Read();
  const int *d_listElems = listElems.Read();

  const int totDofs = x.Size() / num_equation;
  const int numBdrElem = listElems.Size();

  MFEM_FORALL_2D(el,wallElems.Size()/7,maxDofs,1,1,     // NOLINT
  {
    MFEM_FOREACH_THREAD(i,x,maxDofs) {                  // NOLINT
      MFEM_SHARED double Ui[216*5], Fcontrib[216*5];
      MFEM_SHARED double shape[216];
      MFEM_SHARED double Rflux[5], u1[5], u2[5], nor[3];
      MFEM_SHARED double weight;

      const int numFaces = d_wallElems[0+el*7];
      bool elemDataRecovered = false;
      int elOffset;
      int elDof;
      int elID;
      int indexi;

      for (int f=0; f < numFaces; f++) {
        const int n = d_wallElems[1 + f + el * 7];

        const int el = d_listElems[n];
        const int Q = d_intPointsElIDBC[2 * el];

        if (!elemDataRecovered) {
          elID = d_intPointsElIDBC[2 * el + 1];

          elOffset = d_posDofIds[2 * elID];
          elDof = d_posDofIds[2 * elID + 1];

          if (i < elDof) indexi = d_nodesIDs[elOffset + i];
        }

        // retreive data
        if (i < elDof && !elemDataRecovered) {
          for (int eq = 0; eq < num_equation; eq++) {
            Ui[i + eq * elDof] = d_U[indexi + eq * totDofs];
            Fcontrib[i + eq * elDof] = 0.;
          }
          elemDataRecovered = true;
        }

        for (int q = 0; q < Q; q++) {  // loop over int. points
          if (i < elDof) shape[i] = d_shapesBC[i + q * maxDofs + el * maxIntPoints * maxDofs];
          if (i < dim) nor[i] = d_normW[i + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
          if (dim == 2 && i == maxDofs - 2) nor[2] = 0.;
          if (i == maxDofs - 1) weight = d_normW[dim + q * (dim + 1) + el * maxIntPoints * (dim + 1)];
          MFEM_SYNC_THREAD;

          // interpolate to int. point
          if (i < num_equation) {
            u1[i] = 0.;
            for (int k = 0; k < elDof; k++) u1[i] += Ui[k + i * elDof] * shape[k];
          }
          MFEM_SYNC_THREAD;

          // compute mirror state
          switch (type) {
            case WallType::INV:
              if (i < num_equation) computeInvWallState(i, &u1[0], &u2[0], &nor[0], dim, num_equation);
              break;
            case WallType::VISC_ISOTH:
              if (i < num_equation)
                computeIsothermalState(i, &u1[0], &u2[0], &nor[0], wallTemp, gamma, Rg, dim, num_equation);
              break;
            case WallType::VISC_ADIAB:
              break;
          }
          MFEM_SYNC_THREAD;

          // compute flux
          if (i == 0) RiemannSolver::riemannLF_gpu(&u1[0], &u2[0], &Rflux[0], &nor[0], gamma, Rg, dim, num_equation);
          MFEM_SYNC_THREAD;

          // sum contributions to integral
          if (i < elDof) {
            for (int eq = 0; eq < num_equation; eq++) Fcontrib[i + eq * elDof] -= Rflux[eq] * shape[i] * weight;
          }
          MFEM_SYNC_THREAD;
        }
      }

      // add to global data
      if (i < elDof) {
        for (int eq = 0; eq < num_equation; eq++) d_y[indexi + eq * totDofs] += Fcontrib[i + eq * elDof];
      }
    }
  });
#endif
}
