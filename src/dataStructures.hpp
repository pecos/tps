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
#ifndef DATASTRUCTURES_HPP_
#define DATASTRUCTURES_HPP_

#include <mfem.hpp>
#include <string>

using namespace mfem;

enum Equations {
  EULER,      // Euler equations
  NS,         // Navier-Stokes equations
  NS_PASSIVE  // NS with passive scalar equation
};

enum WorkingFluid { DRY_AIR, USER_DEFINED };

enum InletType {
  SUB_DENS_VEL,      // Subsonic inlet specified by the density and velocity components
  SUB_DENS_VEL_NR,   // Non-reflecting subsonic inlet specified by the density and velocity components
  SUB_VEL_CONST_ENT  // Subsonic non-reflecting. Specified vel, keeps entropy constant
};

enum OutletType {
  SUB_P,        // subsonic outlet specified with pressure
  SUB_P_NR,     // non-reflecting subsonic outlet specified with pressure
  SUB_MF_NR,    // Mass-flow non-reflecting
  SUB_MF_NR_PW  // point-based non-reflecting massflow BC
};

enum WallType {
  INV,         // Inviscid wall
  VISC_ADIAB,  // Viscous adiabatic wall
  VISC_ISOTH   // Viscous isothermal wall
};

// The following four keywords define two planes in which
// a linearly varying viscosity can be defined between these two.
// The planes are defined by the normal and one point being the
// normal equal for both planes. The visc ratio will vary linearly
// from 0 to viscRatio from the plane defined by pointInit and
// the plane defined by point0.
//  |-----------------------------------|
//  |------------------------------<====|
//  |-----------------------------------|
//  p_init                     normal  p0
struct linearlyVaryingVisc {
  Vector normal;
  Vector point0;
  Vector pointInit;
  double viscRatio;
  bool isEnabled;
};

enum SpongeZoneSolution {
  USERDEF,   // User defined target solution
  MIXEDOUT,  // Mixed-out values will be computed at the pInit plane and
             // used as target solution in the sponge zone
  NONE = -1  // Invalid value
};

enum SpongeZoneType {
  PLANAR,  // The sponge solution is imposed between two planes
  ANNULUS  // The sponge solution is applied to an annular region
           // defined by an axis and inner and outer radius
};

struct SpongeZoneData {
  Vector normal;  // These planes are defined in the same manner as
  Vector point0;  // in the linearlyVaryingVisc struct case.
  Vector pointInit;

  double r1;  // inner radius in the case of annulus
  double r2;  // outer radius in the case of annulus

  SpongeZoneSolution szSolType;
  SpongeZoneType szType;

  double tol;  // Tolerance for finding nodes at pInit plane
               // These points are used to compute mixed-out values.
               // Required if szType==MIXEDOUT

  Vector targetUp;  // Target user-defined solution defined with primitive
                    // variables: rho, V, p. Required if szType==USERDEF

  double multFactor;  // Factor multiplying the product of sigma*(U-U*) where
                      // U* is the target solution. Currently sigam=a/l where
                      // a is the mixed-out speed of sound and l is the lenght
                      // of the spnge zone. [OPTIONAL]
};

struct volumeFaceIntegrationArrays {
  // nodes IDs and indirection array
  Array<int> nodesIDs;
  Array<int> posDofIds;
  // count of number of elements of each type
  Array<int> numElems;
  // Array<int> posDofQshape1; // position, num. dof and integration points for each face
  Vector shapeWnor1;  // shape functions, weight and normal for each face at ach integration point
  Vector shape2;

  Array<int> elemFaces;  // number and faces IDs of each element
  Array<int> elems12Q;   // elements connecting a face and integration points

  // used in gradient computation:
  // gradients of shape functions for all nodes and weight multiplied by det(Jac)
  // at each integration point
  Vector elemShapeDshapeWJ;           // [...l_0(i),...,l_dof(i),l_0_x(i),...,l_dof_d(i), w_i*detJac_i ...]
  Array<int> elemPosQ_shapeDshapeWJ;  // position and num. of integration points for each element
};

struct parallelFacesIntegrationArrays {
  Vector sharedShapeWnor1;
  Vector sharedShape2;
  Array<int> sharedElem1Dof12Q;
  Array<int> sharedVdofs;
  Array<int> sharedVdofsGradUp;
  Array<int> sharedElemsFaces;
};

struct dataTransferArrays {
  // vectors for data transfer
  Vector face_nbr_data;
  Vector send_data;

  // MPI communicators data
  int num_face_nbrs;
  MPI_Request *requests;
  MPI_Status *statuses;
};

// structure to encapsulate passive scalar data
struct passiveScalarData {
  Vector coords;
  double radius;  // distance in which nodes will be looked for
  double value;   // value of the passive scalar at the location
  Array<int> nodes;
};

// heat source structure
struct heatSourceData {
  bool isEnabled;
  double value;
  std::string type;
  Vector point1;
  Vector point2;
  double radius;
};

#endif  // DATASTRUCTURES_HPP_
