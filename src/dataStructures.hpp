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
#ifndef DATASTRUCTURES_HPP_
#define DATASTRUCTURES_HPP_

#include <string>

#include "tps_mfem_wrap.hpp"

using namespace mfem;

namespace gpudata {
// nodes IDs and indirection array
const int MAXINTPOINTS = 64;  // corresponding to HEX face with p=5
const int MAXDOFS = 216;      // corresponding to HEX with p=5

const int MAXDIM = 3;
const int MAXSPECIES = 5;
const int MAXEQUATIONS = MAXDIM + 2 + MAXSPECIES;  // momentum + two energies + species
// NOTE: (presumably from marc) lets make sure we don't have more than 20 eq.
// NOTE(kevin): with MAXEQUATIONS=20, marvin fails with out-of-memery with 3 MPI process.

const int MAXREACTIONS = 20;
const int MAXCHEMPARAMS = 3;

const int MAXTABLE = 512;
const int MAXTABLEDIM = 2;

const int MAXVISUAL = 128;
}  // namespace gpudata

enum Equations {
  EULER,      // Euler equations
  NS,         // Navier-Stokes equations
  NS_PASSIVE  // NS with passive scalar equation
};

// These are presets of combination of EquationOfState, TranportProperties, and Chemistry.
enum WorkingFluid { DRY_AIR, USER_DEFINED, LTE_FLUID };

// These are the type of EquationOfState.
enum GasModel { /* PERFECT_SINGLE, */ PERFECT_MIXTURE, /* CANTERA, */ NUM_GASMODEL };

enum TransportModel { ARGON_MINIMAL, ARGON_MIXTURE, CONSTANT, NUM_TRANSPORTMODEL };

enum ChemistryModel { /* CANTERA, */ NUM_CHEMISTRYMODEL };

enum ReactionModel { ARRHENIUS, HOFFERTLIEN, TABULATED_RXN, NUM_REACTIONMODEL };

enum RadiationModel { NONE_RAD, NET_EMISSION, NUM_RADIATIONMODEL };

enum NetEmissionCoefficientModel { TABULATED_NEC, NUM_NECMODEL };

enum GasParams { SPECIES_MW, SPECIES_CHARGES, FORMATION_ENERGY, /* SPECIES_HEAT_RATIO, */ NUM_GASPARAMS };

enum FluxTrns { VISCOSITY, BULK_VISCOSITY, HEAVY_THERMAL_CONDUCTIVITY, ELECTRON_THERMAL_CONDUCTIVITY, NUM_FLUX_TRANS };

enum SrcTrns { ELECTRIC_CONDUCTIVITY, NUM_SRC_TRANS };

enum SpeciesTrns { /* DIFFUSIVITY, MOBILITY, */ MF_FREQUENCY, NUM_SPECIES_COEFFS };

enum TransportOutputPrimitives {
  TOTAL_PRESSURE,
  HEAVY_TEMPERATURE,
  ELECTRON_PRESSURE,
  ELECTRON_TEMPERATURE,
  TOTAL_NUMBER_DENSITY,
  NUM_OUTPUT_PRIMITIVES
};

enum ArgonColl {
  CLMB_ATT,      // Attractive Coulomb potential
  CLMB_REP,      // Repulsive Coulomb potential
  AR_AR1P,       // Ar - Ar.1+ (include excited states)
  AR_E,          // Ar - E (include excited states)
  AR_AR,         // Ar - Ar (include excited states)
  /*DMR_ION, */  // Ar2 - Ar.1+ (include excited states)
  NONE_ARGCOLL
};

enum ArgonSpcs {
  AR,        // Monomer Argon neutral, including excited states
  AR1P,      // Ar.1+, including excited states
  ELECTRON,  // Electron
  NONE_ARGSPCS
};

// Type of primitive variable which requires gradient evaulation for diffusion velocity.
// Also used for species output primitive variables.
enum SpeciesPrimitiveType { MASS_FRACTION, MOLE_FRACTION, NUMBER_DENSITY, NUM_SPECIES_PRIMITIVES };

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
  VISC_ISOTH,  // Viscous isothermal wall
  VISC_GNRL    // Viscous general wall
};

enum ThermalCondition {
  ADIAB,  // Adiabatic
  ISOTH,  // Isothermal
  SHTH,   // Sheath (only for electron temperature)
  NONE_THMCND
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
                    // When the Annular zone is selected, velocity components are
                    // given as V = {radial, azimuthal, axial} (axial in the
                    // direction of the normal)

  double multFactor;  // Factor multiplying the product of sigma*(U-U*) where
                      // U* is the target solution. Currently sigam=a/l where
                      // a is the mixed-out speed of sound and l is the lenght
                      // of the spnge zone. [OPTIONAL]

  bool singleTemperature;
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

struct constantTransportData {
  double viscosity;
  double bulkViscosity;
  double diffusivity[gpudata::MAXSPECIES];
  double thermalConductivity;
  double electronThermalConductivity;
  double mtFreq[gpudata::MAXSPECIES];  // momentum transfer frequency

  int electronIndex;
};

struct collisionInputs {
  double debyeCircle;
  double Th;
  double Te;
  double ndimTh;
  double ndimTe;
};

struct WallData {
  ThermalCondition hvyThermalCond;
  ThermalCondition elecThermalCond;

  double Th;
  double Te;
};

struct BoundaryViscousFluxData {
  double normal[gpudata::MAXDIM];
  /* Primitive viscous flux index order:
   diffusion velocity - numSpecies,
   stress             - nvel,
   heavy_heat         - 1,
   electron_heat      - 1
  */
  double primFlux[gpudata::MAXEQUATIONS];
  bool primFluxIdxs[gpudata::MAXEQUATIONS];

  // NOTE(kevin): while auto-generated operator= works in the same way,
  //              the function is still defined to be safe.
  MFEM_HOST_DEVICE BoundaryViscousFluxData &operator=(const BoundaryViscousFluxData &rhs) {
    for (int eq = 0; eq < gpudata::MAXEQUATIONS; eq++) {
      primFlux[eq] = rhs.primFlux[eq];
      primFluxIdxs[eq] = rhs.primFluxIdxs[eq];
    }
    for (int d = 0; d < gpudata::MAXDIM; d++) normal[d] = rhs.normal[d];
    return *this;
  }
};

struct BoundaryPrimitiveData {
  double prim[gpudata::MAXEQUATIONS];
  bool primIdxs[gpudata::MAXEQUATIONS];

  MFEM_HOST_DEVICE BoundaryPrimitiveData &operator=(const BoundaryPrimitiveData &rhs) {
    for (int eq = 0; eq < gpudata::MAXEQUATIONS; eq++) {
      prim[eq] = rhs.prim[eq];
      primIdxs[eq] = rhs.primIdxs[eq];
    }
    return *this;
  }
};

// pack up inputs for all GasMixture-derived classes.
struct DryAirInput {
  WorkingFluid f;

  // DryAir inputs.
  Equations eq_sys;
#if defined(_HIP_)
  double visc_mult;
  double bulk_visc_mult;
#else
#endif
};

struct PerfectMixtureInput {
  WorkingFluid f;

  int numSpecies;

  bool isElectronIncluded;
  bool ambipolar;
  bool twoTemperature;

  double gasParams[gpudata::MAXSPECIES * GasParams::NUM_GASPARAMS];
  double molarCV[gpudata::MAXSPECIES];
};

struct LteMixtureInput {
  WorkingFluid f;
  std::string thermo_file_name;
};

struct ArgonTransportInput {
  int neutralIndex;
  int ionIndex;
  int electronIndex;

  bool thirdOrderkElectron;

  // ArgonMixtureTransport
  ArgonColl collisionIndex[gpudata::MAXSPECIES * gpudata::MAXSPECIES];

  // artificial multipliers
  bool multiply;
  double fluxTrnsMultiplier[FluxTrns::NUM_FLUX_TRANS];
  double spcsTrnsMultiplier[SpeciesTrns::NUM_SPECIES_COEFFS];
  double diffMult;
  double mobilMult;
};

struct TableInput {
  int Ndata;
  const double *xdata;
  const double *fdata;
  bool xLogScale;
  bool fLogScale;

  int order;  // interpolation order. Currently only support order=1.
};

// ReactionInput must be able to support various types of evaluation.
// At the same time, allocating maximum size of memory can be prohibitive in gpu device.
// the datatype therefore contains pointers, which can be allocated only when they are used.
struct ReactionInput {
  TableInput tableInput;
  // NOTE(kevin): with gpu, this pointer is only valid on the device.
  const double *modelParams;
};

struct ChemistryInput {
  ChemistryModel model;

  int electronIndex;
  int numReactions;
  double reactionEnergies[gpudata::MAXREACTIONS];
  bool detailedBalance[gpudata::MAXREACTIONS];
  double reactantStoich[gpudata::MAXSPECIES * gpudata::MAXREACTIONS];
  double productStoich[gpudata::MAXSPECIES * gpudata::MAXREACTIONS];
  ReactionModel reactionModels[gpudata::MAXREACTIONS];
  double equilibriumConstantParams[gpudata::MAXCHEMPARAMS * gpudata::MAXREACTIONS];
  ReactionInput reactionInputs[gpudata::MAXREACTIONS];
};

struct PostProcessInput {
  std::string prefix;
  int startIter;
  int endIter;
  int freq;
};

struct AuxiliaryVisualizationIndexes {
  int Xsp, Ysp, nsp;                            // species primitive.
  int FluxTrns, SrcTrns, SpeciesTrns, diffVel;  // transport.
  int rxn;                                      // reactions.
};

struct RadiationInput {
  RadiationModel model;

  NetEmissionCoefficientModel necModel;
  TableInput necTableInput;
};

#endif  // DATASTRUCTURES_HPP_
