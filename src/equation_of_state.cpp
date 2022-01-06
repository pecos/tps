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

#include "equation_of_state.hpp"

// EquationOfState::EquationOfState() {}

GasMixture::GasMixture(WorkingFluid _fluid, int _dim) {
  fluid = _fluid;
  dim = _dim;

};

//////////////////////////////////////////////////////
//////// Dry Air mixture
//////////////////////////////////////////////////////

DryAir::DryAir(RunConfiguration &_runfile, int _dim) : GasMixture(WorkingFluid::DRY_AIR,_dim)
{
  numSpecies = (_runfile.GetEquationSystem()==NS_PASSIVE) ? 2 : 1;
  ambipolar = false;
  twoTemperature = false;

  SetNumActiveSpecies();
  SetNumEquations();

  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;

  gasParams.SetSize(numSpecies,GasParams::NUM_GASPARAMS);
  gasParams = 0.0;
  gasParams(0,GasParams::SPECIES_MW) = UNIVERSALGASCONSTANT / gas_constant;
  gasParams(0,GasParams::SPECIES_HEAT_RATIO) = specific_heat_ratio;


  // TODO: replace Nconservative/Nprimitive.
  // add extra equation for passive scalar
  if (_runfile.GetEquationSystem() == Equations::NS_PASSIVE) {
    Nconservative++;
    Nprimitive++;
    num_equation++;
  }
}

DryAir::DryAir() {
  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;
}

DryAir::DryAir(int _dim, int _num_equation) {
  gas_constant = 287.058;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;

  dim = _dim;
  num_equation = _num_equation;
}

void DryAir::setNumEquations() {
  Nconservative = dim + 2;
  Nprimitive = Nconservative;
  num_equation = Nconservative;
}

// void EquationOfState::SetFluid(WorkingFluid _fluid) {
//   fluid = _fluid;
//   switch (fluid) {
//     case DRY_AIR:
//       gas_constant = 287.058;
//       // gas_constant = 1.; // for comparison against ex18
//       specific_heat_ratio = 1.4;
//       visc_mult = 1.;
//       Pr = 0.71;
//       cp_div_pr = specific_heat_ratio * gas_constant / (Pr * (specific_heat_ratio - 1.));
//       Sc = 0.71;
//       break;
//     default:
//       break;
//   }
// }

bool DryAir::StateIsPhysical(const mfem::Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  if (den < 0) {
    cout << "Negative density: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  if (den_energy <= 0) {
    cout << "Negative energy: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }

  double den_vel2 = 0;
  for (int i = 0; i < dim; i++) {
    den_vel2 += den_vel(i) * den_vel(i);
  }
  den_vel2 /= den;

  const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

  if (pres <= 0) {
    cout << "Negative pressure: " << pres << ", state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  return true;
}

// TODO: We need to move this routine to upper level, i.e. M2ulPhys.
// Diffusion velocity contributes to the characteristic speed, which mixture cannot handle or know.
// Compute the maximum characteristic speed.
double DryAir::ComputeMaxCharSpeed(const Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state);
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

void DryAir::GetConservativesFromPrimitives(const Vector& primit, Vector& conserv) {
  conserv = primit;

  double v2 = 0.;
  for (int d = 0; d < dim; d++) {
    v2 += primit[1 + d] * primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  // total energy
  conserv[1 + dim] = gas_constant * primit[0] * primit[1 + dim] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;

  // case of passive scalar
  if (num_equation > dim + 2) {
    for (int n = 0; n < num_equation - dim - 2; n++) {
      conserv[dim + 2 + n] *= primit[0];
    }
  }
}

void DryAir::GetPrimitivesFromConservatives(const Vector& conserv, Vector& primit) {
  double T = ComputeTemperature(conserv);
  primit = conserv;

  for (int d = 0; d < dim; d++) primit[1 + d] /= conserv[0];

  primit[dim + 1] = T;

  // case of passive scalar
  if (num_equation > dim + 2) {
    for (int n = 0; n < num_equation - dim - 2; n++) {
      primit[dim + 2 + n] /= primit[0];
    }
  }
}

double DryAir::ComputeSpeedOfSound(const mfem::Vector& Uin, bool primitive) {
  double T;

  if (primitive) {
    T = Uin[1 + dim];
  } else {
    // conservatives passed in
    T = ComputeTemperature(Uin);
  }

  return sqrt(specific_heat_ratio * gas_constant * T);
}

double DryAir::ComputePressureDerivative(const Vector& dUp_dx, const Vector& Uin, bool primitive) {
  double T, p;
  if (primitive) {
    T = Uin[1 + dim];
  } else {
    T = ComputeTemperature(Uin);
  }

  return gas_constant * (T * dUp_dx[0] + Uin[0] * dUp_dx[1 + dim]);
}

double DryAir::ComputePressureFromPrimitives(const mfem::Vector& Up) { return gas_constant * Up[0] * Up[1 + dim]; }

void DryAir::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up) {
  double* pGridFunc = press->HostWrite();
  const double* UpData = Up->HostRead();

  const int nnode = press->FESpace()->GetNDofs();

  for (int n = 0; n < nnode; n++) {
    double tmp = UpData[n + (1 + dim) * nnode];
    double rho = UpData[n];
    pGridFunc[n] = rho * gas_constant * tmp;
  }
}

// GPU FUNCTIONS
/*#ifdef _GPU_
double EquationOfState::pressure( double *state,
                                  double *KE,
                                  const double gamma,
                                  const int dim,
                                  const int num_equation)
{
  double p = 0.;
  for(int k=0;k<dim;k++) p += KE[k];
  return (gamma-1.)*(state[num_equation-1] - p);
}
#endif*/

//////////////////////////////////////////////////////
//////// Test Binary Air mixture
//////////////////////////////////////////////////////
TestBinaryAir::TestBinaryAir(RunConfiguration &_runfile, int _dim) : GasMixture(WorkingFluid::TEST_BINARY_AIR,_dim)
{
  numSpecies = 2;
  ambipolar = false;
  twoTemperature = false;

  SetNumActiveSpecies();
  SetNumEquations();

  const double gas_constant = 287.058;;
  // gas_constant = 1.; // for comparison against ex18
  const double specific_heat_ratio = 1.4;

  gasParams.SetSize(numSpecies, GasParams::NUM_GASPARAMS);
  gasParams = 0.0;
  for (int sp = 0; sp < numSpecies; sp++){
    gasParams(sp, GasParams::SPECIES_MW) = UNIVERSALGASCONSTANT / gas_constant;
    gasParams(sp, GasParams::SPECIES_HEAT_RATIO) = specific_heat_ratio;
  }
}

void TestBinaryAir::GetConservativesFromPrimitives(const Vector& primit,
                                                     Vector& conserv) {
  const double specific_heat_ratio = gasParams(0, GasParams::SPECIES_HEAT_RATIO);

  conserv = primit;

  double v2 = 0.;
  for (int d = 0; d < dim; d++) {
    v2 += primit[1 + d];
    conserv[1 + d] *= primit[0];
  }
  conserv[dim + 1] = primit[dim + 1] / (specific_heat_ratio - 1.) + 0.5 * primit[0] * v2;

  // NOTE: we unify to number density for species primitive.
  // This is beneficial for simple gradient computation, even for X or Y.
  for (int sp = 0; sp < numActiveSpecies; sp++)
    conserv[dim + 2 + sp] = primit[dim + 2 + sp] * gasParams(sp, GasParams::SPECIES_MW);
}

void TestBinaryAir::GetPrimitivesFromConservatives(const Vector& conserv, Vector& primit) {
  double p = ComputePressure(conserv);
  primit = conserv;

  for (int d = 0; d < dim; d++) primit[1 + d] /= conserv[0];

  primit[dim + 1] = p;

  // NOTE: we unify to number density for species primitive.
  // This is beneficial for simple gradient computation, even for X or Y.
  for (int sp = 0; sp < numActiveSpecies; sp++)
    primit[dim + 2 + sp] = conserv[dim + 2 + sp] / gasParams(sp, GasParams::SPECIES_MW * numSpecies);

  // std::cout << "conserved: " << conserv[0] << ", "
  //                            << conserv[1] << ", "
  //                            << conserv[2] << ", "
  //                            << conserv[3] << ", "
  //                            << conserv[4] << std::endl;
  //
  // std::cout << "primitive: " << primit[0] << ", "
  //                            << primit[1] << ", "
  //                            << primit[2] << ", "
  //                            << primit[3] << ", "
  //                            << primit[4] << std::endl;
}

bool TestBinaryAir::StateIsPhysical(const mfem::Vector& state) {
  bool physical = true;

  const double pres = ComputePressure(state);

  if (state(0) < 0) {
    cout << "Negative density: " << state(0) << endl;
    physical = false;
  } else if (state(dim+1) <= 0) {
    cout << "Negative energy: " << state(dim+1) << endl;
    physical = false;
  } else if (pres <= 0.) {
    cout << "Negative pressure: " << pres << endl;
    physical = false;
  } else {
    // we do not re-compute species primitives here. only check conserved variables.
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      if (state(dim + 2 + sp) < 0.0) {
        cout << "Negative species: " << state(dim + 2 + sp) << endl;
        physical = false;
      }
    }
  }

  if(~physical) {
    cout << "state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
  }

  return physical;
}

// Compute the maximum characteristic speed.
double TestBinaryAir::ComputeMaxCharSpeed(const Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double pres = ComputePressure(state);
  const double sound = sqrt(gasParams(0,GasParams::SPECIES_HEAT_RATIO) * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

// NOTE: no ambipolar, no electron, no two temperature.
void TestBinaryAir::ComputeSpeciesPrimitives(const Vector &conservedState,
                                              Vector &X_sp, Vector &Y_sp, Vector &n_sp){
  X_sp.SetSize(numSpecies);
  Y_sp.SetSize(numSpecies);
  n_sp.SetSize(numSpecies);
  X_sp = 0.0;
  Y_sp = 0.0;
  n_sp = 0.0;

  double Yb = conservedState[0];
  double n = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_sp[sp] = conservedState[dim + 2 + sp] / gasParams(sp, GasParams::SPECIES_MW);
    Y_sp[sp] = conservedState[dim + 2 + sp] / conservedState[0];
    Yb -= conservedState[dim + 2 + sp]; // will be divided by density later.
    n += n_sp[sp];
  }

  n_sp[numSpecies - 1] = Yb / gasParams(numSpecies - 1,GasParams::SPECIES_MW);
  Y_sp[numSpecies - 1] = Yb / conservedState[0];
  n += n_sp[numSpecies - 1];

  for (int sp = 0; sp < numSpecies; sp++) X_sp[sp] = n_sp[sp] / n;

  // check for physicality.
  for (int sp = 0; sp < numSpecies; sp++) {
    assert( n_sp[sp] >= 0.0 );
    assert( (X_sp[sp] >= 0.0) && (X_sp[sp] <= 1.0) );
    assert( (Y_sp[sp] >= 0.0) && (Y_sp[sp] <= 1.0) );
  }
}

void TestBinaryAir::ComputeMassFractionGradient(const Vector &state,
                                                const DenseMatrix &gradUp,
                                                DenseMatrix &massFractionGrad) {
  // Only need active species.
  massFractionGrad.SetSize(numActiveSpecies,dim);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      massFractionGrad(sp,d) = gasParams(sp,GasParams::SPECIES_MW) * gradUp(dim + 2 + sp, d) / state(0)
                                - state(dim + 2 + sp) / state(0) / state(0) * gradUp(0, d);
  }
}

void TestBinaryAir::ComputeMoleFractionGradient(const Vector &state,
                                                const DenseMatrix &gradUp,
                                                DenseMatrix &moleFractionGrad) {
  // TODO: Fluxes need to take Up as input, so that we won't recompute primitives again.
  Vector X_sp;
  Vector Y_sp;
  Vector n_sp;
  ComputeSpeciesPrimitives(state,X_sp,Y_sp,n_sp);

  double n = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) n += n_sp[sp];

  Vector nGrad(dim);
  nGrad = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) nGrad(d) += gradUp(dim + 2 + sp, d)
                                              * (1.0 - gasParams(sp,GasParams::SPECIES_MW) / gasParams(numSpecies-1,GasParams::SPECIES_MW));
  }
  for (int d = 0; d < dim; d++) nGrad(d) += gradUp(0, d) / gasParams(numSpecies-1,GasParams::SPECIES_MW);

  // Only need active species.
  moleFractionGrad.SetSize(numActiveSpecies,dim);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      moleFractionGrad(sp,d) = gradUp(dim + 2 + sp, d) / n
                                - X_sp[sp] / n * nGrad(d);
  }
}
