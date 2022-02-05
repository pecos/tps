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

void GasMixture::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up)
{
  double *pGridFunc = press->HostWrite();
  const double *UpData = Up->HostRead();

  const int nnode = press->FESpace()->GetNDofs();

  for(int n=0;n<nnode;n++){
    Vector UpAtNode(num_equation);
    for (int eq = 0; eq < num_equation; eq++) {
      UpAtNode[eq] = UpData[n + eq * nnode];
    }
    pGridFunc[n] = ComputePressureFromPrimitives(UpAtNode);
  }
}

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


// void DryAir::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up)
// {
//   double *pGridFunc = press->HostWrite();
//   const double *UpData = Up->HostRead();
//
//   const int nnode = press->FESpace()->GetNDofs();
//
//   for(int n=0;n<nnode;n++){
//     double tmp = UpData[n + (1+dim)*nnode];
//     double rho = UpData[n];
//     pGridFunc[n] = rho*gas_constant*tmp;
//   }
// }

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

  gas_constant = 287.058;;
  // gas_constant = 1.; // for comparison against ex18
  specific_heat_ratio = 1.4;

  gasParams.SetSize(numSpecies, GasParams::NUM_GASPARAMS);
  gasParams = 0.0;
  for (int sp = 0; sp < numSpecies; sp++){
    gasParams(sp, GasParams::SPECIES_MW) = UNIVERSALGASCONSTANT / gas_constant;
  }
}

void TestBinaryAir::GetConservativesFromPrimitives(const Vector& primit,
                                                     Vector& conserv) {

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
    primit[dim + 2 + sp] = conserv[dim + 2 + sp] / gasParams(sp, GasParams::SPECIES_MW);
}

double TestBinaryAir::ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive)
{
  double T, p;
  if(primitive){
    T = Uin[1+dim];
  }else{
    T = ComputeTemperature(Uin);
  }

  return gas_constant*(T*dUp_dx[0] + Uin[0]*dUp_dx[1+dim]);
}

double TestBinaryAir::ComputePressureFromPrimitives(const mfem::Vector& Up)
{
  return gas_constant*Up[0]*Up[1+dim];
}

// void TestBinaryAir::UpdatePressureGridFunction(ParGridFunction* press, const ParGridFunction* Up)
// {
//   double *pGridFunc = press->HostWrite();
//   const double *UpData = Up->HostRead();
//
//   const int nnode = press->FESpace()->GetNDofs();
//
//   for(int n=0;n<nnode;n++){
//     double tmp = UpData[n + (1+dim)*nnode];
//     double rho = UpData[n];
//     pGridFunc[n] = rho*gas_constant*tmp;
//   }
// }

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
  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

double TestBinaryAir::ComputeSpeedOfSound(const mfem::Vector& Uin, bool primitive)
{
  double T;

  if(primitive){
    T = Uin[1+dim];
  }else{
    // conservatives passed in
    T = ComputeTemperature(Uin);

  }

  return sqrt(specific_heat_ratio*gas_constant*T);
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

//////////////////////////////////////////////////////////////////////////
////// Perfect Mixture GasMixture                     ////////////////////
//////////////////////////////////////////////////////////////////////////

PerfectMixture::PerfectMixture(RunConfiguration &_runfile, int _dim)
    : GasMixture(WorkingFluid::USER_DEFINED, _dim)
{
  numSpecies = _runfile.GetNumSpecies();
  ambipolar = _runfile.IsAmbipolar();
  twoTemperature = _runfile.IsTwoTemperature();

  SetNumActiveSpecies();
  SetNumEquations();

  /*
    TODO: Currently, background species is assumed to be the last species, and electron the second to last.
    These are enforced at input parsing.
  */
  gasParams.SetSize(numSpecies, GasParams::NUM_GASPARAMS);
  molarCV_.SetSize(numSpecies);
  molarCP_.SetSize(numSpecies);
  specificGasConstants_.SetSize(numSpecies);
  specificHeatRatios_.SetSize(numSpecies);

  gasParams = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    for (int param = 0; param < GasParams::NUM_GASPARAMS; param++)
      gasParams(sp, param) = _runfile.GetGasParams(sp,(GasParams) param);

    specificGasConstants_(sp) = UNIVERSALGASCONSTANT / gasParams(sp, GasParams::SPECIES_MW);

    // TODO: read these from input parser.
    molarCV_(sp) = _runfile.getConstantMolarCV(sp) * UNIVERSALGASCONSTANT;
    molarCP_(sp) = _runfile.getConstantMolarCP(sp) * UNIVERSALGASCONSTANT;
    specificHeatRatios_(sp) = molarCP_(sp) / molarCV_(sp);
  }

  // We assume the background species is neutral.
  assert( gasParams(numSpecies - 1, GasParams::SPECIES_CHARGES) == 0.0 );
}

// compute heavy-species heat capacity from number densities.
double PerfectMixture::computeHeaviesHeatCapacity(const double *n_sp, const double &nB){
  double heatCapacity = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp==numSpecies-2) continue; // neglect electron.
    heatCapacity += n_sp[sp] * molarCV_(sp);
  }
  heatCapacity += nB * molarCV_(numSpecies - 1);

  return heatCapacity;
}

double PerfectMixture::computeAmbipolarElectronNumberDensity(const double *n_sp){
  double n_e = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_e += gasParams(sp, GasParams::SPECIES_CHARGES) * n_sp[sp];
  } // Background species doesn't have to be included due to its neutral charge.
  assert( n_e >= 0.0 );

  return n_e;
}

double PerfectMixture::computeBackgroundMassDensity(const double &rho, const double *n_sp,
                                                    double &n_e, bool isElectronComputed ) {
  if ( (~isElectronComputed) && (ambipolar) ) {
    n_e = computeAmbipolarElectronNumberDensity(n_sp);
  }

  // computing background number density.
  double rhoB = rho;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    // If not ambipolar, electron number density is included in this loop.
    rhoB -= gasParams(sp, GasParams::SPECIES_MW) * n_sp[sp];
  }

  if (ambipolar) { // Electron species is assumed be to the second to last species.
    rhoB -= n_e * gasParams(numSpecies - 2, GasParams::SPECIES_MW);
  }

  assert( rhoB >= 0.0 );

  return rhoB;
}

// TODO: Need to expand the number of primitive variables and store all of them.
// Right now, it's alwasy the same as conserved variables, to minimize gradient computation.
// Additional primitive variables (such as temperature currently) needs to be evaluated every time it is needed.
// Gradient computation can still keep the same number, while expanding the primitive variables.
void PerfectMixture::GetPrimitivesFromConservatives(const Vector &conserv, Vector &primit) {
  Vector n_sp;
  computeNumberDensities(conserv, n_sp);

  // NOTE: we fix the number density as "necessary" species primitive and
  // compute its gradient only.
  // Conversion from grad n to grad X or grad Y will be provided.
  // Conversion routines will benefit from expanding Up with all X, Y and n.
  for (int sp = 0; sp < numActiveSpecies; sp++) primit[dim + 2 + sp] = n_sp[sp];

  primit[0] = conserv[0];
  for (int d = 0; d < dim; d++) primit[d+1] = conserv[d+1] / conserv[0];

  double T_h, T_e;
  computeTemperaturesBase(conserv, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  primit[dim + 1] = T_h;

  if (twoTemperature) // electron temperature as primitive variable.
    primit[num_equation - 1] = T_e;

}

void PerfectMixture::GetConservativesFromPrimitives(const Vector &primit, Vector &conserv){
  conserv[0] = primit[0];
  for (int d = 0; d < dim; d++) conserv[d + 1] = primit[d + 1] * primit[0];

  // Convert species rhoY first.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    conserv[dim + 2 + sp] = primit[dim + 2 + sp] * gasParams(sp,GasParams::SPECIES_MW);
  }

  // NOTE: For now, we do not include all species number densities into Up.
  // This requires us to re-evaluate electron/background-species number density.
  double n_e = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&primit[dim+2]);
  } else {
    n_e = primit[dim + 2 + numSpecies - 2];
  }
  double rhoB = computeBackgroundMassDensity(primit[0], &primit[dim+2], n_e, true);
  double nB = rhoB / gasParams(numSpecies - 1, GasParams::SPECIES_MW);

  if (twoTemperature)
    conserv[num_equation - 1] = n_e * molarCV_(numSpecies - 2) * primit[num_equation - 1];

  // compute mixture heat capacity.
  double totalHeatCapacity = computeHeaviesHeatCapacity(&primit[dim+2], nB);
  if (~twoTemperature)
    totalHeatCapacity += n_e * molarCV_(numSpecies - 2);

  double totalEnergy = 0.0;
  for (int d = 0; d < dim; d++) totalEnergy += primit[d + 1] * primit[d + 1];
  totalEnergy *= 0.5 * primit[0];
  totalEnergy += totalHeatCapacity * primit[dim + 1];
  if (twoTemperature) {
    totalEnergy += conserv[num_equation - 1];
  }

  conserv[dim + 1] = totalEnergy;
}

// NOTE: Almost for sure ambipolar will remain true, then we have to always compute at least both Y and n.
// Mole fraction X will be needed almost everywhere, though it requires Y and n to be evaluated first.
// TODO: It is better to include all X, Y, n into primitive variable Up,
// in order to reduce repeated evaluation.
void PerfectMixture::computeSpeciesPrimitives(const Vector &conservedState,
                                              Vector &X_sp, Vector &Y_sp, Vector &n_sp){
  X_sp.SetSize(numSpecies);
  Y_sp.SetSize(numSpecies);
  n_sp.SetSize(numSpecies);
  X_sp = 0.0;
  Y_sp = 0.0;
  n_sp = 0.0;

  double n_e = 0.0;
  double n = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_sp[sp] = conservedState[dim + 2 + sp] / gasParams(sp, GasParams::SPECIES_MW);
    n += n_sp[sp];
    if (ambipolar)
      n_e += gasParams(sp, GasParams::SPECIES_CHARGES) * n_sp[sp];
  } // Background species doesn't have to be included due to its neutral charge.
  if (ambipolar) {
    n_sp[numSpecies - 2] = n_e; // Electron species is assumed be to the second to last species.
    n += n_e;
  }

  double Yb = 1.;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    Y_sp[sp] = conservedState[dim + 2 + sp] / conservedState[0];
    Yb -= Y_sp[sp];
  }

  if (ambipolar) { // Electron species is assumed be to the second to last species.
    Y_sp[numSpecies - 2] = n_e * gasParams(numSpecies - 2, GasParams::SPECIES_MW) / conservedState[0];
    Yb -= Y_sp[numSpecies - 2];
  }
  assert( Yb >= 0.0 ); // In case of negative mass fraction.
  Y_sp[numSpecies - 1] = Yb;

  n_sp[numSpecies - 1] = Y_sp[numSpecies - 1] * conservedState[0] / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  n += n_sp[numSpecies - 1];

  for (int sp = 0; sp < numSpecies; sp++)
    X_sp[sp] = n_sp[sp] / n;
}

void PerfectMixture::computeNumberDensities(const Vector &conservedState, Vector &n_sp){
  n_sp.SetSize(numSpecies);
  n_sp = 0.0;

  double n_e = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    n_sp[sp] = conservedState[dim + 2 + sp] / gasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&n_sp[0]);
    n_sp[numSpecies - 2] = n_e; // Electron species is assumed be to the second to last species.
  }
  double rhoB = computeBackgroundMassDensity(conservedState[0], &n_sp[0], n_e, true);

  n_sp[numSpecies - 1] = rhoB / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
}

double PerfectMixture::ComputePressureFromPrimitives(const mfem::Vector& Up)
{
  // NOTE: For now, we do not include all species number densities into Up.
  // This requires us to re-evaluate electron/background-species number density.
  double n_e = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&Up[dim+2]);
  } else {
    n_e = Up[dim + 2 + numSpecies - 2];
  }
  double rhoB = computeBackgroundMassDensity(Up[0], &Up[dim+2], n_e, true);
  double nB = rhoB / gasParams(numSpecies - 1, GasParams::SPECIES_MW);

  double T_h = Up[dim + 1];
  double T_e;
  if (twoTemperature) {
    T_e = Up[num_equation - 1];
  } else {
    T_e = Up[dim + 1];
  }
  double p = computePressureBase(&Up[dim+2], n_e, nB, T_h, T_e);

  return p;
}

// NOTE: This is almost the same as GetPrimitivesFromConservatives except storing other primitive variables.
double PerfectMixture::ComputePressure(const Vector &state) {
  Vector n_sp;
  computeNumberDensities(state, n_sp);

  // compute mixture heat capacity.
  double totalHeatCapacity = computeHeaviesHeatCapacity(&n_sp[0], n_sp[numSpecies - 1]);
  if (~twoTemperature)
    totalHeatCapacity += n_sp[numSpecies - 2] * molarCV_(numSpecies - 2);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  // NOTE: compute pressure.
  double p = computePressureBase(&n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  return p;
}

double PerfectMixture::computePressureBase(const double *n_sp, const double n_e, const double n_B,
                                           const double T_h, const double T_e) {
  // NOTE: compute pressure.
  double n_h = 0.0; // total number density of all heavy species.
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp==numSpecies-2) continue; // treat electron separately.
    n_h += n_sp[sp];
  }
  n_h += n_B;

  double p = n_h * T_h;
  if (twoTemperature) {
    p += n_e * T_e;
  } else {
    p += n_e * T_h;
  }
  p *= UNIVERSALGASCONSTANT;

  return p;
}

bool PerfectMixture::StateIsPhysical(const Vector& state) {
  bool physical = true;

  if (state(0) < 0) {
    cout << "Negative density! " << endl;
    physical = false;
  }
  if (state(dim+1) <= 0) {
    cout << "Negative energy! " << endl;
    physical = false;
  }

  /* TODO: take primitive variables as input for physicality check. */
  // Vector primitiveState(num_equation);
  // GetPrimitivesFromConservatives(state, primitiveState);
  // if (primitiveState(dim+1) < 0) {
  //   cout << "Negative pressure: " << primitiveState(dim+1) << "! " << endl;
  //   physical = false;
  // }
  // if (primitiveState(num_equation-1) <= 0) {
  //   cout << "Negative electron temperature: " << primitiveState(num_equation-1) << "! " << endl;
  //   physical = false;
  // }
  //
  // Vector n_sp;
  // Vector Y_sp;
  // Vector X_sp;
  // ComputeSpeciesPrimitives(state, X_sp, Y_sp, n_sp);
  // for (int sp = 0; sp < numSpecies; sp++){
  //   if (n_sp[sp] < 0.0) {
  //     cout << "Negative " << sp << "-th species number density: " << n_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  //   if ((X_sp[sp] < 0.0) || (X_sp[sp] > 1.0)) {
  //     cout << "Unphysical " << sp << "-th species mole fraction: " << X_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  //   if ((Y_sp[sp] < 0.0) || (Y_sp[sp] > 1.0)) {
  //     cout << "Unphysical " << sp << "-th species mass fraction: " << Y_sp[sp] << "! " << endl;
  //     physical = false;
  //   }
  // }

  if (!physical) {
    cout << "Unphysical state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
  }
  return physical;
}

// NOTE: this routine only return heavy-species temperature. Need name-change.
double PerfectMixture::ComputeTemperature(const Vector &state) {
  Vector n_sp;
  computeNumberDensities(state, n_sp);

  double T_h, T_e;
  computeTemperaturesBase(state, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  return T_h;
}

void PerfectMixture::computeTemperaturesBase(const Vector &conservedState,
                                             const double *n_sp, const double n_e, const double n_B,
                                             double &T_h, double &T_e) {
  // compute mixture heat capacity.
  double totalHeatCapacity = computeHeaviesHeatCapacity(&n_sp[0], n_B);
  if (~twoTemperature)
    totalHeatCapacity += n_e * molarCV_(numSpecies - 2);

  // Comptue heavy-species temperature. If not two temperature, then this works as the unique temperature.
  T_h = 0.0;
  for (int d = 0; d < dim; d++) T_h -= conservedState[d+1] * conservedState[d+1];
  T_h *= 0.5 / conservedState[0];
  T_h += conservedState[dim + 1];
  if (twoTemperature) T_h -= conservedState[num_equation - 1];
  T_h /= totalHeatCapacity;

  // electron temperature as primitive variable.
  if (twoTemperature) {
    T_e = conservedState[num_equation - 1] / n_e / molarCV_(numSpecies - 2);
  } else {
    T_e = T_h;
  }

  return;
}

double PerfectMixture::ComputePressureDerivative(const Vector &dUp_dx, const Vector &Uin, bool primitive) {
  if (primitive) {
    return computePressureDerivativeFromPrimitives(dUp_dx, Uin);
  } else {
    return computePressureDerivativeFromConservatives(dUp_dx, Uin);
  }
}

double PerfectMixture::computePressureDerivativeFromPrimitives(const Vector &dUp_dx, const Vector &Uin) {
  double pressureGradient = 0.0;

  double n_e = 0.0;
  double dne_dx = 0.0;
  if (ambipolar) {
    n_e = computeAmbipolarElectronNumberDensity(&Uin[dim+2]);
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      dne_dx += dUp_dx[dim + 2 + sp] * gasParams(sp, GasParams::SPECIES_CHARGES);
    }
  } else {
    n_e = Uin[dim + 2 + numSpecies - 2];
    dne_dx = dUp_dx[dim + 2 + numSpecies - 2];
  }

  double rhoB = computeBackgroundMassDensity(Uin[0], &Uin[dim+2], n_e, true);
  double nB = rhoB / gasParams(numSpecies - 1, GasParams::SPECIES_MW);

  double n_h = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp==numSpecies - 2) continue;
    n_h += Uin[dim + 2 + sp];
  }
  n_h += nB;
  pressureGradient += n_h * dUp_dx[dim + 1];

  double numDenGrad = dUp_dx[0] / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp==numSpecies - 2) continue;
    numDenGrad += dUp_dx[dim+2+sp] * (1.0 - gasParams(sp, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW));
  }
  // Kevin: this electron-related term comes from background species.
  numDenGrad -= dne_dx * gasParams(numSpecies - 2, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  pressureGradient += numDenGrad * Uin[dim + 1];

  if (twoTemperature){
    pressureGradient += n_e * dUp_dx[num_equation - 1] + dne_dx * Uin[num_equation - 1];
  } else {
    pressureGradient += n_e * dUp_dx[dim + 1] + dne_dx * Uin[dim + 1];
  }

  pressureGradient *= UNIVERSALGASCONSTANT;
  return pressureGradient;
}

double PerfectMixture::computePressureDerivativeFromConservatives(const Vector &dUp_dx, const Vector &Uin) {
  double pressureGradient = 0.0;

  Vector n_sp;
  computeNumberDensities(Uin, n_sp);
  double dne_dx = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      dne_dx += dUp_dx[dim + 2 + sp] * gasParams(sp, GasParams::SPECIES_CHARGES);
    }
  } else {
    dne_dx = dUp_dx[dim + 2 + numSpecies - 2];
  }

  double T_h, T_e;
  computeTemperaturesBase(Uin, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

  double n_h = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) {
    if (sp==numSpecies - 2) continue;
    n_h += n_sp[sp];
  }
  pressureGradient += n_h * dUp_dx[dim + 1];

  double numDenGrad = dUp_dx[0] / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp==numSpecies - 2) continue;
    numDenGrad += dUp_dx[dim+2+sp] * (1.0 - gasParams(sp, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW));
  }
  // Kevin: this electron-related term comes from background species.
  numDenGrad -= dne_dx * gasParams(numSpecies - 2, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW);
  pressureGradient += numDenGrad * T_h;

  if (twoTemperature){
    pressureGradient += n_sp[numSpecies - 2] * dUp_dx[num_equation - 1] + dne_dx * T_e;
  } else {
    pressureGradient += n_sp[numSpecies - 2] * dUp_dx[dim + 1] + dne_dx * T_h;
  }

  pressureGradient *= UNIVERSALGASCONSTANT;
  return pressureGradient;
}

double PerfectMixture::computeHeaviesMixtureCV(const double *n_sp, const double n_B) {
  double mixtureCV = 0.0;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    mixtureCV += n_sp[sp] * molarCV_(sp);
  }
  mixtureCV += n_B * molarCV_(numSpecies - 1);

  return mixtureCV;
}

double PerfectMixture::computeHeaviesMixtureHeatRatio(const double *n_sp, const double n_B) {
  double mixtureCV = computeHeaviesMixtureCV(n_sp, n_B);
  double n_h = n_B;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    if (sp == numSpecies - 2) continue;
    n_h += n_sp[sp];
  }

  return 1.0 + n_h * UNIVERSALGASCONSTANT / mixtureCV;
}

double PerfectMixture::computeSpeedOfSoundBase(const double *n_sp, const double n_B, const double rho, const double p) {
  double gamma = computeHeaviesMixtureHeatRatio(n_sp, n_B);

  return sqrt(gamma * p / rho);
}

// Compute the maximum characteristic speed.
double PerfectMixture::ComputeMaxCharSpeed(const Vector& state) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  const double sound = ComputeSpeedOfSound(state, false);
  const double vel = sqrt(den_vel2 / den);

  return vel + sound;
}

double PerfectMixture::ComputeSpeedOfSound(const mfem::Vector& Uin, bool primitive)
{
  if(primitive){

    double n_e = 0.0;
    if (ambipolar) {
      n_e = computeAmbipolarElectronNumberDensity(&Uin[dim+2]);
    } else {
      n_e = Uin[dim + 2 + numSpecies - 2];
    }
    double rhoB = computeBackgroundMassDensity(Uin[0], &Uin[dim+2], n_e, true);
    double nB = rhoB / gasParams(numSpecies - 1, GasParams::SPECIES_MW);

    double T_e = (twoTemperature) ? Uin[num_equation-1] : Uin[dim+1];
    double p = computePressureBase(&Uin[dim+2], n_e, nB, Uin[dim+1], T_e);

    return computeSpeedOfSoundBase(&Uin[dim+2], nB, Uin[0], p);

  } else {

    Vector n_sp;
    computeNumberDensities(Uin, n_sp);

    double T_h, T_e;
    computeTemperaturesBase(Uin, &n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

    double p = computePressureBase(&n_sp[0], n_sp[numSpecies - 2], n_sp[numSpecies - 1], T_h, T_e);

    return computeSpeedOfSoundBase(&n_sp[0], n_sp[numSpecies - 1], Uin[0], p);

  }
}

// NOTE: numberDensities have all species number density.
void PerfectMixture::computeMassFractionGradient(const double rho,
                                                 const Vector &numberDensities,
                                                 const DenseMatrix &gradUp,
                                                 DenseMatrix &massFractionGrad) {
  massFractionGrad.SetSize(numSpecies,dim);
  massFractionGrad = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) { // if not ambipolar, electron is included.
    for (int d = 0; d < dim; d++) {
      massFractionGrad(sp,d) = gradUp(dim + 2 + sp, d) / rho - numberDensities(sp) / rho / rho * gradUp(0, d);
      massFractionGrad(sp,d) *= gasParams(sp,GasParams::SPECIES_MW);
    }
  }

  Vector neGrad(dim);
  neGrad = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      for (int d = 0; d < dim; d++)
        neGrad(d) += gradUp(dim + 2 + sp, d) * gasParams(sp, GasParams::SPECIES_CHARGES);
    }
    for (int d = 0; d < dim; d++) {
      massFractionGrad(numSpecies - 2,d) = neGrad(d) / rho - numberDensities(numSpecies - 2) / rho / rho * gradUp(0, d);
      massFractionGrad(numSpecies - 2,d) *= gasParams(numSpecies - 2,GasParams::SPECIES_MW);
    }
  }

  Vector mGradN(dim);
  mGradN = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      mGradN(d) += gradUp(dim+2+sp, d) * gasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    for (int d = 0; d < dim; d++)
      mGradN(d) += neGrad(d) * gasParams(numSpecies - 2, GasParams::SPECIES_MW);
  }
  double Yb = numberDensities(numSpecies - 1) * gasParams(numSpecies - 1, GasParams::SPECIES_MW) / rho;
  for (int d = 0; d < dim; d++) {
    massFractionGrad(numSpecies - 1,d) = ( (1.0 - Yb) * gradUp(0, d) - mGradN(d) ) / rho;
  }

}

void PerfectMixture::computeMoleFractionGradient(const Vector &numberDensities,
                                                 const DenseMatrix &gradUp,
                                                 DenseMatrix &moleFractionGrad) {
  moleFractionGrad.SetSize(numSpecies,dim);
  double totalN = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) totalN += numberDensities(sp);

  Vector neGrad;
  neGrad.SetSize(dim);
  neGrad = 0.0;
  if (ambipolar) {
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      for (int d = 0; d < dim; d++)
        neGrad(d) += gradUp(dim + 2 + sp, d) * gasParams(sp, GasParams::SPECIES_CHARGES);
    }
  }

  Vector nBGrad(dim);
  for (int d = 0; d < dim; d++) nBGrad(d) = gradUp(0,d);
  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++)
      nBGrad(d) -= gradUp(dim+2+sp, d) * gasParams(sp, GasParams::SPECIES_MW);
  }
  if (ambipolar) {
    // nBGrad -= gasParams(numSpecies - 2, GasParams::SPECIES_MW) / gasParams(numSpecies - 1, GasParams::SPECIES_MW) * neGrad;
    nBGrad.Add( -gasParams(numSpecies - 2, GasParams::SPECIES_MW), neGrad);
  }
  nBGrad /= gasParams(numSpecies - 1, GasParams::SPECIES_MW);

  Vector totalNGrad;
  totalNGrad.SetSize(dim);
  totalNGrad = 0.0;
  for (int sp = 0; sp < numActiveSpecies; sp++) { // if not ambipolar, electron is included.
    for (int d = 0; d < dim; d++)
      totalNGrad(d) += gradUp(dim + 2 + sp, d);
  }
  if (ambipolar) totalNGrad += neGrad;
  totalNGrad += nBGrad;

  for (int sp = 0; sp < numActiveSpecies; sp++) {
    for (int d = 0; d < dim; d++) {
      moleFractionGrad(sp,d) = gradUp(dim + 2 + sp, d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
    }
  }
  if (ambipolar) {
    int sp = numSpecies - 2;
    for (int d = 0; d < dim; d++) {
      moleFractionGrad(sp,d) = neGrad(d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
    }
  }
  int sp = numSpecies - 1;
  for (int d = 0; d < dim; d++) {
    moleFractionGrad(sp,d) = nBGrad(d) / totalN - numberDensities(sp) / totalN / totalN * totalNGrad(d);
  }

}
