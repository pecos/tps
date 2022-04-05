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
#include <tps_config.h>

#ifdef HAVE_MASA
#include "M2ulPhyS.hpp"
#include "masa_handler.hpp"

void M2ulPhyS::initMasaHandler() {
  assert(config.use_mms_);

  // Initialize MASA
  if (config.mms_name_ == "navierstokes_2d_compressible") {
    dryair3d::initNS2DCompressible(dim, config);
  } else if (config.mms_name_ == "euler_transient_3d") {
    dryair3d::initEuler3DTransient(dim, config);
  } else if (config.mms_name_ == "navierstokes_3d_transient_sutherland") {
    dryair3d::initNS3DTransient(dim, config);
  } else if (config.mms_name_ == "periodic_argon_ternary_2d") {
    double Lx, Ly;
    std::string basepath("mms/periodic_argon_ternary_2d");
    tpsP->getRequiredInput((basepath + "/Lx").c_str(), Lx);
    tpsP->getRequiredInput((basepath + "/Ly").c_str(), Ly);

    argon2d::initPeriodicArgonTernary2D(mixture, config, Lx, Ly);
  } else {
    grvy_printf(GRVY_ERROR, "Unknown manufactured solution > %s\n", config.mms_name_.c_str());
    exit(-1);
  }

  // check that masa at least thinks things are okie-dokie
  int ierr = MASA::masa_sanity_check<double>();
  if (ierr != 0) {
    std::cout << "*** WARNING: MASA sanity check returned error = " << ierr << " ***" << std::endl;
    std::cout << "Current parameters are as follows" << std::endl;
    MASA::masa_display_param<double>();
    exit(-1);
  }

  // Initialize mms vector function coefficients
  initMMSCoefficients();
}

void M2ulPhyS::projectExactSolution(const double _time) {
  void (*exactSolnFunction)(const Vector &, double, Vector &);

  if (config.workFluid == DRY_AIR) {
    if (dim == 2) {
      grvy_printf(GRVY_ERROR, "Currently does not support MASA solution for dry air, 2d ns equation.\n");
      exit(-1);
    }
    exactSolnFunction = &(dryair3d::exactSolnFunction);
  } else if (config.mms_name_ == "periodic_argon_ternary_2d") {
    exactSolnFunction = &(argon2d::exactSolnFunction);
  }

  VectorFunctionCoefficient u0(num_equation, exactSolnFunction);
  u0.SetTime(_time);
  U->ProjectCoefficient(u0);
}

void M2ulPhyS::initMMSCoefficients() {
  if (config.workFluid == DRY_AIR) {
    DenMMS_ = new VectorFunctionCoefficient(1, &(dryair3d::exactDenFunction));
    VelMMS_ = new VectorFunctionCoefficient(dim, &(dryair3d::exactVelFunction));
    PreMMS_ = new VectorFunctionCoefficient(1, &(dryair3d::exactPreFunction));
  } else if (config.mms_name_ == "periodic_argon_ternary_2d") {
    stateMMS_ = new VectorFunctionCoefficient(num_equation, &(argon2d::exactSolnFunction));

    Vector componentWindow(num_equation);
    componentWindow_.resize(num_equation);
    for (int eq = 0; eq < num_equation; eq++) {
      componentWindow = 0.0;
      componentWindow(eq) = 1.0;
      componentWindow_[eq] = new VectorConstantCoefficient(componentWindow);
    }

    // set up origin vector to compute L2 norm via ComputeLpError.
    zeroUBlock_ = new BlockVector(*offsets);
    zeroU_ = new ParGridFunction(vfes, zeroUBlock_->HostReadWrite());
    double *dataZeros = zeroU_->HostReadWrite();
    int NDof = vfes->GetNDofs();
    for (int i = 0; i < NDof; i++) {
      for (int eq = 0; eq < num_equation; eq++) {
        dataZeros[i + eq * NDof] = 0.0;
      }
    }
  }
}

void M2ulPhyS::checkSolutionError(const double _time) {
  rhsOperator->updatePrimitives(*U);
  mixture->UpdatePressureGridFunction(press, Up);

  // and dump error before we take any steps
  if (config.workFluid == DRY_AIR) {
    DenMMS_->SetTime(_time);
    VelMMS_->SetTime(_time);
    PreMMS_->SetTime(_time);
    const double errorDen = dens->ComputeLpError(2, *DenMMS_);
    const double errorVel = vel->ComputeLpError(2, *VelMMS_);
    const double errorPre = press->ComputeLpError(2, *PreMMS_);
    if (mpi.Root())
      cout << "time step: " << iter << ", physical time " << _time << "s"
           << ", Dens. error: " << errorDen << " Vel. " << errorVel << " press. " << errorPre << endl;
  } else if (config.mms_name_ == "periodic_argon_ternary_2d") {
    Coefficient *nullPtr = NULL;

    stateMMS_->SetTime(_time);
    Vector componentErrors(num_equation), componentRelErrors(num_equation);
    componentErrors = 0.0;
    componentRelErrors = 0.0;
    for (int eq = 0; eq < num_equation; eq++) {
      componentErrors(eq) = U->ComputeLpError(2, *stateMMS_, nullPtr, componentWindow_[eq]);
      componentRelErrors(eq) = componentErrors(eq) / zeroU_->ComputeLpError(2, *stateMMS_, nullPtr, componentWindow_[eq]);
    }
    if (mpi.Root()) {
      grvy_printf(GRVY_INFO, "\ntime step: %d, physical time: %.5E\n", iter, _time);
      grvy_printf(GRVY_INFO, "component L2-error: (%.8E, %.8E, %.8E, %.8E, %.8E) \n",
                             componentErrors(0), componentErrors(1), componentErrors(2), componentErrors(3), componentErrors(4));
      grvy_printf(GRVY_INFO, "component relative-error: (%.4E, %.4E, %.4E, %.4E, %.4E) \n",
                             componentRelErrors(0), componentRelErrors(1), componentRelErrors(2), componentRelErrors(3), componentRelErrors(4));
    }
  }
}

namespace dryair3d {

void evaluateForcing(const Vector &x, double time, Array<double> &y) {
  y[0] = MASA::masa_eval_source_rho<double>(x[0], x[1], x[2], time);  // rho
  y[1] = MASA::masa_eval_source_u<double>(x[0], x[1], x[2], time);    // rho*u
  y[2] = MASA::masa_eval_source_v<double>(x[0], x[1], x[2], time);    // rho*v
  y[3] = MASA::masa_eval_source_w<double>(x[0], x[1], x[2], time);    // rho*w
  y[4] = MASA::masa_eval_source_e<double>(x[0], x[1], x[2], time);    // rhp*e
}

void exactSolnFunction(const Vector &x, double tin, Vector &y) {
  // TODO(kevin): make one for NS2DCompressible.
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1], x[2], tin);  // rho
  y(1) = y[0] * MASA::masa_eval_exact_u<double>(x[0], x[1], x[2], tin);
  y(2) = y[0] * MASA::masa_eval_exact_v<double>(x[0], x[1], x[2], tin);
  y(3) = y[0] * MASA::masa_eval_exact_w<double>(x[0], x[1], x[2], tin);
  y(4) = MASA::masa_eval_exact_p<double>(x[0], x[1], x[2], tin) / (gamma - 1.);

  double k = 0.;
  for (int d = 0; d < x.Size(); d++) k += y[1 + d] * y[1 + d];
  k *= 0.5 / y[0];
  y[4] += k;
}

void exactDenFunction(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1], x[2], tin);  // rho
}

void exactVelFunction(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_u<double>(x[0], x[1], x[2], tin);
  y(1) = MASA::masa_eval_exact_v<double>(x[0], x[1], x[2], tin);
  y(2) = MASA::masa_eval_exact_w<double>(x[0], x[1], x[2], tin);
}

void exactPreFunction(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_p<double>(x[0], x[1], x[2], tin);
}

void initNS2DCompressible(const int dim, RunConfiguration &config) {
  assert(dim == 2);
  assert(config.workFluid == DRY_AIR);
  assert(config.mms_name_ == "navierstokes_2d_compressible");

  MASA::masa_init<double>("exact", "navierstokes_2d_compressible");
  MASA::masa_init<double>("forcing", "navierstokes_2d_compressible");
}

void initEuler3DTransient(const int dim, RunConfiguration &config) {
  assert(dim == 3);
  assert(config.workFluid == DRY_AIR);
  assert(config.GetEquationSystem() == EULER);
  assert(config.mms_name_ == "euler_transient_3d");

  MASA::masa_init<double>("forcing handler", "euler_transient_3d");

  // fluid parameters
  MASA::masa_set_param<double>("Gamma", 1.4);

  // solution parameters
  MASA::masa_set_param<double>("L", 2);

  MASA::masa_set_param<double>("rho_0", 1.0);
  MASA::masa_set_param<double>("rho_x", 0.1);
  MASA::masa_set_param<double>("rho_y", 0.1);
  MASA::masa_set_param<double>("rho_z", 0.0);
  MASA::masa_set_param<double>("rho_t", 0.15);

  MASA::masa_set_param<double>("u_0", 130.0);
  MASA::masa_set_param<double>("u_x", 10.0);
  MASA::masa_set_param<double>("u_y", 5.0);
  MASA::masa_set_param<double>("u_z", 0.0);
  MASA::masa_set_param<double>("u_t", 10.0);

  MASA::masa_set_param<double>("v_0", 5.0);
  MASA::masa_set_param<double>("v_x", 1.0);
  MASA::masa_set_param<double>("v_y", -1.0);
  MASA::masa_set_param<double>("v_z", 0.0);
  MASA::masa_set_param<double>("v_t", 2.0);

  MASA::masa_set_param<double>("w_0", 0.0);
  MASA::masa_set_param<double>("w_x", 2.0);
  MASA::masa_set_param<double>("w_y", 1.0);
  MASA::masa_set_param<double>("w_z", 0.0);
  MASA::masa_set_param<double>("w_t", -1.0);

  MASA::masa_set_param<double>("p_0", 101300.0);
  MASA::masa_set_param<double>("p_x", 101.0);
  MASA::masa_set_param<double>("p_y", 101.0);
  MASA::masa_set_param<double>("p_z", 0.0);
  MASA::masa_set_param<double>("p_t", 1013.0);

  MASA::masa_set_param<double>("a_rhox", 2.);
  MASA::masa_set_param<double>("a_rhoy", 2.);
  MASA::masa_set_param<double>("a_rhoz", 0.);
  MASA::masa_set_param<double>("a_rhot", 400.);

  MASA::masa_set_param<double>("a_ux", 2.);
  MASA::masa_set_param<double>("a_uy", 2.);
  MASA::masa_set_param<double>("a_uz", 0.);
  MASA::masa_set_param<double>("a_ut", 400.);

  MASA::masa_set_param<double>("a_vx", 2.);
  MASA::masa_set_param<double>("a_vy", 2.);
  MASA::masa_set_param<double>("a_vz", 0.);
  MASA::masa_set_param<double>("a_vt", 400.);

  MASA::masa_set_param<double>("a_wx", 2.);
  MASA::masa_set_param<double>("a_wy", 2.);
  MASA::masa_set_param<double>("a_wz", 0.);
  MASA::masa_set_param<double>("a_wt", 0.);

  MASA::masa_set_param<double>("a_px", 2.);
  MASA::masa_set_param<double>("a_py", 2.);
  MASA::masa_set_param<double>("a_pz", 0.);
  MASA::masa_set_param<double>("a_pt", 400.);
}

void initNS3DTransient(const int dim, RunConfiguration &config) {
  assert(dim == 3);
  assert(config.workFluid == DRY_AIR);
  assert(config.GetEquationSystem() == NS);
  assert(config.mms_name_ == "navierstokes_3d_transient_sutherland");

  const double viscMult = config.visc_mult;

  MASA::masa_init<double>("forcing handler", "navierstokes_3d_transient_sutherland");

  // fluid parameters
  MASA::masa_set_param<double>("Gamma", 1.4);
  MASA::masa_set_param<double>("R", 287.058);
  MASA::masa_set_param<double>("Pr", 0.71);
  MASA::masa_set_param<double>("B_mu", 110.4);
  MASA::masa_set_param<double>("A_mu", viscMult * 1.458e-6);

  // soln parameters
  MASA::masa_set_param<double>("L", 2);
  MASA::masa_set_param<double>("Lt", 2);

  MASA::masa_set_param<double>("rho_0", 1.0);
  MASA::masa_set_param<double>("rho_x", 0.1);
  MASA::masa_set_param<double>("rho_y", 0.1);
  MASA::masa_set_param<double>("rho_z", 0.0);
  MASA::masa_set_param<double>("rho_t", 0.15);

  MASA::masa_set_param<double>("u_0", 130.0);
  MASA::masa_set_param<double>("u_x", 10.0);
  MASA::masa_set_param<double>("u_y", 5.0);
  MASA::masa_set_param<double>("u_z", 0.0);
  MASA::masa_set_param<double>("u_t", 10.0);

  MASA::masa_set_param<double>("v_0", 5.0);
  MASA::masa_set_param<double>("v_x", 1.0);
  MASA::masa_set_param<double>("v_y", -1.0);
  MASA::masa_set_param<double>("v_z", 0.0);
  MASA::masa_set_param<double>("v_t", 2.0);

  MASA::masa_set_param<double>("w_0", 0.0);
  MASA::masa_set_param<double>("w_x", 2.0);
  MASA::masa_set_param<double>("w_y", 1.0);
  MASA::masa_set_param<double>("w_z", 0.0);
  MASA::masa_set_param<double>("w_t", -1.0);

  MASA::masa_set_param<double>("p_0", 101300.0);
  MASA::masa_set_param<double>("p_x", 101.0);
  MASA::masa_set_param<double>("p_y", 101.0);
  MASA::masa_set_param<double>("p_z", 0.0);
  MASA::masa_set_param<double>("p_t", 1013.0);

  MASA::masa_set_param<double>("a_rhox", 2.);
  MASA::masa_set_param<double>("a_rhoy", 2.);
  MASA::masa_set_param<double>("a_rhoz", 0.);
  MASA::masa_set_param<double>("a_rhot", 400.);

  MASA::masa_set_param<double>("a_ux", 2.);
  MASA::masa_set_param<double>("a_uy", 2.);
  MASA::masa_set_param<double>("a_uz", 0.);
  MASA::masa_set_param<double>("a_ut", 400.);

  MASA::masa_set_param<double>("a_vx", 2.);
  MASA::masa_set_param<double>("a_vy", 2.);
  MASA::masa_set_param<double>("a_vz", 0.);
  MASA::masa_set_param<double>("a_vt", 400.);

  MASA::masa_set_param<double>("a_wx", 2.);
  MASA::masa_set_param<double>("a_wy", 2.);
  MASA::masa_set_param<double>("a_wz", 0.);
  MASA::masa_set_param<double>("a_wt", 0.);

  MASA::masa_set_param<double>("a_px", 2.);
  MASA::masa_set_param<double>("a_py", 2.);
  MASA::masa_set_param<double>("a_pz", 0.);
  MASA::masa_set_param<double>("a_pt", 400.);
}

}  // namespace dryair3d

namespace argon2d {

void initPeriodicArgonTernary2D(GasMixture *mixture, RunConfiguration &config,
                                const double Lx, const double Ly) {
  // assert(dim == 2);
  // assert(config.workFluid == DRY_AIR);
  assert(config.mms_name_ == "periodic_argon_ternary_2d");
  assert(config.numSpecies == 3);
  assert(config.transportModel == CONSTANT);

  MASA::masa_init<double>("forcing handler", "periodic_argon_ternary_2d");

  // MASA::masa_set_param<double>("u0", 1.5);
  // MASA::masa_set_param<double>("dux", 0.1);
  // MASA::masa_set_param<double>("duy", 0.2);
  MASA::masa_set_param<double>("u0", 1.0);
  MASA::masa_set_param<double>("dux", 0.0);
  MASA::masa_set_param<double>("duy", 0.0);

  MASA::masa_set_param<double>("kux", 1.0);
  MASA::masa_set_param<double>("kuy", 2.0);
  MASA::masa_set_param<double>("offset_ux", - 0.33);
  MASA::masa_set_param<double>("offset_uy", 0.47);

  // MASA::masa_set_param<double>("v0", 0.91);
  // MASA::masa_set_param<double>("dvx", 0.13);
  // MASA::masa_set_param<double>("dvy", 0.11);
  MASA::masa_set_param<double>("v0", 1.0);
  MASA::masa_set_param<double>("dvx", 0.0);
  MASA::masa_set_param<double>("dvy", 0.0);

  MASA::masa_set_param<double>("kvx", 2.0);
  MASA::masa_set_param<double>("kvy", 1.0);
  MASA::masa_set_param<double>("offset_vx", 0.11);
  MASA::masa_set_param<double>("offset_vy", 0.92);

  MASA::masa_set_param<double>("n0", 40.0);
  MASA::masa_set_param<double>("X0", 0.3);
  MASA::masa_set_param<double>("dX", 0.11);
  // MASA::masa_set_param<double>("dX", 0.0);
  MASA::masa_set_param<double>("T0", 500.0);
  // MASA::masa_set_param<double>("dT", 37.0);
  MASA::masa_set_param<double>("dT", 0.0);

  const int numSpecies = mixture->GetNumSpecies();
  MASA::masa_set_param<double>("mA", mixture->GetGasParams(numSpecies - 1, GasParams::SPECIES_MW));
  MASA::masa_set_param<double>("mI", mixture->GetGasParams(0, GasParams::SPECIES_MW));
  MASA::masa_set_param<double>("mE", mixture->GetGasParams(numSpecies - 2, GasParams::SPECIES_MW));

  MASA::masa_set_param<double>("R", UNIVERSALGASCONSTANT);

  MASA::masa_set_param<double>("CV_A", mixture->getMolarCV(numSpecies - 1));
  MASA::masa_set_param<double>("CV_I", mixture->getMolarCV(0));
  MASA::masa_set_param<double>("CV_E", mixture->getMolarCV(numSpecies - 2));

  MASA::masa_set_param<double>("CP_A", mixture->getMolarCP(numSpecies - 1));
  MASA::masa_set_param<double>("CP_I", mixture->getMolarCP(0));
  MASA::masa_set_param<double>("CP_E", mixture->getMolarCP(numSpecies - 2));

  MASA::masa_set_param<double>("formEnergy_I", mixture->GetGasParams(0, GasParams::FORMATION_ENERGY));

  MASA::masa_set_param<double>("Lx", Lx);
  MASA::masa_set_param<double>("Ly", Ly);
  MASA::masa_set_param<double>("kx", 2.0);
  MASA::masa_set_param<double>("ky", 1.0);
  MASA::masa_set_param<double>("offset_x", 0.17);
  MASA::masa_set_param<double>("offset_y", 0.58);

  MASA::masa_set_param<double>("kTx", 1.0);
  MASA::masa_set_param<double>("kTy", 1.0);
  MASA::masa_set_param<double>("offset_Tx", 0.71);
  MASA::masa_set_param<double>("offset_Ty", 0.29);

  MASA::masa_set_param<double>("mu", config.constantTransport.viscosity);
  MASA::masa_set_param<double>("muB", config.constantTransport.bulkViscosity);
  MASA::masa_set_param<double>("k_heat", config.constantTransport.thermalConductivity
                                          + config.constantTransport.electronThermalConductivity);
  MASA::masa_set_param<double>("D_A", config.constantTransport.diffusivity(numSpecies - 1));
  MASA::masa_set_param<double>("D_I", config.constantTransport.diffusivity(0));
  MASA::masa_set_param<double>("D_E", config.constantTransport.diffusivity(numSpecies - 2));

  MASA::masa_set_param<double>("qe", ELECTRONCHARGE);
  MASA::masa_set_param<double>("kB", BOLTZMANNCONSTANT);

  MASA::masa_set_param<double>("ZI", mixture->GetGasParams(0, GasParams::SPECIES_CHARGES));
  MASA::masa_set_param<double>("ZE", mixture->GetGasParams(numSpecies - 2, GasParams::SPECIES_CHARGES));
}

// TODO(kevin): make it numSpecies-insensitive.
void exactSolnFunction(const Vector &x, double tin, Vector &y) {
  const int num_equation = 5;
  for (int eq = 0; eq < num_equation; eq++) {
    y(eq) = MASA::masa_eval_exact_state<double>(x[0], x[1], eq);
  }
}

void evaluateForcing(const Vector &x, double time, Array<double> &y) {
  const int num_equation = 5;
  for (int eq = 0; eq < num_equation; eq++) {
    y[eq] = MASA::masa_eval_source_state<double>(x[0], x[1], eq);
  }
}

}  // namespace argon2d
#endif
