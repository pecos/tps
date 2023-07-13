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
#include <tps_config.h>

#ifdef HAVE_MASA
#include "M2ulPhyS.hpp"
#include "masa_handler.hpp"

void M2ulPhyS::initMasaHandler() {
  assert(config.use_mms_);

  // Initialize MASA
  if (config.mms_name_ == "ad_cns_2d_sutherlands") {
    dryair2d::initCNS2DSutherlands(dim, config);
  } else if (config.mms_name_ == "euler_2d") {
    dryair2d::initEuler2D(dim, config);
  } else if (config.mms_name_ == "euler_transient_3d") {
    dryair3d::initEuler3DTransient(dim, config);
  } else if (config.mms_name_ == "navierstokes_3d_transient_sutherland") {
    dryair3d::initNS3DTransient(dim, config);
  } else if (config.mms_name_.substr(0, 10) == "ternary_2d") {
    double Lx, Ly;
    std::string basepath("mms/ternary_2d");
    tpsP->getRequiredInput((basepath + "/Lx").c_str(), Lx);
    tpsP->getRequiredInput((basepath + "/Ly").c_str(), Ly);

    if (config.mms_name_ == "ternary_2d_periodic") {
      ternary2d::initTernary2DPeriodic(mixture, config, Lx, Ly);
    } else if (config.mms_name_ == "ternary_2d_periodic_ambipolar") {
      ternary2d::initTernary2DPeriodicAmbipolar(mixture, config, Lx, Ly);
    } else if (config.mms_name_ == "ternary_2d_2t_periodic_ambipolar") {
      ternary2d::initTernary2D2TPeriodicAmbipolar(mixture, config, Lx, Ly);
    } else if (config.mms_name_ == "ternary_2d_2t_ambipolar_wall") {
      ternary2d::initTernary2D2TAmbipolarWall(mixture, config, Lx, Ly);
    } else if (config.mms_name_ == "ternary_2d_2t_ambipolar_inoutlet") {
      ternary2d::initTernary2D2TAmbipolarInoutlet(mixture, config, Lx, Ly);
    } else if (config.mms_name_ == "ternary_2d_sheath") {
      ternary2d::initTernary2DSheath(mixture, config, Lx, Ly);
    } else {
      grvy_printf(GRVY_ERROR, "Unknown ternary 2d solution > %s\n", config.mms_name_.c_str());
      exit(-1);
    }
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

void M2ulPhyS::projectExactSolution(const double _time, ParGridFunction *prjU) {
  void (*exactSolnFunction)(const Vector &, double, Vector &);

  if (config.workFluid == DRY_AIR || config.workFluid == LTE_FLUID) {
    if (dim == 2) {
      exactSolnFunction = &(dryair2d::exactSolnFunction);
    } else {
      exactSolnFunction = &(dryair3d::exactSolnFunction);
    }
  } else {
    exactSolnFunction = &(mms::exactSolnFunction);
  }

  VectorFunctionCoefficient u0(num_equation, exactSolnFunction);
  u0.SetTime(_time);
  prjU->ProjectCoefficient(u0);
}

void M2ulPhyS::initMMSCoefficients() {
  if (config.workFluid == DRY_AIR || config.workFluid == LTE_FLUID) {
    DenMMS_ = new VectorFunctionCoefficient(1, &(dryair3d::exactDenFunction));
    VelMMS_ = new VectorFunctionCoefficient(dim, &(dryair3d::exactVelFunction));
    PreMMS_ = new VectorFunctionCoefficient(1, &(dryair3d::exactPreFunction));
  } else {
    stateMMS_ = new VectorFunctionCoefficient(num_equation, &(mms::exactSolnFunction));

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

void M2ulPhyS::checkSolutionError(const double _time, const bool final) {
  rhsOperator->updatePrimitives(*U);
  mixture->UpdatePressureGridFunction(press, Up);

  // and dump error before we take any steps
  if (config.workFluid == DRY_AIR || config.workFluid == LTE_FLUID) {
    DenMMS_->SetTime(_time);
    VelMMS_->SetTime(_time);
    PreMMS_->SetTime(_time);
    const double errorDen = dens->ComputeLpError(2, *DenMMS_);
    const double errorVel = vel->ComputeLpError(2, *VelMMS_);
    const double errorPre = press->ComputeLpError(2, *PreMMS_);
    if (rank0_)
      cout << "time step: " << iter << ", physical time " << _time << "s"
           << ", Dens. error: " << errorDen << " Vel. " << errorVel << " press. " << errorPre << endl;
  } else {
    Coefficient *nullPtr = NULL;

    stateMMS_->SetTime(_time);
    Vector componentErrors(num_equation), componentRelErrors(num_equation);
    componentErrors = 0.0;
    componentRelErrors = 0.0;
    for (int eq = 0; eq < num_equation; eq++) {
      componentErrors(eq) = U->ComputeLpError(2, *stateMMS_, nullPtr, componentWindow_[eq]);
      componentRelErrors(eq) =
          componentErrors(eq) / zeroU_->ComputeLpError(2, *stateMMS_, nullPtr, componentWindow_[eq]);
    }

    int numElems;
    if (final) {
      int localElems = mesh->GetNE();
      MPI_Allreduce(&localElems, &numElems, 1, MPI_INT, MPI_SUM, mesh->GetComm());
    }

    if (rank0_) {
      grvy_printf(GRVY_INFO, "\ntime step: %d, physical time: %.5E\n", iter, _time);
      grvy_printf(GRVY_INFO, "component L2-error: (");
      for (int eq = 0; eq < num_equation; eq++) {
        std::string format = (eq == num_equation - 1) ? "%.8E) \n" : "%.8E, ";
        grvy_printf(GRVY_INFO, format.c_str(), componentErrors(eq));
      }
      grvy_printf(GRVY_INFO, "component relative-error: (");
      for (int eq = 0; eq < num_equation; eq++) {
        std::string format = (eq == num_equation - 1) ? "%.8E) \n" : "%.8E, ";
        grvy_printf(GRVY_INFO, format.c_str(), componentRelErrors(eq));
      }

      if (final) {
        ofstream fID;
        std::string filename = config.mms_name_ + ".rel_error.txt";
        fID.open(filename, std::ios_base::out);
        fID << numElems << "\t";
        for (int eq = 0; eq < num_equation; eq++) fID << componentRelErrors(eq) << "\t";
        fID << "\n";
        fID.close();
      }
    }
  }
}

namespace mms {

void exactSolnFunction(const Vector &x, double tin, Vector &y) {
  std::vector<double> y1(y.Size());
  MASA::masa_eval_exact_state<double>(x[0], x[1], y1);
  for (int eq = 0; eq < y.Size(); eq++) y[eq] = y1[eq];
}

void evaluateForcing(const Vector &x, double time, Array<double> &y) {
  std::vector<double> y1(y.Size());
  MASA::masa_eval_source_state<double>(x[0], x[1], y1);
  for (int eq = 0; eq < y.Size(); eq++) y[eq] = y1[eq];
}

}  // namespace mms

namespace dryair2d {

void evaluateForcing(const Vector &x, double time, Array<double> &y) {
  y[0] = MASA::masa_eval_source_rho<double>(x[0], x[1]);    // rho
  y[1] = MASA::masa_eval_source_rho_u<double>(x[0], x[1]);  // rho*u
  y[2] = MASA::masa_eval_source_rho_v<double>(x[0], x[1]);  // rho*v
  y[3] = MASA::masa_eval_source_rho_e<double>(x[0], x[1]);  // rhp*e
}

void exactSolnFunction(const Vector &x, double tin, Vector &y) {
  // TODO(kevin): make one for NS2DCompressible.
  MFEM_ASSERT(x.Size() == 2, "");

  DryAir eqState;
  const double gamma = eqState.GetSpecificHeatRatio();

  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1]);  // rho
  y(1) = y[0] * MASA::masa_eval_exact_u<double>(x[0], x[1]);
  y(2) = y[0] * MASA::masa_eval_exact_v<double>(x[0], x[1]);
  y(3) = MASA::masa_eval_exact_p<double>(x[0], x[1]) / (gamma - 1.);

  double k = 0.;
  for (int d = 0; d < x.Size(); d++) k += y[1 + d] * y[1 + d];
  k *= 0.5 / y[0];
  y[3] += k;
}

void initEuler2D(const int dim, RunConfiguration &config) {
  assert(dim == 2);
  assert(config.workFluid == DRY_AIR);
  assert(config.GetEquationSystem() == EULER);
  assert(config.mms_name_ == "euler_2d");

  MASA::masa_init<double>("forcing handler", "euler_2d");

  MASA::masa_set_param<double>("L", 3.02);

  MASA::masa_set_param<double>("Gamma", 1.4);

  MASA::masa_set_param<double>("a_rhox", 2.);
  MASA::masa_set_param<double>("a_rhoy", 2.);

  MASA::masa_set_param<double>("a_ux", 2.);
  MASA::masa_set_param<double>("a_uy", 2.);

  MASA::masa_set_param<double>("a_vx", 2.);
  MASA::masa_set_param<double>("a_vy", 2.);

  MASA::masa_set_param<double>("a_px", 2.);
  MASA::masa_set_param<double>("a_py", 2.);
}

void initCNS2DSutherlands(const int dim, RunConfiguration &config) {
  assert(dim == 2);
  assert(config.workFluid == DRY_AIR);
  assert(config.mms_name_ == "ad_cns_2d_sutherlands");

  const double viscMult = config.visc_mult;
  const double bulkVisc = config.bulk_visc;

  MASA::masa_init<double>("forcing handler", "ad_cns_2d_sutherlands");

  // fluid parameters
  MASA::masa_set_param<double>("L", 3.02);

  MASA::masa_set_param<double>("Gamma", 1.4);
  MASA::masa_set_param<double>("R", 287.058);
  MASA::masa_set_param<double>("Pr", 0.71);
  MASA::masa_set_param<double>("Amu", viscMult * 1.458e-6);
  MASA::masa_set_param<double>("Bmu", 1.5);
  MASA::masa_set_param<double>("Cmu", 110.4);
  MASA::masa_set_param<double>("bulkViscMult", bulkVisc);

  MASA::masa_set_param<double>("rho_0", 1.02);
  MASA::masa_set_param<double>("rho_x", 0.11);
  MASA::masa_set_param<double>("rho_y", 0.13);
  MASA::masa_set_param<double>("a_rhox", 2.);
  MASA::masa_set_param<double>("a_rhoy", 2.);

  MASA::masa_set_param<double>("a_ux", 2.);
  MASA::masa_set_param<double>("a_uy", 2.);

  MASA::masa_set_param<double>("a_vx", 2.);
  MASA::masa_set_param<double>("a_vy", 2.);

  MASA::masa_set_param<double>("a_px", 2.);
  MASA::masa_set_param<double>("a_py", 2.);
}

}  // namespace dryair2d

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
  y(0) = MASA::masa_eval_exact_rho<double>(x[0], x[1], x[2], tin);  // rho
}

void exactVelFunction(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");
  y(0) = MASA::masa_eval_exact_u<double>(x[0], x[1], x[2], tin);
  y(1) = MASA::masa_eval_exact_v<double>(x[0], x[1], x[2], tin);
  y(2) = MASA::masa_eval_exact_w<double>(x[0], x[1], x[2], tin);
}

void exactPreFunction(const Vector &x, double tin, Vector &y) {
  MFEM_ASSERT(x.Size() == 3, "");
  y(0) = MASA::masa_eval_exact_p<double>(x[0], x[1], x[2], tin);
}

void initEuler3DTransient(const int dim, RunConfiguration &config) {
  assert(dim == 3);
  assert(config.workFluid == DRY_AIR || config.workFluid == LTE_FLUID);
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

namespace ternary2d {

void initTernary2DBase(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.numSpecies == 3);
  assert(config.transportModel == CONSTANT);
  assert(config.gasModel == PERFECT_MIXTURE);

  MASA::masa_set_param<double>("u0", 1.5);
  MASA::masa_set_param<double>("dux", 0.1);
  MASA::masa_set_param<double>("duy", 0.2);

  MASA::masa_set_param<double>("kux", 1.0);
  MASA::masa_set_param<double>("kuy", 2.0);
  MASA::masa_set_param<double>("offset_ux", -0.33);
  MASA::masa_set_param<double>("offset_uy", 0.47);

  MASA::masa_set_param<double>("v0", 0.91);
  MASA::masa_set_param<double>("dvx", 0.13);
  MASA::masa_set_param<double>("dvy", 0.11);

  MASA::masa_set_param<double>("kvx", 2.0);
  MASA::masa_set_param<double>("kvy", 1.0);
  MASA::masa_set_param<double>("offset_vx", 0.11);
  MASA::masa_set_param<double>("offset_vy", 0.92);

  MASA::masa_set_param<double>("rho0", 1.2);
  MASA::masa_set_param<double>("drhox", 0.17);
  MASA::masa_set_param<double>("drhoy", 0.09);
  MASA::masa_set_param<double>("krhox", 1.0);
  MASA::masa_set_param<double>("krhoy", 1.0);
  MASA::masa_set_param<double>("offset_rhox", 0.74);
  MASA::masa_set_param<double>("offset_rhoy", 0.19);

  MASA::masa_set_param<double>("Y0", 0.34);
  MASA::masa_set_param<double>("dY0x", 0.13);
  MASA::masa_set_param<double>("dY0y", 0.07);
  MASA::masa_set_param<double>("kx0", 2.0);
  MASA::masa_set_param<double>("ky0", 1.0);
  MASA::masa_set_param<double>("offset_x0", 0.17);
  MASA::masa_set_param<double>("offset_y0", 0.58);

  MASA::masa_set_param<double>("T0", 500.0);
  MASA::masa_set_param<double>("dTx", 37.0);
  MASA::masa_set_param<double>("dTy", 29.0);
  MASA::masa_set_param<double>("kTx", 1.0);
  MASA::masa_set_param<double>("kTy", 1.0);
  MASA::masa_set_param<double>("offset_Tx", 0.71);
  MASA::masa_set_param<double>("offset_Ty", 0.29);

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

  MASA::masa_set_param<double>("mu", config.constantTransport.viscosity);
  MASA::masa_set_param<double>("muB", config.constantTransport.bulkViscosity);
  MASA::masa_set_param<double>(
      "k_heat", config.constantTransport.thermalConductivity + config.constantTransport.electronThermalConductivity);

  // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  MASA::masa_set_param<double>("D_A", config.constantTransport.diffusivity[numSpecies - 1]);
  MASA::masa_set_param<double>("D_I", config.constantTransport.diffusivity[0]);
  MASA::masa_set_param<double>("D_E", config.constantTransport.diffusivity[numSpecies - 2]);

  MASA::masa_set_param<double>("qe", ELECTRONCHARGE);
  MASA::masa_set_param<double>("kB", BOLTZMANNCONSTANT);

  MASA::masa_set_param<double>("ZI", mixture->GetGasParams(0, GasParams::SPECIES_CHARGES));
  MASA::masa_set_param<double>("ZE", mixture->GetGasParams(numSpecies - 2, GasParams::SPECIES_CHARGES));

  double Af = 0., bf = 0., Ef = 1., Ab = 1., bb = 0., Eb = 1., rE = 0.;
  if (config.numReactions > 0) {
    assert(config.numReactions == 1);
    assert(config.reactionModels[0] == ARRHENIUS);
    assert(config.detailedBalance[0]);
    assert(config.reactantStoich(0, 0) == 0);
    assert(config.reactantStoich(1, 0) == 1);
    assert(config.reactantStoich(2, 0) == 1);
    assert(config.productStoich(0, 0) == 1);
    assert(config.productStoich(1, 0) == 2);
    assert(config.productStoich(2, 0) == 0);

    // NOTE(kevin): need to read params from host.
    Af = config.rxnModelParamsHost[0](0);
    bf = config.rxnModelParamsHost[0](1);
    Ef = config.rxnModelParamsHost[0](2);
    // Af = config.chemistryInput.reactionInputs[0].modelParams[0];
    // bf = config.chemistryInput.reactionInputs[0].modelParams[1];
    // Ef = config.chemistryInput.reactionInputs[0].modelParams[2];

    Ab = (config.equilibriumConstantParams[0 + 0]);
    bb = (config.equilibriumConstantParams[1 + 0]);
    Eb = (config.equilibriumConstantParams[2 + 0]);

    rE = config.reactionEnergies[0];
  }
  MASA::masa_set_param<double>("Af", Af);
  MASA::masa_set_param<double>("bf", bf);
  MASA::masa_set_param<double>("Ef", Ef);

  MASA::masa_set_param<double>("Ab", Ab);
  MASA::masa_set_param<double>("bb", bb);
  MASA::masa_set_param<double>("Eb", Eb);

  MASA::masa_set_param<double>("rE", rE);
}

void initTernary2DPeriodic(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_periodic");
  assert(!config.ambipolar);
  assert(!config.twoTemperature);

  MASA::masa_init<double>("forcing handler", "ternary_2d_periodic");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);

  MASA::masa_set_param<double>("Y1", 0.13);
  MASA::masa_set_param<double>("dY1x", 0.03);
  MASA::masa_set_param<double>("dY1y", 0.04);
  MASA::masa_set_param<double>("kx1", 1.0);
  MASA::masa_set_param<double>("ky1", 2.0);
  MASA::masa_set_param<double>("offset_x1", 0.94);
  MASA::masa_set_param<double>("offset_y1", 0.29);
}

void initTernary2DPeriodicAmbipolar(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_periodic_ambipolar");
  assert(config.ambipolar);
  assert(!config.twoTemperature);

  MASA::masa_init<double>("forcing handler", "ternary_2d_periodic_ambipolar");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);
}

void initTernary2D2TPeriodicAmbipolar(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_2t_periodic_ambipolar");
  assert(config.ambipolar);
  assert(config.twoTemperature);

  MASA::masa_init<double>("forcing handler", "ternary_2d_2t_periodic_ambipolar");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);

  MASA::masa_set_param<double>("TE0", 700.0);
  MASA::masa_set_param<double>("dTEx", 49.3);
  MASA::masa_set_param<double>("dTEy", 23.1);
  MASA::masa_set_param<double>("kTEx", 2.0);
  MASA::masa_set_param<double>("kTEy", 1.0);
  MASA::masa_set_param<double>("offset_TEx", 0.31);
  MASA::masa_set_param<double>("offset_TEy", 0.91);

  MASA::masa_set_param<double>("k_heat", config.constantTransport.thermalConductivity);
  MASA::masa_set_param<double>("k_E", config.constantTransport.electronThermalConductivity);

  const int numSpecies = mixture->GetNumSpecies();
  // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  MASA::masa_set_param<double>("nu_I", config.constantTransport.mtFreq[0]);
  MASA::masa_set_param<double>("nu_A", config.constantTransport.mtFreq[numSpecies - 1]);
}

void initTernary2D2TAmbipolarWall(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_2t_ambipolar_wall");
  assert(config.ambipolar);
  assert(config.twoTemperature);

  MASA::masa_init<double>("forcing handler", "ternary_2d_2t_ambipolar_wall");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);

  MASA::masa_set_param<double>("dTEx", 49.3);
  MASA::masa_set_param<double>("dTEy", 23.1);
  MASA::masa_set_param<double>("kTEx", 2.0);
  MASA::masa_set_param<double>("kTEy", 1.0);
  MASA::masa_set_param<double>("offset_TEx", 0.31);
  MASA::masa_set_param<double>("offset_TEy", 0.91);

  MASA::masa_set_param<double>("k_heat", config.constantTransport.thermalConductivity);
  MASA::masa_set_param<double>("k_E", config.constantTransport.electronThermalConductivity);

  const int numSpecies = mixture->GetNumSpecies();
  // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  MASA::masa_set_param<double>("nu_I", config.constantTransport.mtFreq[0]);
  MASA::masa_set_param<double>("nu_A", config.constantTransport.mtFreq[numSpecies - 1]);

  assert(config.wallBC.size() == 2);
  double T0 = -1.0;
  for (int w = 0; w < 2; w++) {
    if (config.wallBC[w].Th > 0.0) {
      T0 = config.wallBC[w].Th;
      break;
    }
  }
  assert(T0 > 0.0);
  MASA::masa_set_param<double>("T0", T0);
  MASA::masa_set_param<double>("TE0", T0);

  MASA::masa_set_param<double>("n0", 40.0);
  MASA::masa_set_param<double>("dnx", 5.7);
  MASA::masa_set_param<double>("dny", 8.9);
  MASA::masa_set_param<double>("knx", 2.0);
  MASA::masa_set_param<double>("kny", 1.0);
  MASA::masa_set_param<double>("offset_nx", 0.29);
  MASA::masa_set_param<double>("offset_ny", 0.87);

  MASA::masa_set_param<double>("X0", 0.21);
  MASA::masa_set_param<double>("dX0x", 0.08);
  MASA::masa_set_param<double>("dX0y", 0.045);
}

void initTernary2D2TAmbipolarInoutlet(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_2t_ambipolar_inoutlet");
  assert(config.ambipolar);
  assert(config.twoTemperature);
  const int dim = 2;

  MASA::masa_init<double>("forcing handler", "ternary_2d_2t_ambipolar_inoutlet");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);

  MASA::masa_set_param<double>("dTEx", 49.3);
  MASA::masa_set_param<double>("dTEy", 23.1);
  MASA::masa_set_param<double>("kTEx", 2.0);
  MASA::masa_set_param<double>("kTEy", 1.0);
  MASA::masa_set_param<double>("offset_TEx", 0.31);
  MASA::masa_set_param<double>("offset_TEy", 0.91);

  MASA::masa_set_param<double>("k_heat", config.constantTransport.thermalConductivity);
  MASA::masa_set_param<double>("k_E", config.constantTransport.electronThermalConductivity);

  const int numSpecies = mixture->GetNumSpecies();
  // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  MASA::masa_set_param<double>("nu_I", config.constantTransport.mtFreq[0]);
  MASA::masa_set_param<double>("nu_A", config.constantTransport.mtFreq[numSpecies - 1]);

  assert(config.GetOutletPatchType()->size() == 1);
  Array<double> p0 = config.GetOutletData(0);
  assert(config.GetInletPatchType()->size() == 1);
  Array<double> inlet0 = config.GetInletData(0);
  double rho0 = inlet0[0];
  double u0 = inlet0[1];
  double v0 = inlet0[2];
  // int ionIndex = (*mixtureToInputMap)[0];
  // double YI = inlet0[4 + ionIndex];
  double YI = inlet0[4 + 0];
  assert(p0[0] > 0.0);
  assert(rho0 > 0.0);
  assert(YI > 0.0);

  Vector conservedState(mixture->GetNumEquations());
  conservedState = 0.0;
  // NOTE: fill only necessary information for computing X, Y, n.
  conservedState(0) = rho0;
  conservedState(dim + 2) = rho0 * YI;
  Vector X_sp(3), Y_sp(3), n_sp(3);
  mixture->computeSpeciesPrimitives(conservedState, X_sp, Y_sp, n_sp);
  double nTotal = 0.0;
  for (int sp = 0; sp < 3; sp++) nTotal += n_sp(sp);

  MASA::masa_set_param<double>("u0", u0);
  MASA::masa_set_param<double>("v0", v0);

  MASA::masa_set_param<double>("n0", nTotal);
  MASA::masa_set_param<double>("dnx", 5.7);
  MASA::masa_set_param<double>("dny", 8.9);
  MASA::masa_set_param<double>("knx", 2.0);
  MASA::masa_set_param<double>("kny", 2.0);
  MASA::masa_set_param<double>("offset_ny", 0.87);

  MASA::masa_set_param<double>("X0", X_sp(0));
  MASA::masa_set_param<double>("dX0x", 0.24 * X_sp(0));
  MASA::masa_set_param<double>("dX0y", 0.37 * X_sp(0));

  MASA::masa_set_param<double>("pLx", p0[0]);
  MASA::masa_set_param<double>("dpx", 0.11 * p0[0]);
  MASA::masa_set_param<double>("dpy", 0.07 * p0[0]);
  MASA::masa_set_param<double>("kpx", 2.0);
  MASA::masa_set_param<double>("kpy", 1.0);
  MASA::masa_set_param<double>("offset_py", 0.87);
}

void initTernary2DSheath(GasMixture *mixture, RunConfiguration &config, const double Lx, const double Ly) {
  assert(config.mms_name_ == "ternary_2d_sheath");
  assert(config.ambipolar);
  assert(config.twoTemperature);

  MASA::masa_init<double>("forcing handler", "ternary_2d_sheath");
  ternary2d::initTernary2DBase(mixture, config, Lx, Ly);

  MASA::masa_set_param<double>("k_heat", config.constantTransport.thermalConductivity);
  MASA::masa_set_param<double>("k_E", config.constantTransport.electronThermalConductivity);

  const int numSpecies = mixture->GetNumSpecies();
  // std::map<int, int> *mixtureToInputMap = mixture->getMixtureToInputMap();
  MASA::masa_set_param<double>("nu_I", config.constantTransport.mtFreq[0]);
  MASA::masa_set_param<double>("nu_A", config.constantTransport.mtFreq[numSpecies - 1]);

  MASA::masa_set_param<double>("Te0", 5.0e-1);
  MASA::masa_set_param<double>("Th0", 10.0e-1);
  MASA::masa_set_param<double>("XI0", 0.11);

  MASA::masa_set_param<double>("TeL", 4.0e-1);
  MASA::masa_set_param<double>("XIL", 0.13);

  assert(config.wallBC.size() == 2);
  double T0 = -1.0;
  for (int w = 0; w < 2; w++) {
    if (config.wallBC[w].Th > 0.0) {
      T0 = config.wallBC[w].Th;
      break;
    }
  }
  assert(T0 > 0.0);
  MASA::masa_set_param<double>("ThL", T0);
  MASA::masa_set_param<double>("dTx", 3.7e-2);
  // MASA::masa_set_param<double>("dTx", 0.0);

  MASA::masa_set_param<double>("n0", 40.0e-1);
  MASA::masa_set_param<double>("dnx", 5.7e-1);
  MASA::masa_set_param<double>("dny", 8.9e-1);
  // MASA::masa_set_param<double>("dnx", 0.0);
  // MASA::masa_set_param<double>("dny", 0.0);
  MASA::masa_set_param<double>("knx", 2.0);
  MASA::masa_set_param<double>("kny", 1.0);
  MASA::masa_set_param<double>("offset_nx", 0.29);
  MASA::masa_set_param<double>("offset_ny", 0.87);

  // MASA::masa_set_param<double>("u0", 0.0);
  // MASA::masa_set_param<double>("dux", 0.0);
  // MASA::masa_set_param<double>("duy", 0.0);
  //
  // MASA::masa_set_param<double>("v0", 0.0);
  // MASA::masa_set_param<double>("dvx", 0.0);
  // MASA::masa_set_param<double>("dvy", 0.0);
}

}  // namespace ternary2d
#endif
