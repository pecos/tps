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
#include "masa_handler.hpp"

// void initMasaHandler(std::string name, int dim, const Equations& eqn, const double& viscMult,
//                      const std::string mms_name) {
//   // Initialize MASA
//   if (dim == 2) {
//     assert(mms_name == "navierstokes_2d_compressible");
//     MASA::masa_init<double>(name, "navierstokes_2d_compressible");
//   } else if (dim == 3) {
//     switch (eqn) {
//       // 3-D euler equations
//       case EULER:
//         assert(mms_name == "euler_transient_3d");
//         MASA::masa_init<double>("forcing handler", "euler_transient_3d");
//
//         // fluid parameters
//         MASA::masa_set_param<double>("Gamma", 1.4);
//
//         // solution parameters
//         MASA::masa_set_param<double>("L", 2);
//
//         MASA::masa_set_param<double>("rho_0", 1.0);
//         MASA::masa_set_param<double>("rho_x", 0.1);
//         MASA::masa_set_param<double>("rho_y", 0.1);
//         MASA::masa_set_param<double>("rho_z", 0.0);
//         MASA::masa_set_param<double>("rho_t", 0.15);
//
//         MASA::masa_set_param<double>("u_0", 130.0);
//         MASA::masa_set_param<double>("u_x", 10.0);
//         MASA::masa_set_param<double>("u_y", 5.0);
//         MASA::masa_set_param<double>("u_z", 0.0);
//         MASA::masa_set_param<double>("u_t", 10.0);
//
//         MASA::masa_set_param<double>("v_0", 5.0);
//         MASA::masa_set_param<double>("v_x", 1.0);
//         MASA::masa_set_param<double>("v_y", -1.0);
//         MASA::masa_set_param<double>("v_z", 0.0);
//         MASA::masa_set_param<double>("v_t", 2.0);
//
//         MASA::masa_set_param<double>("w_0", 0.0);
//         MASA::masa_set_param<double>("w_x", 2.0);
//         MASA::masa_set_param<double>("w_y", 1.0);
//         MASA::masa_set_param<double>("w_z", 0.0);
//         MASA::masa_set_param<double>("w_t", -1.0);
//
//         MASA::masa_set_param<double>("p_0", 101300.0);
//         MASA::masa_set_param<double>("p_x", 101.0);
//         MASA::masa_set_param<double>("p_y", 101.0);
//         MASA::masa_set_param<double>("p_z", 0.0);
//         MASA::masa_set_param<double>("p_t", 1013.0);
//
//         MASA::masa_set_param<double>("a_rhox", 2.);
//         MASA::masa_set_param<double>("a_rhoy", 2.);
//         MASA::masa_set_param<double>("a_rhoz", 0.);
//         MASA::masa_set_param<double>("a_rhot", 400.);
//
//         MASA::masa_set_param<double>("a_ux", 2.);
//         MASA::masa_set_param<double>("a_uy", 2.);
//         MASA::masa_set_param<double>("a_uz", 0.);
//         MASA::masa_set_param<double>("a_ut", 400.);
//
//         MASA::masa_set_param<double>("a_vx", 2.);
//         MASA::masa_set_param<double>("a_vy", 2.);
//         MASA::masa_set_param<double>("a_vz", 0.);
//         MASA::masa_set_param<double>("a_vt", 400.);
//
//         MASA::masa_set_param<double>("a_wx", 2.);
//         MASA::masa_set_param<double>("a_wy", 2.);
//         MASA::masa_set_param<double>("a_wz", 0.);
//         MASA::masa_set_param<double>("a_wt", 0.);
//
//         MASA::masa_set_param<double>("a_px", 2.);
//         MASA::masa_set_param<double>("a_py", 2.);
//         MASA::masa_set_param<double>("a_pz", 0.);
//         MASA::masa_set_param<double>("a_pt", 400.);
//
//         break;
//
//       // 3-D navier-stokes equations
//       case NS:
//         assert(mms_name == "navierstokes_3d_transient_sutherland");
//         MASA::masa_init<double>("forcing handler", "navierstokes_3d_transient_sutherland");
//
//         // fluid parameters
//         MASA::masa_set_param<double>("Gamma", 1.4);
//         MASA::masa_set_param<double>("R", 287.058);
//         MASA::masa_set_param<double>("Pr", 0.71);
//         MASA::masa_set_param<double>("B_mu", 110.4);
//         MASA::masa_set_param<double>("A_mu", viscMult * 1.458e-6);
//
//         // soln parameters
//         MASA::masa_set_param<double>("L", 2);
//         MASA::masa_set_param<double>("Lt", 2);
//
//         MASA::masa_set_param<double>("rho_0", 1.0);
//         MASA::masa_set_param<double>("rho_x", 0.1);
//         MASA::masa_set_param<double>("rho_y", 0.1);
//         MASA::masa_set_param<double>("rho_z", 0.0);
//         MASA::masa_set_param<double>("rho_t", 0.15);
//
//         MASA::masa_set_param<double>("u_0", 130.0);
//         MASA::masa_set_param<double>("u_x", 10.0);
//         MASA::masa_set_param<double>("u_y", 5.0);
//         MASA::masa_set_param<double>("u_z", 0.0);
//         MASA::masa_set_param<double>("u_t", 10.0);
//
//         MASA::masa_set_param<double>("v_0", 5.0);
//         MASA::masa_set_param<double>("v_x", 1.0);
//         MASA::masa_set_param<double>("v_y", -1.0);
//         MASA::masa_set_param<double>("v_z", 0.0);
//         MASA::masa_set_param<double>("v_t", 2.0);
//
//         MASA::masa_set_param<double>("w_0", 0.0);
//         MASA::masa_set_param<double>("w_x", 2.0);
//         MASA::masa_set_param<double>("w_y", 1.0);
//         MASA::masa_set_param<double>("w_z", 0.0);
//         MASA::masa_set_param<double>("w_t", -1.0);
//
//         MASA::masa_set_param<double>("p_0", 101300.0);
//         MASA::masa_set_param<double>("p_x", 101.0);
//         MASA::masa_set_param<double>("p_y", 101.0);
//         MASA::masa_set_param<double>("p_z", 0.0);
//         MASA::masa_set_param<double>("p_t", 1013.0);
//
//         MASA::masa_set_param<double>("a_rhox", 2.);
//         MASA::masa_set_param<double>("a_rhoy", 2.);
//         MASA::masa_set_param<double>("a_rhoz", 0.);
//         MASA::masa_set_param<double>("a_rhot", 400.);
//
//         MASA::masa_set_param<double>("a_ux", 2.);
//         MASA::masa_set_param<double>("a_uy", 2.);
//         MASA::masa_set_param<double>("a_uz", 0.);
//         MASA::masa_set_param<double>("a_ut", 400.);
//
//         MASA::masa_set_param<double>("a_vx", 2.);
//         MASA::masa_set_param<double>("a_vy", 2.);
//         MASA::masa_set_param<double>("a_vz", 0.);
//         MASA::masa_set_param<double>("a_vt", 400.);
//
//         MASA::masa_set_param<double>("a_wx", 2.);
//         MASA::masa_set_param<double>("a_wy", 2.);
//         MASA::masa_set_param<double>("a_wz", 0.);
//         MASA::masa_set_param<double>("a_wt", 0.);
//
//         MASA::masa_set_param<double>("a_px", 2.);
//         MASA::masa_set_param<double>("a_py", 2.);
//         MASA::masa_set_param<double>("a_pz", 0.);
//         MASA::masa_set_param<double>("a_pt", 400.);
//
//         break;
//
//       // other equations: we don't support anything else right now!
//       default:
//         std::cout << "*** WARNING: No MASA solution for equation set number " << eqn << "!  MASA is uninitialized. ***"
//                   << std::endl;
//         break;
//     }  // end switch
//   }

MasaHandler::MasaHandler(const int dim, RunConfiguration& config)
    : config_(&config) {
  const Equations eqn = config.GetEquationSystem();
  const double viscMult = config.visc_mult;
  const std::string mms_name = config.mms_name_;

  // Initialize MASA
  switch (config.workFluid) {
    case DRY_AIR:
      if (dim == 2) {
        assert(mms_name == "navierstokes_2d_compressible");
        MASA::masa_init<double>("exact", "navierstokes_2d_compressible");
        MASA::masa_init<double>("forcing", "navierstokes_2d_compressible");
      } else if (dim == 3) {
        switch (eqn) {
          // 3-D euler equations
          case EULER:
            assert(mms_name == "euler_transient_3d");
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

            break;

          // 3-D navier-stokes equations
          case NS:
            assert(mms_name == "navierstokes_3d_transient_sutherland");
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

            break;

          // other equations: we don't support anything else right now!
          default:
            std::cout << "*** WARNING: No MASA solution for equation set number " << eqn
                      << "!  MASA is uninitialized. ***" << std::endl;
            break;
        }  // end dry_air 3d switch
      }
      break;
    case USER_DEFINED:
      break;
  }  // end WorkingFluid switch

  // check that masa at least thinks things are okie-dokie
  int ierr = MASA::masa_sanity_check<double>();
  if (ierr != 0) {
    std::cout << "*** WARNING: MASA sanity check returned error = " << ierr << " ***" << std::endl;
    std::cout << "Current parameters are as follows" << std::endl;
    MASA::masa_display_param<double>();
  }
}
#endif
