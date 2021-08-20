#ifdef _MASA_
#include "masa_handler.hpp"

void initMasaHandler(std::string name, 
                     int dim, 
                     const Equations& eqn,
                     const double &viscMult )
{
  // Initialize MASA
  if( dim==2 )
  {
    MASA::masa_init<double>(name,"navierstokes_2d_compressible");
  }else if( dim==3 )
  {
    switch (eqn) {

    // 3-D euler equations
    case EULER:
      MASA::masa_init<double>("forcing handler","euler_transient_3d");

      // fluid parameters
      MASA::masa_set_param<double>("Gamma",1.4);

      // solution parameters
      MASA::masa_set_param<double>("L",2);

      MASA::masa_set_param<double>("rho_0",1.0 );
      MASA::masa_set_param<double>("rho_x",0.1 );
      MASA::masa_set_param<double>("rho_y",0.1 );
      MASA::masa_set_param<double>("rho_z",0.0 );
      MASA::masa_set_param<double>("rho_t",0.15);

      MASA::masa_set_param<double>("u_0",130.0);
      MASA::masa_set_param<double>("u_x", 10.0);
      MASA::masa_set_param<double>("u_y",  5.0);
      MASA::masa_set_param<double>("u_z",  0.0);
      MASA::masa_set_param<double>("u_t", 10.0);

      MASA::masa_set_param<double>("v_0",  5.0);
      MASA::masa_set_param<double>("v_x",  1.0);
      MASA::masa_set_param<double>("v_y", -1.0);
      MASA::masa_set_param<double>("v_z",  0.0);
      MASA::masa_set_param<double>("v_t",  2.0);

      MASA::masa_set_param<double>("w_0",  0.0);
      MASA::masa_set_param<double>("w_x",  2.0);
      MASA::masa_set_param<double>("w_y",  1.0);
      MASA::masa_set_param<double>("w_z",  0.0);
      MASA::masa_set_param<double>("w_t", -1.0);

      MASA::masa_set_param<double>("p_0",101300.0);
      MASA::masa_set_param<double>("p_x",   101.0);
      MASA::masa_set_param<double>("p_y",   101.0);
      MASA::masa_set_param<double>("p_z",     0.0);
      MASA::masa_set_param<double>("p_t",  1013.0);

      MASA::masa_set_param<double>("a_rhox",  2.);
      MASA::masa_set_param<double>("a_rhoy",  2.);
      MASA::masa_set_param<double>("a_rhoz",  0.);
      MASA::masa_set_param<double>("a_rhot",400.);

      MASA::masa_set_param<double>("a_ux",  2.);
      MASA::masa_set_param<double>("a_uy",  2.);
      MASA::masa_set_param<double>("a_uz",  0.);
      MASA::masa_set_param<double>("a_ut",400.);

      MASA::masa_set_param<double>("a_vx",  2.);
      MASA::masa_set_param<double>("a_vy",  2.);
      MASA::masa_set_param<double>("a_vz",  0.);
      MASA::masa_set_param<double>("a_vt",400.);

      MASA::masa_set_param<double>("a_wx",  2.);
      MASA::masa_set_param<double>("a_wy",  2.);
      MASA::masa_set_param<double>("a_wz",  0.);
      MASA::masa_set_param<double>("a_wt",  0.);

      MASA::masa_set_param<double>("a_px",  2.);
      MASA::masa_set_param<double>("a_py",  2.);
      MASA::masa_set_param<double>("a_pz",  0.);
      MASA::masa_set_param<double>("a_pt",400.);

      break;

    // 3-D navier-stokes equations
    case NS:
      MASA::masa_init<double>("forcing handler","navierstokes_3d_transient_sutherland");

      // fluid parameters
      MASA::masa_set_param<double>("Gamma",1.4);
      MASA::masa_set_param<double>("R",287.058);
      MASA::masa_set_param<double>("Pr",0.71);
      MASA::masa_set_param<double>("B_mu",110.4);
      MASA::masa_set_param<double>("A_mu",viscMult*1.458e-6);

      // soln parameters
      MASA::masa_set_param<double>("L",2);
      MASA::masa_set_param<double>("Lt",2);

      MASA::masa_set_param<double>("rho_0",1.0 );
      MASA::masa_set_param<double>("rho_x",0.1 );
      MASA::masa_set_param<double>("rho_y",0.1 );
      MASA::masa_set_param<double>("rho_z",0.0 );
      MASA::masa_set_param<double>("rho_t",0.15);

      MASA::masa_set_param<double>("u_0",130.0);
      MASA::masa_set_param<double>("u_x", 10.0);
      MASA::masa_set_param<double>("u_y",  5.0);
      MASA::masa_set_param<double>("u_z",  0.0);
      MASA::masa_set_param<double>("u_t", 10.0);

      MASA::masa_set_param<double>("v_0",  5.0);
      MASA::masa_set_param<double>("v_x",  1.0);
      MASA::masa_set_param<double>("v_y", -1.0);
      MASA::masa_set_param<double>("v_z",  0.0);
      MASA::masa_set_param<double>("v_t",  2.0);

      MASA::masa_set_param<double>("w_0",  0.0);
      MASA::masa_set_param<double>("w_x",  2.0);
      MASA::masa_set_param<double>("w_y",  1.0);
      MASA::masa_set_param<double>("w_z",  0.0);
      MASA::masa_set_param<double>("w_t", -1.0);

      MASA::masa_set_param<double>("p_0",101300.0);
      MASA::masa_set_param<double>("p_x",   101.0);
      MASA::masa_set_param<double>("p_y",   101.0);
      MASA::masa_set_param<double>("p_z",     0.0);
      MASA::masa_set_param<double>("p_t",  1013.0);

      MASA::masa_set_param<double>("a_rhox",  2.);
      MASA::masa_set_param<double>("a_rhoy",  2.);
      MASA::masa_set_param<double>("a_rhoz",  0.);
      MASA::masa_set_param<double>("a_rhot",400.);

      MASA::masa_set_param<double>("a_ux",  2.);
      MASA::masa_set_param<double>("a_uy",  2.);
      MASA::masa_set_param<double>("a_uz",  0.);
      MASA::masa_set_param<double>("a_ut",400.);

      MASA::masa_set_param<double>("a_vx",  2.);
      MASA::masa_set_param<double>("a_vy",  2.);
      MASA::masa_set_param<double>("a_vz",  0.);
      MASA::masa_set_param<double>("a_vt",400.);

      MASA::masa_set_param<double>("a_wx",  2.);
      MASA::masa_set_param<double>("a_wy",  2.);
      MASA::masa_set_param<double>("a_wz",  0.);
      MASA::masa_set_param<double>("a_wt",  0.);

      MASA::masa_set_param<double>("a_px",  2.);
      MASA::masa_set_param<double>("a_py",  2.);
      MASA::masa_set_param<double>("a_pz",  0.);
      MASA::masa_set_param<double>("a_pt",400.);

      break;

    // other equations: we don't support anything else right now!
    default:
      std::cout << "*** WARNING: No MASA solution for equation set number "
                << eqn << "!  MASA is uninitialized. ***" << std::endl;
      break;

    } // end switch

  }


  // check that masa at least thinks things are okie-dokie
  int ierr = MASA::masa_sanity_check<double>();
  if (ierr!=0) {
    std::cout << "*** WARNING: MASA sanity check returned error = "
              << ierr << " ***" << std::endl;
    std::cout << "Current parameters are as follows" << std::endl;
    MASA::masa_display_param<double>();
  }
}
#endif
