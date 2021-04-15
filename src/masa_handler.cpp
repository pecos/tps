#ifdef _MASA_
#include "masa_handler.hpp"

void initMasaHandler(std::string name, int dim)
{
  // Initialize MASA
  if( dim==2 )
  {
    MASA::masa_init<double>(name,"navierstokes_2d_compressible");
  }else if( dim==3 )
  {
//     MASA::masa_init<double>(name,"navierstokes_3d_compressible");
    MASA::masa_init<double>("forcing handler","euler_3d");
//     MASA::masa_init<double>("forcing handler","navierstokes_3d_transient_sutherland");
  }
  
  MASA::masa_set_param<double>("L",2);
  //MASA::masa_set_param<double>("mu",1e-4);
  MASA::masa_set_param<double>("mu",0);
  MASA::masa_set_param<double>("k",0);
//   MASA::masa_set_param<double>("Pr",0.71);
  
  MASA::masa_set_param<double>("rho_0",1.);
  MASA::masa_set_param<double>("rho_x",0.1);
  MASA::masa_set_param<double>("rho_y",0.1);
  MASA::masa_set_param<double>("rho_z",0);
  MASA::masa_set_param<double>("u_0",70.);
  MASA::masa_set_param<double>("u_x",0.1);
  MASA::masa_set_param<double>("u_y",0.1);
  MASA::masa_set_param<double>("u_z",0);
  MASA::masa_set_param<double>("v_0",0);
  MASA::masa_set_param<double>("v_x",0.1);
  MASA::masa_set_param<double>("v_y",0.1);
  MASA::masa_set_param<double>("v_z",0);
  MASA::masa_set_param<double>("w_0",0);
  MASA::masa_set_param<double>("w_x",0.1);
  MASA::masa_set_param<double>("w_y",0.1);
  MASA::masa_set_param<double>("w_z",0);
  MASA::masa_set_param<double>("p_0",101300);
  MASA::masa_set_param<double>("p_x",101);
  MASA::masa_set_param<double>("p_y",101);
  MASA::masa_set_param<double>("p_z",0);
  
  MASA::masa_set_param<double>("a_rhox",1.*M_PI/2);
  MASA::masa_set_param<double>("a_rhoy",2.*M_PI/2);
  MASA::masa_set_param<double>("a_rhoz",0.*M_PI/2);
  MASA::masa_set_param<double>("a_ux",1.*M_PI/2);
  MASA::masa_set_param<double>("a_uy",1.*M_PI/2);
  MASA::masa_set_param<double>("a_uz",0.*M_PI/2);
  MASA::masa_set_param<double>("a_vx",1.*M_PI/2);
  MASA::masa_set_param<double>("a_vy",1.*M_PI/2);
  MASA::masa_set_param<double>("a_vz",0.*M_PI/2);
  MASA::masa_set_param<double>("a_wx",1.*M_PI/2);
  MASA::masa_set_param<double>("a_wy",1.*M_PI/2);
  MASA::masa_set_param<double>("a_wz",0.*M_PI/2);
  MASA::masa_set_param<double>("a_px",1.*M_PI/2);
  MASA::masa_set_param<double>("a_py",2.*M_PI/2);
  MASA::masa_set_param<double>("a_pz",0.*M_PI/2);
  MASA::masa_display_param<double>();
}
#endif 
