#include "forcing_terms.hpp"

ForcingTerms::ForcingTerms( const int &_dim,
                            const int &_num_equation,
                            const int &_order,
                            const int &_intRuleType,
                            IntegrationRules *_intRules,
                            ParFiniteElementSpace *_vfes,
                            ParGridFunction *_Up,
                            ParGridFunction *_gradUp ):
dim(_dim),
num_equation(_num_equation),
order(_order),
intRuleType(_intRuleType),
intRules(_intRules),
vfes(_vfes),
Up(_Up),
gradUp(_gradUp)
{
  b = new ParGridFunction(vfes);
  
  // Initialize to zero
  int dof = vfes->GetNDofs();
  double *data = b->GetData();
  for(int ii=0;ii<dof*num_equation;ii++) data[ii] = 0.;
}

ForcingTerms::~ForcingTerms()
{
  delete b;
}

void ForcingTerms::addForcingIntegrals(Vector &in)
{
  double *dataB = b->GetData();
  for(int i=0; i<in.Size(); i++)
  {
    in[i] = in[i] + dataB[i];
  }
}

ConstantPressureGradient::ConstantPressureGradient(const int& _dim, 
                                                   const int& _num_equation, 
                                                   const int& _order, 
                                                   const int& _intRuleType,
                                                   IntegrationRules* _intRules, 
                                                   ParFiniteElementSpace* _vfes,
                                                   ParGridFunction *_Up,
                                                   ParGridFunction *_gradUp,
                                                   RunConfiguration& _config ):
ForcingTerms(_dim,
             _num_equation,
             _order,
             _intRuleType,
             _intRules,
             _vfes,
             _Up,
             _gradUp )
{
  {
    double *data = _config.GetImposedPressureGradient();
    for(int jj=0;jj<3;jj++) pressGrad[jj] = data[jj];
  }
  
  updateTerms();
}

void ConstantPressureGradient::updateTerms()
{
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();
  
  double *data = b->GetData();
  const double *dataUp = Up->GetData();
  const double *dataGradUp = gradUp->GetData();
  
  for(int el=0; el<numElem; el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();
    
    // nodes of the element 
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);
    
    // set to 0
    for(int n=0;n<dof_elem;n++)
    {
      int i = nodes[n];
      for(int eq=1;eq<num_equation;eq++) data[i+eq*dof] = 0.;
    }
    
    
    const int order = elem->GetOrder();
    //cout<<"order :"<<maxorder<<" dof: "<<dof_elem<<endl;
    int intorder = 2*order;
    //if(intRuleType==1 && trial_fe.GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
    for (int k= 0; k< ir->GetNPoints(); k++)
    {
      const IntegrationPoint &ip = ir->IntPoint(k);
      
      Vector shape(dof_elem);
      elem->CalcShape(ip, shape);

      Tr->SetIntPoint(&ip);
      shape *= Tr->Jacobian().Det()*ip.weight;

      for (int j= 0;j<dof_elem;j++)
      {
        int i = nodes[j];
        Vector vel(dim);
        double p = dataUp[i+(num_equation-1)*dof];
        double grad_pV = 0.;
        
        for(int d=0;d<dim;d++)
        {
          vel[d] = dataUp[i+(d+1)*dof];
          data[i +(d+1)*dof] -= pressGrad[d]*shape[j];
          grad_pV -= vel[d]*pressGrad[d];
          grad_pV -= p*dataGradUp[i+(d+1)*dof + d*dof*num_equation];
        }
        
        data[i +(num_equation-1)*dof] += grad_pV*shape[j];
      }
      
    }
  }
}


#ifdef _MASA_
MASA_forcings::MASA_forcings( const int& _dim, 
                              const int& _num_equation, 
                              const int& _order, 
                              const int& _intRuleType,
                              IntegrationRules* _intRules, 
                              ParFiniteElementSpace* _vfes,
                              ParGridFunction *_Up,
                              ParGridFunction *_gradUp,
                              double &_time):
ForcingTerms(_dim,
             _num_equation,
             _order,
             _intRuleType,
             _intRules,
             _vfes,
             _Up,
             _gradUp ),
time(_time)
{
  // Initialize MASA
  if( dim==2 )
  {
    MASA::masa_init<double>("forcing handler","navierstokes_2d_compressible");
  }else if( dim==3 )
  {
//     MASA::masa_init<double>("forcing handler","navierstokes_3d_compressible");
//     MASA::masa_init<double>("forcing handler","euler_3d");
    MASA::masa_init<double>("forcing handler","navierstokes_3d_transient_sutherland");
  }
  
  MASA::masa_set_param<double>("L",2);
  MASA::masa_set_param<double>("mu",1e-4);
  MASA::masa_set_param<double>("k",0);
  MASA::masa_set_param<double>("Pr",0.71);
  
  MASA::masa_set_param<double>("rho_x",1.01);
  MASA::masa_set_param<double>("rho_y",1.01);
  MASA::masa_set_param<double>("rho_z",0);
  MASA::masa_set_param<double>("u_x",1.01);
  MASA::masa_set_param<double>("u_y",1.01);
  MASA::masa_set_param<double>("u_z",0);
  MASA::masa_set_param<double>("v_x",1.01);
  MASA::masa_set_param<double>("v_y",1.01);
  MASA::masa_set_param<double>("v_z",0);
  MASA::masa_set_param<double>("w_x",1.01);
  MASA::masa_set_param<double>("w_y",1.01);
  MASA::masa_set_param<double>("w_z",0);
  MASA::masa_set_param<double>("p_x",1.01);
  MASA::masa_set_param<double>("p_y",1.01);
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

void MASA_forcings::updateTerms()
{
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();
  
  double *data = b->GetData();
  //const double *dataUp = Up->GetData();
  
  for(int el=0; el<numElem; el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();
    
    // nodes of the element 
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);
    
    // set to 0
    for(int n=0;n<dof_elem;n++)
    {
      int i = nodes[n];
      for(int eq=0;eq<num_equation;eq++) data[i+eq*dof] = 0.;
    }
    
    
    const int order = elem->GetOrder();
    //cout<<"order :"<<maxorder<<" dof: "<<dof_elem<<endl;
    int intorder = 2*order;
    //if(intRuleType==1 && trial_fe.GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
    for (int k= 0; k< ir->GetNPoints(); k++)
    {
      const IntegrationPoint &ip = ir->IntPoint(k);
      
      Vector shape(dof_elem);
      elem->CalcShape(ip, shape);

      Tr->SetIntPoint(&ip);
      shape *= Tr->Jacobian().Det()*ip.weight;
      
      // get coordinates of integration point
      double x[3];
      Vector transip(x, 3);
      Tr->Transform(ip,transip);
      
      for(int i=0;i<3;i++) transip[i] += 1.;
//       cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<endl;

      Array<double> ip_forcing(num_equation);

      // MASA forcing
//       ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2]); // rho
//       ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2]); // rho*u
//       ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2]); // rho*v
//       ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2]); // rho*w
//       ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2]); // rhp*e
      ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2],time); // rho
      ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2],time); // rho*u
      ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2],time); // rho*v
      ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2],time); // rho*w
      ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2],time); // rhp*e

      for (int j= 0;j<dof_elem;j++)
      {
        int i = nodes[j];
        
        for(int eq=0;eq<num_equation;eq++) data[i +eq*dof] += ip_forcing[eq]*shape[j];
      }
      
    }
  }
}
#endif // _MASA_

