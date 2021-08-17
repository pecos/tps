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
                              RunConfiguration& _config):
ForcingTerms(_dim,
             _num_equation,
             _order,
             _intRuleType,
             _intRules,
             _vfes,
             _Up,
             _gradUp )
{
  initMasaHandler("forcing",dim, _config.GetEquationSystem(),_config.GetViscMult());
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
      
//       for(int i=0;i<3;i++) transip[i] += 1.;
//       cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<endl;

      Array<double> ip_forcing(num_equation);

      // MASA forcing
      // ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2]); // rho
      // ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2]); // rho*u
      // ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2]); // rho*v
      // ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2]); // rho*w
      // ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2]); // rhp*e

      // ip_forcing[0] = MASA::masa_eval_source_rho<double>  (x[0],x[1],x[2],time); // rho
      // ip_forcing[1] = MASA::masa_eval_source_rho_u<double>(x[0],x[1],x[2],time); // rho*u
      // ip_forcing[2] = MASA::masa_eval_source_rho_v<double>(x[0],x[1],x[2],time); // rho*v
      // ip_forcing[3] = MASA::masa_eval_source_rho_w<double>(x[0],x[1],x[2],time); // rho*w
      // ip_forcing[4] = MASA::masa_eval_source_rho_e<double>(x[0],x[1],x[2],time); // rhp*e

      ip_forcing[0] = MASA::masa_eval_source_rho<double>(x[0],x[1],x[2],time); // rho
      ip_forcing[1] = MASA::masa_eval_source_u<double>  (x[0],x[1],x[2],time); // rho*u
      ip_forcing[2] = MASA::masa_eval_source_v<double>  (x[0],x[1],x[2],time); // rho*v
      ip_forcing[3] = MASA::masa_eval_source_w<double>  (x[0],x[1],x[2],time); // rho*w
      ip_forcing[4] = MASA::masa_eval_source_e<double>  (x[0],x[1],x[2],time); // rhp*e

      for (int j= 0;j<dof_elem;j++)
      {
        int i = nodes[j];
        for(int eq=0;eq<num_equation;eq++) data[i +eq*dof] += ip_forcing[eq]*shape[j];
      }

    }
  }
}
#endif // _MASA_

