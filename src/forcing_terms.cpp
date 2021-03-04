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
          data[i +(d+1)*dof] += pressGrad[d]*shape[j];
          grad_pV += vel[d]*pressGrad[d];
          grad_pV += p*dataGradUp[i+(d+1)*dof + d*dof*num_equation];
        }
        
        data[i +(num_equation-1)*dof] += grad_pV*shape[j];
      }
      
    }
  }
}



void ConstantPressureGradient::addForcingIntegrals(Vector &in)
{
  double *dataB = b->GetData();
  for(int i=0; i<in.Size(); i++)
  {
    in[i] = in[i] + dataB[i];
  }
}

