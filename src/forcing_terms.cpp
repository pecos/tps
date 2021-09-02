#include "forcing_terms.hpp"
#include <general/forall.hpp>

ForcingTerms::ForcingTerms( const int &_dim,
                            const int &_num_equation,
                            const int &_order,
                            const int &_intRuleType,
                            IntegrationRules *_intRules,
                            ParFiniteElementSpace *_vfes,
                            ParGridFunction *_Up,
                            ParGridFunction *_gradUp,
                            const volumeFaceIntegrationArrays &_gpuArrays ):
dim(_dim),
num_equation(_num_equation),
order(_order),
intRuleType(_intRuleType),
intRules(_intRules),
vfes(_vfes),
Up(_Up),
gradUp(_gradUp),
gpuArrays(_gpuArrays)
{
  h_numElems  = gpuArrays.numElems.HostRead();
  h_posDofIds = gpuArrays.posDofIds.HostRead();
}

ForcingTerms::~ForcingTerms()
{
}

ConstantPressureGradient::ConstantPressureGradient(const int& _dim, 
                                                   const int& _num_equation, 
                                                   const int& _order, 
                                                   const int& _intRuleType,
                                                   IntegrationRules* _intRules, 
                                                   ParFiniteElementSpace* _vfes,
                                                   ParGridFunction *_Up,
                                                   ParGridFunction *_gradUp,
                                                   const volumeFaceIntegrationArrays &_gpuArrays,
                                                   RunConfiguration& _config ):
ForcingTerms(_dim,
             _num_equation,
             _order,
             _intRuleType,
             _intRules,
             _vfes,
             _Up,
             _gradUp,
             _gpuArrays )
{
  pressGrad.UseDevice(true);
  pressGrad.SetSize(3);
  pressGrad = 0.;
  double *h_pressGrad = pressGrad.HostWrite();
  {
    double *data = _config.GetImposedPressureGradient();
    for(int jj=0;jj<3;jj++) h_pressGrad[jj] = data[jj];
  }
  pressGrad.ReadWrite();
}

void ConstantPressureGradient::updateTerms(Vector &in)
{
#ifdef _GPU_
  for(int elType=0;elType<gpuArrays.numElems.Size();elType++)
  {
    int elemOffset = 0;
    if( elType!=0 )
    {
      for(int i=0;i<elType;i++) elemOffset += h_numElems[i];
    }
    int dof_el = h_posDofIds[2*elemOffset +1];
    
    updateTerms_gpu(in,
                    h_numElems[elType],
                    elemOffset,
                    dof_el,
                    vfes->GetNDofs(),
                    pressGrad,
                    b,
                    *Up,
                    *gradUp,
                    num_equation,
                    dim,
                    gpuArrays );
  }
#else
  int numElem = vfes->GetNE();
  int dof = vfes->GetNDofs();
  
  double *data = in.GetData();
  const double *dataUp = Up->GetData();
  const double *dataGradUp = gradUp->GetData();
  
  Vector gradUpk; // interpolated gradient
  gradUpk.SetSize(num_equation*dim);
  Vector upk;        // interpolated Up
  upk.SetSize(num_equation);
  double grad_pV;

  for(int el=0; el<numElem; el++)
  {
    const FiniteElement *elem = vfes->GetFE(el);
    ElementTransformation *Tr = vfes->GetElementTransformation(el);
    const int dof_elem = elem->GetDof();
    
    // nodes of the element 
    Array<int> nodes;
    vfes->GetElementVDofs(el, nodes);
    
    
    const int order = elem->GetOrder();
    int intorder = 2*order;
    //if(intRuleType==1 && trial_fe.GetGeomType()==Geometry::SQUARE) intorder--; // when Gauss-Lobatto
    const IntegrationRule *ir = &intRules->Get(elem->GetGeomType(), intorder);
    for (int k= 0; k< ir->GetNPoints(); k++)
    {
      const IntegrationPoint &ip = ir->IntPoint(k);
      
      Vector shape(dof_elem);
      elem->CalcShape(ip, shape);
      
      // interpolate primitives and their gradients
      upk = 0.;
      gradUpk = 0.;
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int j= 0;j<dof_elem;j++)
        {
          int i = nodes[j];
          upk[eq] += dataUp[i+eq*dof]*shape[j];
          for(int d=0;d<dim;d++) gradUpk[eq+d*num_equation] += 
            dataGradUp[i+eq*dof+d*num_equation*dof]*shape[j];
        }
      }

      Tr->SetIntPoint(&ip);
      shape *= Tr->Jacobian().Det()*ip.weight;
      
      grad_pV = 0.;
      for(int d=0;d<dim;d++)
      {
        grad_pV -= upk[d+1]*pressGrad[d];
        grad_pV -= upk[num_equation-1]*gradUpk[d+1 + d*num_equation];
      }
          
      for (int j= 0;j<dof_elem;j++)
      {
        int i = nodes[j];
        for(int d=0;d<dim;d++)
        {
          data[i +(d+1)*dof] -= pressGrad[d]*shape[j];
        }
        data[i +(num_equation-1)*dof] += grad_pV*shape[j];
      }
      
    }
  }
#endif
}


#ifdef _GPU_
void ConstantPressureGradient::updateTerms_gpu( Vector &in,
                                                const int numElems,
                                                const int offsetElems,
                                                const int elDof,
                                                const int totalDofs,
                                                Vector &pressGrad,
                                                ParGridFunction *b,
                                                const Vector &Up,
                                                Vector &gradUp,
                                                const int num_equation,
                                                const int dim,
                                                const volumeFaceIntegrationArrays &gpuArrays)
{
  const double *d_pressGrad = pressGrad.Read();
  double *d_in = in.ReadWrite();
  
  const double *d_Up = Up.Read();
  double *d_gradUp = gradUp.ReadWrite();
  auto d_posDofIds = gpuArrays.posDofIds.Read();
  auto d_nodesIDs = gpuArrays.nodesIDs.Read();
  const double *d_elemShapeDshapeWJ = gpuArrays.elemShapeDshapeWJ.Read();
  auto d_elemPosQ_shapeDshapeWJ = gpuArrays.elemPosQ_shapeDshapeWJ.Read();
  
  MFEM_FORALL_2D(el,numElems,elDof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,elDof)
    {
        
      MFEM_SHARED double Ui[216*5],gradUpi[216*5*3];
      MFEM_SHARED double l1[216];
      MFEM_SHARED double contrib[216*4];
      
      const int eli = el + offsetElems;
      const int offsetIDs    = d_posDofIds[2*eli];
      const int offsetDShape = d_elemPosQ_shapeDshapeWJ[2*eli   ];
      const int Q            = d_elemPosQ_shapeDshapeWJ[2*eli +1];
      
      const int indexi = d_nodesIDs[offsetIDs + i];
      
      // fill out Ui and init gradUpi
      // NOTE: density is not being loaded to save mem. accesses
      for(int eq=1;eq<num_equation;eq++)
      {
        Ui[i + eq*elDof] = d_Up[indexi + eq*totalDofs];
        contrib[i+(eq-1)*elDof] = 0.;
        
        for(int d=0;d<dim;d++)
        {
          gradUpi[i +eq*elDof +d*num_equation*elDof] = 
             d_gradUp[indexi+eq*totalDofs+d*num_equation*totalDofs];
        }
      }
      MFEM_SYNC_THREAD;
      
      // volume integral
      double gradUpk[5*3]; // interpolated gradient
      double upk[5];        // interpolated Up
      double grad_pV;
      for(int k=0;k<Q;k++)
      {
        const double weightDetJac = d_elemShapeDshapeWJ[offsetDShape+elDof+dim*elDof+k*((dim+1)*elDof+1)];
        l1[i]                     = d_elemShapeDshapeWJ[offsetDShape+i+              k*((dim+1)*elDof+1)];
        for(int j=0;j<num_equation;j++) upk[j] = 0.;
        for(int j=0;j<num_equation*dim;j++) gradUpk[j] = 0.;
        MFEM_SYNC_THREAD;
        
        for(int j=0;j<elDof;j++) 
        {
          for(int eq=1;eq<num_equation;eq++)
          {
            upk[eq] += Ui[j+eq*elDof]*l1[j];
            for(int d=0;d<dim;d++) 
              gradUpk[eq+d*num_equation] += gradUpi[j+eq*elDof+d*num_equation*elDof]*l1[j];
          }
        }
        
        grad_pV = 0.;
        // add integration point contribution
        for(int d=0;d<dim;d++)
        {
          contrib[i+d*elDof] -= d_pressGrad[d]*l1[i]*weightDetJac;
          
          grad_pV -= upk[1+d]*d_pressGrad[d];
          grad_pV -= upk[num_equation-1]*gradUpk[1+d +d*num_equation];
        }
        contrib[i+dim*elDof] += grad_pV*l1[i]*weightDetJac;
      }
      
      // write to global memory
      for(int eq=1;eq<num_equation;eq++)
      {
        d_in[indexi+eq*totalDofs] += contrib[i+(eq-1)*elDof];
      }
    }
  });
}

#endif

#ifdef _MASA_
MASA_forcings::MASA_forcings( const int& _dim, 
                              const int& _num_equation, 
                              const int& _order, 
                              const int& _intRuleType,
                              IntegrationRules* _intRules, 
                              ParFiniteElementSpace* _vfes,
                              ParGridFunction *_Up,
                              ParGridFunction *_gradUp,
                              const volumeFaceIntegrationArrays &gpuArrays,
                              RunConfiguration& _config ):
ForcingTerms(_dim,
             _num_equation,
             _order,
             _intRuleType,
             _intRules,
             _vfes,
             _Up,
             _gradUp,
             gpuArrays )
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
