#include "inletBC.hpp"

InletBC::InletBC(RiemannSolver *_rsolver, 
            EquationOfState *_eqState,
            FiniteElementSpace *_vfes,
            IntegrationRules *_intRules,
            double &_dt,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            InletType _bcType,
            const Array<double> &_inputData ):
BoundaryCondition(_rsolver, 
                  _eqState,
                  _vfes,
                  _intRules,
                  _dt,
                  _dim,
                  _num_equation,
                  _patchNumber),
patchNumber(_patchNumber),
inletType(_bcType),
inputState(_inputData)
{
  meanUp.SetSize(num_equation);
  
  meanIsSet = false;
  iters = 0;
  
  meanUp[0] = 1.2;
  meanUp[1] = 60;
  meanUp[2] = 0;
  meanUp[3] = 101300;
  
  initMeanUp = meanUp;
}  
  

InletBC::~InletBC()
{
}

void InletBC::computeBdrFlux( Vector &normal,
                              Vector &stateIn, 
                              DenseMatrix &gradState,
                              Vector &bdrFlux)
{
  switch(inletType)
  {
    case SUB_DENS_VEL:
      subsonicReflectingPressureVelocity(normal,stateIn,gradState,bdrFlux);
      break;
    case SUB_DENS_VEL_NR:
      subsonicNonReflectingPressureVelocity(normal,stateIn,gradState,bdrFlux);
      break;
  }
}

void InletBC::updateMean(IntegrationRules *intRules,
                         GridFunction *Up)
{
  //if( !meanIsSet )
  if(iters%100==0)
  {
    meanUp = 0.;
    int Nbdr = 0;
    
    double *data = Up->GetData();
    for(int bel=0;bel<vfes->GetNBE(); bel++)
    {
      int attr = vfes->GetBdrAttribute(bel);
      if( attr==patchNumber )
      {
        FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
        Array<int> dofs;
        
        vfes->GetElementVDofs(Tr->Elem1No, dofs);
        //cout<< "bel "<<bel<<" attr "<<attr <<" dofs "<<dofs.Size()<<endl;
        
        // retreive data 
        int elDofs = vfes->GetFE(Tr->Elem1No)->GetDof();
        DenseMatrix elUp(elDofs,num_equation);
        for(int d=0;d<elDofs;d++)
        {
          int index = dofs[d];
          for(int eq=0;eq<num_equation;eq++) elUp(d,eq) = data[index+eq*vfes->GetNDofs()];
        }
        
        int intorder = Tr->Elem1->OrderW() + 2*vfes->GetFE(Tr->Elem1No)->GetOrder();
        if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk)
        {
          intorder++;
        }
        const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);
        for(int i=0;i<ir.GetNPoints();i++)
        {
          IntegrationPoint ip = ir.IntPoint(i);
          Tr->SetAllIntPoints(&ip);
          Vector shape(elDofs);
          vfes->GetFE(Tr->Elem1No)->CalcShape(Tr->GetElement1IntPoint(), shape);
          Vector iUp(num_equation);
          for(int eq=0;eq<num_equation;eq++)
          {
            double sum = 0.;
            for(int d=0;d<elDofs;d++)
            {
              sum += shape[d]*elUp(d,eq);
            }
            meanUp[eq] += sum;
          }
          Nbdr++;
        }
      }
    }
    
    meanUp /= double(Nbdr);
    meanIsSet = true;
//     for(int i=0;i<num_equation;i++)cout<<meanUp[i]<<" ";
//     cout<<endl;
  }
}


void InletBC::subsonicNonReflectingPressureVelocity(Vector &normal,
                                                    Vector &stateIn, 
                                                    DenseMatrix &gradState,
                                                    Vector &bdrFlux)
{
  const double gamma = eqState->GetSpecificHeatRatio();
  const double p = eqState->ComputePressure(stateIn, dim);
  
  const double rho = stateIn[0];
  const double u = stateIn[1]/rho;
  const double v = stateIn[2]/rho;
  const double k = 0.5*(u*u + v*v);
  
  const double speedSound = sqrt(gamma*meanUp[3]/meanUp[0]);
  const double meanK = 0.5*(meanUp[1]*meanUp[1] + meanUp[2]*meanUp[2]);
  
  // compute outgoing characteristic
  double L1 = gradState(3,0) -meanUp[0]*speedSound*gradState(1,0);
  L1 *= meanUp[1] - speedSound;
  
  // estimate ingoing characteristic
  const double sigma = speedSound/2.*0.;
  const double L5 = sigma*2.*meanUp[0]*speedSound*(/*u*/meanUp[1] - inputState[1]);
  const double L3 = sigma*(/*v*/meanUp[2]-inputState[2]);
  double L2 = sigma*speedSound*speedSound*(/*rho*/meanUp[0]-inputState[0]) - 0.5*L5;
  
  // calc vector d
  const double d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
  const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d3 = L3;
  const double d5 = 0.5*(L5+L1);
  
  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanUp[1]*d1 + meanUp[0]*d2;
  bdrFlux[2] = meanUp[2]*d1 + meanUp[0]*d3;
  bdrFlux[3] = meanK*d1 + meanUp[0]*meanUp[1]*d2 + meanUp[0]*meanUp[2]*d3 + d5/(gamma-1.);
  
// for(int i=0;i<num_equation;i++)cout<<bdrFlux[i]<<" ";
// cout<<endl;
  
  // flux gradients in other directions
  const double dy_rv = gradState(0,1)*v + rho*gradState(2,1);
  const double dy_ruv = u*dy_rv + rho*v*gradState(1,1);
  const double dy_rv2 = v*dy_rv + rho*v*gradState(2,1);
  const double dy_p = gradState(3,1);
  const double dy_E = dy_p/(gamma-1.) + k*gradState(0,1) +
                      rho*(u*gradState(1,1) + v*gradState(2,1) );
  const double dy_vp = v*dy_p + p*gradState(2,1);
  
  // dG/dy
  Vector fluxY(num_equation);
  fluxY[0] = dy_rv;
  fluxY[1] = dy_ruv;
  fluxY[2] = dy_rv2 + dy_p;
  fluxY[3] = stateIn(3)/rho*dy_rv + rho*v*dy_E + dy_vp;
  
  Vector state2(num_equation);
  state2 = stateIn;
//   state2[0] = initMeanUp[0];
//   state2[1] = initMeanUp[0]*initMeanUp[1];
//   state2[2] = initMeanUp[0]*initMeanUp[2];
//   state2[3] = initMeanUp[3]/(gamma-1.) + 0.5*initMeanUp[0]*(initMeanUp[1]*initMeanUp[1] +
//                                                     initMeanUp[2]*initMeanUp[2] );
//   state2[0] = meanUp[0];
//   state2[1] = meanUp[0]*meanUp[1];
//   state2[2] = meanUp[0]*meanUp[2];
//   state2[3] = meanUp[3]/(gamma-1.) + 0.5*meanUp[0]*(meanUp[1]*meanUp[1] +
//                                                     meanUp[2]*meanUp[2] );
  
//   cout << -dt*(bdrFlux[1] + fluxY[1]) << " ";
  for(int i=0; i<num_equation;i++) state2[i] -= dt*(bdrFlux[i] /*+ fluxY[i]*/);
for(int i=0;i<num_equation;i++)cout<<state2[i]<<" ";
cout<<endl;
  rsolver->Eval(state2,state2,normal,bdrFlux);
}

void InletBC::subsonicReflectingPressureVelocity( Vector &normal,
                                                  Vector &stateIn, 
                                                  DenseMatrix &gradState,
                                                  Vector &bdrFlux)
{
  const double gamma = eqState->GetSpecificHeatRatio();
  const double p = eqState->ComputePressure(stateIn, dim);
  
  Vector state2(num_equation);
  state2[0] = inputState[0];
  state2[1] = inputState[0]*inputState[1];
  state2[2] = inputState[0]*inputState[2];
  state2[3] = p/(gamma-1.) + 0.5*(state2[1]*state2[1] +
                                  state2[2]*state2[2] )/state2[0];
  
  
  rsolver->Eval(stateIn,state2,normal,bdrFlux);
}
