#include "outletBC.hpp"

OutletBC::OutletBC(RiemannSolver *_rsolver, 
            EquationOfState *_eqState,
            FiniteElementSpace *_vfes,
            IntegrationRules *_intRules,
            double &_dt,
            const int _dim,
            const int _num_equation,
            int _patchNumber,
            OutletType _bcType,
            const Array<double> &_inputData ):
BoundaryCondition(_rsolver, 
                  _eqState,
                  _vfes,
                  _intRules,
                  _dt,
                  _dim,
                  _num_equation,
                  _patchNumber),
outletType(_bcType),
inputState(_inputData)
{
  meanUp.SetSize(num_equation);
  
  meanUp[0] = 1.2;
  meanUp[1] = 60;
  meanUp[2] = 0;
  meanUp[3] = 101300;
  
  // init boundary U
  bdrN = 0;
  for(int bel=0;bel<vfes->GetNBE(); bel++)
  {
    int attr = vfes->GetBdrAttribute(bel);
    if( attr==patchNumber )
    {
      FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
      Array<int> dofs;
      vfes->GetElementVDofs(Tr->Elem1No, dofs);
      
      int intorder = Tr->Elem1->OrderW() + 2*vfes->GetFE(Tr->Elem1No)->GetOrder();
      if (vfes->GetFE(Tr->Elem1No)->Space() == FunctionSpace::Pk)
      {
        intorder++;
      }
      const IntegrationRule ir = intRules->Get(Tr->GetGeometryType(), intorder);
      for(int i=0;i<ir.GetNPoints();i++)
      {
        bdrN++;
      }
    }
  }
  
  boundaryU.SetSize(bdrN,num_equation);
  for(int i=0;i<bdrN;i++)
  {
    Vector iState(num_equation);
    iState = meanUp;
    double gamma = eqState->GetSpecificHeatRatio();
    double rE = iState[3]/(gamma-1.) + 0.5*iState[0]*(
                iState[1]*iState[1] +
                iState[2]*iState[2] );
    iState[1] *= iState[0];
    iState[2] *= iState[0];
    iState[3]  = rE;
    boundaryU.SetRow(i,iState);
  }
  bdrUInit = false;
  bdrN = 0;
}

OutletBC::~OutletBC()
{
}

void OutletBC::computeBdrFlux(Vector &normal,
                              Vector &stateIn, 
                              DenseMatrix &gradState,
                              Vector &bdrFlux)
{
  switch(outletType)
  {
    case SUB_P:
      subsonicReflectingPressure(normal,stateIn,bdrFlux);
      break;
    case SUB_P_NR:
      subsonicNonReflectingPressure(normal,stateIn,gradState,bdrFlux);
      break;
  }
}

void OutletBC::updateMean(IntegrationRules *intRules,
                          GridFunction *Up)
{
  bdrN = 0;
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
          if( !bdrUInit ) boundaryU(Nbdr,eq) = sum;
        }
        Nbdr++;
      }
    }
  }
  
  meanUp /= double(Nbdr);
  if( !bdrUInit )
  {
    for(int i=0;i<Nbdr;i++)
    {
      Vector iState(num_equation);
      boundaryU.GetRow(i,iState);
      double gamma = eqState->GetSpecificHeatRatio();
      double rE = iState[3]/(gamma-1.) + 0.5*iState[0]*(
                  iState[1]*iState[1] +
                  iState[2]*iState[2] );
      iState[1] *= iState[0];
      iState[2] *= iState[0];
      iState[3]  = rE;
      boundaryU.SetRow(i,iState);
    }
    bdrUInit = true;
  }
}

void OutletBC::subsonicNonReflectingPressure( Vector &normal,
                                              Vector &stateIn, 
                                              DenseMatrix &gradState,
                                              Vector &bdrFlux)
{
  const double gamma = eqState->GetSpecificHeatRatio();
  
  const double rho = stateIn[0];
  const double u = stateIn[1]/rho;
  const double v = stateIn[2]/rho;
  const double k = 0.5*(u*u + v*v);
  
  const double speedSound = sqrt(gamma*meanUp[3]/meanUp[0]);
  const double meanK = 0.5*(meanUp[1]*meanUp[1] + meanUp[2]*meanUp[2]);
  
  // compute outgoing characteristics
  double L2 = meanUp[1]*(speedSound*speedSound*gradState(0,0) - gradState(3,0) );
  double L3 = meanUp[1]*gradState(2,0);
  double L5 = gradState(3,0) +rho*speedSound*gradState(1,0);
  L5 *= meanUp[1] + speedSound;
  
//   cout<<"grads :";
//   for(int i=0;i<num_equation;i++)cout<<gradState(i,1)<<" ";
//   cout<<endl;
  
  const double p = eqState->ComputePressure(stateIn, dim);
  
  // estimate ingoing characteristic
  const double sigma = speedSound/2./2.;
  double L1 = sigma*(meanUp[3] - inputState[0]);
  
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
  
  // flux gradients in other directions
  const double dy_rv = gradState(0,1)*v + rho*gradState(2,1);
  const double dy_ruv = u*dy_rv + rho*v*gradState(1,1);
  const double dy_rv2 = v*dy_rv + rho*v*gradState(2,1);
  const double dy_p = gradState(3,1);
  const double dy_E = dy_p/(gamma-1.) + k*gradState(0,1) +
                      rho*(u*gradState(1,1) + v*gradState(2,1) );
  const double dy_vp = v*dy_p + p*gradState(2,1);
  
  Vector fluxY(num_equation);
  fluxY[0] = dy_rv;
  fluxY[1] = dy_ruv;
  fluxY[2] = dy_rv2 + dy_p;
  fluxY[3] = stateIn(3)/rho*dy_rv + rho*v*dy_E + dy_vp;
  
  Vector state2(num_equation);
//   state2 = stateIn;
  boundaryU.GetRow(bdrN,state2);
/*for(int i=0;i<num_equation;i++)cout<<state2[i]<<" ";
cout<<endl;*/
  Vector newU(num_equation);
  for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  boundaryU.SetRow(bdrN,newU);
  bdrN++;
  
  rsolver->Eval(stateIn,state2,normal,bdrFlux);
}


void OutletBC::subsonicReflectingPressure(Vector &normal,
                                          Vector &stateIn, 
                                          Vector &bdrFlux)
{
  const double gamma = eqState->GetSpecificHeatRatio();
  Vector state2(num_equation);
  state2 = stateIn;
  state2[3] = inputState[0]/(gamma-1.) + 0.5*(state2[1]*state2[1] +
                                              state2[2]*state2[2] )/state2[0];
  
  rsolver->Eval(stateIn,state2,normal,bdrFlux);
}
