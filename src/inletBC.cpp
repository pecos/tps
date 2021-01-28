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
inletType(_bcType),
inputState(_inputData)
{
  meanUp.SetSize(num_equation);
  
  meanUp[0] = 1.2;
  meanUp[1] = 60;
  meanUp[2] = 0;
  meanUp[3] = 101300;
  
  Array<double> coords;
  
  
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
        IntegrationPoint ip = ir.IntPoint(i);
        double x[3];
        Vector transip(x, 3);
        Tr->Transform(ip,transip);
        for(int d=0;d<3;d++) coords.Append( transip[d] );
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
  
  // init. unit tangent vector tangent1
  Array<double> coords1,coords2;
  for(int d=0;d<dim;d++)
  {
    coords1.Append( coords[d] );
    coords2.Append( coords[d+3] );
  }
  tangent1.SetSize(dim);
  double modVec = 0.;
  for(int d=0;d<dim;d++)
  {
    modVec += (coords2[d]-coords1[d])*(coords2[d]-coords1[d]);
    tangent1[d] = coords2[d]-coords1[d];
  }
  tangent1 *= 1./sqrt( modVec );
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
      subsonicReflectingDensityVelocity(normal,stateIn,bdrFlux);
      break;
    case SUB_DENS_VEL_NR:
      subsonicNonReflectingDensityVelocity(normal,stateIn,gradState,bdrFlux);
      break;
  }
}

void InletBC::updateMean(IntegrationRules *intRules,
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


void InletBC::subsonicNonReflectingDensityVelocity(Vector &normal,
                                                    Vector &stateIn, 
                                                    DenseMatrix &gradState,
                                                    Vector &bdrFlux)
{
  const double gamma = eqState->GetSpecificHeatRatio();
  const double p = eqState->ComputePressure(stateIn, dim);
  
  if( patchNumber==5 )
  {
    //cout<<"merde"<<endl;
  }

  Vector unitNorm = normal;
  {
    double mod = 0.;
    for(int d=0;d<dim;d++) mod += normal[d]*normal[d];
    unitNorm *= -1./sqrt(mod); // pointo into domain!!
  }
  
  double meanVn = 0.;
  double meanVt1 = 0.;
  for(int d=0;d<dim;d++)
  {
    meanVn += unitNorm[d]*meanUp[d+1];
    meanVt1 += tangent1[d]*meanUp[d+1];
  }
  
  double meanDVx = meanUp[1] -inputState[1];
  double meanDVy = meanUp[2] -inputState[2];
  
  // normal gradients
  Vector normGrad(num_equation);
  normGrad = 0.;
  for(int eq=0;eq<num_equation;eq++)
  {
    for(int d=0;d<dim;d++) normGrad[eq] += unitNorm[d]*gradState(eq,d);
  }
  
  const double rho = stateIn[0];
  const double u = stateIn[1]/rho;
  const double v = stateIn[2]/rho;
  const double k = 0.5*(u*u + v*v);
  
  const double speedSound = sqrt(gamma*meanUp[3]/meanUp[0]);
  const double meanK = 0.5*(meanUp[1]*meanUp[1] + meanUp[2]*meanUp[2]);
  
  // compute outgoing characteristic
  double L1 = normGrad[3] -meanUp[0]*speedSound*(normGrad[1]*unitNorm[0] +
                                                 normGrad[2]*unitNorm[1] );
  L1 *= meanVn - speedSound;
  
  // estimate ingoing characteristic
  const double sigma = speedSound/2.;
  const double L5 = sigma*2.*meanUp[0]*speedSound*(meanDVx*unitNorm[0]+
                                                   meanDVy*unitNorm[1] );
  const double L3 = sigma*(meanDVx*tangent1[0]+
                           meanDVy*tangent1[1] );
  double L2 = sigma*speedSound*speedSound*(meanUp[0]-inputState[0]) - 0.5*L5;
  
  // calc vector d
  const double d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
  const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d3 = L3;
  const double d5 = 0.5*(L5+L1);
  
  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVn*d1  + meanUp[0]*d2;
  bdrFlux[2] = meanVt1*d1 + meanUp[0]*d3;
  bdrFlux[3] = meanK*d1   + meanUp[0]*meanVn*d2 + 
                            meanUp[0]*meanVt1*d3 + d5/(gamma-1.);
  
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
  boundaryU.GetRow(bdrN,state2);
  
  Vector stateN = state2;
  stateN[1] = state2[1]*unitNorm[0] + state2[2]*unitNorm[1];
  stateN[2] = state2[1]*tangent1[0] + state2[2]*tangent1[1];
  
  Vector newU(num_equation);
  //for(int i=0; i<num_equation;i++) newU[i] = state2[i]- dt*(bdrFlux[i] /*+ fluxY[i]*/);
  for(int i=0; i<num_equation;i++) newU[i] = stateN[i]- dt*bdrFlux[i];
  
  // transform back into x-y coords
  {
    DenseMatrix M(dim,dim);
    M(0,0) = unitNorm[0]; M(0,1) = unitNorm[1];
    M(1,0) = tangent1[0]; M(1,1) = tangent1[1];
    DenseMatrix invM(dim,dim);
    mfem::CalcInverse(M,invM);
    Vector momN(dim), momX(dim);
    for(int d=0;d<dim;d++) momN[d] = newU[1+d];
    invM.Mult(momN,momX);
    for(int d=0;d<dim;d++) newU[1+d] = momX[d]; 
  }
  boundaryU.SetRow(bdrN,newU);
  bdrN++;
  
  rsolver->Eval(stateIn,state2,normal,bdrFlux);
}

void InletBC::subsonicReflectingDensityVelocity(Vector &normal,
                                                Vector &stateIn,
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
