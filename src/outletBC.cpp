#include "outletBC.hpp"

OutletBC::OutletBC( MPI_Groups *_groupsMPI,
                    RiemannSolver *_rsolver, 
                    EquationOfState *_eqState,
                    ParFiniteElementSpace *_vfes,
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
groupsMPI(_groupsMPI),
outletType(_bcType),
inputState(_inputData)
{
  groupsMPI->setAsOutlet();
  
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
  
  tangent2.SetSize(dim);
  tangent2 = 0.;
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
                          ParGridFunction *Up)
{
  bdrN = 0;
  Vector localMeanUp(num_equation+1);
  localMeanUp = 0.;
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
          localMeanUp[eq] += sum;
          if( !bdrUInit ) boundaryU(Nbdr,eq) = sum;
        }
        Nbdr++;
      }
    }
  }
 
  localMeanUp[num_equation] = (double)Nbdr;
  Vector sum(num_equation+1);
  MPI_Allreduce(localMeanUp.GetData(), sum.GetData(),
                num_equation+1, MPI_DOUBLE, MPI_SUM, groupsMPI->getOutletComm());
  
  for(int eq=0;eq<num_equation;eq++) meanUp[eq] = sum[eq]/sum[num_equation];
  
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
  
  Vector unitNorm = normal;
  {
    double mod = 0.;
    for(int d=0;d<dim;d++) mod += normal[d]*normal[d];
    unitNorm *= 1./sqrt(mod);
  }
  
  // mean velocity in inlet normal and tangent directions
  Vector meanVel(dim);
  meanVel = 0.;
  for(int d=0;d<dim;d++)
  {
    meanVel[0] += unitNorm[d]*meanUp[d+1];
    meanVel[1] += tangent1[d]*meanUp[d+1];
  }
  if( dim==3)
  {
    tangent2[0] = unitNorm[1]*tangent1[2]-unitNorm[2]*tangent1[1];
    tangent2[1] = unitNorm[2]*tangent1[0]-unitNorm[0]*tangent1[2];
    tangent2[2] = unitNorm[0]*tangent1[1]-unitNorm[1]*tangent1[0];
    for(int d=0;d<dim;d++) meanVel[2] += tangent2[d]*meanUp[d+1];
  }
  
  // velocity difference between actual and desired values
  Vector meanDV(dim);
  for(int d=0;d<dim;d++) meanDV[d] = meanUp[1+d] -inputState[1+d];
  
  
  // normal gradients
  Vector normGrad(num_equation);
  normGrad = 0.;
  for(int eq=0;eq<num_equation;eq++)
  {
    for(int d=0;d<dim;d++) normGrad[eq] += unitNorm[d]*gradState(eq,d);
  }
  
  
  const double rho = stateIn[0];
  double k = 0.;
  for(int d=0;d<dim;d++) k += stateIn[1+d]*stateIn[1+d];
  k *= 0.5/rho/rho;
  
  const double speedSound = sqrt(gamma*meanUp[num_equation-1]/meanUp[0]);
  double meanK = 0.;
  for(int d=0;d<dim;d++) meanK += meanUp[1+d]*meanUp[1+d];
  meanK *= 0.5;
  
  // compute outgoing characteristics
  double L2 = speedSound*speedSound*normGrad[0] - normGrad[num_equation-1];
  L2 *= meanVel[0];
  
  double L3 = 0;
  for(int d=0;d<dim;d++) L3 += tangent1[d]*normGrad[1+d];
  L3 *= meanVel[0];
  
  double L4 = 0.;
  if(dim==3)
  {
    for(int d=0;d<dim;d++) L3 += tangent2[d]*normGrad[1+d];
    L4 *= meanVel[0];
  }
  
  double L5 = 0.;
  for(int d=0;d<dim;d++) L5 += unitNorm[d]*normGrad[1+d];
  L5 = normGrad[num_equation-1] +rho*speedSound*L5;
  L5 *= meanVel[0] + speedSound;
  
  //const double p = eqState->ComputePressure(stateIn, dim);
  
  // estimate ingoing characteristic
  const double sigma = speedSound/2.;
  double L1 = sigma*(meanUp[3] - inputState[0]);
  
  // calc vector d
  const double d1 = (L2+0.5*(L5+L1))/speedSound/speedSound;
  const double d2 = 0.5*(L5-L1)/rho/speedSound;
  const double d3 = L3;
  const double d4 = L4;
  const double d5 = 0.5*(L5+L1);
  
  // dF/dx
  bdrFlux[0] = d1;
  bdrFlux[1] = meanVel[0]*d1 + meanUp[0]*d2;
  bdrFlux[2] = meanVel[1]*d1 + meanUp[0]*d3;
  if(dim==3) bdrFlux[3] = meanVel[2]*d1 + meanUp[0]*d4;
  bdrFlux[num_equation-1] = 0.;
  for(int d=0;d<dim;d++) bdrFlux[num_equation-1] += meanUp[0]*meanVel[d];
  bdrFlux[num_equation-1] += meanK*d1 + d5/(gamma-1.);
  
                            
  // flux gradients in other directions
//   Vector fluxY(num_equation);
//   {
//     // gradients in tangent1 direction
//     Vector tg1Grads(num_equation);
//     tg1Grads = 0.;
//     for(int eq=0;eq<num_equation;eq++)
//     {
//       for(int d=0;d<dim;d++) tg1Grads[eq] += tangent1[d]*gradState(eq,d);
//     }
//     
//     
//     const double dy_rv = tg1Grads(0)*meanVt1 + rho*tg1Grads(2);
//     const double dy_ruv = meanVn*dy_rv + rho*meanVt1*tg1Grads(1);
//     const double dy_rv2 = meanVt1*dy_rv + rho*meanVt1*tg1Grads(2);
//     const double dy_p = tg1Grads(3);
//     const double dy_E = dy_p/(gamma-1.) + k*tg1Grads(0) +
//                         rho*(meanVn*tg1Grads(1) + meanVt1*tg1Grads(2) );
//     const double dy_vp = meanVt1*dy_p + p*tg1Grads(2);
//     
//     fluxY[0] = dy_rv;
//     fluxY[1] = dy_ruv;
//     fluxY[2] = dy_rv2 + dy_p;
//     fluxY[3] = stateIn(3)/rho*dy_rv + rho*meanVt1*dy_E + dy_vp;
//   }
  
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
