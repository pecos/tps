
#include "face_integrator.hpp"

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(IntegrationRules *_intRules,
                               RiemannSolver *rsolver_, 
                               Fluxes *_fluxClass,
                               ParFiniteElementSpace *_vfes,
                               bool _useLinear,
                               const int _dim,
                               const int _num_equation,
                               ParGridFunction *_gradUp,
                               ParFiniteElementSpace *_gradUpfes,
                               double &_max_char_speed ):
rsolver(rsolver_),
fluxClass(_fluxClass),
vfes(_vfes),
dim(_dim),
num_equation(_num_equation),
max_char_speed(_max_char_speed),
gradUp(_gradUp),
gradUpfes(_gradUpfes),
intRules(_intRules),
useLinear(_useLinear)
{
  if( useLinear )
  {
    faceMassMatrix1 = new DenseMatrix[vfes->GetNF()-vfes->GetNBE()];
    faceMassMatrix2 = new DenseMatrix[vfes->GetNF()-vfes->GetNBE()];
    faceMassMatrixComputed = false;
    faceNum = 0;
  }
}

FaceIntegrator::~FaceIntegrator()
{
  if( useLinear )
  {
    delete[] faceMassMatrix1;
    delete[] faceMassMatrix2;
  }
}


void FaceIntegrator::getElementsGrads(FaceElementTransformations &Tr, 
                                      const FiniteElement &el1, 
                                      const FiniteElement &el2, 
                                      DenseMatrix &gradUp1, 
                                      DenseMatrix &gradUp2)
{
  double *dataGradUp = gradUp->GetData();
  
  const int totDofs = vfes->GetNDofs();
  Array<int> vdofs1;
  vfes->GetElementVDofs(Tr.Elem1->ElementNo, vdofs1);
  int eldDof = el1.GetDof();
  for(int n=0; n<eldDof; n++)
  {
    int index = vdofs1[n];
    for(int eq=0;eq<num_equation;eq++)
    {
      for(int d=0;d<dim;d++) gradUp1(n,eq +d*num_equation) = 
        dataGradUp[index + eq*totDofs + d*num_equation*totDofs];
    } 
  }
  
  eldDof = el2.GetDof();
  Array<int> vdofs2;
  int no2 = Tr.Elem2->ElementNo;
  int NE  = vfes->GetNE();
  if( no2>=NE )
  {
    int Elem2NbrNo = no2 - NE;
    gradUpfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
    
    Array<double> arrayGrad2(vdofs2.Size());
    gradUp->FaceNbrData().GetSubVector(vdofs2, arrayGrad2.GetData() );
    for(int n=0; n<eldDof; n++)
    {
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) gradUp2(n,eq +d*num_equation) = 
          arrayGrad2[n + eq*eldDof + d*num_equation*eldDof];
      } 
    }
    
  }else
  {
    vfes->GetElementVDofs(no2, vdofs2);
    
    for(int n=0; n<eldDof; n++)
    {
      int index = vdofs2[n];
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) gradUp2(n,eq +d*num_equation) = 
          dataGradUp[index + eq*totDofs + d*num_equation*totDofs];
      } 
    }
    
  }

}

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
  if( useLinear )
  {
    MassMatrixFaceIntegral(el1, el2, Tr, elfun, elvect);
  }else
  {
    NonLinearFaceIntegration(el1, el2, Tr, elfun, elvect);
  }
}


void FaceIntegrator::NonLinearFaceIntegration(const FiniteElement &el1,
                                              const FiniteElement &el2,
                                              FaceElementTransformations &Tr,
                                              const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
  
   Vector shape1;
   Vector shape2;
   Vector funval1(num_equation);
   Vector funval2(num_equation);
   Vector nor(dim);
   Vector fluxN(num_equation);
  
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2,
                          num_equation);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2,
                           num_equation);
   
   // Get gradients of elements
   DenseMatrix gradUp1(dof1,num_equation*dim);
   DenseMatrix gradUp2(dof2,num_equation*dim);
   getElementsGrads(Tr, el1, el2, gradUp1, gradUp2);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
   else
   {
      intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      intorder++;
   }
   //IntegrationRules IntRules2(0, Quadrature1D::GaussLobatto);
   const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
   
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, funval1);
      elfun2_mat.MultTranspose(shape2, funval2);
      
      // Interpolate gradients at int. point
      DenseMatrix gradUp1i(num_equation,dim);
      DenseMatrix gradUp2i(num_equation,dim);
      gradUp1i = 0.; gradUp2i = 0.;
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          for(int k=0;k<dof1;k++) gradUp1i(eq,d) += 
                                  gradUp1(k,eq+d*num_equation)*shape1(k);
          for(int k=0;k<dof2;k++) gradUp2i(eq,d) += 
                                  gradUp2(k,eq+d*num_equation)*shape2(k);
        }
      }

      // Get the normal vector and the convective flux on the face
      CalcOrtho(Tr.Jacobian(), nor);
      rsolver->Eval(funval1, funval2, nor, fluxN);
      
      // compute viscous fluxes
      DenseMatrix viscF1(num_equation,dim);
      DenseMatrix viscF2(num_equation,dim);
      fluxClass->ComputeViscousFluxes(funval1,gradUp1i,viscF1);
      fluxClass->ComputeViscousFluxes(funval2,gradUp2i,viscF2);
      // compute mean flux
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) viscF1(eq,d) += viscF2(eq,d);
      }
      viscF1 *= -0.5;
      
      // add to convective fluxes
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) fluxN[eq] += viscF1(eq,d)*nor[d];
      }

      fluxN *= ip.weight;
      for (int k = 0; k < num_equation; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}

void FaceIntegrator::MassMatrixFaceIntegral(const FiniteElement &el1,
                                            const FiniteElement &el2,
                                            FaceElementTransformations &Tr,
                                            const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
  
   Vector shape1;
   Vector shape2;
   Vector nor(dim);
  
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1*num_equation, dof2,
                          num_equation);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1*num_equation, dof2,
                           num_equation);
   
   // Get gradients of elements
   DenseMatrix gradUp1(dof1,num_equation*dim);
   DenseMatrix gradUp2(dof2,num_equation*dim);
   getElementsGrads(Tr, el1, el2, gradUp1, gradUp2);
   
//    DenseMatrix faceMass1( dof1 );
//    DenseMatrix faceMass2( dof2 );
//    faceMass1 = 0.;
//    faceMass2 = 0.;
   // generate face mass matrix
   if( !faceMassMatrixComputed )
   {
     //cout<<faceNum<<" "<<vfes->GetNF()<<" "<<vfes->GetNBE()<<endl;
     faceMassMatrix1[faceNum].SetSize(dof1);
     faceMassMatrix2[faceNum].SetSize(dof2);
     faceMassMatrix1[faceNum] = 0.;
     faceMassMatrix2[faceNum] = 0.;
     
    // Integration order calculation from DGTraceIntegrator
    int intorder = el1.GetOrder() + el2.GetOrder();
    const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Tr.SetAllIntPoints(&ip); // set face and element int. points

        // Calculate basis functions on both elements at the face
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);
        
        for (int j=0;j<dof1;j++)
        {
          for(int ii=0;ii<dof1;ii++) faceMassMatrix1[faceNum](ii,j) += shape1[ii]*shape1[j]*ip.weight;
        }
        for (int j=0;j<dof2;j++)
        {
          for(int ii=0;ii<dof2;ii++) faceMassMatrix2[faceNum](ii,j) += shape2[ii]*shape2[j]*ip.weight;
        }
    }
   }
   
   // face mass matrix
   DenseMatrix faceMass1( faceMassMatrix1[faceNum] );
   DenseMatrix faceMass2( faceMassMatrix2[faceNum] );
   
  // Riemann fluxes
  {
    // make sure we have as many integration points as face points
    DG_FECollection fec(el1.GetOrder(), dim-1, Tr.GetGeometryType());
    int numDofFace = fec.DofForGeometry( Tr.GetGeometryType() );
    bool getIntOrder = true;
    int intorder = el1.GetOrder();
    // intorder is calculated in a non-elegant way in order to keep going
    // This should be calculated in a proper way; REVISIT THIS!
    while( getIntOrder )
    {
      const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
      if( ir->GetNPoints()==numDofFace )
      {
        getIntOrder = false;
      }else
      {
        intorder++;
      }
    }
    
    const IntegrationRule *ir = &intRules->Get(Tr.GetGeometryType(), intorder);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      
      int index1, index2;
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);
      for(int j=0;j<dof1;j++) if(fabs(shape1[j])>1e-10) index1 = j;
      for(int j=0;j<dof2;j++) if(fabs(shape2[j])>1e-10) index2 = j;
      
      Vector state1(num_equation), state2(num_equation);
      for(int eq=0;eq<num_equation;eq++)
      {
        state1[eq] = elfun1_mat(index1,eq);
        state2[eq] = elfun2_mat(index2,eq);
      }
      
      Vector fluxN(num_equation);
      rsolver->Eval(state1, state2, nor, fluxN);
      
      DenseMatrix igradUp1(num_equation,dim), igradUp2(num_equation,dim);
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++)
        {
          igradUp1(eq,d) = gradUp1(index1,eq +d*num_equation);
          igradUp2(eq,d) = gradUp2(index2,eq +d*num_equation);
        }
      }
      
      DenseMatrix viscF1(num_equation,dim),viscF2(num_equation,dim);
      fluxClass->ComputeViscousFluxes(state1,igradUp1,viscF1);
      fluxClass->ComputeViscousFluxes(state2,igradUp2,viscF2);
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) viscF1(eq,d) += viscF2(eq,d);
      }
      viscF1 *= 0.5;
      // add to convective fluxes
      for(int eq=0;eq<num_equation;eq++)
      {
        for(int d=0;d<dim;d++) fluxN[eq] -= viscF1(eq,d)*nor[d];
      }
      
      for(int j=0;j<dof1;j++)
      {
        //cout<<faceMass1(index1,j)<<endl;
        for(int eq=0;eq<num_equation;eq++)
        {
          elvect1_mat(j,eq) -= fluxN[eq]*faceMass1(index1,j);
        }
      }
      for(int j=0;j<dof2;j++)
      {
        for(int eq=0;eq<num_equation;eq++)
        {
          elvect2_mat(j,eq) += fluxN[eq]*faceMass2(index2,j);
        }
      }
    }
  }
  faceNum++;
  if(faceNum==vfes->GetNF()-vfes->GetNBE())
  {
    faceNum = 0;
    faceMassMatrixComputed = true;
  }
}
