#include "fluxes.hpp"

Fluxes::Fluxes(EquationOfState *_eqState,
               Equations &_eqSystem,
               const int &_num_equations,
               const int &_dim
              ):
eqState(_eqState),
eqSystem(_eqSystem),
dim(_dim),
num_equations(_num_equations)
{
  gradT.SetSize(dim);
  vel.SetSize(dim);
  vtmp.SetSize(dim);
  stress.SetSize(dim,dim);
  Rg = eqState->GetGasConstant();
}

void Fluxes::ComputeTotalFlux(const Vector& state, 
                              const DenseMatrix& gradUpi, 
                              DenseMatrix& flux)
{
  switch( eqSystem )
  {
    case EULER:
      ComputeConvectiveFluxes(state,flux);
      break;
    case NS:
    {
      DenseMatrix convF(num_equations,dim);
      ComputeConvectiveFluxes(state,convF);
      
      DenseMatrix viscF(num_equations,dim);
      ComputeViscousFluxes(state,gradUpi,viscF);
      for(int eq=0;eq<num_equations;eq++)
      {
        for(int d=0;d<dim;d++)flux(eq,d) = convF(eq,d) - viscF(eq,d);
      }
    }
      break;
    case MHD:
      break;
  }
}


void Fluxes::ComputeConvectiveFluxes(const Vector &state, DenseMatrix &flux)
{
   const double pres =  eqState->ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = state(d+1);
      for (int i=0;i<dim;i++)
      {
         flux(1+i,d) = state(i+1)*state(d+1)/state[0];
      }
      flux(1+d, d) += pres;
   }

   const double H = (state[1+dim] + pres)/state[0];
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim,d) = state(d+1)*H;
   }
}

void Fluxes::ComputeViscousFluxes(const Vector& state, 
                                  const DenseMatrix& gradUp, 
                                  DenseMatrix& flux)
{
  switch( eqSystem )
  {
    case NS:
    {

      const double p    = eqState->ComputePressure(state,dim);
      const double temp = p/state[0]/Rg;
      const double visc = eqState->GetViscosity(temp);
      const double bulkViscMult = eqState->GetBulkViscMultiplyer();
      const double k    = eqState->GetThermalConductivity(visc);

      // make sure density visc. flux is 0
      for(int d=0;d<dim;d++) flux(0,d) = 0.;
      
      double divV = 0.;
      for(int i=0;i<dim;i++)
      {
        for(int j=0;j<dim;j++)
          stress(i,j) = gradUp(1+j,i) + gradUp(1+i,j);
        divV += gradUp(1+i,i);
      }

      for(int i=0;i<dim;i++) stress(i,i) += (bulkViscMult -2./3.)*divV;
      stress *= visc;

      for(int i=0;i<dim;i++)
        for(int j=0;j<dim;j++)
          flux(1+i,j) = stress(i,j);
      
      // temperature gradient

      for(int d=0;d<dim;d++)
        gradT[d] = temp*(gradUp(1+dim,d)/p - gradUp(0,d)/state[0]);
      
      for(int d=0;d<dim;d++)
        vel(d) = state[1+d]/state[0];

      stress.Mult(vel,vtmp);
      for(int d=0;d<dim;d++)
        {
          flux(1+dim,d) += vtmp[d];
          flux(1+dim,d) += k*gradT[d];
        }
    }
      break;
    default:
      flux = 0.;
      break;
  }
}


void Fluxes::ComputeSplitFlux(const mfem::Vector &state, 
                              mfem::DenseMatrix &a_mat, 
                              mfem::DenseMatrix &c_mat)
{
  const int num_equations = state.Size();
  const int dim = num_equations-2;
  
  a_mat.SetSize(num_equations, dim);
  c_mat.SetSize(num_equations, dim);
  
  const double rho  = state( 0 );
  const Vector rhoV(state.GetData()+1, dim);
  const double rhoE = state(num_equations-1);
  //const double p = eqState->ComputePressure(state, dim);
  //cout<<"*"<<p<<" "<<rho<<" "<<rhoE<<endl;
  
  for(int d=0; d<dim; d++)
  {
    for(int i=0; i<num_equations-1; i++) a_mat(i,d) = rhoV(d);
    a_mat(num_equations-1,d) = rhoV(d)/rho;
    
    c_mat(0,d) = 1.;
    for(int i=0; i<dim; i++)
    {
      c_mat(i+1,d) = rhoV(i)/rho;
    }
    c_mat(num_equations-1, d) = rhoE /*+p*/;
  }
  
}

void Fluxes::convectiveFluxes_gpu( const Vector &x, 
                                  DenseTensor &flux,
                                  const double &gamma, 
                                  const int &dof, 
                                  const int &dim,
                                  const int &num_equation)
{
#ifdef _GPU_
  auto dataIn = x.Read();
  auto d_flux = flux.ReadWrite();
  
  MFEM_FORALL_2D(n,dof,num_equation,1,1,
  {
    MFEM_SHARED double Un[5];
    MFEM_SHARED double KE[3];
    MFEM_SHARED double p;
    
    MFEM_FOREACH_THREAD(eq,x,num_equation)
    {
      Un[eq] = dataIn[n + eq*dof];
      MFEM_SYNC_THREAD;
      
      if( eq<dim ) KE[eq] = 0.5*Un[1+eq]*Un[1+eq]/Un[0];
      if( dim!=3 && eq==1 ) KE[2] = 0.;
      MFEM_SYNC_THREAD;
      
      if( eq==0 ) p = EquationOfState::pressure(&Un[0],&KE[0],gamma,dim,num_equation);
      MFEM_SYNC_THREAD;
      
      double temp;
      for(int d=0;d<dim;d++)
      {
        if( eq==0 ) d_flux[n + d*dof + eq*dof*dim] = Un[1+d];
        if( eq>0 && eq<=dim )
        {
          temp = Un[eq]*Un[1+d]/Un[0];
          if( eq-1==d ) temp += p;
          d_flux[n + d*dof + eq*dof*dim] = temp;
        }
        if( eq==num_equation-1)
        {
          d_flux[n + d*dof + eq*dof*dim] = Un[1+d]*(Un[num_equation-1]+p)/Un[0];
        }
        
      }
    }
  });
#endif
}

void Fluxes::viscousFluxes_gpu( const Vector &x, 
                                ParGridFunction *gradUp,
                                DenseTensor &flux,
                                const double &gamma, 
                                const double &Rg, // gas constant
                                const double &Pr, // Prandtl number
                                const double &viscMult,
                                const double &bulkViscMult,
                                const ParGridFunction *spaceVaryViscMult,
                                const linearlyVaryingVisc &linViscData,
                                const int &dof, 
                                const int &dim,
                                const int &num_equation)
{
#ifdef _GPU_
  const double *dataIn = x.Read();
  double *d_flux = flux.ReadWrite();
  const double *d_gradUp = gradUp->Read();
  
  const double* d_spaceVaryViscMult;
  if( spaceVaryViscMult!=NULL)
  {
    d_spaceVaryViscMult = spaceVaryViscMult->Read();
  }else d_spaceVaryViscMult = NULL;
  
  MFEM_FORALL_2D(n,dof,num_equation,1,1,
  {
    MFEM_FOREACH_THREAD(eq,x,num_equation)
    {
      MFEM_SHARED double Un[5];
      MFEM_SHARED double gradUpn[5*3];
      MFEM_SHARED double vFlux[5*3];
      MFEM_SHARED double linVisc;
      
      // init. State
      Un[eq] = dataIn[n+eq*dof];
      
      for(int d=0;d<dim;d++)
      {
        gradUpn[eq+d*num_equation] = d_gradUp[n + eq*dof + d*dof*num_equation];
      }
      MFEM_SYNC_THREAD;
      
      Fluxes::viscousFlux_gpu(&vFlux[0],
                              &Un[0],
                              &gradUpn[0],
                              gamma,
                              Rg,
                              viscMult,
                              bulkViscMult,
                              Pr,
                              eq,
                              num_equation,
                              dim,
                              num_equation );
      MFEM_SYNC_THREAD;
      
      if( d_spaceVaryViscMult!=NULL )
      { 
        if(eq==0) linVisc = d_spaceVaryViscMult[n];
        MFEM_SYNC_THREAD;
        
        for(int d=0;d<dim;d++) vFlux[eq+d*num_equation] *= linVisc;
      }
      
      // write to global memory
      for(int d=0;d<dim;d++)
      {
        d_flux[n + d*dof + eq*dof*dim] -= vFlux[eq+d*num_equation];
      }
    }
  });
#endif
}

void Fluxes::FluxesVolumeIntegrals_gpu(const ParGridFunction *Up, 
                                       Vector& z, 
                                       const ParGridFunction* gradUp, 
                                       const EquationOfState* eqState, 
                                       const ParGridFunction* spaceVaryViscMult, 
                                       const linearlyVaryingVisc& linViscData, 
                                       const volumeFaceIntegrationArrays *gpuArrays,
                                       const Vector &invMArray,
                                       const Array<int> &posDofInvM,
                                       const Equations &eqSystem,
                                       const int num_equation, 
                                       const int& dim, 
                                       const int& totNumDof, 
                                       const int& NE, 
                                       const int& elemOffset, 
                                       const int& dof)
{
#ifdef _GPU_
  const double *d_Up = Up->Read();
  const double *d_gradUp = gradUp->Read();
  double *d_z = z.ReadWrite();
  
  auto d_posDofInvM = posDofInvM.Read();
  auto d_posDofIds  = gpuArrays->posDofIds.Read();
  auto d_nodesIDs   = gpuArrays->nodesIDs.Read();
  
  auto d_Dx = gpuArrays->Dx.Read();
  auto d_Dy = gpuArrays->Dy.Read();
  const double *d_Dz = NULL;
  if(dim==3) d_Dz = gpuArrays->Dz.Read();
  auto d_invMArray = invMArray.Read();
  
  const double gamma        = eqState->GetSpecificHeatRatio();
  const double Rg           = eqState->GetGasConstant();
  const double Pr           = eqState->GetPrandtlNum();
  const double viscMult     = eqState->GetViscMultiplyer();
  const double bulkViscMult = eqState->GetBulkViscMultiplyer();
  
  const double* d_spaceVaryViscMult;
  if( spaceVaryViscMult!=NULL)
  {
    d_spaceVaryViscMult = spaceVaryViscMult->Read();
  }else d_spaceVaryViscMult = NULL;
  
  
  
  MFEM_FORALL_2D(el,NE,dof,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,dof)
    {
      MFEM_SHARED double elUp[216*5], gradUpel[216*5*3];
      MFEM_SHARED double Dx[216];
      MFEM_SHARED double temp[216*5], contrib[216*5];
      
      
      int eli = el + elemOffset;
      int offsetInv = d_posDofInvM[2*eli];
      int offsetIds = d_posDofIds[2*eli];
      
      int indexi = d_nodesIDs[offsetIds+i];
      
      // fill Up
      for(int eq=0;eq<num_equation;eq++){
        elUp[i+eq*dof] = d_Up[indexi + eq*totNumDof];
        
        contrib[i+eq*dof] = 0.;
        
        if(eqSystem==NS)
          for(int d=0;d<dim;d++) gradUpel[i+eq*dof+d*num_equation*dof] = 
            d_gradUp[indexi+eq*totNumDof+d*num_equation*totNumDof];
      }
      MFEM_SYNC_THREAD;
      
      double divV = 0.;
      double rE = elUp[i+(num_equation-1)*dof]/(gamma-1.);
      for(int d=0;d<dim;d++){
        rE += 0.5*elUp[i]*elUp[i+(1+d)*dof]*elUp[i+(1+d)*dof];
        if(eqSystem==NS) divV += gradUpel[i+(1+d)*dof+d*num_equation*dof];
      }
      double temperature = elUp[i+(num_equation-1)*dof]/elUp[i]/Rg;
      double linVisc = viscMult;
      
      if( d_spaceVaryViscMult!=NULL ){ 
        linVisc *= d_spaceVaryViscMult[indexi];
      }
                 
      
      // Compute convective fluxes in x direction
      temp[i]       = elUp[i]*elUp[i+dof]; // rho*u
      temp[i+  dof] = temp[i]*elUp[i+dof] + elUp[i+(num_equation-1)*dof]; // rho*u^2+p
      temp[i+2*dof] = temp[i]*elUp[i+2*dof]; // rho*u*v
      if(dim==3) temp[i+3*dof] = temp[i]*elUp[i+3*dof]; // rho*u*w
      temp[i+(num_equation-1)*dof] = elUp[i+dof]*(rE + elUp[i+(num_equation-1)*dof]);
      
      if( eqSystem==NS ){ // compute visc fluxes x
        double visc = EquationOfState::GetViscosity_gpu(temperature);
        double k = EquationOfState::GetThermalConductivity_gpu(visc,gamma,Rg,Pr);
        visc *= linVisc;
        k *= linVisc;
        double stress[3];
        for(int d=0;d<dim;d++) stress[d] = gradUpel[i+(1+d)*dof]+gradUpel[i+dof+d*num_equation*dof];
        stress[0] += (bulkViscMult -2./3.)*divV;
        
        // add viscous fluxes to temp[]
        for(int d=0;d<dim;d++){
          temp[i+(1+d)*dof] -= visc*stress[d];
          temp[i+(num_equation-1)*dof] -= elUp[i+(1+d)*dof]*visc*stress[d];
        }
        temp[i+(num_equation-1)*dof] -= k*temperature*(gradUpel[i+(num_equation-1)*dof]/elUp[i+(num_equation-1)*dof] -
                                                       gradUpel[i                     ]/elUp[i] );
      }
      
      // mult by Dx matrix
      for(int j=0;j<dof;j++){
        Dx[i] = d_Dx[offsetInv +i +j*dof];
        MFEM_SYNC_THREAD;
        
        for(int eq=0;eq<num_equation;eq++) contrib[i+eq*dof] += Dx[i]*temp[i+eq*dof];
        MFEM_SYNC_THREAD;
      }
      
      // Compute convective fluxes in y direction
      temp[i]       = elUp[i]*elUp[i+2*dof]; // rho*v
      temp[i+  dof] = temp[i]*elUp[i+dof]; // rho*v*u
      temp[i+2*dof] = temp[i]*elUp[i+2*dof] + elUp[i+(num_equation-1)*dof]; // rho*v^2+p
      if(dim==3) temp[i+3*dof] = temp[i]*elUp[i+3*dof]; // rho*v*w
      temp[i+(num_equation-1)*dof] = elUp[i+2*dof]*(rE + elUp[i+(num_equation-1)*dof]); // v*(rE+p)
      
      if( eqSystem==NS ){ // compute visc fluxes y
        double visc = EquationOfState::GetViscosity_gpu(temperature);
        double k = EquationOfState::GetThermalConductivity_gpu(visc,gamma,Rg,Pr);
        visc *= linVisc;
        k *= linVisc;
        double stress[3];
        for(int d=0;d<dim;d++) stress[d] = gradUpel[i+(1+d)*dof   +num_equation*dof]+
                                           gradUpel[i+2*dof     +d*num_equation*dof];
        stress[1] += (bulkViscMult -2./3.)*divV;
        
        // add viscous fluxes to temp[]
        for(int d=0;d<dim;d++){
          temp[i+(1+d)*dof] -= visc*stress[d];
          temp[i+(num_equation-1)*dof] -= elUp[i+(1+d)*dof]*visc*stress[d];
        }
        temp[i+(num_equation-1)*dof] -= k*temperature*(gradUpel[i+(num_equation-1)*dof+num_equation*dof]/
                                                       elUp[i+(num_equation-1)*dof] -
                                                       gradUpel[i +num_equation*dof]/elUp[i] );
      }
      
      // mult by Dx matrix
      for(int j=0;j<dof;j++){
        Dx[i] = d_Dy[offsetInv +i +j*dof];
        MFEM_SYNC_THREAD;
        
        for(int eq=0;eq<num_equation;eq++) contrib[i+eq*dof] += Dx[i]*temp[i+eq*dof];
        MFEM_SYNC_THREAD;
      }
      
      if(dim==3){
        // Compute convective fluxes in z direction
        temp[i] = elUp[i]*elUp[i+3*dof]; // rho*w
        temp[i+  dof] = temp[i]*elUp[i+dof]; // rho*w*u
        temp[i+2*dof] = temp[i]*elUp[i+2*dof]; // rho*w*v
        temp[i+3*dof] = temp[i]*elUp[i+3*dof] + elUp[i+(num_equation-1)*dof]; // rho*w^2+p
        temp[i+(num_equation-1)*dof] = elUp[i+3*dof]*(rE + elUp[i+(num_equation-1)*dof]); // w*(rE+p)
        
        if( eqSystem==NS ){ // compute visc fluxes z
          double visc = EquationOfState::GetViscosity_gpu(temperature);
          double k = EquationOfState::GetThermalConductivity_gpu(visc,gamma,Rg,Pr);
          visc *= linVisc;
          k *= linVisc;
          double stress[3];
          for(int d=0;d<dim;d++) stress[d] = gradUpel[i+(1+d)*dof +2*num_equation*dof]+
                                            gradUpel[i+3*dof      +d*num_equation*dof];
          stress[1] += (bulkViscMult -2./3.)*divV;
          
          // add viscous fluxes to temp[]
          for(int d=0;d<dim;d++){
            temp[i+(1+d)*dof] -= visc*stress[d];
            temp[i+(num_equation-1)*dof] -= elUp[i+(1+d)*dof]*visc*stress[d];
          }
          temp[i+(num_equation-1)*dof] -= k*temperature*(gradUpel[i+(num_equation-1)*dof+2*num_equation*dof]/
                                                        elUp[i+(num_equation-1)*dof] -
                                                        gradUpel[i +2*num_equation*dof]/elUp[i] );
        }
        
        // mult by Dx matrix
        for(int j=0;j<dof;j++){
          Dx[i] = d_Dz[offsetInv +i +j*dof];
          MFEM_SYNC_THREAD;
          
          for(int eq=0;eq<num_equation;eq++) contrib[i+eq*dof] += Dx[i]*temp[i+eq*dof];
          MFEM_SYNC_THREAD;
        }
      }
      
      // GET PREVIOUS TERMS IN Z AND MULT. BY INVERSE OF MASS MATRIX
      // get terms from global memory
      for(int eq=0;eq<num_equation;eq++){
        contrib[i+eq*dof] += d_z[indexi+eq*totNumDof];
        temp[i+eq*dof] = 0.;
      }
      MFEM_SYNC_THREAD;
      
      // multiply by inverse
      for(int j=0;j<dof;j++){
        Dx[i] = d_invMArray[offsetInv +i +j*dof];
        MFEM_SYNC_THREAD;
        
        for(int eq=0;eq<num_equation;eq++) temp[i+eq*dof] += Dx[i]*contrib[i+eq*dof];
        MFEM_SYNC_THREAD;
      }
      
      // set contribution to global memory
      for(int eq=0;eq<num_equation;eq++) d_z[indexi+eq*totNumDof] = temp[i+eq*dof];
    }
  });
#endif
}

