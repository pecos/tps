

#include "turbModel.hpp"
#include "thermoChem.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include <hdf5.h>
#include "../utils/mfem_extras/pfem_extras.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace mfem::common;


TurbModel::TurbModel(mfem::ParMesh *pmesh_, RunConfiguration *config_, LoMachOptions *loMach_opts_)
  : pmesh(pmesh_),
    loMach_opts(loMach_opts_),
    config(config_) {}


void TurbModel::initialize() {

    //groupsMPI = pmesh->GetComm(); 
    rank = pmesh->GetMyRank();
    //rank0 = pmesh->isWorldRoot();
    if(rank == 0) {rank0 = true;}
    dim = pmesh->Dimension();
    nvel = dim;
  
   bool verbose = rank0;
   if (verbose) grvy_printf(ginfo, "Initializing TurbModel solver.\n");

   if (loMach_opts->uOrder == -1) {
     order = std::max(config->solOrder,1);
     double no;
     no = ceil( ((double)order * 1.5) );   
     //norder = int(no);
   } else {
     order = loMach_opts->uOrder;
     //norder = loMach_->loMach_opts_.nOrder;
   }
   
   MaxIters = config->GetNumIters();
   num_equation = 1; // hard code for now, fix with species

   //-----------------------------------------------------
   // 2) Prepare the required finite elements
   //-----------------------------------------------------

   // scalar
   sfec = new H1_FECollection(order);
   sfes = new ParFiniteElementSpace(pmesh, sfec);   

   // vector
   vfec = new H1_FECollection(order, dim);
   vfes = new ParFiniteElementSpace(pmesh, vfec, dim);
   
   // dealias nonlinear term 
   //nfec = new H1_FECollection(norder, dim);   
   //nfes = new ParFiniteElementSpace(pmesh, nfec, dim);   
   //nfecR0 = new H1_FECollection(norder);   
   //nfesR0 = new ParFiniteElementSpace(pmesh, nfecR0);         
   
   // Check if fully periodic mesh
   //if (!(pmesh->bdr_attributes.Size() == 0))
   /* update for tke/epsi
   if (!(pmesh->bdr_attributes.Size() == 0))     
   {
      temp_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      temp_ess_attr = 0;

      Qt_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      Qt_ess_attr = 0;
   }
   if (verbose) grvy_printf(ginfo, "ThermoChem paces constructed...\n");
   */

   int vfes_truevsize = vfes->GetTrueVSize();   
   int sfes_truevsize = sfes->GetTrueVSize();
   //int nfes_truevsize = nfes->GetTrueVSize();
   //int nfesR0_truevsize = nfesR0->GetTrueVSize();

   subgridViscSml.SetSize(sfes_truevsize);
   subgridVisc_gf.SetSpace(sfes);
   delta.SetSize(sfes_truevsize);   
   
   if (verbose) grvy_printf(ginfo, "TurbModel vectors and gf initialized...\n");     
   
}

void TurbModel::initializeExternal(Vector *gradU_, Vector *gradV_, Vector *gradW_, ParGridFunction *delta_gf_) {

  gradU = gradU_;
  gradV = gradV_;
  gradW = gradW_;
  delta_gf = delta_gf_;
  
}


void TurbModel::Setup(double dt)
{

   // HARD CODE
   //partial_assembly = true;
   partial_assembly = false;
   //if (verbose) grvy_printf(ginfo, "in Setup...\n");

   Sdof = sfes->GetNDofs();   
   SdofInt = sfes->GetTrueVSize();

   // Initial conditions
   
   // Boundary conditions
      
   // inlet bc
   // outlet bc   
   // wall bc

   
   Array<int> empty;

   // unsteady: p+p [+p] = 2p [3p]
   // convection: p+p+(p-1) [+p] = 3p-1 [4p-1]
   // diffusion: (p-1)+(p-1) [+p] = 2p-2 [3p-2]
   
   // GLL integration rule (Numerical Integration)
   /*
   const IntegrationRule &ir_i  = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 2*order + 1);  //3 5
   const IntegrationRule &ir_nli = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 4*order);    //4 8
   const IntegrationRule &ir_di  = gll_rules.Get(sfes->GetFE(0)->GetGeomType(), 3*order - 1); //2 5  
   if (rank0) std::cout << "Integration rules set" << endl;
   */

   // coeffs for solvers
   // density coefficient
   
}

void TurbModel::turbModelStep(double &time, double dt, const int current_step, const int start_step, std::vector<double> ab, std::vector<double> bdf, bool provisional)
{

   ab1 = ab[0];
   ab2 = ab[1];
   ab3 = ab[2];  

   bd0 = bdf[0];
   bd1 = bdf[1];
   bd2 = bdf[2];
   bd3 = bdf[3];  
  
   // Set current time for scalar Dirichlet boundary conditions.
   //for (auto &temp_dbc : temp_dbcs) {temp_dbc.coeff->SetTime(time + dt);}   

   subgridViscSml = 0.0;
   delta_gf->GetTrueDofs(delta);  
   if (config->sgsModelType > 0) {
     
     double *dGradU = gradU->HostReadWrite();
     double *dGradV = gradV->HostReadWrite();
     double *dGradW = gradW->HostReadWrite(); 
     double *del = delta.HostReadWrite();
     double *data = subgridViscSml.HostReadWrite();
     
     if (config->sgsModelType == 1) {
       for (int i = 0; i < SdofInt; i++) {     
         double nu_sgs = 0.;
	 DenseMatrix gradUp;
	 gradUp.SetSize(nvel, dim);
	 for (int dir = 0; dir < dim; dir++) { gradUp(0,dir) = dGradU[i + dir * SdofInt]; }
	 for (int dir = 0; dir < dim; dir++) { gradUp(1,dir) = dGradV[i + dir * SdofInt]; }
	 for (int dir = 0; dir < dim; dir++) { gradUp(2,dir) = dGradW[i + dir * SdofInt]; }
         sgsSmag(gradUp, del[i], nu_sgs);
         data[i] = nu_sgs;
       }
       
     } else if (config->sgsModelType == 2) {       
       for (int i = 0; i < SdofInt; i++) {     
         double nu_sgs = 0.;
	 DenseMatrix gradUp;
	 gradUp.SetSize(nvel, dim);
	 for (int dir = 0; dir < dim; dir++) { gradUp(0,dir) = dGradU[i + dir * SdofInt]; }
	 for (int dir = 0; dir < dim; dir++) { gradUp(1,dir) = dGradV[i + dir * SdofInt]; }
	 for (int dir = 0; dir < dim; dir++) { gradUp(2,dir) = dGradW[i + dir * SdofInt]; }
         sgsSigma(gradUp, del[i], nu_sgs);
	 data[i] = nu_sgs;
       }
       
     } else {
       
       if(rank0) std::cout << " => turbulence model not supported!!!" << endl;
       exit(1);
       
     }

   }
   
   subgridVisc_gf.SetFromTrueDofs(subgridViscSml);          
   bufferSubgridVisc->SetFromTrueDofs(subgridViscSml);     
     
}


   //void TurbModel::computeExplicitConvectionOP(bool extrap) {}

   //void TurbModel::updateBC(int current_step) {}

/*
void ThermoChem::EliminateRHS(Operator &A,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A.GetOutputProlongation();
   const Operator *Pi = A.GetProlongation();
   const Operator *Ri = A.GetRestriction();
   A.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   constrainedA.EliminateRHS(X, B);
}
*/

/*
void ThermoChem::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, sfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, sfes->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}
*/

/* 
void TurbModel::interpolateInlet() {
}
*/

/*
void ThermoChem::uniformInlet() {  
}
*/

/**
Basic Smagorinksy subgrid model with user-specified cutoff grid length
*/
void TurbModel::sgsSmag(const DenseMatrix &gradUp, double delta, double &nu) {

  Vector Sij(6);
  double Smag = 0.;
  double Cd;
  double l_floor;
  double d_model;
  Cd = config->sgs_model_const;
  

  // gradUp is in (eq,dim) form
  Sij[0] = gradUp(0,0);
  Sij[1] = gradUp(1,1);
  Sij[2] = gradUp(2,2);  
  Sij[3] = 0.5*(gradUp(0,1) + gradUp(1,0));
  Sij[4] = 0.5*(gradUp(0,2) + gradUp(2,0));
  Sij[5] = 0.5*(gradUp(1,2) + gradUp(2,1));

  // strain magnitude with silly sqrt(2) factor
  for (int i = 0; i < 3; i++) Smag += Sij[i]*Sij[i];
  for (int i = 3; i < 6; i++) Smag += 2.0*Sij[i]*Sij[i];
  Smag = sqrt(2.0*Smag);

  // eddy viscosity with delta shift
  //l_floor = config->GetSgsFloor();
  //d_model = Cd * max(delta-l_floor,0.0);
  d_model = Cd * delta;
  nu = d_model * d_model * Smag;
  
}

/**
NOT TESTED: Sigma subgrid model following Nicoud et.al., "Using singular values to build a 
subgrid-scale model for large eddy simulations", PoF 2011.
*/
void TurbModel::sgsSigma(const DenseMatrix &gradUp, double delta, double &nu) {

  DenseMatrix Qij(dim,dim);
  DenseMatrix du(dim,dim);
  DenseMatrix B(dim,dim);  
  Vector ev(dim);
  Vector sigma(dim);  
  double Cd;
  double sml = 1.0e-12;
  double pi = 3.14159265359;
  double onethird = 1./3.;  
  double l_floor, d_model, d4;
  double p1, p2, p, q, detB, r, phi;
  Cd = config->sgs_model_const;
  
  // Qij = u_{k,i}*u_{k,j}
  for (int j = 0; j < dim; j++) {  
    for (int i = 0; i < dim; i++) {
      Qij(i,j) = 0.;
    }    
  }
  for (int k = 0; k < dim; k++) {  
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
        Qij(i,j) += gradUp(k,i) * gradUp(k,j);
      }           
    }    
  }

  // shifted grid scale, d should really be sqrt of J^T*J
  //l_floor = config->GetSgsFloor();
  //d_model = max((delta-l_floor), sml);
  d_model = delta;
  d4 = pow(d_model,4);
  for (int j = 0; j < dim; j++) { 
    for (int i = 0; i < dim; i++) {      
      Qij(i,j) *= d4;
    }    
  }  
  
  // eigenvalues for symmetric pos-def 3x3
  p1 = Qij(0,1) * Qij(0,1) + \
       Qij(0,2) * Qij(0,2) + \
       Qij(1,2) * Qij(1,2);
  q = onethird * (Qij(0,0) + Qij(1,1) + Qij(2,2));
  p2 = (Qij(0,0) - q) * (Qij(0,0) - q) + \
       (Qij(1,1) - q) * (Qij(1,1) - q) + \
       (Qij(2,2) - q) * (Qij(2,2) - q) + 2.0*p1;
  p = std::sqrt(max(p2,0.0)/6.0);  
  //cout << "p1, q, p2, p: " << p1 << " " << q << " " << p2 << " " << p << endl; fflush(stdout);
  
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {       
      B(i,j) = Qij(i,j);
    }    
  }    
  for (int i = 0; i < dim; i++) {
     B(i,i) -= q;
  }    
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < dim; i++) {       
      B(i,j) *= (1.0/max(p,sml));
    }    
  }    
  detB = B(0,0) * (B(1,1)*B(2,2) - B(2,1)*B(1,2)) - \
         B(0,1) * (B(1,0)*B(2,2) - B(2,0)*B(1,2)) + \
         B(0,2) * (B(1,0)*B(2,1) - B(2,0)*B(1,1));
  r = 0.5*detB;
  //cout << "r: " << r << endl; fflush(stdout);
  
  if (r <= -1.0) {
    phi = onethird*pi;
  } else if (r >= 1.0) {
    phi = 0.0;
  } else {
    phi = onethird*acos(r);
  }

  // eigenvalues satisfy eig3 <= eig2 <= eig1 (L^4/T^2)
  ev[0] = q + 2.0 * p * cos(phi);
  ev[2] = q + 2.0 * p * cos(phi + (2.0*onethird*pi));
  ev[1] = 3.0 * q - ev[0] - ev[2];

  // actual sigma (L^2/T)
  sigma[0] = sqrt(max(ev[0],sml));
  sigma[1] = sqrt(max(ev[1],sml));
  sigma[2] = sqrt(max(ev[2],sml));
  //cout << "sigma: " << sigma[0] << " " << sigma[1] << " " << sigma[2] << endl; fflush(stdout);
  
  // eddy viscosity
  nu = sigma[2] * (sigma[0]-sigma[1]) * (sigma[1]-sigma[2]);
  nu = max(nu, 0.0);  
  nu /= (sigma[0]*sigma[0]); 
  nu *= (Cd*Cd);
  //cout << "mu: " << mu << endl; fflush(stdout);

  // shouldnt be necessary
  if (nu != nu) nu = 0.0;
  
}

/* 
void TurbModel::AddTKEDirichletBC(Coefficient *coeff, Array<int> &attr)
{

   temp_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Temperature Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i) {
         if (attr[i] == 1) {
	   mfem::out << i << " ";
	 }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((temp_ess_attr[i] && attr[i]) == 0,"Duplicate boundary definition deteceted.");
      if (attr[i] == 1) { temp_ess_attr[i] = 1; }
   }
}

void ThermoChem::AddTempDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddTempDirichletBC(new FunctionCoefficient(f), attr);
}

void ThermoChem::AddQtDirichletBC(Coefficient *coeff, Array<int> &attr)
{

   Qt_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Qt Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i) {
         if (attr[i] == 1) {
	   mfem::out << i << " ";
	 }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((Qt_ess_attr[i] && attr[i]) == 0,"Duplicate boundary definition deteceted.");
      if (attr[i] == 1) { Qt_ess_attr[i] = 1; }
   }
}

void ThermoChem::AddQtDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddQtDirichletBC(new FunctionCoefficient(f), attr);
}
*/

