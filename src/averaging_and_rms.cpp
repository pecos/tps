#include "averaging_and_rms.hpp"
#include <general/forall.hpp>


Averaging::Averaging( ParGridFunction *_Up,
                      ParMesh *_mesh,
                      FiniteElementCollection *_fec,
                      ParFiniteElementSpace *_fes,
                      ParFiniteElementSpace *_dfes,
                      ParFiniteElementSpace *_vfes,
                      const int &_num_equation,
                      const int &_dim,
                      RunConfiguration &_config,
                      MPI_Groups *_groupsMPI
):
Up(_Up),
mesh(_mesh),
fec(_fec),
fes(_fes),
dfes(_dfes),
vfes(_vfes),
num_equation(_num_equation),
dim(_dim),
config(_config),
groupsMPI(_groupsMPI)
{
  // Always assume 6 components of the Reynolds stress tensor
  numRMS = 6;
  
  sampleInterval = config.GetMeanSampleInterval();
  samplesMean = 0;
  startMean = config.GetMeanStartIter();
  computeMean = false;
  if( sampleInterval!=0 ) computeMean = true;
  
  if( computeMean )
  {
    rmsFes = new ParFiniteElementSpace(mesh, fec, numRMS, Ordering::byNODES);
    
    meanUp  = new ParGridFunction(vfes);
    rms     = new ParGridFunction(rmsFes);
    
    meanRho = new ParGridFunction(fes, meanUp->GetData() );
    meanV   = new ParGridFunction(dfes, meanUp->GetData() +fes->GetNDofs() );
    meanP   = new ParGridFunction(fes, 
                    meanUp->GetData()+(num_equation-1)*fes->GetNDofs() );
    
    initiMeanAndRMS();
    
    // ParaviewMean 
    string name = "mean_";
    name.append( config.GetOutputName() );
    paraviewMean = new ParaViewDataCollection(name, mesh );
    paraviewMean->SetLevelsOfDetail( config.GetSolutionOrder() );
    paraviewMean->SetHighOrderOutput(true);
    paraviewMean->SetPrecision(8);
    
    paraviewMean->RegisterField("dens",meanRho);
    paraviewMean->RegisterField("vel",meanV);
    paraviewMean->RegisterField("press",meanP);
    paraviewMean->RegisterField("rms",rms);
  }
}


Averaging::~Averaging()
{
  if( computeMean )
  {
    delete paraviewMean;
    
    delete meanP;
    delete meanV;
    delete meanRho;
    
    delete rms;
    delete meanUp;
    
    delete rmsFes;
  }
}


void Averaging::addSampleMean(const int &iter)
{
  if( computeMean )
  {
    if( iter%sampleInterval==0 && iter>=startMean )
    {
      if( iter==startMean && groupsMPI->getSession()->Root()) cout<<"Starting mean calculation."<<endl;
#ifdef _GPU_
      addSample_gpu(meanUp, 
                    rms, 
                    samplesMean, 
                    Up, 
                    fes->GetNDofs(), 
                    dim,
                    num_equation);
#else
      addSample_cpu();
#endif
      samplesMean++;
    }
  }
}

void Averaging::addSample_cpu()
{
  double *dataUp = Up->GetData();
  double *dataMean = meanUp->GetData();
  double *dataRMS = rms->GetData();
  int dof = fes->GetNDofs();
  
  for(int n=0;n<dof;n++)
  {
    // mean
    for(int eq=0;eq<num_equation;eq++)
    {
      double mVal = double(samplesMean)*dataMean[n+eq*dof];
      dataMean[n+eq*dof] = (mVal+dataUp[n+eq*dof])/double(samplesMean+1);
    }

    // ----- RMS -----
    // velocities
    Vector meanVel(3), vel(3);
    meanVel = 0.; vel = 0.;
    for(int d=0;d<dim;d++)
    {
      meanVel[d] = dataMean[n+dof*(d+1)];
      vel[d]     = dataUp[n+dof*(d+1)];
    }
    
    // xx
    double val = dataRMS[n];
    dataRMS[n] = (val*double(samplesMean) + 
                  (vel[0]-meanVel[0])*(vel[0]-meanVel[0]))/double(samplesMean+1);
    // yy
    val = dataRMS[n +dof];
    dataRMS[n + dof] = (val*double(samplesMean) + 
                      (vel[1]-meanVel[1])*(vel[1]-meanVel[1]))/double(samplesMean+1);
    // zz
    val = dataRMS[n +2*dof];
    dataRMS[n +2*dof] = (val*double(samplesMean) + 
                        (vel[2]-meanVel[2])*(vel[2]-meanVel[2]))/double(samplesMean+1);
    // xy
    val = dataRMS[n +3*dof];
    dataRMS[n +3*dof] = (val*double(samplesMean) + 
                        (vel[0]-meanVel[0])*(vel[1]-meanVel[1]))/double(samplesMean+1);
    // xz
    val = dataRMS[n +4*dof];
    dataRMS[n +4*dof] = (val*double(samplesMean) + 
                        (vel[0]-meanVel[0])*(vel[2]-meanVel[2]))/double(samplesMean+1);
    // yz
    val = dataRMS[n +5*dof];
    dataRMS[n +5*dof] = (val*double(samplesMean) + 
                          (vel[1]-meanVel[1])*(vel[2]-meanVel[2]))/double(samplesMean+1);
  }
}


void Averaging::write_meanANDrms_restart_files()
{
  if( computeMean )
  {
    paraviewMean->Save();
    
//     string serialName = "restart_mean_";
//     serialName.append( config.GetOutputName() );
//     serialName.append( ".sol" );
//     
//     string serialNameRMS = "restart_rms_";
//     serialNameRMS.append( config.GetOutputName() );
//     serialNameRMS.append( ".sol" );
//     
//     string fileName = groupsMPI->getParallelName( serialName );
//     string rmsName  = groupsMPI->getParallelName( serialNameRMS );
//     ofstream file( fileName, std::ofstream::trunc );
//     ofstream rmsfile( rmsName, std::ofstream::trunc );
//     file.precision(8);
//     rmsfile.precision(8);
//     
//     double *data = meanUp->GetData();
//     double *dataRMS = rms->GetData();
//     
//     // write cycle and time
//     file<<samplesMean<<" "<<sampleInterval<<endl;
//     
//     
//     int dof = vfes->GetNDofs();
//     
//     for(int i=0;i<dof*num_equation;i++)
//     {
//       file << data[i] <<endl;
//     }
//     file.close();
//     
//     for(int i=0;i<dof*numRMS;i++)
//     {
//       rmsfile << dataRMS[i] <<endl;
//     }
//     rmsfile.close();
  }
}

void Averaging::read_meanANDrms_restart_files()
{
//   if( computeMean && config.GetRestartMean() )
//   {
//     string serialName = "restart_mean_";
//     serialName.append( config.GetOutputName() );
//     serialName.append( ".sol" );
//     
//     string serialNameRMS = "restart_rms_";
//     serialNameRMS.append( config.GetOutputName() );
//     serialNameRMS.append( ".sol" );
//     
//     string fileName = groupsMPI->getParallelName( serialName ); 
//     ifstream file( fileName );
//     
//     if( !file.is_open() )
//     {
//       cout<< "Could not open file \""<<fileName<<"\""<<endl;
//       return;
//     }else
//     {
//       double *dataUp = meanUp->GetData();
//       
//       string line;
//       // read time and iters
//       {
//         getline(file,line);
//         istringstream ss(line);
//         string word;
//         ss >> word;
//         samplesMean = stoi( word );
//         
//         ss >> word;
//         sampleInterval = stof( word );
//       }
//     
//       int lines = 0;
//       while( getline(file,line) )
//       {
//         istringstream ss(line);
//         string word;
//         ss >> word;
//         
//         dataUp[lines] = stof( word );
//         lines++;
//       }
//       file.close();
//     }
//     
//     // read RMS
//     string rmsName  = groupsMPI->getParallelName( serialNameRMS );
//     ifstream rmsfile( rmsName );
//     
//     if( !rmsfile.is_open() )
//     {
//       cout<< "Could not open file \""<<rmsName<<"\""<<endl;
//       return;
//     }else
//     {
//       double *dataRMS = rms->GetData();
//       
//       string line;
//     
//       int lines = 0;
//       while( getline(rmsfile,line) )
//       {
//         istringstream ss(line);
//         string word;
//         ss >> word;
//         
//         dataRMS[lines] = stof( word );
//         lines++;
//       }
//       rmsfile.close();
//     }
//   }
}

void Averaging::initiMeanAndRMS()
{
  double *dataMean = meanUp->GetData();
  double *dataRMS  = rms->GetData();
  int dof = vfes->GetNDofs();
  
  for(int i=0; i<dof; i++)
  {
    for(int eq=0;eq<num_equation;eq++) dataMean[i+eq*dof] = 0.;
    for(int n=0;n<numRMS;n++) dataRMS[i+dof*n] = 0.;
  }
}


#ifdef _GPU_
void Averaging::addSample_gpu(ParGridFunction* meanUp, 
                              ParGridFunction* rms, 
                              int& samplesMean, 
                              const ParGridFunction* Up, 
                              const int& Ndof, 
                              const int &dim,
                              const int& num_equation)
{
  double *d_meanUp = meanUp->ReadWrite();
  double *d_rms    = rms->ReadWrite();
  const double *d_Up = Up->Read();
  
  double dSamplesMean = (double)samplesMean;
  
  MFEM_FORALL_2D(n,Ndof,6,1,1,
  {
    MFEM_FOREACH_THREAD(i,x,6)
    {
      MFEM_SHARED double meanVel[3], vel[3];
      
      // mean
      for(int eq=i;eq<num_equation;eq+=6)
      {
        double mUpi = d_meanUp[n+eq*Ndof];
        double Upi = d_Up[n+eq*Ndof];
        
        double mVal = dSamplesMean*mUpi;
        double newMeanUp = (mVal+Upi)/(dSamplesMean+1);
        
        d_meanUp[n+eq*Ndof] = newMeanUp;
        
        // fill out mean velocity array
        if( i>0 && i<=3 )
        {
          meanVel[i-1] = newMeanUp;
          vel[i-1] = Upi;
        }
        if( i==3 && dim!=3 )
        {
          meanVel[2] = 0.;
          vel[2] = 0.;
        }
      }
      MFEM_SYNC_THREAD;
      
      // ----- RMS -----
      // xx
      if(i==0)
      {
        double val = d_rms[n];
        d_rms[n] = (val*dSamplesMean + 
                      (vel[0]-meanVel[0])*(vel[0]-meanVel[0]))/(dSamplesMean+1);
      }
      
      // yy
      if(i==1)
      {
        double val = d_rms[n +Ndof];
        d_rms[n + Ndof] = (val*dSamplesMean + 
                        (vel[1]-meanVel[1])*(vel[1]-meanVel[1]))/(dSamplesMean+1);
      }
    
      // zz
      if(i==2)
      {
        double val = d_rms[n +2*Ndof];
        d_rms[n +2*Ndof] = (val*dSamplesMean + 
                          (vel[2]-meanVel[2])*(vel[2]-meanVel[2]))/(dSamplesMean+1);
      }
    
      // xy
      if(i==3)
      {
        double val = d_rms[n +3*Ndof];
        d_rms[n +3*Ndof] = (val*dSamplesMean + 
                            (vel[0]-meanVel[0])*(vel[1]-meanVel[1]))/(dSamplesMean+1);
      }

      // xz
      if(i==4)
      {
        double val = d_rms[n +4*Ndof];
        d_rms[n +4*Ndof] = (val*dSamplesMean + 
                          (vel[0]-meanVel[0])*(vel[2]-meanVel[2]))/(dSamplesMean+1);
      }
    
      // yz
      if(i==5)
      {
        double val = d_rms[n +5*Ndof];
        d_rms[n +5*Ndof] = (val*dSamplesMean + 
                            (vel[1]-meanVel[1])*(vel[2]-meanVel[2]))/(dSamplesMean+1);
      }
    }
  });
}

#endif
