#include "run_configuration.hpp"

#include <string>
#include <fstream>
#include <sstream>

RunConfiguration::RunConfiguration()
{
  // Initialize with default values
  timeIntegratorType = 4;
  solOrder = 4;
  auxOrder = 0;
  
  integrationRule = 1;
  basisType = 1;
  
  cflNum = 0.12;
  constantTimeStep = false;
  numIters = 10;
  useRoe = false;
  restart_cycle = 0;
  restartFromAux = false;
  
  sampleInterval = 0;
  startIter = 0;
  restartMean = false;
  
  itersOut = 50;
  workFluid = DRY_AIR;
  eqSystem = EULER;
  visc_mult = 1.;
  refLength = 1.;
  SBP = false;
  for(int ii=0;ii<5;ii++) initRhoRhoVp[ii] = 0.;
  
  isForcing = false;
  for(int ii=0;ii<3;ii++) gradPress[ii] = 0.;
}

RunConfiguration::~RunConfiguration()
{
}



void RunConfiguration::readInputFile(std::string inpuFileName)
{
  // runfile
  ifstream runFile( inpuFileName );
  
  if( !runFile.is_open() )
  {
    cout<< "Could not open runfile"<<endl;
    return;
  }
  
  string line;
  
  while( getline(runFile,line) )
  {
    istringstream ss(line);
    // check if it is a commented line
    if( !(line.c_str()[0]=='#') && line.size()>0 )
    {
      string word;
      ss >> word;
      
      // mesh file
      if( word.compare("MESH")==0 )
      {
        ss >> word;
        meshFile.clear();
        meshFile = word;
        
//       // linear mesh file
//       }else if( word.compare("LIN_MESH")==0 )
//       {
//         ss >> word;
//         lin_meshFile.clear();
//         lin_meshFile = word;
        
      // output file name
      }else if( word.compare("OUTPUT_NAME")==0 )
      {
        ss >> word;
        outputFile.clear();
        outputFile = word;
        
      // max num. of iters
      }else if( word.compare("NMAX")==0 )
      {
        ss >> word;
        numIters = stoi(word);
        
      // cycles at restart
      }else if( word.compare("RESTART_CYCLE")==0 )
      {
        ss >> word;
        restart_cycle = stoi(word);
        
      // restart from aux. sol
      }else if( word.compare("RESTART_FROM_AUX")==0 )
      {
        restartFromAux = true;
        
        // order of auxiliary solution
      }else if( word.compare("AUX_ORDER")==0 )
      {
        ss >> word;
        auxOrder = stoi( word );
        
      // mean and RMS calculation
      }else if( word.compare("CALC_MEAN_RMS")==0 )
      {
        ss >> word;
        startIter = stoi( word );
        
        ss >> word;
        sampleInterval = stoi( word );
        
      }else if( word.compare("CONTINUE_MEAN_CALC")==0 )
      {
        restartMean = true;
        
      // output interval
      }else if( word.compare("ITERS_OUT")==0 )
      {
        ss >> word;
        itersOut = stoi(word);
        
      // Riemann solver
      }else if( word.compare("USE_ROE")==0 )
      {
        ss >> word;
        if( stoi(word)==1 ) useRoe = true;
        
      // solution order
      }else if( word.compare("POL_ORDER")==0 )
      {
        ss >> word;
        solOrder = stoi(word);
        
      // Integration rule
      }else if( word.compare("INT_RULE")==0 )
      {
        ss >> word;
        integrationRule = stoi(word);
       
      // Basis type
      }else if( word.compare("BASIS_TYPE")==0 )
      {
        ss >> word;
        basisType = stoi(word);
       
      // CFL number
      }else if( word.compare("CFL")==0 )
      {
        ss >> word;
        cflNum = stof( word );
       
      // constant time-step
      }else if( word.compare("DT_CONSTANT")==0 )
      {
        constantTimeStep = true;
        
      // time integrator
      }else if( word.compare("TIME_INTEGRATOR")==0 )
      {
        ss >> word;
        timeIntegratorType = stoi(word);
        
      // working fluid
      }else if( word.compare("FLUID")==0 )
      {
        ss >> word;
        int typeFluid = stoi(word);
        switch( typeFluid )
        {
          case 0:
            workFluid = DRY_AIR;
            break;
          default:
            break;
        }
      }else if( word.compare("EQ_SYSTEM")==0 )
      {
        ss >> word;
        int systemType = stoi(word);
        switch( systemType )
        {
          case 0:
            eqSystem = EULER;
            break;
          case 1:
            eqSystem = NS;
            break;
          case 2:
            eqSystem = MHD;
            break;
          default:
            break;
        }
      }else if( word.compare("IS_SBP")==0 )
      {
        ss >> word;
        int isSBP = stoi( word );
        if( isSBP==1 ) SBP = true;
        
      // Initial solution RHO
      }else if( word.compare("INIT_RHO")==0 )
      {
        ss >> word;
        initRhoRhoVp[0] = stof( word );
        
      // Initial solution RHOVX
      }else if( word.compare("INIT_RHOVX")==0 )
      {
        ss >> word;
        initRhoRhoVp[1] = stof( word );
        
      // Initial solution RHOVY
      }else if( word.compare("INIT_RHOVY")==0 )
      {
        ss >> word;
        initRhoRhoVp[2] = stof( word );
        
      // Initial solution RHOVZ
      }else if( word.compare("INIT_RHOVZ")==0 )
      {
        ss >> word;
        initRhoRhoVp[3] = stof( word );
        
      // Initial solution P
      }else if( word.compare("INIT_P")==0 )
      {
        ss >> word;
        initRhoRhoVp[4] = stof( word );
        
      // viscity factor
      }else if( word.compare("VISC_MULT")==0 )
      {
        ss >> word;
        visc_mult = stof( word );
        
      // Reference length
      }else if( word.compare("REF_LENGTH")==0 )
      {
        ss >> word;
        refLength = stof( word );
        
      }else if( word.compare("GRAD_PRESSURE")==0 )
      {
        ss >> word;
        gradPress[0] = stof( word );
        
        ss >> word;
        gradPress[1] = stof( word );
        
        ss >> word;
        gradPress[2] = stof( word );
        isForcing = true;
        
      }else if( word.compare("INLET")==0 )
      {
        ss >> word;
        std::pair<int,InletType> patchANDtype;
        patchANDtype.first = stoi(word) ;
        
        ss >> word;
        int typ = stoi( word );
        switch(typ)
        {
          case 0:
            patchANDtype.second = SUB_DENS_VEL;
            break;
          case 1:
            patchANDtype.second = SUB_DENS_VEL_NR;
            break;
          case 2:
            patchANDtype.second = SUB_VEL_CONST_ENT;
            break;
        }
        inletPatchType.Append( patchANDtype );
        
        for(int i=0; i<4; i++)
        {
          ss >> word;
          inletBC.Append( stof( word ) );
        }
        
      }else if( word.compare("OUTLET")==0 )
      {
        ss >> word;
        std::pair<int,OutletType> patchANDtype;
        patchANDtype.first = stoi(word) ;
        
        ss >> word;
        int typ = stoi( word );
        switch(typ)
        {
          case 0:
            patchANDtype.second = SUB_P;
            break;
          case 1:
            patchANDtype.second = SUB_P_NR;
            break;
          case 2:
            patchANDtype.second = SUB_MF_NR;
            break;
        }
        outletPatchType.Append( patchANDtype );
        
        ss >> word;
        outletBC.Append (stof( word ) );
        
      }else if( word.compare("WALL")==0 )
      {
        ss >> word;
        std::pair<int,WallType> patchType;
        patchType.first = stoi( word );
        ss >> word;
        switch( stoi(word) )
        {
          case 0:
            patchType.second = INV;
            break;
          case 1:
            patchType.second = VISC_ADIAB;
            break;
          case 2:
            patchType.second = VISC_ISOTH;
            break;
        }
        wallPatchType.Append( patchType );
        if( patchType.second==VISC_ISOTH )
        {
          ss >> word;
          wallBC.Append( stof(word) );
        }else
        {
          wallBC.Append( 0. );
        }
      }
      
    }
  }
  
  runFile.close();
}

Array<double> RunConfiguration::GetInletData(int i)
{
  Array<double> data(4);
  for(int j=0;j<4;j++) data[j] = inletBC[j+4*i];
  
  return data;
}

Array<double> RunConfiguration::GetOutletData(int out)
{
  Array<double> data(1);
  data[0] = outletBC[out];
  
  return data;
}

Array<double> RunConfiguration::GetWallData(int w)
{
  Array<double> data(1);
  data[0] = wallBC[w];
  
  return data;
}

