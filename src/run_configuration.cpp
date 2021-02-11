#include "run_configuration.hpp"

#include <string>
#include <fstream>
#include <sstream>

RunConfiguration::RunConfiguration()
{
  // Initialize with default values
  timeIntegratorType = 4;
  solOrder = 4;
  
  integrationRule = 1;
  basisType = 1;
  
  cflNum = 0.12;
  numIters = 10;
  restart_cycle = 0;
  itersOut = 50;
  workFluid = DRY_AIR;
  eqSystem = EULER;
  visc_mult = 1.;
  SBP = false;
  for(int ii=0;ii<5;ii++) initRhoRhoVp[ii] = 0.;
}

RunConfiguration::~RunConfiguration()
{
  delete[] outputFile;
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
        
      // output file name
      }else if( word.compare("OUTPUT_NAME")==0 )
      {
        ss >> word;
        outputFile = new char[word.length() ];
        for(int i=0;i<(int)word.length(); i++) outputFile[i] = word[i];
        
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
        
      // output interval
      }else if( word.compare("ITERS_OUT")==0 )
      {
        ss >> word;
        itersOut = stoi(word);
        
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
       
      // CFL number
      }else if( word.compare("CFL")==0 )
      {
        ss >> word;
        cflNum = stof( word );
      
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
        }
        wallPatchType.Append( patchType );
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

