
#include <string>
#include <fstream>
#include <sstream>

#include "run_configuration.hpp"

using namespace std;

RunConfiguration::RunConfiguration()
{
  // Initialize with default values
  timeIntegratorType = 4;
  solOrder = 4;
  
  integrationRule = 1;
  basisType = 1;
  
  cflNum = 0.12;
  numIters = 10;
  workFluid = DRY_AIR;
  eqSystem = EULER;
  SBP = false;
  for(int ii=0;ii<5;ii++) initRhoRhoVp[ii] = 0.;
}

RunConfiguration::~RunConfiguration()
{
  delete[] meshFile;
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
        meshFile = new char[word.length() ];
        for(int i=0;i<(int)word.length(); i++) meshFile[i] = word[i];
        
      // max num. of iters
      }else if( word.compare("NMAX")==0 )
      {
        ss >> word;
        numIters = stoi(word);
        
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
      }
      
    }
  }
}
