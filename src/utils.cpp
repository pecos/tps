#include <cstdio>
#include <string>
#include <cstdlib>
#include <array>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <sys/stat.h>
#include <filesystem>
#include <grvy.h>
#include "M2ulPhyS.hpp"

#define HAVE_SLURM

#ifdef HAVE_SLURM
#include <slurm/slurm.h>
#endif

// Run system command and capture output
std::string systemCmd(const char* cmd)
{
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// check if the amount of time left in SLURM job is less than desired threshold
bool slurm_job_almost_done(int threshold,int rank)
{
  char *SLURM_JOB_ID = getenv ("SLURM_JOB_ID");
  if (SLURM_JOB_ID == NULL)
    {
      printf("[ERROR]: SLURM_JOB_ID env variable not set. Unable to query how much time remaining\n");
      return false;
    } 

  long int secsRemaining;
  if( rank == 0 )  
    secsRemaining = slurm_get_rem_time(atoi(SLURM_JOB_ID));

  MPI_Bcast(&secsRemaining,1,MPI_LONG,0,MPI_COMM_WORLD);

  if( secsRemaining < 0 )
    {
      printf("[WARN]: Unable to query how much time left for current SLURM job\n");
      return false;
    }

  if (rank == 0)
    printf("  --> Seconds remaining for current job = %li (threshold = %i)\n",secsRemaining,threshold);
  
  if ( secsRemaining <= threshold )
    {
      if (rank == 0)
	{
	  string delim(80,'=');
	  cout << endl;
	  cout << delim << endl;
	  cout << "Remaining job time is less than " << threshold << " secs." << endl;
	  cout << "--> Dumping final restart and terminating with exit_code = " << JOB_RESTART << endl;
	  cout << "--> Use this restart exit code in your job scripts to auto-trigger a resubmit. " << endl;
	  cout << delim << endl;
	}
    return true;
    }
  else
    return false;
}

// submit a new resource manager job with provided job script
int rm_restart(std::string jobFile,std::string mode)
{

  //grvy_printf(GRVY_INFO,"[RMS] Submitting fresh %s job using script -> %s\n",mode.c_str(),jobFile.c_str());
  std::cout << "[RMS] Submitting fresh" << mode << " job using script -> " << jobFile << std::endl;

  if ( mode == "SLURM" )
    {
      std::string command = "sbatch " + jobFile;
      std::string result = systemCmd(command.c_str());
      std::cout << result << std::endl;
    }

  return 0;
}

// check if file exists
bool file_exists (const std::string &name)
{
  return ( access( name.c_str(), F_OK ) != -1 );
}

bool M2ulPhyS::Check_JobResubmit()
{
  if ( config.isAutoRestart() )
    {
      if ( slurm_job_almost_done(config.rm_threshold(),mpi.WorldRank()) )
	return true;
      else
	return false;
    }
  else
    return false;
}

// Look for existence of output.pvd files in vis output directory and
// keep a sequentially numbered copy. Useful when doing restart,
// otherwise, MFEM ParaviewDataCollection output will delete old
// timesteps from the file on restarts.
void M2ulPhyS::Cache_Paraview_Timesteps()
{
  const int MAX_FILES = 9999;
  char suffix[5];

  std::string fileName = config.GetOutputName() + "/output.pvd";
  if ( file_exists (fileName ) )
    {
      for(int i=1;i<=MAX_FILES;i++)
	{
	  sprintf(suffix,"%04i",i);
	  std::string testFile = fileName + "." + suffix;
	  if ( file_exists(testFile) )
	    continue;
	  else
	    {
	      std::cout << "--> caching output.pvd file -> " << testFile << std::endl;
	      std::filesystem::copy_file(fileName,testFile);
	      return;
	    }
	}
    }

  return;
}


