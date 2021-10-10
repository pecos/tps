// -----------------------------------------------------------------------------------bl-
// -----------------------------------------------------------------------------------el-
#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "M2ulPhyS.hpp"
#include <unistd.h>

int main(int argc, char *argv[])
{
  MPI_Session mpi(argc, argv);
  
  const char *inputFile = "../data/periodic-square.mesh";
  //const char *device_config = "cpu";

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&inputFile, "-run", "--runFile",
                "Name of the input file with run options.");
  //  args.AddOption(&device_config, "-d", "--device",
  // "Device configuration string, see Device::Configure().");

  // device_config inferred from build setup
  std::string device_config = "cpu";
#ifdef _HIP_
  device_config = "hip";
#elif _CUDA_
  device_config = "cuda";
#endif

#ifdef DEBUG 
  int threads = 0.;
  args.AddOption(&threads, "-thr", "--threads",
                " Set -thr 1 so that the program stops at the \
 beginning in debug mode for gdb attach.");
#endif

  args.Parse();
  if (!args.Good())
  {
    if(mpi.Root() ) args.PrintUsage(cout);
    return 1;
  }
  if(mpi.Root() ) args.PrintOptions(cout);
  
#ifdef DEBUG 
  if( threads != 0 )
  {
    int gdb = 0;
    cout<<"Process "<<mpi.WorldRank()+1<<"/"<<mpi.WorldSize()<<", id: "<<getpid()<<endl;
    while( gdb==0 )
        sleep(5);
  }
#endif

  const int NUM_GPUS_NODE = 4;
  Device device(device_config, mpi.WorldRank()%NUM_GPUS_NODE);
  if (mpi.Root() )  device.Print();


  string inputFileName(inputFile);
  M2ulPhyS solver( mpi, inputFileName );


  // Start the timer.
  tic_toc.Clear();
  tic_toc.Start();

  solver.Iterate();

  tic_toc.Stop();
  if(mpi.Root() ) cout << " done, " << tic_toc.RealTime() << "s." << endl;

  return(solver.GetStatus());
}
