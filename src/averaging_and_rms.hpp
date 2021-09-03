#ifndef AVERAGING_AND_RMS
#define AVERAGING_AND_RMS

#include <string>
#include <tps_config.h>
#include <mfem.hpp>
#include "run_configuration.hpp"
#include "mpi_groups.hpp"

using namespace mfem;
using namespace std;

class Averaging
{
private:
  // Primitive variables
  ParGridFunction *Up;
  ParMesh *mesh;
  FiniteElementCollection *fec;
  ParFiniteElementSpace *fes;
  ParFiniteElementSpace *dfes;
  ParFiniteElementSpace *vfes;
  const int &num_equation;
  const int &dim;
  RunConfiguration &config;
  MPI_Groups *groupsMPI;
  
  
  // FES for RMS 
  ParFiniteElementSpace *rmsFes;
  int numRMS;
  
   // time averaged primitive variables
  ParGridFunction *meanUp;
  ParGridFunction *rms;
  
  // time averaged p, rho, vel (pointers to meanUp) for Visualization
  ParGridFunction *meanP, *meanRho, *meanV;
  
  ParaViewDataCollection *paraviewMean = NULL;
  
  int samplesMean;
  
  // iteration interval between samples
  int sampleInterval;
  int startMean;
  bool computeMean;
  
  void initiMeanAndRMS();

  void addSample_cpu();
  
  Vector local_sums;
  Vector tmp_vector;;
  
public:
  Averaging( ParGridFunction *_Up,
             ParMesh *_mesh,
             FiniteElementCollection *_fec,
             ParFiniteElementSpace *_fes,
             ParFiniteElementSpace *_dfes,
             ParFiniteElementSpace *_vfes,
             const int &_num_equation,
             const int &_dim,
             RunConfiguration &_config,
             MPI_Groups *_groupsMPI
  );
  ~Averaging();
  
  void addSampleMean(const int &iter);
  void write_meanANDrms_restart_files(const int &iter, const double &time);
  void read_meanANDrms_restart_files();
  
  int GetSamplesMean(){return samplesMean;}
  int GetSamplesInterval(){return sampleInterval;}
  bool ComputeMean(){return computeMean;}
  
  ParGridFunction *GetMeanUp(){return meanUp;}
  ParGridFunction *GetRMS(){return rms;}
  
  void SetSamplesMean(int &samples){samplesMean = samples;}
  void SetSamplesInterval(int &interval){sampleInterval = interval;}
  
  const double *getLocalSums();
  
  // GPU functions
#ifdef _GPU_
  static void addSample_gpu(ParGridFunction *meanUp,
                            ParGridFunction *rms,
                            int &samplesMean,
                            const ParGridFunction *Up,
                            const int &Ndof,
                            const int &dim,
                            const int &num_equation );
  static void sumValues_gpu(const Vector &meanUp,
                            const Vector &rms,
                            Vector &local_sums,
                            Vector &tmp_vector,
                            const int &num_equation,
                            const int &dim );
#endif // _GPU_
};

#endif // AVERAGING_AND_RMS
