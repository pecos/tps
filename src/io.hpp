#ifndef IO_OPERATION
#define IO_OPERATION

#include <mfem.hpp>
#include <hdf5.h>
#include "utils.hpp"

#ifdef HAVE_GRVY
#include "grvy.h"
#endif

#include "mpi_groups.hpp"
#include "run_configuration.hpp"

using namespace mfem;
using namespace std;

class IO_operations
{
private:
  MPI_Groups *groupsMPI;
  RunConfiguration *config;
  int* locToGlobElem;
  Array<int> &partitioning_;
  const int &defaultPartMethod;
  int &num_equation;
  double &time;
  double &dt;
  int &iter;
  int &order;
  int &dim;
  
  bool loadFromAuxSol;
  
  int  nprocs_;
  int rank_;
  bool rank0_;

  Mesh *serial_mesh;
  FiniteElementSpace *serial_fes;
  GridFunction *serial_soln;
  
  ParGridFunction *U;
  ParGridFunction *meanUp;
  ParGridFunction *rms;
  
  ParMesh *mesh;
  ParFiniteElementSpace *vfes;
  const FiniteElementCollection *fec;

  string prefix_serial;
  
  int nelemGlobal_;


  void read_partitioned_soln_data(hid_t file, string varName, size_t index, double *data);
  void read_serialized_soln_data (hid_t file, string varName, int numDof,   int varOffset, double *data);
  void partitioning_file_hdf5(string mode);
  void serialize_soln_for_write();
public:
  IO_operations(string filePrefix,
                MPI_Groups *_groupMPI,
                RunConfiguration *_config,
                int* _locToGlobElem,
                Array<int> &_partitioning_,
                const int &_defaultPartMethod,
                int &_num_equation,
                double &_time,
                double &_dt,
                int &_iter,
                int &_order,
                int&_dim );
  ~IO_operations();

  void setSolutionGridFunction(ParGridFunction *_U);
  void setAveragesGridFunction(ParGridFunction *_meanUp, ParGridFunction *_rms);
  
  void restart_files_hdf5(string mode);
};

#endif // IO_OPERATION
