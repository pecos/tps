#ifndef MPI_COMM_MANAGER
#define MPI_COMM_MANAGER

#include <mpi.h>
#include "mfem.hpp"

using namespace mfem;

class MPI_Groups
{
private:
  MPI_Session *mpi;
  
  MPI_Comm inlet_comm;
  MPI_Comm outlet_comm;
  
  int isOutlet;
  int isInlet;
  
  
public:
  MPI_Groups(MPI_Session *mpi);
  ~MPI_Groups();
  
  // function that creates the groups 
  void init();
  
  void setAsInlet(){isInlet = 1;};
  void setAsOutlet(){isOutlet = 2;};
  
  MPI_Session* getSession(){return mpi;}
  
  MPI_Comm getInletComm(){return inlet_comm;}
  MPI_Comm getOutletComm(){return outlet_comm;}
};

#endif // MPI_COMM_MANAGER
