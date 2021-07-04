#ifndef MPI_COMM_MANAGER
#define MPI_COMM_MANAGER

#include <mpi.h>
#include "mfem.hpp"

#include <string>

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

  bool isGroupRoot(MPI_Comm);      // check if rank 0 on local group
  int groupSize(MPI_Comm);         // size of BC MPI group
  
  // function that creates the groups 
  void init();
  
  void setAsInlet(int patchNum ){isInlet = patchNum +1000;}
  void setAsOutlet(int patchNum ){isOutlet = patchNum +2000;}
  
  MPI_Session* getSession(){return mpi;}
  
  MPI_Comm getInletComm(){return inlet_comm;}
  MPI_Comm getOutletComm(){return outlet_comm;}
  
  std::string getParallelName(std::string name );
};

#endif // MPI_COMM_MANAGER
