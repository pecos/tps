#include "mpi_groups.hpp"

MPI_Groups::MPI_Groups(MPI_Session* _mpi):
mpi(_mpi)
{
  isInlet = 0;
  isOutlet = 0;
}

MPI_Groups::~MPI_Groups()
{
  MPI_Comm_free(&inlet_comm);
  MPI_Comm_free(&outlet_comm);
}


void MPI_Groups::init()
{
  // create inlet communicator
  MPI_Comm_split(MPI_COMM_WORLD, isInlet, mpi->WorldRank(), &inlet_comm);
  
  // create outlet communicator
  MPI_Comm_split(MPI_COMM_WORLD, isOutlet, mpi->WorldRank(), &outlet_comm);
  
}
