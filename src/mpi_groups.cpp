#include "mpi_groups.hpp"
#include <vector>

using namespace std;

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


string MPI_Groups::getParallelName(string serialName)
{
  if( mpi->WorldSize()==1 ) // serial name
  {
    return serialName;
  }else
  {
    // find the dots
    vector<size_t> dots;
    size_t lastPoint = serialName.find(".");
    dots.push_back( lastPoint );
    size_t newPoint = 0;
    while( newPoint!=string::npos )
    {
      newPoint = serialName.find(".",lastPoint+1);
      if( newPoint!=string::npos )
      {
        lastPoint = newPoint;
        dots.push_back( lastPoint );
      }
    }
    
    string baseName = serialName.substr(0,lastPoint);
    string extension = serialName.substr(lastPoint+1,serialName.length()-1);
    
    string parallelName = baseName + "." + to_string(mpi->WorldRank()) + "." + extension;
    return parallelName;
  }
}
