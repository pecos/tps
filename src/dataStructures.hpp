#ifndef STRUCTURES
#define STRUCTURES

#include <mfem.hpp>

using namespace mfem;

// The following four keywords define two planes in which
// a linearly varying viscosity can be defined between these two.
// The planes are defined by the normal and one point being the 
// normal equal for both planes. The visc ratio will vary linearly
// from 0 to viscRatio from the plane defined by pointInit and 
// the plane defined by point0.
struct linearlyVaryingVisc
{
  Vector normal;
  Vector point0;
  Vector pointInit;
  double viscRatio;
};

struct parallelFacesIntegrationArrays
{
  Vector sharedShapeWnor1;
  Vector sharedShape2;
  Array<int> sharedElem1Dof12Q;
  Array<int> sharedVdofs;
  Array<int> sharedVdofsGradUp;
  Array<int> sharedElemsFaces;
};

struct dataTransferArrays
{
  // vectors for data transfer
  Vector face_nbr_data;
  Vector send_data;
  
  // MPI communicators data
  int num_face_nbrs;
  MPI_Request *send_requests;
  MPI_Request *recv_requests;
  MPI_Status  *statuses;
};

#endif
