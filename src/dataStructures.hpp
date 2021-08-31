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

struct volumeFaceIntegrationArrays
{
  // nodes IDs and indirection array
  Array<int> nodesIDs;
  Array<int> posDofIds;
  // count of number of elements of each type
  Array<int> numElems;
  //Array<int> posDofQshape1; // position, num. dof and integration points for each face
  Vector shapeWnor1; // shape functions, weight and normal for each face at ach integration point
  Vector shape2;

  Array<int> elemFaces; // number and faces IDs of each element
  Array<int> elems12Q; // elements connecting a face and integration points

  // used in gradient computation
  Vector elemShapeDshapeWJ; // [...l_0(i),...,l_dof(i),l_0_x(i),...,l_dof_d(i), w_i*detJac_i ...]
  Array<int> elemPosQ_shapeDshapeWJ; // position and num. of integration points for each element
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
  MPI_Request *requests;
  MPI_Status  *statuses;
};

#endif
