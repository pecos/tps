#ifndef STRUCTURES
#define STRUCTURES

#include <mfem.hpp>

using namespace mfem;

// application exit codes
enum ExitCodes {NORMAL = 0, ERROR = 1, JOB_RESTART = 10 };

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

#endif
