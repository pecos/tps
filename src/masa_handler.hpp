#ifndef _MASA_HANDLER_
#define _MASA_HANDLER_

#include "masa.h"
#include "mfem.hpp"

#include "fluxes.hpp" // for enum Equations

using namespace mfem;
using namespace std;

void initMasaHandler(string name, int dim, const Equations& eqn);

#endif
