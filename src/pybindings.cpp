#include <sys/types.h>
#include <tps_config.h>
#include <unistd.h>

#ifdef HAVE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef HAVE_MPI4PY
#include <mpi4py/mpi4py.h>
#endif

namespace py = pybind11;

namespace tps_wrappers {
void tps(py::module& m);
void tps2bolzmann(py::module& m);
}  // namespace tps_wrappers

PYBIND11_MODULE(libtps, m) {
  m.doc() = "TPS Python Interface";

  tps_wrappers::tps(m);
  tps_wrappers::tps2bolzmann(m);
}

#endif