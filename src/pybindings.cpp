#include <sys/types.h>
#include <unistd.h>
#include <tps_config.h>

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
}


PYBIND11_MODULE(libtps, m) {
#ifdef HAVE_MPI4PY
  // initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    // mpi4py calls the Python C API
    // we let pybind11 give us the detailed traceback
    throw py::error_already_set();
  }
#endif

  m.doc() = "TPS Python Interface";

  tps_wrappers::tps(m);
  tps_wrappers::tps2bolzmann(m);

}

#endif