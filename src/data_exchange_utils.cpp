#include "data_exchange_utils.hpp"

#ifdef HAVE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tps_wrappers {
void data_exchange_utils(py::module &m) {
  // Can by read in numpy as np.array(data_instance, copy = False)
  py::class_<TPS::CPUDataRead>(m, "CPUDataRead", py::buffer_protocol())
      .def_buffer([](TPS::CPUDataRead &d) -> py::buffer_info {
        return py::buffer_info(d.data(),                                /*pointer to buffer*/
                               sizeof(double),                          /*size of one element*/
                               py::format_descriptor<double>::format(), /*python struct-style format descriptor*/
                               1,                                       /*number of dimensions*/
                               {d.size()},                              /*buffer dimension(s)*/
                               {d.stride() * sizeof(const double)},     /*Stride in bytes for each index*/
                               true /*read only*/);
      });

  py::class_<TPS::CPUData>(m, "CPUData", py::buffer_protocol()).def_buffer([](TPS::CPUData &d) -> py::buffer_info {
    return py::buffer_info(d.data(),                                /*pointer to buffer*/
                           sizeof(double),                          /*size of one element*/
                           py::format_descriptor<double>::format(), /*python struct-style format descriptor*/
                           1,                                       /*number of dimensions*/
                           {d.size()},                              /*buffer dimension(s)*/
                           {d.stride() * sizeof(const double)},     /*Stride in bytes for each index*/
                           false /*writetable*/);
  });
}
}

#endif