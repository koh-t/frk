#include <pybind11/pybind11.h>

#include "kernel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfrk, m) {
    m.doc() = "Frechet Kernels";

    py::class_<frk::MatrixF64>(m, "MatrixF64")
        .def(py::init<size_t, size_t>())
        .def("get", &frk::MatrixF64::get)
        .def("set", &frk::MatrixF64::set)
        .def("nrows", &frk::MatrixF64::nrows)
        .def("ncols", &frk::MatrixF64::ncols);

    py::class_<frk::TopkFrechet>(m, "TopkFrechet")  //
        .def(py::init<const frk::MatrixF64&>())
        .def("next", &frk::TopkFrechet::next, "Next computation");

    py::class_<frk::Kernel>(m, "Kernel")  //
        .def(py::init<size_t>())
        .def("compute", &frk::Kernel::compute, "Computation",  //
             py::arg("emat"), py::arg("nsamples"), py::arg("beta"), py::arg("diag_wgt"), py::arg("seed"))
        .def("compute_with_topk", &frk::Kernel::compute_with_topk, "Computation with TopK",  //
             py::arg("emat"), py::arg("tf"), py::arg("nsamples"),  //
             py::arg("beta"), py::arg("diag_wgt"), py::arg("seed"));
}
