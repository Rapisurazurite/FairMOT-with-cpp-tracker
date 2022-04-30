//
// Created by Lazurite on 4/29/2022.
//

#include "Ndarray.h"

Ndarray::Ndarray(RowMatrixXd::Scalar *data_ptr, size_t N, size_t rows, size_t cols) : data_ptr(data_ptr), own_data(false) {
    shape = {N, rows, cols};
    for (size_t i = 0; i < N; i++) {
        ndarray.emplace_back(data_ptr + i * rows * cols,
                             rows,
                             cols
        );
    }
}

Ndarray::Ndarray(py::array_t<double> &input) : own_data(false) {
    if (input.ndim() != 3)
        throw std::runtime_error("Input array must be 3 dimensional.");
    py::buffer_info info = input.request();
    shape = {static_cast<unsigned long>(info.shape[0]),
             static_cast<unsigned long>(info.shape[1]),
             static_cast<unsigned long>(info.shape[2])};
    data_ptr = static_cast<RowMatrixXd::Scalar *>(info.ptr);
    for (size_t i = 0; i < shape[0]; i++) {
        ndarray.emplace_back(data_ptr + i * shape[1] * shape[2],
                             shape[1],
                             shape[2]
        );
    }
}

Ndarray::Ndarray(size_t N, size_t rows, size_t cols) : local_data(new double[N * rows * cols]), own_data(true) {
    shape = {N, rows, cols};
    for (size_t i = 0; i < N; i++) {
        ndarray.emplace_back(local_data.get() + i * rows * cols,
                             rows,
                             cols
        );
    }
}

std::string Ndarray::ptr() {
    std::stringstream ss;
    ss << (own_data ? local_data.get() : data_ptr);
    return ss.str();
}


void init_module_dnarray(py::module &m) {
    py::class_<Ndarray>(m, "Ndarray", py::buffer_protocol())
            .def(py::init([](py::buffer b) {
                typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;
                py::buffer_info info = b.request();
                if (info.ndim != 3)
                    throw std::runtime_error("Input array must be 3 dimensional.");
                if (info.format != py::format_descriptor<double>::format())
                    throw std::runtime_error("Input array must be double.");
                return Ndarray(
                        static_cast<RowMatrixXd::Scalar *>(info.ptr),
                        info.shape[0], info.shape[1], info.shape[2]);
            }))
            .def("create", [](size_t N, size_t rows, size_t cols) {
                return Ndarray(N, rows, cols);
            })
            .def_buffer([](Ndarray &ndarray) -> py::buffer_info {
                return py::buffer_info(
                        (ndarray.own_data ? ndarray.local_data.get() : ndarray.data_ptr),
                        sizeof(RowMatrixXd::Scalar),
                        py::format_descriptor<double>::format(),
                        3,
                        ndarray.shape,
                        {sizeof(RowMatrixXd::Scalar) * ndarray.shape[1] * ndarray.shape[2],
                         sizeof(RowMatrixXd::Scalar) * ndarray.shape[2],
                         sizeof(RowMatrixXd::Scalar)});
            })
            .def_readonly("shape", &Ndarray::shape)
            .def("ptr", &Ndarray::ptr)
            .def("__repr__", [](const Ndarray &ndarray) {
                std::stringstream ss;
                ss << "Ndarray object in pybind" << std::endl;
                ss << "own_data: " << ndarray.own_data << std::endl;
                ss << "address: " << (ndarray.own_data ? ndarray.local_data.get() : ndarray.data_ptr) << std::endl;
                ss << "shape: " << ndarray.shape[0] << " " << ndarray.shape[1] << " " << ndarray.shape[2] << std::endl;
                return ss.str();
            })
//            .def("__getitem__", [](Ndarray &ndarray, int i) {
//                return ndarray.ndarray[i];
//            }, py::return_value_policy::reference_internal)
            .def("print", [](const Ndarray &ndarray) {
                for (size_t i = 0; i < ndarray.shape[0]; i++) {
                    std::cout << "[" << ndarray.ndarray[i] << "]" << std::endl;
                }
            });
}