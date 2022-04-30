#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>

void Greet() { std::cout << "Hello Pytorch from C++" << std::endl; }


void init_module_cpp_extension(pybind11::module &m);
void init_module_kalman_filter(pybind11::module &m);
void init_module_dnarray(py::module &m);
void init_module_basetrack(py::module &m);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greet", &Greet);

    /*
     * Register the C++ functions in modules directory
     */
    init_module_cpp_extension(m);

    init_module_dnarray(m);
    init_module_kalman_filter(m);
//    init_module_basetrack(m);
}