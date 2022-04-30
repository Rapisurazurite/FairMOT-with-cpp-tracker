//
// Created by Lazurite on 4/29/2022.
//

#ifndef TRACKLET_NDARRAY_H
#define TRACKLET_NDARRAY_H

#include <torch/extension.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>


namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename _Scalar, int _Rows, int _Cols>
using RowMatrix = Eigen::Matrix<_Scalar, _Rows, _Cols, Eigen::RowMajor>;

class Ndarray {
public:
    std::unique_ptr<double> local_data;
    // data ptr from numpy array
    RowMatrixXd::Scalar *data_ptr = nullptr;
    bool own_data;

    std::vector<Eigen::Map<RowMatrixXd>> ndarray;
    std::vector<size_t> shape;

    Ndarray(RowMatrixXd::Scalar *data_ptr, size_t N, size_t rows, size_t cols);

    Ndarray(py::array_t<double> &input);

    Ndarray(size_t N, size_t rows, size_t cols);

    std::string ptr();
};

extern void init_module_dnarray(py::module &m);

#endif //TRACKLET_NDARRAY_H
