//
// Created by Lazurite on 4/29/2022.
//

#ifndef TRACKLET_BASETRACK_H
#define TRACKLET_BASETRACK_H


#include <torch/extension.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include "KalmanFilter.h"

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename _Scalar, int _Rows, int _Cols>
using RowMatrix = Eigen::Matrix<_Scalar, _Rows, _Cols, Eigen::RowMajor>;

namespace py = pybind11;

extern void init_module_basetrack(py::module &m);

#endif //TRACKLET_BASETRACK_H
