//
// Created by Lazurite on 4/22/2022.
//

#ifndef PYBIND_CUDA_KALMANFILTER_H
#define PYBIND_CUDA_KALMANFILTER_H

#include <torch/extension.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include "Ndarray.h"

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename _Scalar, int _Rows, int _Cols>
using RowMatrix = Eigen::Matrix<_Scalar, _Rows, _Cols, Eigen::RowMajor>;

// Return multiple eigen in here is so bad
// You should write normal function and the edit the value using reference
// And in pybind wrapper, create the tuple which contains all the output value
// Then pass the reference to the function
// Also return the tuple in the wrapper function, not here!

class KalmanFilter {
    /*
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
     */
public:
    // Create Kalman filter model matrices.
    RowMatrix<double, 8, 8> _motion_mat;
    RowMatrix<double, 4, 8> _update_mat;
    double _std_weight_position = 1.0 / 20;
    double _std_weight_velocity = 1.0 / 160;

    KalmanFilter();

    void initiate(Eigen::Ref<Eigen::Vector4d> &measurement, Eigen::Matrix<double, 8, 1> &mean,
                  Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance);

    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
    initiate_py(Eigen::Ref<Eigen::Vector4d> measurement);

    void predict(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                 Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                 Eigen::Matrix<double, 8, 1> &mean_,
                 Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_);

    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
    predict_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
               Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance);

    void project(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                 Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                 Eigen::Matrix<double, 4, 1> &mean_,
                 Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &covariance_);

    std::tuple<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>
    project_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
               Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance);

    void multi_predict(Eigen::Ref<RowMatrix<double, -1, 8>> &mean, Ndarray &covariance, RowMatrix<double, -1, 8> &mean_,
                       Ndarray &covariance_);

    std::tuple<RowMatrix<double, -1, 8>, Ndarray>
    multi_predict_py(Eigen::Ref<RowMatrix<double, -1, 8>> mean, Ndarray &covariance);

    std::tuple<RowMatrix<double, -1, 8>, Ndarray>
    multi_predict_py(Eigen::Ref<RowMatrix<double, -1, 8>> mean, py::array_t<double> &covariance);

    void update(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                Eigen::Ref<Eigen::Vector4d> &measurement,
                Eigen::Matrix<double, 8, 1> &mean_,
                Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_);

    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
    update_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
              Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance,
              Eigen::Ref<Eigen::Vector4d> measurement);

    Eigen::VectorXd gating_distance(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
                                    Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance,
                                    Eigen::Ref<Eigen::Matrix<double, -1, 4, Eigen::RowMajor>> measurement,
                                    bool only_position = false,
                                    std::string metric = "maha");

    virtual ~KalmanFilter() = default;
};

extern void init_module_kalman_filter(py::module &m);

#endif // PYBIND_CUDA_KALMANFILTER_H
