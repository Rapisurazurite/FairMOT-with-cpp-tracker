//
// Created by Lazurite on 4/22/2022.
//
#include "KalmanFilter.h"
#include <Eigen/Dense>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

/*
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
*/

double chi2inv95[10] = {
        0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};

void KalmanFilter::initiate(Eigen::Ref<Eigen::Vector4d> &measurement, Eigen::Matrix<double, 8, 1> &mean,
                            Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance) {
    /*Create track from unassociated measurement.

    Parameters
    ----------
    measurement : ndarray
        Bounding box coordinates (x, y, a, h) with center position (x, y),
        aspect ratio a, and height h.

    Returns
    -------
    (ndarray, ndarray)
        Returns the mean vector (8 dimensional) and covariance matrix (8x8
        dimensional) of the new track. Unobserved velocities are initialized
        to 0 mean.
     */
    mean << measurement, Eigen::Vector4d::Zero();
    Eigen::Matrix<double, 8, 1> std;
    std << 2 * _std_weight_position * measurement(3),
            2 * _std_weight_position * measurement(3),
            1e-2,
            2 * _std_weight_position * measurement(3),
            10 * _std_weight_velocity * measurement(3),
            10 * _std_weight_velocity * measurement(3),
            1e-5,
            10 * _std_weight_velocity * measurement(3);
    std = std.array().square();
    covariance = std.asDiagonal();
}

std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
KalmanFilter::initiate_py(Eigen::Ref<Eigen::Vector4d> measurement) {
    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> result;

    Eigen::Matrix<double, 8, 1> &mean = std::get<0>(result);
    Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance = std::get<1>(result);

    initiate(measurement, mean, covariance);
    return result;
}

void KalmanFilter::predict(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                           Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                           Eigen::Matrix<double, 8, 1> &mean_,
                           Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_) {
    /*Run Kalman filter prediction step.

    Parameters
    ----------
    mean : ndarray
        The 8 dimensional mean vector of the object state at the previous
        time step.
    covariance : ndarray
        The 8x8 dimensional covariance matrix of the object state at the
        previous time step.

    Returns
    -------
    (ndarray, ndarray)
        Returns the mean vector and covariance matrix of the predicted
        state. Unobserved velocities are initialized to 0 mean.
     */
    Eigen::Vector4d std_pos{
            _std_weight_position * mean(3),
            _std_weight_position * mean(3),
            1e-2,
            _std_weight_position * mean(3)};
    Eigen::Vector4d std_vel{
            _std_weight_velocity * mean(3),
            _std_weight_velocity * mean(3),
            1e-5,
            _std_weight_velocity * mean(3)};
    Eigen::Matrix<double, 8, 1> std_pos_vel;
    std_pos_vel << std_pos, std_vel;
    RowMatrix<double, 8, 8> motion_cov = std_pos_vel.array().square().matrix().asDiagonal();

    mean_ = _motion_mat * mean;
    covariance_ = _motion_mat * covariance * _motion_mat.transpose() + motion_cov;
}

std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
KalmanFilter::predict_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
                         Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance) {
    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> result;
    Eigen::Matrix<double, 8, 1> &mean_ = std::get<0>(result);
    Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_ = std::get<1>(result);

    predict(mean, covariance, mean_, covariance_);

    return result;
}

void KalmanFilter::project(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                           Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                           Eigen::Matrix<double, 4, 1> &mean_,
                           Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &covariance_) {
    /*Project state distribution to measurement space.

    Parameters
    ----------
    mean : ndarray
        The mean vector of the state distribution.
    covariance : ndarray
        The covariance matrix of the state distribution.

    Returns
    -------
    (ndarray, ndarray)
        Returns the projected mean and covariance matrix of the given state
        distribution.
     */

    Eigen::Vector4d std{
            _std_weight_position * mean(3),
            _std_weight_position * mean(3),
            1e-1,
            _std_weight_position * mean(3)};
    Eigen::Matrix<double, 4, 4> innovation_cov = std.array().square().matrix().asDiagonal();
    mean_ = _update_mat * mean;
    covariance_ = _update_mat * covariance * _update_mat.transpose() + innovation_cov;
}

std::tuple<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>
KalmanFilter::project_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
                         Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance) {

    std::tuple<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> result;
    Eigen::Matrix<double, 4, 1> &mean_ = std::get<0>(result);
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &covariance_ = std::get<1>(result);

    project(mean, covariance, mean_, covariance_);

    return result;
}

void KalmanFilter::multi_predict(Eigen::Ref<RowMatrix<double, -1, 8>> &mean, Ndarray &covariance,
                                 RowMatrix<double, -1, 8> &mean_, Ndarray &covariance_) {
    /*Run Kalman filter prediction step (Vectorized version).
    Parameters
    ----------
    mean : ndarray
        The Nx8 dimensional mean matrix of the object states at the previous
        time step.
    covariance : ndarray
        The Nx8x8 dimensional covariance matrics of the object states at the
        previous time step.
    Returns
    -------
    (ndarray, ndarray)
        Returns the mean vector and covariance matrix of the predicted
        state. Unobserved velocities are initialized to 0 mean.
     */
    int N = mean.rows();

    RowMatrix<double, 8, -1> std_pos = RowMatrixXd::Zero(8, N);
    Eigen::Matrix<double, 8, 1> std_pos_std = Eigen::Matrix<double, 8, 1>::Zero();
    std_pos_std << _std_weight_position,
            _std_weight_position,
            1e-2,
            _std_weight_position,
            _std_weight_velocity, _std_weight_velocity, 1e-5, _std_weight_velocity;
    std_pos = std_pos_std * mean.col(3).transpose();
    std_pos.row(2) = RowMatrixXd::Constant(1, N, 1e-2);
    std_pos.row(6) = RowMatrixXd::Constant(1, N, 1e-5);

    RowMatrix<double, -1, 8> sqr = std_pos.array().square().matrix().transpose();
    //        std::vector<RowMatrix<double, 8, 8>> motion_cov(N, RowMatrix<double, 8, 8>::Zero());
    //        for (int i = 0; i < N; i++) {
    //            motion_cov[i] = sqr.row(i).asDiagonal();
    //        }

    mean_ = mean * _motion_mat.transpose();

    for (int i = 0; i < N; i++) {
        RowMatrix<double, 8, 8> motion_cov = sqr.row(i).asDiagonal();
        covariance_.ndarray[i] = _motion_mat * covariance.ndarray[i] * _motion_mat.transpose() + motion_cov;
    }
}

std::tuple<RowMatrix<double, -1, 8>, Ndarray>
KalmanFilter::multi_predict_py(Eigen::Ref<RowMatrix<double, -1, 8>> mean, Ndarray &covariance) {

    //        std::cout << "Address of mean: " << &mean(0, 0) << std::endl;
    //        std::cout << "Address of covariance: " << covariance.ptr() << std::endl;
    int N = mean.rows();
    std::tuple<RowMatrix<double, -1, 8>, Ndarray> result(
            RowMatrix<double, -1, 8>(N, 8),
            Ndarray(N, 8, 8));

    auto &mean_ = std::get<0>(result);
    auto &covariance_ = std::get<1>(result);

    multi_predict(mean, covariance, mean_, covariance_);
    return result;
}

KalmanFilter::KalmanFilter() {
    int ndim = 4;
    double dt = 1;
    _motion_mat = RowMatrixXd::Identity(2 * ndim, 2 * ndim);
    for (int i = 0; i < ndim; i++) {
        _motion_mat(i, ndim + i) = dt;
    }
    _update_mat = RowMatrixXd::Identity(ndim, 2 * ndim);
    //# Motion and observation uncertainty are chosen relative to the current
    //# state estimate. These weights control the amount of uncertainty in
    //# the model. This is a bit hacky.
}

std::tuple<RowMatrix<double, -1, 8>, Ndarray>
KalmanFilter::multi_predict_py(Eigen::Ref<RowMatrix<double, -1, 8>> mean, py::array_t<double> &covariance) {
    auto ndarray = Ndarray(covariance);
    return multi_predict_py(mean, ndarray);
}

void KalmanFilter::update(Eigen::Ref<Eigen::Matrix<double, 8, 1>> &mean,
                          Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> &covariance,
                          Eigen::Ref<Eigen::Vector4d> &measurement,
                          Eigen::Matrix<double, 8, 1> &mean_,
                          Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_) {
    /*Run Kalman filter correction step.

    Parameters
    ----------
    mean : ndarray
        The predicted state's mean vector (8 dimensional).
    covariance : ndarray
        The state's covariance matrix (8x8 dimensional).
    measurement : ndarray
        The 4 dimensional measurement vector (x, y, a, h), where (x, y)
        is the center position, a the aspect ratio, and h the height of the
        bounding box.

    Returns
    -------
    (ndarray, ndarray)
        Returns the measurement-corrected state distribution.
     */
    auto project_result = project_py(mean, covariance);
    Eigen::Matrix<double, 4, 1> &projected_mean = std::get<0>(project_result);
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &projected_cov = std::get<1>(project_result);

    Eigen::LLT<RowMatrixXd, Eigen::Upper> llt_projected_cov(projected_cov);
    RowMatrixXd kalman_gain = llt_projected_cov.solve(
                    (covariance * _update_mat.transpose()).transpose())
            .transpose();

    Eigen::Vector4d innovation = measurement - projected_mean;
    mean_ = mean + kalman_gain * innovation;
    covariance_ = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
}

std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>>
KalmanFilter::update_py(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
                        Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance,
                        Eigen::Ref<Eigen::Vector4d> measurement) {
    std::tuple<Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> result;
    Eigen::Matrix<double, 8, 1> &mean_ = std::get<0>(result);
    Eigen::Matrix<double, 8, 8, Eigen::RowMajor> &covariance_ = std::get<1>(result);

    update(mean, covariance, measurement, mean_, covariance_);

    return result;
}

Eigen::VectorXd KalmanFilter::gating_distance(Eigen::Ref<Eigen::Matrix<double, 8, 1>> mean,
                                              Eigen::Ref<Eigen::Matrix<double, 8, 8, Eigen::RowMajor>> covariance,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 4, Eigen::RowMajor>> measurement,
                                              bool only_position, std::string metric) {
    /*Compute gating distance between state distribution and measurements.
    A suitable distance threshold can be obtained from `chi2inv95`. If
    `only_position` is False, the chi-square distribution has 4 degrees of
    freedom, otherwise 2.
    Parameters
    ----------
    mean : ndarray
        Mean vector over the state distribution (8 dimensional).
    covariance : ndarray
        Covariance of the state distribution (8x8 dimensional).
    measurements : ndarray
        An Nx4 dimensional matrix of N measurements, each in
        format (x, y, a, h) where (x, y) is the bounding box center
        position, a the aspect ratio, and h the height.
    only_position : Optional[bool]
        If True, distance computation is done with respect to the bounding
        box center position only.
    Returns
    -------
    ndarray
        Returns an array of length N, where the i-th element contains the
        squared Mahalanobis distance between (mean, covariance) and
        `measurements[i]`.
     */
    int N = measurement.rows();
    auto project_result = project_py(mean, covariance);
    Eigen::Matrix<double, 4, 1> &projected_mean = std::get<0>(project_result);
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &projected_cov = std::get<1>(project_result);

    RowMatrixXd mean_ = projected_mean;
    RowMatrixXd covariance_ = projected_cov;
    RowMatrixXd measurement_ = measurement;
    if (only_position) {
        mean_.resize(2, 1);
        covariance_.resize(2, 2);
        measurement_.resize(N, 2);

        mean_ = projected_mean.head(2);
        covariance_ = projected_cov.block(0, 0, 2, 2);
        measurement_ = measurement.leftCols(2);
    }

    // repeat mean_ N rows
    RowMatrixXd d = measurement_.array() - mean_.transpose().replicate(N, 1).array();

    if (metric == "gaussian") {
        return (d.array() * d.array()).rowwise().sum().transpose();
    } else if (metric == "maha") {
        Eigen::LLT<RowMatrixXd, Eigen::Upper> llt_covariance_(covariance_);
        RowMatrixXd cholesky_factor = llt_covariance_.matrixL();
        RowMatrixXd z = cholesky_factor.triangularView<Eigen::Upper>().solve(d.transpose());
        RowMatrixXd squared_maha = (z.array() * z.array()).matrix();
        return squared_maha.colwise().sum();
    }
    throw std::runtime_error("Not support metric: " + metric);
}

void init_module_kalman_filter(py::module &m) {
    py::module kalman_filter = m.def_submodule("kalman_filter");
    kalman_filter.def("is_kalman_filter_init", []() { return true; });
    kalman_filter.attr("__version__") = "0.1.0";
    kalman_filter.attr("chi2inv95") = chi2inv95;
    py::class_<KalmanFilter>(kalman_filter, "KalmanFilter")
            .def(py::init<>())
            .def_readonly("_motion_mat", &KalmanFilter::_motion_mat, pybind11::return_value_policy::reference_internal)
            .def_readonly("_update_mat", &KalmanFilter::_update_mat, pybind11::return_value_policy::reference_internal)
            .def("initiate", &KalmanFilter::initiate_py,
                 py::arg("measurement"))
            .def("predict", &KalmanFilter::predict_py,
                 py::arg("mean"), py::arg("covariance"))
            .def("project", &KalmanFilter::project_py,
                 py::arg("mean"), py::arg("covariance"))
            .def("multi_predict", py::overload_cast<Eigen::Ref<RowMatrix<double, -1, 8>>, py::array_t<double> &>(
                         &KalmanFilter::multi_predict_py),
                 py::arg("mean"), py::arg("covariance"))
            .def("update", &KalmanFilter::update_py,
                 py::arg("mean"), py::arg("covariance"), py::arg("measurement"))
            .def("gating_distance", &KalmanFilter::gating_distance,
                 py::arg("mean"), py::arg("covariance"), py::arg("measurement"), py::arg("only_position") = false,
                 py::arg("metric") = "maha")
            .def("__repr__", [](const KalmanFilter &kf) {
                std::stringstream ss;
                ss << "KalmanFilter object in pybind" << std::endl;
                return ss.str();
            });
}