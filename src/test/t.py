# -*- coding: utf-8 -*-
# @Time    : 4/28/2022 10:24 PM
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : t.py
# @Software: PyCharm
import numpy as np
import torch
from tracklet import kalman_filter as kalman_filter_c

import tracking_utils.kalman_filter as kalman_filter

a = kalman_filter.KalmanFilter()
b = kalman_filter_c.KalmanFilter()

for i in range(1000):
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    measurement = np.random.rand(2, 4)
    c_res = b.gating_distance(mean, covariance, measurement, metric="gaussian")
    py_res = a.gating_distance(mean, covariance, measurement, metric="gaussian")
    assert np.allclose(c_res, py_res)

for i in range(1000):
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    measurement = np.random.rand(2, 4)
    c_res = b.gating_distance(mean, covariance, measurement, only_position=True, metric="gaussian")
    py_res = a.gating_distance(mean, covariance, measurement, only_position=True, metric="gaussian")
    assert np.allclose(c_res, py_res)

for i in range(1000):
    mean = np.array([1418.5, 599.3, 0.46051, 382.1, 0, 0, 0, 0])
    covariance = np.array([[2395.3, 0, 0, 0, 570.32, 0, 0, 0],
                           [0, 2395.3, 0, 0, 0, 570.32, 0, 0],
                           [0, 0, 0.0002, 0, 0, 0, 1e-10, 0],
                           [0, 0, 0, 2395.3, 0, 0, 0, 570.32],
                           [570.32, 0, 0, 0, 576.02, 0, 0, 0],
                           [0, 570.32, 0, 0, 0, 576.02, 0, 0],
                           [0, 0, 1e-10, 0, 0, 0, 2e-10, 0],
                           [0, 0, 0, 570.32, 0, 0, 0, 576.02]])
    measurement = np.array([[1425.4, 600.12, 0.47235, 387.63],
                            [628.77, 577.23, 0.34893, 260.66],
                            [494.51, 584.53, 0.41566, 275.83],
                            [1517.1, 594.59, 0.4934, 342.36],
                            [665.41, 551.57, 0.36027, 191.52],
                            [516.38, 585.19, 0.36156, 257.2],
                            [954.88, 492.25, 0.36799, 117.16],
                            [588.88, 452.27, 0.5078, 45.824],
                            [1105.3, 540.93, 0.42875, 112.36],
                            [1075.4, 537.16, 0.41348, 111.23],
                            [566.2, 513.18, 0.3521, 93.838],
                            [602.04, 451.37, 0.47462, 41.473],
                            [1035.8, 488.66, 0.42282, 115.24],
                            [1117.8, 492.94, 0.36998, 104.98],
                            [438.84, 500.28, 0.44797, 92.546],
                            [991.69, 494.48, 0.3918, 88.883]])
    c_res = b.gating_distance(mean, covariance, measurement, metric="maha")
    py_res = a.gating_distance(mean, covariance, measurement, metric="maha")
    assert np.allclose(c_res, py_res)
