# -*- coding: utf-8 -*-
# @Time    : 4/24/2022 8:18 PM
# @Author  : Lazurite
# @Email   : lazurite@tom.com
# @File    : kalman_filter_test.py.py
# @Software: PyCharm

# %%
import sys
import torch
import numpy as np
import tracking_utils.kalman_filter as kalman_filter
import tracklet
from tracklet import kalman_filter as kalman_filter_c, Ndarray
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup
import time


# %%

# 计算时间函数
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print(f'current Function {func.__name__} run time is {time.time() - local_time}')

    return wrapper


# %%

a = np.random.rand(3, 3, 3)
a_c = Ndarray(a)
a_np = np.array(a_c, copy=False)
print(hex(a_np.__array_interface__['data'][0]))
print(a_c.ptr())
assert np.allclose(a, a_np)

a_c = Ndarray.create(3, 3, 3)
print(a_c)
a_np = np.array(a_c, copy=False)
assert np.allclose(a_c, a_np)

# %%
a = kalman_filter.KalmanFilter()
b = kalman_filter_c.KalmanFilter()

for i in range(100):
    measurement = np.random.rand(4)
    r1, r2 = a.initiate(measurement)
    r3, r4 = b.initiate(measurement)
    assert np.all(r1 == r3)
    assert np.all(r2 == r4)

# %%
for i in range(100):
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    r1, r2 = a.predict(mean, covariance)
    r3, r4 = b.predict(mean, covariance)
    assert np.allclose(r1, r3)
    assert np.allclose(r2, r4)

# %%
for i in range(100):
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    r1, r2 = a.project(mean, covariance)
    r3, r4 = b.project(mean, covariance)
    assert np.allclose(r1, r3)
    assert np.allclose(r2, r4)

# %%
# test multi_predict
for i in range(100):
    mean = np.random.rand(2, 8)
    covariance = np.random.rand(2, 8, 8)
    mean_np, covariance_np = a.multi_predict(mean, covariance)
    mean_c, covariance_c = b.multi_predict(mean, covariance)
    assert np.allclose(mean_np, mean_c)
    assert np.allclose(covariance_np, covariance_c)

# %%
# test update
mean = np.array([1418.5, 599.3, 0.46051, 382.1, 0, 0, 0, 0])

covariance = np.array([[2395.3, 0, 0, 0, 570.32, 0, 0, 0],
                       [0, 2395.3, 0, 0, 0, 570.32, 0, 0],
                       [0, 0, 0.0002, 0, 0, 0, 1e-10, 0],
                       [0, 0, 0, 2395.3, 0, 0, 0, 570.32],
                       [570.32, 0, 0, 0, 576.02, 0, 0, 0],
                       [0, 570.32, 0, 0, 0, 576.02, 0, 0],
                       [0, 0, 1e-10, 0, 0, 0, 2e-10, 0],
                       [0, 0, 0, 570.32, 0, 0, 0, 576.02]])

measurement = np.array([1425.4, 600.12, 0.47235, 387.63])
for i in range(100):
    c_res0, c_res1 = b.update(mean, covariance, measurement)
    py_res0, py_res1 = a.update(mean, covariance, measurement)
    assert np.allclose(c_res0, py_res0)
    assert np.allclose(c_res1, py_res1)

# %%
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
for i in range(100):
    c_res = b.gating_distance(mean, covariance, measurement, metric="maha")
    py_res = a.gating_distance(mean, covariance, measurement, metric="maha")
    assert np.allclose(c_res, py_res)


# %%
@print_run_time
def test_np_initiate():
    a = kalman_filter.KalmanFilter()
    measurement = np.random.rand(4)
    for i in range(1000):
        r1, r2 = a.initiate(measurement)


@print_run_time
def test_c_initiate():
    b = kalman_filter_c.KalmanFilter()
    measurement = np.random.rand(4)
    for i in range(1000):
        r3, r4 = b.initiate(measurement)


test_np_initiate()
test_c_initiate()


@print_run_time
def test_np_predict():
    a = kalman_filter.KalmanFilter()
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    for i in range(1000):
        r1, r2 = a.predict(mean, covariance)


@print_run_time
def test_c_predict():
    b = kalman_filter_c.KalmanFilter()
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    for i in range(1000):
        r3, r4 = b.predict(mean, covariance)


test_np_predict()
test_c_predict()


# %%
@print_run_time
def test_np_project():
    a = kalman_filter.KalmanFilter()
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    for i in range(1000):
        r1, r2 = a.project(mean, covariance)


@print_run_time
def test_c_project():
    b = kalman_filter_c.KalmanFilter()
    mean = np.random.rand(8)
    covariance = np.random.rand(8, 8)
    for i in range(1000):
        r3, r4 = b.project(mean, covariance)


test_np_project()
test_c_project()


# %%
@print_run_time
def test_np_multi_predict():
    a = kalman_filter.KalmanFilter()
    mean = np.random.rand(2, 8)
    covariance = np.random.rand(2, 8, 8)
    for i in range(1000):
        r1, r2 = a.multi_predict(mean, covariance)


@print_run_time
def test_c_multi_predict():
    b = kalman_filter_c.KalmanFilter()
    mean = np.random.rand(2, 8)
    covariance = np.random.rand(2, 8, 8)
    for i in range(1000):
        r3, r4 = b.multi_predict(mean, covariance)


test_np_multi_predict()
test_c_multi_predict()

# %%
mean = np.array([1418.5, 599.3, 0.46051, 382.1, 0, 0, 0, 0])

covariance = np.array([[2395.3, 0, 0, 0, 570.32, 0, 0, 0],
                       [0, 2395.3, 0, 0, 0, 570.32, 0, 0],
                       [0, 0, 0.0002, 0, 0, 0, 1e-10, 0],
                       [0, 0, 0, 2395.3, 0, 0, 0, 570.32],
                       [570.32, 0, 0, 0, 576.02, 0, 0, 0],
                       [0, 570.32, 0, 0, 0, 576.02, 0, 0],
                       [0, 0, 1e-10, 0, 0, 0, 2e-10, 0],
                       [0, 0, 0, 570.32, 0, 0, 0, 576.02]])

measurement = np.array([1425.4, 600.12, 0.47235, 387.63])


@print_run_time
def test_np_update():
    a = kalman_filter.KalmanFilter()
    for i in range(1000):
        r1, r2 = a.update(mean, covariance, measurement)


@print_run_time
def test_c_update():
    b = kalman_filter_c.KalmanFilter()
    for i in range(1000):
        r3, r4 = b.update(mean, covariance, measurement)


test_np_update()
test_c_update()

# %%
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


@print_run_time
def test_py_gating_distance():
    a = kalman_filter.KalmanFilter()
    for i in range(1000):
        r2 = a.gating_distance(mean, covariance, measurement)


@print_run_time
def test_c_gating_distance():
    b = kalman_filter_c.KalmanFilter()
    for i in range(1000):
        r4 = b.gating_distance(mean, covariance, measurement)


test_py_gating_distance()
test_c_gating_distance()


# current Function test_np_initiate run time is 0.020568132400512695
# current Function test_c_initiate run time is 0.0008282661437988281
# current Function test_np_predict run time is 0.03365349769592285
# current Function test_c_predict run time is 0.0015561580657958984
# current Function test_np_project run time is 0.012738943099975586
# current Function test_c_project run time is 0.0012729167938232422
# current Function test_np_multi_predict run time is 0.05970144271850586
# current Function test_c_multi_predict run time is 0.0026209354400634766
# current Function test_np_update run time is 0.040341854095458984
# current Function test_c_update run time is 0.0023005008697509766
# current Function test_py_gating_distance run time is 0.04303550720214844
# current Function test_c_gating_distance run time is 0.002392292022705078