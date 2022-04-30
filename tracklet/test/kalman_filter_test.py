# import import_helper
import torch
from tracklet import kalman_filter
import numpy as np
print(kalman_filter.test_stl_ownership())
if __name__ == '__main__':
    # Test Kalman Filter
    print(f"Version: {kalman_filter.__version__}")
    print(f"chi2inv95: {kalman_filter.chi2inv95}")
    print(f"chi2inv95[1]: {kalman_filter.chi2inv95[1]}")
    print(f"chi2inv95[2]: {kalman_filter.chi2inv95[2]}")

    Filter = kalman_filter.KalmanFilter()
    mean = np.array([1,2,3,4], dtype=np.float64)
    a, b = Filter.initiate(mean)
    print(f"a: {a}, b: {b}")

    print(kalman_filter.test_stl_ownership())

