import import_helper
import torch
import tracklet
import numpy as np

a = tracklet.return_tuple_matrix()[0]
print("Address of a:", hex(a.__array_interface__['data'][0]))

a2 = tracklet.return_tuple_matrix()[0]
print("Address of a:", hex(a2.__array_interface__['data'][0]))

b = tracklet.return_tuple_matrix2()[0]
print("Address of b:", hex(b.__array_interface__['data'][0]))

b2 = tracklet.return_tuple_matrix2()[0]
print("Address of b:", hex(b2.__array_interface__['data'][0]))