# change to the directory of this script
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# remove -c from sys.argv

setup(name='pybind_cuda',
      ext_modules=[CppExtension('pybind_torch', ['main.cpp', 'src/cpp_extension.cpp', 'src/KalmanFilter.cpp',
                                                 'src/Ndarray.cpp'],
                                extra_compile_args={'cxx': ['-O3', '-march=native', '-I/usr/include/eigen3']},
                                )],
      cmdclass={'build_ext': BuildExtension})
