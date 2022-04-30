# change to the directory of this script
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

print("current file path: ", current_dir)
print(sys.path)

# if '--src' in sys.argv, remove it and the list after it
# Note: --src command must be the last argument
if '--src' in sys.argv:
    src_index = sys.argv.index('--src')
    src_list = sys.argv[src_index+1:]
    sys.argv = sys.argv[:src_index]

if '--module_name' in sys.argv:
    module_name_index = sys.argv.index('--module_name')
    module_name = sys.argv[module_name_index+1]
    sys.argv = sys.argv[:module_name_index]


ext_type = CppExtension
if 'CUDA' in sys.argv:
    print("USING CUDA build")
    ext_type = CUDAExtension
    sys.argv.remove('CUDA')

print('sys.argv:', sys.argv)
print('module_name:', module_name)
print('src_list:', src_list)


setup(name='pybind_cuda',
      ext_modules=[ext_type(module_name, src_list,
                            # extra_compile_args={'cxx':['-O0', '-g','-I/usr/include/eigen3']})],
                            extra_compile_args={'cxx':['-O3', '-march=native','-I/usr/include/eigen3']},
                            use_ninja=False)],
      cmdclass={'build_ext': BuildExtension})