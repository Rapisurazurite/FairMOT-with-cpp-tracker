import os
import sys

# get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
build_dir = os.path.join(project_dir, 'build')
exec_file_dir = [dir_name for dir_name in os.listdir(build_dir) if dir_name.startswith("lib") ][0]
extension_dir = os.path.join(build_dir, exec_file_dir)
sys.path.append(extension_dir)

print(f"extension_dir:{extension_dir}")