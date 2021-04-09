##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

# References
'''
https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
https://github.com/thewtex/cython-cmake-example/blob/master/setup.py
'''

import os
import sys
import sysconfig

import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_lib

version = versioneer.get_version(),
cmdclass = versioneer.get_cmdclass(),

# make sure conda is activated or, conda-build is used
if ("CONDA_PREFIX" not in os.environ and "CONDA_BUILD" not in os.environ):
    print("Neither CONDA_PREFIX nor CONDA_BUILD is set. Activate conda environment or use conda-build")
    sys.exit()


# os.environ["CXX"] = "mpic++"

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

cython_files = ["pygcylon/*.pyx"]

if "GCYLON_HOME" not in os.environ:
    raise ValueError("GCYLON_HOME not set")

if "CYLON_HOME" not in os.environ:
    raise ValueError("CYLON_HOME not set")

std_version = '-std=c++14'
additional_compile_args = [std_version]

extra_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
extra_compile_args = extra_compile_args + extra_link_args + additional_compile_args
#  extra_compile_args = additional_compile_args
extra_link_args.append("-Wl,-rpath")

#glob_library_directory = os.path.join(CYLON_PREFIX, "glog", "install", "lib")


#glog_lib_include_dir = os.path.join(CYLON_PREFIX, "glog", "install", "include")
gcylon_library_directory = os.path.join(os.environ.get('GCYLON_HOME'), "build/lib")

if "CONDA_PREFIX" in os.environ:
    conda_lib_dir = os.path.join(os.environ.get('CONDA_PREFIX'), "lib")
    conda_include_dir = os.path.join(os.environ.get('CONDA_PREFIX'), "include")
elif "CONDA_BUILD" in os.environ:
    conda_lib_dir = os.path.join(os.environ.get('BUILD_PREFIX'), "lib") + " "
    conda_lib_dir += os.path.join(os.environ.get('PREFIX'), "lib")
    conda_include_dir = os.path.join(os.environ.get('BUILD_PREFIX'), "include") + " "
    conda_include_dir += os.path.join(os.environ.get('PREFIX'), "include")

print("gcylon_library_directory: ", gcylon_library_directory)
print("conda_library_directory: ", conda_lib_dir)

library_directories = [
    gcylon_library_directory,
    conda_lib_dir,
    get_python_lib(),
    os.path.join(os.sys.prefix, "lib")]

libraries = ["gcylon", "cudf"]
#libraries = ["gcylon", "cylon", "glog"]

cylon_include_dir = os.path.join(os.environ.get('CYLON_HOME'), "cpp/src/cylon")

_include_dirs = ["../cpp/src/cylon/cudf",
                 cylon_include_dir,
                 conda_include_dir,
                 os.path.dirname(sysconfig.get_path("include"))]

# Adopted the Cudf Python Build format
# https://github.com/rapidsai/cudf

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=_include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_directories,
    )
]

compiler_directives = {"language_level": 3, "embedsignature": True}
packages = find_packages(include=["pygcylon", "pygcylon.*"])

setup(
    name="pygcylon",
    packages=packages,
    version=versioneer.get_version(),
    setup_requires=["cython", "setuptools"],
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
    ),
    python_requires='>=3.7',
    install_requires=[
        'cython',
    ],
    zip_safe=False,
)
