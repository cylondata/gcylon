#!/bin/bash

SOURCE_DIR=$(pwd)/cpp
CONDA_CPP_BUILD="OFF"
CYTHON_BUILD="OFF"
CONDA_CYTHON_BUILD="OFF"
JAVA_BUILD="OFF"
BUILD_ALL="OFF"

BUILD_MODE=Release
BUILD_MODE_DEBUG="OFF"
BUILD_MODE_RELEASE="OFF"
PYTHON_RELEASE="OFF"
RUN_CPP_TESTS="OFF"
RUN_PYTHON_TESTS="OFF"
STYLE_CHECK="OFF"
INSTALL_PATH=
BUILD_PATH=$(pwd)/build
CMAKE_FLAGS=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"


case $key in
    -bpath|--build_path)
    BUILD_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -ipath|--install_path)
    INSTALL_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    --cpp)
    CONDA_CPP_BUILD="ON"
    shift # past argument
    ;;
    --conda_cpp)
    CONDA_CPP_BUILD="ON"
    shift # past argument
    ;;
    --conda_cython)
    CONDA_CYTHON_BUILD="ON"
    shift # past argument
    ;;
    --cython)
    CPP_BUILD="ON"
    CYTHON_BUILD="ON"
    shift # past argument
    ;;
    --java)
    CPP_BUILD="ON"
    JAVA_BUILD="ON"
    shift # past argument
    ;;
    --debug)
    BUILD_MODE_DEBUG="ON"
    BUILD_MODE_RELEASE="OFF"
    shift # past argument
    ;;
    --release)
    BUILD_MODE_RELEASE="ON"
    BUILD_MODE_DEBUG="OFF"
    shift # past argument
    ;;
    --test)
    RUN_CPP_TESTS="ON"
    shift # past argument
    ;;
    --pytest)
    RUN_PYTHON_TESTS="ON"
    shift # past argument
    ;;
    --style-check)
    STYLE_CHECK="ON"
    shift # past argument
    ;;
    --py-release)
    PYTHON_RELEASE="ON"
    CPP_BUILD="OFF"
    shift # past argument
    ;;
    --cmake-flags)
    CMAKE_FLAGS="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "PYTHON ENV PATH       = ${PYTHON_ENV_PATH}"
echo "BUILD PATH            = ${BUILD_PATH}"
echo "CONDA CPP BUILD       = ${CONDA_CPP_BUILD}"
echo "CYTHON BUILD          = ${CYTHON_BUILD}"
echo "CONDA CYTHON BUILD    = ${CONDA_CYTHON_BUILD}"
echo "BUILD ALL             = ${BUILD_ALL}"
echo "BUILD DEBUG           = ${BUILD_MODE_DEBUG}"
echo "BUILD RELEASE         = ${BUILD_MODE_RELEASE}"
echo "RUN CPP TEST          = ${RUN_CPP_TESTS}"
echo "RUN PYTHON TEST       = ${RUN_PYTHON_TESTS}"
echo "STYLE CHECK           = ${STYLE_CHECK}"
echo "ADDITIONAL CMAKE FLAGS= ${CMAKE_FLAGS}"

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

CPPLINT_COMMAND=" \"-DCMAKE_CXX_CPPLINT=cpplint;--linelength=100;--headers=h,hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" "

################################## Util Functions ##################################################
quit() {
exit
}

print_line() {
echo "=================================================================";
}

read_python_requirements(){
  pip3 install -r requirements.txt || exit 1
}

check_python_pre_requisites(){
  echo "Checking Python Pre_-requisites"
  response=$(python3 -c \
    "import numpy; print('Numpy Installation');\
    print('Version {}'.format(numpy.__version__));\
    print('Library Installation Path {}'.format(numpy.get_include()))")
  echo "${response}"
}

INSTALL_CMD=
if [ -z "$INSTALL_PATH" ]
then
  echo "\-ipath|--install_path is NOT set default to cmake"
else
  INSTALL_CMD="-DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}"
  echo "Install location set to: ${INSTALL_PATH}"
fi

build_cpp_conda(){
  print_line
  echo "Building Conda CPP in ${BUILD_MODE} mode"
  print_line

  # set install path to conda dir if it is not already set
  INSTALL_PATH=${INSTALL_PATH:=${PREFIX:=${CONDA_PREFIX}}}

  echo "SOURCE_DIR: ${SOURCE_DIR}"
  BUILD_PATH=$(pwd)/build
  mkdir -p ${BUILD_PATH}
  pushd ${BUILD_PATH} || exit 1

  cmake -DPYCYLON_BUILD=${CYTHON_BUILD} -DPYTHON_EXEC_PATH=${PYTHON_ENV_PATH} \
      -DCMAKE_BUILD_TYPE=${BUILD_MODE} -DCYLON_WITH_TEST=${RUN_CPP_TESTS} -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
      ${CMAKE_FLAGS} \
      ${SOURCE_DIR} \
      || exit 1
  make -j 4 || exit 1
  printf "Cylon CPP Built Successfully!"
  make install || exit 1
  printf "Cylon Installed Successfully!"
  popd || exit 1
  print_line
}

build_cython_conda() {
  print_line
  echo "Building Conda Python"
  echo "================================ Conda PREFIX: ${PREFIX}"
  echo "================================ BUILD_PATH: ${BUILD_PATH}"
  CYLON_LIB=${CYLON_HOME}/build/lib
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CYLON_LIB}:${BUILD_PATH}/lib:${LD_LIBRARY_PATH}" || exit 1
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

  export GCYLON_HOME=${PWD}
  echo "GCYLON_HOME: ${GCYLON_HOME}"

  pushd python || exit 1
  make clean
  echo "current dir before executing setup.py: ${PWD}"
  python3 setup.py install || exit 1
  popd || exit 1
  print_line
}

build_python() {
  print_line
  echo "Building Python"
  if [ -z "$CONDA_PREFIX" ]; then
    echo "CONDA_PREFIX is not set. Please activate gcylon conda environment"
    exit 1
  fi

  echo "\-ipath|--install_path is NOT set default to cmake"
#  export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH || exit 1
  export LD_LIBRARY_PATH=${BUILD_PATH}/lib:$CYLON_HOME/build/lib:$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib || exit 1
  echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH

  export GCYLON_HOME=${PWD}
  echo "GCYLON_HOME: ${GCYLON_HOME}"

  export PARALLEL_LEVEL=4

  # shellcheck disable=SC1090
  read_python_requirements
#  check_python_pre_requisites
  pushd python || exit 1
  pip3 uninstall -y pygcylon
  make clean
  python3 setup.py install || exit 1
  popd || exit 1
  print_line
}

release_python() {
  print_line
  echo "Building Python"
  export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
  echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
  source "${PYTHON_ENV_PATH}"/bin/activate
  read_python_requirements
  check_python_pre_requisites
  pushd python || exit 1
  pip3 uninstall -y pycylon
  make clean
  # https://www.scivision.dev/easy-upload-to-pypi/ [solution to linux wheel issue]
  #ARROW_HOME=${BUILD_PATH} python3 setup.py sdist bdist_wheel
  ARROW_HOME=${BUILD_PATH} python3 setup.py build_ext --inplace --library-dir=${BUILD_PATH} || exit 1
  popd || exit 1
  print_line
}

export_info(){
  print_line
  echo "Add the following to your LD_LIBRARY_PATH";
  echo "export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:"\$"LD_LIBRARY_PATH";
  print_line
}

check_pyarrow_installation(){
  export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
  response=$(python3 -c \
    "import pyarrow; print('PyArrow Installation');\
    print('Version {}'.format(pyarrow.__version__));\
    print('Library Installation Path {}'.format(pyarrow.get_library_dirs()))")
  echo "${response}"
}

check_pygcylon_installation(){
  response=$(python3 python/test/test_pygcylon.py)
  echo "${response}"
}

python_test(){
  python3 -m pytest python/test/test_all.py
}

build_java(){
  echo "Building Java"
  cd java
  mvn clean install -Dcylon.core.libs=$BUILD_PATH/lib -Dcylon.arrow.dir=$BUILD_PATH/arrow/install || exit 1
  echo "Cylon Java built Successufully!"
  cd ../
}

####################################################################################################

if [ "${BUILD_MODE_DEBUG}" = "ON" ]; then
   	BUILD_MODE=Debug	
fi

if [ "${BUILD_MODE_RELEASE}" = "ON" ]; then
   	BUILD_MODE=Release	
fi

if [ "${CONDA_CPP_BUILD}" = "ON" ]; then
	echo "Running conda build"
	build_cpp_conda
fi

if [ "${CONDA_CYTHON_BUILD}" = "ON" ]; then
	echo "Running conda build"
	build_cython_conda
fi


if [ "${CYTHON_BUILD}" = "ON" ]; then
	export_info	
	build_python
	check_pygcylon_installation
fi

if [ "${JAVA_BUILD}" = "ON" ]; then
	build_java
fi

if [ "${PYTHON_RELEASE}" = "ON" ]; then	
	export_info
	check_pyarrow_installation
	release_python
fi

if [ "${RUN_CPP_TESTS}" = "ON" ]; then
	echo "Running CPP tests"
	CTEST_OUTPUT_ON_FAILURE=1 make -C "$BUILD_PATH" test
fi

if [ "${RUN_PYTHON_TESTS}" = "ON" ]; then
	echo "Running Python tests"
	python_test
fi




