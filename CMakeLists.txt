# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  hello #${SKBUILD_PROJECT_NAME}
  VERSION 1#${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(GPU_operations_module MODULE src/briann/c_plus_plus/GPU/operations.cpp WITH_SOABI)
target_link_libraries(GPU_operations_module PRIVATE pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(GPU_operations_module PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS GPU_operations_module DESTINATION pybind11_builts)
