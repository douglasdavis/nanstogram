project(nanogram)
cmake_minimum_required(VERSION 3.18...3.22)

if (NOT SKBUILD)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if (SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
elseif (MSVC)
  # MSVC needs a little extra help finding the Python library
  find_package(PythonInterp)
  find_package(Python)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

add_subdirectory(extern/nanobind)
find_package(nanobind CONFIG REQUIRED)

# Build the actual extension module
nanobind_add_module(
  ext
  NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
  src/ext.cpp
)

# Install directive for scikit-build
install(TARGETS ext LIBRARY DESTINATION .)
