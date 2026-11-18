cmake_minimum_required(VERSION 3.24)
project(herdsight_core VERSION 0.0.1 LANGUAGES CXX)

# HerdSight C++ core: geometry, matching and intervals shared by the onboard ROS 2 nodes and,
# through pybind11, the Python evaluation. Every function here has a Python reference in
# python/herdsight and a parity test in tests/test_core_parity.py.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HERDSIGHT_BUILD_TESTS "Build the GoogleTest suite" ON)
option(HERDSIGHT_BUILD_PYTHON "Build the pybind11 module herdsight._core" OFF)
option(HERDSIGHT_SANITIZE "Build with address and undefined-behaviour sanitizers" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
if(HERDSIGHT_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(herdsight_core src/frames.cpp)
target_include_directories(herdsight_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(herdsight_core PUBLIC Eigen3::Eigen)

if(HERDSIGHT_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(herdsight_core_tests tests/test_frames.cpp)
  target_link_libraries(herdsight_core_tests PRIVATE herdsight_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(herdsight_core_tests)
endif()

if(HERDSIGHT_BUILD_PYTHON)
  if(HERDSIGHT_SANITIZE)
    message(FATAL_ERROR "Build the Python module without sanitizers, in a separate build folder.")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE herdsight_core)
  # Put the module inside the Python package so `import herdsight._core` works in place.
  set(HERDSIGHT_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python/herdsight"
      CACHE PATH "Where to put the built Python module")
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HERDSIGHT_PYTHON_PACKAGE_DIR}")
endif()
