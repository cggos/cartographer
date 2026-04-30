# Cartographer Project Context

Cartographer is a real-time Simultaneous Localization and Mapping (SLAM) system in 2D and 3D across multiple platforms and sensor configurations.

## Project Overview

- **Core Technology:** C++11, Eigen (linear algebra), Ceres Solver (optimization), Protobuf (serialization), Lua (configuration).
- **Architecture:** Trajectory-based SLAM. Sensor data is organized into trajectories, which consist of nodes and submaps. Loop closure is performed via pose graph optimization.
- **Maintenance Status:** This repository is no longer actively maintained by Google. ROS-specific forks are maintained at `github.com/ros2/cartographer`.

## Key Directories

- `cartographer/`: Main library source code.
    - `common/`: Core utilities (time, math, thread pool).
    - `mapping/`: SLAM algorithms (2D/3D trajectory builders, pose graph).
    - `io/`: Input/Output (pbstream serialization, points processors).
    - `sensor/`: Sensor data types (point clouds, IMU, odometry).
    - `transform/`: Coordinate transformations.
- `configuration_files/`: Default Lua configuration files for the SLAM algorithms.
- `cmake/` & `bazel/`: Build system configurations.
- `scripts/`: Installation and helper scripts.

## Building and Running

### Prerequisites
Cartographer depends on several libraries:
- Abseil, Boost, Cairo, Ceres Solver, Eigen3, gflags, glog, Lua 5.2, Protobuf.

### Build with CMake (Recommended)
The project defaults to a **Release** build for performance. Debug builds are intentionally restricted unless `FORCE_DEBUG_BUILD=True` is passed.

```bash
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

### Running Tests
```bash
ctest
# Or via Ninja:
ninja test
```

### Build with Bazel
```bash
bazel build //...
bazel test //...
```

## Development Conventions

- **Coding Style:** Adheres to the Google C++ Style Guide. A `.clang-format` file is provided in the `cartographer/` directory.
- **Testing:** Extensive use of `gtest` and `gmock`. Most classes have a corresponding `_test.cc` file.
- **Configuration:** SLAM parameters are configured via Lua. Use `cartographer_print_configuration` to debug configuration issues.
- **Serialization:** Uses `.pbstream` files for saving and loading SLAM state, based on Protobuf.

## Tooling and Utilities

- `cartographer_pbstream`: Utility for inspecting and manipulating pbstream files.
- `cartographer_print_configuration`: Validates and prints the Lua configuration.
- `cartographer_autogenerate_ground_truth`: Tools for ground truth generation and evaluation.
