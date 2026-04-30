# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**CMake (primary):**
```bash
mkdir build && cd build
cmake .. -G Ninja
ninja
```

Optional flags: `-DBUILD_GRPC=ON`, `-DBUILD_PROMETHEUS=ON`

**Bazel (alternative):**
```bash
bazel build //...
```

Bazel enforces `-c opt --force_pic` via `.bazelrc`.

## Tests

```bash
# All tests (from build dir)
ninja test
# or with output on failure
CTEST_OUTPUT_ON_FAILURE=1 ninja test

# Bazel
bazel test //...

# Single test binary (after build)
./build/cartographer/mapping/internal/2d/pose_graph_2d_test
```

Test files are named `*_test.cc` and located alongside their implementation files (91 total).

## Code Style

Google C++ Style Guide. A `.clang-format` file is at `cartographer/.clang-format`.

```bash
git clang-format  # format staged changes
```

## Architecture Overview

Cartographer is a **real-time SLAM system** with a two-stage pipeline:

### Pipeline
```
Sensor Input (LiDAR, IMU, Odometry)
    ↓
SensorCollator — synchronizes multi-sensor streams per trajectory
    ↓
Local SLAM (per trajectory, no loop closure)
    ├─ PoseExtrapolator — predicts poses using IMU/gravity
    ├─ RangeDataCollator — accumulates point clouds
    ├─ MotionFilter — drops redundant measurements
    ├─ RealTimeCorrelativeScanMatcher — fast coarse alignment
    ├─ CeresScanMatcher — fine-grained optimization
    └─ ActiveSubmaps — maintains 2 active submaps (for matching + initialization)
    ↓
Global SLAM (PoseGraph)
    ├─ ConstraintBuilder — detects loop closures by matching non-adjacent submaps
    └─ Ceres optimizer — globally optimizes all trajectory node poses
    ↓
Output: Optimized poses + serialized map (*.pbstream)
```

### Key Interfaces
- `MapBuilderInterface` (`mapping/map_builder_interface.h`) — top-level entry point
- `TrajectoryBuilderInterface` — per-trajectory local SLAM
- `PoseGraphInterface` — global optimization and serialization
- `CollatorInterface` — sensor data synchronization

### Major Subsystems

| Directory | Role |
|-----------|------|
| `mapping/internal/2d/` | Local SLAM for 2D (LocalTrajectoryBuilder2D, ActiveSubmaps2D) |
| `mapping/internal/3d/` | Local SLAM for 3D (LocalTrajectoryBuilder3D, ActiveSubmaps3D) |
| `mapping/2d/` / `mapping/3d/` | Submap and grid implementations |
| `mapping/internal/optimization/` | Ceres-based pose graph optimization |
| `mapping/internal/constraints/` | Loop closure constraint builders (2D & 3D) |
| `mapping/internal/scan_matching/` | Correlative and Ceres scan matchers |
| `sensor/` | Data types: PointCloud, IMU, Odometry, RangeData, LandmarkData |
| `transform/` | Rigid3d, rotation utilities |
| `common/` | Thread pool, Lua config, time utilities, math helpers |
| `io/` | pbstream serialization, point cloud I/O, submap painting |
| `cloud/` | gRPC server/client for distributed deployment |

### Submaps
- Always 2 active submaps per trajectory: one old (used for scan matching), one new (being built)
- **2D:** `Submap2D` backed by `Grid2D` (probability grids)
- **3D:** `Submap3D` backed by `HybridGrid` (voxel-based, high + low resolution)

### Multi-Trajectory Support
Each trajectory is independent (supports multi-robot SLAM). Trajectories can be linked via initial pose constraints. The pose graph optimizes across all trajectories simultaneously.

### Configuration
Lua-based configuration files in `configuration_files/`. Parsed by `LuaParameterDictionary`. Key config groups: `TrajectoryBuilderOptions`, `PoseGraphOptions`, `SubmapsOptions`, scan matcher parameters.

### Serialization
Maps are serialized to `.pbstream` (Protocol Buffer stream) format. Use `cartographer_pbstream` binary to inspect/manipulate them.

## Notable Tools

- `cartographer_print_configuration` — validates and prints Lua configs
- `cartographer_pbstream` — inspects/manipulates pbstream files
- `cartographer_autogenerate_ground_truth` — ground truth generation
- `scripts/install_abseil.sh`, `scripts/install_lua.sh` — dependency install scripts

## Project Status

No longer actively maintained by Google. Active ROS forks:
- ROS 1: `github.com/cartographer-project/cartographer_ros`
- ROS 2: `github.com/ros2/cartographer`
