# Rescue Robot Simulator — Codebase Overview

## Summary
This is a Webots-based rescue robot simulation codebase for a university competition (IITA/Simulated Rescue Maze). The robot autonomously navigates a maze, detects victims (fixtures with Greek symbols Φ/Ψ/Ω), maps the environment (walls, holes, swamps, checkpoints), and sends a final map matrix to a server. The system runs on a tightly integrated pipeline of LIDAR-based wall mapping, camera-based floor classification, vision-based fixture detection, multi-strategy pathfinding agents, and final matrix generation.

## Architecture

### Primary Pattern: Layered + State Machine

```
RescueRobot (user API layer)
    └── Executor (orchestrator — runs state machine)
            ├── Mapper (sensor fusion → pixel grid)
            │   ├── WallMapper (LIDAR → walls, navigation margins)
            │   ├── FloorMapper (camera IPM → floor colors, holes, swamps)
            │   ├── RobotMapper (traversed, seen_by_camera, discovered)
            │   ├── OccupiedMapper (walls OR holes)
            │   ├── FixtureMapper (victim proximity zones)
            │   ├── FixtureDetector (camera → victim positions on map)
            │   └── ArrayFilterer (noise removal)
            ├── Agent (navigation strategy)
            │   └── SubagentPriorityCombiner
            │       ├── GoToFixturesAgent (priority 1)
            │       ├── FollowWallsAgent (priority 2)
            │       └── GoToNonDiscoveredAgent (priority 3)
            ├── Robot (hardware abstraction)
            │   ├── DriveBase (wheel control, rotation, movement)
            │   ├── PoseManager (GPS/Gyroscope fusion)
            │   ├── Lidar (point cloud)
            │   ├── Camera (3 cameras)
            │   └── Comunicator (server communication)
            ├── StuckDetector
            └── DelayManager
```

### Technology Stack
- **Language**: Python 3
- **Simulator**: Webots (robot controller API via `controller` module)
- **Key Libraries**: numpy, OpenCV (cv2), scikit-image, imutils
- **Concurrency**: Single-threaded, synchronous, time-step-based loop (32ms per step)
- **Coordinate System**: 2D top-down, meters for world coords, pixels for grid

### Execution Flow
1. `run.py` → `main.py` → `RescueRobot.run_autonomous()` → `Executor.run()`
2. `Executor.run()` loops: `robot.step()` (sim step) → `robot.update()` (sensors) → `mapper.update()` (build map) → `state_machine.run()` (decide action)
3. State machine: `init` → `explore` ↔ `report_fixture` | `stuck` | `send_map` → `end`

## Directory Structure

```
src/
├── main.py                           # Entry point with usage examples
├── run.py                            # Webots controller entry
├── rescue_robot.py                   # User-friendly API layer
├── student_example.py                # Student template code
├── flags.py                          # Global debug/feature flags
├── utilities.py                      # Math helpers, drawing utilities
├── map_visualizer.py                 # Real-time OpenCV map visualization
├── mapping/                          # MAP BUILDING CORE
│   ├── mapper.py                     # Mapper — orchestrates all sub-mappers
│   ├── wall_mapper.py                # WallMapper — LIDAR → walls
│   ├── floor_mapper.py               # FloorMapper — camera → floor colors
│   ├── robot_mapper.py               # RobotMapper — traversed, seen_by_camera
│   ├── occupied_mapping.py           # OccupiedMapper — walls OR holes
│   ├── fixture_mapper.py             # FixtureMapper — victim zones
│   └── array_filtering.py            # ArrayFilterer — noise removal
├── executor/                         # ORCHESTRATION
│   ├── executor.py                   # Executor — state machine, mission control
│   └── stuck_detector.py             # StuckDetector
├── agent/                            # NAVIGATION
│   ├── agent.py                      # Agent + SubagentPriorityCombiner
│   ├── agent_interface.py            # Abstract interfaces
│   ├── pathfinding/                  # A* pathfinding
│   └── subagents/                    # 4 sub-agents
├── robot/                            # HARDWARE ABSTRACTION
│   ├── robot.py                      # Robot — top-level hardware interface
│   ├── drive_base.py                 # DriveBase + movement control
│   ├── pose_manager.py               # GPS/Gyroscope fusion
│   └── devices/                      # lidar, camera, gps, gyroscope, wheel
├── fixture_detection/                # VISION — symbol recognition
├── final_matrix_creation/            # MAP OUTPUT — final matrix generation
├── data_structures/                  # CompoundExpandablePixelGrid, vectors, angle
├── algorithms/                       # A*, BFS
└── flow_control/                     # StateMachine, Sequencer, Delay, StepCounter
```

## Key Abstractions

### CompoundExpandablePixelGrid
- **File**: `data_structures/compound_pixel_grid.py`
- Central multi-layer grid storing ALL map information. Grows dynamically as the robot explores.
- **Resolution**: ~166.67 pixels/meter (10 pixels per quarter_tile, quarter_tile = 0.06m)
- **28 layers**: walls, detected_points, occupied, traversable, navigation_preference, traversed, seen_by_camera, seen_by_lidar, discovered, floor_color, holes, swamps, victims, etc.

### WallMapper
- **File**: `mapping/wall_mapper.py`
- Converts LIDAR point cloud → wall pixels with noise filtering and gap filling
- **`to_boolean_threshold = 3`**: must detect a point 3 times before confirming as wall
- **`delete_threshold = 1`**: single-hit points are deleted as noise
- Generates `traversable` (with 1px erosion for passage) and `navigation_preference` (gradient from walls for A* avoidance)

### Mapper
- **File**: `mapping/mapper.py`
- Orchestrates all sub-mappers in `update()` each timestep
- Call order: WallMapper → RobotMapper → FixtureMapper → FloorMapper → FixtureDetector → OccupiedMapper → ArrayFilterer

### Executor
- **File**: `executor/executor.py`
- Top-level orchestrator running the state machine loop
- States: `init` → `explore` ↔ `report_fixture` | `stuck` | `send_map` → `end`

### Agent
- **File**: `agent/agent.py`
- Chooses navigation targets via priority combiner: GoToFixtures > FollowWalls > GoToNonDiscovered
- Two stages: `explore` → `return_to_start`

## Data Flow — Mapping Pipeline

1. **LIDAR scans** → `Lidar.__update_point_clouds()` → polar to Cartesian → list of `[x, y]`
2. **Mapper.update()** receives point clouds + robot position + orientation
3. **WallMapper**: each point → `coordinates_to_grid_index()` → expand grid → increment `detected_points`; if > 3 AND not `traversed` → mark as `walls=True`; draw LIDAR beam on `seen_by_lidar` (Bresenham line); remove noise; fill 1-pixel gaps; generate navigation margins
4. **RobotMapper**: mark `traversed`, `seen_by_camera`, `discovered`
5. **FloorMapper**: camera IPM → top-down floor color grid
6. **OccupiedMapper**: `occupied = walls OR holes` (minus traversed)

## 🔍 Wall Accuracy Analysis — THE KEY FINDING

### Primary cause: Coordinate truncation (this is almost certainly your issue)

**File**: `data_structures/compound_pixel_grid.py`, line 93:
```python
def coordinates_to_grid_index(self, coordinates: np.ndarray) -> np.ndarray:
    coords = (coordinates * self.resolution).astype(int)  # ❌ TRUNCATES toward zero!
```

`.astype(int)` **truncates** toward zero (positive → floor, negative → ceil), equivalent to `np.trunc()`. This means:
- A wall at x=0.12m → grid index 20 (correct — 12cm × 166.67 px/m = 20.0)
- A wall at x=0.125m → grid index **20** instead of **21** (wrong by ~3mm)
- This is a **systematic -0.5 pixel bias** (~3mm error at every wall)

**Fix**: Use `np.round()`:
```python
coords = np.round(coordinates * self.resolution).astype(int)
```

### Secondary issues

2. **Wall thickening from moving robot**: Each LIDAR scan sees the wall from a different angle/position, so the same wall registers at 2-3 adjacent pixels. The 3-hit threshold helps but cannot fully prevent this.

3. **Gap filling (`close_wall_pixel_gaps`)**: Correctly fills only 1-pixel gaps. No overfill concern.

4. **Isolated point removal is slow**: Only runs every 100 steps (3.2 seconds). A false wall pixel persists that long.

5. **LIDAR update interval**: Every 6 timesteps (192ms). Between frames the robot moves, so wall hits from different frames have different position offsets.

6. **Detection threshold (3 hits)**: A thin wall viewed edge-on might be detected only 1-2 times and never registered. Conversely, a wall seen from many angles becomes a thick band.

## Recommended Fixes for Wall Accuracy

1. **✅ Round instead of truncate in `coordinates_to_grid_index`** — immediate improvement, zero side effects. This is the highest-impact, lowest-risk change.

2. **Increase pixel resolution** — `pixels_per_tile` from 10 to 15 or 20. Trade-off: more memory, slower A*.

3. **Reduce LIDAR update interval** — from 6 to 3 timesteps. Trade-off: more CPU.

4. **Use sub-pixel accumulation** — store hit positions with floating-point precision in a sub-pixel grid before downsampling.

---

I'm in **Explore Mode** — a codebase investigation mode. I can analyze code and produce documentation, but **I can't implement changes** here. To implement the fixes above, switch to **Act Mode** using the mode selector at the bottom of the chat.

Your exploration findings file has been saved as `project_info__1.md` in the project root directory.