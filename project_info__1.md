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
├── run.py                            # Webots controller entry (copied as robot0Controller.py)
├── rescue_robot.py                   # User-friendly API layer (RescueRobot class)
├── student_example.py                # Student template code
├── flags.py                          # Global debug/feature flags
├── utilities.py                      # Math helpers, drawing utilities, color filter tuner
├── map_visualizer.py                 # Real-time OpenCV map visualization
├── mapping/                          # MAP BUILDING CORE
│   ├── mapper.py                     # Mapper — orchestrates all sub-mappers
│   ├── wall_mapper.py                # WallMapper — LIDAR → walls, navigation margins
│   ├── floor_mapper.py               # FloorMapper — camera IPM → floor colors, holes, swamps
│   ├── robot_mapper.py               # RobotMapper — traversed, seen_by_camera, discovered
│   ├── occupied_mapping.py           # OccupiedMapper — walls OR holes = occupied
│   ├── fixture_mapper.py             # FixtureMapper — victim proximity zones
│   └── array_filtering.py             # ArrayFilterer — noise removal
├── executor/                         # ORCHESTRATION
│   ├── executor.py                   # Executor — state machine, sequencer, mission control
│   └── stuck_detector.py             # StuckDetector — detects wheel spin without movement
├── agent/                            # NAVIGATION
│   ├── agent.py                      # Agent + SubagentPriorityCombiner
│   ├── agent_interface.py            # Abstract interfaces
│   ├── pathfinding/
│   │   ├── pathfinder.py             # A* pathfinding
│   │   └── path_smoothing.py         # Path post-processing
│   └── subagents/
│       ├── follow_walls/             # FollowWallsAgent
│       ├── go_to_fixtures/           # GoToFixturesAgent
│       ├── go_to_non_discovered/     # GoToNonDiscoveredAgent
│       └── return_to_start/          # ReturnToStartAgent
├── robot/                            # HARDWARE ABSTRACTION
│   ├── robot.py                      # Robot — top-level hardware interface
│   ├── drive_base.py                 # DriveBase + RotationManager + SmoothMovementManager
│   ├── pose_manager.py               # PoseManager — GPS/Gyroscope fusion
│   └── devices/
│       ├── lidar.py                  # Lidar sensor
│       ├── camera.py                 # Camera sensor (×3)
│       ├── gps.py                    # GPS sensor
│       ├── gyroscope.py              # Gyroscope sensor
│       ├── wheel.py                  # Motor wheel
│       ├── sensor.py                 # Base sensor class
│       └── comunicator.py            # Server communication (emitter/receiver)
├── fixture_detection/                # VISION
│   ├── fixture_detection.py          # FixtureDetector — map fixture positions
│   ├── fixture_clasification.py      # FixtureClasiffier — symbol recognition (Φ/Ψ/Ω)
│   ├── victim_clasification.py       # VictimClasification (legacy)
│   ├── color_filter.py               # ColorFilter + wall mask generation
│   └── non_fixture_filterer.py       # NonFixtureFilter — background removal
├── final_matrix_creation/            # MAP OUTPUT
│   └── final_matrix_creator.py       # WallMatrixCreator + FloorMatrixCreator + FinalMatrixCreator
├── data_structures/                  # CORE DATA TYPES
│   ├── compound_pixel_grid.py        # CompoundExpandablePixelGrid — multi-layer dynamic grid
│   ├── vectors.py                    # Position2D, Vector2D
│   └── angle.py                      # Angle class
├── algorithms/
│   └── np_bool_array/
│       ├── efficient_a_star.py       # Heap-based A* with octile heuristic
│       └── bfs.py                    # BFS pathfinding
└── flow_control/
    ├── state_machine.py              # StateMachine
    ├── sequencer.py                  # Sequencer (sequential action execution)
    ├── delay.py                      # DelayManager (non-blocking delays)
    └── step_counter.py               # StepCounter (n-step interval triggers)
```

## Key Abstractions

### CompoundExpandablePixelGrid
- **File**: `data_structures/compound_pixel_grid.py`
- **Responsibility**: Central multi-layer grid that stores ALL map information. Grows dynamically as the robot explores. Provides coordinate transformations (world meters ↔ grid index ↔ array index).
- **Resolution**: ~166.67 pixels/meter (10 pixels per quarter_tile, quarter_tile = 0.06m)
- **Key layers** (28 total): `walls`, `detected_points`, `occupied`, `traversable`, `navigation_preference`, `traversed`, `seen_by_camera`, `seen_by_lidar`, `discovered`, `floor_color`, `holes`, `swamps`, `victims`, etc.
- **Grid expansion**: Expandable in all 4 directions. Any coordinate access outside bounds triggers padding.

### WallMapper
- **File**: `mapping/wall_mapper.py`
- **Responsibility**: Converts LIDAR point cloud to wall pixels with noise filtering, gap filling, and navigation margin generation.
- **Key thresholds**: `to_boolean_threshold = 3` (detect 3× to confirm wall), `delete_threshold = 1` (remove single-hit points)
- **Gap filling**: `close_wall_pixel_gaps()` — fills 1-pixel gaps by checking flanking walls 2 pixels apart
- **Navigation margins**: Generates `traversable` (with 1px erosion to allow passage) and `navigation_preference` (gradient from walls for A* avoidance)

### Mapper
- **File**: `mapping/mapper.py`
- **Responsibility**: Orchestrates all sub-mappers in `update()`. Called every timestep.
- **Call order**: WallMapper (LIDAR) → RobotMapper (trajectory) → FixtureMapper (zones) → FloorMapper (camera colors) → FixtureDetector (victims) → OccupiedMapper → ArrayFilterer

### Executor
- **File**: `executor/executor.py`
- **Responsibility**: Top-level orchestrator. Runs the state machine loop. Manages mapping toggle, stuck detection, fixture reporting sequences, and timed map sending.
- **States**: `init` → `explore` ↔ `report_fixture` | `stuck` | `send_map` → `end`

### Agent
- **File**: `agent/agent.py`
- **Responsibility**: Chooses navigation targets via priority-based subagent combiner.
- **Priorities**: GoToFixtures > FollowWalls > GoToNonDiscovered
- **Two stages**: `explore` → `return_to_start`

### PoseManager
- **File**: `robot/pose_manager.py`
- **Responsibility**: Fuses GPS and gyroscope for position/orientation. GPS used during straight travel (more accurate), gyroscope during turns (GPS too noisy). Automatically decides sensor source.

### DriveBase
- **File**: `robot/drive_base.py`
- **Responsibility**: SmoothMovementToCoordinatesManager drives with three modes: straight (small angle error ≤3°), gentle curve (3°-30°), strong rotation (>45°).

## Data Flow — Mapping Pipeline

1. **LIDAR scans** → `Lidar.__update_point_clouds()` → polar to Cartesian → list of `[x, y]` offset from robot
2. **Mapper.update()** receives point clouds + robot position + orientation
3. **WallMapper.load_point_cloud()**:
   - Each in-bounds point: `coordinates_to_grid_index()` (world coords → grid index), expand grid, increment `detected_points` counter
   - `occupy_point()`: if `detected_points > 3` AND not `traversed` → set `walls=True`
   - Draw LIDAR beam path on `seen_by_lidar` layer (Bresenham line)
   - `filter_out_noise()`: clear points with count ≤ 1
   - `close_wall_pixel_gaps()`: fills 1-pixel holes in walls
   - `generate_navigation_margins()`: create `traversable` and `navigation_preference`
4. **RobotMapper** marks `traversed` (robot body), `seen_by_camera` (camera frustum AND LIDAR beam), `discovered` (170° forward cone AND LIDAR beam)
5. **FloorMapper** converts camera images via IPM (Inverse Perspective Mapping) to top-down floor color grid
6. **OccupiedMapper** sets `occupied = walls OR holes` (minus traversed)

## Data Flow — Final Matrix Generation

1. **WallMatrixCreator**: Takes wall pixel array → splits into 10×10px tiles → matches directional wall templates (straight/corner with rotation) → outputs 2×2 node array per tile
2. **FloorMatrixCreator**: Takes floor color array → splits into 20×20px tiles → HSV color matching → tile type codes (0=floor, 2=hole, 3=swamp, 4=checkpoint)
3. **FinalMatrixCreator**: Combines wall nodes + floor codes + start position (5) + obstacle tiles (x) → final text grid (each tile = 4×4 text nodes, type code placed at center)

## Non-Obvious Behaviors & Design Decisions

### Wall Mapping Issues (the user's concern)

1. **coordinates_to_grid_index uses truncation, not rounding**:
   ```python
   coords = (coordinates * self.resolution).astype(int)  # truncates toward zero!
   ```
   Instead of `np.round(...).astype(int)`. This introduces a systematic -0.5 pixel bias (~3mm). For a wall pixel that should be at boundary, this can misplace it by half a pixel.

2. **LIDAR data accumulation from moving robot creates wall thickening**:
   Each LIDAR scan sees the wall from a slightly different angle/position. The 3-hit threshold helps but doesn't prevent a wall from registering as 2-3 pixels wide instead of 1. The `close_wall_pixel_gaps()` method may exacerbate this.

3. **`close_wall_pixel_gaps()` fills gaps that may be actual passages**:
   ```python
   horizontal_gap[:, 1:-1] = walls[:, :-2] & walls[:, 2:]
   ```
   Checks if pixel (y, x-2) AND (y, x) are both walls → fills (y, x-1). This correctly fills 1-pixel gaps. BUT it does NOT fill gaps if there's a 3+ pixel wide opening (which would be a real corridor). This should be fine for accuracy.

4. **The `to_boolean_threshold = 3` is critical for wall precision**:
   A wall must be detected from 3 different LIDAR scans before being confirmed. A thin wall (e.g., 0.006m) viewed edge-on might be detected only 1-2 times and never registered. Conversely, a wall seen from many angles will register as a thick band.

5. **`delete_threshold = 1` is aggressive**:
   Points detected exactly once are deleted. In tight corridors, a glance detection might be the only data point for a thin obstacle.

6. **LIDAR timing**: LIDAR updates every 6 timesteps (192ms real-time at 32ms/timestep). Between LIDAR frames, the robot may have moved, so wall points from different frames will have different position offsets.

7. **The wall mapper receives `in_bounds_point_cloud` AND `out_of_bounds_point_cloud`**: In-bounds points (distance < ~0.48m) add wall hits. Out-of-bounds points (distance >= 0.48m or infinite) only draw on `seen_by_lidar` — they don't create wall pixels. This is correct behavior.

8. **RobotMapper's `traversed` overrides walls**: In `OccupiedMapper.map_occupied()`:
   ```python
   self.__grid.arrays["occupied"][self.__grid.arrays["traversed"]] = False
   ```
   If the robot drives through an area that was mistakenly marked as wall, traversed data removes it from occupied. This is an invariant: the robot physically cannot be inside a wall.

### Other Non-Obvious Behaviors

9. **PoseManager GPS/Gyroscope switching criteria**:
   - Uses GPS for orientation when: `angular_velocity < 1°/step` AND `avg_wheel_velocity >= 1` AND `wheel_speed_diff < 1`
   - Otherwise uses gyroscope
   - When switching to GPS: `gps.reset_orientation_baseline()` is called to prevent the baseline from spanning a rotation segment

10. **Fixture report deduplication is position-based, not vision-based**:
    `has_detected_victim_from_position()` checks a 10cm radius (`detected_from_radius`) around current robot position. This means if the robot returns to the same location, it won't re-report. But it could miss a different fixture near the same location.

11. **False positive cooldown in Executor**:
    If fixture is confirmed during explore but disappears during report_fixture (re-check), the position is blacklisted AND a 150-frame (~4.8s) detection cooldown is triggered. This helps avoid getting stuck on visual noise.

12. **Hole detection requires 2+ observations**:
    ```python
    confirmed_holes = self.pixel_grid.arrays["hole_detections"] >= 2
    ```
    A hole pixel must be detected as dark in ≥2 separate frames before being confirmed.

13. **Isolated point removal is slow (every 100 steps)**:
    A single false wall pixel can persist for 100 timesteps (3.2 seconds) before being cleaned up by `ArrayFilterer`.

14. **The Swamp is detected by floor color, not by robot behavior**:
    A brown-colored floor tile (HSV: H=19, S=112-141, V=32-166) triggers swamp mode. When close, `PoseManager` is forced to use gyroscope only (GPS becomes unreliable).

15. **Return path uses A* pathfinding, not behavioral**: The `ReturnToStartAgent` computes a shortest path through the discovered traversable area, while exploration agents use behavioral strategies (follow walls, go to undiscovered).

## Module Reference

| File | Purpose |
|------|---------|
| `mapping/mapper.py` | Mapper — orchestrates all sub-mappers |
| `mapping/wall_mapper.py` | WallMapper — LIDAR → walls, gap filling, navigation margins |
| `mapping/floor_mapper.py` | FloorMapper — camera IPM → floor colors, holes/swamps/checkpoints |
| `mapping/robot_mapper.py` | RobotMapper — traversed/seen_by_camera/discovered layers |
| `mapping/occupied_mapping.py` | OccupiedMapper — walls OR holes = occupied |
| `mapping/fixture_mapper.py` | FixtureMapper — victim proximity zones, duplicate prevention |
| `mapping/array_filtering.py` | ArrayFilterer — isolated point + jagged edge removal |
| `executor/executor.py` | Executor — state machine, fixture reporting sequence |
| `executor/stuck_detector.py` | StuckDetector — wheel spin without movement detection |
| `agent/agent.py` | Agent + SubagentPriorityCombiner |
| `agent/subagents/follow_walls/` | FollowWallsAgent — edge-following navigation |
| `agent/subagents/go_to_non_discovered/` | GoToNonDiscoveredAgent — explore unknown areas |
| `agent/subagents/go_to_fixtures/` | GoToFixturesAgent — approach fixture proximity zones |
| `agent/subagents/return_to_start/` | ReturnToStartAgent — A* path to start |
| `data_structures/compound_pixel_grid.py` | CompoundExpandablePixelGrid — dynamic multi-layer grid |
| `robot/robot.py` | Robot — hardware abstraction |
| `robot/pose_manager.py` | PoseManager — GPS/Gyroscope fusion |
| `robot/drive_base.py` | DriveBase — rotation + smooth movement control |
| `robot/devices/lidar.py` | Lidar — point cloud generation |
| `robot/devices/gps.py` | Gps — position + baseline orientation |
| `robot/devices/gyroscope.py` | Gyroscope — angular velocity integration + drift correction |
| `fixture_detection/fixture_detection.py` | FixtureDetector — victim mapping on grid |
| `fixture_detection/fixture_clasification.py` | FixtureClasiffier — symbol recognition |
| `final_matrix_creation/final_matrix_creator.py` | FinalMatrixCreator, WallMatrixCreator, FloorMatrixCreator |
| `flow_control/state_machine.py` | StateMachine — state machine framework |
| `flow_control/sequencer.py` | Sequencer — sequential action execution |
| `map_visualizer.py` | MapVisualizer — OpenCV real-time map display |

## Wall Accuracy Improvement — Analysis

The user reports that wall mapping is not accurate enough. Based on the code analysis, here are the key factors affecting wall precision and possible improvement directions:

### Primary cause: Coordinate truncation (likely the biggest issue)
**File**: `data_structures/compound_pixel_grid.py`, line ~93
```python
def coordinates_to_grid_index(self, coordinates: np.ndarray) -> np.ndarray:
    coords = (coordinates * self.resolution).astype(int)  # uses floor for positive, ceil for negative
```
This truncates toward zero (positive → floor, negative → ceil), equivalent to `np.trunc()`. This means:
- A wall at x=0.12m would map to grid index 20 (12cm × 166.67 px/m = 20.0 → exactly 20, okay)  
- But a wall at x=0.125m would map to 20 instead of 21
- This introduces a ~0.5 pixel systematic error (~3mm at 166.67 px/m)

**Fix**: Use `np.round()` instead of `.astype(int)`:
```python
coords = np.round(coordinates * self.resolution).astype(int)
```

### Secondary issue: Wall thickness from multiple scans
Since the robot moves between LIDAR scans, the same physical wall point can register at 2-3 adjacent pixel positions when viewed from different angles. This creates "fat walls."

### Design constraints that limit accuracy:
1. **LIDAR update interval**: 6 timesteps (~192ms) — limits how often new data is incorporated
2. **Detection threshold (3 hits)**: Good for noise, but means a wall must be seen from 3 positions before being confirmed. This is a deliberate trade-off for reliability.
3. **Gap filling may overfill in noisy conditions**: `close_wall_pixel_gaps` only fills 1-pixel gaps, which is appropriate. No overfill concern.

### Recommended Priority Fixes:
1. **Round instead of truncate in `coordinates_to_grid_index`** — immediate improvement, zero side effects
2. **Increase resolution** — `pixels_per_tile` from 10 to 15 or 20 would double/halve pixel size. Trade-off: more memory and slower A* search.
3. **Reduce LIDAR update interval** — from 6 to 3 timesteps for faster wall convergence. Trade-off: more CPU usage.
4. **Use sub-pixel accumulation** — store hit positions with floating-point precision and accumulate in a sub-pixel grid before downsampling to pixel grid.

## Suggested Reading Order

1. `mapping/wall_mapper.py` — Understand the core wall generation pipeline (this is the user's concern)
2. `data_structures/compound_pixel_grid.py` — Understand the grid system and coordinate transforms
3. `mapping/mapper.py` — See how all sub-mappers are orchestrated
4. `executor/executor.py` — Understand the mission flow and state machine
5. `robot/pose_manager.py` — Understand how position/orientation accuracy affects mapping
6. `final_matrix_creation/final_matrix_creator.py` — Understand how walls become the final submission matrix
