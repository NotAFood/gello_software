# Camera Calibration System for GELLO Robot

## Overview
This document describes the discrete pose capture and replay system for camera calibration with YAM robotic arms.

## Robot Control Architecture

### Core Components
- **Robot Interface**: `gello/robots/robot.py` defines the `Robot` protocol with methods:
  - `num_dofs()` - Get number of degrees of freedom
  - `get_joint_state()` - Get current joint positions
  - `command_joint_state(joints)` - Command joint positions
  - `get_observations()` - Get full observation dict (joint_positions, joint_velocities, ee_pos_quat, gripper_position)

- **YAM Robot**: `gello/robots/yam.py` - Hardware implementation for YAM arms via CAN bus (i2rt library)
  - 7 DOFs: 6 arm joints + 1 gripper
  - Note: `ee_pos_quat` currently returns zeros (FK not implemented)

- **GELLO Agent**: `gello/agents/gello_agent.py` - Leader arm control via Dynamixel servos
  - `act(obs)` returns joint positions from leader arm
  - Leader arm mirrors operator's hand movements to control follower robot

- **Environment**: `gello/env.py` - `RobotEnv` class
  - `step(joints)` - Command robot and return observations
  - `get_obs()` - Get current robot state + optional camera data
  - Rate-limited control loop via `Rate` class

- **ZMQ Communication**: `gello/zmq_core/robot_node.py`
  - `ZMQServerRobot` - Wraps robot hardware in network server
  - `ZMQClientRobot` - Client implements Robot protocol over network
  - Allows robot control from separate processes
  - Default ports: 5556 (sim), 6001-6002 (hardware)

### Launch Infrastructure
- **Config System**: Uses OmegaConf YAML configs in `configs/`
  - `yam_auto_generated.yaml` - Left arm config
  - `yam_auto_generated_right.yaml` - Right arm config
  - Configs specify robot, agent, start_joints (home position), hz (control rate)

- **Launch Utils**: `gello/utils/launch_utils.py`
  - `instantiate_from_dict(cfg)` - Creates objects from config dicts
  - `move_to_start_position(env, bimanual, left_cfg, right_cfg)` - Moves robot to home

- **Main Launch**: `experiments/launch_yaml.py`
  - Supports single arm or bimanual operation
  - Creates ZMQ servers for hardware robots
  - Optional `--use-save-interface` for continuous data recording

## Calibration Pose Capture System

### Capture Script (`experiments/capture_calibration_poses.py`)
Captures discrete robot poses for camera calibration.

**Features:**
- Single arm operation (left or right via config)
- Interactive GELLO control to position robot
- Discrete pose capture on keypress
- Saves joint positions only (no camera data)

**Controls:**
- **SPACE** or **ENTER**: Capture current pose
- **Q**: Quit and save all poses

**Output Format:**
- Saved to: `data/calibration_poses/calibration_poses_{arm}_{timestamp}.json`
- JSON structure:
```json
{
  "arm_name": "left",
  "num_poses": 10,
  "poses": [
    {
      "pose_number": 1,
      "timestamp": "2025-12-14T23:57:25.829204",
      "joint_positions": [0.0, -0.785, 0.0, -1.571, 0.0, 0.785, 1.0]
    }
  ]
}
```

**Usage:**
```bash
# Left arm
uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated.yaml

# Right arm
uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated_right.yaml
```

### Replay Script (`experiments/replay_calibration_poses.py`)
Replays saved calibration poses with smooth transitions for camera data collection.

**Features:**
- Loads JSON pose file
- Smooth interpolated transitions (ease-in-out curve)
- Configurable timing for safe motion
- Camera capture placeholder at each pose
- Terminal bell feedback
- Returns to home position after completion
- Double bell at end to signal completion

**Key Functions:**
- `smooth_transition_to_pose()` - Interpolates between poses over duration
  - Uses ease-in-out curve: `alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)`
  - Protects hardware from jerky motion
  
- `capture_camera_image()` - **PLACEHOLDER** at line ~145
  - Replace with actual camera capture code
  - Should trigger camera(s), save images, return success status

**Parameters:**
- `--config-path`: Robot config YAML
- `--poses-path`: Calibration poses JSON file
- `--transition-duration`: Seconds for pose transitions (default: 2.0)
- `--settle-time`: Wait time before capture (default: 0.5)
- `--start-pose-index`: Resume from specific pose (0-based)

**Usage:**
```bash
uv run experiments/replay_calibration_poses.py \
  --config-path configs/yam_auto_generated.yaml \
  --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json \
  --transition-duration 2.0 \
  --settle-time 0.5
```

**Safety Features:**
- Confirmation prompt before starting motion
- Returns to home on normal completion
- Attempts to return home even on Ctrl+C interrupt
- Smooth transitions prevent hardware damage

## Common Workflows

### Full Calibration Data Collection

1. **Capture poses** (manual positioning with GELLO):
```bash
uv run experiments/capture_calibration_poses.py \
  --config-path configs/yam_auto_generated.yaml
# Move robot to each calibration pose, press SPACE to capture
# Press Q when done
```

2. **Replay poses** (automatic with camera capture):
```bash
./kill_nodes.sh  # Clean up any previous processes
uv run experiments/replay_calibration_poses.py \
  --config-path configs/yam_auto_generated.yaml \
  --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json
```

3. **Integrate camera code**:
- Edit `capture_camera_image()` function in `replay_calibration_poses.py` (~line 145)
- Add camera initialization, image capture, and file saving
- Return `True` on success, `False` on failure

## Important Technical Notes

### ZMQ Process Management
- Hardware robots run in non-daemon background threads with infinite `serve()` loops
- Processes won't exit naturally - require `os._exit(0)` for forced termination
- Always run `./kill_nodes.sh` or `killall python3` before starting new sessions
- Port conflicts occur if previous sessions didn't clean up properly
- Default hardware ports: 6001-6002 (6002 used in replay to avoid conflicts)

### Data Formats
- Joint positions: 7 values in radians (6 arm joints + 1 gripper)
- Gripper: ~1.0 = open, ~0.0 = closed
- Control loop: typically 30 Hz
- Home position: Defined by `start_joints` in agent config

### Timing Considerations
- `transition_duration`: Too fast can damage hardware, too slow wastes time
- `settle_time`: Needed for vibration damping before camera capture
- Control rate: From config `hz` field, typically 30 Hz
- Interpolation steps: `duration * control_hz` steps per transition

### Error Handling
- Port conflicts: Clear error message with solutions
- Keyboard interrupt: Graceful shutdown with home return attempt
- Missing home position: Skips return-to-home step
- Failed captures: Logged but doesn't stop execution

## Future Integration Points

### Camera Integration (`capture_camera_image()`)
Replace placeholder with:
1. Camera driver initialization (RealSense, etc.)
2. Image capture trigger
3. Save with metadata: `{pose_number}_{timestamp}.png`
4. Optional: Save intrinsics/extrinsics
5. Error handling and retry logic

### Additional Features to Consider
- Multi-camera support (multiple captures per pose)
- Image quality verification before proceeding
- Automatic lighting adjustments
- Pose verification (compare actual vs target joints)
- Resume capability from failed captures
- Calibration pattern detection feedback
