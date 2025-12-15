# Camera Calibration Workflow

Scripts to capture and replay robot poses for camera calibration.

## Overview

The calibration workflow consists of two steps:

1. **Capture calibration poses** - Manually move the robot to various poses and record joint positions
2. **Replay calibration poses** - Automatically move the robot through saved poses while capturing camera images

## Step 1: Capture Calibration Poses

Script to capture discrete robot poses for later replay with cameras.

### Usage

#### Left Arm
```bash
uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated.yaml
```

#### Right Arm
```bash
uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated_right.yaml
```

#### Custom Output Directory
```bash
uv run experiments/capture_calibration_poses.py \
  --config-path configs/yam_auto_generated.yaml \
  --output-dir my_calibration_data
```

### Controls

- **SPACE** or **ENTER**: Capture current pose
- **Q**: Quit and save all poses

### Output Format

Poses are saved to `data/calibration_poses/calibration_poses_{arm_name}_{timestamp}.json`

Example output:
```json
{
  "arm_name": "left",
  "num_poses": 10,
  "poses": [
    {
      "pose_number": 1,
      "timestamp": "2025-12-14T15:30:45.123456",
      "joint_positions": [0.0, -0.785, 0.0, -1.571, 0.0, 0.785, 1.0]
    },
    ...
  ]
}
```

Each pose contains:
- `pose_number`: Sequential pose number
- `timestamp`: ISO format timestamp
- `joint_positions`: Array of joint angles in radians

## Step 2: Replay Calibration Poses

Script to automatically move the robot through saved poses and capture camera images.

### Prerequisites

Install DepthAI for OAK camera support:
```bash
pip install depthai
```

### Finding Camera MXIDs

To find your OAK camera MXIDs:
```bash
python -c "import depthai as dai; [print(f'{d.getMxId()}') for d in dai.Device.getAllAvailableDevices()]"
```

### Usage Examples

#### Without Cameras (Testing)
```bash
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json
```

#### With Single Camera
```bash
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json \
    --camera-mxids wrist:14442C10E1F94CD800
```

#### With Multiple Cameras
```bash
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json \
    --camera-mxids wrist_left:14442C10E1F94CD800 wrist_right:14442C10E1F94CDA00
```

#### With Custom Output Directory
```bash
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json \
    --camera-mxids wrist:14442C10E1F94CD800 \
    --output-dir data/my_calibration_session
```

#### Resume from Specific Pose
```bash
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_arm_2025-12-14_23-57-55.json \
    --camera-mxids wrist:14442C10E1F94CD800 \
    --start-pose-index 5
```

### Camera MXID Format

Cameras are specified as `name:mxid` pairs:
- `wrist:14442C10E1F94CD800` - Camera named "wrist" with MXID 14442C10E1F94CD800
- `wrist_left:14442C10E1F94CD800 wrist_right:14442C10E1F94CDA00` - Two cameras

If you don't specify a name (just the MXID), it will be named `camera_0`, `camera_1`, etc.

### Output

Images are saved to `data/calibration_images/{arm_name}_{timestamp}/` with format:
```
{camera_name}_pose_{num:03d}_{timestamp}.png
```

Example filenames:
```
wrist_left_pose_001_20251214_153045.png
wrist_left_pose_002_20251214_153050.png
wrist_right_pose_001_20251214_153045.png
wrist_right_pose_002_20251214_153050.png
```

### Parameters

- `--config-path`: Path to robot configuration YAML
- `--poses-path`: Path to saved calibration poses JSON
- `--camera-mxids`: List of camera specifications (name:mxid)
- `--output-dir`: Custom output directory (optional)
- `--transition-duration`: Time for smooth transitions between poses (default: 2.0s)
- `--settle-time`: Time to wait before capturing after reaching pose (default: 0.5s)
- `--start-pose-index`: Start from specific pose index (default: 0)

### Behavior

For each pose, the script will:
1. Print pose information (number, timestamp, joint positions)
2. Smoothly transition to the target pose
3. Wait for settling
4. Capture images from all configured cameras
5. Ring terminal bell on successful capture
6. Save images with timestamped filenames

After all poses are complete, the robot returns to the home position (if defined in config).

## Complete Workflow Example

```bash
# Step 1: Capture calibration poses
uv run experiments/capture_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml

# (Move robot to various poses and press SPACE to capture each one)
# Output: data/calibration_poses/calibration_poses_left_2025-12-15_10-30-00.json

# Step 2: Find your camera MXIDs
python -c "import depthai as dai; [print(f'{d.getMxId()}') for d in dai.Device.getAllAvailableDevices()]"

# Step 3: Replay poses with cameras
uv run experiments/replay_calibration_poses.py \
    --config-path configs/yam_auto_generated.yaml \
    --poses-path data/calibration_poses/calibration_poses_left_2025-12-15_10-30-00.json \
    --camera-mxids wrist:14442C10E1F94CD800

# Output: data/calibration_images/left_2025-12-15_10-35-00/
#   wrist_pose_001_20251215_103500.png
#   wrist_pose_002_20251215_103505.png
#   ...
```
- `timestamp`: ISO format timestamp
- `joint_positions`: Array of joint positions in radians (6 arm joints + 1 gripper)

## Workflow

1. Start the script with your desired arm configuration
2. Move the robot to a calibration pose using the GELLO controller
3. Press SPACE or ENTER to capture the current joint positions
4. Repeat for all calibration poses
5. Press Q when done - all poses will be saved to a single JSON file
