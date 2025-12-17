"""Shared utilities for camera calibration workflows."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from gello.cameras.oak_camera import OakColorCamera


def ring_bell():
    """Ring the terminal bell."""
    print("\a", end="", flush=True)


def normalize_arm_identifier(raw_arm_name: str) -> tuple[str, str]:
    """Return the arm identifier and a filename-safe slug.

    Args:
        raw_arm_name: The raw arm identifier string

    Returns:
        Tuple of (identifier, slug) where slug is filename-safe
    """
    identifier = str(raw_arm_name)
    slug = identifier.replace("\\", "/").replace("/", "__").replace(" ", "_")
    return identifier, slug


def resolve_config_path(config_identifier: str) -> Path:
    """Resolve config identifier (relative or absolute) to an absolute path.

    Args:
        config_identifier: Config path (relative or absolute)

    Returns:
        Resolved absolute Path
    """
    candidate = Path(config_identifier).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    repo_root = Path(__file__).resolve().parent.parent.parent
    return (repo_root / candidate).resolve()


def interpolate_joint_positions(
    start: np.ndarray, end: np.ndarray, alpha: float
) -> np.ndarray:
    """Linearly interpolate between two joint configurations.

    Args:
        start: Starting joint positions
        end: Target joint positions
        alpha: Interpolation factor (0.0 = start, 1.0 = end)

    Returns:
        Interpolated joint positions
    """
    return start + alpha * (end - start)


def smooth_transition_to_pose(
    env,
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    duration_sec: float = 2.0,
    control_hz: float = 30.0,
) -> np.ndarray:
    """Smoothly transition robot from current pose to target pose.

    Uses linear interpolation with small steps to avoid jerky motion.

    Args:
        env: Robot environment
        current_joints: Current joint positions
        target_joints: Target joint positions
        duration_sec: Duration of transition in seconds
        control_hz: Control rate in Hz

    Returns:
        Final observed joint positions
    """
    num_steps = int(duration_sec * control_hz)
    obs: Dict = {}

    for step in range(num_steps + 1):
        alpha = step / num_steps
        # Use smooth easing for more natural motion (ease-in-out)
        alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)

        interpolated_joints = interpolate_joint_positions(
            current_joints, target_joints, alpha_smooth
        )
        obs = env.step(interpolated_joints)

    return obs["joint_positions"]


def capture_camera_image(
    cameras: Dict[str, OakColorCamera],
    output_dir: Path,
    pose_number: int,
    timestamp: str,
) -> bool:
    """Capture images from all cameras and save to disk.

    Args:
        cameras: Dictionary mapping camera names to OakColorCamera instances
        output_dir: Directory to save captured images
        pose_number: Current pose number for filename
        timestamp: Timestamp string for filename

    Returns:
        True if all captures succeeded, False otherwise
    """
    if not cameras:
        print("    Warning: No cameras configured, skipping capture")
        return True

    success = True
    for cam_name, camera in cameras.items():
        try:
            # Read frame from camera
            rgb_image, img_timestamp = camera.read()

            # Create filename: <camera_name>_pose_<num>_<timestamp>.png
            filename = f"{cam_name}_pose_{pose_number:03d}_{timestamp}.png"
            filepath = output_dir / filename

            # Save image (using cv2 or PIL)
            # Convert RGB to BGR for cv2
            bgr_image = rgb_image[:, :, ::-1]
            cv2.imwrite(str(filepath), bgr_image)

            print(f"    Saved: {filename}")

        except Exception as e:
            print(f"    Error capturing from {cam_name}: {e}")
            success = False

    return success


def load_calibration_poses(poses_path: Path) -> dict:
    """Load calibration poses from JSON file.

    Args:
        poses_path: Path to the calibration poses JSON file

    Returns:
        Dictionary containing poses data with joint_positions as numpy arrays
    """
    with open(poses_path, "r") as f:
        data = json.load(f)

    # Convert joint positions back to numpy arrays
    for pose in data["poses"]:
        pose["joint_positions"] = np.array(pose["joint_positions"])

    return data


def save_calibration_poses(
    poses: list, output_path: Path, arm_identifier: str, filename_tag: str
) -> Path:
    """Save all captured poses to a JSON file.

    Args:
        poses: List of pose dictionaries with keys: pose_number, timestamp, joint_positions
        output_path: Directory to save the poses file
        arm_identifier: Identifier for the arm (used in JSON)
        filename_tag: Tag to include in the filename (usually a slug)

    Returns:
        Path to the created JSON file
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = output_path / f"calibration_poses_{filename_tag}_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    poses_serializable = []
    for pose in poses:
        pose_data = {
            "pose_number": pose["pose_number"],
            "timestamp": pose["timestamp"],
            "joint_positions": pose["joint_positions"].tolist(),
        }
        poses_serializable.append(pose_data)

    # Save to JSON
    with open(filename, "w") as f:
        json.dump(
            {
                "arm_name": arm_identifier,
                "num_poses": len(poses),
                "poses": poses_serializable,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved {len(poses)} poses to {filename}")
    return filename


def write_session_metadata(
    output_dir: Path,
    config_path: Path,
    poses_path: Optional[Path],
    arm_identifier: str,
    pose_numbers_replayed: list[int],
    start_pose_index: int,
    total_poses: int,
) -> None:
    """Persist metadata for downstream consumers of a calibration session.

    Args:
        output_dir: Directory where metadata will be saved
        config_path: Path to the robot configuration file used
        poses_path: Path to the calibration poses file (can be None for capture sessions)
        arm_identifier: Identifier for the arm
        pose_numbers_replayed: List of pose numbers that were replayed
        start_pose_index: Starting index for the session (0-based)
        total_poses: Total number of poses in the file or captured
    """
    metadata = {
        "config_path": str(config_path),
        "poses_path": str(poses_path) if poses_path else None,
        "arm_identifier": arm_identifier,
        "start_pose_index": start_pose_index,
        "total_poses_in_file": total_poses,
        "pose_numbers_replayed": pose_numbers_replayed,
        "saved_at": datetime.now().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "session_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to {metadata_path}")
