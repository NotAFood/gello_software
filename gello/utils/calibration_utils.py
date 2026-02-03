"""Shared utilities for camera calibration workflows."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
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
    raise RuntimeError(
        "capture_camera_image (legacy PNG writer) has been removed; use HDF5 append functions instead"
    )


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

    # Legacy JSON session metadata writing is deprecated for HDF5-only sessions.
    # This function is intentionally left to allow callers to fail fast if still used.
    raise RuntimeError(
        "write_session_metadata has been removed: HDF5 sessions store metadata as root attributes"
    )


def init_overhead_hdf5_file(
    h5_path: Path, image_shape: tuple, joint_dim: int, metadata: dict
) -> None:
    """Create and initialize an HDF5 session file for overhead calibration.

    Args:
        h5_path: Path to the HDF5 file to create.
        image_shape: Tuple (H, W, C) describing image shape.
        joint_dim: Number of joint values per pose.
        metadata: Dict of root-level metadata to store as attributes.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    H, W, C = image_shape

    with h5py.File(h5_path, "w") as f:
        # Root attributes
        for k, v in metadata.items():
            try:
                f.attrs[k] = v
            except Exception:
                f.attrs[k] = str(v)

        # Create extendable root datasets for synchronized per-pose data
        f.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(1024,),
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "joint_angles",
            shape=(0, joint_dim),
            maxshape=(None, joint_dim),
            dtype="f4",
            chunks=(1, joint_dim),
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "pose_index",
            shape=(0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(1024,),
            compression="gzip",
            compression_opts=4,
        )


def append_frame_to_hdf5(
    h5_path: Path,
    camera_images: Dict[str, np.ndarray],
    timestamp: float,
    joint_positions: np.ndarray,
    pose_index: int,
) -> None:
    """Append one captured pose (images from one or more cameras) to the HDF5 session file.

    Args:
        h5_path: Path to HDF5 file (will be created if missing).
        camera_images: Dict mapping camera name -> RGB numpy array (H,W,3) uint8
        timestamp: Epoch seconds (float) timestamp for this capture
        joint_positions: 1D numpy array of joint positions
        pose_index: Integer pose index
    """
    # Validate camera_images
    if not camera_images:
        raise ValueError("camera_images must contain at least one camera image")

    # Lazy init: create file and datasets if missing using first image shape
    first_cam = next(iter(camera_images.items()))
    _, first_img = first_cam
    H, W, C = first_img.shape

    joint_dim = int(joint_positions.shape[0])

    if not h5_path.exists():
        meta = {"created_at": datetime.now().isoformat(), "hdf5_version": 1}
        init_overhead_hdf5_file(h5_path, (H, W, C), joint_dim, meta)

    with h5py.File(h5_path, "a") as f:
        # Ensure root datasets exist and have matching joint_dim
        if "joint_angles" not in f:
            init_overhead_hdf5_file(
                h5_path,
                (H, W, C),
                joint_dim,
                {"created_at": datetime.now().isoformat()},
            )

        # Resize root datasets by 1
        n = f["timestamps"].shape[0]
        f["timestamps"].resize((n + 1,))
        f["joint_angles"].resize((n + 1, joint_dim))
        f["pose_index"].resize((n + 1,))

        # Write per-pose root data
        f["timestamps"][n] = float(timestamp)
        f["joint_angles"][n, :] = joint_positions.astype("f4")
        f["pose_index"][n] = int(pose_index)

        # For each camera, ensure dataset and append image
        for cam_name, img in camera_images.items():
            grp = f.require_group(f"cameras/{cam_name}")
            if "images" not in grp:
                grp.create_dataset(
                    "images",
                    shape=(0, H, W, C),
                    maxshape=(None, H, W, C),
                    dtype="u1",
                    chunks=(1, H, W, C),
                    compression="gzip",
                    compression_opts=4,
                )
                grp["images"].attrs["encoding"] = "RGB"

            img_ds = grp["images"]
            img_ds.resize((img_ds.shape[0] + 1, H, W, C))
            img_ds[img_ds.shape[0] - 1] = img.astype("u1")


def finalize_hdf5_metadata(h5_path: Path, metadata_updates: dict) -> None:
    """Update root attributes with final metadata (e.g., total_poses).

    Args:
        h5_path: Path to HDF5 file
        metadata_updates: Dict of attributes to set/update
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "a") as f:
        for k, v in metadata_updates.items():
            try:
                f.attrs[k] = v
            except Exception:
                f.attrs[k] = str(v)


def load_session_from_hdf5(h5_path: Path) -> dict:
    """Load an HDF5 calibration session and return structured data.

    Returns a dict with keys:
      - 'metadata': dict of root attributes
      - 'timestamps': numpy array (N,)
      - 'joint_angles': numpy array (N, J)
      - 'pose_index': numpy array (N,)
      - 'cameras': dict mapping camera name -> images array (N, H, W, C)
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    out = {
        "metadata": {},
        "timestamps": None,
        "joint_angles": None,
        "pose_index": None,
        "cameras": {},
    }
    with h5py.File(h5_path, "r") as f:
        # Root attrs
        for k, v in f.attrs.items():
            try:
                out["metadata"][k] = v
            except Exception:
                out["metadata"][k] = str(v)

        # Root datasets
        if "timestamps" in f:
            out["timestamps"] = f["timestamps"][()]
        if "joint_angles" in f:
            out["joint_angles"] = f["joint_angles"][()]
        if "pose_index" in f:
            out["pose_index"] = f["pose_index"][()]

        # Cameras
        if "cameras" in f:
            cam_grp = f["cameras"]
            for cam_name in cam_grp:
                grp = cam_grp[cam_name]
                if "images" in grp:
                    out["cameras"][cam_name] = grp["images"][()]

    return out
