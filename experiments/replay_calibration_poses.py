#!/usr/bin/env python3
"""
Replay calibration poses for camera calibration data collection.

Usage:
    # Basic usage (config inferred from poses file):
    uv run experiments/replay_calibration_poses.py \
        --poses-path data/calibration_poses/calibration_poses_left_2025-12-15_12-34-56.json

    # With custom output directory:
    uv run experiments/replay_calibration_poses.py \
        --poses-path data/calibration_poses/calibration_poses_left_2025-12-15_12-34-56.json \
        --output-dir data/my_calibration_session

This script will:
1. Load the saved calibration poses
2. Initialize OAK camera (if mxid specified in config)
3. Move the robot to each pose with smooth transitions
4. Wait briefly for settling
5. Capture images from the camera
6. Ring terminal bell and print confirmation
"""

import atexit
import json
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.cameras.oak_camera import OakColorCamera
from gello.utils.launch_utils import instantiate_from_dict

# Global variables for cleanup
active_threads = []
active_servers = []
active_cameras = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("\nCleaning up resources...")

    # Stop cameras
    for camera in active_cameras:
        try:
            camera.stop()
        except Exception as e:
            print(f"Error stopping camera: {e}")

    # Close servers
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    # Join threads
    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


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
            import cv2

            # Convert RGB to BGR for cv2
            bgr_image = rgb_image[:, :, ::-1]
            cv2.imwrite(str(filepath), bgr_image)

            print(f"    Saved: {filename}")

        except Exception as e:
            print(f"    Error capturing from {cam_name}: {e}")
            success = False

    return success


def normalize_arm_identifier(raw_arm_name: str) -> tuple[str, str]:
    """Return the arm identifier and a filename-safe slug."""
    identifier = str(raw_arm_name)
    slug = identifier.replace("\\", "/").replace("/", "__").replace(" ", "_")
    return identifier, slug


def write_session_metadata(
    output_dir: Path,
    config_path: Path,
    poses_path: Path,
    arm_identifier: str,
    pose_numbers_replayed: list[int],
    start_pose_index: int,
    total_poses: int,
):
    """Persist metadata for downstream consumers of a calibration session."""

    metadata = {
        "config_path": str(config_path),
        "poses_path": str(poses_path),
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


def ring_bell():
    """Ring the terminal bell."""
    print("\a", end="", flush=True)


@dataclass
class Args:
    poses_path: str
    """Path to the calibration poses JSON file."""

    output_dir: Optional[str] = None
    """Directory to save captured images. Defaults to data/calibration_images/<timestamp>"""

    transition_duration: float = 2.0
    """Duration for smooth transition between poses in seconds."""

    settle_time: float = 0.5
    """Time to wait after reaching pose before capturing, in seconds."""

    start_pose_index: int = 0
    """Start from this pose index (0-based). Useful for resuming."""


def load_calibration_poses(poses_path: Path) -> dict:
    """Load calibration poses from JSON file."""
    with open(poses_path, "r") as f:
        data = json.load(f)

    # Convert joint positions back to numpy arrays
    for pose in data["poses"]:
        pose["joint_positions"] = np.array(pose["joint_positions"])

    return data


def resolve_config_path(config_identifier: str) -> Path:
    """Resolve config identifier (relative or absolute) to an absolute path."""
    candidate = Path(config_identifier).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / candidate).resolve()


def main():
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    # Load calibration poses
    poses_path = Path(args.poses_path)
    poses_data = load_calibration_poses(poses_path)
    poses = poses_data["poses"]

    arm_identifier_raw = poses_data["arm_name"]
    config_path = resolve_config_path(arm_identifier_raw)
    arm_identifier, arm_slug = normalize_arm_identifier(arm_identifier_raw)

    # Load robot config
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(f"data/calibration_images/{arm_slug}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_numbers_replayed: list[int] = []

    # Initialize cameras from config
    cameras = {}
    if isinstance(cfg, dict):
        mxid = cfg.get("mxid")
        if mxid:
            print("\nInitializing camera...")
            cam_name = f"wrist_{arm_slug}"
            try:
                camera = OakColorCamera(
                    name=cam_name,
                    device_mxid=mxid,
                    output_size=(640, 400),
                    fps=30,
                )
                cameras[cam_name] = camera
                active_cameras.append(camera)
                print(f"  ✓ {cam_name}: {camera.get_device_id()}")
            except Exception as e:
                print(f"  ✗ Failed to initialize {cam_name} ({mxid}): {e}")
                cleanup()
                import os

                os._exit(1)
            print()
        else:
            print("\nNo camera mxid in config, skipping camera initialization.")

    print(f"\n{'=' * 70}")
    print(f"Calibration Pose Replay - {arm_identifier}")
    print(f"{'=' * 70}")
    print(f"Config: {config_path}")
    print(f"Poses file: {poses_path}")
    print(f"Total poses: {len(poses)}")
    print(f"Starting from pose: {args.start_pose_index + 1}")
    print(f"Cameras: {len(cameras)}")
    print(f"Output directory: {output_dir}")
    print(f"Transition duration: {args.transition_duration}s")
    print(f"Settle time: {args.settle_time}s")
    print(f"{'=' * 70}\n")

    # Create robot
    robot_cfg = cfg["robot"]
    if isinstance(robot_cfg.get("config"), str):
        robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(robot_cfg["config"]), resolve=True
        )

    robot = instantiate_from_dict(robot_cfg)

    # Setup robot connection
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        active_threads.append(server_thread)
        active_servers.append(robot)

        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        hardware_port = cfg.get(
            "hardware_server_port", 6002
        )  # Use 6002 to avoid conflicts
        hardware_host = "127.0.0.1"

        try:
            server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                print(f"\n✗ Error: Port {hardware_port} is already in use.")
                print(
                    "This usually happens when another robot control script is running."
                )
                print("\nTry one of these solutions:")
                print("  1. Kill the other process using: kill_nodes.sh")
                print(
                    "  2. Use a different port by adding to your config: hardware_server_port: <port>"
                )
                print("  3. Wait a few seconds and try again")
                cleanup()
                import os

                os._exit(1)
            else:
                raise

        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        active_threads.append(server_thread)
        active_servers.append(server)

        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    print(f"\n✓ Robot initialized: {robot.__class__.__name__}")
    print(f"✓ Control loop: {cfg.get('hz', 30)} Hz\n")

    # Get current joint positions
    obs = env.get_obs()
    current_joints = obs["joint_positions"]

    print("Current joint positions:")
    for i, pos in enumerate(current_joints):
        print(f"  Joint {i + 1}: {pos:8.4f} rad")
    print()

    # Ask for confirmation before starting
    print("Ready to replay calibration poses.")
    print("The robot will move through each pose and capture images.")
    response = input("Continue? [y/N]: ")
    if response.lower() != "y":
        print("Aborted by user.")
        cleanup()
        return

    print(f"\n{'=' * 70}")
    print("Starting pose replay...")
    print(f"{'=' * 70}\n")

    # Replay each pose
    try:
        for i, pose in enumerate(poses):
            if i < args.start_pose_index:
                continue

            pose_num = pose["pose_number"]
            target_joints = pose["joint_positions"]

            pose_numbers_replayed.append(pose_num)

            print(f"\n[{i + 1}/{len(poses)}] Moving to Pose #{pose_num}...")
            print(f"  Timestamp: {pose['timestamp']}")
            print("  Target joints:")
            for j, pos in enumerate(target_joints):
                print(f"    Joint {j + 1}: {pos:8.4f} rad")

            # Smooth transition to target pose
            print(
                f"  Transitioning (duration: {args.transition_duration}s)...",
                end="",
                flush=True,
            )
            current_joints = smooth_transition_to_pose(
                env,
                current_joints,
                target_joints,
                duration_sec=args.transition_duration,
                control_hz=cfg.get("hz", 30),
            )
            print(" Done!")

            # Wait for settling
            if args.settle_time > 0:
                print(f"  Settling ({args.settle_time}s)...", end="", flush=True)
                time.sleep(args.settle_time)
                print(" Done!")

            # Capture image
            print("  Capturing image...", end="", flush=True)
            capture_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            success = capture_camera_image(
                cameras, output_dir, pose_num, capture_timestamp
            )
            if success:
                print(" Done!")
                ring_bell()
                print(f"  ✓ Pose #{pose_num} captured successfully!")
            else:
                print(" Failed!")
                print(f"  ✗ Pose #{pose_num} capture failed!")

            print(f"  {'-' * 66}")

        print(f"\n{'=' * 70}")
        print(f"✓ All {len(poses)} poses completed successfully!")
        print(f"{'=' * 70}\n")

        # Return to home position
        home_joints = cfg.get("agent", {}).get("start_joints")
        if home_joints is not None:
            print("Returning to home position...")
            home_joints = np.array(home_joints)
            current_joints = smooth_transition_to_pose(
                env,
                current_joints,
                home_joints,
                duration_sec=args.transition_duration,
                control_hz=cfg.get("hz", 30),
            )
            print("✓ Returned to home position\n")
        else:
            print("No home position defined in config, skipping return to home\n")

        # Double bell to signal completion
        ring_bell()
        time.sleep(0.2)
        ring_bell()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
        # Try to return to home even after interrupt
        try:
            home_joints = cfg.get("agent", {}).get("start_joints")
            if home_joints is not None:
                print("\nReturning to home position...")
                home_joints = np.array(home_joints)
                obs = env.get_obs()
                smooth_transition_to_pose(
                    env,
                    obs["joint_positions"],
                    home_joints,
                    duration_sec=args.transition_duration,
                    control_hz=cfg.get("hz", 30),
                )
                print("✓ Returned to home position")
        except Exception as e:
            print(f"Could not return to home: {e}")
    finally:
        try:
            write_session_metadata(
                output_dir=output_dir,
                config_path=config_path,
                poses_path=poses_path,
                arm_identifier=arm_identifier,
                pose_numbers_replayed=pose_numbers_replayed,
                start_pose_index=args.start_pose_index,
                total_poses=len(poses),
            )
        except Exception as e:
            print(f"Failed to write session metadata: {e}")

        cleanup()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
