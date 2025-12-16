#!/usr/bin/env python3
"""
Capture discrete robot poses for camera calibration.

Usage:
    # Left arm only
    uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated.yaml

    # Right arm only
    uv run experiments/capture_calibration_poses.py --config-path configs/yam_auto_generated_right.yaml

Controls:
    SPACE/ENTER: Capture current pose
    Q: Quit and save all poses
"""

import atexit
import json
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("\nCleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

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


class SingleKeyCapture:
    """Minimal keyboard input handler for pose capture without pygame."""

    def __init__(self):
        self.old_settings = None
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def get_key(self) -> Optional[str]:
        """Get a single keypress if available (non-blocking)."""
        if not sys.stdin.isatty():
            return None

        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None

    def cleanup(self):
        """Restore terminal settings."""
        if self.old_settings and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


@dataclass
class Args:
    config_path: str
    """Path to the arm configuration YAML file."""

    output_dir: Path = Path("data/calibration_poses")
    """Directory to save calibration poses."""

    arm_name: Optional[str] = None
    """Optional name for the arm (e.g., 'left', 'right'). Auto-detected from config if not provided."""


def save_calibration_poses(
    poses: list, output_path: Path, arm_identifier: str, filename_tag: str
):
    """Save all captured poses to a JSON file."""
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


def print_pose_info(pose_number: int, joint_positions: np.ndarray):
    """Print current pose information."""
    print(f"\n{'=' * 60}")
    print(f"Pose #{pose_number} captured!")
    print(f"{'=' * 60}")
    print("Joint Positions:")
    for i, pos in enumerate(joint_positions):
        print(f"  Joint {i + 1}: {pos:8.4f} rad ({np.degrees(pos):8.2f}°)")
    print(f"{'=' * 60}\n")


def infer_arm_name_from_channel(channel: str) -> str:
    """Infer arm name (left/right) from a CAN channel string."""
    channel_lower = channel.lower()

    if any(tag in channel_lower for tag in ["left", "_l", "-l", " l", "l_"]):
        return "left"
    if any(tag in channel_lower for tag in ["right", "_r", "-r", " r", "r_"]):
        return "right"

    return "arm"


def resolve_config_identifier(config_path: str) -> tuple[str, str, Path]:
    """Return the arm identifier (relative path) and a filename-safe slug."""
    resolved = Path(config_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent

    try:
        relative_path = resolved.relative_to(repo_root)
    except ValueError:
        relative_path = resolved

    identifier = relative_path.as_posix()
    slug = identifier.replace("\\", "/").replace("/", "__").replace(" ", "_")
    return identifier, slug, resolved


def main():
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    arm_identifier, arm_slug, resolved_config_path = resolve_config_identifier(
        args.config_path
    )

    # Load config
    cfg = OmegaConf.to_container(OmegaConf.load(resolved_config_path), resolve=True)

    # Auto-detect arm side from config path if not provided
    arm_side = args.arm_name
    if arm_side is None:
        channel = cfg.get("robot", {}).get("channel") if isinstance(cfg, dict) else None
        if isinstance(channel, str):
            arm_side = infer_arm_name_from_channel(channel)
        else:
            config_filename = Path(args.config_path).stem
            if "right" in config_filename.lower():
                arm_side = "right"
            elif "left" in config_filename.lower():
                arm_side = "left"
            else:
                arm_side = "arm"

    print(f"\n{'=' * 60}")
    print("Calibration Pose Capture")
    print(f"Arm identifier: {arm_identifier}")
    if arm_side:
        print(f"Arm side (inferred): {arm_side}")
    print(f"{'=' * 60}")
    print(f"Config: {resolved_config_path}")
    print(f"Output: {args.output_dir}")
    print(f"\nControls:")
    print(f"  b: Capture current pose")
    print(f"  Q: Quit and save all poses")
    print(f"{'=' * 60}\n")

    # Create agent
    agent = instantiate_from_dict(cfg["agent"])

    # Create robot
    robot_cfg = cfg["robot"]
    if isinstance(robot_cfg.get("config"), str):
        robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(robot_cfg["config"]), resolve=True
        )

    robot = instantiate_from_dict(robot_cfg)

    # Setup robot connection (same as launch_yaml.py)
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

        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
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

    # Move robot to start position if specified
    move_to_start_position(env, bimanual=False, left_cfg=cfg)

    print(f"\n✓ Robot initialized: {robot.__class__.__name__}")
    print(f"✓ Agent initialized: {agent.__class__.__name__}")
    print(f"✓ Control loop: {cfg.get('hz', 30)} Hz")
    print(
        f"\nReady to capture poses! Move the robot and press SPACE/ENTER to capture.\n"
    )

    # Initialize keyboard handler
    key_handler = SingleKeyCapture()
    atexit.register(key_handler.cleanup)

    # Storage for captured poses
    captured_poses = []
    pose_count = 0

    try:
        obs = env.get_obs()

        while True:
            # Get agent action (control from leader arm)
            action = agent.act(obs)

            # Check for keyboard input
            key = key_handler.get_key()

            if key:
                if key.lower() == "q":
                    print("\n\nQuitting...")
                    break
                elif key in ["b"]:  # b key
                    # Capture current pose
                    pose_count += 1
                    joint_positions = obs["joint_positions"]

                    captured_pose = {
                        "pose_number": pose_count,
                        "timestamp": datetime.now().isoformat(),
                        "joint_positions": joint_positions.copy(),
                    }
                    captured_poses.append(captured_pose)

                    print_pose_info(pose_count, joint_positions)
                    print(
                        f"Total poses captured: {pose_count} | Press SPACE/ENTER for more, Q to quit\n"
                    )
                    # Play the terminal bell as feedback
                    print("\a", end="", flush=True)

            # Step the environment
            obs = env.step(action)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    finally:
        key_handler.cleanup()

        # Save all captured poses
        if captured_poses:
            save_calibration_poses(
                captured_poses, args.output_dir, arm_identifier, arm_slug
            )
        else:
            print("\nNo poses captured. Exiting without saving.")

        cleanup()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
