import argparse
import json
import re
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
from i2rt.robots.kinematics import Kinematics
from omegaconf import OmegaConf

# ChArUco board parameters (set to match your physical board)
SQUARES_X = 14
SQUARES_Y = 9
SQUARE_LENGTH_M = 0.020  # 20mm in meters
MARKER_LENGTH_M = 0.015  # 15mm in meters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# YAM robot XML path and FK site name
YAM_XML_PATH = "third_party/mujoco_menagerie/i2rt_yam/yam.xml"
YAM_SITE_NAME = "grasp_site"  # End-effector site for FK


def _create_charuco_board() -> aruco.CharucoBoard:
    """Create a ChArUco board across OpenCV API variants.

    Note: Board dimensions are in meters to match FK output units.
    """

    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(  # type: ignore[attr-defined]
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, ARUCO_DICT
        )

    if hasattr(aruco.CharucoBoard, "create"):
        return aruco.CharucoBoard.create(  # type: ignore[attr-defined]
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, ARUCO_DICT
        )

    # Newer OpenCV exposes the constructor directly
    try:
        return aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, ARUCO_DICT
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise AttributeError(
            "cv2.aruco CharucoBoard factory not found in this OpenCV build"
        ) from exc


def _create_detector_parameters():
    """Create detector parameters across OpenCV API variants."""

    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()  # type: ignore[attr-defined]

    if hasattr(aruco.DetectorParameters, "create"):
        return aruco.DetectorParameters.create()  # type: ignore[attr-defined]

    # Newer OpenCV exposes the constructor directly
    try:
        return aruco.DetectorParameters()
    except Exception as exc:  # pragma: no cover - defensive
        raise AttributeError(
            "cv2.aruco DetectorParameters factory not found in this OpenCV build"
        ) from exc


def _detect_markers(gray: np.ndarray, aruco_params):
    """Detect markers across OpenCV aruco API variants."""

    if hasattr(aruco, "detectMarkers"):
        return aruco.detectMarkers(gray, ARUCO_DICT, parameters=aruco_params)  # type: ignore[attr-defined]

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(ARUCO_DICT, aruco_params)  # type: ignore[attr-defined]
        return detector.detectMarkers(gray)

    raise AttributeError(
        "cv2.aruco marker detection API not found in this OpenCV build"
    )


BOARD = _create_charuco_board()

# Resolution used when reading intrinsics from the DepthAI device
RGB_W = 1280
RGB_H = 800


def load_session_metadata(session_dir: Path) -> dict:
    """Load session metadata.json from the session directory."""

    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing session_metadata.json in {session_dir}")

    with open(metadata_path, "r") as f:
        return json.load(f)


def load_mxid_from_config(config_path: Path) -> str:
    """Extract mxid from the robot config referenced in metadata."""

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Robot config did not resolve to a dictionary.")

    mxid = cfg.get("mxid")
    if not mxid:
        raise ValueError("mxid not found in robot config.")

    return str(mxid)


def get_camera_intrinsics(
    mxid: str, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read intrinsics and distortion coefficients from the DepthAI device."""

    rgb_socket = (
        dai.CameraBoardSocket.CAM_A
        if hasattr(dai.CameraBoardSocket, "CAM_A")
        else getattr(dai.CameraBoardSocket, "RGB", dai.CameraBoardSocket.AUTO)
    )

    with dai.Device(mxid) as device:
        calib_data = device.readCalibration()
        intrinsics = np.array(calib_data.getCameraIntrinsics(rgb_socket, width, height))
        distortion = np.array(calib_data.getDistortionCoefficients(rgb_socket))

    return intrinsics, distortion


def load_calibration_poses(poses_path: Path) -> dict:
    """Load calibration poses from JSON file.

    Returns:
        Dictionary with 'arm_name', 'num_poses', and 'poses' (list of pose dicts).
        Each pose has 'pose_number', 'timestamp', and 'joint_positions' (np.ndarray).
    """
    with open(poses_path, "r") as f:
        data = json.load(f)

    # Convert joint positions to numpy arrays
    for pose in data["poses"]:
        pose["joint_positions"] = np.array(pose["joint_positions"])

    return data


def extract_pose_number_from_filename(image_path: Path) -> int | None:
    """Extract pose number from image filename like 'wrist_*_pose_003_*.png'."""
    match = re.search(r"pose_(\d+)", image_path.name)
    if match:
        return int(match.group(1))
    return None


def estimate_charuco_pose(
    image_path: Path, K: np.ndarray, dist_coeffs: np.ndarray, board
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect ChArUco board and estimate its pose in camera frame.

    Returns:
        Tuple of (R_cam_board, t_cam_board) or None if detection failed.
        R_cam_board: 3×3 rotation matrix
        t_cam_board: 3×1 translation vector in meters
    """

    print(f"  Processing image: {image_path.name}")

    if not image_path.exists():
        print(f"    ERROR: Image file not found at {image_path}")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print("    ERROR: Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_params = _create_detector_parameters()

    if hasattr(aruco, "interpolateCornersCharuco") and hasattr(
        aruco, "estimatePoseCharucoBoard"
    ):
        marker_corners, marker_ids, _ = _detect_markers(gray, aruco_params)
        if marker_ids is None or len(marker_ids) == 0:
            print("    FAILED: No ArUco markers detected.")
            return None

        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(  # type: ignore[attr-defined]
            marker_corners,
            marker_ids,
            gray,
            board,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
        )

        if charuco_ids is None or len(charuco_corners) < 4:
            print("    FAILED: Too few ChArUco corners interpolated.")
            return None

        pose_ret, rvec_cam_board, tvec_cam_board = aruco.estimatePoseCharucoBoard(  # type: ignore[attr-defined]
            charuco_corners, charuco_ids, board, K, dist_coeffs
        )

        if not pose_ret:
            print("    FAILED: PnP could not solve the pose.")
            return None

    elif hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(  # type: ignore[attr-defined]
            board, detectorParams=aruco_params
        )
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

        if charuco_ids is None or len(charuco_corners) < 4:
            print("    FAILED: Too few ChArUco corners detected.")
            return None

        # Build object/image points and solve PnP manually
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
        img_points = charuco_corners.reshape(-1, 2)
        success, rvec_cam_board, tvec_cam_board = cv2.solvePnP(
            obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("    FAILED: PnP could not solve the pose.")
            return None

    else:
        raise AttributeError("cv2.aruco ChArUco pose estimation API not found")

    R_cam_board = cv2.Rodrigues(rvec_cam_board)[0]
    t_cam_board = tvec_cam_board.flatten()

    print(f"    SUCCESS: Detected {len(charuco_ids)} corners")
    return R_cam_board, t_cam_board


def find_images(session_dir: Path, pattern: str) -> list[Path]:
    """Return sorted list of images matching the pattern inside session_dir."""

    images = sorted(session_dir.glob(pattern))
    if not images:
        raise FileNotFoundError(
            f"No images found in {session_dir} matching pattern '{pattern}'."
        )
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera extrinsics from a calibration session directory.",
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Calibration session directory with images and session_metadata.json.",
    )
    parser.add_argument(
        "--image-glob",
        default="*.png",
        help="Glob pattern for calibration images inside the session directory.",
    )
    parser.add_argument(
        "--rgb-width",
        type=int,
        default=RGB_W,
        help="RGB image width used when reading intrinsics from the device.",
    )
    parser.add_argument(
        "--rgb-height",
        type=int,
        default=RGB_H,
        help="RGB image height used when reading intrinsics from the device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = args.session_dir

    print("\n" + "=" * 80)
    print("Hand-Eye Calibration using cv2.calibrateHandEye")
    print("=" * 80)
    print(f"Session directory: {session_dir}\n")

    # Load session metadata
    metadata = load_session_metadata(session_dir)
    config_path = Path(metadata["config_path"]).expanduser().resolve()
    poses_path = Path(metadata["poses_path"]).expanduser().resolve()

    print(f"Robot config: {config_path}")
    print(f"Poses file: {poses_path}\n")

    # Load camera intrinsics
    try:
        mxid = load_mxid_from_config(config_path)
        print(f"Camera mxid: {mxid}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load mxid from {config_path}") from exc

    try:
        K, dist_coeffs = get_camera_intrinsics(mxid, args.rgb_width, args.rgb_height)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read camera intrinsics from DepthAI device."
        ) from exc

    print(f"\nCamera intrinsics (K):\n{K}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}\n")

    # Load calibration poses (joint angles)
    poses_data = load_calibration_poses(poses_path)
    poses = poses_data["poses"]
    print(f"Loaded {len(poses)} calibration poses\n")

    # Initialize forward kinematics
    repo_root = Path(__file__).resolve().parent.parent
    xml_path = repo_root / YAM_XML_PATH
    if not xml_path.exists():
        raise FileNotFoundError(f"YAM XML not found at {xml_path}")

    print(f"Initializing FK with: {xml_path}")
    print(f"End-effector site: {YAM_SITE_NAME}\n")
    kinematics = Kinematics(str(xml_path), YAM_SITE_NAME)

    # Find all images
    images = find_images(session_dir, args.image_glob)
    print(f"Found {len(images)} image(s) matching '{args.image_glob}'\n")

    # Collect hand-eye calibration pairs
    R_base_gripper_list = []  # Robot base → gripper transforms (from FK)
    t_base_gripper_list = []
    R_cam_board_list = []  # Camera → board transforms (from detection)
    t_cam_board_list = []

    print("=" * 80)
    print("Processing images...")
    print("=" * 80)

    for image_path in images:
        pose_num = extract_pose_number_from_filename(image_path)
        if pose_num is None:
            print(f"⚠ Skipping {image_path.name}: cannot extract pose number")
            continue

        # Find matching pose
        matching_pose = None
        for pose in poses:
            if pose["pose_number"] == pose_num:
                matching_pose = pose
                break

        if matching_pose is None:
            print(f"⚠ Skipping {image_path.name}: no matching pose #{pose_num}")
            continue

        print(f"\nPose #{pose_num}:")

        # Compute FK: T_base_gripper
        # YAM MuJoCo model has 8 DOFs: 6 arm joints + 2 gripper fingers
        # We need all joint positions, but only the first 6 affect the end-effector pose
        joint_positions = matching_pose["joint_positions"]
        if len(joint_positions) == 7:
            # Format: [j1, j2, j3, j4, j5, j6, gripper]
            # Convert to MuJoCo format: [j1, j2, j3, j4, j5, j6, left_finger, right_finger]
            # The gripper value represents the left finger; right finger is negated (from equality constraint)
            arm_joints = joint_positions[:6]
            gripper_pos = joint_positions[6]
            joint_angles = np.concatenate([arm_joints, [gripper_pos, -gripper_pos]])
        else:
            # Fallback: pad with zeros if unexpected format
            joint_angles = np.zeros(8)
            joint_angles[: len(joint_positions)] = joint_positions

        T_base_gripper = kinematics.fk(joint_angles)
        R_base_gripper = T_base_gripper[:3, :3]
        t_base_gripper = T_base_gripper[:3, 3]

        # Detect ChArUco board: T_cam_board
        detection_result = estimate_charuco_pose(image_path, K, dist_coeffs, BOARD)
        if detection_result is None:
            print(f"  ⚠ Skipping pose #{pose_num}: board detection failed")
            continue

        R_cam_board, t_cam_board = detection_result

        # Add to lists
        R_base_gripper_list.append(R_base_gripper)
        t_base_gripper_list.append(t_base_gripper.reshape(3, 1))
        R_cam_board_list.append(R_cam_board)
        t_cam_board_list.append(t_cam_board.reshape(3, 1))

        # Compute board position in base frame for verification
        # T_base_board = T_base_gripper @ T_gripper_cam @ T_cam_board
        T_cam_board_full = np.eye(4)
        T_cam_board_full[:3, :3] = R_cam_board
        T_cam_board_full[:3, 3] = t_cam_board

        # Note: We don't have T_gripper_cam yet, but we can show board in camera frame
        print(
            f"  Board in camera frame: [{t_cam_board[0]:.3f}, {t_cam_board[1]:.3f}, {t_cam_board[2]:.3f}] m"
        )
        print(f"  ✓ Added to calibration set ({len(R_base_gripper_list)} pairs total)")

    print(f"\n{'=' * 80}")
    print(f"Collected {len(R_base_gripper_list)} valid calibration pairs")
    print("=" * 80)

    if len(R_base_gripper_list) < 3:
        print("\n❌ ERROR: Need at least 3 valid pairs for hand-eye calibration")
        return

    # Run hand-eye calibration
    print("\nRunning cv2.calibrateHandEye (Tsai method)...\n")

    R_gripper_cam, t_gripper_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_gripper_list,
        t_gripper2base=t_base_gripper_list,
        R_target2cam=R_cam_board_list,
        t_target2cam=t_cam_board_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # Build full transform
    T_gripper_cam = np.eye(4)
    T_gripper_cam[:3, :3] = R_gripper_cam
    T_gripper_cam[:3, 3] = t_gripper_cam.flatten()

    print("=" * 80)
    print("CALIBRATION RESULT: T_gripper_cam (Camera in Gripper Frame)")
    print("=" * 80)
    print(f"\n{T_gripper_cam}\n")
    print(f"Translation (meters): {t_gripper_cam.flatten()}")
    print(f"Rotation matrix:\n{R_gripper_cam}\n")

    # Verify calibration quality by computing board positions
    print("=" * 80)
    print("VERIFICATION: Board positions in base frame")
    print("=" * 80)
    print("(Should be consistent across all poses if calibration is accurate)\n")

    board_positions_base = []
    for i, (R_base_grip, t_base_grip, R_cam_brd, t_cam_brd) in enumerate(
        zip(
            R_base_gripper_list, t_base_gripper_list, R_cam_board_list, t_cam_board_list
        ),
        1,
    ):
        # Build full transforms
        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = R_base_grip
        T_base_gripper[:3, 3] = t_base_grip.flatten()

        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cam_brd
        T_cam_board[:3, 3] = t_cam_brd.flatten()

        # T_base_board = T_base_gripper @ T_gripper_cam @ T_cam_board
        T_base_board = T_base_gripper @ T_gripper_cam @ T_cam_board

        board_pos = T_base_board[:3, 3]
        board_positions_base.append(board_pos)
        print(
            f"  Pose #{i:2d}: [{board_pos[0]:7.4f}, {board_pos[1]:7.4f}, {board_pos[2]:7.4f}] m"
        )

    # Compute statistics
    board_positions_base = np.array(board_positions_base)
    mean_pos = np.mean(board_positions_base, axis=0)
    std_pos = np.std(board_positions_base, axis=0)
    max_deviation = np.max(np.linalg.norm(board_positions_base - mean_pos, axis=1))

    print(
        f"\n  Mean position: [{mean_pos[0]:7.4f}, {mean_pos[1]:7.4f}, {mean_pos[2]:7.4f}] m"
    )
    print(
        f"  Std deviation: [{std_pos[0]:7.4f}, {std_pos[1]:7.4f}, {std_pos[2]:7.4f}] m"
    )
    print(f"  Max deviation from mean: {max_deviation:.4f} m")
    print(f"\n  (Lower values indicate better calibration quality)")
    print("=" * 80 + "\n")

    # Save result
    output_path = session_dir / "hand_eye_calibration.json"
    result = {
        "T_gripper_cam": T_gripper_cam.tolist(),
        "R_gripper_cam": R_gripper_cam.tolist(),
        "t_gripper_cam": t_gripper_cam.flatten().tolist(),
        "num_pairs": len(R_base_gripper_list),
        "method": "cv2.CALIB_HAND_EYE_TSAI",
        "board_square_size_m": SQUARE_LENGTH_M,
        "board_marker_size_m": MARKER_LENGTH_M,
        "board_dimensions": [SQUARES_X, SQUARES_Y],
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved calibration to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
