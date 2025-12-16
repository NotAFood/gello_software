import argparse
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
from omegaconf import OmegaConf

# ChArUco board parameters (set to match your physical board)
SQUARES_X = 14
SQUARES_Y = 9
SQUARE_LENGTH_MM = 20
MARKER_LENGTH_MM = 15
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)


def _create_charuco_board() -> aruco.CharucoBoard:
    """Create a ChArUco board across OpenCV API variants."""

    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(  # type: ignore[attr-defined]
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
        )

    if hasattr(aruco.CharucoBoard, "create"):
        return aruco.CharucoBoard.create(  # type: ignore[attr-defined]
            SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
        )

    # Newer OpenCV exposes the constructor directly
    try:
        return aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
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

# Default world-to-board pose (identity if board frame == world frame)
T_W_C_ROBOT_POSE = np.eye(4)


def calculate_T_C_cam(T_cam_C: np.ndarray) -> np.ndarray:
    """Invert T_cam_C to obtain T_C_cam (camera pose in board frame)."""

    R_cam_C = T_cam_C[:3, :3]
    t_cam_C = T_cam_C[:3, 3]

    R_C_cam = R_cam_C.T
    t_C_cam = -R_C_cam @ t_cam_C

    T_C_cam = np.eye(4)
    T_C_cam[:3, :3] = R_C_cam
    T_C_cam[:3, 3] = t_C_cam
    return T_C_cam


def calculate_T_W_cam(T_W_C: np.ndarray, T_cam_C: np.ndarray) -> np.ndarray:
    """Compute camera pose in world frame given board pose in world."""

    T_C_cam = calculate_T_C_cam(T_cam_C)
    return T_W_C @ T_C_cam


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


def estimate_charuco_pose(
    image_path: Path, K: np.ndarray, dist_coeffs: np.ndarray, board
) -> np.ndarray | None:
    """Detect ChArUco board and estimate T_cam_C (board pose in camera frame)."""

    print(f"Processing image: {image_path}")

    if not image_path.exists():
        print(f"ERROR: Image file not found at {image_path}")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print("ERROR: Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_params = _create_detector_parameters()

    if hasattr(aruco, "interpolateCornersCharuco") and hasattr(
        aruco, "estimatePoseCharucoBoard"
    ):
        marker_corners, marker_ids, _ = _detect_markers(gray, aruco_params)
        if marker_ids is None or len(marker_ids) == 0:
            print("FAILED: No ArUco markers detected in the image.")
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
            print("FAILED: Too few ChArUco corners interpolated.")
            return None

        pose_ret, rvec_cam_C, tvec_cam_C = aruco.estimatePoseCharucoBoard(  # type: ignore[attr-defined]
            charuco_corners, charuco_ids, board, K, dist_coeffs
        )

        if not pose_ret:
            print(
                "FAILED: PnP (estimatePoseCharucoBoard) could not solve the pose."
            )
            return None

    elif hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(  # type: ignore[attr-defined]
            board, detectorParams=aruco_params
        )
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

        if charuco_ids is None or len(charuco_corners) < 4:
            print("FAILED: Too few ChArUco corners detected.")
            return None

        # Build object/image points and solve PnP manually since pose helpers are absent.
        obj_points = board.getChessboardCorners()[charuco_ids.flatten()]
        img_points = charuco_corners.reshape(-1, 2)
        success, rvec_cam_C, tvec_cam_C = cv2.solvePnP(
            obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("FAILED: PnP (solvePnP) could not solve the pose.")
            return None

    else:
        raise AttributeError("cv2.aruco ChArUco pose estimation API not found")

    R_cam_C = cv2.Rodrigues(rvec_cam_C)[0]
    T_cam_C = np.eye(4)
    T_cam_C[:3, :3] = R_cam_C
    T_cam_C[:3, 3] = tvec_cam_C.flatten()

    print(f"SUCCESS: ChArUco pose detected with {len(charuco_ids)} corners.")
    return T_cam_C


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

    print("--- ChArUco Extrinsics Calculation ---")
    print(f"Session directory: {session_dir}")

    metadata = load_session_metadata(session_dir)
    config_path = Path(metadata["config_path"]).expanduser()
    print(f"Robot config: {config_path}")

    try:
        mxid = load_mxid_from_config(config_path)
        print(f"Using mxid: {mxid}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load mxid from {config_path}") from exc

    try:
        K, dist_coeffs = get_camera_intrinsics(mxid, args.rgb_width, args.rgb_height)
    except Exception as exc:
        raise RuntimeError(
            "Unable to read camera intrinsics from DepthAI device."
        ) from exc

    print("Camera intrinsics (K):\n", K)
    print("Distortion coefficients:\n", dist_coeffs)

    images = find_images(session_dir, args.image_glob)
    print(f"Found {len(images)} image(s) matching {args.image_glob}.")

    successes = 0
    for image_path in images:
        T_cam_C = estimate_charuco_pose(image_path, K, dist_coeffs, BOARD)
        if T_cam_C is None:
            continue

        T_W_cam = calculate_T_W_cam(T_W_C_ROBOT_POSE, T_cam_C)
        successes += 1

        print("\n[Intermediate Result] T_cam_C (Chessboard in Camera Frame):")
        print(T_cam_C)
        print("\n[FINAL RESULT] T_W_cam (Camera in World Frame):")
        print(T_W_cam)

    if successes == 0:
        print("\nCould not calculate extrinsics for any image.")
    else:
        print(f"\nSuccessfully estimated extrinsics for {successes} image(s).")


if __name__ == "__main__":
    main()
