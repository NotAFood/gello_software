import argparse
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
from omegaconf import OmegaConf

# --- 1. CONFIGURATION PARAMETERS (EDIT THESE) ---

# 1.1 ChArUco Board Parameters (Must match the physical board)
SQUARES_X = 14  # Number of squares in X direction
SQUARES_Y = 9   # Number of squares in Y direction
SQUARE_LENGTH_MM = 20  # Chessboard square side length (e.g., mm)
MARKER_LENGTH_MM = 15  # ArUco marker side length
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
BOARD = aruco.CharucoBoard.create(
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
)

# 1.2 Camera intrinsics resolution (matches the DepthAI calibration query)
RGB_W = 1280
RGB_H = 800

# 1.3 Default robot pose for the board (identity means board frame == world frame)
T_W_C_ROBOT_POSE = np.eye(4)


# --- 2. CORE FUNCTIONS ---


def calculate_T_C_cam(T_cam_C: np.ndarray) -> np.ndarray:
    """Invert T_cam_C to obtain T_C_cam."""

    R_cam_C = T_cam_C[:3, :3]
    t_cam_C = T_cam_C[:3, 3]

    R_C_cam = R_cam_C.T
    t_C_cam = -R_C_cam @ t_cam_C

    T_C_cam = np.eye(4)
    T_C_cam[:3, :3] = R_C_cam
    T_C_cam[:3, 3] = t_C_cam

    return T_C_cam


def calculate_T_W_cam(T_W_C: np.ndarray, T_cam_C: np.ndarray) -> np.ndarray:
    """Compute camera pose in world frame."""

    T_C_cam = calculate_T_C_cam(T_cam_C)
    return T_W_C @ T_C_cam


def load_session_metadata(session_dir: Path) -> dict:
    """Load session metadata from the calibration run."""

    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing session_metadata.json in {session_dir}."
        )

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


def get_camera_intrinsics(mxid: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
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


def estimate_charuco_pose(image_path: Path, K: np.ndarray, dist_coeffs: np.ndarray, board) -> np.ndarray | None:
    """Detect the ChArUco board and estimate T_cam_C (board pose in camera frame)."""

    print(f"Processing image: {image_path}")

    if not image_path.exists():
        print(f"ERROR: Image file not found at {image_path}")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print("ERROR: Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_params = aruco.DetectorParameters.create()

    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray, ARUCO_DICT, parameters=aruco_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        print("FAILED: No ArUco markers detected in the image.")
        return None

    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
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

    pose_ret, rvec_cam_C, tvec_cam_C = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist_coeffs
    )

    if not pose_ret:
        print("FAILED: PnP (estimatePoseCharucoBoard) could not solve the pose.")
        return None

    R_cam_C = cv2.Rodrigues(rvec_cam_C)[0]
    T_cam_C = np.eye(4)
    T_cam_C[:3, :3] = R_cam_C
    T_cam_C[:3, 3] = tvec_cam_C.flatten()

    print(f"SUCCESS: ChArUco pose detected with {len(charuco_ids)} corners.")
    return T_cam_C


def find_images(session_dir: Path, pattern: str) -> list[Path]:
    """Return sorted list of image paths matching the pattern inside session_dir."""

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
        help="Path to a calibration session directory containing images and session_metadata.json.",
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
        raise RuntimeError("Unable to read camera intrinsics from DepthAI device.") from exc

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
    main()import argparse
import json
import os
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
from omegaconf import OmegaConf

# --- 1. CONFIGURATION PARAMETERS (EDIT THESE) ---

# 1.1 ChArUco Board Parameters (Must match the physical board)
SQUARES_X = 14  # Number of squares in X direction
SQUARES_Y = 9  # Number of squares in Y direction
SQUARE_LENGTH_MM = 20  # Chessboard square side length (e.g., mm)
MARKER_LENGTH_MM = 15  # ArUco marker side length
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
BOARD = aruco.CharucoBoard.create(
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
)

# 1.2 Camera intrinsics resolution (matches the DepthAI calibration query)
RGB_W = 1280
RGB_H = 800

# 1.3 Default robot pose for the board (identity means board frame == world frame)
T_W_C_ROBOT_POSE = np.eye(4)


# --- 2. CORE FUNCTIONS ---


def calculate_T_C_cam(T_cam_C: np.ndarray) -> np.ndarray:
    """Invert T_cam_C to obtain T_C_cam."""

    R_cam_C = T_cam_C[:3, :3]
    t_cam_C = T_cam_C[:3, 3]

    R_C_cam = R_cam_C.T
    t_C_cam = -R_C_cam @ t_cam_C

    T_C_cam = np.eye(4)
    T_C_cam[:3, :3] = R_C_cam
    T_C_cam[:3, 3] = t_C_cam

    return T_C_cam


def calculate_T_W_cam(T_W_C: np.ndarray, T_cam_C: np.ndarray) -> np.ndarray:
    """Compute camera pose in world frame."""

    T_C_cam = calculate_T_C_cam(T_cam_C)
    return T_W_C @ T_C_cam


def load_session_metadata(session_dir: Path) -> dict:
    """Load session metadata from the calibration run."""

    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing session_metadata.json in {session_dir}."
        )

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


def get_camera_intrinsics(mxid: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
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


def estimate_charuco_pose(image_path: Path, K: np.ndarray, dist_coeffs: np.ndarray, board) -> np.ndarray | None:
    """Detect the ChArUco board and estimate T_cam_C (board pose in camera frame)."""

    print(f"Processing image: {image_path}")

    if not image_path.exists():
        print(f"ERROR: Image file not found at {image_path}")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print("ERROR: Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_params = aruco.DetectorParameters.create()

    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray, ARUCO_DICT, parameters=aruco_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        print("FAILED: No ArUco markers detected in the image.")
        return None

    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
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

    pose_ret, rvec_cam_C, tvec_cam_C = aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist_coeffs
    )

    if not pose_ret:
        print("FAILED: PnP (estimatePoseCharucoBoard) could not solve the pose.")
        return None

    R_cam_C = cv2.Rodrigues(rvec_cam_C)[0]
    T_cam_C = np.eye(4)
    T_cam_C[:3, :3] = R_cam_C
    T_cam_C[:3, 3] = tvec_cam_C.flatten()

    print(f"SUCCESS: ChArUco pose detected with {len(charuco_ids)} corners.")
    return T_cam_C


def find_images(session_dir: Path, pattern: str) -> list[Path]:
    """Return sorted list of image paths matching the pattern inside session_dir."""

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
        help="Path to a calibration session directory containing images and session_metadata.json.",
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
        raise RuntimeError("Unable to read camera intrinsics from DepthAI device.") from exc

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
    main()import argparse
import json
import os
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
from omegaconf import OmegaConf

# --- 1. CONFIGURATION PARAMETERS (EDIT THESE) ---
# NOTE: Replace these with the actual paths and values for your setup.

# 1.1 ChArUco Board Parameters (Must match the physical board)
SQUARES_X = 14  # Number of squares in X direction
SQUARES_Y = 9  # Number of squares in Y direction
SQUARE_LENGTH_MM = (
    20  # Length of a chessboard square side (in your world units, e.g., mm)
)
MARKER_LENGTH_MM = 15  # Length of the ArUco marker side (e.g., 75% of square length)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
BOARD = aruco.CharucoBoard.create(
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH_MM, MARKER_LENGTH_MM, ARUCO_DICT
)

RGB_W = 1280
RGB_H = 800

# T_W_C: The 4x4 Homogeneous Transformation Matrix of the Chessboard (C)
# relative to the World/Robot Base (W) Frame, obtained from your robot's FK.
# This example uses an identity matrix (board is at the world origin)
T_W_C_ROBOT_POSE = np.eye(4)
# Example: Robot places the board at (X=500mm, Y=200mm, Z=0mm) with no rotation
# T_W_C_ROBOT_POSE[:3, 3] = [500.0, 200.0, 0.0]


# --- 2. CORE FUNCTIONS ---


def calculate_T_C_cam(T_cam_C):
    """
    Calculates the inverse transformation T_C_cam from T_cam_C.
    T_C_cam is the pose of the Camera in the Chessboard frame.
    """
    R_cam_C = T_cam_C[:3, :3]
    t_cam_C = T_cam_C[:3, 3]

    # R_inv = R.T
    R_C_cam = R_cam_C.T
    # t_inv = -R_inv @ t
    t_C_cam = -R_C_cam @ t_cam_C

    T_C_cam = np.eye(4)
    T_C_cam[:3, :3] = R_C_cam
    T_C_cam[:3, 3] = t_C_cam

    return T_C_cam


def calculate_T_W_cam(T_W_C, T_cam_C):
    """
    Calculates the final Camera Extrinsic Matrix T_W_cam (Camera in World Frame).
    T_W_cam = T_W_C @ T_C_cam
    """
    T_C_cam = calculate_T_C_cam(T_cam_C)
    T_W_cam = T_W_C @ T_C_cam
    return T_W_cam


def load_session_metadata(session_dir: Path) -> dict:
    """Load session metadata from the calibration run."""

    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing session_metadata.json in {session_dir}."
        )

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


def get_camera_intrinsics(mxid: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
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


def estimate_charuco_pose(image_path, K, dist_coeffs, board):
    """
    Detects the ChArUco board and estimates its pose (T_cam_C) relative to the camera frame.
    """
    print(f"Processing image: {image_path}")

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoParams = aruco.DetectorParameters.create()

    # 1. Detect the ArUco markers
    marker_corners, marker_ids, _ = aruco.detectMarkers(
        gray, ARUCO_DICT, parameters=arucoParams
    )

    if marker_ids is not None and len(marker_ids) > 0:
        # 2. Interpolate the ChArUco corners
        # Note: Using K and dist_coeffs here helps refine corner location accuracy
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
    def find_images(session_dir: Path, pattern: str) -> list[Path]:
        """Return sorted list of image paths matching the pattern inside session_dir."""

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
            help="Path to a calibration session directory containing images and session_metadata.json.",
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


            board,
        args = parse_args()
        session_dir = args.session_dir


        def find_images(session_dir: Path, pattern: str) -> list[Path]:
            """Return sorted list of image paths matching the pattern inside session_dir."""

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
                help="Path to a calibration session directory containing images and session_metadata.json.",
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
                raise RuntimeError("Unable to read camera intrinsics from DepthAI device.") from exc

            print("Camera intrinsics (K):\n", K)
            print("Distortion coefficients:\n", dist_coeffs)

            images = find_images(session_dir, args.image_glob)
            print(f"Found {len(images)} image(s) matching {args.image_glob}.")

            successes = 0
            for image_path in images:
                T_cam_C = estimate_charuco_pose(str(image_path), K, dist_coeffs, BOARD)
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

        print("--- ChArUco Extrinsics Calculation ---")
        print(f"Session directory: {session_dir}")

        metadata = load_session_metadata(session_dir)
            main()
            print("\n[Intermediate Result] T_cam_C (Chessboard in Camera Frame):")
            print(T_cam_C)
            print("\n[FINAL RESULT] T_W_cam (Camera in World Frame):")
            print(T_W_cam)

        if successes == 0:
            print("\nCould not calculate extrinsics for any image.")
        else:
            print(f"\nSuccessfully estimated extrinsics for {successes} image(s).")
    else:
        print("FAILED: No ArUco markers detected in the image.")
        return None


# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    print("--- ChArUco Extrinsics Calculation ---")

    # Step A: Get T_cam_C from the image using ChArUco PnP
    T_cam_C = estimate_charuco_pose(IMAGE_PATH, K, DIST_COEFFS, BOARD)

    if T_cam_C is not None:
        print("\n[Intermediate Result] T_cam_C (Chessboard in Camera Frame):")
        print(T_cam_C)

        # Step B: Calculate the final T_W_cam using the robot's pose
        T_W_cam = calculate_T_W_cam(T_W_C_ROBOT_POSE, T_cam_C)

        print("\n========================================================")
        print("[FINAL RESULT] Camera Extrinsic Matrix T_W_cam (Camera in World Frame):")
        print(T_W_cam)
        print("========================================================")

        # Optional: Verify the pose by extracting Euler angles and position
        R_W_cam = T_W_cam[:3, :3]
        t_W_cam = T_W_cam[:3, 3]

        # Note: Converting R to Euler angles is complex and depends on rotation order (e.g., ZYX)
        # Always use the matrix T_W_cam for 3D transformations.

        print(f"\nCamera Position in World Frame (X, Y, Z, in mm):")
        print(t_W_cam)

    else:
        print("\nCould not calculate extrinsics due to failure in pose estimation.")
