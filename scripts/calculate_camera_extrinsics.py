import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from i2rt.robots.kinematics import Kinematics

# Chessboard parameters (set to match your physical board)
CHESSBOARD_ROWS = 4  # Number of internal corners along rows
CHESSBOARD_COLS = 6  # Number of internal corners along columns
CHECKER_SIZE_M = 0.03  # Size of each checker square in meters (30mm)

# YAM robot XML path and FK site name
YAM_XML_PATH = "third_party/mujoco_menagerie/i2rt_yam/yam.xml"
YAM_SITE_NAME = "grasp_site"  # End-effector site for FK


def get_intrinsics_from_hdf5(h5_path: Path, cam_name: str = "overhead") -> tuple[np.ndarray, np.ndarray] | None:
    """Load camera intrinsics from HDF5 file if stored.

    Args:
        h5_path: Path to HDF5 file
        cam_name: Camera name to load intrinsics for (default: "overhead")

    Returns:
        Tuple of (camera_matrix, dist_coeffs) or None if not found
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Try to load from root attributes first
            fx_key = f"{cam_name}_intrinsics_fx"
            if fx_key in f.attrs:
                fx = float(f.attrs[fx_key])
                fy = float(f.attrs[f"{cam_name}_intrinsics_fy"])
                ppx = float(f.attrs[f"{cam_name}_intrinsics_ppx"])
                ppy = float(f.attrs[f"{cam_name}_intrinsics_ppy"])
                dist_str = f.attrs.get(f"{cam_name}_distortion_coeffs", "[]")
                dist_coeffs = np.array(json.loads(dist_str), dtype=np.float32)

                camera_matrix = np.array(
                    [
                        [fx, 0, ppx],
                        [0, fy, ppy],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )

                print(
                    f"Loaded {cam_name} intrinsics from HDF5: fx={fx:.2f}, fy={fy:.2f}, "
                    f"ppx={ppx:.2f}, ppy={ppy:.2f}"
                )
                return camera_matrix, dist_coeffs

            # Try to load from camera group attributes
            cam_grp = f.get(f"cameras/{cam_name}")
            if cam_grp and f"{cam_name}_fx" in cam_grp.attrs:
                fx = float(cam_grp.attrs[f"{cam_name}_fx"])
                fy = float(cam_grp.attrs[f"{cam_name}_fy"])
                ppx = float(cam_grp.attrs[f"{cam_name}_ppx"])
                ppy = float(cam_grp.attrs[f"{cam_name}_ppy"])
                dist_str = cam_grp.attrs.get(f"{cam_name}_distortion_coeffs", "[]")
                dist_coeffs = np.array(json.loads(dist_str), dtype=np.float32)

                camera_matrix = np.array(
                    [
                        [fx, 0, ppx],
                        [0, fy, ppy],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )

                print(
                    f"Loaded {cam_name} intrinsics from HDF5 camera group: fx={fx:.2f}, fy={fy:.2f}, "
                    f"ppx={ppx:.2f}, ppy={ppy:.2f}"
                )
                return camera_matrix, dist_coeffs

    except Exception as e:
        print(f"Failed to load intrinsics from HDF5: {e}")

    return None


def estimate_chessboard_pose(
    img_rgb: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray, pattern: tuple, objp: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect chessboard and estimate its pose in camera frame.

    Args:
        img_rgb: RGB image from H5 file
        K: Camera intrinsics matrix
        dist_coeffs: Distortion coefficients
        pattern: Chessboard pattern (cols, rows)
        objp: 3D object points for the chessboard

    Returns:
        Tuple of (R_cam_board, t_cam_board) or None if detection failed.
        R_cam_board: 3×3 rotation matrix
        t_cam_board: 3×1 translation vector in meters
    """
    # Convert RGB to grayscale for detection
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(gray, pattern, None)

    if not found:
        print("    FAILED: Chessboard not detected.")
        return None

    # Refine corner positions
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Estimate board pose using solvePnP
    pose_ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coeffs)

    if not pose_ret:
        print("    FAILED: PnP could not solve the pose.")
        return None

    # Calculate RMS reprojection error
    projected_points, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected_points, axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    R_cam_board = cv2.Rodrigues(rvec)[0]
    t_cam_board = tvec.flatten()

    print(f"    SUCCESS: Detected {len(corners)} corners, RMS error: {rms_error:.2f}px")
    return R_cam_board, t_cam_board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera extrinsics from an HDF5 calibration file.",
    )
    parser.add_argument(
        "h5_path",
        type=Path,
        help="Path to the HDF5 calibration file containing images and joint angles.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=CHESSBOARD_ROWS,
        help="Number of chessboard rows (internal corners).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=CHESSBOARD_COLS,
        help="Number of chessboard columns (internal corners).",
    )
    parser.add_argument(
        "--checker-size",
        type=float,
        default=CHECKER_SIZE_M,
        help="Size of each checker square in meters.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="overhead",
        help="Camera name to use from the H5 file (e.g., 'overhead', 'wrist').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    h5_path = args.h5_path

    if not h5_path.exists():
        print(f"ERROR: File not found: {h5_path}")
        return

    print("\n" + "=" * 80)
    print("Hand-Eye Calibration using cv2.calibrateHandEye")
    print("=" * 80)
    print(f"HDF5 file: {h5_path}\n")

    # Load camera intrinsics from H5 file
    intrinsics = get_intrinsics_from_hdf5(h5_path, cam_name=args.camera)
    if intrinsics is None:
        print(f"ERROR: Could not load intrinsics for camera '{args.camera}' from H5 file.")
        print("Make sure the H5 file contains intrinsics attributes.")
        return

    K, dist_coeffs = intrinsics
    print(f"\nCamera intrinsics (K):\n{K}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}\n")

    # Chessboard pattern setup
    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.checker_size

    print(f"Chessboard pattern: {args.cols}x{args.rows} (cols x rows)")
    print(f"Checker size: {args.checker_size} m\n")

    # Initialize forward kinematics
    repo_root = Path(__file__).resolve().parent.parent
    xml_path = repo_root / YAM_XML_PATH
    if not xml_path.exists():
        raise FileNotFoundError(f"YAM XML not found at {xml_path}")

    print(f"Initializing FK with: {xml_path}")
    print(f"End-effector site: {YAM_SITE_NAME}\n")
    kinematics = Kinematics(str(xml_path), YAM_SITE_NAME)

    # Load data from H5 file
    print("=" * 80)
    print("Loading data from HDF5 file...")
    print("=" * 80)

    with h5py.File(h5_path, "r") as f:
        # Check structure
        if "cameras" not in f:
            print("ERROR: No 'cameras' group found in HDF5.")
            return

        if args.camera not in f["cameras"]:
            available = list(f["cameras"].keys())
            print(f"ERROR: Camera '{args.camera}' not found in H5 file.")
            print(f"Available cameras: {available}")
            return

        # Load images
        images = f[f"cameras/{args.camera}/images"][()]
        print(f"Loaded {len(images)} images from camera '{args.camera}'")

        # Load joint angles
        if "joint_angles" not in f:
            print("ERROR: No 'joint_angles' dataset found in HDF5.")
            return

        joint_angles_data = f["joint_angles"][()]
        print(f"Loaded {len(joint_angles_data)} joint angle sets\n")

        if len(images) != len(joint_angles_data):
            print(
                f"WARNING: Number of images ({len(images)}) doesn't match "
                f"joint angles ({len(joint_angles_data)})"
            )

        num_entries = min(len(images), len(joint_angles_data))

        # Collect hand-eye calibration pairs
        R_base_gripper_list = []  # Robot base → gripper transforms (from FK)
        t_base_gripper_list = []
        R_cam_board_list = []  # Camera → board transforms (from detection)
        t_cam_board_list = []

        print("=" * 80)
        print("Processing images and computing hand-eye calibration pairs...")
        print("=" * 80)

        for idx in range(num_entries):
            print(f"\nEntry {idx + 1}/{num_entries}:")

            # Get image
            img_rgb = images[idx]

            # Get joint angles
            joint_positions = joint_angles_data[idx]

            # Compute FK: T_base_gripper
            # YAM MuJoCo model has 8 DOFs: 6 arm joints + 2 gripper fingers
            if len(joint_positions) == 7:
                # Format: [j1, j2, j3, j4, j5, j6, gripper]
                # Convert to MuJoCo format: [j1, j2, j3, j4, j5, j6, left_finger, right_finger]
                arm_joints = joint_positions[:6]
                gripper_pos = joint_positions[6]
                joint_angles = np.concatenate([arm_joints, [gripper_pos, -gripper_pos]])
            else:
                # Assume full 8 DOF format or pad with zeros
                joint_angles = np.zeros(8)
                joint_angles[: len(joint_positions)] = joint_positions

            T_base_gripper = kinematics.fk(joint_angles)
            R_base_gripper = T_base_gripper[:3, :3]
            t_base_gripper = T_base_gripper[:3, 3]

            # Detect chessboard: T_cam_board
            detection_result = estimate_chessboard_pose(img_rgb, K, dist_coeffs, pattern, objp)
            if detection_result is None:
                print(f"  ⚠ Skipping entry {idx + 1}: board detection failed")
                continue

            R_cam_board, t_cam_board = detection_result

            # For eye-to-hand calibration (fixed camera, moving board),
            # we need base→gripper transform (inverse of gripper→base from FK)
            R_gripper_base = R_base_gripper.T  # Inverse of rotation
            t_gripper_base = -R_gripper_base @ t_base_gripper  # Transform translation

            # Add to lists
            R_base_gripper_list.append(R_gripper_base)  # Actually base→gripper
            t_base_gripper_list.append(t_gripper_base.reshape(3, 1))
            R_cam_board_list.append(R_cam_board)
            t_cam_board_list.append(t_cam_board.reshape(3, 1))

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

    # Run hand-eye calibration (eye-to-hand: fixed camera, moving board)
    print("\nRunning cv2.calibrateHandEye (Tsai method) for eye-to-hand setup...\n")

    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_gripper_list,  # Actually base→gripper transforms
        t_gripper2base=t_base_gripper_list,
        R_target2cam=R_cam_board_list,
        t_target2cam=t_cam_board_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # Build full transform
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = t_base_cam.flatten()

    print("=" * 80)
    print("CALIBRATION RESULT: T_base_cam (Camera in Robot Base Frame)")
    print("=" * 80)
    print(f"\n{T_base_cam}\n")
    print(f"Translation (meters): {t_base_cam.flatten()}")
    print(f"Rotation matrix:\n{R_base_cam}\n")

    # Estimate T_gripper_board offset (board frame relative to gripper frame)
    # Since the board is rigidly attached, we can estimate this offset
    print("=" * 80)
    print("ESTIMATING T_gripper_board (Board offset from gripper)")
    print("=" * 80)
    
    # Invert T_base_cam to get T_cam_base
    T_cam_base = np.linalg.inv(T_base_cam)
    
    # For each pose: T_cam_board_detected = T_cam_base @ T_base_gripper @ T_gripper_board
    # Solve for T_gripper_board: T_gripper_board = T_base_gripper^-1 @ T_cam_base^-1 @ T_cam_board_detected
    
    T_gripper_board_list = []
    for R_gripper_base, t_gripper_base, R_cam_brd, t_cam_brd in zip(
        R_base_gripper_list, t_base_gripper_list, R_cam_board_list, t_cam_board_list
    ):
        # T_base_gripper (inverse of the stored gripper→base)
        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = R_gripper_base.T
        T_base_gripper[:3, 3] = -R_gripper_base.T @ t_gripper_base.flatten()
        
        # T_cam_board detected
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cam_brd
        T_cam_board[:3, 3] = t_cam_brd.flatten()
        
        # Solve: T_gripper_board = T_gripper_base @ T_base_cam @ T_cam_board
        T_gripper_base = np.linalg.inv(T_base_gripper)
        T_base_cam_inv = np.linalg.inv(T_cam_base)
        T_gripper_board = T_gripper_base @ T_base_cam_inv @ T_cam_board
        
        T_gripper_board_list.append(T_gripper_board)
    
    # Average the transforms (simple approach: average translation and rotation separately)
    translations = np.array([T[:3, 3] for T in T_gripper_board_list])
    t_gripper_board_avg = np.mean(translations, axis=0)
    
    # For rotation, average using quaternions or matrix averaging (simple: just use median transform)
    median_idx = len(T_gripper_board_list) // 2
    R_gripper_board_avg = T_gripper_board_list[median_idx][:3, :3]
    
    T_gripper_board = np.eye(4)
    T_gripper_board[:3, :3] = R_gripper_board_avg
    T_gripper_board[:3, 3] = t_gripper_board_avg
    
    print(f"\nEstimated T_gripper_board (Board in Gripper Frame):")
    print(f"{T_gripper_board}\n")
    print(f"Translation offset: [{t_gripper_board_avg[0]:.4f}, {t_gripper_board_avg[1]:.4f}, {t_gripper_board_avg[2]:.4f}] m")
    
    # Check consistency of T_gripper_board estimates
    translation_std = np.std(translations, axis=0)
    print(f"Translation std dev: [{translation_std[0]:.4f}, {translation_std[1]:.4f}, {translation_std[2]:.4f}] m")
    print(f"(Low std dev indicates board was rigidly attached)\n")

    # Verify calibration quality by computing board positions in camera frame
    print("=" * 80)
    print("VERIFICATION: Board positions in camera frame (with T_gripper_board)")
    print("=" * 80)
    print("(Should match detected positions if calibration is accurate)\n")

    board_positions_cam = []
    board_positions_detected = []
    for i, (R_gripper_base, t_gripper_base, R_cam_brd, t_cam_brd) in enumerate(
        zip(
            R_base_gripper_list, t_base_gripper_list, R_cam_board_list, t_cam_board_list
        ),
        1,
    ):
        # Get T_base_gripper (inverse of the stored gripper→base)
        T_base_gripper = np.eye(4)
        T_base_gripper[:3, :3] = R_gripper_base.T
        T_base_gripper[:3, 3] = -R_gripper_base.T @ t_gripper_base.flatten()

        # T_cam_board_computed = T_cam_base @ T_base_gripper @ T_gripper_board
        T_cam_board_computed = T_cam_base @ T_base_gripper @ T_gripper_board

        board_pos_computed = T_cam_board_computed[:3, 3]
        board_pos_detected = t_cam_brd.flatten()
        
        board_positions_cam.append(board_pos_computed)
        board_positions_detected.append(board_pos_detected)
        
        error = np.linalg.norm(board_pos_computed - board_pos_detected)
        print(
            f"  Entry #{i:2d}: Computed [{board_pos_computed[0]:7.4f}, {board_pos_computed[1]:7.4f}, {board_pos_computed[2]:7.4f}] m "
            f"| Detected [{board_pos_detected[0]:7.4f}, {board_pos_detected[1]:7.4f}, {board_pos_detected[2]:7.4f}] m "
            f"| Error: {error:.4f} m"
        )

    # Compute statistics
    board_positions_cam = np.array(board_positions_cam)
    board_positions_detected = np.array(board_positions_detected)
    errors = np.linalg.norm(board_positions_cam - board_positions_detected, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rms_error = np.sqrt(np.mean(errors**2))

    print(f"\n  Mean reprojection error: {mean_error:.4f} m ({mean_error*1000:.2f} mm)")
    print(f"  RMS reprojection error: {rms_error:.4f} m ({rms_error*1000:.2f} mm)")
    print(f"  Max reprojection error: {max_error:.4f} m ({max_error*1000:.2f} mm)")
    print("\n  (Lower values indicate better calibration quality)")
    print("=" * 80 + "\n")

    # Save result
    output_path = h5_path.parent / f"{h5_path.stem}_hand_eye_calibration.json"
    result = {
        "T_base_cam": T_base_cam.tolist(),
        "R_base_cam": R_base_cam.tolist(),
        "t_base_cam": t_base_cam.flatten().tolist(),
        "T_gripper_board": T_gripper_board.tolist(),
        "t_gripper_board": t_gripper_board_avg.tolist(),
        "num_pairs": len(R_base_gripper_list),
        "method": "cv2.CALIB_HAND_EYE_TSAI",
        "calibration_type": "eye-to-hand",
        "chessboard_rows": args.rows,
        "chessboard_cols": args.cols,
        "checker_size_m": args.checker_size,
        "camera_name": args.camera,
        "source_h5": str(h5_path),
        "mean_reprojection_error_m": float(mean_error),
        "rms_reprojection_error_m": float(rms_error),
        "max_reprojection_error_m": float(max_error),
        "gripper_board_translation_std_m": translation_std.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved calibration to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
