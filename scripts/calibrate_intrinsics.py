import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def get_intrinsics_from_hdf5(h5_path: Path, cam_name: str = "overhead"):
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


def detect_chessboard_and_corners(
    img_rgb: np.ndarray, pattern: tuple, checker_size: float
):
    """Detect chessboard corners in image.

    Args:
        img_rgb: RGB image (H, W, 3)
        pattern: Tuple of (cols, rows)
        checker_size: Size of each checker square in meters

    Returns:
        Tuple of (found, corners, objp) where objp is 3D object points
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(gray, pattern, None)

    if found:
        # Refine corner positions
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Generate 3D object points for the chessboard
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= checker_size

    return found, corners, objp


def compute_rms_error(
    objp: np.ndarray,
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """Compute RMS reprojection error.

    Args:
        objp: 3D object points (N, 3)
        corners: 2D corner points (N, 1, 2)
        camera_matrix: Camera intrinsic matrix (3, 3)
        dist_coeffs: Distortion coefficients

    Returns:
        RMS reprojection error in pixels
    """
    # Estimate board pose
    pose_ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if not pose_ret or rvec is None or tvec is None:
        return float("nan")

    # Project 3D points to image plane
    projected_points, _ = cv2.projectPoints(
        objp, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)

    # Calculate RMS error
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected_points, axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    return rms_error


def visualize_detection(
    img_rgb: np.ndarray,
    pattern: tuple,
    found: bool,
    corners: np.ndarray,
    rms_error: float,
) -> np.ndarray:
    """Create visualization of chessboard detection.

    Args:
        img_rgb: RGB image (H, W, 3)
        pattern: Tuple of (cols, rows)
        found: Whether chessboard was detected
        corners: Detected corners
        rms_error: RMS reprojection error

    Returns:
        BGR image with overlay
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    vis_img = img_bgr.copy()

    if found and corners is not None:
        # Draw chessboard corners
        cv2.drawChessboardCorners(vis_img, pattern, corners, found)
        color = (0, 255, 0)
        status = "DETECTED"
    else:
        color = (0, 0, 255)
        status = "NOT DETECTED"

    # Overlay status
    status_text = f"Overhead: {status}"
    if not np.isnan(rms_error):
        status_text += f" | RMS: {rms_error:.2f}px"
    cv2.putText(
        vis_img,
        status_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        color,
        3,
    )

    return vis_img


def main():
    parser = argparse.ArgumentParser(
        description="Calculate camera intrinsics from calibration images in HDF5 file."
    )
    parser.add_argument("h5_path", type=Path, help="Path to the HDF5 file.")
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of chessboard rows.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of chessboard columns.",
    )
    parser.add_argument(
        "--checker-size",
        type=float,
        default=0.03,
        help="Size of each checker square in meters.",
    )
    args = parser.parse_args()

    if not args.h5_path.exists():
        print(f"File not found: {args.h5_path}")
        return

    # Load file and check structure
    with h5py.File(args.h5_path, "r") as f:
        if "cameras" not in f:
            print("No 'cameras' group found in HDF5.")
            return

        if "overhead" not in f["cameras"]:
            print("No 'overhead' camera found in HDF5.")
            return

        if "images" not in f["cameras/overhead"]:
            print("No 'images' dataset found for overhead camera.")
            return

        num_entries = f["cameras/overhead/images"].shape[0]

    print(f"Found {num_entries} images from overhead camera in {args.h5_path}")
    print("-" * 60)
    print("Controls:")
    print("  [k] : Keep current image")
    print("  [d] : Discard current image")
    print("  [n] or [space] : Next image")
    print("  [b] or [backspace] : Previous image")
    print("  [q] : Finish review")
    print("  [Esc] : Exit without calibration")
    print("-" * 60)

    # Load current intrinsics
    print("\nLoading current hardware intrinsics...")
    old_intrinsics = get_intrinsics_from_hdf5(args.h5_path, cam_name="overhead")
    if old_intrinsics is None:
        print("WARNING: No existing intrinsics found in HDF5.")
        old_camera_matrix = None
        old_dist_coeffs = None
    else:
        old_camera_matrix, old_dist_coeffs = old_intrinsics

    pattern = (args.cols, args.rows)

    # Interactive review and selection
    to_keep = np.ones(num_entries, dtype=bool)
    idx = 0

    with h5py.File(args.h5_path, "r") as f:
        while idx < num_entries:
            # Read image
            img_rgb = f["cameras/overhead/images"][idx]

            # Detect chessboard
            found, corners, objp = detect_chessboard_and_corners(
                img_rgb, pattern, args.checker_size
            )

            # Compute RMS error if we have old intrinsics and detection succeeded
            rms_error = float("nan")
            if found and old_camera_matrix is not None:
                rms_error = compute_rms_error(
                    objp, corners, old_camera_matrix, old_dist_coeffs
                )

            # Visualize
            vis_img = visualize_detection(img_rgb, pattern, found, corners, rms_error)

            # Scale down if too large
            screen_max_w, screen_max_h = 1600, 900
            h, w = vis_img.shape[:2]
            scale = min(screen_max_w / w, (screen_max_h - 100) / h, 1.0)
            if scale < 1.0:
                vis_img = cv2.resize(vis_img, (0, 0), fx=scale, fy=scale)

            # Draw UI footer
            footer_h = 80
            footer = np.zeros((footer_h, vis_img.shape[1], 3), dtype=np.uint8)

            status_text = f"Image {idx + 1}/{num_entries} | Action: {'[DELETE]' if not to_keep[idx] else '[KEEP]'}"
            footer_color = (0, 0, 255) if not to_keep[idx] else (0, 255, 0)
            cv2.putText(
                footer,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                footer_color,
                2,
            )

            final_display = np.vstack([vis_img, footer])

            cv2.imshow("Calibration Image Review", final_display)

            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # Escape
                cv2.destroyAllWindows()
                print("Review cancelled. No calibration performed.")
                return
            elif key == ord("q"):
                break
            elif key == ord("k"):
                to_keep[idx] = True
                print(f"Image {idx + 1}: Set to KEEP")
                idx += 1
            elif key == ord("d"):
                to_keep[idx] = False
                print(f"Image {idx + 1}: Set to DISCARD")
                idx += 1
            elif key in [ord("n"), ord(" ")]:
                idx += 1
            elif key in [ord("b"), 8]:  # 8 is backspace
                idx = max(0, idx - 1)

    cv2.destroyAllWindows()

    num_to_use = np.sum(to_keep)
    if num_to_use == 0:
        print("No images marked to keep. Cannot calibrate.")
        return

    print(f"\nSelected {num_to_use}/{num_entries} images for calibration.")

    # Collect calibration data from approved images
    print("Collecting chessboard corners and computing intrinsics...")
    all_objps = []
    all_corners_list = []

    with h5py.File(args.h5_path, "r") as f:
        indices = np.where(to_keep)[0]
        for idx in indices:
            img_rgb = f["cameras/overhead/images"][idx]
            found, corners, objp = detect_chessboard_and_corners(
                img_rgb, pattern, args.checker_size
            )

            if found:
                all_objps.append(objp)
                all_corners_list.append(corners)

    if len(all_objps) == 0:
        print("No images with detected chessboards. Cannot calibrate.")
        return

    print(f"Using {len(all_objps)} images with detected chessboards.")

    # Get image size for calibration
    with h5py.File(args.h5_path, "r") as f:
        first_img = f["cameras/overhead/images"][int(np.where(to_keep)[0][0])]
        image_size = (first_img.shape[1], first_img.shape[0])  # (width, height)

    # Run OpenCV camera calibration
    print(f"Running cv2.calibrateCamera with {len(all_objps)} poses...")
    ret, new_camera_matrix, new_dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_objps,
        all_corners_list,
        image_size,
        None,
        None,
    )

    if not ret:
        print("ERROR: Camera calibration failed.")
        return

    # Extract parameters
    new_fx = new_camera_matrix[0, 0]
    new_fy = new_camera_matrix[1, 1]
    new_ppx = new_camera_matrix[0, 2]
    new_ppy = new_camera_matrix[1, 2]

    # Display comparison
    print("\n" + "=" * 60)
    print("INTRINSICS COMPARISON:")
    print("=" * 60)
    print(
        f"{'Parameter':<15} {'Old (Hardware)':<20} {'New (Computed)':<20} {'Difference':<15}"
    )
    print("-" * 60)

    if old_camera_matrix is not None:
        old_fx = old_camera_matrix[0, 0]
        old_fy = old_camera_matrix[1, 1]
        old_ppx = old_camera_matrix[0, 2]
        old_ppy = old_camera_matrix[1, 2]

        print(f"{'fx':<15} {old_fx:<20.4f} {new_fx:<20.4f} {new_fx - old_fx:>+14.4f}")
        print(f"{'fy':<15} {old_fy:<20.4f} {new_fy:<20.4f} {new_fy - old_fy:>+14.4f}")
        print(
            f"{'ppx':<15} {old_ppx:<20.4f} {new_ppx:<20.4f} {new_ppx - old_ppx:>+14.4f}"
        )
        print(
            f"{'ppy':<15} {old_ppy:<20.4f} {new_ppy:<20.4f} {new_ppy - old_ppy:>+14.4f}"
        )
        print("-" * 60)
        print(f"{'Distortion (old):':<15} {json.dumps(old_dist_coeffs.tolist())}")
        print(
            f"{'Distortion (new):':<15} {json.dumps(new_dist_coeffs.reshape(-1).tolist())}"
        )
    else:
        print(f"{'fx':<15} {'N/A':<20} {new_fx:<20.4f}")
        print(f"{'fy':<15} {'N/A':<20} {new_fy:<20.4f}")
        print(f"{'ppx':<15} {'N/A':<20} {new_ppx:<20.4f}")
        print(f"{'ppy':<15} {'N/A':<20} {new_ppy:<20.4f}")
        print("-" * 60)
        print(
            f"{'Distortion (new):':<15} {json.dumps(new_dist_coeffs.reshape(-1).tolist())}"
        )

    print(
        f"\nCalibration used {len(all_objps)} poses with {len(all_corners_list[0])} corners per image."
    )
    print("=" * 60)

    # Ask for confirmation
    confirm = input(
        f"\nConfirm PERMANENT overwrite of overhead intrinsics in {args.h5_path.name}? [y/N]: "
    )
    if confirm.lower() != "y":
        print("Changes discarded.")
        return

    # Create temporary file and save new intrinsics
    temp_path = args.h5_path.with_suffix(".h5.tmp")
    try:
        with h5py.File(temp_path, "w") as f_new:
            with h5py.File(args.h5_path, "r") as f_old:
                # Copy all root attributes
                for k, v in f_old.attrs.items():
                    f_new.attrs[k] = v

                # Update intrinsics in root attributes
                f_new.attrs["overhead_intrinsics_fx"] = float(new_fx)
                f_new.attrs["overhead_intrinsics_fy"] = float(new_fy)
                f_new.attrs["overhead_intrinsics_ppx"] = float(new_ppx)
                f_new.attrs["overhead_intrinsics_ppy"] = float(new_ppy)
                f_new.attrs["overhead_distortion_coeffs"] = json.dumps(
                    new_dist_coeffs.reshape(-1).tolist()
                )
                f_new.attrs["overhead_camera_matrix"] = json.dumps(
                    new_camera_matrix.tolist()
                )

                # Copy all root datasets
                for ds_name in f_old.keys():
                    if ds_name not in ["cameras"]:
                        f_new.create_dataset(ds_name, data=f_old[ds_name][()])

                # Copy camera group
                cam_grp_new = f_new.create_group("cameras")
                cam_grp_old = f_old["cameras"]

                for cam_name in cam_grp_old:
                    cam_grp_new.create_group(cam_name)
                    c_old = cam_grp_old[cam_name]

                    # Copy all datasets in camera group
                    for ds_name in c_old.keys():
                        f_new.create_dataset(
                            f"cameras/{cam_name}/{ds_name}",
                            data=c_old[ds_name][()],
                            compression=c_old[ds_name].compression,
                            compression_opts=c_old[ds_name].compression_opts,
                        )

                    # Update intrinsics in camera group attributes (if overhead camera)
                    if cam_name == "overhead":
                        for k, v in c_old.attrs.items():
                            cam_grp_new[cam_name].attrs[k] = v

                        cam_grp_new[cam_name].attrs["overhead_intrinsics_fx"] = float(
                            new_fx
                        )
                        cam_grp_new[cam_name].attrs["overhead_intrinsics_fy"] = float(
                            new_fy
                        )
                        cam_grp_new[cam_name].attrs["overhead_intrinsics_ppx"] = float(
                            new_ppx
                        )
                        cam_grp_new[cam_name].attrs["overhead_intrinsics_ppy"] = float(
                            new_ppy
                        )
                        cam_grp_new[cam_name].attrs["overhead_distortion_coeffs"] = (
                            json.dumps(new_dist_coeffs.reshape(-1).tolist())
                        )
                        cam_grp_new[cam_name].attrs["overhead_camera_matrix"] = (
                            json.dumps(new_camera_matrix.tolist())
                        )
                    else:
                        for k, v in c_old.attrs.items():
                            cam_grp_new[cam_name].attrs[k] = v

        # Replace original with temp file
        temp_path.replace(args.h5_path)
        print(f"\nSuccessfully saved new intrinsics to {args.h5_path}!")
        print(
            f"New intrinsics: fx={new_fx:.2f}, fy={new_fy:.2f}, ppx={new_ppx:.2f}, ppy={new_ppy:.2f}"
        )

    except Exception as e:
        print(f"ERROR during save: {e}")
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
