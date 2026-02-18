import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def check_orthonormal_rotation(rvec: np.ndarray, tolerance: float = 1e-3) -> tuple:
    """Check if rotation vector produces an orthonormal rotation matrix.

    Args:
        rvec: Rotation vector (3,)
        tolerance: Tolerance for deviation from orthonormality

    Returns:
        Tuple of (is_orthonormal, frobenius_error, determinant)
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Check if R @ R.T = I (orthogonality)
    should_be_I = R @ R.T
    frobenius_error = np.linalg.norm(should_be_I - np.eye(3), "fro")

    # Check if det(R) = 1 (proper rotation, not reflection)
    det_R = np.linalg.det(R)

    # Is it orthonormal?
    is_orthonormal = frobenius_error < tolerance and abs(det_R - 1.0) < tolerance

    return is_orthonormal, frobenius_error, det_R


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


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup HDF5 calibration data by reviewing images and detecting chessboard."
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

    # Load file
    with h5py.File(args.h5_path, "r") as f:
        # Check structure
        if "cameras" not in f:
            print("No 'cameras' group found in HDF5.")
            return

        cam_names = sorted(list(f["cameras"].keys()))
        if not cam_names:
            print("No camera datasets found under /cameras.")
            return

        # Use pose_index to determine number of entries
        if "pose_index" not in f:
            print("No 'pose_index' dataset found.")
            return

        num_entries = f["pose_index"].shape[0]
        to_keep = np.ones(num_entries, dtype=bool)

        print(f"Reviewing {num_entries} entries from {args.h5_path}")
        print("-" * 40)
        print("Controls:")
        print("  [n] or [space] : Next entry")
        print("  [b] or [backspace] : Previous entry")
        print("  [d] or [x] : Toggle Delete (mark/unmark current entry)")
        print("  [q] : Finish review and prompt to save changes")
        print("  [Esc] : Exit without saving any changes")
        print("-" * 40)

        # Chessboard pattern
        pattern = (args.cols, args.rows)

        # Generate 3D object points for the chessboard
        objp = np.zeros((args.rows * args.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
        objp *= args.checker_size

        # Load camera intrinsics from HDF5 file
        realsense_intrinsics = get_intrinsics_from_hdf5(
            args.h5_path, cam_name="overhead"
        )

        if realsense_intrinsics is None:
            print("Intrinsics not found in HDF5. Using default camera matrix fallback.")

        idx = 0
        while idx < num_entries:
            # Gather images for current pose
            display_imgs = []

            for cam_name in cam_names:
                # Read image
                img_rgb = f["cameras"][cam_name]["images"][idx]

                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # Find chessboard corners
                found, corners = cv2.findChessboardCorners(gray, pattern, None)

                pose_ret = False
                rvec = tvec = None

                if found:
                    # Refine corner positions
                    criteria = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    )
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )

                    print(
                        f"  {cam_name}: Chessboard detected with {len(corners)} corners"
                    )

                    # Estimate pose
                    # Use RealSense intrinsics if available, otherwise use default
                    if realsense_intrinsics is not None:
                        camera_matrix, dist_coeffs = realsense_intrinsics
                    else:
                        # Create default camera matrix based on image size
                        h, w = img_bgr.shape[:2]
                        focal_length = w
                        camera_matrix = np.array(
                            [
                                [focal_length, 0, w / 2],
                                [0, focal_length, h / 2],
                                [0, 0, 1],
                            ],
                            dtype=np.float32,
                        )
                        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

                    # Estimate board pose using solvePnP
                    pose_ret, rvec, tvec = cv2.solvePnP(
                        objp, corners, camera_matrix, dist_coeffs
                    )

                    # Calculate RMS reprojection error if pose estimation succeeded
                    rms_error = None
                    ortho_status = None
                    if pose_ret and rvec is not None and tvec is not None:
                        # Project 3D points to image plane
                        projected_points, _ = cv2.projectPoints(
                            objp, rvec, tvec, camera_matrix, dist_coeffs
                        )
                        projected_points = projected_points.reshape(-1, 2)

                        # Calculate errors
                        errors = np.linalg.norm(
                            corners.reshape(-1, 2) - projected_points, axis=1
                        )
                        rms_error = np.sqrt(np.mean(errors**2))

                        # Check rotation matrix orthonormality
                        is_ortho, frob_err, det_r = check_orthonormal_rotation(rvec)
                        ortho_status = "OK" if is_ortho else "BAD"
                        print(
                            f"    Rotation: {ortho_status} (frob_err={frob_err:.4f}, det={det_r:.4f})"
                        )
                else:
                    corners = None
                    rms_error = None
                    ortho_status = None
                    print(f"  {cam_name}: Chessboard NOT detected")

                # Visualization
                vis_img = img_bgr.copy()

                if found and corners is not None:
                    # Draw chessboard corners
                    cv2.drawChessboardCorners(vis_img, pattern, corners, found)

                    # Draw pose axes if available
                    if pose_ret and rvec is not None and tvec is not None:
                        # Use RealSense intrinsics if available, otherwise use default
                        if realsense_intrinsics is not None:
                            camera_matrix, dist_coeffs = realsense_intrinsics
                        else:
                            raise ValueError("Intrinsics should be available here.")
                        cv2.drawFrameAxes(
                            vis_img,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            length=0.05,
                            thickness=3,
                        )

                    color = (0, 255, 0)
                    status = "DETECTED"
                else:
                    color = (0, 0, 255)
                    status = "NOT DETECTED"

                # Overlay status
                status_text = f"{cam_name}: {status}"
                if rms_error is not None:
                    status_text += f" | RMS: {rms_error:.2f}px"
                if ortho_status is not None:
                    status_text += f" | R: {ortho_status}"
                cv2.putText(
                    vis_img,
                    status_text,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    3,
                )

                display_imgs.append(vis_img)

            # Combine images horizontally
            combined = np.hstack(display_imgs)

            # Scale down if too large for screen
            screen_max_w, screen_max_h = 1600, 900
            h, w = combined.shape[:2]
            scale = min(screen_max_w / w, (screen_max_h - 100) / h, 1.0)
            if scale < 1.0:
                combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)

            # Draw UI footer
            footer_h = 80
            footer = np.zeros((footer_h, combined.shape[1], 3), dtype=np.uint8)

            status_text = f"Entry {idx + 1}/{num_entries} | Action: {'[DELETE]' if not to_keep[idx] else '[KEEP]'}"
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

            final_display = np.vstack([combined, footer])

            cv2.imshow("Calibration Data Review", final_display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                print("Review cancelled. No changes made.")
                return
            elif key == ord("d") or key == ord("x"):
                to_keep[idx] = not to_keep[idx]
                print(
                    f"Entry {idx + 1}: Set to {'DELETE' if not to_keep[idx] else 'KEEP'}"
                )
            elif key in [ord("n"), ord(" ")]:
                idx += 1
            elif key in [ord("b"), 8]:  # 8 is backspace
                idx = max(0, idx - 1)

        cv2.destroyAllWindows()

        num_to_delete = np.sum(~to_keep)
        if num_to_delete == 0:
            print("No entries marked for deletion. Exiting.")
            return

        print(f"\nReview complete. {num_to_delete} entries marked for deletion.")
        confirm = input(f"Confirm PERMANENT deletion from {args.h5_path.name}? [y/N]: ")
        if confirm.lower() != "y":
            print("Changes discarded.")
            return

        # Create a new file without the deleted entries
        temp_path = args.h5_path.with_suffix(".h5.tmp")
        try:
            with h5py.File(temp_path, "w") as f_new:
                with h5py.File(args.h5_path, "r") as f_old:
                    # Copy root attributes
                    for k, v in f_old.attrs.items():
                        f_new.attrs[k] = v

                    # Identify valid indices
                    indices = np.where(to_keep)[0]
                    new_total = len(indices)

                    # Update total_poses if it exists
                    if "total_poses" in f_new.attrs:
                        f_new.attrs["total_poses"] = new_total

                    # Copy root datasets
                    for ds_name in ["timestamps", "joint_angles", "pose_index"]:
                        if ds_name in f_old:
                            f_new.create_dataset(
                                ds_name, data=f_old[ds_name][()][indices]
                            )

                    # Copy camera group and datasets
                    cam_grp_new = f_new.create_group("cameras")
                    cam_grp_old = f_old["cameras"]
                    for cam_name in cam_grp_old:
                        c_old = cam_grp_old[cam_name]
                        cam_grp_new.create_group(cam_name)

                        # Copy images with same properties
                        if "images" in c_old:
                            imgs_ds = c_old["images"]
                            # Copy image data
                            f_new.create_dataset(
                                f"cameras/{cam_name}/images",
                                data=imgs_ds[()][indices],
                                compression=imgs_ds.compression,
                                compression_opts=imgs_ds.compression_opts,
                            )

            # Replace original
            temp_path.replace(args.h5_path)
            print(f"Successfully cleaned file. {new_total} poses remaining.")

        except Exception as e:
            print(f"ERROR during save: {e}")
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()
