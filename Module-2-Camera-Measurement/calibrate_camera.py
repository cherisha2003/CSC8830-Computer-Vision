"""
CSC 8830 - Module 2: Smartphone Camera Calibration

This script calibrates an iPhone camera from checkerboard photographs.

Before running:
    1. Put the checkerboard photographs in calibration_images/.
    2. Keep every image at the same resolution and camera setting.

Run:
    python3 calibrate_camera.py

Outputs are saved in calibration_results/.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# The printed board has 10 x 7 squares and therefore 9 x 6 inner corners.
CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0

IMAGE_FOLDER = Path("calibration_images")
RESULTS_FOLDER = Path("calibration_results")
CORNER_FOLDER = RESULTS_FOLDER / "detected_corners"


def make_checkerboard_points():
    """Return the known 3D coordinates of the flat checkerboard corners."""
    points = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
        dtype=np.float32,
    )
    points[:, :2] = (
        np.mgrid[
            0 : CHECKERBOARD_SIZE[0],
            0 : CHECKERBOARD_SIZE[1],
        ]
        .T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )
    return points


def find_image_files():
    """Find supported calibration photographs."""
    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError("The calibration_images folder was not found.")

    files = sorted(
        path
        for path in IMAGE_FOLDER.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not files:
        raise FileNotFoundError(
            "No JPG, JPEG, or PNG files were found in calibration_images/."
        )
    return files


def detect_checkerboards(image_files):
    """Detect checkerboard corners and save corner-preview images."""
    board_points = make_checkerboard_points()
    object_points = []
    image_points = []
    accepted = []
    rejected = []
    image_size = None

    detection_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            rejected.append(image_path.name)
            print(f"REJECTED: {image_path.name} could not be opened")
            continue

        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected.append(image_path.name)
            print(f"REJECTED: {image_path.name} has a different resolution")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            detection_flags,
        )
        used_sb_detector = False

        if not found:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                CHECKERBOARD_SIZE,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            used_sb_detector = found

        if not found:
            rejected.append(image_path.name)
            print(f"REJECTED: {image_path.name} - corners not detected")
            continue

        if not used_sb_detector:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )

        object_points.append(board_points.copy())
        image_points.append(corners)
        accepted.append(image_path.name)

        preview = image.copy()
        cv2.drawChessboardCorners(
            preview,
            CHECKERBOARD_SIZE,
            corners,
            True,
        )
        preview_width = 900
        preview_scale = preview_width / preview.shape[1]
        preview = cv2.resize(
            preview,
            (preview_width, int(preview.shape[0] * preview_scale)),
        )
        output_path = CORNER_FOLDER / f"{image_path.stem}_corners.jpg"
        cv2.imwrite(str(output_path), preview)
        print(f"ACCEPTED: {image_path.name}")

    return object_points, image_points, accepted, rejected, image_size


def calculate_reprojection_errors(
    object_points,
    image_points,
    rotation_vectors,
    translation_vectors,
    camera_matrix,
    distortion_coefficients,
):
    """Calculate one reprojection error value for every accepted image."""
    errors = []
    for index in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[index],
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion_coefficients,
        )
        detected = image_points[index].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum((detected - projected) ** 2, axis=1)))
        errors.append(float(error))
    return errors


def main():
    RESULTS_FOLDER.mkdir(exist_ok=True)
    CORNER_FOLDER.mkdir(exist_ok=True)

    image_files = find_image_files()
    print(f"Found {len(image_files)} calibration images.\n")

    (
        object_points,
        image_points,
        accepted,
        rejected,
        image_size,
    ) = detect_checkerboards(image_files)

    if len(accepted) < 10:
        raise RuntimeError(
            f"Only {len(accepted)} images were accepted. At least 10 are required."
        )

    # Estimate k1 and k2. Tangential distortion and k3 are fixed to zero.
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    (
        rms_error,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    per_image_errors = calculate_reprojection_errors(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        camera_matrix,
        distortion_coefficients,
    )
    mean_error = float(np.mean(per_image_errors))
    maximum_error = float(np.max(per_image_errors))

    np.savez(
        RESULTS_FOLDER / "calibration_data.npz",
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        image_width=image_size[0],
        image_height=image_size[1],
        square_size_mm=SQUARE_SIZE_MM,
    )

    report = {
        "checkerboard_internal_corners": list(CHECKERBOARD_SIZE),
        "square_size_mm": SQUARE_SIZE_MM,
        "image_resolution": list(image_size),
        "total_images": len(image_files),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_images": accepted,
        "rejected_images": rejected,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion_coefficients.tolist(),
        "opencv_rms_reprojection_error_pixels": float(rms_error),
        "mean_per_image_reprojection_error_pixels": mean_error,
        "maximum_per_image_reprojection_error_pixels": maximum_error,
        "per_image_errors_pixels": dict(zip(accepted, per_image_errors)),
    }
    with open(
        RESULTS_FOLDER / "calibration_report.json",
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(report, report_file, indent=4)

    sample_image = cv2.imread(str(IMAGE_FOLDER / accepted[0]))
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        image_size,
        0,
        image_size,
    )
    undistorted = cv2.undistort(
        sample_image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )
    cv2.imwrite(
        str(RESULTS_FOLDER / "undistorted_example.jpg"),
        undistorted,
    )

    print("\nCAMERA CALIBRATION COMPLETED")
    print("--------------------------------------------")
    print(f"Image resolution: {image_size[0]} x {image_size[1]}")
    print(f"Total images: {len(image_files)}")
    print(f"Accepted images: {len(accepted)}")
    print(f"Rejected images: {len(rejected)}")
    print(f"Rejected files: {rejected}")
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion_coefficients)
    print(f"\nOpenCV RMS error: {rms_error:.4f} pixels")
    print(f"Mean per-image reprojection error: {mean_error:.4f} pixels")
    print(f"Maximum per-image reprojection error: {maximum_error:.4f} pixels")
    print("\nResults saved in calibration_results/")


if __name__ == "__main__":
    main()
