"""
CSC 8830 - Assignment 6
Part 2 - Structure from Motion / Planar Reconstruction

This script uses four images of a printed checkerboard taken
from different viewpoints.

The checkerboard has:
- 10 x 7 squares
- 9 x 6 inner corners
- square size = 19.05 mm

The program:
1. Detects checkerboard corners in all four images.
2. Uses the known square size to define real-world coordinates.
3. Estimates camera calibration parameters.
4. Estimates the camera pose for each viewpoint.
5. Projects the known checkerboard boundary back into each image.
6. Saves the detected corners and reconstructed boundary.
"""

import cv2
import numpy as np
import os


# ----------------------------------------------------
# Checkerboard information
# ----------------------------------------------------

CHECKERBOARD = (9, 6)

SQUARE_SIZE = 19.05  # millimeters

image_paths = [
    "sfm/view1.jpg",
    "sfm/view2.jpg",
    "sfm/view3.jpg",
    "sfm/view4.jpg"
]

os.makedirs("outputs/sfm", exist_ok=True)


# ----------------------------------------------------
# Real-world coordinates of the 9 x 6 inner corners
# ----------------------------------------------------

object_points_single = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_single[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
)

object_points_single *= SQUARE_SIZE


object_points = []
image_points = []

valid_images = []
image_size = None


# ----------------------------------------------------
# Detect checkerboard corners
# ----------------------------------------------------

for i, image_path in enumerate(image_paths):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not open {image_path}")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if not found:
        print(
            f"Checkerboard not detected in {image_path}"
        )
        continue

    # Improve the corner positions to subpixel accuracy
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )

    object_points.append(
        object_points_single.copy()
    )

    image_points.append(corners)

    valid_images.append(image_path)

    # Draw detected corners
    detected = image.copy()

    cv2.drawChessboardCorners(
        detected,
        CHECKERBOARD,
        corners,
        found
    )

    cv2.imwrite(
        f"outputs/sfm/view{i + 1}_corners.jpg",
        detected
    )

    print(
        f"View {i + 1}: detected "
        f"{len(corners)} corners"
    )


# ----------------------------------------------------
# Camera calibration
# ----------------------------------------------------

if len(object_points) < 2:
    print("Not enough valid images.")
    exit()


rms, camera_matrix, distortion, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)


print("\n----------------------------------------")
print("CAMERA CALIBRATION")
print("----------------------------------------")

print(f"\nRMS reprojection error: {rms:.4f}")

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion)


# ----------------------------------------------------
# Full checkerboard boundary
#
# Inner corners begin one square inside the outer edge.
# Therefore the full board extends:
#
# x: -19.05 mm to 171.45 mm
# y: -19.05 mm to 114.30 mm
#
# Width = 190.50 mm
# Height = 133.35 mm
# ----------------------------------------------------

board_boundary = np.array(
    [
        [-SQUARE_SIZE, -SQUARE_SIZE, 0],
        [9 * SQUARE_SIZE, -SQUARE_SIZE, 0],
        [9 * SQUARE_SIZE, 6 * SQUARE_SIZE, 0],
        [-SQUARE_SIZE, 6 * SQUARE_SIZE, 0]
    ],
    dtype=np.float32
)


# ----------------------------------------------------
# Estimate and display camera pose for each view
# ----------------------------------------------------

for i in range(len(valid_images)):

    image = cv2.imread(valid_images[i])

    rvec = rvecs[i]
    tvec = tvecs[i]

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Camera center in world coordinates
    camera_position = (
        -rotation_matrix.T @ tvec
    )

    print("\n----------------------------------------")
    print(f"VIEW {i + 1}")
    print("----------------------------------------")

    print("\nRotation vector:")
    print(rvec.flatten())

    print("\nTranslation vector (mm):")
    print(tvec.flatten())

    print("\nEstimated camera position (mm):")
    print(camera_position.flatten())


    # ------------------------------------------------
    # Project board boundary into the image
    # ------------------------------------------------

    projected_boundary, _ = cv2.projectPoints(
        board_boundary,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    projected_boundary = (
        projected_boundary
        .reshape(-1, 2)
        .astype(int)
    )

    boundary_image = image.copy()

    cv2.polylines(
        boundary_image,
        [projected_boundary],
        True,
        (0, 0, 255),
        5
    )

    # Label corners
    labels = [
        "P1",
        "P2",
        "P3",
        "P4"
    ]

    for point, label in zip(
        projected_boundary,
        labels
    ):

        x, y = point

        cv2.circle(
            boundary_image,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            boundary_image,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imwrite(
        f"outputs/sfm/"
        f"view{i + 1}_boundary.jpg",
        boundary_image
    )


# ----------------------------------------------------
# Save calibration values to text file
# ----------------------------------------------------

with open(
    "outputs/sfm/calibration_results.txt",
    "w"
) as file:

    file.write(
        f"RMS reprojection error:\n"
        f"{rms}\n\n"
    )

    file.write(
        "Camera Matrix:\n"
    )

    file.write(
        str(camera_matrix)
    )

    file.write(
        "\n\nDistortion Coefficients:\n"
    )

    file.write(
        str(distortion)
    )

    file.write("\n\n")

    for i in range(len(rvecs)):

        rotation_matrix, _ = (
            cv2.Rodrigues(rvecs[i])
        )

        camera_position = (
            -rotation_matrix.T
            @ tvecs[i]
        )

        file.write(
            f"View {i + 1}\n"
        )

        file.write(
            f"Rotation Vector:\n"
            f"{rvecs[i].flatten()}\n"
        )

        file.write(
            f"Translation Vector (mm):\n"
            f"{tvecs[i].flatten()}\n"
        )

        file.write(
            f"Camera Position (mm):\n"
            f"{camera_position.flatten()}\n\n"
        )


print("\nFinished Part 2 processing.")

print(
    "\nResults saved in: outputs/sfm/"
)