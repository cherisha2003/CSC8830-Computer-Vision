"""
CSC 8830 - Assignment 6
Tracking Validation

This script selects two consecutive frames and tracks feature
points between them using Lucas-Kanade optical flow.

For the cycle video, the search is limited to the cyclist area
so the tracked points represent the moving object instead of
mostly the background.
"""

import cv2
import numpy as np
import os


def validate_tracking(
    video_path,
    output_prefix,
    start_frame,
    use_roi=False
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if not ret1 or not ret2:
        print("Could not read frames.")
        cap.release()
        return

    width = 960

    original_width = frame1.shape[1]
    original_height = frame1.shape[0]

    scale = width / original_width
    height = int(original_height * scale)

    frame1 = cv2.resize(
        frame1,
        (width, height)
    )

    frame2 = cv2.resize(
        frame2,
        (width, height)
    )

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    mask = None

    # Limit feature detection to cyclist area
    if use_roi:

        mask = np.zeros_like(gray1)

        # Cyclist is roughly in the center
        mask[
            170:500,
            400:610
        ] = 255

    points1 = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=15,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
        mask=mask
    )

    if points1 is None:
        print("No features found.")
        cap.release()
        return

    points2, status, error = (
        cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            points1,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )
    )

    good_old = points1[status == 1]
    good_new = points2[status == 1]

    result = frame2.copy()

    print(
        f"\nTracking results for {video_path}"
    )

    print("-" * 70)

    for i, (new, old) in enumerate(
        zip(good_new, good_old)
    ):

        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()

        u = x_new - x_old
        v = y_new - y_old

        magnitude = np.sqrt(
            u ** 2 + v ** 2
        )

        print(
            f"Point {i + 1}: "
            f"Frame 1 = "
            f"({x_old:.2f}, {y_old:.2f}) "
            f"-> Frame 2 = "
            f"({x_new:.2f}, {y_new:.2f}) "
            f"| u = {u:.2f}, "
            f"v = {v:.2f}, "
            f"magnitude = "
            f"{magnitude:.2f}"
        )

        old_pt = (
            int(x_old),
            int(y_old)
        )

        new_pt = (
            int(x_new),
            int(y_new)
        )

        cv2.circle(
            result,
            old_pt,
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            result,
            new_pt,
            5,
            (0, 0, 255),
            -1
        )

        cv2.arrowedLine(
            result,
            old_pt,
            new_pt,
            (255, 0, 0),
            2,
            tipLength=0.4
        )

        cv2.putText(
            result,
            str(i + 1),
            new_pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    os.makedirs(
        "outputs/tracking",
        exist_ok=True
    )

    cv2.imwrite(
        f"outputs/tracking/"
        f"{output_prefix}_frame1.jpg",
        frame1
    )

    cv2.imwrite(
        f"outputs/tracking/"
        f"{output_prefix}_frame2.jpg",
        frame2
    )

    cv2.imwrite(
        f"outputs/tracking/"
        f"{output_prefix}_tracking.jpg",
        result
    )

    cap.release()

    print(
        f"Saved tracking results "
        f"for {video_path}"
    )


# Cycle: restrict detection to cyclist
validate_tracking(
    "cycle.mp4",
    "cycle",
    150,
    True
)

# Beach: whole scene is fine
validate_tracking(
    "beach.mp4",
    "beach",
    150,
    False
)