"""
CSC 8830 - Assignment 6
Optical Flow Visualization
"""

import cv2
import numpy as np
import os


def create_writer(output_path, fps, width, height):

    codecs = ["avc1", "mp4v"]

    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if writer.isOpened():
            print(f"Using codec: {codec}")
            return writer

        writer.release()

    return None


def create_optical_flow_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Could not open input video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"Input: {input_path} | "
        f"{original_width}x{original_height} | "
        f"{fps:.2f} FPS"
    )

    output_width = 960

    scale = output_width / original_width

    output_height = int(
        original_height * scale
    )

    # Make dimensions even for video encoding
    if output_height % 2 != 0:
        output_height -= 1

    writer = create_writer(
        output_path,
        fps,
        output_width,
        output_height
    )

    if writer is None:
        print(
            f"Could not create output video: "
            f"{output_path}"
        )

        cap.release()
        return

    ret, first_frame = cap.read()

    if not ret:
        print(
            f"Could not read first frame "
            f"from {input_path}"
        )

        cap.release()
        writer.release()
        return

    first_frame = cv2.resize(
        first_frame,
        (output_width, output_height)
    )

    previous_gray = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY
    )

    hsv = np.zeros_like(first_frame)

    hsv[:, :, 1] = 255

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(
            frame,
            (output_width, output_height)
        )

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        horizontal_flow = flow[:, :, 0]

        vertical_flow = flow[:, :, 1]

        magnitude, angle = cv2.cartToPolar(
            horizontal_flow,
            vertical_flow
        )

        hsv[:, :, 0] = (
            angle * 180 / np.pi / 2
        )

        hsv[:, :, 2] = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        flow_visualization = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

        writer.write(
            flow_visualization
        )

        previous_gray = current_gray

        frame_count += 1

    cap.release()
    writer.release()

    print(
        f"Finished {output_path} "
        f"with {frame_count} frames"
    )


os.makedirs(
    "outputs",
    exist_ok=True
)


print("\nProcessing cycle video...")

create_optical_flow_video(
    "cycle.mp4",
    "outputs/cycle_optical_flow.mp4"
)


print("\nProcessing beach video...")

create_optical_flow_video(
    "beach.mp4",
    "outputs/beach_optical_flow.mp4"
)


print("\nDone processing both videos.")