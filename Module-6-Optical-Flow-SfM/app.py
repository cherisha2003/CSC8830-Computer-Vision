import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="CSC 8830 - Assignment 6",
    page_icon="🎥",
    layout="wide"
)

st.title("CSC 8830 - Assignment 6")
st.caption("Optical Flow and Planar Structure from Motion")

tab1, tab2 = st.tabs(["Part 1: Optical Flow", "Part 2: Structure from Motion"])


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_uploaded_file(uploaded_file, suffix):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.close()
    return temp.name


def optical_flow_video(input_path, output_path, width=640):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = width / original_width
    height = int(original_height * scale)

    if height % 2 != 0:
        height -= 1

    # Try common MP4 codecs.
    writer = None
    used_codec = None

    for codec in ["mp4v", "avc1"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if test_writer.isOpened():
            writer = test_writer
            used_codec = codec
            break

        test_writer.release()

    if writer is None:
        cap.release()
        raise RuntimeError("Could not create the output video on this server.")

    ret, first_frame = cap.read()

    if not ret:
        cap.release()
        writer.release()
        raise RuntimeError("Could not read the first video frame.")

    first_frame = cv2.resize(first_frame, (width, height))
    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(first_frame)
    hsv[:, :, 1] = 255

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        horizontal = flow[:, :, 0]
        vertical = flow[:, :, 1]

        magnitude, angle = cv2.cartToPolar(horizontal, vertical)

        hsv[:, :, 0] = angle * 180 / np.pi / 2
        hsv[:, :, 2] = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(visualization)

        previous_gray = current_gray
        frame_count += 1

        if total_frames > 0:
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    writer.release()
    progress.empty()

    return used_codec, frame_count


def decode_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not decode {uploaded_file.name}.")

    return image


def run_planar_sfm(uploaded_images):
    checkerboard = (9, 6)
    square_size = 19.05

    object_points_single = np.zeros(
        (checkerboard[0] * checkerboard[1], 3),
        np.float32
    )

    object_points_single[:, :2] = (
        np.mgrid[
            0:checkerboard[0],
            0:checkerboard[1]
        ].T.reshape(-1, 2)
    )

    object_points_single *= square_size

    object_points = []
    image_points = []
    original_images = []
    corner_images = []
    image_size = None

    for uploaded in uploaded_images:
        image = decode_image(uploaded)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            None
        )

        if not found:
            raise RuntimeError(
                f"Checkerboard corners were not detected in {uploaded.name}."
            )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        object_points.append(object_points_single.copy())
        image_points.append(corners)
        original_images.append(image)

        detected = image.copy()
        cv2.drawChessboardCorners(
            detected,
            checkerboard,
            corners,
            found
        )

        corner_images.append(detected)

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    board_boundary = np.array(
        [
            [-square_size, -square_size, 0],
            [9 * square_size, -square_size, 0],
            [9 * square_size, 6 * square_size, 0],
            [-square_size, 6 * square_size, 0]
        ],
        dtype=np.float32
    )

    boundary_images = []
    camera_positions = []

    for image, rvec, tvec in zip(
        original_images,
        rvecs,
        tvecs
    ):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_position = -rotation_matrix.T @ tvec
        camera_positions.append(camera_position.flatten())

        projected, _ = cv2.projectPoints(
            board_boundary,
            rvec,
            tvec,
            camera_matrix,
            distortion
        )

        projected = projected.reshape(-1, 2).astype(int)

        result = image.copy()

        cv2.polylines(
            result,
            [projected],
            True,
            (0, 0, 255),
            5
        )

        labels = ["P1", "P2", "P3", "P4"]

        for point, label in zip(projected, labels):
            x, y = point

            cv2.circle(
                result,
                (x, y),
                8,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                result,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        boundary_images.append(result)

    return {
        "rms": rms,
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "corner_images": corner_images,
        "boundary_images": boundary_images,
        "camera_positions": camera_positions
    }


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# Part 1
# ---------------------------------------------------------

with tab1:
    st.subheader("Dense Optical Flow")

    st.write(
        "Upload a video with visible motion. The app calculates dense "
        "Farneback optical flow between consecutive frames. Color represents "
        "motion direction and brightness represents motion magnitude."
    )

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi"],
        key="flow_video"
    )

    if video_file is not None:
        st.video(video_file)

        if st.button("Generate optical flow", type="primary"):
            input_suffix = Path(video_file.name).suffix or ".mp4"
            input_path = save_uploaded_file(video_file, input_suffix)

            output_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )
            output_path = output_temp.name
            output_temp.close()

            try:
                with st.spinner("Calculating optical flow..."):
                    codec, frames = optical_flow_video(
                        input_path,
                        output_path
                    )

                st.success(
                    f"Finished processing {frames} frame pairs "
                    f"using {codec}."
                )

                st.video(output_path)

                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download optical-flow video",
                        file,
                        file_name="optical_flow_output.mp4",
                        mime="video/mp4"
                    )

            except Exception as error:
                st.error(str(error))

            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)


# ---------------------------------------------------------
# Part 2
# ---------------------------------------------------------

with tab2:
    st.subheader("Planar Structure from Motion")

    st.write(
        "Upload four views of the same 10 × 7 checkerboard. "
        "The printed square size used for this assignment is 19.05 mm."
    )

    sfm_files = st.file_uploader(
        "Upload exactly four checkerboard images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="sfm_images"
    )

    if len(sfm_files) > 0:
        st.write(f"Selected images: {len(sfm_files)}")

    if st.button("Run four-view reconstruction", type="primary"):
        if len(sfm_files) != 4:
            st.error("Please upload exactly four images.")
        else:
            try:
                with st.spinner("Detecting corners and calibrating the camera..."):
                    results = run_planar_sfm(sfm_files)

                st.success("All four images were processed successfully.")

                st.metric(
                    "RMS reprojection error",
                    f"{results['rms']:.4f} px"
                )

                st.markdown("#### Camera intrinsic matrix")
                st.code(
                    np.array2string(
                        results["camera_matrix"],
                        precision=3,
                        suppress_small=True
                    )
                )

                st.markdown("#### Distortion coefficients")
                st.code(
                    np.array2string(
                        results["distortion"],
                        precision=4,
                        suppress_small=True
                    )
                )

                st.markdown("#### Detected 9 × 6 inner corners")

                columns = st.columns(2)

                for i, image in enumerate(results["corner_images"]):
                    with columns[i % 2]:
                        st.image(
                            bgr_to_rgb(image),
                            caption=f"View {i + 1}: 54 detected inner corners",
                            use_container_width=True
                        )

                st.markdown("#### Reconstructed checkerboard boundary")

                columns = st.columns(2)

                for i, image in enumerate(results["boundary_images"]):
                    with columns[i % 2]:
                        st.image(
                            bgr_to_rgb(image),
                            caption=f"View {i + 1}: projected outer boundary",
                            use_container_width=True
                        )

                st.markdown("#### Estimated camera positions")

                for i, position in enumerate(
                    results["camera_positions"],
                    start=1
                ):
                    st.write(
                        f"View {i}: "
                        f"({position[0]:.2f}, "
                        f"{position[1]:.2f}, "
                        f"{position[2]:.2f}) mm"
                    )

                report_lines = [
                    f"RMS reprojection error: {results['rms']:.6f}",
                    "",
                    "Camera matrix:",
                    str(results["camera_matrix"]),
                    "",
                    "Distortion coefficients:",
                    str(results["distortion"]),
                    "",
                    "Estimated camera positions (mm):"
                ]

                for i, position in enumerate(
                    results["camera_positions"],
                    start=1
                ):
                    report_lines.append(
                        f"View {i}: {position}"
                    )

                result_text = "\n".join(report_lines)

                st.download_button(
                    "Download calibration results",
                    result_text,
                    file_name="calibration_results.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(str(error))


st.divider()
st.caption(
    "CSC 8830 - Computer Vision | Assignment 6 | "
    "Optical Flow and Planar Structure from Motion"
)
