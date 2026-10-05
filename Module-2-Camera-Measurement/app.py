"""
CSC 8830 - Module 2: Camera Calibration and Object Measurement

This Streamlit application presents the calibration results, measures the
width and height of a flat rectangular object, summarizes the validation
experiment, and explains the two-camera projection relationship.

Before running:
    1. Run python3 calibrate_camera.py at least once.
    2. Keep calibration_results/calibration_data.npz in the project.

Run:
    streamlit run app.py
"""

from io import BytesIO
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates


CALIBRATION_FILE = Path("calibration_results/calibration_data.npz")
CALIBRATION_REPORT = Path("calibration_results/calibration_report.json")
UNDISTORTED_IMAGE = Path("calibration_results/undistorted_example.jpg")
CORNER_FOLDER = Path("calibration_results/detected_corners")
EXPERIMENT_FILE = Path("results/measurement_results.csv")


def scale_camera_matrix(camera_matrix, calibration_size, image_size):
    """Scale focal lengths and principal point for the uploaded resolution."""
    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size
    scale_x = image_width / calibration_width
    scale_y = image_height / calibration_height

    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def point_distance(point_a, point_b):
    """Return the Euclidean distance between two image points."""
    return float(np.linalg.norm(point_a - point_b))


def measurement_error(estimated, actual):
    """Return absolute error and percentage error."""
    absolute = abs(estimated - actual)
    percentage = absolute / actual * 100
    return absolute, percentage


def initialize_state():
    """Create the session values used by the interactive measurement page."""
    defaults = {
        "points": [],
        "last_click": None,
        "uploaded_image_id": None,
        "reset_number": 0,
        "measurements": [],
    }
    for name, value in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = value


def load_calibration():
    """Load the camera matrix and distortion coefficients."""
    if not CALIBRATION_FILE.exists():
        st.error(
            "Calibration data was not found. Run calibrate_camera.py first."
        )
        st.stop()

    calibration = np.load(CALIBRATION_FILE)
    return (
        calibration["camera_matrix"].astype(np.float64),
        calibration["distortion_coefficients"].astype(np.float64),
        int(calibration["image_width"]),
        int(calibration["image_height"]),
    )


def show_overview():
    st.header("CSC 8830 Module 2")
    st.write(
        "This application calibrates a smartphone camera and uses perspective "
        "projection to estimate the real-world width and height of flat objects."
    )
    st.markdown(
        """
        **Project sequence**

        1. Calibrate the iPhone camera using a checkerboard.
        2. Measure a rectangular object from its image coordinates and depth.
        3. Validate the method using 20 dimension measurements at 2.50 meters.
        4. Derive the relationship between observations from two cameras.
        """
    )
    st.info(
        "The experiment used 10 objects. Width and height were measured for "
        "each object, producing 20 dimension measurements."
    )


def show_calibration(camera_matrix, distortion, width, height):
    st.header("Step 1: Smartphone camera calibration")
    st.write(
        "The iPhone camera was calibrated in OpenCV using a printed "
        "10 × 7 square checkerboard with 9 × 6 internal corners. "
        "Each square measured 25.0 mm."
    )

    report = {}
    if CALIBRATION_REPORT.exists():
        with open(CALIBRATION_REPORT, encoding="utf-8") as report_file:
            report = json.load(report_file)

    column_1, column_2, column_3 = st.columns(3)
    column_1.metric("Calibration images", report.get("total_images", 36))
    column_2.metric("Accepted images", report.get("accepted_count", 18))
    column_3.metric(
        "RMS reprojection error",
        f"{report.get('opencv_rms_reprojection_error_pixels', 0.9242):.4f} px",
    )

    st.write(f"Calibration resolution: {width} × {height}")
    st.write("Camera intrinsic matrix:")
    st.code(np.array2string(camera_matrix, precision=4))
    st.write("Distortion coefficients:")
    st.code(np.array2string(distortion, precision=6))

    if report:
        st.write(
            "Mean per-image reprojection error: "
            f"{report['mean_per_image_reprojection_error_pixels']:.4f} pixels"
        )
        st.write(
            "Maximum per-image reprojection error: "
            f"{report['maximum_per_image_reprojection_error_pixels']:.4f} pixels"
        )

    image_column_1, image_column_2 = st.columns(2)
    corner_images = sorted(CORNER_FOLDER.glob("*.jpg"))
    if corner_images:
        image_column_1.image(
            str(corner_images[0]),
            caption="Detected checkerboard corners",
            width="stretch",
        )
    if UNDISTORTED_IMAGE.exists():
        image_column_2.image(
            str(UNDISTORTED_IMAGE),
            caption="Undistorted calibration image",
            width="stretch",
        )


def draw_selected_points(image, points):
    """Draw selected corners and connecting lines on the displayed image."""
    result = image.copy()
    drawing = ImageDraw.Draw(result)

    for index, (x, y) in enumerate(points):
        radius = 7
        drawing.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="red",
            outline="white",
            width=2,
        )
        drawing.text((x + 10, y - 15), str(index + 1), fill="yellow")

    if len(points) >= 2:
        drawing.line(points, fill="lime", width=4)
    if len(points) == 4:
        drawing.line([points[-1], points[0]], fill="lime", width=4)
    return result


def show_measurement_tool(
    original_camera_matrix,
    distortion,
    calibration_width,
    calibration_height,
):
    st.header("Step 2: Real-world 2D object measurement")
    st.write(
        "Upload a portrait photograph, enter the measured depth, and select "
        "the four corners of the same flat rectangular face."
    )

    uploaded_file = st.file_uploader(
        "Object photograph",
        type=["jpg", "jpeg", "png"],
    )
    distance_mm = st.number_input(
        "Camera-to-object distance (mm)",
        min_value=1.0,
        value=2500.0,
        step=10.0,
    )
    if distance_mm <= 2000:
        st.warning(
            "The validation experiment requires a distance greater than 2000 mm."
        )

    object_name = st.text_input("Object name", value="Object 1")

    if uploaded_file is None:
        return

    uploaded_bytes = uploaded_file.getvalue()
    image_id = hashlib.md5(uploaded_bytes).hexdigest()
    if st.session_state.uploaded_image_id != image_id:
        st.session_state.uploaded_image_id = image_id
        st.session_state.points = []
        st.session_state.last_click = None
        st.session_state.reset_number += 1

    image = ImageOps.exif_transpose(Image.open(BytesIO(uploaded_bytes)))
    image = image.convert("RGB")
    image_width, image_height = image.size
    st.write(f"Uploaded resolution: {image_width} × {image_height}")

    calibration_ratio = calibration_width / calibration_height
    image_ratio = image_width / image_height
    if abs(calibration_ratio - image_ratio) > 0.02:
        st.error(
            "The image orientation or aspect ratio does not match the "
            "calibration photographs."
        )
        return

    camera_matrix = scale_camera_matrix(
        original_camera_matrix,
        (calibration_width, calibration_height),
        (image_width, image_height),
    )

    display_width = 650
    display_height = round(image_height * display_width / image_width)
    display_image = image.resize((display_width, display_height))
    display_image = draw_selected_points(display_image, st.session_state.points)

    st.subheader("Select four corners")
    st.write(
        "Click in this order: **top-left, top-right, bottom-right, bottom-left**."
    )
    if st.button("Reset selected points"):
        st.session_state.points = []
        st.session_state.last_click = None
        st.session_state.reset_number += 1
        st.rerun()

    clicked = streamlit_image_coordinates(
        display_image,
        width=display_width,
        key=f"image_{image_id}_{st.session_state.reset_number}",
        cursor="crosshair",
    )
    if clicked is not None and len(st.session_state.points) < 4:
        click_id = (clicked["x"], clicked["y"], clicked.get("unix_time"))
        if click_id != st.session_state.last_click:
            st.session_state.points.append((clicked["x"], clicked["y"]))
            st.session_state.last_click = click_id
            st.rerun()

    st.write(f"Selected points: {len(st.session_state.points)} / 4")
    if len(st.session_state.points) != 4:
        return

    display_points = np.array(st.session_state.points, dtype=np.float32)
    image_points = display_points.copy()
    image_points[:, 0] *= image_width / display_width
    image_points[:, 1] *= image_height / display_height

    undistorted = cv2.undistortPoints(
        image_points.reshape(-1, 1, 2),
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(-1, 2)

    top_left, top_right, bottom_right, bottom_left = undistorted
    pixel_width = (
        point_distance(top_left, top_right)
        + point_distance(bottom_left, bottom_right)
    ) / 2
    pixel_height = (
        point_distance(top_left, bottom_left)
        + point_distance(top_right, bottom_right)
    ) / 2

    focal_x = camera_matrix[0, 0]
    focal_y = camera_matrix[1, 1]
    estimated_width = pixel_width * distance_mm / focal_x
    estimated_height = pixel_height * distance_mm / focal_y

    st.subheader("Estimated dimensions")
    result_1, result_2 = st.columns(2)
    result_1.metric("Estimated width", f"{estimated_width:.2f} mm")
    result_2.metric("Estimated height", f"{estimated_height:.2f} mm")

    st.write("Perspective projection equations:")
    st.latex(r"W=\frac{w_pZ}{f_x},\qquad H=\frac{h_pZ}{f_y}")
    st.write(f"Pixel width: {pixel_width:.2f} pixels")
    st.write(f"Pixel height: {pixel_height:.2f} pixels")

    st.subheader("Validation values")
    input_1, input_2 = st.columns(2)
    actual_width = input_1.number_input(
        "Actual width (mm)", min_value=0.0, value=0.0, step=1.0
    )
    actual_height = input_2.number_input(
        "Actual height (mm)", min_value=0.0, value=0.0, step=1.0
    )

    if actual_width > 0 and actual_height > 0:
        width_absolute, width_percentage = measurement_error(
            estimated_width, actual_width
        )
        height_absolute, height_percentage = measurement_error(
            estimated_height, actual_height
        )

        st.write(
            f"Width error: {width_absolute:.2f} mm ({width_percentage:.2f}%)"
        )
        st.write(
            f"Height error: {height_absolute:.2f} mm ({height_percentage:.2f}%)"
        )

        if st.button("Record this measurement"):
            st.session_state.measurements.append(
                {
                    "Object": object_name,
                    "Distance (mm)": distance_mm,
                    "Estimated width (mm)": estimated_width,
                    "Actual width (mm)": actual_width,
                    "Width absolute error (mm)": width_absolute,
                    "Width error (%)": width_percentage,
                    "Estimated height (mm)": estimated_height,
                    "Actual height (mm)": actual_height,
                    "Height absolute error (mm)": height_absolute,
                    "Height error (%)": height_percentage,
                }
            )
            st.success("Measurement recorded.")

    if st.session_state.measurements:
        session_results = pd.DataFrame(st.session_state.measurements)
        st.subheader("Measurements recorded in this session")
        st.dataframe(session_results, width="stretch")
        st.download_button(
            "Download session measurements",
            data=session_results.to_csv(index=False).encode("utf-8"),
            file_name="measurement_results.csv",
            mime="text/csv",
        )


def show_validation():
    st.header("Step 3: Experimental validation")
    st.write(
        "Ten rectangular objects were photographed at approximately "
        "2.50 meters (8 ft 2 in). Width and height were evaluated for each "
        "object, giving 20 dimension measurements."
    )

    if not EXPERIMENT_FILE.exists():
        st.warning("The file results/measurement_results.csv was not found.")
        return

    results = pd.read_csv(EXPERIMENT_FILE)
    st.dataframe(results, width="stretch")

    error_columns = [
        "Width absolute error (mm)",
        "Width error (%)",
        "Height absolute error (mm)",
        "Height error (%)",
    ]
    statistics = results[error_columns].agg(["mean", "std", "min", "max"])
    st.subheader("Width and height error statistics")
    st.dataframe(statistics, width="stretch")

    combined_absolute = pd.concat(
        [
            results["Width absolute error (mm)"],
            results["Height absolute error (mm)"],
        ],
        ignore_index=True,
    )
    combined_percentage = pd.concat(
        [results["Width error (%)"], results["Height error (%)"]],
        ignore_index=True,
    )

    st.subheader("Combined statistics for all 20 measurements")
    combined_table = pd.DataFrame(
        {
            "Statistic": ["Mean", "Standard deviation", "Minimum", "Maximum"],
            "Absolute error (mm)": [
                combined_absolute.mean(),
                combined_absolute.std(),
                combined_absolute.min(),
                combined_absolute.max(),
            ],
            "Percentage error (%)": [
                combined_percentage.mean(),
                combined_percentage.std(),
                combined_percentage.min(),
                combined_percentage.max(),
            ],
        }
    )
    st.dataframe(combined_table, width="stretch", hide_index=True)

    st.write(
        "The main sources of error were manual corner selection, small object "
        "size in the image, distance uncertainty, and imperfect alignment "
        "between the camera and object plane."
    )


def show_theory():
    st.header("Theory: Relationship between two camera images")
    st.write(
        "Camera 1 defines the world coordinate system. Camera 2 is displaced "
        "and rotated relative to Camera 1. The scene point is "
        r"$P=[X,Y,Z,1]^T$, and its homogeneous image points are "
        r"$p_1=[u_1,v_1,1]^T$ and $p_2=[u_2,v_2,1]^T$."
    )

    st.subheader("Projection equations")
    st.latex(r"s_1p_1=K_1[I\mid0]P")
    st.latex(r"s_2p_2=K_2[R\mid t]P")
    st.write(
        r"$K_1$ and $K_2$ are the intrinsic matrices. $R$ is the rotation "
        r"from Camera 1 to Camera 2. If the second camera center is $C_2$ "
        r"in Camera 1 coordinates, then $t=-RC_2$."
    )

    st.subheader("Direct relationship when depth is known")
    st.write(
        r"The 3D point reconstructed along the Camera 1 ray is "
        r"$P_{3D}=ZK_1^{-1}p_1$. Substitution into Camera 2 gives:"
    )
    st.latex(r"p_2\sim K_2\left(RZK_1^{-1}p_1+t\right)")

    st.subheader("Relationship when depth is unknown")
    st.write(
        "Without the depth, the matching point in Camera 2 is constrained to "
        "an epipolar line rather than one exact image location."
    )
    st.latex(r"p_2^TFp_1=0")
    st.latex(r"E=[t]_\times R,\qquad F=K_2^{-T}EK_1^{-1}")

    st.subheader("Parameters and assumptions")
    st.markdown(
        """
        - The cameras follow the pinhole model after distortion correction.
        - The cameras and scene point remain fixed while the images are taken.
        - The point is visible in both images and the baseline is nonzero.
        - Intrinsic matrices are obtained by individual camera calibration.
        - Relative rotation and translation can be obtained by stereo calibration,
          or from matched points using the essential matrix when intrinsics are known.
        - Pixel correspondences can be detected manually or with feature matching.
        """
    )


st.set_page_config(
    page_title="Camera Calibration and Measurement",
    page_icon="📐",
    layout="wide",
)
initialize_state()
camera_matrix, distortion, calibration_width, calibration_height = load_calibration()

st.title("Smartphone Camera Calibration and Object Measurement")
tabs = st.tabs(
    [
        "Overview",
        "Step 1: Calibration",
        "Step 2: Measurement",
        "Step 3: Validation",
        "Theory",
    ]
)

with tabs[0]:
    show_overview()
with tabs[1]:
    show_calibration(
        camera_matrix,
        distortion,
        calibration_width,
        calibration_height,
    )
with tabs[2]:
    show_measurement_tool(
        camera_matrix,
        distortion,
        calibration_width,
        calibration_height,
    )
with tabs[3]:
    show_validation()
with tabs[4]:
    show_theory()
