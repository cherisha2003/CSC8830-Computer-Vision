
"""CSC 8830 - Module 7: Calibrated Stereo Measurement.

Run with:
    streamlit run app.py

The two video recordings are alternative inputs.
Two frames are extracted from the selected recording.
"""

from pathlib import Path
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from room_video import (
    frame_at,
    video_info,
    scale_camera_matrix,
    reconstruct_matched_points,
    point_pair_distance,
)
from stereo_geometry import (
    decode_image,
    calibrate_single_camera,
    estimate_pose,
    triangulate_points,
    measure_shape,
    rotate_calibration_90,
)


st.set_page_config(
    page_title="CSC 8830 - Stereo Measurement",
    layout="wide",
)

st.title("Module 7 · Calibrated Stereo Object Measurement")
st.caption(
    "Task 1: Calibrated stereo objects · "
    "Task 3: Video-based room reconstruction"
)

BASE = Path(__file__).resolve().parent

SAMPLES = {
    "Journal / rectangle": (
        "journal_view1.jpeg",
        "journal_view2.jpeg",
        "Rectangle",
    ),
    "Blue bowl / circle": (
        "lid_view1.jpeg",
        "lid_view2.jpeg",
        "Circle",
    ),
    "Pink pentagon / polygon": (
        "pentagon_view1.jpeg",
        "pentagon_view2.jpeg",
        "Polygon",
    ),
}


def as_pil(image):
    return Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )


def marked(image, points):
    result = as_pil(image)
    draw = ImageDraw.Draw(result)

    for i, (x, y) in enumerate(points):
        draw.ellipse(
            (x - 8, y - 8, x + 8, y + 8),
            fill="#ff4a52",
            outline="white",
            width=3,
        )
        draw.text(
            (x + 11, y - 12),
            str(i + 1),
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
    return result


def receive_image(upload, filename):
    if upload is not None:
        return decode_image(upload.getvalue())
    return cv2.imread(str(BASE / "sample_images" / filename))


cal_tab, measure_tab, room_tab, deriv_tab = st.tabs(
    [
        "1 · Camera calibration",
        "2 · Object measurement",
        "3 · Room video analysis",
        "4 · Uncalibrated derivation (plan)",
    ]
)


# ---------------------------------------------------
# TASK 1 - CAMERA CALIBRATION
# ---------------------------------------------------

with cal_tab:
    st.subheader("Calibrate iPhone intrinsics")

    st.write(
        "Use 15–20 photographs of the printed checkerboard "
        "at different angles and positions. The checkerboard "
        "has 9 × 6 inner corners and 2 cm squares."
    )

    a, b, c = st.columns(3)

    with a:
        nx = st.number_input(
            "Inner corners across",
            min_value=3,
            max_value=20,
            value=9,
        )
    with b:
        ny = st.number_input(
            "Inner corners down",
            min_value=3,
            max_value=20,
            value=6,
        )
    with c:
        square_cm = st.number_input(
            "Printed square side (cm)",
            min_value=0.1,
            value=2.0,
            step=0.1,
        )

    uploads = st.file_uploader(
        "Checkerboard images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button(
        "Load and calibrate using the 20 included checkerboard photographs"
    ):
        try:
            files = sorted(
                (BASE / "calibration_images").glob("*.jpg")
            )
            images = [
                cv2.imread(str(f))
                for f in files
            ]

            K, dist, rms, n, size = calibrate_single_camera(
                images,
                (int(nx), int(ny)),
                float(square_cm),
            )

            st.session_state["camera_cal"] = (
                K, dist, size
            )

            st.success(
                f"Calibrated using {n} photographs. "
                f"RMS reprojection error: {rms:.3f} pixels."
            )
            st.dataframe(K)

        except Exception as e:
            st.error(str(e))

    if st.button(
        "Run checkerboard calibration",
        type="primary",
    ):
        try:
            images = [
                decode_image(u.getvalue())
                for u in uploads
            ]
            K, dist, rms, n, size = calibrate_single_camera(
                images,
                (int(nx), int(ny)),
                float(square_cm),
            )

            st.session_state["camera_cal"] = (
                K, dist, size
            )

            st.success(
                f"Calibrated using {n} images. "
                f"RMS: {rms:.3f} pixels."
            )
            st.dataframe(K)

        except Exception as e:
            st.error(str(e))

    if "camera_cal" not in st.session_state:
        st.info(
            "No checkerboard calibration loaded yet."
        )
    else:
        st.caption(
            "Checkerboard intrinsics loaded. "
            "Verify that the camera lens and field of view "
            "match the images or video being measured."
        )


# ---------------------------------------------------
# TASK 1 - OBJECT MEASUREMENT
# ---------------------------------------------------

with measure_tab:
    example = st.selectbox(
        "Choose your captured object",
        list(SAMPLES),
    )

    left_col, right_col = st.columns(2)

    with left_col:
        upl = st.file_uploader(
            "Or replace view 1",
            type=["jpg", "jpeg", "png"],
            key="left_upload",
        )

    with right_col:
        upr = st.file_uploader(
            "Or replace view 2",
            type=["jpg", "jpeg", "png"],
            key="right_upload",
        )

    left_name, right_name, shape_default = SAMPLES[example]

    L = receive_image(upl, left_name)
    R = receive_image(upr, right_name)

    if L is None or R is None:
        st.error("One or both sample images are missing.")
        st.stop()

    if L.shape[:2] != R.shape[:2]:
        st.error(
            "Image dimensions differ. Both views must "
            "use compatible capture settings."
        )
        st.stop()

    st.caption(
        f"View 1: {left_name} · View 2: {right_name}"
    )

    with st.expander("Compare original photographs"):
        p1, p2 = st.columns(2)
        with p1:
            st.image(
                as_pil(L),
                caption=left_name,
                use_container_width=True,
            )
        with p2:
            st.image(
                as_pil(R),
                caption=right_name,
                use_container_width=True,
            )

    shape = st.selectbox(
        "Object shape",
        ["Rectangle", "Circle", "Polygon"],
        index=[
            "Rectangle", "Circle", "Polygon"
        ].index(shape_default),
    )

    baseline = st.number_input(
        "Camera displacement baseline (cm)",
        min_value=0.01,
        value=15.24,
        step=0.5,
        key=f"baseline-{example}",
        help=(
            "This must be the displacement between "
            "the two camera positions, not the "
            "camera-to-object distance."
        ),
    )

    st.caption(
        "Reference: bowl radius ≈ 10.16 cm, "
        "diameter ≈ 20.32 cm. "
        "The baseline must be verified independently."
    )

    calibrated = "camera_cal" in st.session_state
    orientation_setting = "same resolution"

    if calibrated:
        K, dist, cal_size = st.session_state["camera_cal"]
        photo_size = (L.shape[1], L.shape[0])

        if cal_size == photo_size:
            st.success(
                "Actual checkerboard intrinsics loaded."
            )

        elif cal_size == photo_size[::-1]:
            rotation_choice = st.selectbox(
                "Calibration image rotation",
                [
                    "Clockwise 90°",
                    "Counterclockwise 90°",
                ],
            )

            orientation_setting = rotation_choice

            K, dist, rotated_size = rotate_calibration_90(
                K,
                dist,
                cal_size,
                (
                    "clockwise"
                    if rotation_choice == "Clockwise 90°"
                    else "counterclockwise"
                ),
            )

            assert rotated_size == photo_size

        else:
            st.error(
                f"Calibration size {cal_size} is "
                f"incompatible with object size {photo_size}."
            )
            st.stop()

    else:
        st.warning(
            "EXPLORATORY MODE: no actual calibration."
        )

        approximate_f = st.number_input(
            "Hypothetical focal length in pixels",
            min_value=100.0,
            value=1250.0,
            step=50.0,
        )

        K = np.array(
            [
                [approximate_f, 0, L.shape[1] / 2],
                [0, approximate_f, L.shape[0] / 2],
                [0, 0, 1],
            ],
            dtype=float,
        )
        dist = np.zeros((1, 5), np.float64)

    pair_signature = (
        example,
        (
            (upl.name, len(upl.getvalue()))
            if upl is not None else left_name
        ),
        (
            (upr.name, len(upr.getvalue()))
            if upr is not None else right_name
        ),
        calibrated,
        float(K[0, 0]),
        orientation_setting,
    )

    if st.session_state.get("pose_for") != pair_signature:
        st.session_state.pop("pose", None)

    if st.button("Estimate motion from feature matches"):
        try:
            rotation, direction, stats = estimate_pose(
                L, R, K, dist
            )

            st.session_state["pose"] = (
                rotation, direction
            )
            st.session_state["pose_for"] = pair_signature

            st.success(
                f"{stats['pose_inliers']} inliers from "
                f"{stats['matches']} feature matches."
            )

            if abs(float(direction[2])) > 0.65:
                st.warning(
                    "Mostly forward/backward motion. "
                    "Depth measurements may be unreliable."
                )

        except Exception as e:
            st.session_state.pop("pose", None)
            st.error(str(e))

    current = pair_signature

    if st.session_state.get("selected_example") != current:
        st.session_state["selected_example"] = current
        st.session_state["pts_l"] = []
        st.session_state["pts_r"] = []

    st.markdown("#### Select corresponding object points")

    st.write(
        "Rectangle: top-left, top-right, bottom-right, "
        "bottom-left. Circle: opposite diameter endpoints. "
        "Pentagon: five corners clockwise. "
        "Use the same order in both views."
    )

    zoom = st.checkbox(
        "Zoom for precise corner selection",
        value=("pentagon" in example.lower()),
    )

    side = st.radio(
        "Place the next point on",
        ["View 1", "View 2"],
        horizontal=True,
    )

    pts_key = "pts_l" if side == "View 1" else "pts_r"
    img = L if side == "View 1" else R
    h, w = img.shape[:2]

    if zoom:
        if "pentagon" in example.lower():
            x0, y0, x1, y1 = (
                int(.29*w), int(.17*h),
                int(.83*w), int(.88*h)
            )
        elif "journal" in example.lower():
            x0, y0, x1, y1 = (
                int(.31*w), int(.23*h),
                int(.82*w), int(.89*h)
            )
        else:
            x0, y0, x1, y1 = (
                int(.29*w), int(.21*h),
                int(.82*w), int(.84*h)
            )
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    crop = img[y0:y1, x0:x1]

    transformed_points = [
        (x-x0, y-y0)
        for x, y in st.session_state[pts_key]
        if x0 <= x < x1 and y0 <= y < y1
    ]

    screen = marked(crop, transformed_points)
    max_display_width = 960

    if screen.width > max_display_width:
        display_scale = max_display_width / screen.width
        screen = screen.resize(
            (
                max_display_width,
                round(screen.height * display_scale),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        display_scale = 1.0

    click = streamlit_image_coordinates(
        screen,
        key=(
            f"click-{example}-{side}-{zoom}-"
            f"{len(st.session_state[pts_key])}"
        ),
    )

    if click is not None:
        new_point = (
            min(
                w - 1,
                x0 + round(click["x"] / display_scale),
            ),
            min(
                h - 1,
                y0 + round(click["y"] / display_scale),
            ),
        )

        st.session_state[pts_key].append(new_point)
        st.rerun()

    cl, cr = st.columns(2)

    with cl:
        if st.button("Undo last point"):
            if st.session_state[pts_key]:
                st.session_state[pts_key].pop()
            st.rerun()

    with cr:
        if st.button("Clear BOTH point sets"):
            st.session_state["pts_l"] = []
            st.session_state["pts_r"] = []
            st.rerun()

    st.write(
        "View 1:", st.session_state["pts_l"],
        "View 2:", st.session_state["pts_r"],
    )

    if st.button(
        "Calculate Z and dimensions",
        type="primary",
    ):
        try:
            if (
                "pose" not in st.session_state
                or st.session_state.get("pose_for") != current
            ):
                raise ValueError(
                    "Estimate camera pose for this pair first."
                )

            n = {
                "Rectangle": 4,
                "Circle": 2,
                "Polygon": 5,
            }[shape]

            if (
                len(st.session_state["pts_l"]) != n
                or len(st.session_state["pts_r"]) != n
            ):
                raise ValueError(
                    f"Select exactly {n} points per view."
                )

            rotation, t = st.session_state["pose"]

            xyz = triangulate_points(
                st.session_state["pts_l"],
                st.session_state["pts_r"],
                K, dist, rotation, t, baseline,
            )

            result = measure_shape(xyz, shape)
            zs = xyz[:, 2]

            if np.ptp(zs) > max(
                5.0,
                0.15 * abs(float(np.mean(zs))),
            ):
                st.error(
                    "Object points have widely differing "
                    "depths. Reconstruction may be unreliable."
                )

            st.subheader("Estimated reconstruction")

            st.metric(
                "Mean Z distance",
                f"{result['mean_depth_cm']:.2f} cm",
            )

            if shape == "Rectangle":
                a, b = st.columns(2)
                a.metric(
                    "Width",
                    f"{result['width_cm']:.2f} cm",
                )
                b.metric(
                    "Height",
                    f"{result['height_cm']:.2f} cm",
                )

            elif shape == "Circle":
                st.metric(
                    "Diameter",
                    f"{result['diameter_cm']:.2f} cm",
                )

            else:
                st.write(
                    "Polygon edges (cm)",
                    {
                        f"Edge {i+1}": round(e, 2)
                        for i, e in enumerate(
                            result["edge_lengths_cm"]
                        )
                    },
                )

            st.write(
                "3D points [X, Y, Z] in cm:",
                np.round(xyz, 2),
            )

            if not calibrated:
                st.error(
                    "EXPLORATORY ONLY: calibration was assumed."
                )

            st.warning(
                "The estimated baseline introduces uncertainty."
            )

        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------
# TASK 3 - ROOM VIDEO ANALYSIS
# ---------------------------------------------------

with room_tab:
    st.subheader("Task 3 · Room measurement using video frames")

    st.write(
        "Choose ONE room recording. Frame 1 and Frame 2 "
        "are taken at different timestamps within that "
        "same recording. Each video is an alternative "
        "camera pass, not a separate stereo camera."
    )

    room_videos = {
        "Video 1 - Portrait recording":
            BASE / "room_videos" / "Module7_Room_Video_1.mp4",
        "Video 2 - Portrait recording":
            BASE / "room_videos" / "Module7_Room_Video_2.mp4",
    }

    room_choice = st.selectbox(
        "Room video",
        list(room_videos) + ["Upload my own video"],
    )

    video_file = None
    video_path = None

    if room_choice == "Upload my own video":
        video_file = st.file_uploader(
            "Upload MP4 or MOV",
            type=["mp4", "mov", "m4v"],
            key="room_upload",
        )

        if video_file is not None:
            signature = (
                video_file.name,
                len(video_file.getvalue()),
            )

            if (
                st.session_state.get("room_upload_signature")
                != signature
            ):
                old = st.session_state.get(
                    "room_uploaded_path"
                )

                if old and os.path.exists(old):
                    os.unlink(old)

                with tempfile.NamedTemporaryFile(
                    suffix=".mp4",
                    delete=False,
                ) as tmp:
                    tmp.write(video_file.getvalue())
                    st.session_state[
                        "room_uploaded_path"
                    ] = tmp.name

                st.session_state[
                    "room_upload_signature"
                ] = signature

            video_path = Path(
                st.session_state["room_uploaded_path"]
            )
        else:
            st.info("Upload a room video to begin.")

    else:
        video_path = room_videos[room_choice]

        if video_path.is_file():
            st.caption(
                f"Loaded: {video_path.name} "
                f"({video_path.stat().st_size/1048576:.2f} MB)"
            )

            with st.expander(
                "Preview the selected full recording"
            ):
                st.video(video_path.read_bytes())
        else:
            st.error(
                f"Missing video: {video_path.name}. "
                "Check the room_videos folder."
            )

    if video_path is not None and video_path.is_file():
        try:
            meta = video_info(video_path)

            st.caption(
                f"Original video: "
                f"{meta['width']} × {meta['height']} · "
                f"{meta['fps']:.1f} FPS · "
                f"{meta['duration_s']:.1f} seconds"
            )

            a, b = st.columns(2)

            end = max(
                0.2,
                round(float(meta["duration_s"]) - 0.15, 1),
            )

            if room_choice.startswith("Video 1"):
                video_key = "video1"
                defaults = (3.0, 6.0)
            elif room_choice.startswith("Video 2"):
                video_key = "video2"
                defaults = (2.0, 4.0)
            else:
                video_key = "upload"
                defaults = (2.0, 4.0)

            with a:
                time1 = st.slider(
                    "Frame 1 timestamp (seconds)",
                    0.0, end,
                    min(defaults[0], end),
                    0.1,
                    key=f"room_t1_{video_key}",
                )

            with b:
                time2 = st.slider(
                    "Frame 2 timestamp (seconds)",
                    0.0, end,
                    min(defaults[1], end),
                    0.1,
                    key=f"room_t2_{video_key}",
                )

            if abs(time2 - time1) < 0.4:
                st.warning(
                    "Choose frames at least 0.4 seconds apart."
                )

            # Both frames intentionally come from the
            # currently selected video.
            F1, _ = frame_at(video_path, time1)
            F2, _ = frame_at(video_path, time2)

            if F1.shape != F2.shape:
                raise ValueError("Frame sizes differ.")

            room_sig = (
                str(video_path),
                round(time1, 2),
                round(time2, 2),
            )

            if st.session_state.get("room_sig") != room_sig:
                st.session_state["room_sig"] = room_sig
                st.session_state["room_pts1"] = []
                st.session_state["room_pts2"] = []
                st.session_state.pop("room_pose", None)

            qa, qb = st.columns(2)

            with qa:
                st.image(
                    as_pil(F1),
                    caption=f"Frame 1 · {time1:.1f}s",
                    use_container_width=True,
                )

            with qb:
                st.image(
                    as_pil(F2),
                    caption=f"Frame 2 · {time2:.1f}s",
                    use_container_width=True,
                )

            K_room = None
            D_room = None

            # -----------------------------------------
            # CORRECTED VIDEO CALIBRATION
            # -----------------------------------------

            if "camera_cal" in st.session_state:
                raw_K, raw_D, cal_size = (
                    st.session_state["camera_cal"]
                )

                vid_size = (F1.shape[1], F1.shape[0])

                if vid_size == cal_size:
                    K_room = np.array(
                        raw_K,
                        dtype=float,
                    )
                    D_room = raw_D

                elif abs(
                    vid_size[0] / vid_size[1]
                    - cal_size[0] / cal_size[1]
                ) < 0.015:
                    # Same orientation; different resolution.
                    K_room = scale_camera_matrix(
                        raw_K,
                        cal_size,
                        vid_size,
                    )
                    D_room = raw_D

                    st.info(
                        "Intrinsics scaled to video resolution. "
                        "Only valid if the video uses the same "
                        "lens, framing and field of view."
                    )

                elif abs(
                    vid_size[0] / vid_size[1]
                    - cal_size[1] / cal_size[0]
                ) < 0.015:
                    choice = st.selectbox(
                        "Checkerboard-to-video orientation",
                        [
                            "Clockwise 90°",
                            "Counterclockwise 90°",
                        ],
                        key="room_cal_rotation",
                    )

                    rot_K, rot_D, rot_size = (
                        rotate_calibration_90(
                            raw_K,
                            raw_D,
                            cal_size,
                            (
                                "clockwise"
                                if choice.startswith("Clockwise")
                                else "counterclockwise"
                            ),
                        )
                    )

                    K_room = scale_camera_matrix(
                        rot_K,
                        rot_size,
                        vid_size,
                    )
                    D_room = rot_D

                    st.info(
                        "Calibration rotated and scaled. "
                        "Verify that camera settings match."
                    )

                else:
                    st.error(
                        "Calibration and video aspect ratios "
                        "are incompatible."
                    )

            else:
                st.warning(
                    "Load the actual checkerboard calibration "
                    "in Tab 1 before estimating camera pose."
                )

            # -----------------------------------------
            # ESTIMATE CAMERA POSE
            # -----------------------------------------

            if K_room is not None:
                if st.button(
                    "Estimate camera pose between selected video frames",
                    type="primary",
                ):
                    try:
                        R_room, t_room, stats_room = (
                            estimate_pose(
                                F1, F2, K_room, D_room
                            )
                        )

                        st.session_state["room_pose"] = (
                            R_room,
                            t_room,
                            stats_room,
                        )

                        st.success(
                            f"{stats_room['pose_inliers']} "
                            "geometric inliers from "
                            f"{stats_room['matches']} "
                            "matched features."
                        )

                    except Exception as e:
                        st.session_state.pop(
                            "room_pose", None
                        )
                        st.error(str(e))

                if "room_pose" in st.session_state:
                    R_room, t_room, stats_room = (
                        st.session_state["room_pose"]
                    )

                    st.write(
                        "Estimated camera translation "
                        "direction [x, y, z]:",
                        np.round(t_room, 3).tolist(),
                    )

                    if abs(float(t_room[2])) > 0.65:
                        st.warning(
                            "Large forward/backward motion. "
                            "Depth may be unreliable."
                        )

                    st.markdown(
                        "#### Mark matching room landmarks"
                    )

                    st.caption(
                        "Doorframe test: point 1 = top-left, "
                        "point 2 = top-right (90.17 cm width), "
                        "point 3 = top-right again, "
                        "point 4 = bottom-right (height). "
                        "Mark the same physical corners "
                        "in both frames, in the same order."
                    )

                    marked_side = st.radio(
                        "Mark points on",
                        ["Frame 1", "Frame 2"],
                        horizontal=True,
                        key="room_side",
                    )

                    pts_key = (
                        "room_pts1"
                        if marked_side == "Frame 1"
                        else "room_pts2"
                    )

                    display_frame = (
                        F1
                        if marked_side == "Frame 1"
                        else F2
                    )

                    display_width = min(
                        750, display_frame.shape[1]
                    )
                    scale = (
                        display_width /
                        display_frame.shape[1]
                    )

                    shown = cv2.resize(
                        display_frame,
                        (
                            display_width,
                            round(
                                display_frame.shape[0] * scale
                            ),
                        ),
                    )

                    pts_render = [
                        (round(x*scale), round(y*scale))
                        for x, y in st.session_state[pts_key]
                    ]

                    click = streamlit_image_coordinates(
                        marked(shown, pts_render),
                        key=(
                            f"room_click_"
                            f"{marked_side}_{room_sig}"
                        ),
                        width=display_width,
                    )

                    click_key = (
                        f"{marked_side}_{room_sig}"
                    )

                    if (
                        click is not None
                        and st.session_state.get(
                            "last_room_click"
                        ) != (
                            click_key,
                            click["x"],
                            click["y"],
                        )
                    ):
                        st.session_state[
                            "last_room_click"
                        ] = (
                            click_key,
                            click["x"],
                            click["y"],
                        )

                        if len(
                            st.session_state[pts_key]
                        ) < 8:
                            st.session_state[
                                pts_key
                            ].append(
                                (
                                    click["x"] / scale,
                                    click["y"] / scale,
                                )
                            )
                            st.rerun()

                    k1, k2, k3 = st.columns(3)

                    k1.write(f"Frame 1: {len(st.session_state['room_pts1'])} selected")
                    k2.write(f"Frame 2: {len(st.session_state['room_pts2'])} selected")

                    if k3.button("Clear room points"):
                        st.session_state["room_pts1"] = []
                        st.session_state["room_pts2"] = []
                        st.rerun()

                    method = st.radio(
                        "How will you establish metric scale?",
                        [
                            "Measured room reference between points 1–2",
                            "Measured camera displacement between THESE frames",
                            "No metric scale (relative units only)",
                        ],
                        key="room_scale_method",
                    )

                    baseline_cm = None
                    reference_cm = None

                    if method.startswith("Measured room"):
                        reference_cm = st.number_input(
                            "Actual distance between points 1 and 2 (cm)",
                            min_value=0.1,
                            value=90.17,
                            step=1.0,
                            help=(
                                "Doorframe width = 90.17 cm. "
                                "Use only if points 1–2 "
                                "are its measured endpoints."
                            ),
                        )

                        st.caption(
                            "This reference establishes scale. "
                            "It is not an independent accuracy test."
                        )

                    elif method.startswith("Measured camera"):
                        baseline_cm = st.number_input(
                            "Camera displacement between these frames (cm)",
                            min_value=0.1,
                            value=15.0,
                            step=1.0,
                        )

                    if st.button(
                        "Triangulate landmarks and estimate room spans"
                    ):
                        try:
                            P1 = st.session_state["room_pts1"]
                            P2 = st.session_state["room_pts2"]

                            X, sc, source = (
                                reconstruct_matched_points(
                                    P1, P2,
                                    K_room, D_room,
                                    R_room, t_room,
                                    baseline_cm,
                                    reference_cm,
                                )
                            )

                            st.subheader(
                                "Reconstructed landmarks"
                            )

                            st.dataframe(np.round(X, 2))

                            if sc is None:
                                st.warning(
                                    "Coordinates are relative, "
                                    "not centimeters."
                                )

                            else:
                                st.success(
                                    f"Metric scale supplied by: {source}"
                                )

                                for i in range(
                                    0, len(X)-1, 2
                                ):
                                    label = (
                                        "Reference span (used for scale)"
                                        if (
                                            i == 0
                                            and reference_cm is not None
                                        )
                                        else (
                                            "Room span from marked "
                                            f"points {i+1}–{i+2}"
                                        )
                                    )

                                    st.metric(
                                        label,
                                        f"{point_pair_distance(X, i, i+1):.2f} cm",
                                    )

                                if np.ptp(X[:, 2]) > (
                                    0.3 *
                                    max(
                                        1,
                                        np.mean(X[:, 2]),
                                    )
                                ):
                                    st.warning(
                                        "Landmark depths vary widely. "
                                        "Check geometry and point matching."
                                    )

                            st.warning(
                                "Only marked spans are reconstructed. "
                                "A complete room model requires "
                                "additional visible landmarks."
                            )

                        except Exception as e:
                            st.error(str(e))

            st.markdown(
                "#### Independent room measurements"
            )

            st.write(
                "Doorframe width: 90.17 cm. "
                "Doorframe height: approximately 210.82 cm. "
                "Room measurements from the iPhone Measure "
                "app are approximate and should be compared "
                "with reconstructed values independently."
            )

            st.caption(
                "The known reference width establishes "
                "metric scale; it cannot independently "
                "validate that same width."
            )

        except Exception as e:
            st.error(
                f"Room-video analysis error: {e}"
            )


# ---------------------------------------------------
# TASK 2 - UNCALIBRATED DERIVATION
# ---------------------------------------------------

with deriv_tab:
    st.subheader(
        "Task 2 · Handwritten derivation"
    )

    st.write(
        "The professor requires handwritten, scanned "
        "derivations of uncalibrated stereo. "
        "This application does not replace them."
    )

    st.write(
        "Outline: correspondence → fundamental matrix "
        "and epipolar constraint → canonical projective "
        "cameras → triangulation → metric ambiguity "
        "→ constraints needed for physical measurements."
    )

    st.warning(
        "Uncalibrated stereo reconstruction has "
        "projective ambiguity. Absolute physical "
        "dimensions require additional constraints."
    )
