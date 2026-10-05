"""Streamlit demo: `streamlit run app.py`. See README for inputs and SAM2 setup."""
import io
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from segmentation import rgb_mask, thermal_mask, overlay, scores, fourier_views
from explanations import explain_method, explain_scores

st.set_page_config(page_title="Module 4 | Human Boundaries", layout="wide")
st.title("Human boundary detection | Module 4")
st.caption("Classical OpenCV segmentation and an independently generated SAM2 reference mask")

def read_image(upload, mode="RGB"):
    img = Image.open(upload)
    if mode == "L" and img.mode in ("I;16", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        return np.uint8(np.clip((arr-lo)*255/max(hi-lo, 1), 0, 255))
    return np.asarray(img.convert(mode))

def png_button(label, array, filename):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, "PNG")
    st.download_button(label, buf.getvalue(), filename, "image/png")

example_dir = Path(__file__).parent / "examples"
tab_examples, tab_rgb, tab_thermal, tab_fourier, tab_theory = st.tabs(
    ["Saved examples", "Try an RGB image", "Try a thermal image", "Fourier analysis", "Theory & derivations"]
)

with tab_examples:
    st.subheader("Example results")
    st.write("This experiment estimates a person’s visible boundary from RGB and thermal images, then compares the results with SAM2. The examples below show the inputs, masks, boundaries, and measured agreement.")
    explain_scores()
    settings = json.loads((example_dir / "settings.json").read_text())
    for kind, title in [("rgb", "RGB: football player"), ("thermal", "Thermal: pedestrian")]:
        st.markdown(f"### {title}")
        explain_method(kind)
        for columns, names in [(st.columns(3), [(f"{kind}.jpg", "Input image"),
                                               (f"{kind}_classical_mask.png", "Classical mask"),
                                               (f"{kind}_classical.png", "Classical boundary (red)")]),
                               (st.columns(3), [(f"{kind}_sam2_mask.png", "SAM2 mask"),
                                               (f"{kind}_sam2.png", "SAM2 boundary (green)"),
                                               (f"{kind}_comparison.png", "Both boundaries")])]:
            for column, (filename, caption) in zip(columns, names):
                column.image(str(example_dir / filename), caption=caption, width="stretch")
        result = settings[kind]
        left, right = st.columns(2)
        left.metric("IoU with SAM2", f"{result['scores']['IoU']:.3f}")
        right.metric("Dice with SAM2", f"{result['scores']['Dice']:.3f}")
        st.caption("Agreement with SAM2 is not ground-truth accuracy. Both methods received the same target box.")
        if kind == "rgb":
            st.write("Watershed uses hand-picked foreground and background points. SAM2 received those same points. Thin parts and changes in clothing color can split the classical mask.")
            with st.expander("View the points used for the RGB example"):
                st.image(str(example_dir / "rgb_seeds.png"), caption="Green: foreground. Red: background. Yellow: target box.")
        else:
            st.write("The classical method uses Otsu thresholding on thermal intensity. This example is a crop from a real LLVIP infrared frame; it is not a recolored RGB photo.")
        with st.expander(f"{title}: settings and downloads"):
            st.json(result)
            for suffix in ["classical_mask", "sam2_mask", "comparison"]:
                path = example_dir / f"{kind}_{suffix}.png"
                st.download_button(f"Download {suffix.replace('_', ' ')}", path.read_bytes(), path.name,
                                   "image/png", key=f"saved_{kind}_{suffix}")
    st.markdown("Image sources: [OpenCV sample](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg) · [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/)")


def read_points(text):
    """Read coordinates entered as x,y; x,y."""
    if not text.strip():
        return []
    return [tuple(map(int, pair.strip().split(","))) for pair in text.split(";") if pair.strip()]


def run_tab(kind):
    upload = st.file_uploader(f"Upload {kind} image", type=["png", "jpg", "jpeg", "tif", "tiff"], key=kind)
    if upload is None:
        st.info("Upload a frame with one clearly visible person. Draw a close box around that person using the controls below.")
        return
    image = read_image(upload, "RGB" if kind == "RGB" else "L")
    h, w = image.shape[:2]
    if min(h, w) < 10:
        st.error("Please use an image at least 10 pixels wide and tall.")
        return
    st.image(image, caption=f"Input, {w} × {h} px", width="stretch")
    st.write("Person box (pixel coordinates). Include the whole person while excluding other people when possible.")
    c = st.columns(4)
    x0 = c[0].number_input("Left x", 0, w-2, int(.1*w), key=kind+"x0")
    y0 = c[1].number_input("Top y", 0, h-2, int(.05*h), key=kind+"y0")
    x1 = c[2].number_input("Right x", int(x0)+2, w, max(int(x0)+2, int(.9*w)), key=kind+"x1")
    y1 = c[3].number_input("Bottom y", int(y0)+2, h, max(int(y0)+2, int(.95*h)), key=kind+"y1")
    box = (x0, y0, x1, y1)
    kernel = st.slider("Morphology kernel (odd pixels)", 1, 15, 3, step=2, key=kind+"kernel")
    try:
        if kind == "RGB":
            st.caption("Add points inside the person, including differently colored clothes and limbs. Format: x,y; x,y.")
            foreground = st.text_input("Foreground points", f"{(x0+x1)//2},{(y0+y1)//2}")
            background = st.text_input("Background points inside the box (optional)", "")
            mask, raw = rgb_mask(image, box, read_points(foreground), read_points(background), kernel)
            st.caption("Watershed grows regions from the marked points using image boundaries. No model is trained or fitted.")
        else:
            polarity = st.selectbox("Target appearance", ["hot", "cold"])
            mask, raw, threshold = thermal_mask(image, box, polarity, kernel)
            st.caption(f"Otsu threshold within the box: {threshold:.1f} display-intensity units. False-color images should be converted to raw intensity before use.")
    except (cv2.error, ValueError) as exc:
        st.error(f"Segmentation could not run on this image and box: {exc}")
        return
    if not np.any(mask):
        st.warning("No component remained. Try a closer box, opposite thermal polarity, or smaller morphology kernel.")
    c1, c2, c3 = st.columns(3)
    c1.image(raw, caption="Initial binary mask", width="stretch")
    c2.image(mask, caption="Cleaned classical mask", width="stretch")
    c3.image(overlay(image, mask), caption="Classical boundary (red)", width="stretch")
    png_button("Download classical mask", mask, f"{kind.lower()}_classical_mask.png")

    st.subheader("Compare with SAM2")
    ref = st.file_uploader("Upload the SAM2 person mask for this exact frame (white person, black background)",
                           type=["png", "jpg", "jpeg"], key=kind+"sam")
    if ref is None:
        st.info("Run the optional sam2_reference.py locally or export a SAM2 mask, then upload it here. Metrics appear only after a real reference mask is supplied.")
    else:
        sam = read_image(ref, "L")
        if sam.shape != mask.shape:
            st.error(f"Mask shape {sam.shape} does not match input {mask.shape}. Use the same uncropped image, with identical dimensions.")
            return
        sam = np.uint8(sam > 127)*255
        if not np.any(sam):
            st.error("The SAM2 mask contains no foreground pixels.")
            return
        metrics = scores(mask, sam)
        a, b = st.columns(2)
        a.image(overlay(image, sam, (0, 230, 100)), caption="SAM2 boundary (green)", width="stretch")
        both = overlay(overlay(image, mask), sam, (0, 230, 100))
        b.image(both, caption="Classical (red) and SAM2 (green)", width="stretch")
        st.metric("Intersection over Union", f"{metrics['IoU']:.3f}")
        st.metric("Dice coefficient", f"{metrics['Dice']:.3f}")
        st.caption("These scores measure agreement with SAM2, not absolute ground-truth accuracy.")

with tab_rgb:
    explain_method("rgb")
    run_tab("RGB")
with tab_thermal:
    explain_method("thermal")
    run_tab("Thermal")
with tab_fourier:
    st.subheader("How frequency filtering changes an image")
    st.write("The Fourier transform represents repeating intensity patterns. Low frequencies describe broad changes; higher frequencies include edges, fine textures, and noise. The center of the displayed spectrum is zero frequency.")
    st.write("Move the radius slider: a smaller radius keeps fewer frequencies and smooths more detail. The high-pass view contains the removed changes. Full derivations are in the Theory & derivations tab.")
    uploaded = st.file_uploader("Upload an image to explore its spectrum", type=["png", "jpg", "jpeg", "tif", "tiff"])
    st.caption("The saved RGB example is shown by default. You can also upload another image.")
    if uploaded is not None or (example_dir / "rgb.jpg").exists():
        image = read_image(uploaded if uploaded is not None else example_dir / "rgb.jpg", "L")
        fraction = st.slider("Low-pass radius (% of shorter image side)", 1, 40, 12)/100
        magnitude, low, high = fourier_views(image, fraction)
        for col, arr, title in zip(st.columns(4), [image, magnitude, low, high],
                                   ["Intensity", "Log Fourier magnitude", "Low frequencies", "High frequencies"]):
            col.image(arr, caption=title, width="stretch")
        st.caption("The high-pass display is contrast-normalized and shows texture as well as edges; it is not itself a human mask.")

        st.subheader("From a reconstruction to regions")
        st.write("This extra step thresholds the low-pass reconstruction with Otsu's method. It shows how filtering can support segmentation. The white regions include any sufficiently bright objects, not just the person.")
        threshold, regions = cv2.threshold(low, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.image(regions, caption=f"Thresholded low-pass reconstruction (Otsu threshold: {threshold:.0f})", width=500)
        st.write("Low-pass filtering is linear; converting its output into foreground/background regions requires a nonlinear decision such as this threshold. For a human mask, a target prompt and further region selection are still needed.")

with tab_theory:
    st.markdown((Path(__file__).parent / "THEORY.md").read_text())
