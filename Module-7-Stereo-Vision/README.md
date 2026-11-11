# CSC 8830 · Assignment 7 — Stereo Object Size Estimation (In Progress)

This folder implements the **Task 1 workflow** from Assignment 7, using original camera photos of a journal, transparent lid, and pink pentagon. The code estimates **Z (depth) before size** and supports rectangle width/height, circle diameter, and each polygon edge.

**Current status:** functional Task 1 prototype and calibration workflow, **not yet calibrated/validated**. Tasks 2 (handwritten uncalibrated derivation) and 3 (room measurements from video) are **not completed**. Do not represent this folder as a finished assignment.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Task 1 steps

1. Print a checkerboard with **9 × 6 inner corners** (10 × 7 squares). Measure the side of an actual printed square. Capture at least 8 and preferably 15–20 **varied views** using the same iPhone 1× lens, orientation, mode, and resolution as the object photos.
2. Upload checkerboard frames on the **Camera calibration** tab and click **Run checkerboard calibration**. Observe the reprojection RMS, and reject poor calibration.
3. In **Object measurement**, select original journal/lid/pentagon photo pair. The scene must remain still between views.
4. Enter the actual measured displacement of the camera optical center. The current **15 cm value is an estimate**, not verified; entering it does not make the setup calibrated.
5. Click **Estimate motion from feature matches**. This estimates the direction of camera translation and rotation via the essential matrix; it does not recover an absolute baseline.
6. Click corresponding points **in the same order** on View 1 and View 2, then **Calculate Z and dimensions**. Report results only after checking the reconstructed geometry and comparing with independent physical measurements.

### Original reference readings (approximate iPhone Measure app)

- Journal rectangle: width 16.5 cm, height 21.6 cm.
- Plastic lid diameter: 14 cm.
- Paper pentagon: two visible sides measured at about 3.5 inches (8.9 cm); **do not assume every edge was measured**.
- Camera translation for each pair: **about** 15 cm by hand; uncertainty not measured.

**Crucial:** A moving iPhone is *not a fixed calibrated stereo rig*. Intrinsic calibration of the iPhone alone does not calibrate the extrinsic geometry of each stereo pair. The prototype estimates that pose from visual feature matching; feature degeneracy, lens changes, baseline inaccuracies, planar scenes, and reflections can make measurements unusable. The provided no-calibration mode uses a *hypothetical* focal length for interface testing and **does not satisfy Task 1's calibrated-stereo requirement**.

## How the math works

Camera intrinsics and distortion: `cv2.calibrateCamera` from checkerboard. Feature correspondence: ORB + ratio filtering. Recover pairwise relative pose: essential matrix `cv2.findEssentialMat` / `cv2.recoverPose` with normalized undistorted coordinates. Absolute translation length: independent baseline measurement. 3D points: `cv2.triangulatePoints` using projection matrices `[I | 0]`, `[R | t]` in normalized coordinates. Depth is the 3D **Z coordinate** in the first camera frame; physical edge length is the Euclidean distance between 3D vertices.

## Remaining work before submitting

- [ ] Capture checkerboard images and verify calibration quality (and lens consistency).
- [ ] Re-capture stereo pairs with a reliably measured baseline if results are unstable.
- [ ] Validate all three objects with independent ruler readings; provide error percentages and depth plots.
- [ ] **Problem 2:** write, scan, and include handwritten uncalibrated stereo derivation; discuss projective/scale ambiguity.
- [ ] **Problem 3:** implement and test video-based **room length, width and height**, not pose/hand tracking. Record independent scale measurements.
- [ ] Add screenshots, references, final report PDF, GitHub link and demo video for Classroom.

## References

- OpenCV official camera calibration documentation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- OpenCV `calib3d` reference (`findEssentialMat`, `recoverPose`, `triangulatePoints`): https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- CSC 8830 Assignment 7, user-provided professor specification.
