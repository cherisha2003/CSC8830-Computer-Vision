# CSC 8830 · Assignment 7 — Stereo Object Size Estimation (In Progress)

This folder implements the **Task 1 workflow** from Assignment 7, using original camera photos of a journal, blue bowl, and pink pentagon. The code estimates **Z (depth) before size** and supports rectangle width/height, circle diameter, and each polygon edge.

**Current status:** Task 1 prototype (syntax and geometry-unit tests verified; Streamlit UI has not been end-to-end tested in this environment) and calibration workflow, **not yet calibrated/validated**. Tasks 2 (handwritten uncalibrated derivation) and 3 (room measurements from video) are **not completed**. Do not represent this folder as a finished assignment.

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
3. In **Object measurement**, select original journal/bowl/pentagon photo pair. The scene must remain still between views.
4. Enter the actual measured displacement of the camera optical center. The current **15.24 cm (6 inch) default is an approximate reported measurement**, not verified; entering it does not make the setup calibrated.
5. Click **Estimate motion from feature matches**. This estimates the direction of camera translation and rotation via the essential matrix; it does not recover an absolute baseline.
6. Click corresponding points **in the same order** on View 1 and View 2, then **Calculate Z and dimensions**. Report results only after checking the reconstructed geometry and comparing with independent physical measurements.

### Original reference readings (approximate iPhone Measure app)

- Journal rectangle: width 16.5 cm, height 21.6 cm.
- Blue bowl diameter: about 4 inches (10.16 cm), based on an approximate iPhone Measure reading.
- Paper pentagon: two visible sides measured at about 3.5 inches (8.9 cm); **do not assume every edge was measured**.
- Reported displacement: about 6 inches (15.24 cm) based on an iPhone Measure screenshot; confirm this is actual lens-to-lens translation for each stereo pair. Camera-to-object reference distance was about 22 inches (55.88 cm) in the blue-bowl setup, and is NOT a baseline.

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

## Initial photo-pair diagnostic (exploratory focal length ONLY)

ORB matches / recoverPose inliers: journal 2261 / 1333, lid 175 / 143, pentagon 229 / 192. The journal pair estimated a translation direction dominated by forward/backward motion rather than the stated 15 cm lateral movement; **do not use it for final depth measurements without recalibration/recapture and validation**. These are feature-matching diagnostics, NOT evidence of accurate size.

## Interface update and public hosting

The zoomed image selector maps displayed click coordinates back to original image coordinates. The gallery shows both original photos, with distinct filenames. The journal pair is the user-confirmed IMG_4838.jpeg (View 1) and IMG_4840.jpeg (View 2), and both are different images.

To publish via Streamlit Community Cloud, commit this folder to your GitHub repository, then use [share.streamlit.io](https://share.streamlit.io) → Create app → Existing app. Choose `cherisha2003/CSC8830-Computer-Vision`, branch `main`, and entrypoint `Module-7-Stereo-Vision/app.py`. Keep `requirements.txt` beside `app.py`. The `*.app.github.dev` address is a Codespaces development preview; public deployments use `*.streamlit.app`.

**Please retain all exploration/calibration warnings until the actual photos have been calibrated and all measurements validated.**

### Portrait checkerboard / landscape object photographs

The Object measurement tab detects a 90-degree orientation difference and lets you choose a clockwise or counterclockwise transform of the intrinsic camera matrix and distortion coefficients. This is valid only if the two sets of photos are the same lens, sensor mode, zoom and resolution after rotation, with no cropping. Check the correct rotation direction against the originals. The translation baseline of about 15 cm was hand-estimated and remains a source of uncertainty. Successful checkerboard reprojection alone does not validate 3D dimensions.

## November 12 portrait photo update

All three object pairs use portrait photos (1152 × 1536):

| Object | View 1 | View 2 |
|---|---|---|
| Journal | `IMG_4912.jpeg` | `IMG_4914.jpeg` |
| Pentagon | `IMG_4915.jpeg` | `IMG_4917.jpeg` |
| Blue bowl | `IMG_4922.jpeg` | `IMG_4920.jpeg` |

The filenames above are source originals; the app loads them under `sample_images/` as `*_view1.jpeg` and `*_view2.jpeg`. The 20 checkerboard photos are included for upload in `calibration_images/`, but calibration is **not** automatically persisted across separate browser sessions. Upload the originals in the Calibration tab and rerun calibration (9 × 6 inner corners, provisional printed square size 2.00 cm).

The object reference screenshots are preserved in `reference_measurements/`: approximately 22-inch camera-to-object distance, 6-inch camera displacement, and 4-inch blue-bowl diameter. These are approximate iPhone Measure estimates, not tape-measured ground truth. The bowl's curved rim and surface can also create inconsistent depths; click two opposite points on the same rim plane. Do not report the measurements as validated until the 3D checks agree with independent physical measurements.

**Still not complete:** room-video application (Task 3) and handwritten uncalibrated stereo derivation (Task 2).
