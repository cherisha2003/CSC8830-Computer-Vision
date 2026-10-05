# Human Boundary Detection - Module 4

Cherisha Killari Giribabu · CSC 8830 · Panther ID: 002901729

This app finds a person's boundary in an RGB image and a thermal image. It also compares the masks with SAM2 and includes a Fourier filtering demonstration.

## Run it

Use Python 3.10 or newer. Open Terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## What the webpage shows

The **Saved examples** tab opens first. It already contains the input images, classical masks, classical boundaries, actual SAM2 masks, overlays, and scores. Nothing needs to be uploaded to see these results. Use these same examples when recording the video.

The next two tabs let someone upload an RGB or thermal image. The **Fourier analysis** tab displays the saved RGB example by default and also accepts uploads. It includes a thresholded low-pass reconstruction to demonstrate region formation. The **Theory & derivations** tab explains the DFT, edge operators, convolution theorem, region thresholds, and a worked numerical example.

Keep the entire `examples` folder when uploading to GitHub. The webpage reads those files directly, so the images still appear after deployment. It does not need to download a model or reach another website to display them.

## Methods

**RGB:** Marker-based watershed. The user supplies a box around the person, points inside the person, and optional points on the background. Pixels outside the box are background markers. After slight Gaussian smoothing, watershed grows regions from those markers. We keep the marked foreground regions and extract their contours. The saved example shows every point used. This approach uses no learned or fitted statistical model. It can miss differently colored body parts, so a point on each part helps.

**Thermal:** Gaussian smoothing, Otsu thresholding inside the box, opening/closing, component selection, and contour extraction. Select “hot” for a brighter person or “cold” for a darker person. Prefer single-channel thermal intensity; grayscale from a false-color display is not calibrated temperature. The saved example uses a crop from a real LLVIP infrared frame.

**Fourier:** The 2-D FFT separates spatial frequencies. A circular low-pass filter produces a smoother reconstruction; the complementary high-pass filter emphasizes edges and texture. The high-pass display is contrast-normalized. It is not a human segmentation mask.

These methods estimate visible outlines. They cannot guarantee exact boundaries in every scene and do not automatically recognize which object is a human.

## Saved comparison

| Image | Classical method | IoU with SAM2 | Dice with SAM2 |
| --- | --- | ---: | ---: |
| RGB football player | Watershed | 0.7672 | 0.8682 |
| Thermal pedestrian | Otsu + morphology | 0.8263 | 0.9049 |

The reference model is the original **SAM2 Hiera Tiny**, with one output mask and optional postprocessing disabled. Both methods received the same box. For RGB, SAM2 also received the same foreground/background points. Settings are recorded in `examples/settings.json`.

IoU = intersection / union. Dice = 2 × intersection / total foreground area. These numbers measure agreement with SAM2, not accuracy against an independently labeled ground truth. The RGB result misses some clothing and narrow structures; the thermal result differs at the coat, bag, and limbs. Inspect the displayed overlays.

## Rebuild the saved outputs

```bash
python generate_examples.py
```

This reruns the classical methods and reuses the included SAM2 masks. To regenerate the reference masks as well, install SAM2 using [Meta's instructions](https://github.com/facebookresearch/sam2), then run:

```bash
python generate_examples.py --run-sam2
```

The first SAM2 run downloads weights and needs more memory than the web app. Alternatively, pass the original SAM2 Tiny checkpoint with `--checkpoint /path/to/sam2_hiera_tiny.pt`. Model weights are not included in this ZIP. The webpage needs only `requirements.txt`.

For a new image and box:

```bash
python sam2_reference.py frame.jpg sam2_mask.png 20 10 400 500
```

Use your own coordinates. Upload the generated mask to the matching image tab. Masks must be black and white and have the exact same dimensions as the input image.

## Files

- `app.py`: webpage and uploads.
- `explanations.py`: method explanations beside the examples.
- `THEORY.md`: mathematical derivations displayed in the theory tab.
- `segmentation.py`: watershed, thermal processing, contours, metrics, and Fourier views.
- `sam2_reference.py`: separate optional SAM2 inference.
- `generate_examples.py`: rebuild the bundled example outputs.
- `examples/`: inputs, outputs, settings, and source notes.
- `DEMO_SCRIPT.md`: short guide for recording the website.

## Deploy and submit

Create a public GitHub repository and upload all these files, including the `examples` folder. In Streamlit Community Cloud, select that repo and `app.py` as the main file. Check that the saved examples appear on the deployed page.

Add the repository URL, deployed URL, actual screenshots, and these measured results to the final report. The earlier PDF draft described GrabCut; update that section to watershed before submitting. Record the website showing the saved RGB, thermal, and Fourier examples, then demonstrate an upload. Submit the final PDF and video in Classroom.

## Sources

See `examples/SOURCES.md` for image credits and the infrared crop coordinates. The masks in this project are newly computed outputs, not the datasets' official annotations.
