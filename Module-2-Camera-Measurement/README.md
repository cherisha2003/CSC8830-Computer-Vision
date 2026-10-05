# CSC 8830 Module 2: Camera Calibration and Object Measurement

This project calibrates an iPhone camera with OpenCV and estimates the
real-world width and height of a flat rectangular object from one photograph.
It also reports results from an experiment containing 20 dimension
measurements.

## Author

Cherisha Killari Giribabu  
CSC 8830 Computer Vision

## Web application

Public Streamlit application: https://csc8830-camera-measurement-ia4ndc6tsdrg4sqmn24rx6.streamlit.app/

## Assignment components

### Step 1: Camera calibration

The camera was calibrated using 36 photographs of a printed checkerboard.
The board contained 10 × 7 squares, which gave OpenCV 9 × 6 internal corners.
Each printed square measured 25.0 mm.

OpenCV detected the checkerboard in 18 photographs. The final calibration
results were:

- Image resolution: 3024 × 4032 pixels
- OpenCV RMS reprojection error: 0.9242 pixels
- Mean per-image reprojection error: 0.9071 pixels
- Maximum per-image reprojection error: 1.1984 pixels

The calibration script saves the camera matrix, distortion coefficients,
corner-detection previews, an undistorted example, and a JSON report.

### Step 2: Real-world dimension measurement

The Streamlit application asks the user to upload a photograph, enter the
camera-to-object distance, and click four corners in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

The selected coordinates are corrected for lens distortion. The program then
averages the two horizontal sides and the two vertical sides. Under the
assumption that the object face is parallel to the image plane, the dimensions
are calculated using:

```text
W = wp Z / fx
H = hp Z / fy
```

Here, `wp` and `hp` are the measured pixel dimensions, `Z` is the measured
depth in millimeters, and `fx` and `fy` are the calibrated focal lengths in
pixels.

### Step 3: Experimental validation

Ten rectangular objects were photographed at approximately 2.50 meters
(8 ft 2 in). Width and height were evaluated for each object, producing 20
dimension measurements.

Combined results for the 20 measurements:

- Mean absolute error: 16.69 mm
- Standard deviation of absolute error: 9.80 mm
- Mean percentage error: 8.05%
- Standard deviation of percentage error: 4.88%
- Minimum percentage error: 0.98%
- Maximum percentage error: 14.75%

The experimental CSV is stored in `results/measurement_results.csv`.

## Assumptions

- The same iPhone rear 1× camera setting is used for calibration and testing.
- The uploaded photograph has the same orientation and aspect ratio as the
  calibration images.
- Lens distortion is corrected using the saved OpenCV calibration parameters.
- The measured object face is flat and approximately parallel to the camera
  sensor.
- All four corners belong to the same object plane.
- The camera-to-object distance is measured from the lens to the object face.

## Project structure

```text
CSC8830-Camera-Measurement/
├── app.py
├── calibrate_camera.py
├── requirements.txt
├── README.md
├── calibration_images/
├── calibration_results/
│   ├── calibration_data.npz
│   ├── calibration_report.json
│   ├── detected_corners/
│   └── undistorted_example.jpg
├── measurement_images/
├── evidence/
└── results/
    └── measurement_results.csv
```

The local `venv/` and `calibration_images_original/` folders are excluded from
GitHub.

## Local installation

Open Terminal in the project directory and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run camera calibration

Place the calibration photographs in `calibration_images/`, then run:

```bash
python3 calibrate_camera.py
```

## Run the web application

```bash
streamlit run app.py
```

Open the local address displayed in Terminal, normally
`http://localhost:8501`.

## Main limitations

The result is sensitive to the accuracy of the depth measurement, manual
corner selection, camera-object alignment, and the number of pixels occupied
by the object. Small or oblique objects generally produce larger errors.
