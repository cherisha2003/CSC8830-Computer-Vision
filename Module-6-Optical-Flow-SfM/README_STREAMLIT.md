# CSC 8830 - Assignment 6

This repository contains the implementation and web demonstration for Assignment 6.

## Parts

### Part 1 - Optical Flow
- Dense Farneback optical flow
- Optical-flow video visualization
- Lucas-Kanade feature tracking and pixel validation
- Bilinear interpolation discussion and calculations

### Part 2 - Planar Structure from Motion
- Four checkerboard viewpoints
- 9 x 6 inner-corner detection
- Camera calibration
- Camera pose estimation
- Checkerboard boundary reconstruction

## Checkerboard Measurements

- 10 x 7 printed squares
- 9 x 6 inner corners
- Square size: 19.05 mm
- Full checkerboard size: 190.50 mm x 133.35 mm

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit deployment

Push `app.py` and `requirements.txt` to GitHub. Then create a Streamlit Community Cloud app and select `app.py` as the entrypoint.
