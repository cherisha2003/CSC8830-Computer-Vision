"""Classical human-mask pipelines. Run the web app with `streamlit run app.py`.

Inputs are RGB uint8 arrays and an optional (x0, y0, x1, y1) person box.
Neither pipeline imports or invokes a learned model.
"""
import cv2
import numpy as np


def box_for(image, box=None):
    h, w = image.shape[:2]
    if box is None:
        return (int(.1*w), int(.05*h), int(.9*w), int(.95*h))
    x0, y0, x1, y1 = map(int, box)
    x0, x1 = sorted((max(0, min(w-1, x0)), max(1, min(w, x1))))
    y0, y1 = sorted((max(0, min(h-1, y0)), max(1, min(h, y1))))
    if x1-x0 < 2 or y1-y0 < 2:
        raise ValueError("The person box needs a positive width and height.")
    return x0, y0, x1, y1


def select_component(mask, box):
    """Keep the component overlapping the box center most, else the largest."""
    x0, y0, x1, y1 = box
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, np.uint8)
    center = labels[(y0+y1)//2, (x0+x1)//2]
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 24]
    if not ids:
        return np.zeros_like(mask, np.uint8)
    chosen = int(center) if center in ids else max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == chosen).astype(np.uint8) * 255


def clean(mask, box, kernel_size=3):
    k = max(1, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return select_component(mask, box)


def rgb_mask(image, box, foreground=None, background=None, kernel_size=3):
    """Watershed from hand-picked points; green seeds mark the person."""
    x0, y0, x1, y1 = box_for(image, box)
    markers = np.ones(image.shape[:2], np.int32)
    markers[y0:y1, x0:x1] = 0
    foreground = foreground or [((x0+x1)//2, (y0+y1)//2)]
    background = background or []
    for points, label in [(background, 1), (foreground, 2)]:
        for x, y in points:
            if not (x0 <= x < x1 and y0 <= y < y1):
                raise ValueError("Seed points must be inside the person box.")
            cv2.circle(markers, (int(x), int(y)), 2, label, -1)
    # Smoothing reduces tiny regions caused by image noise.
    smooth = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.watershed(cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR), markers)
    raw = np.uint8(markers == 2) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    # Keep separate seeded parts such as a hand or shoe.
    return mask, raw


def thermal_mask(image, box, polarity="hot", kernel_size=3):
    """Otsu threshold on a thermal frame; box gates competing hot/cold regions.

    For false-color frames, luminance is a display value, not calibrated temperature.
    Prefer a raw single-channel image where available.
    """
    x0, y0, x1, y1 = box_for(image, box)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    roi = gray[y0:y1, x0:x1]
    threshold, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mode = cv2.THRESH_BINARY if polarity == "hot" else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(gray, threshold, 255, mode)
    gate = np.zeros_like(gray, np.uint8)
    gate[y0:y1, x0:x1] = 255
    raw = cv2.bitwise_and(binary, gate)
    return clean(raw, (x0, y0, x1, y1), kernel_size), raw, float(threshold)


def overlay(image, mask, color=(255, 40, 40)):
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, color, 2)
    return base


def scores(a, b):
    a, b = a > 0, b > 0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    total = int(a.sum() + b.sum())
    return {"IoU": intersection / union if union else 1.,
            "Dice": 2*intersection / total if total else 1.}


def fourier_views(image, radius_fraction=.12):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    yy, xx = np.ogrid[:h, :w]
    radius = max(1, radius_fraction*min(h, w))
    low = ((yy-h//2)**2 + (xx-w//2)**2) <= radius**2
    def display(arr):
        return cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = display(np.log1p(np.abs(spectrum)))
    smooth = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum*low)))
    high = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum*(~low))))
    return magnitude, np.clip(smooth, 0, 255).astype(np.uint8), display(high)
