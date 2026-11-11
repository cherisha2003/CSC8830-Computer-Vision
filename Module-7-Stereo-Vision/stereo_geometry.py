"""Calibrated two-view reconstruction helpers for CSC 8830 Module 7.

Single moving camera: K and distortion calibrated from checkerboard photos;
relative rotation/translation DIRECTION estimated per pair from scene feature matches;
translation SCALE supplied by independently measured camera displacement.
"""
from __future__ import annotations
import cv2
import numpy as np


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not open image')
    return img


def calibrate_single_camera(images, inner_corners=(9, 6), square_size_cm=2.0):
    """Estimate intrinsics from diverse checkerboard views (same lens/resolution)."""
    nx, ny = inner_corners
    obj = np.zeros((nx * ny, 3), np.float32)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size_cm
    obj_points, img_points, size = [], [], None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is not None and (gray.shape[1], gray.shape[0]) != size:
            raise ValueError('Calibration photos must all have the same resolution')
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, (nx, ny),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(obj.copy()); img_points.append(corners)
    if len(obj_points) < 8:
        raise ValueError(f'Only {len(obj_points)} usable checkerboard photos. Capture at least 8, preferably 15–20 varied views.')
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return K, dist, rms, len(obj_points), size


def estimate_pose(img1, img2, K, dist):
    """ORB correspondence + essential matrix + relative pose from SAME moving camera.

    Two-view recovery returns translation direction, NOT metric translation size.
    This can be unreliable with mostly planar scenes or small motion.
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=5500)
    kp1, d1 = orb.detectAndCompute(g1, None)
    kp2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        raise ValueError('Not enough visual features to estimate camera motion')
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < .74 * b.distance]
    if len(good) < 20:
        raise ValueError(f'Only {len(good)} feature correspondences; need at least 20')
    p1 = np.float64([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float64([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    u1 = cv2.undistortPoints(p1, K, dist).reshape(-1, 2)
    u2 = cv2.undistortPoints(p2, K, dist).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(u1, u2, np.eye(3), method=cv2.RANSAC,
                                    prob=0.999, threshold=0.002)
    if E is None:
        raise ValueError('Essential-matrix estimation failed')
    # Several E candidates can be returned; recoverPose accepts stacked matrices in many OpenCV builds.
    if E.shape[0] > 3:
        E = E[:3, :]
    n, R, t, posemask = cv2.recoverPose(E, u1, u2, np.eye(3), mask=mask)
    inlier_count = int(np.count_nonzero(posemask))
    if inlier_count < 12:
        raise ValueError(f'Pose estimation found only {inlier_count} consistent matches')
    return R, t.reshape(3), {'matches': len(good), 'pose_inliers': inlier_count}


def triangulate_points(left_xy, right_xy, K, dist, R, t_direction, baseline_cm):
    """Triangulate matched pixels into first-camera 3D frame, in centimeters."""
    L = np.asarray(left_xy, dtype=np.float64).reshape(-1, 1, 2)
    Q = np.asarray(right_xy, dtype=np.float64).reshape(-1, 1, 2)
    if len(L) != len(Q) or len(L) < 2:
        raise ValueError('Select at least 2 matching points in both images')
    if baseline_cm <= 0:
        raise ValueError('Measured baseline must be positive')
    l = cv2.undistortPoints(L, K, dist).reshape(-1, 2).T
    r = cv2.undistortPoints(Q, K, dist).reshape(-1, 2).T
    t = baseline_cm * np.asarray(t_direction, dtype=float).reshape(3, 1) / np.linalg.norm(t_direction)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])
    h = cv2.triangulatePoints(P1, P2, l, r)
    XYZ = (h[:3] / h[3]).T
    return XYZ


def measure_shape(points_3d, shape):
    xyz = np.asarray(points_3d, float)
    if not np.isfinite(xyz).all():
        raise ValueError('Triangulated points are invalid')
    if np.any(xyz[:, 2] <= 0):
        raise ValueError('One or more points are behind the camera. Check match order and camera geometry.')
    out = {'depths_cm': xyz[:, 2].tolist(), 'mean_depth_cm': float(np.mean(xyz[:, 2]))}
    if shape == 'Circle':
        if len(xyz) != 2:
            raise ValueError('For a circle select exactly two opposite diameter endpoints')
        out['diameter_cm'] = float(np.linalg.norm(xyz[1]-xyz[0]))
    elif shape == 'Rectangle':
        if len(xyz) != 4:
            raise ValueError('For rectangle select TL, TR, BR, BL in that order')
        edges = [float(np.linalg.norm(xyz[(i+1)%4]-xyz[i])) for i in range(4)]
        out['edge_lengths_cm'] = edges
        out['width_cm'] = (edges[0]+edges[2])/2
        out['height_cm'] = (edges[1]+edges[3])/2
    else:
        if len(xyz) < 3:
            raise ValueError('For polygon select 3 or more consecutive vertices')
        out['edge_lengths_cm'] = [float(np.linalg.norm(xyz[(i+1)%len(xyz)]-xyz[i])) for i in range(len(xyz))]
    return out
