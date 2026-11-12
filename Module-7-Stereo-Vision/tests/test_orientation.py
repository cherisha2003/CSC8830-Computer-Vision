import cv2
import numpy as np
from stereo_geometry import rotate_calibration_90


def test_rotate_intrinsics_both_directions():
    K = np.array([[1184, 0, 587], [0, 1183, 763], [0, 0, 1]], float)
    distortion = np.array([[.03, -.04, .002, -.003, .01]], float)
    points = np.array([[.3, .22, 1], [-.1, .4, 1], [.05, -.13, 1]], float)
    original = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, distortion)[0][:, 0, :]
    for direction in ('clockwise', 'counterclockwise'):
        K2, d2, size = rotate_calibration_90(K, distortion, (1152, 1536), direction)
        assert size == (1536, 1152)
        if direction == 'clockwise':
            p2 = np.stack([-points[:, 1], points[:, 0], points[:, 2]], axis=1)
            expected = np.stack([1535-original[:, 1], original[:, 0]], axis=1)
        else:
            p2 = np.stack([points[:, 1], -points[:, 0], points[:, 2]], axis=1)
            expected = np.stack([original[:, 1], 1151-original[:, 0]], axis=1)
        actual = cv2.projectPoints(p2, np.zeros(3), np.zeros(3), K2, d2)[0][:, 0, :]
        np.testing.assert_allclose(actual, expected, atol=1e-8)
