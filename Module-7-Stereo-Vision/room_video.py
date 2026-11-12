"""Video keyframe analysis and metric reconstruction for Module 7 Task 3.

Camera pose from a monocular video has an unknown global translation scale.
Any centimeter measurement must have an independent measured scale input.
"""
import cv2
import numpy as np
from stereo_geometry import estimate_pose, triangulate_points


def frame_at(path, second, target_width=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Could not open room video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < 2:
        cap.release(); raise ValueError('Missing video timestamps or frames')
    duration = count / fps
    t = float(np.clip(second, 0, max(0, duration - 1/fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok: raise ValueError('Failed to decode selected video frame')
    if target_width and target_width < frame.shape[1]:
        scale = target_width / frame.shape[1]
        frame = cv2.resize(frame, (target_width, round(frame.shape[0] * scale)))
    return frame, duration


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Could not open video')
    fps = float(cap.get(cv2.CAP_PROP_FPS)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0: raise ValueError('Invalid video FPS')
    return {'fps': fps, 'duration_s': frames/fps, 'frames':frames, 'width':width,'height':height}


def scale_camera_matrix(K, old_size, new_size):
    """Only use for pure resize, with no cropping or camera/lens changes."""
    ow,oh = old_size; nw,nh = new_size
    if min(ow,oh,nw,nh) <= 0: raise ValueError('Bad camera resolution')
    if abs(nw/ow - nh/oh) > .015:
        raise ValueError('Aspect ratio changed too much. Camera intrinsics cannot safely be resized.')
    out = np.asarray(K,float).copy()
    out[0,:] *= nw/ow
    out[1,:] *= nh/oh
    return out


def reconstruct_matched_points(points1, points2, K, D, R, t, baseline_cm=None,
                               reference_cm=None, reference_indices=(0,1)):
    """Triangulate selected matching pixels. Scale by known camera baseline OR
    independently measured length between first two reconstructed reference points.
    The reference length itself must not be claimed as independent validation.
    """
    a=np.asarray(points1,float).reshape(-1,2)
    b=np.asarray(points2,float).reshape(-1,2)
    if a.shape != b.shape or len(a)<2:
        raise ValueError('Select the same number of corresponding points in both frames (at least 2).')
    unit_xyz = triangulate_points(a,b,K,D,R,t,1.0)
    if not np.isfinite(unit_xyz).all() or np.any(unit_xyz[:,2] <= 0):
        raise ValueError('Points behind the camera or invalid: check matches and change frame pair.')
    if baseline_cm is not None:
        if baseline_cm <= 0: raise ValueError('Measured camera movement must be positive')
        scale=float(baseline_cm)
        source='Measured camera-center displacement between the selected frames'
    elif reference_cm is not None:
        i,j=reference_indices
        if reference_cm <= 0: raise ValueError('Reference length must be positive')
        unit_length=np.linalg.norm(unit_xyz[i]-unit_xyz[j])
        if unit_length <= 1e-6:raise ValueError('Reference endpoints triangulate too close together')
        scale=float(reference_cm)/unit_length
        source='Independent reference length between points 1 and 2'
    else:
        return unit_xyz, None, 'UNKNOWN (unit translation only)'
    return unit_xyz*scale, scale, source


def point_pair_distance(points, i, j):
    pts=np.asarray(points,float)
    return float(np.linalg.norm(pts[i]-pts[j]))
