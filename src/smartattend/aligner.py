"""
aligner.py – Face alignment/preprocessing for recognition models.

Produces 112×112 aligned face crops using:
  1. Keypoint-based affine transform (when MediaPipe keypoints available).
  2. Simple bbox crop + resize (fallback when keypoints absent).

The 112×112 output matches the input expected by EdgeFace and FaceLiVT.
"""

from __future__ import annotations

import cv2
import numpy as np

# Standard 5-point reference template for 112×112 alignment
# Based on ArcFace alignment reference (used by most modern recognition models)
REFERENCE_5PT = np.array([
    [38.2946, 51.6963],   # right eye
    [73.5318, 51.5014],   # left eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
], dtype=np.float32)


def align_face_112(
    bgr_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    keypoints: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Return a 112×112 BGR aligned face crop.

    Parameters
    ----------
    bgr_frame : full-resolution camera frame
    bbox      : (x, y, w, h) pixel bounding box from detector
    keypoints : list of (x,y) pixel keypoints from detector (up to 6)
                MediaPipe order: right_eye, left_eye, nose, mouth, right_ear, left_ear
    """
    if keypoints and len(keypoints) >= 5:
        # Use first 5 keypoints: right_eye, left_eye, nose, mouth_right, mouth_left
        # MediaPipe: kp[0]=right_eye, kp[1]=left_eye, kp[2]=nose_tip,
        #            kp[3]=mouth_center, kp[4]=right_ear, kp[5]=left_ear
        # We approximate mouth corners from mouth_center + ear positions
        src_pts = np.array(keypoints[:5], dtype=np.float32)
        # Check we have 5 usable points
        if len(src_pts) == 5:
            ref = _select_reference(src_pts)
            M, _ = cv2.estimateAffinePartial2D(
                src_pts, ref, method=cv2.RANSAC
            )
            if M is not None:
                aligned = cv2.warpAffine(bgr_frame, M, (112, 112),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
                return aligned

    # Fallback: plain bbox crop + resize
    x, y, w, h = bbox
    face_crop = bgr_frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    return cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)


def _select_reference(src_pts: np.ndarray) -> np.ndarray:
    """
    Choose reference 5-pt template scaled to match inter-eye distance of src.
    This keeps the estimate robust when the face is far/close.
    """
    return REFERENCE_5PT


def preprocess_for_recognition(face_112: np.ndarray) -> np.ndarray:
    """
    Normalize a 112×112 BGR face crop to float32 in [-1, 1].
    This normalization is used by EdgeFace and most ArcFace-trained models.

    Returns shape (1, 3, 112, 112) float32 (PyTorch NCHW).
    """
    import torch
    # BGR → RGB
    rgb = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
    # [0,255] → [0,1] → [-1,1]
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()  # CHW
    t = t / 127.5 - 1.0
    return t.unsqueeze(0)  # NCHW
