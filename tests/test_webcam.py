"""
tests/test_webcam.py – pytest suite for Milestone 1 (webcam).

These tests run headlessly (no display window) and are safe for CI.
They test the REAL webcam. If no camera is present, tests are skipped.
"""

import importlib.util
import sys
import time
from pathlib import Path

import cv2
import pytest

# ── allow importing from scripts/ ────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from webcam_test import open_camera, probe_frame, run_headless, BACKENDS


# ── fixtures ──────────────────────────────────────────────────────────────────

def camera_available() -> bool:
    """Quick probe: is index 0 openable?"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    cap.release()
    return ok


SKIP_NO_CAM = pytest.mark.skipif(
    not camera_available(),
    reason="No camera available at index 0 – skipping live webcam tests."
)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_opencv_importable():
    """OpenCV must be importable and report a version."""
    assert hasattr(cv2, "__version__"), "cv2.__version__ missing"
    major = int(cv2.__version__.split(".")[0])
    assert major >= 4, f"Expected OpenCV 4.x, got {cv2.__version__}"


@SKIP_NO_CAM
def test_open_camera_returns_cap_and_backend_name():
    cap, backend_name = open_camera(0)
    try:
        assert cap.isOpened(), "VideoCapture not opened"
        assert isinstance(backend_name, str) and len(backend_name) > 0
    finally:
        cap.release()


@SKIP_NO_CAM
def test_probe_frame_returns_valid_frame():
    cap, _ = open_camera(0)
    try:
        ret, frame = probe_frame(cap)
        assert ret, "probe_frame returned ret=False"
        assert frame is not None, "frame is None"
        assert frame.ndim == 3, f"Expected 3-channel frame, got ndim={frame.ndim}"
        h, w, c = frame.shape
        assert w > 0 and h > 0, f"Bad frame dimensions: {w}x{h}"
        assert c == 3, f"Expected 3 channels, got {c}"
    finally:
        cap.release()


@SKIP_NO_CAM
def test_run_headless_passes():
    cap, _ = open_camera(0)
    try:
        result = run_headless(cap)
    finally:
        cap.release()

    assert result["passed"] is True, f"run_headless failed: {result}"
    assert result["width"] > 0
    assert result["height"] > 0


@SKIP_NO_CAM
def test_camera_releases_cleanly():
    """Open and close the camera twice; second open must succeed."""
    for i in range(2):
        cap, _ = open_camera(0)
        ret, _ = probe_frame(cap)
        cap.release()
        # Small pause to let OS free the device
        time.sleep(0.1)
        assert ret, f"Iteration {i}: probe_frame failed"


@SKIP_NO_CAM
def test_headless_result_has_required_keys():
    cap, _ = open_camera(0)
    try:
        result = run_headless(cap)
    finally:
        cap.release()

    required = {"passed", "width", "height", "channels", "dtype"}
    missing = required - result.keys()
    assert not missing, f"Result missing keys: {missing}"


def test_invalid_camera_index_raises():
    """An absurd camera index should raise RuntimeError, not crash."""
    with pytest.raises(RuntimeError, match="Could not open camera"):
        open_camera(99)
