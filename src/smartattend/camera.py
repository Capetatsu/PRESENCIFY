"""
camera.py – Camera capture, diagnostics, and multi-backend fallback.

Provides robust camera initialization, device probing, and frame acquisition
for the SmartAttend Streamlit dashboard and CLI diagnostics.

Key capabilities:
  - Multi-backend fallback on Windows (DirectShow -> MSMF -> ANY)
  - Diagnostic hardware probe with actionable error hints
  - Transient frame read retry mechanism
  - Safe camera resource release
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


BACKENDS: List[Tuple[str, int]] = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("Default/ANY", cv2.CAP_ANY),
]


def open_camera(
    index: int = 0,
    preferred_backend: Optional[str] = None,
    warmup_frames: int = 2,
) -> Tuple[Optional[cv2.VideoCapture], str, Optional[str]]:
    """
    Attempts to open a camera device with fallback across Windows backends.

    Args:
        index: OpenCV camera device index (0, 1, 2, ...).
        preferred_backend: Preferred backend name substring (e.g. "DirectShow", "MSMF").
        warmup_frames: Number of initial frames to read to complete auto-exposure/driver handshake.

    Returns:
        (cap, backend_name, error_message)
        If successful, cap is open VideoCapture and error_message is None.
        If failed, cap is None, backend_name is empty, error_message contains details.
    """
    # Order backends placing preferred first
    order: List[Tuple[str, int]] = []
    if preferred_backend:
        for name, b in BACKENDS:
            if preferred_backend.lower() in name.lower():
                order.append((name, b))
    for item in BACKENDS:
        if item not in order:
            order.append(item)

    last_error: Optional[str] = None

    for name, backend in order:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Perform warm-up reads to verify frame buffer delivery
                ret, frame = False, None
                for _ in range(max(1, warmup_frames)):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.04)

                if ret and frame is not None:
                    return cap, name, None
                else:
                    last_error = (
                        f"Device index {index} opened with backend '{name}', "
                        "but initial frame read failed (blank or dropped frame)."
                    )
                    cap.release()
            else:
                last_error = f"Device index {index} failed to open with backend '{name}'."
                cap.release()
        except Exception as exc:
            last_error = f"Exception opening index {index} with backend '{name}': {exc}"

    return None, "", last_error or f"Could not access camera device index {index} on any backend."


def diagnose_camera(index: int = 0) -> Dict[str, Any]:
    """
    Runs a non-destructive diagnostic check across all available camera backends.

    Returns a structured dictionary with availability, resolution, latency,
    and troubleshooting recommendations.
    """
    backend_results: Dict[str, Any] = {}
    is_accessible = False
    recommended_backend: Optional[str] = None

    for name, backend in BACKENDS:
        t0 = time.perf_counter()
        try:
            cap = cv2.VideoCapture(index, backend)
            opened = cap.isOpened()
            ret, frame = False, None
            if opened:
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.04)
                cap.release()

            dt_ms = (time.perf_counter() - t0) * 1000
            shape = list(frame.shape) if (ret and frame is not None) else None
            backend_results[name] = {
                "opened": opened,
                "frame_read": bool(ret),
                "resolution": f"{shape[1]}x{shape[0]}" if shape else None,
                "channels": shape[2] if shape else None,
                "init_latency_ms": round(dt_ms, 1),
            }
            if opened and ret:
                is_accessible = True
                if recommended_backend is None:
                    recommended_backend = name
        except Exception as exc:
            backend_results[name] = {
                "opened": False,
                "frame_read": False,
                "error": str(exc),
            }

    hints: List[str] = []
    if not is_accessible:
        hints.append(
            "Check if another process (e.g. Zoom, Microsoft Teams, browser tabs) is currently using the camera."
        )
        hints.append(
            "Verify Windows Settings: Privacy & security -> Camera -> 'Let desktop apps access your camera' must be ON."
        )
        hints.append(
            f"If an external webcam is attached, try changing the device index from {index} to 1 or 2."
        )
        hints.append(
            "Check physical webcam privacy shutter / switch on the laptop bezel."
        )
    else:
        hints.append(f"Camera index {index} is functional and ready for live capture using backend '{recommended_backend}'.")

    return {
        "device_index": index,
        "is_accessible": is_accessible,
        "recommended_backend": recommended_backend,
        "backends": backend_results,
        "hints": hints,
    }


def read_frame_with_retry(
    cap: cv2.VideoCapture,
    max_retries: int = 3,
    delay_sec: float = 0.02,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Attempts to read a frame from an open VideoCapture with brief retry tolerance.
    Prevents single dropped frames from terminating live sessions prematurely.
    """
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None:
            return True, frame
        if attempt < max_retries - 1:
            time.sleep(delay_sec)
    return False, None


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """Safely releases the camera device handle."""
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def format_similarity(
    score: Any,
    decimals: int = 3,
    default: str = "N/A",
) -> str:
    """
    Safely format a cosine similarity score.
    Returns 'default' (e.g. 'N/A') on None, NaN, inf, or unparseable input.
    Correctly displays 0.000 without treating 0.0 as False.
    """
    if score is None:
        return default
    try:
        f = float(score)
        if np.isnan(f) or np.isinf(f):
            return default
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return default
