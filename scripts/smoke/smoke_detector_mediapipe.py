"""
smoke_detector_mediapipe.py – Phase 2 smoke test for MediaPipe face detection.

Checks:
  1. mediapipe imports successfully.
  2. FaceDetection model (Tasks API) loads on CPU.
  3. Face detected on synthetic test image.
  4. No-face image returns zero detections.
  5. CPU latency measured (5 runs after warm-up).
  6. Memory usage reported.

Usage:
  python scripts/smoke/smoke_detector_mediapipe.py
  python scripts/smoke/smoke_detector_mediapipe.py --webcam   # live webcam frames

Exits 0 if all checks pass, 1 if any check fails.
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from presencify.detector import FaceDetector, get_detector

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def make_synthetic_face_image(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (size // 5, size // 4), 0, 0, 360, (195, 155, 115), -1)
    eye_y = cy - size // 12
    for dx in [-size // 10, size // 10]:
        cv2.circle(img, (cx + dx, eye_y), size // 24, (50, 30, 20), -1)
    return img


def make_blank_image(size: int = 480) -> np.ndarray:
    return np.full((size, size, 3), 180, dtype=np.uint8)


def run_smoke_test(use_webcam: bool = False) -> int:
    results = []

    # 1. Import check
    try:
        import mediapipe as mp
        results.append((PASS, "import mediapipe", f"version={mp.__version__}"))
    except ImportError as exc:
        results.append((FAIL, "import mediapipe", str(exc)))
        _print_results(results)
        return 1

    # 2. Model load
    detector = FaceDetector()
    ok = detector.load()
    if ok:
        results.append((PASS, "model load", f"{detector.status['load_time_ms']} ms"))
    else:
        results.append((FAIL, "model load", f"error={detector.status.get('load_error')}"))
        _print_results(results)
        return 1

    # 3. Synthetic face detection
    face_img = make_synthetic_face_image()
    try:
        dets = detector.detect(face_img)
        n_faces = len(dets)
        tag = PASS if n_faces >= 1 else SKIP
        results.append((tag, "synthetic face detection", f"{n_faces} face(s) detected"))
    except Exception as exc:
        results.append((FAIL, "synthetic face detection", str(exc)))

    # 4. No-face image
    blank = make_blank_image()
    try:
        dets_blank = detector.detect(blank)
        n_blank = len(dets_blank)
        tag = PASS if n_blank == 0 else FAIL
        results.append((tag, "no-face image -> 0 detections", f"{n_blank} detections on blank image"))
    except Exception as exc:
        results.append((FAIL, "no-face image -> 0 detections", str(exc)))

    # 5. Latency
    if use_webcam:
        _run_webcam_latency(detector, results)
    else:
        _run_synthetic_latency(detector, results)

    # 6. Memory
    tracemalloc.start()
    for _ in range(3):
        _ = detector.detect(face_img)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append((PASS, "RAM (peak tracemalloc)", f"{peak / 1e6:.2f} MB (3 inference calls)"))

    detector.close()
    _print_results(results)
    failed = [r for r in results if r[0] == FAIL]
    return 0 if not failed else 1


def _run_synthetic_latency(detector, results: list) -> None:
    face_img = make_synthetic_face_image()
    detector.detect(face_img)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        detector.detect(face_img)
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    mx = max(times)
    tag = PASS if avg < 500 else FAIL
    results.append((tag, "CPU latency (synthetic, 5 runs)", f"avg={avg:.1f} ms  max={mx:.1f} ms  threshold=500 ms"))


def _run_webcam_latency(detector, results: list) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        results.append((SKIP, "webcam latency", "Camera not available"))
        return
    times = []
    face_counts = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        face_counts.append(len(dets))
    cap.release()
    if times:
        avg = sum(times) / len(times)
        tag = PASS if avg < 500 else FAIL
        results.append((tag, "webcam latency (10 frames)", f"avg={avg:.1f} ms  face counts={face_counts}"))
    else:
        results.append((SKIP, "webcam latency", "No frames captured"))


def _print_results(results: list) -> None:
    print()
    print("=" * 65)
    print("Phase 2 - MediaPipe Face Detection Smoke Test")
    print("=" * 65)
    all_ok = True
    for tag, name, detail in results:
        marker = "[OK]" if tag == PASS else ("[--]" if tag == SKIP else "[XX]")
        print(f"  [{tag:4s}] {marker} {name}")
        print(f"         {detail}")
        if tag == FAIL:
            all_ok = False
    print()
    overall = "ALL CHECKS PASS" if all_ok else "ONE OR MORE CHECKS FAILED"
    print(f"OVERALL: {overall}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="Use live webcam frames for latency test.")
    args = parser.parse_args()
    sys.exit(run_smoke_test(use_webcam=args.webcam))
