"""
webcam_test.py – PRESENCIFY Milestone 1: Webcam smoke test.

Tests that:
  1. A webcam can be opened on Windows (DirectShow → MSMF → any).
  2. At least one frame can be read.
  3. Resources are released cleanly.

Usage:
  python scripts/webcam_test.py              # opens live window (press q to quit)
  python scripts/webcam_test.py --headless   # no window; just checks one frame
  python scripts/webcam_test.py --save-frame # saves one frame to experiments/

This script NEVER fakes results. If the camera fails it says so and exits non-zero.
"""

import argparse
import sys
import time
import cv2


BACKENDS = [
    (cv2.CAP_DSHOW,  "DirectShow"),
    (cv2.CAP_MSMF,   "MSMF"),
    (cv2.CAP_ANY,    "ANY"),
]


def open_camera(index: int = 0) -> tuple[cv2.VideoCapture, str]:
    """Try Windows backends in order; return the first that opens."""
    for backend, name in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"[camera] opened with backend={name} index={index}")
            return cap, name
        cap.release()
    raise RuntimeError(
        f"Could not open camera index {index} on any backend "
        f"({[n for _, n in BACKENDS]}). "
        "Check that no other app is using the webcam."
    )


def probe_frame(cap: cv2.VideoCapture) -> tuple[bool, object]:
    """Read one frame; return (success, frame)."""
    # Warm-up: some backends return a black frame on the first read.
    for _ in range(3):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.05)
    return ret, frame


def run_headless(cap: cv2.VideoCapture) -> dict:
    """Read one frame and report; no window."""
    ret, frame = probe_frame(cap)
    if not ret or frame is None:
        return {"passed": False, "error": "cap.read() returned False after warm-up"}
    h, w = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim == 3 else 1
    return {
        "passed": True,
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(frame.dtype),
    }


def run_live(cap: cv2.VideoCapture, save_frame: bool = False) -> dict:
    """Display live feed; press q or ESC to quit."""
    result = {"passed": False}
    frame_count = 0
    save_path = None

    print("[camera] Live view started. Press  q  or  ESC  to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[camera] WARNING: cap.read() returned False – skipping frame")
            continue

        frame_count += 1
        if frame_count == 1:
            h, w = frame.shape[:2]
            result.update({"passed": True, "width": w, "height": h})
            print(f"[camera] First frame: {w}x{h}  dtype={frame.dtype}")

            if save_frame:
                import os
                os.makedirs("experiments", exist_ok=True)
                save_path = "experiments/webcam_test_frame.png"
                cv2.imwrite(save_path, frame)
                print(f"[camera] Frame saved to {save_path}")

        # Overlay
        label = f"PRESENCIFY – webcam test  |  frame {frame_count}  |  press q to quit"
        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)

        cv2.imshow("PRESENCIFY – Webcam Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    result["frames_shown"] = frame_count
    if save_path:
        result["saved_frame"] = save_path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRESENCIFY webcam smoke test (Milestone 1)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No display window; just verify one frame can be read."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Camera device index (default 0)."
    )
    parser.add_argument(
        "--save-frame", action="store_true",
        help="Save first frame to experiments/webcam_test_frame.png."
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Run comprehensive diagnostic scan across all available backends."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PRESENCIFY - Milestone 1: Webcam Test")
    print(f"OpenCV version : {cv2.__version__}")
    print(f"Camera index   : {args.camera_index}")
    print(f"Mode           : {'diagnose' if args.diagnose else ('headless' if args.headless else 'live')}")
    print("=" * 60)

    if args.diagnose:
        from pathlib import Path
        src_path = Path(__file__).resolve().parent.parent / "src"
        sys.path.insert(0, str(src_path))
        from presencify.camera import diagnose_camera
        diag = diagnose_camera(args.camera_index)
        import json
        print(json.dumps(diag, indent=2))
        return 0 if diag["is_accessible"] else 1

    try:
        cap, backend = open_camera(args.camera_index)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    try:
        if args.headless:
            result = run_headless(cap)
        else:
            result = run_live(cap, save_frame=args.save_frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    if result.get("passed"):
        print("RESULT : PASS")
        for k, v in result.items():
            if k != "passed":
                print(f"  {k}: {v}")
    else:
        print("RESULT : FAIL")
        print(f"  error: {result.get('error', 'unknown')}")
    print("=" * 60)

    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
