"""
Gesture Sense - Webcam capability benchmark
Tests webcam backend / resolution / pixel-format combinations.

Run from the project root:
    python benchmark_camera_modes.py

Optional:
    python benchmark_camera_modes.py --camera 0 --seconds 6
    python benchmark_camera_modes.py --camera 1 --seconds 8

Notes:
- Close other apps using the webcam before running.
- Each configuration is opened and tested sequentially.
- A camera/driver may ignore requested resolution or FOURCC. The script
  reports the actual values returned by the driver and measured FPS.
"""

import argparse
import time
import cv2
import numpy as np


BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]

MODES = [
    (640, 480, "MJPG"),
    (640, 480, "YUY2"),
    (1280, 720, "MJPG"),
    (1280, 720, "YUY2"),
    (1920, 1080, "MJPG"),
    (1920, 1080, "YUY2"),
]


def fourcc_text(value):
    try:
        value = int(value)
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "????"


def test_mode(camera_index, backend_name, backend, width, height, pixel_format, seconds):
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        return {"status": "OPEN FAILED"}

    # Set FOURCC first, then dimensions, then request a frame rate.
    requested_fourcc = cv2.VideoWriter_fourcc(*pixel_format)
    cap.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Let the driver settle and discard initial frames.
    for _ in range(8):
        cap.read()

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))

    read_times = []
    frame_count = 0
    start = time.perf_counter()
    deadline = start + seconds

    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        read_times.append((t1 - t0) * 1000.0)
        if ok and frame is not None:
            frame_count += 1

    elapsed = time.perf_counter() - start
    cap.release()

    if not read_times or frame_count == 0:
        return {
            "status": "NO FRAMES",
            "actual": f"{actual_width}x{actual_height}",
            "reported_fps": reported_fps,
            "fourcc": actual_fourcc,
        }

    arr = np.asarray(read_times, dtype=np.float64)
    return {
        "status": "OK",
        "actual": f"{actual_width}x{actual_height}",
        "reported_fps": reported_fps,
        "fourcc": actual_fourcc,
        "measured_fps": frame_count / elapsed,
        "read_mean": float(np.mean(arr)),
        "read_p95": float(np.percentile(arr, 95)),
        "frames": frame_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--seconds", type=float, default=6.0, help="Seconds per configuration (default: 6)")
    args = parser.parse_args()

    print("\nGESTURE SENSE - WEBCAM MODE BENCHMARK")
    print("=" * 92)
    print(f"OpenCV: {cv2.__version__} | Camera index: {args.camera} | Test duration: {args.seconds:.1f}s")
    print("Testing sequentially. Close Zoom, Teams, browser camera tabs, etc.\n")

    results = []
    for backend_name, backend in BACKENDS:
        for width, height, pixel_format in MODES:
            print(f"Testing {backend_name:5s} | {width:4d}x{height:<4d} | {pixel_format} ...", end=" ", flush=True)
            result = test_mode(
                args.camera, backend_name, backend,
                width, height, pixel_format, args.seconds
            )
            result.update({
                "backend": backend_name,
                "requested": f"{width}x{height}",
                "requested_fourcc": pixel_format,
            })
            results.append(result)

            if result["status"] == "OK":
                print(
                    f"{result['actual']:>9s} | driver {result['reported_fps']:5.1f} FPS "
                    f"| measured {result['measured_fps']:5.2f} FPS "
                    f"| FOURCC {result['fourcc']} "
                    f"| read mean {result['read_mean']:6.2f} ms "
                    f"| p95 {result['read_p95']:6.2f} ms"
                )
            else:
                print(result["status"])

    print("\n" + "=" * 92)
    print("SUMMARY (successful modes, sorted by measured FPS)")
    successful = [r for r in results if r["status"] == "OK"]
    successful.sort(key=lambda r: r["measured_fps"], reverse=True)

    if not successful:
        print("No successful camera configurations.")
    else:
        print(f"{'Backend':<8} {'Requested':<12} {'Actual':<12} {'Format':<8} {'FPS':>8} {'Read p95':>12}")
        for r in successful:
            print(
                f"{r['backend']:<8} {r['requested']:<12} {r['actual']:<12} "
                f"{r['fourcc']:<8} {r['measured_fps']:>7.2f} "
                f"{r['read_p95']:>9.2f} ms"
            )

    print("\nUse the measured FPS and actual resolution, not the driver's reported FPS.")
    print("For Gesture Sense, prioritize a stable 1280x720 at ~30 FPS; otherwise choose")
    print("the highest stable measured FPS at 640x480 or 720p.")
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
