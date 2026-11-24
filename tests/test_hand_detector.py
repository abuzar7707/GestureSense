
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from camera.camera_manager import CameraManager
from detection.hand_detector import HandDetector


def main():

    camera = CameraManager()
    detector = None

    try:
        print("Discovering cameras...")

        cameras = camera.discover_cameras()

        if not cameras:
            print("No cameras found.")
            return

        print("\nAvailable cameras:")

        for item in cameras:
            print(item)

        camera_index = int(input("\nSelect camera index: "))

        if not camera.connect_camera(camera_index):
            print("Camera error:", camera.error_message)
            return

        if not camera.start_camera():
            print("Could not start camera.")
            return

        print("\nInitializing hand detector...")

        detector = HandDetector(max_hands=2)

        print("Hand detector initialized.")
        print("Show your hands to the camera.")
        print("Press Q to quit.")

        while True:

            frame = camera.read_frame()

            if frame is None:
                continue

            result = detector.detect(frame)

            output = detector.draw_landmarks(
                frame,
                result
            )

            cv2.putText(
                output,
                f"Hands Detected: {result['count']}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2
            )

            cv2.imshow(
                "Gesture Sense - Hand Detection",
                output
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print("ERROR:", error)

    finally:
        if detector is not None:
            detector.close()

        camera.release_camera()
        cv2.destroyAllWindows()

        print("Resources released.")


if __name__ == "__main__":
    main()
