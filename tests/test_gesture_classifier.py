
import sys
from pathlib import Path
from unittest import result

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from camera.camera_manager import CameraManager
from detection.hand_detector import HandDetector
from detection.gesture_classifier import GestureClassifier


def main():

    camera = CameraManager()
    detector = None

    classifier = GestureClassifier()

    try:
        print("Discovering cameras...")

        cameras = camera.discover_cameras()

        if not cameras:
            print("No cameras detected.")
            return

        for item in cameras:
            print(item)

        camera_index = int(input("\nSelect camera index: "))

        if not camera.connect_camera(camera_index):
            print("Camera error:", camera.error_message)
            return

        camera.start_camera()

        detector = HandDetector(max_hands=2)

        print("\nGesture Classifier initialized.")
        print("Try these gestures:")
        print("Thumbs Up | Peace | Open Palm | Rock Sign")
        print("Press Q to quit.")

        while True:

            frame = camera.read_frame()

            if frame is None:
                continue

            detection = detector.detect(frame)

            classification = classifier.classify(
                detection
            )

            output = detector.draw_landmarks(
                frame,
                detection
            )

            for index, result in enumerate(
                classification["gestures"]
            ):

                gesture = result["gesture"]
                confidence = result["confidence"]
                handedness = result["handedness"]

                text = (
                    f"{handedness}: {gesture} "
                    f"({confidence:.2f})"
                )

                cv2.putText(
                    output,
                    text,
                    (20, 75 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 140, 255),
                    2
                )

                states = result["finger_states"]
                margins = result["finger_margins"]

                print(
                    f"\n{text}"
                    f"\n  States: {states}"
                    f"\n  Margins: {margins}"
                )

            cv2.imshow(
                "Gesture Sense - Gesture Classifier",
                output
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as error:
        print("\nERROR:", error)

    finally:
        if detector is not None:
            detector.close()

        camera.release_camera()
        cv2.destroyAllWindows()

        print("\nResources released.")


if __name__ == "__main__":
    main()
