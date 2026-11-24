import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from camera.camera_manager import CameraManager


def main():
    manager = CameraManager()

    print("Searching for cameras...")
    cameras = manager.discover_cameras()

    if not cameras:
        print("No cameras detected.")
        return

    print("\nAvailable cameras:")
    for camera in cameras:
        print(f"Index: {camera['index']} | {camera['name']}")

    try:
        selected_index = int(input("\nEnter camera index: "))
    except ValueError:
        print("Invalid camera index.")
        return

    print("\nConnecting camera...")

    if not manager.connect_camera(selected_index):
        print("Connection failed:", manager.error_message)
        return

    print("Camera connected:", manager.get_camera_info())

    if not manager.start_camera():
        print("Could not start camera:", manager.error_message)
        manager.release_camera()
        return

    print("\nCamera preview started.")
    print("Press Q to quit.")

    try:
        while True:
            frame = manager.read_frame()

            if frame is None:
                continue

            cv2.imshow("Gesture Sense - Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        manager.release_camera()
        cv2.destroyAllWindows()
        print("Camera released successfully.")


if __name__ == "__main__":
    main()
