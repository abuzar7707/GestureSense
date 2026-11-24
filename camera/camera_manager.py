
import cv2
import threading
import time


class CameraManager:
    def __init__(self, max_cameras=6, width=1280, height=720):
        self.max_cameras = max_cameras
        self.width = width
        self.height = height

        self.camera = None
        self.camera_index = None

        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest_frame = None

        self.error_message = None

    def discover_cameras(self):
        """Detect available cameras by probing device indexes."""
        available_cameras = []

        for index in range(self.max_cameras):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            if cap.isOpened():
                success, frame = cap.read()

                if success and frame is not None:
                    available_cameras.append({
                        "index": index,
                        "name": f"Camera {index}"
                    })

            cap.release()

        return available_cameras

    def connect_camera(self, camera_index):
        """Connect to the selected camera."""
        self.stop_camera()

        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            self.error_message = (
                f"Could not open camera at index {camera_index}"
            )
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        success, frame = cap.read()

        if not success or frame is None:
            cap.release()
            self.error_message = "Camera opened but failed to capture a frame."
            return False

        self.camera = cap
        self.camera_index = camera_index
        self.error_message = None

        with self._lock:
            self._latest_frame = frame

        return True

    def start_camera(self):
        """Start continuous frame capture."""
        if self.camera is None or not self.camera.isOpened():
            self.error_message = "No camera is connected."
            return False

        if self.is_running:
            return True

        self.is_running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._thread.start()

        return True

    def _capture_loop(self):
        """Continuously capture frames in a background thread."""
        while self.is_running:
            if self.camera is None:
                break

            success, frame = self.camera.read()

            if success and frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                self.error_message = "Failed to read from camera."
                time.sleep(0.05)

    def read_frame(self):
        """Return a copy of the latest captured frame."""
        with self._lock:
            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()

    def stop_camera(self):
        """Stop the capture thread safely."""
        self.is_running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

        self._thread = None

    def release_camera(self):
        """Stop streaming and release the camera."""
        self.stop_camera()

        if self.camera is not None:
            self.camera.release()

        self.camera = None
        self.camera_index = None

        with self._lock:
            self._latest_frame = None

    def get_camera_info(self):
        """Return information about the currently connected camera."""
        if self.camera is None:
            return None

        return {
            "index": self.camera_index,
            "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.camera.get(cv2.CAP_PROP_FPS)
        }
