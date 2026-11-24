
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
)

from camera.camera_manager import CameraManager


class CameraPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.camera_manager = CameraManager()

        self.setup_ui()
        self.setup_timer()
        self.refresh_cameras()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #F7F7F7;
                color: #222222;
                font-family: Arial;
            }

            QLabel#page_title {
                font-size: 24px;
                font-weight: bold;
            }

            QFrame#preview_container {
                background-color: #171717;
                border-radius: 12px;
            }

            QLabel#camera_preview {
                background-color: #171717;
                color: #AAAAAA;
                font-size: 15px;
                border-radius: 12px;
            }

            QPushButton {
                background-color: #EEEEEE;
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #DDDDDD;
            }

            QPushButton#start_button {
                background-color: #FF4B00;
                color: white;
                font-weight: bold;
            }

            QPushButton#start_button:hover {
                background-color: #E84300;
            }

            QComboBox {
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 7px;
                padding: 9px;
                min-width: 150px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(18)

        # Header
        title = QLabel("Camera")
        title.setObjectName("page_title")

        description = QLabel(
            "Manage your camera and preview your live video feed."
        )

        main_layout.addWidget(title)
        main_layout.addWidget(description)

        # Camera controls
        controls_layout = QHBoxLayout()

        self.camera_selector = QComboBox()
        self.camera_selector.setPlaceholderText("Select camera")

        self.refresh_button = QPushButton("Refresh")
        self.start_button = QPushButton("Start Camera")
        self.start_button.setObjectName("start_button")
        self.stop_button = QPushButton("Stop Camera")

        self.stop_button.setEnabled(False)

        controls_layout.addWidget(self.camera_selector)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        main_layout.addLayout(controls_layout)

        # Preview container
        self.preview_container = QFrame()
        self.preview_container.setObjectName("preview_container")
        self.preview_container.setMinimumHeight(350)

        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        self.preview_label = QLabel("Camera preview is currently inactive")
        self.preview_label.setObjectName("camera_preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)

        preview_layout.addWidget(self.preview_label)

        main_layout.addWidget(self.preview_container, stretch=1)

        # Status
        self.status_label = QLabel("Status: No camera connected")
        main_layout.addWidget(self.status_label)

        # Signals
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(30)

    def refresh_cameras(self):
        was_running = self.camera_manager.is_running

        if was_running:
            self.stop_camera()

        self.camera_selector.clear()

        self.status_label.setText("Status: Searching for cameras...")

        cameras = self.camera_manager.discover_cameras()

        for camera in cameras:
            self.camera_selector.addItem(
                camera["name"],
                camera["index"]
            )

        if cameras:
            self.status_label.setText(
                f"Status: {len(cameras)} camera(s) detected"
            )
        else:
            self.status_label.setText("Status: No cameras detected")

    def start_camera(self):
        camera_index = self.camera_selector.currentData()

        if camera_index is None:
            self.status_label.setText("Status: Please select a camera")
            return

        self.status_label.setText("Status: Connecting...")

        if not self.camera_manager.connect_camera(camera_index):
            self.status_label.setText(
                f"Status: {self.camera_manager.error_message}"
            )
            return

        if not self.camera_manager.start_camera():
            self.status_label.setText(
                f"Status: {self.camera_manager.error_message}"
            )
            self.camera_manager.release_camera()
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_selector.setEnabled(False)

        self.status_label.setText(
            f"Status: Connected to Camera {camera_index}"
        )

    def stop_camera(self):
        self.camera_manager.release_camera()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_selector.setEnabled(True)

        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Camera preview is currently inactive")

        self.status_label.setText("Status: Camera stopped")

    def update_preview(self):
        if not self.camera_manager.is_running:
            return

        frame = self.camera_manager.read_frame()

        if frame is None:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame.shape
        bytes_per_line = channels * width

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.timer.stop()
        self.camera_manager.release_camera()
        event.accept()
