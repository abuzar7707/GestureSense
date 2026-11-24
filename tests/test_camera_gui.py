
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from ui.camera_page import CameraPage


def main():
    app = QApplication(sys.argv)

    window = CameraPage()
    window.resize(1100, 700)
    window.setWindowTitle("Gesture Sense - Camera Test")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
