
import sys

from PySide6.QtWidgets import QApplication
from ui.Home_dashboard import HomeDashboard


def main():

    app = QApplication(sys.argv)

    app.setApplicationName("Gesture Sense")
    app.setOrganizationName("IoTrix")

    window = HomeDashboard()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
