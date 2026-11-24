
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QStackedWidget,
)

from ui.camera_page import CameraPage


class HomeDashboard(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gesture Sense")
        self.resize(1250, 800)
        self.setMinimumSize(950, 650)

        self.navigation_buttons = []

        self.setup_ui()
        self.setup_navigation()

    def setup_ui(self):

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F7F7F7;
            }

            QWidget {
                font-family: Arial;
            }

            QFrame#sidebar {
                background-color: #191919;
                border: none;
            }

            QLabel#brand {
                color: white;
                font-size: 23px;
                font-weight: bold;
            }

            QLabel#brand_accent {
                color: #FF4B00;
                font-size: 23px;
                font-weight: bold;
            }

            QPushButton#nav_button {
                background-color: transparent;
                color: #AAAAAA;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding: 13px 16px;
                font-size: 14px;
            }

            QPushButton#nav_button:hover {
                background-color: #303030;
                color: white;
            }

            QPushButton#nav_button[active="true"] {
                background-color: #FF4B00;
                color: white;
                font-weight: bold;
            }

            QLabel#page_title {
                color: #191919;
                font-size: 27px;
                font-weight: bold;
            }

            QLabel#page_description {
                color: #777777;
                font-size: 13px;
            }

            QFrame#content_card {
                background-color: white;
                border: 1px solid #EEEEEE;
                border-radius: 12px;
            }

            QLabel#card_title {
                color: #333333;
                font-size: 15px;
                font-weight: bold;
            }

            QLabel#card_value {
                color: #FF4B00;
                font-size: 24px;
                font-weight: bold;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(15, 25, 15, 20)
        sidebar_layout.setSpacing(10)

        # Brand
        brand_layout = QHBoxLayout()

        brand = QLabel("GESTURE")
        brand.setObjectName("brand")

        accent = QLabel("SENSE")
        accent.setObjectName("brand_accent")

        brand_layout.addWidget(brand)
        brand_layout.addWidget(accent)
        brand_layout.addStretch()

        sidebar_layout.addLayout(brand_layout)
        sidebar_layout.addSpacing(35)

        # Navigation
        navigation_items = [
            ("Home", "home"),
            ("Camera", "camera"),
            ("Gestures", "gestures"),
            ("Effects", "effects"),
            ("Settings", "settings"),
        ]

        for label, page_id in navigation_items:

            button = QPushButton(label)
            button.setObjectName("nav_button")
            button.setProperty("active", False)

            button.clicked.connect(
                lambda checked=False, pid=page_id:
                self.navigate_to(pid)
            )

            self.navigation_buttons.append((page_id, button))
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch()

        footer = QLabel("GESTURE SENSE\nDesktop Application")
        footer.setStyleSheet("""
            color: #777777;
            font-size: 11px;
            padding: 10px;
        """)

        sidebar_layout.addWidget(footer)

        # Main content
        self.pages = QStackedWidget()

        self.home_page = self.create_home_page()
        self.camera_page = CameraPage()

        self.pages.addWidget(self.home_page)
        self.pages.addWidget(self.camera_page)

        # Placeholder pages
        self.gestures_page = self.create_placeholder(
            "Gestures",
            "Configure and manage your gesture presets."
        )

        self.effects_page = self.create_placeholder(
            "Effects",
            "Configure disappear, fade and dissolve effects."
        )

        self.settings_page = self.create_placeholder(
            "Settings",
            "Manage your Gesture Sense application settings."
        )

        self.pages.addWidget(self.gestures_page)
        self.pages.addWidget(self.effects_page)
        self.pages.addWidget(self.settings_page)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.pages, stretch=1)

        self.navigate_to("home")

    def create_home_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(35, 30, 35, 30)
        layout.setSpacing(20)

        title = QLabel("Welcome to Gesture Sense")
        title.setObjectName("page_title")

        description = QLabel(
            "Control your camera experience with natural hand gestures."
        )
        description.setObjectName("page_description")

        layout.addWidget(title)
        layout.addWidget(description)

        # Status cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)

        cards = [
            ("Camera Status", "Not Connected"),
            ("Gesture Engine", "Not Started"),
            ("Active Effect", "None"),
        ]

        for card_title, card_value in cards:

            card = QFrame()
            card.setObjectName("content_card")

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 20, 20, 20)

            heading = QLabel(card_title)
            heading.setObjectName("card_title")

            value = QLabel(card_value)
            value.setObjectName("card_value")

            card_layout.addWidget(heading)
            card_layout.addWidget(value)
            card_layout.addStretch()

            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # Getting started
        section_title = QLabel("Getting Started")
        section_title.setObjectName("page_title")

        layout.addWidget(section_title)

        info_card = QFrame()
        info_card.setObjectName("content_card")

        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(22, 22, 22, 22)
        info_layout.setSpacing(12)

        instructions = [
            "01  Connect your webcam",
            "02  Configure your gesture presets",
            "03  Select your desired visual effects",
            "04  Start Gesture Sense",
        ]

        for instruction in instructions:
            label = QLabel(instruction)
            label.setStyleSheet(
                "color: #444444; font-size: 14px; padding: 5px;"
            )
            info_layout.addWidget(label)

        layout.addWidget(info_card)
        layout.addStretch()

        return page

    def create_placeholder(self, title, description):

        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(35, 30, 35, 30)

        heading = QLabel(title)
        heading.setObjectName("page_title")

        subtitle = QLabel(description)
        subtitle.setObjectName("page_description")

        coming_soon = QLabel("Coming Soon")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        coming_soon.setStyleSheet("""
            color: #AAAAAA;
            font-size: 20px;
            font-weight: bold;
        """)

        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addWidget(coming_soon)
        layout.addStretch()

        return page

    def setup_navigation(self):
        self.navigate_to("home")

    def navigate_to(self, page_id):

        page_mapping = {
            "home": 0,
            "camera": 1,
            "gestures": 2,
            "effects": 3,
            "settings": 4,
        }

        if page_id not in page_mapping:
            return

        self.pages.setCurrentIndex(page_mapping[page_id])

        for button_id, button in self.navigation_buttons:

            is_active = button_id == page_id

            button.setProperty("active", is_active)

            button.style().unpolish(button)
            button.style().polish(button)

    def closeEvent(self, event):

        self.camera_page.timer.stop()
        self.camera_page.camera_manager.release_camera()

        event.accept()
