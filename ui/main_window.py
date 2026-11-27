
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QStackedWidget,
)

from ui.anonymization_page import AnonymizationPage


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("视频隐私保护与异常检测系统")
        self.resize(1440, 960)
        self.setMinimumSize(1350,900)

        self.init_ui()
        self.apply_styles()

    def init_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航区域
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 25, 18, 20)
        sidebar_layout.setSpacing(15)

        app_title = QLabel("视频隐私保护")
        app_title.setObjectName("sidebarTitle")

        app_subtitle = QLabel("VIDEO PRIVACY SYSTEM")
        app_subtitle.setObjectName("sidebarSubtitle")

        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(app_subtitle)
        sidebar_layout.addSpacing(25)

        navigation_title = QLabel("功能模块")
        navigation_title.setObjectName("sidebarSection")

        sidebar_layout.addWidget(navigation_title)

        self.anonymization_button = QPushButton("  视频匿名化")
        self.anonymization_button.setObjectName("navButton")
        self.anonymization_button.setCheckable(True)
        self.anonymization_button.setChecked(True)

        self.federated_button = QPushButton(
            "  联邦学习 VAD（后续）"
        )
        self.federated_button.setObjectName("navButton")
        self.federated_button.setEnabled(False)

        sidebar_layout.addWidget(self.anonymization_button)
        sidebar_layout.addWidget(self.federated_button)

        sidebar_layout.addStretch()

        version_label = QLabel("毕业设计系统")
        version_label.setObjectName("sidebarSubtitle")

        sidebar_layout.addWidget(version_label)

        # 右侧工作区域
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)

        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 顶部标题栏
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(68)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(25, 0, 25, 0)

        header_title = QLabel(
            "视频隐私保护与异常检测系统"
        )
        header_title.setObjectName("headerTitle")

        header_layout.addWidget(header_title)
        header_layout.addStretch()

        desktop_tag = QLabel("本地桌面端")
        desktop_tag.setObjectName("tagLabel")

        header_layout.addWidget(desktop_tag)

        # 页面切换容器
        self.page_stack = QStackedWidget()

        self.anonymization_page = AnonymizationPage()
        self.page_stack.addWidget(self.anonymization_page)

        right_layout.addWidget(header)
        right_layout.addWidget(self.page_stack)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(right_area)

    def apply_styles(self):

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F4F6F8;
            }

            #sidebar {
                background-color: #263445;
            }

            #sidebarTitle {
                color: white;
                font-size: 22px;
                font-weight: bold;
            }

            #sidebarSubtitle {
                color: #AAB5C2;
                font-size: 10px;
            }

            #sidebarSection {
                color: #AAB5C2;
                font-size: 13px;
            }

            #navButton {
                background-color: transparent;
                color: #E1E7ED;
                border: none;
                border-radius: 5px;
                padding: 13px 10px;
                text-align: left;
                font-size: 14px;
            }

            #navButton:checked {
                background-color: #329B5D;
                color: white;
                font-weight: bold;
            }

            #navButton:disabled {
                color: #768493;
            }

            #header {
                background-color: white;
                border-bottom: 1px solid #E1E5EA;
            }

            #headerTitle {
                color: #253345;
                font-size: 17px;
                font-weight: bold;
            }

            #tagLabel {
                background-color: #E9F5ED;
                color: #27864C;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }

            #pageTitle {
                font-size: 23px;
                font-weight: bold;
                color: #263445;
            }

            #pageDescription {
                font-size: 13px;
                color: #7B8794;
            }

            #card {
                background-color: white;
                border: 1px solid #E3E7EB;
                border-radius: 7px;
            }

            #videoPathLabel {
                color: #687584;
                font-size: 13px;
            }

            #primaryButton {
                background-color: #329B5D;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }

            #primaryButton:hover {
                background-color: #27864C;
            }

            #sectionTitle {
                color: #344256;
                font-size: 15px;
                font-weight: bold;
            }

            #previewArea {
                background-color: #F0F2F5;
                border: 1px dashed #C9D0D8;
                border-radius: 5px;
                color: #929EAA;
                font-size: 13px;
            }

            #secondaryText {
                color: #7B8794;
                font-size: 12px;
            }

            #chartPlaceholder {
                background-color: #F8F9FB;
                color: #929EAA;
                border-radius: 5px;
                font-size: 13px;
            }

            #statusText {
                color: #778491;
                font-size: 12px;
            }

            QSlider::groove:horizontal {
                height: 5px;
                background: #E0E5E9;
                border-radius: 2px;
            }

            QSlider::handle:horizontal {
                background: #329B5D;
                width: 13px;
                margin: -5px 0;
                border-radius: 6px;
            }
        """)
