
import os
import traceback

import cv2

from PySide6.QtCore import (
    QThread,
    Signal,
    Qt,
    QUrl,
    QTimer
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QLabel,
    QPushButton,
    QSlider,
    QFileDialog,
    QProgressBar,
    QSizePolicy,
    QStackedLayout,
    QMessageBox
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget

    QT_MULTIMEDIA_AVAILABLE = True

except (ImportError, OSError):
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None

    QT_MULTIMEDIA_AVAILABLE = False


# ==================================================
# 1. 完整视频处理后台线程
# ==================================================

class VideoPipelineWorker(QThread):

    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)

    pipeline_succeeded = Signal(object)
    pipeline_failed = Signal(str)

    def __init__(self, video_path):
        super().__init__()

        self.video_path = video_path

    def run(self):

        try:
            # 在后台线程中导入统一流水线
            from video_pipeline import run_full_pipeline

            result = run_full_pipeline(
                source_video_path=self.video_path,

                progress_callback=lambda value:
                    self.progress_changed.emit(value),

                status_callback=lambda message:
                    self.status_changed.emit(message),

                log_callback=lambda message:
                    self.log_message.emit(message)
            )

            self.pipeline_succeeded.emit(result)

        except Exception:
            error_message = traceback.format_exc()
            self.pipeline_failed.emit(error_message)


# ==================================================
# 2. 视频匿名化结果页面
# ==================================================

class AnonymizationPage(QWidget):

    def __init__(self):
        super().__init__()

        self.selected_video_path = None

        self.frame_dir = None
        self.frame_paths = []
        self.anonymized_frame_paths = []

        self.current_frame = None

        self.worker = None
        self.pipeline_result = None

        self.media_player = None
        self.audio_output = None
        self.video_widget = None
        self.output_stack = None

        self.active_output_video_path = None

        self.source_fps = 0.0
        self.source_frame_count = 0

        self.pending_frame_seek = None

        self.curve_pixmap = None

        self.metric_widgets = {}

        self.init_ui()

    # ==================================================
    # 3. 项目路径
    # ==================================================

    def get_project_root(self):
        return os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

    # ==================================================
    # 4. 初始化页面布局
    # ==================================================

    def init_ui(self):

        main = QVBoxLayout(self)

        main.setContentsMargins(18, 12, 18, 12)
        main.setSpacing(7)

        # --------------------------------------------------
        # 页面标题
        # --------------------------------------------------

        title = QLabel("视频匿名化与异常检测")

        title.setStyleSheet(
            "font-size:22px;"
            "font-weight:bold;"
            "color:#243b5a;"
        )

        desc = QLabel(
            "上传监控视频，自动执行隐私匿名化、异常响应分析及效果评价。"
        )

        desc.setStyleSheet(
            "color:#718096;"
            "font-size:12px;"
        )

        main.addWidget(title)
        main.addWidget(desc)

        # --------------------------------------------------
        # 视频选择区域
        # --------------------------------------------------

        upload = QFrame()
        upload.setObjectName("uploadCard")

        upload_layout = QHBoxLayout(upload)

        upload_layout.setContentsMargins(14, 7, 14, 7)

        self.video_label = QLabel("尚未选择视频")

        self.video_label.setObjectName("videoPathLabel")

        self.select_button = QPushButton("选择视频")

        self.select_button.setObjectName("primaryButton")
        self.select_button.setFixedSize(120, 36)

        self.select_button.clicked.connect(
            self.select_video
        )

        upload_layout.addWidget(
            self.video_label,
            1
        )

        upload_layout.addWidget(
            self.select_button
        )

        main.addWidget(upload)

        # --------------------------------------------------
        # 总体处理进度
        # --------------------------------------------------

        self.extraction_progress = QProgressBar()

        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_progress.setFormat("%p%")
        self.extraction_progress.setTextVisible(True)
        self.extraction_progress.setFixedHeight(17)

        self.extraction_progress.hide()

        main.addWidget(self.extraction_progress)

        # --------------------------------------------------
        # 原始视频与匿名化视频预览
        # --------------------------------------------------

        previews = QHBoxLayout()
        previews.setSpacing(12)

        self.original_panel = self.create_preview_panel(
            "原始视频帧"
        )

        self.original_image_area = (
            self.original_panel.image_area
        )

        self.anonymized_panel = (
            self.create_video_player_panel()
        )

        previews.addWidget(
            self.original_panel,
            1
        )

        previews.addWidget(
            self.anonymized_panel,
            1
        )

        main.addLayout(
            previews,
            4
        )

        # --------------------------------------------------
        # 视频同步进度与播放控制
        # --------------------------------------------------

        progress_panel = QFrame()
        progress_panel.setObjectName("contentCard")

        progress_layout = QVBoxLayout(progress_panel)

        progress_layout.setContentsMargins(
            12, 6, 12, 6
        )

        progress_layout.setSpacing(4)

        progress_header = QHBoxLayout()

        progress_title = QLabel("视频同步进度")

        progress_title.setObjectName("sectionTitle")

        self.frame_counter = QLabel("0 / 0 帧")

        self.frame_counter.setObjectName("secondaryText")

        progress_header.addWidget(progress_title)
        progress_header.addStretch()
        progress_header.addWidget(self.frame_counter)

        progress_layout.addLayout(progress_header)

        controls = QHBoxLayout()

        self.play_button = QPushButton("播放")

        self.play_button.setFixedSize(76, 30)
        self.play_button.setEnabled(False)

        self.play_button.clicked.connect(
            self.toggle_video_playback
        )

        self.frame_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)

        self.frame_slider.valueChanged.connect(
            self.on_frame_changed
        )

        controls.addWidget(self.play_button)
        controls.addWidget(self.frame_slider, 1)

        progress_layout.addLayout(controls)

        main.addWidget(progress_panel)

        # --------------------------------------------------
        # 评价指标区域
        # --------------------------------------------------

        metrics_panel = QFrame()
        metrics_panel.setObjectName("contentCard")

        metrics_layout = QVBoxLayout(metrics_panel)

        metrics_layout.setContentsMargins(
            12, 7, 12, 7
        )

        metrics_layout.setSpacing(6)

        metrics_title = QLabel("匿名化效果评价")

        metrics_title.setObjectName("sectionTitle")

        metrics_layout.addWidget(metrics_title)

        metric_grid = QGridLayout()

        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(6)

        metric_definitions = [
            # 第一行
            ("psnr", "PSNR 均值"),
            ("original_faces", "原始视频人脸检测框"),
            ("original_rate", "原始视频人脸检出率"),

            # 第二行
            ("ssim", "SSIM 均值"),
            ("anonymized_faces", "匿名化视频人脸检测框"),
            ("anonymized_rate", "匿名化视频人脸检出率")
        ]

        for index, (key, label_text) in enumerate(
            metric_definitions
        ):

            card, value_label, detail_label = (
                self.create_metric_card(label_text)
            )

            row = index // 3
            column = index % 3

            metric_grid.addWidget(
                card,
                row,
                column
            )

            self.metric_widgets[key] = {
                "value": value_label,
                "detail": detail_label
            }

        metrics_layout.addLayout(metric_grid)

        main.addWidget(metrics_panel)

        # --------------------------------------------------
        # FT 异常响应曲线
        # --------------------------------------------------

        chart_panel = QFrame()
        chart_panel.setObjectName("contentCard")

        chart_layout = QVBoxLayout(chart_panel)

        chart_layout.setContentsMargins(
            12, 7, 12, 7
        )

        chart_layout.setSpacing(5)

        chart_title = QLabel("视频异常响应曲线")

        chart_title.setObjectName("sectionTitle")

        chart_layout.addWidget(chart_title)

        self.chart_placeholder = QLabel(
            "完成视频处理后，将在这里显示 FT 异常响应曲线。"
        )

        self.chart_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.chart_placeholder.setMinimumHeight(105)
        self.chart_placeholder.setMaximumHeight(155)

        self.chart_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.chart_placeholder.setStyleSheet(
            "background:#f7f8fa;"
            "color:#8a9bb0;"
            "border-radius:5px;"
            "font-size:12px;"
        )

        chart_layout.addWidget(
            self.chart_placeholder
        )

        main.addWidget(
            chart_panel,
            2
        )

        # --------------------------------------------------
        # 状态栏
        # --------------------------------------------------

        self.status_label = QLabel(
            "系统就绪，等待选择视频。"
        )

        self.status_label.setObjectName("statusText")

        self.status_label.setWordWrap(True)

        main.addWidget(self.status_label)

        # --------------------------------------------------
        # 页面样式
        # --------------------------------------------------

        self.setStyleSheet("""
            QFrame#uploadCard,
            QFrame#contentCard,
            QFrame#metricCard {
                background:white;
                border:1px solid #dce3ea;
                border-radius:7px;
            }

            QPushButton#primaryButton,
            QPushButton {
                background:#2f9e62;
                color:white;
                border:0;
                border-radius:5px;
                padding:5px 10px;
            }

            QPushButton:hover {
                background:#278653;
            }

            QPushButton:disabled {
                background:#aab5bd;
            }

            QLabel#videoPathLabel {
                color:#687584;
                font-size:12px;
            }

            QLabel#sectionTitle {
                color:#344256;
                font-size:13px;
                font-weight:bold;
            }

            QLabel#secondaryText,
            QLabel#statusText {
                color:#7b8794;
                font-size:11px;
            }

            QSlider::groove:horizontal {
                height:5px;
                background:#dce3e8;
                border-radius:2px;
            }

            QSlider::handle:horizontal {
                background:#2f9e62;
                width:14px;
                height:14px;
                margin:-5px 0;
                border-radius:7px;
            }

            QProgressBar {
                background:#e9edf0;
                border:0;
                border-radius:3px;
                color:#526170;
                text-align:center;
                font-size:10px;
            }

            QProgressBar::chunk {
                background:#2f9e62;
                border-radius:3px;
            }
        """)

        self.reset_metrics()

    # ==================================================
    # 5. 创建视频预览区域
    # ==================================================

    def prepare_image_label(self, label, background="#111827"):

        label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        label.setMinimumSize(0, 0)

        label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        label.setScaledContents(False)

        label.setStyleSheet(
            f"background:{background};"
            "color:white;"
            "border-radius:5px;"
        )

    def create_preview_panel(self, panel_title):

        panel = QFrame()
        panel.setObjectName("contentCard")

        panel.setMinimumHeight(175)

        panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            10, 7, 10, 7
        )

        layout.setSpacing(5)

        heading = QLabel(panel_title)

        heading.setObjectName("sectionTitle")

        image = QLabel("等待视频处理")

        self.prepare_image_label(
            image,
            "#f0f2f5"
        )

        image.setStyleSheet(
            "background:#f0f2f5;"
            "color:#8a9bb0;"
            "border:1px dashed #cbd5e1;"
            "border-radius:5px;"
        )

        layout.addWidget(heading)
        layout.addWidget(image, 1)

        panel.image_area = image

        return panel

    def create_video_player_panel(self):

        panel = QFrame()
        panel.setObjectName("contentCard")

        panel.setMinimumHeight(175)

        panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(panel)

        layout.setContentsMargins(
            10, 7, 10, 7
        )

        layout.setSpacing(5)

        heading = QLabel("匿名化结果视频")

        heading.setObjectName("sectionTitle")

        layout.addWidget(heading)

        self.output_preview_label = QLabel(
            "等待匿名化视频"
        )

        self.prepare_image_label(
            self.output_preview_label
        )

        if QT_MULTIMEDIA_AVAILABLE:

            self.video_widget = QVideoWidget()

            self.video_widget.setMinimumSize(0, 0)

            self.video_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )

            self.video_widget.setStyleSheet(
                "background:#111827;"
            )

            self.output_stack = QStackedLayout()

            self.output_stack.setContentsMargins(
                0, 0, 0, 0
            )

            self.output_stack.addWidget(
                self.video_widget
            )

            self.output_stack.addWidget(
                self.output_preview_label
            )

            self.output_stack.setCurrentWidget(
                self.output_preview_label
            )

            layout.addLayout(
                self.output_stack,
                1
            )

            self.audio_output = QAudioOutput(self)

            self.media_player = QMediaPlayer(self)

            self.media_player.setAudioOutput(
                self.audio_output
            )

            self.media_player.setVideoOutput(
                self.video_widget
            )

            self.media_player.durationChanged.connect(
                self.on_media_duration_changed
            )

            self.media_player.positionChanged.connect(
                self.on_media_position_changed
            )

            self.media_player.playbackStateChanged.connect(
                self.on_playback_state_changed
            )

            self.media_player.mediaStatusChanged.connect(
                self.on_media_status_changed
            )

            self.media_player.errorOccurred.connect(
                self.on_media_error
            )

        else:

            layout.addWidget(
                self.output_preview_label,
                1
            )

        return panel

    # ==================================================
    # 6. 创建评价指标卡片
    # ==================================================

    def create_metric_card(self, title_text):

        card = QFrame()

        card.setObjectName("metricCard")

        card.setMinimumHeight(62)

        layout = QVBoxLayout(card)

        layout.setContentsMargins(
            8, 5, 8, 5
        )

        layout.setSpacing(1)

        title = QLabel(title_text)

        title.setStyleSheet(
            "color:#718096;"
            "font-size:10px;"
        )

        value = QLabel("—")

        value.setStyleSheet(
            "color:#243b5a;"
            "font-size:17px;"
            "font-weight:bold;"
        )

        detail = QLabel("等待计算")

        detail.setStyleSheet(
            "color:#8a9bb0;"
            "font-size:9px;"
        )

        layout.addWidget(title)
        layout.addWidget(value)
        layout.addWidget(detail)

        return card, value, detail

    def reset_metrics(self):

        for widgets in self.metric_widgets.values():

            widgets["value"].setText("—")
            widgets["detail"].setText("等待计算")

    def update_metrics(self, result):

        total_frames = result["frame_count"]

        # PSNR
        psnr_mean = result["psnr_mean"]
        psnr_std = result["psnr_std"]

        psnr_text = (
            "∞ dB"
            if psnr_mean == float("inf")
            else f"{psnr_mean:.3f} dB"
        )

        self.metric_widgets["psnr"]["value"].setText(
            psnr_text
        )

        self.metric_widgets["psnr"]["detail"].setText(
            f"标准差：{psnr_std:.3f} dB"
        )

        # SSIM
        self.metric_widgets["ssim"]["value"].setText(
            f"{result['ssim_mean']:.4f}"
        )

        self.metric_widgets["ssim"]["detail"].setText(
            f"标准差：{result['ssim_std']:.4f}"
        )

        # 原始视频人脸检出率
        original_rate = (
            result["original_face_detection_rate"]
        )

        original_detected = (
            result["original_detected_frames"]
        )

        self.metric_widgets["original_rate"]["value"].setText(
            f"{original_rate:.2f}%"
        )

        self.metric_widgets["original_rate"]["detail"].setText(
            f"检出帧数：{original_detected}/{total_frames}"
        )

        # 匿名化视频人脸检出率
        anonymized_rate = (
            result["anonymized_face_detection_rate"]
        )

        anonymized_detected = (
            result["anonymized_detected_frames"]
        )

        self.metric_widgets["anonymized_rate"]["value"].setText(
            f"{anonymized_rate:.2f}%"
        )

        self.metric_widgets["anonymized_rate"]["detail"].setText(
            f"检出帧数：{anonymized_detected}/{total_frames}"
        )

        # 原始视频人脸检测框总数
        self.metric_widgets["original_faces"]["value"].setText(
            str(result["original_total_faces"])
        )

        self.metric_widgets["original_faces"]["detail"].setText(
            "检测器输出的人脸框数量"
        )

        # 匿名化视频人脸检测框总数
        self.metric_widgets["anonymized_faces"]["value"].setText(
            str(result["anonymized_total_faces"])
        )

        self.metric_widgets["anonymized_faces"]["detail"].setText(
            "检测器输出的人脸框数量"
        )

    # ==================================================
    # 7. 曲线图展示
    # ==================================================

    def load_curve_image(self, curve_path):

        if not os.path.isfile(curve_path):

            self.chart_placeholder.setText(
                "未找到 FT 异常响应曲线。"
            )

            self.chart_placeholder.setPixmap(
                QPixmap()
            )

            return

        pixmap = QPixmap(curve_path)

        if pixmap.isNull():

            self.chart_placeholder.setText(
                "异常响应曲线图片无法读取。"
            )

            return

        self.curve_pixmap = pixmap

        self.refresh_curve_image()

    def refresh_curve_image(self):

        if self.curve_pixmap is None:
            return

        area = self.chart_placeholder.contentsRect().size()

        if area.width() < 2 or area.height() < 2:
            return

        scaled = self.curve_pixmap.scaled(
            area,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.chart_placeholder.setPixmap(scaled)

    # ==================================================
    # 8. 选择视频并启动完整流水线
    # ==================================================

    def select_video(self):

        if self.worker is not None and self.worker.isRunning():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择监控视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*.*)"
        )

        if not path:
            return

        self.selected_video_path = os.path.abspath(path)

        self.video_label.setText(
            os.path.basename(path)
        )

        self.clear_previous_results()

        self.select_button.setEnabled(False)

        self.extraction_progress.setValue(0)
        self.extraction_progress.show()

        self.status_label.setText(
            "正在启动完整视频处理流水线……"
        )

        self.worker = VideoPipelineWorker(
            self.selected_video_path
        )

        self.worker.progress_changed.connect(
            self.on_pipeline_progress
        )

        self.worker.status_changed.connect(
            self.on_pipeline_status
        )

        self.worker.log_message.connect(
            self.on_pipeline_log
        )

        self.worker.pipeline_succeeded.connect(
            self.on_pipeline_succeeded
        )

        self.worker.pipeline_failed.connect(
            self.on_pipeline_failed
        )

        self.worker.start()

    def clear_previous_results(self):

        if self.media_player is not None:
            self.media_player.stop()

        self.frame_dir = None

        self.frame_paths = []
        self.anonymized_frame_paths = []

        self.current_frame = None

        self.source_fps = 0.0
        self.source_frame_count = 0

        self.pipeline_result = None

        self.active_output_video_path = None
        self.pending_frame_seek = None

        self.curve_pixmap = None

        self.original_image_area.setText(
            "正在等待处理"
        )

        self.original_image_area.setPixmap(
            QPixmap()
        )

        self.output_preview_label.setText(
            "等待匿名化视频"
        )

        self.output_preview_label.setPixmap(
            QPixmap()
        )

        if self.output_stack is not None:
            self.output_stack.setCurrentWidget(
                self.output_preview_label
            )

        self.chart_placeholder.setText(
            "正在等待 FT 异常响应曲线"
        )

        self.chart_placeholder.setPixmap(
            QPixmap()
        )

        self.reset_metrics()

        self.frame_slider.blockSignals(True)

        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)

        self.frame_slider.blockSignals(False)

        self.frame_counter.setText("0 / 0 帧")

        self.play_button.setEnabled(False)
        self.play_button.setText("播放")

    # ==================================================
    # 9. 流水线信号处理
    # ==================================================

    def on_pipeline_progress(self, value):

        self.extraction_progress.setValue(
            max(0, min(100, int(value)))
        )

    def on_pipeline_status(self, message):

        self.status_label.setText(message)

    def on_pipeline_log(self, message):

        # 这里只接收流水线主动转发的警告、错误和阶段完成日志
        self.status_label.setText(message)

    def on_pipeline_succeeded(self, result):

        self.pipeline_result = result

        # 1. 加载原始帧和匿名化帧列表
        self.frame_dir = result["original_frames_dir"]

        self.frame_paths = sorted([
            os.path.join(self.frame_dir, name)
            for name in os.listdir(self.frame_dir)
            if name.lower().endswith(".jpg")
        ])

        anonymized_dir = result["anonymized_frames_dir"]

        self.anonymized_frame_paths = sorted([
            os.path.join(anonymized_dir, name)
            for name in os.listdir(anonymized_dir)
            if name.lower().endswith(".png")
        ])

        if not self.frame_paths:
            self.status_label.setText(
                "处理完成，但没有找到原始视频帧。"
            )
            self.select_button.setEnabled(True)
            return

        if len(self.frame_paths) != len(self.anonymized_frame_paths):
            self.status_label.setText(
                "处理完成，但原始帧与匿名化帧数量不一致，"
                "请检查输出目录。"
            )
            self.select_button.setEnabled(True)
            return

        # 2. 更新视频基本信息
        self.source_fps = result["fps"]
        self.source_frame_count = result["frame_count"]

        duration = (
            self.source_frame_count / self.source_fps
            if self.source_fps > 0 else 0
        )

        self.video_label.setText(
            f"{os.path.basename(self.selected_video_path)}"
            f" | {result['width']}×{result['height']}"
            f" | {self.source_fps:.2f} FPS"
            f" | {duration:.2f} 秒"
        )

        # 3. 配置同步进度条
        self.frame_slider.blockSignals(True)

        self.frame_slider.setRange(
            0,
            len(self.frame_paths) - 1
        )

        self.frame_slider.setValue(0)

        self.frame_slider.blockSignals(False)

        self.frame_slider.setEnabled(
            len(self.frame_paths) > 1
        )

        self.frame_counter.setText(
            f"1 / {len(self.frame_paths)} 帧"
        )

        # 4. 展示原始帧和匿名化帧
        self.show_frame(0)
        self.show_anonymized_frame(0)

        # 5. 自动加载匿名化视频
        output_video_path = result["anonymized_video_path"]

        if QT_MULTIMEDIA_AVAILABLE:
            self.load_anonymized_video(
                output_video_path
            )
        else:
            self.status_label.setText(
                "处理完成。当前 PySide6 环境缺少 Qt Multimedia，"
                "无法播放视频，但帧预览和评价结果可以正常展示。"
            )

        # 6. 展示评价指标
        self.update_metrics(result)

        # 7. 展示 FT 异常响应曲线
        self.load_curve_image(
            result["curve_path"]
        )

        # 8. 更新处理状态
        self.extraction_progress.setValue(100)
        self.extraction_progress.hide()

        self.select_button.setEnabled(True)

        self.status_label.setText(
            "全部处理完成。匿名化视频、异常响应曲线及评价指标已加载。"
        )

        QTimer.singleShot(
            0,
            self.refresh_preview_images
        )

    def on_pipeline_failed(self, error_message):

        self.extraction_progress.hide()

        self.select_button.setEnabled(True)

        self.status_label.setText(
            "视频处理失败，请检查错误信息。"
        )

        # 保留完整错误信息，方便定位问题
        QMessageBox.critical(
            self,
            "视频处理失败",
            "完整视频处理流水线执行失败。\n\n"
            + error_message[-5000:]
        )

    # ==================================================
    # 10. 匿名化视频播放器
    # ==================================================

    def load_anonymized_video(self, path):

        if self.media_player is None:
            self.status_label.setText(
                "当前环境不支持视频播放，已保留匿名化帧预览。"
            )
            return

        if not os.path.isfile(path):
            self.status_label.setText(
                f"找不到匿名化结果视频：{path}"
            )
            return

        self.media_player.stop()

        self.active_output_video_path = os.path.abspath(
            path
        )

        self.pending_frame_seek = 0

        self.media_player.setSource(
            QUrl.fromLocalFile(
                self.active_output_video_path
            )
        )

        self.play_button.setEnabled(True)
        self.play_button.setText("播放")

        self.output_stack.setCurrentWidget(
            self.output_preview_label
        )

    def toggle_video_playback(self):

        if self.media_player is None:
            return

        if (
            self.media_player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):

            self.media_player.pause()

        else:

            duration = self.media_player.duration()

            if (
                duration > 0
                and self.media_player.position() >= duration
            ):
                self.media_player.setPosition(0)

            self.output_stack.setCurrentWidget(
                self.video_widget
            )

            self.media_player.play()

    def on_playback_state_changed(self, state):

        if self.media_player is None:
            return

        if state == QMediaPlayer.PlaybackState.PlayingState:

            self.play_button.setText("暂停")

            self.output_stack.setCurrentWidget(
                self.video_widget
            )

        else:

            self.play_button.setText("播放")

            if self.frame_paths:
                self.show_anonymized_frame(
                    self.frame_slider.value()
                )

            self.output_stack.setCurrentWidget(
                self.output_preview_label
            )

    def on_media_duration_changed(self, duration):

        if duration > 0 and self.pending_frame_seek is not None:

            index = self.pending_frame_seek

            self.pending_frame_seek = None

            self.seek_anonymized_video_to_frame(
                index
            )

    def on_media_position_changed(self, position):

        if not self.frame_paths:
            return

        if self.source_fps <= 0:
            return

        if self.frame_slider.isSliderDown():
            return

        index = int(
            round(
                position * self.source_fps / 1000
            )
        )

        index = max(
            0,
            min(index, len(self.frame_paths) - 1)
        )

        if self.frame_slider.value() != index:

            self.frame_slider.blockSignals(True)

            self.frame_slider.setValue(index)

            self.frame_slider.blockSignals(False)

        self.show_frame(index)

    def seek_anonymized_video_to_frame(self, index):

        if self.media_player is None:
            return

        if self.source_fps <= 0:
            return

        duration = self.media_player.duration()

        if duration <= 0:
            self.pending_frame_seek = index
            return

        position = int(
            round(index * 1000 / self.source_fps)
        )

        self.media_player.setPosition(
            max(0, min(position, duration))
        )

        if (
            self.media_player.playbackState()
            != QMediaPlayer.PlaybackState.PlayingState
        ):
            self.show_anonymized_frame(index)

    def on_media_status_changed(self, status):

        if status == QMediaPlayer.MediaStatus.LoadedMedia:

            if self.pending_frame_seek is not None:

                index = self.pending_frame_seek

                self.pending_frame_seek = None

                self.seek_anonymized_video_to_frame(
                    index
                )

        elif status == QMediaPlayer.MediaStatus.EndOfMedia:

            self.play_button.setText("播放")

            if self.frame_paths:
                self.show_anonymized_frame(
                    self.frame_slider.value()
                )

            self.output_stack.setCurrentWidget(
                self.output_preview_label
            )

        elif status == QMediaPlayer.MediaStatus.InvalidMedia:

            self.status_label.setText(
                "匿名化视频无法解码，请检查视频编码格式。"
            )

    def on_media_error(self, error, message):

        if error != QMediaPlayer.Error.NoError:

            self.status_label.setText(
                f"视频播放失败：{message}"
            )

    # ==================================================
    # 11. 原始帧和匿名化帧同步显示
    # ==================================================

    def on_frame_changed(self, index):

        if not self.frame_paths:
            return

        self.show_frame(index)

        self.show_anonymized_frame(index)

        self.seek_anonymized_video_to_frame(
            index
        )

    def display_cv_frame(self, frame, label):

        if frame is None or label is None:
            return

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, _ = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        area = label.contentsRect().size()

        if area.width() < 2 or area.height() < 2:

            QTimer.singleShot(
                0,
                self.refresh_preview_images
            )

            return

        label.setPixmap(
            pixmap.scaled(
                area,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def show_frame(self, index):

        if index < 0 or index >= len(self.frame_paths):
            return

        frame = cv2.imread(
            self.frame_paths[index]
        )

        if frame is None:

            self.status_label.setText(
                f"无法读取原始视频第 {index + 1} 帧。"
            )

            return

        self.current_frame = frame

        self.display_cv_frame(
            frame,
            self.original_image_area
        )

        self.frame_counter.setText(
            f"{index + 1} / {len(self.frame_paths)} 帧"
        )

    def show_anonymized_frame(self, index):

        if (
            index < 0
            or index >= len(self.anonymized_frame_paths)
        ):
            return

        frame = cv2.imread(
            self.anonymized_frame_paths[index]
        )

        if frame is None:
            return

        self.display_cv_frame(
            frame,
            self.output_preview_label
        )

    # ==================================================
    # 12. 窗口尺寸变化时刷新图片
    # ==================================================

    def refresh_preview_images(self):

        if self.current_frame is not None:

            self.display_cv_frame(
                self.current_frame,
                self.original_image_area
            )

        if (
            self.anonymized_frame_paths
            and (
                self.media_player is None
                or self.media_player.playbackState()
                != QMediaPlayer.PlaybackState.PlayingState
            )
        ):

            self.show_anonymized_frame(
                self.frame_slider.value()
            )

        self.refresh_curve_image()

    def resizeEvent(self, event):

        super().resizeEvent(event)

        QTimer.singleShot(
            0,
            self.refresh_preview_images
        )

    # ==================================================
    # 13. 关闭窗口保护
    # ==================================================

    def closeEvent(self, event):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            QMessageBox.information(
                self,
                "正在处理视频",
                "视频处理尚未完成，请等待当前任务结束后再关闭系统。"
            )

            event.ignore()
            return

        if self.media_player is not None:
            self.media_player.stop()

        event.accept()
