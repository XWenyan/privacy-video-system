# -*- coding: utf-8 -*-
"""PySide6 page for the one-click video anonymization pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QPushButton, QFileDialog, QProgressBar, QSizePolicy, QMessageBox,
    QScrollArea,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    QT_MULTIMEDIA_AVAILABLE = True
except (ImportError, OSError):
    QAudioOutput = QMediaPlayer = QVideoWidget = None
    QT_MULTIMEDIA_AVAILABLE = False

from video_pipeline import process_video


class PipelineWorker(QThread):
    progress_changed = Signal(int, str)
    pipeline_finished = Signal(dict)
    pipeline_failed = Signal(str)

    def __init__(self, video_path: str, project_root: str):
        super().__init__()
        self.video_path = video_path
        self.project_root = project_root

    def run(self):
        try:
            result = process_video(
                self.video_path,
                self.project_root,
                progress_callback=lambda value, message: self.progress_changed.emit(
                    value, message
                ),
                cancel_check=self.isInterruptionRequested,
            )
            self.pipeline_finished.emit(result)
        except InterruptedError as exc:
            self.pipeline_failed.emit(str(exc))
        except Exception as exc:
            import traceback
            self.pipeline_failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class AnonymizationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[1]
        self.selected_video_path: Path | None = None
        self.worker: PipelineWorker | None = None
        self.result: dict | None = None
        self.media_player = None
        self.audio_output = None
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 16, 22, 16)
        root.setSpacing(10)

        title = QLabel("视频隐私匿名化系统")
        title.setStyleSheet("font-size:23px;font-weight:700;color:#243b5a")
        subtitle = QLabel(
            "上传视频后自动完成抽帧、FA 匿名化、视频重建、FT 异常分析、"
            "PSNR/SSIM 评价及人脸检测。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#718096;font-size:13px")
        root.addWidget(title)
        root.addWidget(subtitle)

        upload = QFrame()
        upload.setObjectName("card")
        upload_layout = QHBoxLayout(upload)
        upload_layout.setContentsMargins(14, 10, 14, 10)
        self.video_name_label = QLabel("尚未选择视频")
        self.video_name_label.setStyleSheet("color:#344256")
        self.select_button = QPushButton("上传视频")
        self.select_button.setFixedSize(112, 36)
        self.select_button.clicked.connect(self.select_video)
        self.start_button = QPushButton("开始自动处理")
        self.start_button.setFixedSize(132, 36)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_pipeline)
        upload_layout.addWidget(self.video_name_label, 1)
        upload_layout.addWidget(self.select_button)
        upload_layout.addWidget(self.start_button)
        root.addWidget(upload)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("系统就绪，请上传视频。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#64748b;font-size:12px")
        root.addWidget(self.status_label)

        previews = QHBoxLayout()
        previews.setSpacing(12)
        self.original_image = self._make_image_panel("原始视频画面", "等待上传视频")
        self.anonymized_panel = self._make_video_panel()
        previews.addWidget(self.original_image["panel"], 1)
        previews.addWidget(self.anonymized_panel, 1)
        root.addLayout(previews, 4)

        chart_panel = QFrame()
        chart_panel.setObjectName("card")
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(12, 9, 12, 9)
        chart_title = QLabel("FT 异常响应曲线")
        chart_title.setStyleSheet("font-weight:700;color:#243b5a")
        self.chart_label = QLabel("完成视频处理后，在这里展示匿名化视频的异常响应曲线。")
        self.chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_label.setMinimumHeight(150)
        self.chart_label.setStyleSheet(
            "background:#f7f8fa;color:#8a9bb0;border-radius:5px;padding:8px"
        )
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(self.chart_label, 1)
        root.addWidget(chart_panel, 3)

        metrics_panel = QFrame()
        metrics_panel.setObjectName("card")
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(12, 9, 12, 9)
        metrics_title = QLabel("自动评价结果")
        metrics_title.setStyleSheet("font-weight:700;color:#243b5a")
        metrics_layout.addWidget(metrics_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.metric_labels = {}
        metric_specs = [
            ("psnr", "PSNR（均值 ± 标准差）"),
            ("ssim", "SSIM（均值 ± 标准差）"),
            ("original_face_rate", "原始视频人脸检出率"),
            ("anonymized_face_rate", "匿名化视频人脸检出率"),
            ("face_delta", "人脸检出率变化"),
            ("face_boxes", "人脸检测框数量（原始 / 匿名化）"),
        ]
        for index, (key, label_text) in enumerate(metric_specs):
            cell = QFrame()
            cell.setStyleSheet("background:#f7f9fb;border-radius:5px")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(10, 7, 10, 7)
            name = QLabel(label_text)
            name.setStyleSheet("color:#718096;font-size:11px")
            value = QLabel("—")
            value.setStyleSheet("color:#243b5a;font-size:17px;font-weight:700")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            cell_layout.addWidget(name)
            cell_layout.addWidget(value)
            grid.addWidget(cell, index // 3, index % 3)
            self.metric_labels[key] = value
        metrics_layout.addLayout(grid)
        root.addWidget(metrics_panel, 1)

        self.setStyleSheet("""
            QFrame#card{background:white;border:1px solid #dce3ea;border-radius:7px}
            QPushButton{background:#2f9e62;color:white;border:0;border-radius:5px;padding:6px 10px}
            QPushButton:hover{background:#278653}
            QPushButton:disabled{background:#aab5bd;color:#f7f7f7}
            QProgressBar{background:#e9edf0;border:0;border-radius:3px;height:7px}
            QProgressBar::chunk{background:#2f9e62;border-radius:3px}
        """)

    def _make_image_panel(self, title, placeholder):
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        heading = QLabel(title)
        heading.setStyleSheet("font-weight:700;color:#243b5a")
        image = QLabel(placeholder)
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setMinimumHeight(150)
        image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image.setStyleSheet(
            "background:#111827;color:#cbd5e1;border-radius:5px"
        )
        layout.addWidget(heading)
        layout.addWidget(image, 1)
        return {"panel": panel, "label": image}

    def _make_video_panel(self):
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        heading = QLabel("匿名化结果视频")
        heading.setStyleSheet("font-weight:700;color:#243b5a")
        layout.addWidget(heading)

        if QT_MULTIMEDIA_AVAILABLE:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(150)
            self.video_widget.setStyleSheet("background:#111827;border-radius:5px")
            layout.addWidget(self.video_widget, 1)
            controls = QHBoxLayout()
            self.play_button = QPushButton("播放")
            self.play_button.setEnabled(False)
            self.play_button.clicked.connect(self.toggle_playback)
            self.output_path_label = QLabel("等待生成匿名化视频")
            self.output_path_label.setStyleSheet("color:#718096;font-size:11px")
            self.output_path_label.setWordWrap(True)
            controls.addWidget(self.play_button)
            controls.addWidget(self.output_path_label, 1)
            layout.addLayout(controls)
            self.audio_output = QAudioOutput(self)
            self.media_player = QMediaPlayer(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            self.media_player.playbackStateChanged.connect(self._playback_state_changed)
            self.media_player.errorOccurred.connect(self._media_error)
        else:
            self.video_widget = None
            self.play_button = None
            self.output_path_label = QLabel(
                "当前 PySide6 环境不支持 Qt Multimedia，"
                "处理完成后仍会保存匿名化视频文件。"
            )
            self.output_path_label.setWordWrap(True)
            layout.addWidget(self.output_path_label, 1)
        return panel

    def select_video(self):
        if self.worker is not None and self.worker.isRunning():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待匿名化视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*.*)",
        )
        if not path:
            return
        self.selected_video_path = Path(path)
        self.video_name_label.setText(self.selected_video_path.name)
        self.start_button.setText("正在准备…")
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.chart_label.setText("等待处理完成后生成异常响应曲线。")
        self.chart_label.setPixmap(QPixmap())
        for label in self.metric_labels.values():
            label.setText("—")
        self.original_image["label"].setText("已选择视频，等待处理")
        self.original_image["label"].setPixmap(QPixmap())
        self.status_label.setText("视频已选择，自动开始处理流程……")
        QTimer.singleShot(0, self.start_pipeline)

    def start_pipeline(self):
        if self.selected_video_path is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.select_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在启动处理流程……")

        self.worker = PipelineWorker(
            str(self.selected_video_path),
            str(self.project_root),
        )
        self.worker.progress_changed.connect(self.on_progress)
        self.worker.pipeline_finished.connect(self.on_pipeline_finished)
        self.worker.pipeline_failed.connect(self.on_pipeline_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_pipeline_finished(self, result):
        self.result = result
        first_frame = Path(result["original_frames_dir"]) / "frame_000000.jpg"
        self._show_image(self.original_image["label"], first_frame)

        curve_path = Path(result["curve_path"])
        pixmap = QPixmap(str(curve_path))
        if pixmap.isNull():
            self.chart_label.setText(f"曲线图片无法加载：{curve_path}")
        else:
            self.chart_label.setPixmap(
                pixmap.scaled(
                    self.chart_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        quality = result["image_quality"]
        face = result["face_detection"]
        self.metric_labels["psnr"].setText(
            f'{quality["psnr_mean"]:.4f} ± {quality["psnr_std"]:.4f} dB'
        )
        self.metric_labels["ssim"].setText(
            f'{quality["ssim_mean"]:.4f} ± {quality["ssim_std"]:.4f}'
        )
        self.metric_labels["original_face_rate"].setText(
            f'{face["original_detection_rate"]:.2f}%'
        )
        self.metric_labels["anonymized_face_rate"].setText(
            f'{face["anonymized_detection_rate"]:.2f}%'
        )
        self.metric_labels["face_delta"].setText(
            f'{face["detection_rate_delta_pp"]:+.2f} 个百分点'
        )
        self.metric_labels["face_boxes"].setText(
            f'{face["original_face_boxes"]} / {face["anonymized_face_boxes"]}'
        )

        output_video = Path(result["anonymized_video_path"])
        self.output_path_label.setText(str(output_video))
        if self.media_player is not None and output_video.is_file():
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(str(output_video)))
            self.play_button.setEnabled(True)

        self.status_label.setText(
            f'处理完成：{result["frame_count"]} 帧，'
            f'结果已保存到 {result["output_dir"]}'
        )
        self.progress_bar.setValue(100)

    def on_pipeline_failed(self, message):
        self.status_label.setText("处理失败，请查看错误信息。")
        QMessageBox.critical(self, "视频处理失败", message)

    def on_worker_finished(self):
        self.select_button.setEnabled(True)
        self.start_button.setText("重新处理")
        self.start_button.setEnabled(self.selected_video_path is not None)

    def _show_image(self, label, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            label.setText(f"无法读取图片：{path}")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        label.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.result:
            self._show_image(
                self.original_image["label"],
                Path(self.result["original_frames_dir"]) / "frame_000000.jpg",
            )
            self.chart_label.setPixmap(
                QPixmap(self.result["curve_path"]).scaled(
                    self.chart_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def toggle_playback(self):
        if self.media_player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _playback_state_changed(self, state):
        from PySide6.QtMultimedia import QMediaPlayer
        if self.play_button is not None:
            self.play_button.setText(
                "暂停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放"
            )

    def _media_error(self, error, message):
        if self.media_player is not None:
            self.status_label.setText(f"匿名化视频播放失败：{message}")
