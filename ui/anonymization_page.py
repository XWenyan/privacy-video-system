import os
import shutil
import cv2
from PySide6.QtCore import QThread, Signal, Qt, QUrl, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QSlider, QFileDialog, QProgressBar, QSizePolicy, QStackedLayout)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    QT_MULTIMEDIA_AVAILABLE = True
except (ImportError, OSError):
    QAudioOutput = QMediaPlayer = QVideoWidget = None
    QT_MULTIMEDIA_AVAILABLE = False


class VideoFrameExtractionWorker(QThread):
    progress_changed = Signal(int, int)
    extraction_finished = Signal(str, int, float, int, int)
    extraction_failed = Signal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.frame_dir = os.path.join(root, "outputs", "extracted_frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        for name in os.listdir(self.frame_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(os.path.join(self.frame_dir, name))

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件，请检查视频格式或文件是否损坏。")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                target = os.path.join(self.frame_dir, f"frame_{count:06d}.jpg")
                if not cv2.imwrite(target, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"第 {count + 1} 帧保存失败。")
                count += 1
                if count % 10 == 0:
                    self.progress_changed.emit(count, total)
            if count == 0:
                raise RuntimeError("没有从视频中读取到有效画面。")
            self.extraction_finished.emit(self.frame_dir, count, max(fps, 0), width, height)
        except Exception as exc:
            self.extraction_failed.emit(str(exc))
        finally:
            cap.release()


class AnonymizationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_video_path = None
        self.frame_dir = None
        self.frame_paths = []
        self.anonymized_frame_paths = []
        self.current_frame = None
        self.worker = None
        self.media_player = None
        self.audio_output = None
        self.active_output_video_path = None
        self.source_fps = 0.0
        self.source_frame_count = 0
        self.pending_frame_seek = None
        self.init_ui()
        self.load_default_anonymized_video()

    def get_project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(25, 18, 25, 16)
        main.setSpacing(10)
        title = QLabel("视频匿名化")
        title.setStyleSheet("font-size:23px;font-weight:bold;color:#243b5a")
        desc = QLabel("上传监控视频，执行逐帧匿名化，并查看视频异常分数变化。")
        desc.setStyleSheet("color:#718096;font-size:13px")
        main.addWidget(title)
        main.addWidget(desc)

        upload = QFrame(); upload.setObjectName("uploadCard")
        ul = QHBoxLayout(upload); ul.setContentsMargins(16, 8, 16, 8)
        self.video_label = QLabel("尚未选择视频")
        self.select_button = QPushButton("选择视频")
        self.select_button.setFixedSize(120, 38)
        self.select_button.clicked.connect(self.select_video)
        ul.addWidget(self.video_label, 1); ul.addWidget(self.select_button)
        main.addWidget(upload)

        self.extraction_progress = QProgressBar()
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setTextVisible(False)
        self.extraction_progress.setFixedHeight(6)
        self.extraction_progress.hide()
        main.addWidget(self.extraction_progress)

        previews = QHBoxLayout(); previews.setSpacing(14)
        self.original_panel = self.create_preview_panel("原始视频帧")
        self.original_image_area = self.original_panel.image_area
        self.anonymized_panel = self.create_video_player_panel()
        previews.addWidget(self.original_panel, 1)
        previews.addWidget(self.anonymized_panel, 1)
        main.addLayout(previews, 5)

        progress = QFrame(); progress.setObjectName("contentCard")
        pl = QVBoxLayout(progress); pl.setContentsMargins(16, 8, 16, 8); pl.setSpacing(6)
        header = QHBoxLayout()
        ptitle = QLabel("视频同步进度"); ptitle.setStyleSheet("font-weight:bold;color:#243b5a")
        self.frame_counter = QLabel("0 / 0 帧"); self.frame_counter.setStyleSheet("color:#718096")
        header.addWidget(ptitle); header.addStretch(); header.addWidget(self.frame_counter)
        pl.addLayout(header)
        controls = QHBoxLayout(); controls.setSpacing(10)
        self.play_button = QPushButton("播放"); self.play_button.setFixedSize(76, 34)
        self.play_button.setEnabled(False); self.play_button.clicked.connect(self.toggle_video_playback)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0); self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        controls.addWidget(self.play_button); controls.addWidget(self.frame_slider, 1)
        pl.addLayout(controls); main.addWidget(progress)

        chart = QFrame(); chart.setObjectName("contentCard")
        cl = QVBoxLayout(chart); cl.setContentsMargins(16, 7, 16, 7)
        ct = QLabel("视频异常分数变化"); ct.setStyleSheet("font-weight:bold;color:#243b5a")
        self.chart_placeholder = QLabel("完成匿名化与 VAD 推理后，将在这里显示逐帧异常分数曲线。")
        self.chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_placeholder.setMinimumHeight(55)
        self.chart_placeholder.setStyleSheet("background:#f7f8fa;color:#8a9bb0;border-radius:5px")
        cl.addWidget(ct); cl.addWidget(self.chart_placeholder); main.addWidget(chart, 1)
        self.status_label = QLabel("系统就绪，等待选择视频。")
        self.status_label.setStyleSheet("color:#718096;font-size:12px")
        main.addWidget(self.status_label)
        self.setStyleSheet("""
            QFrame#uploadCard,QFrame#contentCard{background:white;border:1px solid #dce3ea;border-radius:7px}
            QPushButton{background:#2f9e62;color:white;border:0;border-radius:5px;padding:6px 10px}
            QPushButton:hover{background:#278653} QPushButton:disabled{background:#aab5bd}
            QSlider::groove:horizontal{height:5px;background:#dce3e8;border-radius:2px}
            QSlider::handle:horizontal{background:#2f9e62;width:14px;height:14px;margin:-5px 0;border-radius:7px}
            QProgressBar{background:#e9edf0;border:0;border-radius:3px}
            QProgressBar::chunk{background:#2f9e62;border-radius:3px}
        """)

    def _prepare_label(self, label, bg="#111827"):
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(0, 0)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setScaledContents(False)
        label.setStyleSheet(f"background:{bg};color:white;border-radius:5px")

    def create_preview_panel(self, panel_title):
        panel = QFrame(); panel.setObjectName("contentCard")
        panel.setMinimumHeight(220)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel); layout.setContentsMargins(12, 10, 12, 10); layout.setSpacing(7)
        heading = QLabel(panel_title); heading.setStyleSheet("font-weight:bold;color:#243b5a")
        image = QLabel("等待视频处理")
        self._prepare_label(image, "#f0f2f5")
        image.setStyleSheet("background:#f0f2f5;color:#8a9bb0;border:1px dashed #cbd5e1;border-radius:5px")
        layout.addWidget(heading); layout.addWidget(image, 1)
        panel.image_area = image
        return panel

    def create_video_player_panel(self):
        panel = QFrame(); panel.setObjectName("contentCard")
        panel.setMinimumHeight(220)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel); layout.setContentsMargins(12, 10, 12, 10); layout.setSpacing(7)
        header = QHBoxLayout()
        heading = QLabel("匿名化结果视频"); heading.setStyleSheet("font-weight:bold;color:#243b5a")
        self.select_result_button = QPushButton("选择结果视频")
        self.select_result_button.setFixedHeight(30)
        self.select_result_button.clicked.connect(self.select_anonymized_video)
        self.select_result_button.setEnabled(QT_MULTIMEDIA_AVAILABLE)
        header.addWidget(heading); header.addStretch(); header.addWidget(self.select_result_button)
        layout.addLayout(header)
        if QT_MULTIMEDIA_AVAILABLE:
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumSize(0, 0)
            self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.video_widget.setStyleSheet("background:#111827")
            self.output_preview_label = QLabel("等待匿名化视频")
            self._prepare_label(self.output_preview_label)
            self.output_stack = QStackedLayout(); self.output_stack.setContentsMargins(0, 0, 0, 0)
            self.output_stack.addWidget(self.video_widget)
            self.output_stack.addWidget(self.output_preview_label)
            self.output_stack.setCurrentWidget(self.output_preview_label)
            layout.addLayout(self.output_stack, 1)
            self.audio_output = QAudioOutput(self)
            self.media_player = QMediaPlayer(self)
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)
            self.media_player.durationChanged.connect(self.on_media_duration_changed)
            self.media_player.positionChanged.connect(self.on_media_position_changed)
            self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
            self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
            self.media_player.errorOccurred.connect(self.on_media_error)
        else:
            self.video_widget = QLabel("当前 PySide6 环境缺少 Qt Multimedia 模块。")
            self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.video_widget, 1)
            self.output_preview_label = None
            self.output_stack = None
        return panel

    def load_default_anonymized_video(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            self.status_label.setText("当前 PySide6 环境不支持 Qt Multimedia，视频播放器不可用。"); return
        out = os.path.join(self.get_project_root(), "outputs")
        for name in ("anonymized_video_with_audio.mp4", "anonymized_video.mp4"):
            path = os.path.join(out, name)
            if os.path.isfile(path):
                self.load_anonymized_video(path); return
        self.status_label.setText("未找到输出视频。请先完成匿名化，或手动选择结果视频。")

    def select_anonymized_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择匿名化结果视频",
            os.path.join(self.get_project_root(), "outputs"),
            "视频文件 (*.mp4 *.m4v *.mov *.avi *.mkv);;所有文件 (*.*)")
        if path: self.load_anonymized_video(path)

    def load_anonymized_video(self, path):
        if self.media_player is None or not os.path.isfile(path):
            self.status_label.setText("视频播放器不可用，或找不到所选视频文件。"); return
        self.media_player.stop()
        self.active_output_video_path = os.path.abspath(path)
        self.pending_frame_seek = self.frame_slider.value() if self.frame_paths else 0
        self.media_player.setSource(QUrl.fromLocalFile(self.active_output_video_path))
        self.play_button.setEnabled(True)
        self.play_button.setText("播放")
        self.output_stack.setCurrentWidget(self.output_preview_label)
        self.status_label.setText("匿名化结果视频已加载，可通过公共进度条定位画面。")

    def toggle_video_playback(self):
        if self.media_player is None: return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            if self.media_player.duration() > 0 and self.media_player.position() >= self.media_player.duration():
                self.media_player.setPosition(0)
            self.output_stack.setCurrentWidget(self.video_widget)
            self.media_player.play()

    def on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("暂停")
            self.output_stack.setCurrentWidget(self.video_widget)
        else:
            self.play_button.setText("播放")
            if self.frame_paths: self.show_anonymized_frame(self.frame_slider.value())
            self.output_stack.setCurrentWidget(self.output_preview_label)

    def on_media_duration_changed(self, duration):
        if duration > 0 and self.pending_frame_seek is not None:
            index = self.pending_frame_seek; self.pending_frame_seek = None
            self.seek_anonymized_video_to_frame(index)

    def on_media_position_changed(self, position):
        if not self.frame_paths or self.source_fps <= 0 or self.frame_slider.isSliderDown(): return
        index = max(0, min(int(round(position * self.source_fps / 1000)), len(self.frame_paths)-1))
        if self.frame_slider.value() != index:
            self.frame_slider.blockSignals(True); self.frame_slider.setValue(index); self.frame_slider.blockSignals(False)
        self.show_frame(index)

    def seek_anonymized_video_to_frame(self, index):
        if self.media_player is None or self.source_fps <= 0: return
        duration = self.media_player.duration()
        if duration <= 0:
            self.pending_frame_seek = index; return
        position = int(round(index * 1000 / self.source_fps))
        self.media_player.setPosition(max(0, min(position, duration)))
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.show_anonymized_frame(index)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self.pending_frame_seek is not None:
                index = self.pending_frame_seek; self.pending_frame_seek = None
                self.seek_anonymized_video_to_frame(index)
            self.status_label.setText("匿名化视频已就绪，可以播放或拖动进度条查看。")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.play_button.setText("播放")
            if self.frame_paths: self.show_anonymized_frame(self.frame_slider.value())
            self.output_stack.setCurrentWidget(self.output_preview_label)
            self.status_label.setText("视频播放结束，可以拖动进度条重新查看。")
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self.status_label.setText("匿名化视频无法解码，请检查文件格式或编码。")

    def on_media_error(self, error, message):
        if error != QMediaPlayer.Error.NoError:
            self.status_label.setText(f"视频播放失败：{message}")

    def select_video(self):
        if self.worker is not None and self.worker.isRunning(): return
        path, _ = QFileDialog.getOpenFileName(self, "选择监控视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)")
        if not path: return
        self.selected_video_path = path
        self.frame_paths = []; self.anonymized_frame_paths = []; self.current_frame = None
        self.source_fps = 0.0; self.source_frame_count = 0
        self.video_label.setText(os.path.basename(path))
        self.original_image_area.setText("正在读取视频并抽取帧..."); self.original_image_area.setPixmap(QPixmap())
        if self.output_preview_label is not None:
            self.output_preview_label.setText("等待匿名化视频帧"); self.output_preview_label.setPixmap(QPixmap())
        self.frame_slider.setEnabled(False); self.frame_slider.setRange(0, 0)
        self.frame_counter.setText("0 / 0 帧")
        self.extraction_progress.setValue(0); self.extraction_progress.show()
        self.select_button.setEnabled(False); self.status_label.setText("正在读取视频，请稍候...")
        self.worker = VideoFrameExtractionWorker(path)
        self.worker.progress_changed.connect(self.on_extraction_progress)
        self.worker.extraction_finished.connect(self.on_extraction_finished)
        self.worker.extraction_failed.connect(self.on_extraction_failed)
        self.worker.start()

    def on_extraction_progress(self, current, total):
        if total > 0:
            self.extraction_progress.setValue(min(int(current / total * 100), 100))
            self.status_label.setText(f"正在抽取视频帧：{current} / 约 {total} 帧")
        else: self.status_label.setText(f"正在抽取视频帧：已处理 {current} 帧")

    def on_extraction_finished(self, frame_dir, count, fps, width, height):
        self.frame_dir = frame_dir
        self.frame_paths = [os.path.join(frame_dir, f"frame_{i:06d}.jpg") for i in range(count)]
        self.source_fps = fps if fps > 0 else 0.0; self.source_frame_count = count
        anon_dir = os.path.join(self.get_project_root(), "outputs", "anonymized_frames")
        if os.path.isdir(anon_dir):
            names = [n for n in os.listdir(anon_dir) if n.startswith("frame_") and n.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.anonymized_frame_paths = [os.path.join(anon_dir, n) for n in sorted(names)]
        else: self.anonymized_frame_paths = []
        duration = count / fps if fps > 0 else 0
        self.video_label.setText(f"{os.path.basename(self.selected_video_path)} | {width}×{height} | {fps:.2f} FPS | {duration:.2f} 秒")
        self.frame_slider.blockSignals(True); self.frame_slider.setRange(0, count-1); self.frame_slider.setValue(0); self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(count > 1); self.frame_counter.setText(f"1 / {count} 帧")
        self.extraction_progress.hide(); self.select_button.setEnabled(True)
        self.show_frame(0); self.show_anonymized_frame(0); self.seek_anonymized_video_to_frame(0)
        self.status_label.setText(f"视频读取完成，共抽取 {count} 帧。可通过公共进度条同步查看两侧画面。")
        QTimer.singleShot(0, self.refresh_preview_images)

    def on_extraction_failed(self, message):
        self.extraction_progress.hide(); self.select_button.setEnabled(True)
        self.original_image_area.setText("视频读取失败")
        self.status_label.setText(f"处理失败：{message}")

    def on_frame_changed(self, index):
        if not self.frame_paths: return
        self.show_frame(index); self.show_anonymized_frame(index)
        self.seek_anonymized_video_to_frame(index)

    def _display_cv_frame(self, frame, label):
        if frame is None or label is None: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        area = label.contentsRect().size()
        if area.width() < 2 or area.height() < 2:
            QTimer.singleShot(0, self.refresh_preview_images); return
        label.setPixmap(pixmap.scaled(area, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))

    def show_frame(self, index):
        if index < 0 or index >= len(self.frame_paths): return
        frame = cv2.imread(self.frame_paths[index])
        if frame is None:
            self.status_label.setText(f"无法读取第 {index+1} 帧。"); return
        self.current_frame = frame
        self._display_cv_frame(frame, self.original_image_area)
        self.frame_counter.setText(f"{index+1} / {len(self.frame_paths)} 帧")

    def show_anonymized_frame(self, index):
        if self.output_preview_label is None: return
        if index < 0 or index >= len(self.anonymized_frame_paths):
            if not self.anonymized_frame_paths:
                self.output_preview_label.setText("尚未找到匿名化帧，请先完成视频匿名化")
                self.output_preview_label.setPixmap(QPixmap())
            return
        frame = cv2.imread(self.anonymized_frame_paths[index])
        if frame is not None: self._display_cv_frame(frame, self.output_preview_label)

    def refresh_preview_images(self):
        if self.current_frame is not None: self._display_cv_frame(self.current_frame, self.original_image_area)
        if self.anonymized_frame_paths: self.show_anonymized_frame(self.frame_slider.value())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_preview_images)

    def closeEvent(self, event):
        if self.media_player is not None: self.media_player.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit(); self.worker.wait(1500)
        super().closeEvent(event)
