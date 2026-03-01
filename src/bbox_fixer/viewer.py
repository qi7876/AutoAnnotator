"""GUI for editing MOT boxes."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .mot_io import MotBox, MotStore
from .state import EditorState

try:
    from auto_annotator.annotators.tracker import ObjectTracker
except Exception:  # pragma: no cover - tracker is optional at runtime
    ObjectTracker = None


@dataclass
class ClipEntry:
    sport: str
    event: str
    clip_id: str
    task_name: str
    video_path: Path
    mot_path: Path
    json_path: Path
    ann_index: int
    retrack: Optional[bool] = None
    is_window_consistence: Optional[bool] = None


class OpenCVVideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            tmp = cv2.VideoCapture(str(video_path))
            if not tmp.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            count = 0
            while True:
                ok = tmp.grab()
                if not ok:
                    break
                count += 1
            tmp.release()
            frame_count = count

        self.frame_count = max(1, frame_count)
        self._last_index: int | None = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def read_rgb(self, index: int):
        if index < 0 or index >= self.frame_count:
            return None

        if self._last_index is None or index != self._last_index + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame_bgr = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = self._cap.read()
            if not ok:
                return None

        self._last_index = index
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class HandleItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent: QtWidgets.QGraphicsItem, corner: str):
        super().__init__(-8, -8, 16, 16, parent)
        self.corner = corner
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            parent = self.parentItem()
            if isinstance(parent, BoxItem):
                parent.update_from_handles()
        return super().itemChange(change, value)


class BoxItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, box: MotBox):
        super().__init__()
        self.box = box
        self.setRect(QtCore.QRectF(box.left, box.top, box.width, box.height))
        self.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
        self.setZValue(2)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.handle_tl = HandleItem(self, "tl")
        self.handle_br = HandleItem(self, "br")
        self._sync_handles()

    def _sync_handles(self) -> None:
        rect = self.rect()
        self.handle_tl.setPos(rect.left(), rect.top())
        self.handle_br.setPos(rect.right(), rect.bottom())

    def update_from_handles(self) -> None:
        tl = self.handle_tl.pos()
        br = self.handle_br.pos()
        left = min(tl.x(), br.x())
        top = min(tl.y(), br.y())
        right = max(tl.x(), br.x())
        bottom = max(tl.y(), br.y())
        self.setRect(QtCore.QRectF(left, top, right - left, bottom - top))
        self.box.left = left
        self.box.top = top
        self.box.width = right - left
        self.box.height = bottom - top

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._sync_handles()
        return super().itemChange(change, value)


class FrameView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene())
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._pixmap_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.box_items: List[BoxItem] = []
        self._fit_to_view = True

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                steps = delta / 120.0
                base = 1.1
                factor = base**steps if steps > 0 else (1.0 / base) ** (-steps)
                self.zoom(factor)
            event.accept()
            return

        super().wheelEvent(event)

    def set_frame(self, image: QtGui.QImage, boxes: List[MotBox]) -> None:
        self.scene().clear()
        pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self.box_items = []
        for box in boxes:
            item = BoxItem(box)
            self.scene().addItem(item)
            self.box_items.append(item)
        self.scene().setSceneRect(pixmap.rect())
        if self._fit_to_view:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def sync_boxes(self) -> List[MotBox]:
        for item in self.box_items:
            item.update_from_handles()
        return [item.box for item in self.box_items]

    def set_fit_mode(self, fit: bool) -> None:
        self._fit_to_view = fit
        if self._pixmap_item and fit:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self._fit_to_view = False
        self.scale(factor, factor)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_to_view:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)


class MotEditorWindow(QtWidgets.QMainWindow):
    def __init__(
        self, dataset_root: Path, output_root: Path, state_path: Path, flagged_mode: bool = False
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.state_path = state_path
        self.flagged_mode = flagged_mode
        self.state = EditorState.load(state_path)

        self.clip_entries = self._discover_clips(flagged_only=self.flagged_mode)
        if not self.clip_entries:
            raise RuntimeError("No clips found in dataset.")

        self.clip_index = max(0, min(self.state.clip_index, len(self.clip_entries) - 1))
        self.frame_index = max(1, self.state.frame_index)

        self.store = MotStore()
        self.video_reader: Optional[OpenCVVideoReader] = None
        self.total_frames = 1
        self._last_empty_notice: Optional[int] = None
        self.reviewed = False
        self._annotation_cache: Dict[Path, dict] = {}
        self._retrack_worker: Optional["RetrackWorker"] = None

        self._build_ui()
        self._populate_flagged_list()
        self._load_clip(self.clip_entries[self.clip_index])

    def _build_ui(self) -> None:
        self.setWindowTitle("BBoxFixer")
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        if self.flagged_mode:
            flagged_group = QtWidgets.QGroupBox("标记任务 (retrack / window)")
            fg_layout = QtWidgets.QVBoxLayout(flagged_group)
            self.flagged_list = QtWidgets.QListWidget()
            self.flagged_list.itemClicked.connect(self._handle_flagged_selection)
            self.flagged_list.itemDoubleClicked.connect(self._handle_flagged_selection)
            fg_layout.addWidget(self.flagged_list)
            layout.addWidget(flagged_group)

        self.frame_view = FrameView()
        layout.addWidget(self.frame_view, stretch=1)

        ann_group = QtWidgets.QGroupBox("Annotation 参考")
        ann_layout = QtWidgets.QVBoxLayout(ann_group)
        self.q_window_label = QtWidgets.QLabel("Q_window_frame: -")
        self.q_window_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ann_layout.addWidget(self.q_window_label)
        self.a_window_label = QtWidgets.QLabel("A_window_frame: -")
        self.a_window_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ann_layout.addWidget(self.a_window_label)
        self.answer_window_label = QtWidgets.QLabel("answer_window: -")
        self.answer_window_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ann_layout.addWidget(self.answer_window_label)
        self.mot_ranges_label = QtWidgets.QLabel("MOT 帧区间(有框): -")
        self.mot_ranges_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ann_layout.addWidget(self.mot_ranges_label)
        self.answer_box = QtWidgets.QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setFixedHeight(120)
        ann_layout.addWidget(self.answer_box)
        layout.addWidget(ann_group)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        layout.addWidget(self.log_box)

        self.review_checkbox = QtWidgets.QCheckBox("Reviewed")
        layout.addWidget(self.review_checkbox)

        controls = QtWidgets.QHBoxLayout()
        self.prev_clip_btn = QtWidgets.QPushButton("<< Prev Clip")
        self.next_clip_btn = QtWidgets.QPushButton("Next Clip >>")
        self.prev_frame_btn = QtWidgets.QPushButton("< Prev Frame")
        self.next_frame_btn = QtWidgets.QPushButton("Next Frame >")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        self.frame_input = QtWidgets.QLineEdit()
        self.frame_input.setFixedWidth(80)
        self.frame_input.setPlaceholderText("Frame")
        self.frame_go_btn = QtWidgets.QPushButton("Go")
        self.retrack_btn = QtWidgets.QPushButton("从此帧重跟踪")
        self.retrack_btn.setEnabled(ObjectTracker is not None)
        self.add_box_btn = QtWidgets.QPushButton("新增框")
        self.delete_box_btn = QtWidgets.QPushButton("删除框")

        if self.flagged_mode:
            self.save_window_btn = QtWidgets.QPushButton("保存窗口")
            self.save_window_btn.clicked.connect(self.save_clip_and_window)

            self.delete_frames_start_input = QtWidgets.QLineEdit()
            self.delete_frames_start_input.setFixedWidth(70)
            self.delete_frames_start_input.setPlaceholderText("start0")
            self.delete_frames_end_input = QtWidgets.QLineEdit()
            self.delete_frames_end_input.setFixedWidth(70)
            self.delete_frames_end_input.setPlaceholderText("end0")
            self.delete_frames_btn = QtWidgets.QPushButton("批量删除帧")
            self.delete_frames_btn.clicked.connect(self.delete_frames_range_0based)

        self.prev_clip_btn.clicked.connect(self.prev_clip)
        self.next_clip_btn.clicked.connect(self.next_clip)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.frame_go_btn.clicked.connect(self.jump_to_frame)
        self.frame_input.returnPressed.connect(self.jump_to_frame)
        self.fit_btn.clicked.connect(self.fit_view)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.retrack_btn.clicked.connect(self.retrack_from_current_frame)
        self.add_box_btn.clicked.connect(self.add_box_current_frame)
        self.delete_box_btn.clicked.connect(self.delete_selected_boxes_current_frame)

        controls.addWidget(self.prev_clip_btn)
        if self.flagged_mode:
            controls.addWidget(self.save_window_btn)
            controls.addWidget(QtWidgets.QLabel("删帧(0-based):"))
            controls.addWidget(self.delete_frames_start_input)
            controls.addWidget(self.delete_frames_end_input)
            controls.addWidget(self.delete_frames_btn)
        controls.addStretch(1)
        controls.addWidget(self.prev_frame_btn)
        controls.addWidget(self.fit_btn)
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.frame_input)
        controls.addWidget(self.frame_go_btn)
        controls.addWidget(self.next_frame_btn)
        controls.addStretch(1)
        controls.addWidget(self.add_box_btn)
        controls.addWidget(self.delete_box_btn)
        controls.addWidget(self.retrack_btn)
        controls.addWidget(self.next_clip_btn)

        layout.addLayout(controls)
        self.setCentralWidget(central)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.frame_view.setFocus()

        self.prev_frame_shortcut = QtGui.QShortcut(QtGui.QKeySequence("A"), self)
        self.prev_frame_shortcut.setContext(
            QtCore.Qt.ShortcutContext.ApplicationShortcut
        )
        self.prev_frame_shortcut.activated.connect(self.prev_frame)

        self.next_frame_shortcut = QtGui.QShortcut(QtGui.QKeySequence("D"), self)
        self.next_frame_shortcut.setContext(
            QtCore.Qt.ShortcutContext.ApplicationShortcut
        )
        self.next_frame_shortcut.activated.connect(self.next_frame)

    def _populate_flagged_list(self) -> None:
        if not getattr(self, "flagged_mode", False):
            return
        if not hasattr(self, "flagged_list"):
            return
        self.flagged_list.clear()
        for idx, entry in enumerate(self.clip_entries):
            labels = []
            if entry.retrack:
                labels.append("retrack")
            if entry.is_window_consistence is False:
                labels.append("window_inconsistent")
            flag_text = ",".join(labels) if labels else "-"
            text = (
                f"{entry.sport}/{entry.event}/{entry.clip_id} "
                f"[{entry.task_name}] {flag_text}"
            )
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, idx)
            self.flagged_list.addItem(item)
        if self.clip_entries:
            self.flagged_list.setCurrentRow(self.clip_index)

    def _handle_flagged_selection(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        idx_raw = item.data(QtCore.Qt.ItemDataRole.UserRole)
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            return
        if idx < 0 or idx >= len(self.clip_entries):
            return
        if idx == self.clip_index:
            return
        self._capture_current_frame()
        self._save_current_clip()
        self.clip_index = idx
        self._load_clip(self.clip_entries[self.clip_index])

    def log(self, message: str) -> None:
        self.log_box.append(message)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected_boxes_current_frame()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._capture_current_frame()
        self._save_current_clip()
        self._save_state()
        event.accept()

    def _save_state(self) -> None:
        self.state.clip_index = self.clip_index
        self.state.frame_index = self.frame_index
        self.state.save(self.state_path)

    def _discover_clips(self, flagged_only: bool = False) -> List[ClipEntry]:
        entries: List[ClipEntry] = []
        seen_keys: set[tuple[str, str, str, str]] = set()

        repo_root = self.dataset_root.parent.parent

        def sanitize_task_name_for_filename(task_name: str) -> str:
            # Most task names are already filename-safe (e.g. Spatial_Temporal_Grounding).
            # Keep this conservative: avoid path traversal and weird separators.
            return (
                str(task_name)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .strip("_")
            )

        def safe_load_json(path: Path) -> Optional[dict]:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

        def clip_requires_mot(
            output_path: Path,
        ) -> List[tuple[str, Path, int, Optional[bool], Optional[bool]]]:
            mot_entries: List[tuple[str, Path, int, Optional[bool], Optional[bool]]] = []
            output = safe_load_json(output_path)
            if output and isinstance(output, dict):
                clip_id = str(output_path.stem)
                mot_dir = output_path.parent / "mot"
                for idx, ann in enumerate(output.get("annotations", []) or []):
                    if not isinstance(ann, dict):
                        continue
                    tracking = ann.get("tracking_bboxes")
                    if not isinstance(tracking, dict):
                        continue
                    task_name = ann.get("task_L2", "")
                    retrack_flag = bool(ann.get("retrack", False))
                    window_flag_raw = ann.get("is_window_consistence")
                    window_flag = (
                        bool(window_flag_raw)
                        if isinstance(window_flag_raw, bool)
                        else None
                    )
                    if flagged_only and not (retrack_flag or window_flag is False):
                        continue

                    # Ignore JSON's tracking_bboxes.mot_file path. Always resolve MOT TXT
                    # from the sibling mot/ directory next to this clip JSON.
                    if not mot_dir.exists():
                        continue
                    safe_task = sanitize_task_name_for_filename(task_name or "tracking")
                    preferred = mot_dir / f"{clip_id}_{safe_task}.txt"
                    mot_path = None
                    if preferred.exists():
                        mot_path = preferred
                    else:
                        # Fallback: try to find a unique .txt with prefix clip_id_.
                        prefix = f"{clip_id}_"
                        candidates = [
                            p
                            for p in mot_dir.glob(f"{prefix}*.txt")
                            if p.is_file() and p.suffix == ".txt"
                        ]
                        if safe_task:
                            filtered = [p for p in candidates if safe_task in p.name]
                            if len(filtered) == 1:
                                mot_path = filtered[0]
                            elif len(filtered) > 1:
                                # Choose the shortest match to reduce ambiguity.
                                mot_path = sorted(filtered, key=lambda p: len(p.name))[0]
                        if mot_path is None and len(candidates) == 1:
                            mot_path = candidates[0]
                    if mot_path is None:
                        continue

                    mot_entries.append(
                        (
                            task_name or "tracking",
                            mot_path,
                            idx,
                            retrack_flag,
                            window_flag,
                        )
                    )
            return mot_entries

        # 以 output_root 为准遍历 clips JSON（retrack/is_window_consistence 标记来自 JSON）。
        # Dataset 仅用于定位对应视频文件。
        for sport_dir in self.output_root.iterdir():
            if not sport_dir.is_dir():
                continue
            for event_dir in sport_dir.iterdir():
                if not event_dir.is_dir():
                    continue
                clips_dir = event_dir / "clips"
                if not clips_dir.exists():
                    continue
                for output_path in clips_dir.glob("*.json"):
                    clip_id = output_path.stem
                    mot_entries = clip_requires_mot(output_path)
                    if not mot_entries:
                        continue

                    clip_path = (
                        self.dataset_root
                        / sport_dir.name
                        / event_dir.name
                        / "clips"
                        / f"{clip_id}.mp4"
                    )
                    # 没有对应视频则无法在 GUI 中展示，直接跳过。
                    if not clip_path.exists():
                        continue

                    for (
                        task_name,
                        mot_path,
                        ann_idx,
                        retrack_flag,
                        window_flag,
                    ) in mot_entries:
                        key = (sport_dir.name, event_dir.name, clip_id, task_name)
                        if key in seen_keys:
                            continue
                        entries.append(
                            ClipEntry(
                                sport_dir.name,
                                event_dir.name,
                                clip_id,
                                task_name,
                                clip_path,
                                mot_path,
                                output_path,
                                ann_idx,
                                retrack_flag,
                                window_flag,
                            )
                        )
                        seen_keys.add(key)

        entries.sort(
            key=lambda e: (
                e.sport,
                e.event,
                int(e.clip_id) if e.clip_id.isdigit() else e.clip_id,
                e.task_name,
            )
        )
        return entries

    def _load_clip(self, clip: ClipEntry) -> None:
        if self.video_reader is not None:
            self.video_reader.close()
        self.video_reader = None
        try:
            self.video_reader = OpenCVVideoReader(clip.video_path)
        except Exception as exc:
            self.log(f"Failed to open video with OpenCV: {exc}")
        self.log(f"Loading MOT file: {clip.mot_path}")
        if self.video_reader:
            self.total_frames = self.video_reader.frame_count
        else:
            self.total_frames = 1
        self._last_empty_notice = None
        self.frame_index = 1
        self.store = MotStore.load(clip.mot_path)
        self._update_mot_ranges_label()
        if self.store.frames:
            first_frame = min(self.store.frames.keys())
            if 1 <= first_frame <= self.total_frames:
                self.frame_index = first_frame
            else:
                self.frame_index = max(1, min(first_frame, self.total_frames))
                self.log(
                    f"MOT frame {first_frame} out of range; "
                    f"clamped to {self.frame_index}."
                )
        else:
            self.frame_index = max(1, min(self.frame_index, self.total_frames))
            self.log("No MOT boxes found for this clip.")
        self.frame_input.setValidator(QtGui.QIntValidator(1, self.total_frames))
        self.log(
            f"Loaded clip {clip.sport}/{clip.event}/{clip.clip_id} "
            f"[{clip.task_name}] ({self.total_frames} frames)"
        )
        self._load_review_flag(clip)
        self._update_annotation_panel(clip)
        self._render_frame()

    def _get_annotation(self, clip: ClipEntry) -> Optional[dict]:
        data = self._annotation_cache.get(clip.json_path)
        if data is None:
            try:
                data = json.loads(clip.json_path.read_text(encoding="utf-8"))
                self._annotation_cache[clip.json_path] = data
            except Exception as exc:
                self.log(f"读取 {clip.json_path} 失败：{exc}")
                return None
        anns = data.get("annotations", [])
        if not isinstance(anns, list) or clip.ann_index >= len(anns):
            return None
        ann = anns[clip.ann_index]
        if not isinstance(ann, dict):
            return None
        return ann

    def _update_annotation_panel(self, clip: ClipEntry) -> None:
        ann = self._get_annotation(clip)
        if ann is None:
            self.q_window_label.setText("Q_window_frame: -")
            self.a_window_label.setText("A_window_frame: -")
            self.answer_window_label.setText("answer_window: -")
            self.mot_ranges_label.setText("MOT 帧区间(有框): -")
            self.answer_box.setPlainText("")
            return

        q_window = ann.get("Q_window_frame")
        if isinstance(q_window, list) and len(q_window) == 2:
            self.q_window_label.setText(f"Q_window_frame: [{q_window[0]}, {q_window[1]}]")
        else:
            self.q_window_label.setText("Q_window_frame: -")

        a_window = ann.get("A_window_frame")
        if isinstance(a_window, list) and a_window:
            self.a_window_label.setText(f"A_window_frame: {a_window}")
        else:
            self.a_window_label.setText("A_window_frame: -")

        answer_window = ann.get("answer_window")
        if isinstance(answer_window, list) and answer_window:
            self.answer_window_label.setText(f"answer_window: {answer_window}")
        else:
            self.answer_window_label.setText("answer_window: -")

        self._update_mot_ranges_label()

        answer = ann.get("answer")
        if isinstance(answer, list):
            lines = [f"{i + 1}. {str(v)}" for i, v in enumerate(answer)]
            self.answer_box.setPlainText("\n".join(lines))
        elif isinstance(answer, str):
            self.answer_box.setPlainText(answer)
        else:
            self.answer_box.setPlainText("")

    def _update_mot_ranges_label(self) -> None:
        if not hasattr(self, "mot_ranges_label"):
            return
        frames_with_boxes = [
            int(frame)
            for frame, boxes in (self.store.frames or {}).items()
            if boxes
        ]
        frames_with_boxes.sort()
        if not frames_with_boxes:
            self.mot_ranges_label.setText("MOT 帧区间(有框): -")
            return

        ranges: List[tuple[int, int]] = []
        start = prev = frames_with_boxes[0]
        for f in frames_with_boxes[1:]:
            if f == prev + 1:
                prev = f
                continue
            ranges.append((start, prev))
            start = prev = f
        ranges.append((start, prev))

        parts = []
        for a, b in ranges:
            a0 = max(0, a - 1)
            b0 = max(0, b - 1)
            parts.append(str(a0) if a0 == b0 else f"{a0}-{b0}")
        self.mot_ranges_label.setText(
            f"MOT 帧区间(有框, 0-based): {', '.join(parts)}"
        )

    def _load_review_flag(self, clip: ClipEntry) -> None:
        try:
            data = json.loads(clip.json_path.read_text(encoding="utf-8"))
            anns = data.get("annotations", [])
            if not isinstance(anns, list):
                return
            if clip.ann_index >= len(anns):
                return
            ann = anns[clip.ann_index]
            if not isinstance(ann, dict):
                return
            self.reviewed = bool(ann.get("reviewed", False))
            self.review_checkbox.setChecked(self.reviewed)
        except Exception as exc:
            self.log(f"Failed to load reviewed flag: {exc}")

    def _save_review_flag(self, clip: ClipEntry) -> None:
        try:
            data = json.loads(clip.json_path.read_text(encoding="utf-8"))
            anns = data.get("annotations", [])
            if not isinstance(anns, list) or clip.ann_index >= len(anns):
                return
            ann = anns[clip.ann_index]
            if not isinstance(ann, dict):
                return
            ann["reviewed"] = bool(self.review_checkbox.isChecked())
            data["annotations"] = anns
            clip.json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            self.log(f"Failed to save reviewed flag: {exc}")

    def _capture_current_frame(self) -> None:
        if not self.clip_entries:
            return
        boxes = self.frame_view.sync_boxes()
        current_frame = self.frame_index
        for box in boxes:
            box.frame = current_frame
        self.store.set_frame(current_frame, boxes)
        self.reviewed = bool(self.review_checkbox.isChecked())

    def _save_current_clip(self) -> None:
        if not self.clip_entries:
            return
        current_clip = self.clip_entries[self.clip_index]
        self.store.save(current_clip.mot_path)
        self._save_review_flag(current_clip)

    def save_clip_and_window(self) -> None:
        if not self.clip_entries:
            return
        self._capture_current_frame()
        self._save_current_clip()
        clip = self.clip_entries[self.clip_index]

        # 注意：该按钮不会改动 JSON 的窗口范围（A_window_frame/answer_window）。
        # 仅把 first_bounding_box 同步为“窗口起始帧”对应的 MOT 框坐标。
        try:
            data = json.loads(clip.json_path.read_text(encoding="utf-8"))
            anns = data.get("annotations", [])
            if not isinstance(anns, list) or clip.ann_index >= len(anns):
                self.log("JSON annotations 无效，未更新 first_bounding_box。")
                return
            ann = anns[clip.ann_index]
            if not isinstance(ann, dict):
                self.log("目标 annotation 不是字典，未更新 first_bounding_box。")
                return

            window = self._get_annotation_window(clip)
            if window is None:
                self.log("无法解析任务窗口，未更新 first_bounding_box。")
                return
            window_start_0, _window_end_0 = window
            target_frame_1 = max(1, int(window_start_0) + 1)

            frame_boxes = self.store.get_frame(target_frame_1)
            if not frame_boxes:
                self.log(
                    f"窗口起始帧 {target_frame_1} 没有框，未更新 first_bounding_box。"
                )
                return

            existing = ann.get("first_bounding_box")
            existing_box = None
            if isinstance(existing, list) and len(existing) == 4:
                try:
                    x1, y1, x2, y2 = map(float, existing)
                    existing_box = (x1, y1, x2, y2)
                except Exception:
                    existing_box = None

            def to_xyxy(b: MotBox) -> tuple[float, float, float, float]:
                x1 = float(b.left)
                y1 = float(b.top)
                x2 = x1 + float(b.width)
                y2 = y1 + float(b.height)
                return x1, y1, x2, y2

            def center_dist2(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
                ax = (a[0] + a[2]) / 2.0
                ay = (a[1] + a[3]) / 2.0
                bx = (b[0] + b[2]) / 2.0
                by = (b[1] + b[3]) / 2.0
                dx = ax - bx
                dy = ay - by
                return dx * dx + dy * dy

            chosen = frame_boxes[0]
            if existing_box is not None:
                best_d2 = None
                for cand in frame_boxes:
                    d2 = center_dist2(existing_box, to_xyxy(cand))
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        chosen = cand
            else:
                chosen = min(frame_boxes, key=lambda b: b.track_id)

            x1, y1, x2, y2 = to_xyxy(chosen)
            ann["first_bounding_box"] = [x1, y1, x2, y2]
            data["annotations"] = anns
            clip.json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.log(
                f"已保存 MOT，并将 first_bounding_box 同步到窗口起始帧 {target_frame_1} (txt 帧从1开始, JSON 从0开始)"
            )
        except Exception as exc:
            self.log(f"保存 first_bounding_box 失败: {exc}")

    def _render_frame(self) -> None:
        if not self.video_reader:
            return
        current_clip = self.clip_entries[self.clip_index]
        frame = self._read_frame(self.frame_index)
        if frame is None:
            self.log("Failed to read frame.")
            self.frame_view.scene().clear()
            self.frame_view.box_items = []
            self.statusBar().showMessage(
                f"Clip {current_clip.clip_id} [{current_clip.task_name}] "
                f"Frame {max(0, self.frame_index - 1)}/{max(0, self.total_frames - 1)} (read failed)"
            )
            return
        frame_rgb = frame
        h, w, _ = frame_rgb.shape
        image = QtGui.QImage(
            frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888
        )
        boxes = self.store.get_frame(self.frame_index)
        if not boxes and self._last_empty_notice != self.frame_index:
            self.log(f"No boxes for frame {self.frame_index}.")
            self._last_empty_notice = self.frame_index
        self.frame_view.set_frame(image, boxes)
        self.statusBar().showMessage(
            f"Clip {self.clip_entries[self.clip_index].clip_id} "
            f"[{self.clip_entries[self.clip_index].task_name}] "
            f"Frame {max(0, self.frame_index - 1)}/{max(0, self.total_frames - 1)}"
        )
        self._update_mot_ranges_label()

    def delete_frames_range_0based(self) -> None:
        if not getattr(self, "flagged_mode", False):
            return
        if not self.clip_entries:
            return
        if not hasattr(self, "delete_frames_start_input") or not hasattr(
            self, "delete_frames_end_input"
        ):
            return

        start_raw = self.delete_frames_start_input.text().strip()
        end_raw = self.delete_frames_end_input.text().strip()
        try:
            start0 = int(start_raw)
            end0 = int(end_raw)
        except Exception:
            self.log("批量删除帧：请输入合法的 start0/end0 整数。")
            return

        if start0 < 0 or end0 < 0:
            self.log("批量删除帧：start0/end0 需为非负整数（0-based）。")
            return
        if end0 < start0:
            self.log("批量删除帧：end0 需 >= start0（两端都包含）。")
            return

        max0 = max(0, int(self.total_frames) - 1)
        if start0 > max0 or end0 > max0:
            self.log(f"批量删除帧：范围超出视频长度（允许 0..{max0}）。")
            return

        # Capture current edits first, then delete the requested range.
        self._capture_current_frame()

        start1 = start0 + 1
        end1 = end0 + 1
        deleted_keys = 0
        for frame_1 in range(start1, end1 + 1):
            if frame_1 in self.store.frames:
                deleted_keys += 1
            self.store.frames.pop(frame_1, None)

        # Persist immediately; user expects the button to actually delete the frames.
        self._save_current_clip()

        self.log(
            f"已批量删除 MOT 帧区间(0-based) [{start0}, {end0}]（含两端），"
            f"清理了 {deleted_keys} 个有框帧。"
        )
        self._render_frame()

    def add_box_current_frame(self) -> None:
        if not self.video_reader:
            self.log("当前没有加载视频，无法新增框。")
            return
        frame_rgb = self._read_frame(self.frame_index)
        if frame_rgb is None:
            self.log("读取当前帧失败，无法新增框。")
            return
        h, w, _ = frame_rgb.shape
        boxes = self.frame_view.sync_boxes()
        next_track_id = max([b.track_id for b in boxes], default=0) + 1
        box_w = max(20.0, w * 0.15)
        box_h = max(20.0, h * 0.15)
        left = max(0.0, (w - box_w) / 2.0)
        top = max(0.0, (h - box_h) / 2.0)
        new_box = MotBox(
            frame=self.frame_index,
            track_id=next_track_id,
            left=left,
            top=top,
            width=box_w,
            height=box_h,
        )
        boxes.append(new_box)
        self.store.set_frame(self.frame_index, boxes)
        self.log(
            f"新增框 track_id={next_track_id} 于帧 {self.frame_index}，可拖动调整后再保存。"
        )
        self._render_frame()

    def delete_selected_boxes_current_frame(self) -> None:
        if not self.video_reader:
            return
        selected = [
            item
            for item in self.frame_view.scene().selectedItems()
            if isinstance(item, BoxItem)
        ]
        if not selected:
            self.log("未选中任何框（点击框后再删除）。")
            return
        boxes = self.frame_view.sync_boxes()
        selected_ids = {id(item.box) for item in selected}
        remaining = [b for b in boxes if id(b) not in selected_ids]
        deleted = len(boxes) - len(remaining)
        self.store.set_frame(self.frame_index, remaining)
        self.log(f"已删除当前帧 {deleted} 个框。")
        self._render_frame()

    def _read_frame(self, frame_index: int):
        if not self.video_reader:
            return None
        target = max(1, min(frame_index, self.total_frames))
        try:
            return self.video_reader.read_rgb(target - 1)
        except Exception:
            return None

    def prev_frame(self) -> None:
        if self.frame_index <= 1:
            return
        self._capture_current_frame()
        self.frame_index -= 1
        self._render_frame()

    def next_frame(self) -> None:
        if self.frame_index >= self.total_frames:
            return
        self._capture_current_frame()
        self.frame_index += 1
        self._render_frame()

    def fit_view(self) -> None:
        self.frame_view.resetTransform()
        self.frame_view.set_fit_mode(True)
        self._render_frame()

    def zoom_in(self) -> None:
        self.frame_view.zoom(1.1)

    def zoom_out(self) -> None:
        self.frame_view.zoom(0.9)

    def jump_to_frame(self) -> None:
        text = self.frame_input.text().strip()
        if not text:
            return
        try:
            target = int(text)
        except ValueError:
            self.log(f"Invalid frame: {text}")
            return
        if target < 1 or target > self.total_frames:
            self.log(f"Frame {target} out of range (1-{self.total_frames}).")
            return
        self._capture_current_frame()
        self.frame_index = target
        self._render_frame()

    def prev_clip(self) -> None:
        if self.clip_index <= 0:
            return
        self._capture_current_frame()
        self._save_current_clip()
        self.clip_index -= 1
        self._load_clip(self.clip_entries[self.clip_index])

    def next_clip(self) -> None:
        if self.clip_index >= len(self.clip_entries) - 1:
            return
        self._capture_current_frame()
        self._save_current_clip()
        self.clip_index += 1
        self._load_clip(self.clip_entries[self.clip_index])

    def retrack_from_current_frame(self) -> None:
        if ObjectTracker is None:
            self.log("SAM2 跟踪依赖未就绪，无法重新跟踪。")
            return
        if self._retrack_worker is not None:
            self.log("已有重跟踪任务运行中，请稍候。")
            return
        clip = self.clip_entries[self.clip_index]
        boxes = self.frame_view.sync_boxes()
        if not boxes:
            self.log("当前帧没有可用框，无法作为重新跟踪的起点。")
            return
        window = self._get_annotation_window(clip)
        if window is None:
            self.log("无法解析该任务的时间窗口，取消重新跟踪。")
            return
        window_start, window_end = window
        window_end = min(window_end, max(0, self.total_frames - 1))
        start_frame = max(window_start, self.frame_index - 1)
        if start_frame > window_end:
            self.log(
                f"当前帧 {self.frame_index} 超过任务窗口 (1-{window_end + 1})，取消重新跟踪。"
            )
            return
        initial_bboxes: List[dict] = []
        track_id_map: Dict[int, int] = {}
        object_idx = 0
        for box in boxes:
            xtl = float(box.left)
            ytl = float(box.top)
            xbr = xtl + float(box.width)
            ybr = ytl + float(box.height)
            if xbr <= xtl or ybr <= ytl:
                continue
            initial_bboxes.append({
                "bbox": [xtl, ytl, xbr, ybr],
                "label": f"track-{box.track_id}",
            })
            track_id_map[object_idx] = box.track_id
            object_idx += 1
        if not initial_bboxes:
            self.log("当前帧的框尺寸无效，无法重新跟踪。")
            return
        clip_key = self._clip_key(clip)
        self._retrack_worker = RetrackWorker(
            video_path=clip.video_path,
            clip_key=clip_key,
            start_frame=start_frame,
            end_frame=window_end,
            first_bboxes=initial_bboxes,
            track_id_map=track_id_map,
        )
        self._retrack_worker.result_ready.connect(self._handle_retrack_result)
        self._retrack_worker.error.connect(self._handle_retrack_error)
        self._retrack_worker.finished.connect(self._clear_retrack_worker)
        self.retrack_btn.setEnabled(False)
        self.log(
            f"开始重新跟踪：帧 {start_frame + 1}-{window_end + 1} ({clip.clip_id}). 这可能需要一些时间。"
        )
        self._retrack_worker.start()

    def _handle_retrack_result(self, payload: dict) -> None:
        clip = self.clip_entries[self.clip_index]
        payload_key_raw = payload.get("clip_key")
        if isinstance(payload_key_raw, (list, tuple)):
            payload_key = tuple(payload_key_raw)
        else:
            payload_key = None
        if payload_key != self._clip_key(clip):
            self.log("收到的跟踪结果与当前 clip 不匹配，已忽略。")
            return
        objects = payload.get("objects", [])
        start_frame = int(payload.get("start_frame", 0))
        end_frame = int(payload.get("end_frame", -1))
        track_id_map = payload.get("track_id_map", {})
        if end_frame < start_frame:
            self.log("跟踪结果帧范围无效，已忽略。")
            return
        updated_frames = 0
        for frame_idx in range(start_frame, end_frame + 1):
            mot_frame = frame_idx + 1
            if mot_frame > self.total_frames:
                break
            boxes: List[MotBox] = []
            for obj in objects:
                obj_id = obj.get("id")
                frames = obj.get("frames", {})
                bbox = frames.get(frame_idx) or frames.get(str(frame_idx))
                if not bbox or len(bbox) != 4:
                    continue
                left, top, right, bottom = map(float, bbox)
                width = max(0.0, right - left)
                height = max(0.0, bottom - top)
                if width <= 0 or height <= 0:
                    continue
                track_id = track_id_map.get(obj_id, obj_id + 1)
                boxes.append(
                    MotBox(
                        frame=mot_frame,
                        track_id=track_id,
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                    )
                )
            self.store.set_frame(mot_frame, boxes)
            if boxes:
                updated_frames += 1
        if updated_frames == 0:
            self.log("跟踪完成，但没有生成可用的框。")
        else:
            self.log(
                f"重新跟踪完成，更新 {updated_frames} 帧。记得检查结果并保存。"
            )
            self._render_frame()
            self._save_current_clip()

    def _handle_retrack_error(self, message: str) -> None:
        self.log(f"重新跟踪失败：{message}")

    def _clear_retrack_worker(self) -> None:
        self.retrack_btn.setEnabled(ObjectTracker is not None)
        self._retrack_worker = None

    def _clip_key(self, clip: ClipEntry) -> Tuple[str, str, str, str]:
        return (clip.sport, clip.event, clip.clip_id, clip.task_name)

    def _get_annotation_window(self, clip: ClipEntry) -> Optional[Tuple[int, int]]:
        ann = self._get_annotation(clip)
        if ann is None:
            return None
        q_window = ann.get("Q_window_frame")
        if isinstance(q_window, list) and len(q_window) == 2:
            try:
                start = int(q_window[0])
                end = int(q_window[1])
                return max(0, start), max(start, end)
            except (TypeError, ValueError):
                pass
        a_window = ann.get("A_window_frame")
        # 常见格式：A_window_frame = [start, end]（两个数字表示闭区间）。
        if (
            isinstance(a_window, list)
            and len(a_window) == 2
            and isinstance(a_window[0], (int, float))
            and isinstance(a_window[1], (int, float))
        ):
            start = int(a_window[0])
            end = int(a_window[1])
            return max(0, start), max(start, end)

        if isinstance(a_window, list) and a_window:
            min_start: Optional[int] = None
            max_end: Optional[int] = None
            for entry in a_window:
                if not isinstance(entry, str) or "-" not in entry:
                    continue
                try:
                    start_str, end_str = entry.split("-", 1)
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    continue
                min_start = start if min_start is None else min(min_start, start)
                max_end = end if max_end is None else max(max_end, end)
            if min_start is not None and max_end is not None:
                return max(0, min_start), max(min_start, max_end)
        return 0, max(0, self.total_frames - 1)


def run_app(
    dataset_root: Path, output_root: Path, state_path: Path, flagged_mode: bool = False
) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MotEditorWindow(dataset_root, output_root, state_path, flagged_mode)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())


class RetrackWorker(QtCore.QThread):
    result_ready = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(
        self,
        video_path: Path,
        clip_key: Tuple[str, str, str, str],
        start_frame: int,
        end_frame: int,
        first_bboxes: List[dict],
        track_id_map: Dict[int, int],
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.clip_key = clip_key
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.first_bboxes = first_bboxes
        self.track_id_map = track_id_map

    def run(self) -> None:
        try:
            tracker = ObjectTracker() if ObjectTracker else None
            if tracker is None:
                raise RuntimeError("ObjectTracker 未初始化")
            result = tracker.track_from_first_bbox(
                video_path=self.video_path,
                first_bboxes_with_label=self.first_bboxes,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            payload = {
                "clip_key": self.clip_key,
                "objects": result.objects,
                "start_frame": result.start_frame,
                "end_frame": result.end_frame,
                "track_id_map": self.track_id_map,
            }
            self.result_ready.emit(payload)
        except Exception as exc:  # pragma: no cover - UI thread logs error
            self.error.emit(str(exc))
