"""GUI for editing MOT boxes."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .mot_io import MotBox, MotStore
from .state import EditorState


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


class OpenCVVideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

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
        self.duration_sec: Optional[float] = None
        if self.fps > 0:
            self.duration_sec = self.frame_count / self.fps
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
    def __init__(self, dataset_root: Path, output_root: Path, state_path: Path):
        super().__init__()
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.state_path = state_path
        self.state = EditorState.load(state_path)

        self.clip_entries = self._discover_clips()
        if not self.clip_entries:
            raise RuntimeError("No clips found in dataset.")

        self._selector_index = self._build_selector_index()
        self._updating_selectors = False
        self.clip_index = max(0, min(self.state.clip_index, len(self.clip_entries) - 1))
        self.frame_index = max(1, self.state.frame_index)

        self.store = MotStore()
        self.video_reader: Optional[OpenCVVideoReader] = None
        self.total_frames = 1
        self._last_empty_notice: Optional[int] = None
        self.reviewed = False

        self._build_ui()
        self._load_clip(self.clip_entries[self.clip_index])

    def _build_ui(self) -> None:
        self.setWindowTitle("BBoxFixer")
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(520)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        selector_group = QtWidgets.QGroupBox("Selection")
        selector_form = QtWidgets.QFormLayout(selector_group)
        self.sport_combo = QtWidgets.QComboBox()
        self.event_combo = QtWidgets.QComboBox()
        self.clip_combo = QtWidgets.QComboBox()
        self.annotation_combo = QtWidgets.QComboBox()
        selector_form.addRow("Sport", self.sport_combo)
        selector_form.addRow("Event", self.event_combo)
        selector_form.addRow("Clip", self.clip_combo)
        selector_form.addRow("Annotation", self.annotation_combo)
        left_layout.addWidget(selector_group)

        info_group = QtWidgets.QGroupBox("Clip Info")
        info_form = QtWidgets.QFormLayout(info_group)
        self.video_length_value = QtWidgets.QLabel("-")
        self.tracking_length_value = QtWidgets.QLabel("-")
        self.tracking_start_value = QtWidgets.QLabel("-")
        self.tracking_end_value = QtWidgets.QLabel("-")
        info_form.addRow("Video Length", self.video_length_value)
        info_form.addRow("Tracking Length", self.tracking_length_value)
        info_form.addRow("Tracking Start", self.tracking_start_value)
        info_form.addRow("Tracking End", self.tracking_end_value)
        left_layout.addWidget(info_group)

        self.review_checkbox = QtWidgets.QCheckBox("Reviewed")
        left_layout.addWidget(self.review_checkbox)

        controls_group = QtWidgets.QGroupBox("Frame Controls")
        controls = QtWidgets.QGridLayout(controls_group)
        self.prev_frame_btn = QtWidgets.QPushButton("< Prev Frame")
        self.next_frame_btn = QtWidgets.QPushButton("Next Frame >")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        self.frame_input = QtWidgets.QLineEdit()
        self.frame_input.setFixedWidth(80)
        self.frame_input.setPlaceholderText("Frame")
        self.frame_go_btn = QtWidgets.QPushButton("Go")

        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.frame_go_btn.clicked.connect(self.jump_to_frame)
        self.frame_input.returnPressed.connect(self.jump_to_frame)
        self.fit_btn.clicked.connect(self.fit_view)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.sport_combo.currentIndexChanged.connect(self._on_sport_changed)
        self.event_combo.currentIndexChanged.connect(self._on_event_changed)
        self.clip_combo.currentIndexChanged.connect(self._on_clip_changed)
        self.annotation_combo.currentIndexChanged.connect(self._on_annotation_changed)

        controls.addWidget(self.prev_frame_btn, 0, 0)
        controls.addWidget(self.next_frame_btn, 0, 1)
        controls.addWidget(self.fit_btn, 1, 0)
        controls.addWidget(self.zoom_out_btn, 1, 1)
        controls.addWidget(self.zoom_in_btn, 1, 2)
        controls.addWidget(self.frame_input, 2, 0)
        controls.addWidget(self.frame_go_btn, 2, 1)
        left_layout.addWidget(controls_group)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        left_layout.addWidget(self.log_box, stretch=1)

        layout.addWidget(left_panel, stretch=0)

        self.frame_view = FrameView()
        layout.addWidget(self.frame_view, stretch=1)

        self.setCentralWidget(central)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.frame_view.setFocus()

        self._sync_selectors_to_entry(self.clip_entries[self.clip_index])

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

    def log(self, message: str) -> None:
        self.log_box.append(message)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
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

    @staticmethod
    def _clip_sort_key(clip_id: str):
        return (0, int(clip_id)) if clip_id.isdigit() else (1, clip_id)

    def _build_selector_index(self) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
        index: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
        for idx, entry in enumerate(self.clip_entries):
            sport_map = index.setdefault(entry.sport, {})
            event_map = sport_map.setdefault(entry.event, {})
            event_map.setdefault(entry.clip_id, []).append(idx)
        return index

    def _populate_events(self, sport: str, preferred_event: Optional[str] = None) -> None:
        events = sorted(self._selector_index.get(sport, {}).keys())
        self.event_combo.clear()
        for event in events:
            self.event_combo.addItem(event, event)
        if not events:
            return
        target = preferred_event if preferred_event in events else events[0]
        idx = self.event_combo.findData(target)
        if idx >= 0:
            self.event_combo.setCurrentIndex(idx)

    def _populate_clips(
        self, sport: str, event: str, preferred_clip_id: Optional[str] = None
    ) -> None:
        clip_ids = sorted(
            self._selector_index.get(sport, {}).get(event, {}).keys(),
            key=self._clip_sort_key,
        )
        self.clip_combo.clear()
        for clip_id in clip_ids:
            self.clip_combo.addItem(clip_id, clip_id)
        if not clip_ids:
            return
        target = preferred_clip_id if preferred_clip_id in clip_ids else clip_ids[0]
        idx = self.clip_combo.findData(target)
        if idx >= 0:
            self.clip_combo.setCurrentIndex(idx)

    def _populate_annotations(
        self,
        sport: str,
        event: str,
        clip_id: str,
        preferred_entry_index: Optional[int] = None,
    ) -> None:
        entry_indexes = list(
            self._selector_index.get(sport, {}).get(event, {}).get(clip_id, [])
        )
        entry_indexes.sort(
            key=lambda i: (self.clip_entries[i].ann_index, self.clip_entries[i].task_name)
        )
        self.annotation_combo.clear()
        for entry_index in entry_indexes:
            entry = self.clip_entries[entry_index]
            label = f"{entry.task_name} (ann {entry.ann_index})"
            self.annotation_combo.addItem(label, entry_index)
        if not entry_indexes:
            return
        target = (
            preferred_entry_index
            if preferred_entry_index in entry_indexes
            else entry_indexes[0]
        )
        idx = self.annotation_combo.findData(target)
        if idx >= 0:
            self.annotation_combo.setCurrentIndex(idx)

    def _sync_selectors_to_entry(self, entry: ClipEntry) -> None:
        self._updating_selectors = True
        try:
            sports = sorted(self._selector_index.keys())
            self.sport_combo.clear()
            for sport in sports:
                self.sport_combo.addItem(sport, sport)
            sport_idx = self.sport_combo.findData(entry.sport)
            if sport_idx < 0 and sports:
                sport_idx = 0
            if sport_idx >= 0:
                self.sport_combo.setCurrentIndex(sport_idx)

            sport = self.sport_combo.currentData()
            if not isinstance(sport, str):
                return

            self._populate_events(sport, preferred_event=entry.event)
            event = self.event_combo.currentData()
            if not isinstance(event, str):
                return

            self._populate_clips(sport, event, preferred_clip_id=entry.clip_id)
            clip_id = self.clip_combo.currentData()
            if not isinstance(clip_id, str):
                return

            self._populate_annotations(
                sport,
                event,
                clip_id,
                preferred_entry_index=self.clip_index,
            )
        finally:
            self._updating_selectors = False

    def _selected_entry_index(self) -> Optional[int]:
        data = self.annotation_combo.currentData()
        if isinstance(data, int) and 0 <= data < len(self.clip_entries):
            return data
        return None

    def _switch_to_entry_index(self, entry_index: int) -> None:
        if entry_index == self.clip_index and self.video_reader is not None:
            return
        if self.video_reader is not None:
            self._capture_current_frame()
            self._save_current_clip()
        self.clip_index = entry_index
        self._load_clip(self.clip_entries[self.clip_index])
        self._sync_selectors_to_entry(self.clip_entries[self.clip_index])
        self._save_state()

    def _switch_to_current_selection(self) -> None:
        entry_index = self._selected_entry_index()
        if entry_index is None:
            return
        self._switch_to_entry_index(entry_index)

    def _on_sport_changed(self, _index: int = -1) -> None:
        if self._updating_selectors:
            return
        sport = self.sport_combo.currentData()
        if not isinstance(sport, str):
            return
        self._updating_selectors = True
        try:
            self._populate_events(sport)
            event = self.event_combo.currentData()
            if not isinstance(event, str):
                return
            self._populate_clips(sport, event)
            clip_id = self.clip_combo.currentData()
            if not isinstance(clip_id, str):
                return
            self._populate_annotations(sport, event, clip_id)
        finally:
            self._updating_selectors = False
        self._switch_to_current_selection()

    def _on_event_changed(self, _index: int = -1) -> None:
        if self._updating_selectors:
            return
        sport = self.sport_combo.currentData()
        event = self.event_combo.currentData()
        if not isinstance(sport, str) or not isinstance(event, str):
            return
        self._updating_selectors = True
        try:
            self._populate_clips(sport, event)
            clip_id = self.clip_combo.currentData()
            if not isinstance(clip_id, str):
                return
            self._populate_annotations(sport, event, clip_id)
        finally:
            self._updating_selectors = False
        self._switch_to_current_selection()

    def _on_clip_changed(self, _index: int = -1) -> None:
        if self._updating_selectors:
            return
        sport = self.sport_combo.currentData()
        event = self.event_combo.currentData()
        clip_id = self.clip_combo.currentData()
        if (
            not isinstance(sport, str)
            or not isinstance(event, str)
            or not isinstance(clip_id, str)
        ):
            return
        self._updating_selectors = True
        try:
            self._populate_annotations(sport, event, clip_id)
        finally:
            self._updating_selectors = False
        self._switch_to_current_selection()

    def _on_annotation_changed(self, _index: int = -1) -> None:
        if self._updating_selectors:
            return
        self._switch_to_current_selection()

    def _discover_clips(self) -> List[ClipEntry]:
        entries: List[ClipEntry] = []
        seen_keys: set[tuple[str, str, str, str, int]] = set()

        def safe_load_json(path: Path) -> Optional[dict]:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

        def clip_requires_mot(output_path: Path) -> List[tuple[str, Path, int]]:
            mot_entries: List[tuple[str, Path, int]] = []
            output = safe_load_json(output_path)
            if output and isinstance(output, dict):
                for idx, ann in enumerate(output.get("annotations", []) or []):
                    if not isinstance(ann, dict):
                        continue
                    tracking = ann.get("tracking_bboxes")
                    if not isinstance(tracking, dict):
                        continue
                    mot_ref = tracking.get("mot_file")
                    task_name = ann.get("task_L2", "")
                    if mot_ref:
                        mot_entries.append(
                            (task_name or "tracking", Path(str(mot_ref)), idx)
                        )
            return mot_entries

        for sport_dir in self.dataset_root.iterdir():
            if not sport_dir.is_dir():
                continue
            for event_dir in sport_dir.iterdir():
                if not event_dir.is_dir():
                    continue
                clips_dir = event_dir / "clips"
                if not clips_dir.exists():
                    continue
                for clip_path in clips_dir.glob("*.mp4"):
                    clip_id = clip_path.stem
                    output_path = (
                        self.output_root
                        / sport_dir.name
                        / event_dir.name
                        / "clips"
                        / f"{clip_id}.json"
                    )
                    mot_entries = clip_requires_mot(output_path)
                    if not mot_entries:
                        continue
                    for task_name, mot_path, ann_idx in mot_entries:
                        key = (sport_dir.name, event_dir.name, clip_id, task_name, ann_idx)
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
                            )
                        )
                        seen_keys.add(key)

        entries.sort(
            key=lambda e: (
                e.sport,
                e.event,
                self._clip_sort_key(e.clip_id),
                e.task_name,
                e.ann_index,
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
        self._update_clip_info()
        self._render_frame()

    def _update_clip_info(self) -> None:
        if self.video_reader and self.video_reader.fps > 0:
            self.video_length_value.setText(
                f"{self.total_frames} frames ({self.video_reader.duration_sec:.2f}s @ {self.video_reader.fps:.2f} FPS)"
            )
        else:
            self.video_length_value.setText(f"{self.total_frames} frames (FPS unknown)")

        frames_with_boxes = sorted(
            frame for frame, boxes in self.store.frames.items() if boxes
        )
        if not frames_with_boxes:
            self.tracking_length_value.setText("0 frames")
            self.tracking_start_value.setText("-")
            self.tracking_end_value.setText("-")
            return

        start_frame = frames_with_boxes[0]
        end_frame = frames_with_boxes[-1]
        span_len = end_frame - start_frame + 1
        boxed_frames = len(frames_with_boxes)
        self.tracking_length_value.setText(
            f"{span_len} frames (boxed frames: {boxed_frames})"
        )
        self.tracking_start_value.setText(str(start_frame))
        self.tracking_end_value.setText(str(end_frame))

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
        self._update_clip_info()

    def _save_current_clip(self) -> None:
        if not self.clip_entries:
            return
        current_clip = self.clip_entries[self.clip_index]
        self.store.save(current_clip.mot_path)
        self._save_review_flag(current_clip)

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
                f"Frame {self.frame_index}/{self.total_frames} (read failed)"
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
            f"Frame {self.frame_index}/{self.total_frames}"
        )

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


def run_app(dataset_root: Path, output_root: Path, state_path: Path) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MotEditorWindow(dataset_root, output_root, state_path)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
