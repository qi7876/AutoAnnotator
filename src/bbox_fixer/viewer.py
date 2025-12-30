"""GUI for editing MOT boxes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .mot_io import MotBox, MotStore
from .state import EditorState


@dataclass
class ClipEntry:
    sport: str
    event: str
    clip_id: str
    video_path: Path
    mot_path: Path


class HandleItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent: QtWidgets.QGraphicsItem, corner: str):
        super().__init__(-4, -4, 8, 8, parent)
        self.corner = corner
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)


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
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._sync_handles()
        return super().itemChange(change, value)


class FrameView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene())
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.box_items: List[BoxItem] = []

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

    def sync_boxes(self) -> List[MotBox]:
        for item in self.box_items:
            item.update_from_handles()
        return [item.box for item in self.box_items]


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

        self.clip_index = max(0, min(self.state.clip_index, len(self.clip_entries) - 1))
        self.frame_index = max(1, self.state.frame_index)

        self.store = MotStore()
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 1
        self._last_empty_notice: Optional[int] = None

        self._build_ui()
        self._load_clip(self.clip_entries[self.clip_index])

    def _build_ui(self) -> None:
        self.setWindowTitle("BBoxFixer")
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.frame_view = FrameView()
        layout.addWidget(self.frame_view, stretch=1)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        layout.addWidget(self.log_box)

        controls = QtWidgets.QHBoxLayout()
        self.prev_clip_btn = QtWidgets.QPushButton("<< Prev Clip")
        self.next_clip_btn = QtWidgets.QPushButton("Next Clip >>")
        self.prev_frame_btn = QtWidgets.QPushButton("< Prev Frame")
        self.next_frame_btn = QtWidgets.QPushButton("Next Frame >")
        self.frame_input = QtWidgets.QLineEdit()
        self.frame_input.setFixedWidth(80)
        self.frame_input.setPlaceholderText("Frame")
        self.frame_go_btn = QtWidgets.QPushButton("Go")

        self.prev_clip_btn.clicked.connect(self.prev_clip)
        self.next_clip_btn.clicked.connect(self.next_clip)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.frame_go_btn.clicked.connect(self.jump_to_frame)
        self.frame_input.returnPressed.connect(self.jump_to_frame)

        controls.addWidget(self.prev_clip_btn)
        controls.addStretch(1)
        controls.addWidget(self.prev_frame_btn)
        controls.addWidget(self.frame_input)
        controls.addWidget(self.frame_go_btn)
        controls.addWidget(self.next_frame_btn)
        controls.addStretch(1)
        controls.addWidget(self.next_clip_btn)

        layout.addLayout(controls)
        self.setCentralWidget(central)

    def log(self, message: str) -> None:
        self.log_box.append(message)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Left:
            self.prev_frame()
        elif event.key() == QtCore.Qt.Key_Right:
            self.next_frame()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_current_frame()
        self._save_state()
        event.accept()

    def _save_state(self) -> None:
        self.state.clip_index = self.clip_index
        self.state.frame_index = self.frame_index
        self.state.save(self.state_path)

    def _discover_clips(self) -> List[ClipEntry]:
        entries: List[ClipEntry] = []
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
                    mot_path = (
                        self.output_root
                        / sport_dir.name
                        / event_dir.name
                        / "clips"
                        / "mot"
                        / f"{clip_id}.txt"
                    )
                    entries.append(
                        ClipEntry(
                            sport_dir.name,
                            event_dir.name,
                            clip_id,
                            clip_path,
                            mot_path,
                        )
                    )
        entries.sort(key=lambda e: (e.sport, e.event, int(e.clip_id) if e.clip_id.isdigit() else e.clip_id))
        return entries

    def _load_clip(self, clip: ClipEntry) -> None:
        self._save_current_frame()
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(str(clip.video_path))
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.store = MotStore.load(clip.mot_path)
        if self.store.frames:
            first_frame = min(self.store.frames.keys())
            if self.frame_index < 1 or self.frame_index > self.total_frames:
                self.frame_index = first_frame
            elif self.frame_index not in self.store.frames:
                self.frame_index = first_frame
        else:
            self.frame_index = max(1, min(self.frame_index, self.total_frames))
            self.log("No MOT boxes found for this clip.")
        self.frame_input.setValidator(QtGui.QIntValidator(1, self.total_frames))
        self.log(f"Loaded clip {clip.sport}/{clip.event}/{clip.clip_id} ({self.total_frames} frames)")
        self._render_frame()

    def _save_current_frame(self) -> None:
        if not self.clip_entries:
            return
        boxes = self.frame_view.sync_boxes()
        current_frame = self.frame_index
        for box in boxes:
            box.frame = current_frame
        self.store.set_frame(current_frame, boxes)
        current_clip = self.clip_entries[self.clip_index]
        self.store.save(current_clip.mot_path)

    def _render_frame(self) -> None:
        if not self.video_cap:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index - 1)
        ok, frame = self.video_cap.read()
        if not ok:
            self.log("Failed to read frame.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        image = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        boxes = self.store.get_frame(self.frame_index)
        if not boxes and self._last_empty_notice != self.frame_index:
            self.log(f"No boxes for frame {self.frame_index}.")
            self._last_empty_notice = self.frame_index
        self.frame_view.set_frame(image, boxes)
        self.statusBar().showMessage(
            f"Clip {self.clip_entries[self.clip_index].clip_id} "
            f"Frame {self.frame_index}/{self.total_frames}"
        )

    def prev_frame(self) -> None:
        if self.frame_index <= 1:
            return
        self._save_current_frame()
        self.frame_index -= 1
        self._render_frame()

    def next_frame(self) -> None:
        if self.frame_index >= self.total_frames:
            return
        self._save_current_frame()
        self.frame_index += 1
        self._render_frame()

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
        self._save_current_frame()
        self.frame_index = target
        self._render_frame()

    def prev_clip(self) -> None:
        if self.clip_index <= 0:
            return
        self._save_current_frame()
        self.clip_index -= 1
        self._load_clip(self.clip_entries[self.clip_index])

    def next_clip(self) -> None:
        if self.clip_index >= len(self.clip_entries) - 1:
            return
        self._save_current_frame()
        self.clip_index += 1
        self._load_clip(self.clip_entries[self.clip_index])


def run_app(dataset_root: Path, output_root: Path, state_path: Path) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MotEditorWindow(dataset_root, output_root, state_path)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
