from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class OSREntry:
    sport: str
    event: str
    frame_id: str
    json_path: Path
    image_path: Path
    annotation_index: int


class HandleItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent: QtWidgets.QGraphicsItem, corner: str):
        super().__init__(-8, -8, 16, 16, parent)
        self.corner = corner
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            parent = self.parentItem()
            if isinstance(parent, BoxItem):
                parent.update_from_handles()
        return super().itemChange(change, value)


class BoxItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, box: list[float], label: str, color: QtGui.QColor):
        super().__init__()
        left, top, right, bottom = box
        self.setRect(QtCore.QRectF(left, top, right - left, bottom - top))
        self.setPen(QtGui.QPen(color, 3))
        self.handle_tl = HandleItem(self, "tl")
        self.handle_br = HandleItem(self, "br")
        self.label_item = QtWidgets.QGraphicsSimpleTextItem(label, self)
        self.label_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.label_item.setFont(font)
        self.label_item.setPen(QtGui.QPen(color, 1))
        self._sync_handles()

    def _sync_handles(self) -> None:
        rect = self.rect()
        self.handle_tl.setPos(rect.left(), rect.top())
        self.handle_br.setPos(rect.right(), rect.bottom())
        self.label_item.setPos(rect.left() + 4, rect.top() + 4)

    def update_from_handles(self) -> None:
        tl = self.handle_tl.pos()
        br = self.handle_br.pos()
        left = min(tl.x(), br.x())
        top = min(tl.y(), br.y())
        right = max(tl.x(), br.x())
        bottom = max(tl.y(), br.y())
        self.setRect(QtCore.QRectF(left, top, right - left, bottom - top))

    def update_label(self, label: str) -> None:
        self.label_item.setText(label)
        self._sync_handles()

    def to_box(self) -> list[float]:
        rect = self.rect()
        return [rect.left(), rect.top(), rect.right(), rect.bottom()]


class FrameView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene())
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._pixmap_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.box_items: List[BoxItem] = []
        self._fit_to_view = True

    def set_frame(self, image: QtGui.QImage, boxes: List[list[float]], labels: List[str]):
        self.scene().clear()
        pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self.box_items = []
        colors = [QtGui.QColor(0, 255, 0), QtGui.QColor(255, 165, 0)]
        for idx, box in enumerate(boxes):
            label = labels[idx] if idx < len(labels) else ""
            color = colors[idx % len(colors)]
            item = BoxItem(box, label, color)
            self.scene().addItem(item)
            self.box_items.append(item)
        self.scene().setSceneRect(pixmap.rect())
        if self._fit_to_view:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def sync_boxes(self) -> List[list[float]]:
        return [item.to_box() for item in self.box_items]

    def update_labels(self, labels: List[str]) -> None:
        for idx, item in enumerate(self.box_items):
            label = labels[idx] if idx < len(labels) else ""
            item.update_label(label)

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


class OSREditor(QtWidgets.QMainWindow):
    def __init__(self, dataset_root: Path, output_root: Path):
        super().__init__()
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.entries = self._discover_entries()
        if not self.entries:
            raise RuntimeError("No Objects_Spatial_Relationships annotations found.")

        self.entry_index = 0
        self.current_entry = self.entries[self.entry_index]
        self.annotation: dict = {}

        self._build_ui()
        self._load_entry(self.current_entry)

    def _discover_entries(self) -> List[OSREntry]:
        entries: List[OSREntry] = []
        for json_path in self.output_root.glob("*/*/frames/*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            anns = data.get("annotations", [])
            if not isinstance(anns, list):
                continue
            for idx, ann in enumerate(anns):
                if not isinstance(ann, dict):
                    continue
                if ann.get("task_L2") != "Objects_Spatial_Relationships":
                    continue
                origin = data.get("origin", {})
                sport = origin.get("sport")
                event = origin.get("event")
                frame_id = data.get("id") or json_path.stem
                if not (sport and event and frame_id):
                    continue
                image_path = (
                    self.dataset_root
                    / sport
                    / event
                    / "frames"
                    / f"{frame_id}.jpg"
                )
                entries.append(
                    OSREntry(
                        sport=sport,
                        event=event,
                        frame_id=frame_id,
                        json_path=json_path,
                        image_path=image_path,
                        annotation_index=idx,
                    )
                )
        entries.sort(key=lambda e: (e.sport, e.event, int(e.frame_id) if e.frame_id.isdigit() else e.frame_id))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("OSR Fixer")
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.frame_view = FrameView()
        layout.addWidget(self.frame_view, stretch=1)

        qa_layout = QtWidgets.QHBoxLayout()
        self.question_edit = QtWidgets.QTextEdit()
        self.question_edit.setFixedHeight(80)
        self.answer_edit = QtWidgets.QTextEdit()
        self.answer_edit.setFixedHeight(80)
        qa_layout.addWidget(self.question_edit)
        qa_layout.addWidget(self.answer_edit)

        form = QtWidgets.QFormLayout()
        labels_layout = QtWidgets.QHBoxLayout()
        self.label_edits: List[QtWidgets.QLineEdit] = [QtWidgets.QLineEdit(), QtWidgets.QLineEdit()]
        labels_layout.addWidget(self.label_edits[0])
        labels_layout.addWidget(self.label_edits[1])
        form.addRow("Q / A", qa_layout)
        form.addRow("Labels", labels_layout)
        layout.addLayout(form)

        self.review_checkbox = QtWidgets.QCheckBox("Reviewed")
        layout.addWidget(self.review_checkbox)

        controls = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("<< Prev")
        self.next_btn = QtWidgets.QPushButton("Next >>")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        self.save_btn = QtWidgets.QPushButton("Save")
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addStretch(1)
        controls.addWidget(self.fit_btn)
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.save_btn)
        layout.addLayout(controls)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        layout.addWidget(self.log_box)

        self.prev_btn.clicked.connect(self.prev_entry)
        self.next_btn.clicked.connect(self.next_entry)
        self.fit_btn.clicked.connect(self.fit_view)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.save_btn.clicked.connect(self.save_entry)
        self.label_edits[0].textChanged.connect(self._refresh_labels)
        self.label_edits[1].textChanged.connect(self._refresh_labels)

        self.setCentralWidget(central)

    def log(self, msg: str) -> None:
        self.log_box.append(msg)

    def _load_entry(self, entry: OSREntry) -> None:
        try:
            data = json.loads(entry.json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log(f"Failed to load {entry.json_path}: {exc}")
            return
        anns = data.get("annotations", [])
        if not isinstance(anns, list) or entry.annotation_index >= len(anns):
            self.log(f"No annotation at index {entry.annotation_index} in {entry.json_path}")
            return
        ann = anns[entry.annotation_index]
        if not isinstance(ann, dict):
            self.log(f"Annotation malformed in {entry.json_path}")
            return
        self.annotation = ann

        question = ann.get("question", "")
        answer = ann.get("answer", "")
        boxes = ann.get("bounding_box", [])
        if not isinstance(boxes, list):
            boxes = []
        # Normalize to two boxes
        norm_boxes: List[list[float]] = []
        labels: List[str] = []
        for obj in boxes[:2]:
            if isinstance(obj, dict):
                labels.append(obj.get("label", ""))
                box = obj.get("box")
                if isinstance(box, list) and len(box) == 4:
                    norm_boxes.append([float(x) for x in box])
        while len(norm_boxes) < 2:
            norm_boxes.append([0.0, 0.0, 10.0, 10.0])
        while len(labels) < 2:
            labels.append("")

        self.question_edit.setPlainText(str(question))
        self.answer_edit.setPlainText(str(answer))
        self.label_edits[0].setText(labels[0])
        self.label_edits[1].setText(labels[1])
        self.review_checkbox.setChecked(bool(ann.get("reviewed", False)))

        if not entry.image_path.exists():
            self.log(f"Image not found: {entry.image_path}")
            return
        frame = cv2.imread(str(entry.image_path))
        if frame is None:
            self.log(f"Failed to read image: {entry.image_path}")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        image = QtGui.QImage(frame_rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.frame_view.set_frame(image, norm_boxes, labels)
        self.statusBar().showMessage(
            f"{entry.sport}/{entry.event}/frames/{entry.frame_id}"
        )
        self.log(f"Loaded {entry.json_path}")

    def _capture(self) -> None:
        if not self.annotation:
            return
        boxes = self.frame_view.sync_boxes()
        labels = [self.label_edits[0].text(), self.label_edits[1].text()]
        bbox_out = []
        for i, box in enumerate(boxes[:2]):
            bbox_out.append({
                "label": labels[i],
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            })
        self.annotation["bounding_box"] = bbox_out
        self.annotation["question"] = self.question_edit.toPlainText()
        self.annotation["answer"] = self.answer_edit.toPlainText()
        self.annotation["reviewed"] = bool(self.review_checkbox.isChecked())

    def _refresh_labels(self) -> None:
        labels = [self.label_edits[0].text(), self.label_edits[1].text()]
        self.frame_view.update_labels(labels)

    def save_entry(self) -> None:
        if not self.annotation:
            return
        self._capture()
        entry = self.entries[self.entry_index]
        try:
            data = json.loads(entry.json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log(f"Failed to reload {entry.json_path} for save: {exc}")
            return
        anns = data.get("annotations", [])
        if not isinstance(anns, list):
            self.log(f"Invalid annotations list in {entry.json_path}")
            return
        anns[entry.annotation_index] = self.annotation
        data["annotations"] = anns
        entry.json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.log(f"Saved {entry.json_path}")

    def fit_view(self) -> None:
        self.frame_view.resetTransform()
        self.frame_view.set_fit_mode(True)
        self.frame_view.repaint()

    def zoom_in(self) -> None:
        self.frame_view.zoom(1.1)

    def zoom_out(self) -> None:
        self.frame_view.zoom(0.9)

    def prev_entry(self) -> None:
        if self.entry_index <= 0:
            return
        self.save_entry()
        self.entry_index -= 1
        self.current_entry = self.entries[self.entry_index]
        self._load_entry(self.current_entry)

    def next_entry(self) -> None:
        if self.entry_index >= len(self.entries) - 1:
            return
        self.save_entry()
        self.entry_index += 1
        self.current_entry = self.entries[self.entry_index]
        self._load_entry(self.current_entry)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_entry()
        event.accept()


def run_app(dataset_root: Path, output_root: Path) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = OSREditor(dataset_root, output_root)
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())
