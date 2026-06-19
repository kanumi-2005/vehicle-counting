#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from config_utils import (
    add_common_config_args,
    load_config_from_args,
    project_path,
)


Point = Tuple[int, int]
cv2 = None
np = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_CONFIG_PATH = PROJECT_ROOT / ".apprc"

DEFAULT_APP_CONFIG = {
    "video": None,
    "output": None,
    "width": None,
    "height": None,
    "show_control_panel": True,
    "verbose_tracking": False,
    "verbose_detection": False,
    "quiet_detection": False,
}

ROI_COLORS = [
    (0, 215, 255),
    (80, 220, 120),
    (255, 160, 60),
    (210, 120, 255),
    (80, 180, 255),
]

CONTROL_LINES = [
    "n: new ROI",
    "enter: save ROI / run",
    "s: select ROI",
    "drag vertex: edit selected ROI",
    "x: delete ROI by inside click",
    "r: reset current ROI",
    "esc/c: cancel current command",
    "h: toggle current panel",
    "f1: toggle control panel",
    "f2: toggle counting panel",
    "p: pause / resume",
    "q/window close: quit",
]

F1_KEYS = {190, 65470, 7340032}
F2_KEYS = {191, 65471, 7405568}

COMMAND_HINTS = {
    "draw": [
        "draw ROI: left click adds vertices",
        "enter: save ROI",
        "esc: cancel drawing",
    ],
    "select": [
        "select ROI: click inside a polygon",
        "enter: finish command",
        "esc: cancel selection",
    ],
    "edit": [
        "edit ROI: drag a red vertex",
        "enter: finish editing",
        "esc: cancel editing",
    ],
    "delete": [
        "delete ROI: click inside a polygon",
        "enter: finish deletion",
        "esc: cancel deletion",
    ],
}


@dataclass
class ROI:
    name: str
    points: List[Point]
    counted_track_ids: Set[int] = field(default_factory=set)
    total_count: int = 0


class InteractiveTrackingCounter:
    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        output_path: Optional[Path],
        frame_width: int,
        frame_height: int,
        pipeline: TrackingPipeline,
        show_help_panel: bool,
        verbose_tracking: bool,
    ) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pipeline = pipeline
        self.verbose_tracking = verbose_tracking

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.writer = self._create_writer(output_path)
        self.window_name = "Interactive Tracking Counter"

        self.rois: List[ROI] = []
        self.current_points: List[Point] = []
        self.mouse_position: Point = (0, 0)
        self.mode = "idle"
        self.command_snapshot: Optional[Dict] = None
        self.selected_roi_index: Optional[int] = None
        self.dragged_vertex_index: Optional[int] = None
        self.paused = False
        self.should_quit = False
        self.confirm_quit = False
        self.show_control_panel = show_help_panel
        self.show_tracking_panel = True

        self.total_counted_track_ids: Set[int] = set()
        self.track_history: DefaultDict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self.previous_time = time.time()
        self.frame_index = 0

    def _create_writer(self, output_path: Optional[Path]) -> Optional[cv2.VideoWriter]:
        if output_path is None:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.frame_width, self.frame_height),
        )
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        print(f"[INFO] Output video: {output_path}")
        return writer

    def run(self) -> None:
        try:
            first_frame = self._read_frame()
            self._setup_rois(first_frame)
            if self.should_quit:
                return
            self._run_tracking_loop(first_frame)
        finally:
            self._release()

    def _read_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.resize(frame, (self.frame_width, self.frame_height))

    def _setup_rois(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            raise ValueError("Cannot read the first video frame.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_event)
        self._print_controls()

        while not self.should_quit:
            setup_frame = frame.copy()
            self._draw_rois(setup_frame, show_vertices=True)
            self._draw_current_roi(setup_frame)
            self._draw_control_panel(setup_frame, setup=True)
            self._draw_quit_confirmation(setup_frame)

            cv2.imshow(self.window_name, setup_frame)
            key = cv2.waitKeyEx(20)
            if self._window_was_closed():
                self.should_quit = True
                return

            if self._handle_key(key, setup=True):
                return

    def _run_tracking_loop(self, initial_frame: Optional[np.ndarray]) -> None:
        frame = initial_frame

        while not self.should_quit:
            if frame is None:
                break

            if not self.paused:
                rendered = self._process_frame(frame)
                if self.writer is not None:
                    self.writer.write(rendered)
                cv2.imshow(self.window_name, rendered)

            if self._window_was_closed():
                print("[INFO] Window closed.")
                self.should_quit = True
                break

            key = cv2.waitKeyEx(1 if not self.paused else 30)
            self._handle_key(key, setup=False)

            if not self.paused:
                frame = self._read_frame()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_index += 1
        current_time = time.time()
        fps = 1 / max(current_time - self.previous_time, 1e-6)
        self.previous_time = current_time

        detections = self.pipeline.detect(frame)
        tracked = self.pipeline.track(detections)

        if self.verbose_tracking:
            detection_count = len(detections)
            track_count = 0
            if tracked.tracker_id is not None:
                track_count = len([tid for tid in tracked.tracker_id if tid >= 0])
            print(
                f"[TRACK] frame={self.frame_index} "
                f"detections={detection_count} tracks={track_count}"
            )

        output_frame = frame.copy()
        self._draw_rois(output_frame, show_vertices=False)
        live_stats = self._track_and_count(output_frame, tracked)
        self._draw_runtime_panel(output_frame, fps, live_stats)
        self._draw_control_panel(output_frame, setup=False)
        self._draw_quit_confirmation(output_frame)
        return output_frame

    def _track_and_count(self, frame: np.ndarray, tracked) -> Dict[str, Dict]:
        live_stats: Dict[str, Dict] = {
            roi.name: {"total": 0, "classes": defaultdict(int)}
            for roi in self.rois
        }
        current_track_ids = set()

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            self.track_history.clear()
            return live_stats

        for box, track_id, class_id in zip(
            tracked.xyxy, tracked.tracker_id, tracked.class_id
        ):
            if track_id < 0:
                continue

            track_id = int(track_id)
            current_track_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)
            anchor = ((x1 + x2) // 2, y2)
            class_name = self.pipeline.model.names[int(class_id)]
            self.track_history[track_id].append(anchor)

            roi = self._find_roi_containing_point(anchor)
            color = (150, 150, 150)
            if roi is not None:
                color = (0, 255, 0)
                live_stats[roi.name]["total"] += 1
                live_stats[roi.name]["classes"][class_name] += 1

                if track_id not in roi.counted_track_ids:
                    roi.counted_track_ids.add(track_id)
                    roi.total_count += 1

                self.total_counted_track_ids.add(track_id)

            self._draw_track(frame, (x1, y1, x2, y2), track_id, class_name, color)

        for lost_track_id in set(self.track_history) - current_track_ids:
            del self.track_history[lost_track_id]

        return live_stats

    def _draw_track(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        track_id: int,
        class_name: str,
        color: Tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = box
        history_points = list(self.track_history[track_id])
        direction = ""

        if len(history_points) >= 2:
            points_array = np.array(history_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points_array], False, (0, 165, 255), 2)
            y_delta = history_points[-1][1] - history_points[0][1]
            if abs(y_delta) > 10:
                direction = "DOWN" if y_delta > 0 else "UP"
            if len(history_points) > 5:
                cv2.arrowedLine(
                    frame,
                    history_points[-5],
                    history_points[-1],
                    (255, 0, 255),
                    2,
                    tipLength=0.5,
                )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name.upper()} #{track_id}"
        if direction:
            label = f"{label} | {direction}"
        self._draw_label(frame, label, (x1, max(y1 - 24, 4)), color)
        cv2.circle(frame, history_points[-1], 4, (0, 255, 255), -1)

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Point,
        accent_color: Tuple[int, int, int],
    ) -> None:
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )
        x = int(np.clip(x, 0, max(self.frame_width - text_width - 12, 0)))
        y = int(np.clip(y, 0, max(self.frame_height - text_height - 12, 0)))
        top_left = (x, y)
        bottom_right = (x + text_width + 12, y + text_height + baseline + 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, top_left, bottom_right, accent_color, 1)
        cv2.putText(
            frame,
            text,
            (x + 6, y + text_height + 5),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _handle_mouse_event(
        self, event: int, x: int, y: int, flags: int, param
    ) -> None:
        self.mouse_position = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.confirm_quit = False
            if self.mode == "draw":
                self.current_points.append((x, y))
                print(f"[ROI] Added vertex {len(self.current_points)}: ({x}, {y})")
            elif self.mode == "select":
                self._select_roi_at((x, y))
            elif self.mode == "delete":
                self._delete_roi_at((x, y))
            elif self.mode == "edit":
                self.dragged_vertex_index = self._find_nearest_vertex((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "edit" and self.dragged_vertex_index is not None:
                self._move_selected_vertex((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragged_vertex_index = None

    def _handle_key(self, key: int, setup: bool) -> bool:
        if key in (-1, 255):
            return False
        ascii_key = key & 0xFF

        if key in F1_KEYS:
            self.show_control_panel = not self.show_control_panel
            return False
        if key in F2_KEYS:
            self.show_tracking_panel = not self.show_tracking_panel
            return False

        if self.confirm_quit:
            if ascii_key in (ord("y"), ord("Y"), 13):
                self.should_quit = True
                return True
            if ascii_key in (ord("n"), ord("N"), 27):
                self.confirm_quit = False
                print("[INFO] Quit canceled.")
                return False

        if ascii_key == 27:
            self._cancel_current_command()
            return False

        if ascii_key in (ord("q"), ord("Q")):
            self.confirm_quit = True
            print("[INFO] Quit? Press y/Enter to confirm, n/Esc to cancel.")
            return False

        if ascii_key in (ord("y"), ord("Y")):
            return False

        if ascii_key in (ord("n"), ord("N")) and self.mode != "draw":
            self._start_command("draw", clear_selection=True)
            self.current_points.clear()
            print("[ROI] Drawing a new ROI.")
            return False

        if ascii_key in (ord("n"), ord("N")) and self.mode == "draw":
            return False

        if ascii_key in (ord("c"), ord("C")):
            self._cancel_current_command()
            return False

        if ascii_key in (ord("h"), ord("H")):
            if setup:
                self.show_control_panel = not self.show_control_panel
            else:
                self.show_tracking_panel = not self.show_tracking_panel
            return False

        if ascii_key in (ord("p"), ord("P")) and not setup:
            self.paused = not self.paused
            print("[INFO] Paused." if self.paused else "[INFO] Resumed.")
            return False

        if ascii_key in (ord("s"), ord("S")):
            self._start_command("select", clear_selection=True)
            self.current_points.clear()
            print("[ROI] Click inside an ROI to select it.")
            return False

        if ascii_key in (ord("x"), ord("X"), ord("d"), ord("D")):
            self._start_command("delete", clear_selection=True)
            self.current_points.clear()
            print("[ROI] Click inside an ROI to delete it.")
            return False

        if ascii_key in (ord("r"), ord("R")):
            if self.current_points:
                self.current_points.clear()
                print("[ROI] Current ROI was reset.")
            elif self.selected_roi_index is not None:
                del self.rois[self.selected_roi_index]
                self._rename_rois()
                self.selected_roi_index = None
                self.mode = "idle"
                self.command_snapshot = None
                print("[ROI] Selected ROI was removed.")
            return False

        if ascii_key == 13:
            if self.confirm_quit:
                self.should_quit = True
                return True
            if self.mode == "draw":
                if self.current_points:
                    self._finish_current_roi()
                else:
                    self._finish_current_command()
                return False
            if self.mode in ("select", "edit", "delete"):
                self._finish_current_command()
                return False
            if setup:
                if not self.rois:
                    print("[WARN] Create at least one ROI before running.")
                    return False
                self.mode = "idle"
                return True

        return False

    def _finish_current_roi(self) -> None:
        if len(self.current_points) < 3:
            print("[WARN] An ROI needs at least 3 vertices.")
            return

        roi = ROI(name=f"ROI {len(self.rois) + 1}", points=self.current_points.copy())
        self.rois.append(roi)
        self.current_points.clear()
        self.selected_roi_index = len(self.rois) - 1
        self.mode = "idle"
        self.command_snapshot = None
        print(f"[ROI] Saved {roi.name}.")

    def _start_command(self, mode: str, clear_selection: bool) -> None:
        if self.command_snapshot is None:
            self.command_snapshot = {
                "mode": self.mode,
                "selected_roi_index": self.selected_roi_index,
                "current_points": self.current_points.copy(),
                "rois": [
                    ROI(
                        name=roi.name,
                        points=roi.points.copy(),
                        counted_track_ids=roi.counted_track_ids.copy(),
                        total_count=roi.total_count,
                    )
                    for roi in self.rois
                ],
            }
        self.mode = mode
        self.dragged_vertex_index = None
        if clear_selection:
            self.selected_roi_index = None

    def _finish_current_command(self) -> None:
        self.current_points.clear()
        self.dragged_vertex_index = None
        self.selected_roi_index = None
        self.mode = "idle"
        self.command_snapshot = None
        print("[ROI] Command finished.")

    def _cancel_current_command(self) -> None:
        self.confirm_quit = False
        self.dragged_vertex_index = None
        if self.command_snapshot is not None:
            self.current_points = self.command_snapshot["current_points"]
            self.selected_roi_index = self.command_snapshot["selected_roi_index"]
            self.mode = self.command_snapshot["mode"]
            self.rois = self.command_snapshot["rois"]
            self.command_snapshot = None
        else:
            self.current_points.clear()
            self.selected_roi_index = None
            self.mode = "idle"
        print("[ROI] Command canceled.")

    def _select_roi_at(self, point: Point) -> None:
        for index in reversed(range(len(self.rois))):
            if self._point_inside_roi(point, self.rois[index]):
                self.selected_roi_index = index
                self.mode = "edit"
                print(f"[ROI] Selected {self.rois[index].name}. Drag vertices to edit.")
                return
        print("[ROI] No ROI found at clicked point.")

    def _delete_roi_at(self, point: Point) -> None:
        for index in reversed(range(len(self.rois))):
            if self._point_inside_roi(point, self.rois[index]):
                deleted_name = self.rois[index].name
                del self.rois[index]
                self._rename_rois()
                self.selected_roi_index = None
                self.mode = "idle"
                print(f"[ROI] Deleted {deleted_name}.")
                return
        print("[ROI] No ROI found at clicked point.")

    def _rename_rois(self) -> None:
        for index, roi in enumerate(self.rois, start=1):
            roi.name = f"ROI {index}"

    def _find_nearest_vertex(self, point: Point) -> Optional[int]:
        if self.selected_roi_index is None:
            return None

        roi = self.rois[self.selected_roi_index]
        distances = [
            np.hypot(vertex[0] - point[0], vertex[1] - point[1])
            for vertex in roi.points
        ]
        if not distances:
            return None

        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] <= 14:
            return nearest_index
        return None

    def _move_selected_vertex(self, point: Point) -> None:
        if self.selected_roi_index is None or self.dragged_vertex_index is None:
            return

        x = int(np.clip(point[0], 0, self.frame_width - 1))
        y = int(np.clip(point[1], 0, self.frame_height - 1))
        self.rois[self.selected_roi_index].points[self.dragged_vertex_index] = (x, y)

    def _find_roi_containing_point(self, point: Point) -> Optional[ROI]:
        for roi in self.rois:
            if self._point_inside_roi(point, roi):
                return roi
        return None

    def _point_inside_roi(self, point: Point, roi: ROI) -> bool:
        if len(roi.points) < 3:
            return False
        points = np.array(roi.points, dtype=np.int32)
        return cv2.pointPolygonTest(points, point, False) >= 0

    def _draw_rois(self, frame: np.ndarray, show_vertices: bool) -> None:
        overlay = frame.copy()
        for index, roi in enumerate(self.rois):
            color = ROI_COLORS[index % len(ROI_COLORS)]
            if index == self.selected_roi_index:
                color = (0, 255, 0)
            points = np.array(roi.points, dtype=np.int32)
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)

            center = self._polygon_center(points)
            cv2.putText(
                frame,
                roi.name,
                (center[0] - 26, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if show_vertices or index == self.selected_roi_index:
                for vertex in roi.points:
                    cv2.circle(frame, vertex, 5, (0, 0, 255), -1)

        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    def _draw_current_roi(self, frame: np.ndarray) -> None:
        if not self.current_points:
            return

        points = np.array(self.current_points, dtype=np.int32)
        cv2.polylines(frame, [points], False, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(
            frame,
            self.current_points[-1],
            self.mouse_position,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for point in self.current_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def _draw_control_panel(self, frame: np.ndarray, setup: bool) -> None:
        if self.confirm_quit:
            return

        if self._has_active_command():
            self._draw_command_hint(frame, setup)
            return

        if not self.show_control_panel:
            return

        title = "ROI setup controls" if setup else "Runtime controls"
        lines = [
            title,
            f"mode: {self.mode}",
            f"rois: {len(self.rois)} | current vertices: {len(self.current_points)}",
        ]
        lines.extend(CONTROL_LINES)
        self._draw_panel(frame, 16, 16, 460, lines)

    def _has_active_command(self) -> bool:
        return self.mode in COMMAND_HINTS

    def _draw_command_hint(self, frame: np.ndarray, setup: bool) -> None:
        title = f"command: {self.mode}"
        lines = [title]
        lines.extend(COMMAND_HINTS.get(self.mode, []))
        if self.mode == "draw":
            lines.append(f"vertices: {len(self.current_points)}")
        elif self.selected_roi_index is not None:
            lines.append(f"selected: {self.rois[self.selected_roi_index].name}")
        self._draw_panel(frame, 16, 16, 360, lines)

    def _draw_runtime_panel(
        self,
        frame: np.ndarray,
        fps: float,
        live_stats: Dict[str, Dict],
    ) -> None:
        if self.confirm_quit:
            return

        if self._has_active_command():
            return

        if not self.show_tracking_panel:
            return

        self._draw_tracking_panel(frame, fps, live_stats)

    def _draw_tracking_panel(
        self,
        frame: np.ndarray,
        fps: float,
        live_stats: Dict[str, Dict],
    ) -> None:
        lines = [
            "Tracking / Counting",
            f"fps {fps:.1f} | unique {len(self.total_counted_track_ids)}",
        ]

        for roi in self.rois:
            live_total = live_stats[roi.name]["total"]
            lines.append(f"{roi.name}: live {live_total} | total {roi.total_count}")
            class_parts = [
                f"{class_name[:3].upper()}:{count}"
                for class_name, count in sorted(live_stats[roi.name]["classes"].items())
                if count
            ]
            if class_parts:
                lines.append("  " + "  ".join(class_parts[:4]))

        lines.append("h/F2: hide | F1: controls")
        self._draw_panel(frame, 16, 16, 340, lines)

    def _draw_quit_confirmation(self, frame: np.ndarray) -> None:
        if not self.confirm_quit:
            return

        lines = [
            "quit tracking counter?",
            "enter/y: confirm",
            "esc/n: cancel",
        ]
        self._draw_panel(frame, 16, 16, 300, lines)

    def _draw_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        lines: List[str],
    ) -> None:
        line_height = 22
        height = min(28 + line_height * len(lines), self.frame_height - y - 20)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 24, 33), -1)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        text_y = y + 26
        for index, line in enumerate(lines):
            if text_y > y + height - 8:
                break
            color = (0, 255, 255) if index == 0 else (235, 235, 235)
            cv2.putText(
                frame,
                line,
                (x + 14, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += line_height

    def _polygon_center(self, points: np.ndarray) -> Point:
        moments = cv2.moments(points)
        if moments["m00"] == 0:
            return tuple(points.mean(axis=0).astype(int))
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    def _window_was_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _print_controls(self) -> None:
        print("\n=== INTERACTIVE TRACKING COUNTER ===")
        print(f"Video : {self.video_path}")
        print(f"Model : {self.model_path}")
        print("\nControls:")
        for line in CONTROL_LINES:
            print(f"  {line}")
        print()

    def _release(self) -> None:
        if self.capture.isOpened():
            self.capture.release()
        if self.writer is not None and self.writer.isOpened():
            self.writer.release()
        cv2.destroyAllWindows()
        if self.output_path is not None:
            print(f"[INFO] Saved output video: {self.output_path}")
        print("[INFO] Closed tracking counter.")


def get_effective_tracking_config(
    config: Dict,
    tracker_type_override: Optional[str] = None,
) -> Dict:
    tracking_config = config["tracking"]
    tracker_type = (tracker_type_override or tracking_config["tracker_type"]).lower()
    tracker_defaults = tracking_config.get(tracker_type, {})

    effective_config = {
        "tracker_type": tracker_type,
        "lost_track_buffer": tracker_defaults.get(
            "lost_track_buffer",
            tracking_config["lost_track_buffer"],
        ),
        "track_activation_threshold": tracker_defaults.get(
            "track_activation_threshold",
            tracking_config["track_activation_threshold"],
        ),
        "minimum_consecutive_frames": tracker_defaults.get(
            "minimum_consecutive_frames",
            tracking_config["minimum_consecutive_frames"],
        ),
        "minimum_iou_threshold": tracker_defaults.get(
            "minimum_iou_threshold",
            tracking_config["minimum_iou_threshold"],
        ),
        "high_conf_detection_threshold": tracker_defaults.get(
            "high_conf_detection_threshold",
            tracking_config["high_conf_detection_threshold"],
        ),
    }
    return effective_config


def build_pipeline(config: Dict, args: argparse.Namespace) -> TrackingPipeline:
    from tracking_pipeline import TrackingPipeline

    detection_config = config["detection"]
    tracking_config = get_effective_tracking_config(config, args.tracker)

    detection_verbose = detection_config["verbose"]
    if args.verbose_detection:
        detection_verbose = True
    if args.quiet_detection:
        detection_verbose = False

    return TrackingPipeline(
        model_path=str(project_path(args.model or config["paths"]["model_path"])),
        confidence_threshold=args.confidence_threshold
        if args.confidence_threshold is not None
        else detection_config["confidence_threshold"],
        device=args.device or detection_config["device"],
        verbose=detection_verbose,
        tracker_type=tracking_config["tracker_type"],
        lost_track_buffer=args.lost_track_buffer
        if args.lost_track_buffer is not None
        else tracking_config["lost_track_buffer"],
        frame_rate=args.frame_rate
        if args.frame_rate is not None
        else config["dataset"]["frame_rate"],
        track_activation_threshold=args.track_activation_threshold
        if args.track_activation_threshold is not None
        else tracking_config["track_activation_threshold"],
        minimum_consecutive_frames=args.minimum_consecutive_frames
        if args.minimum_consecutive_frames is not None
        else tracking_config["minimum_consecutive_frames"],
        minimum_iou_threshold=args.minimum_iou_threshold
        if args.minimum_iou_threshold is not None
        else tracking_config["minimum_iou_threshold"],
        high_conf_detection_threshold=args.high_conf_detection_threshold
        if args.high_conf_detection_threshold is not None
        else tracking_config["high_conf_detection_threshold"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive vehicle tracking and ROI-based counting tool."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_config_args(parser)
    parser.add_argument(
        "--app-config",
        default=str(DEFAULT_APP_CONFIG_PATH),
        help="Path to app-level JSON config, e.g. .apprc.",
    )
    parser.add_argument(
        "--write-default-app-config",
        action="store_true",
        help="Write the default app config JSON and exit.",
    )
    parser.add_argument("--video", help="Input video path.")
    parser.add_argument("--model", help="Override paths.model_path.")
    parser.add_argument("--output", help="Optional output video path.")
    parser.add_argument(
        "--save-output",
        nargs="?",
        const="demo/tracking_counter_output.mp4",
        help=(
            "Save rendered tracking video. Optionally pass a path; without "
            "a path it uses demo/tracking_counter_output.mp4."
        ),
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable output video saving.",
    )
    parser.add_argument("--width", type=int, help="Display/output frame width.")
    parser.add_argument("--height", type=int, help="Display/output frame height.")
    parser.add_argument("--tracker", choices=["byte", "sort"], help="Tracker type.")
    parser.add_argument("--device", help="Detection device, e.g. auto, cpu, cuda.")
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--frame-rate", type=int)
    parser.add_argument("--lost-track-buffer", type=int)
    parser.add_argument("--track-activation-threshold", type=float)
    parser.add_argument("--minimum-consecutive-frames", type=int)
    parser.add_argument("--minimum-iou-threshold", type=float)
    parser.add_argument("--high-conf-detection-threshold", type=float)
    parser.add_argument(
        "--verbose-detection",
        action="store_true",
        help="Show detector logs.",
    )
    parser.add_argument(
        "--quiet-detection",
        action="store_true",
        help="Hide detector logs.",
    )
    parser.add_argument(
        "--verbose-tracking",
        action="store_true",
        help="Print per-frame detection and tracking counts.",
    )
    parser.add_argument(
        "--hide-controls",
        action="store_true",
        help="Start with the on-window control panel hidden.",
    )
    return parser.parse_args()


def load_app_config(app_config_path: str) -> Dict:
    path = project_path(app_config_path)
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_APP_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as file_in:
        user_config = json.load(file_in)

    app_config = DEFAULT_APP_CONFIG.copy()
    app_config.update(user_config)
    return app_config


def write_default_app_config(app_config_path: str) -> None:
    path = project_path(app_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_out:
        json.dump(DEFAULT_APP_CONFIG, file_out, indent=2)
        file_out.write("\n")


def load_runtime_dependencies() -> None:
    global cv2, np

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ModuleNotFoundError as error:
        missing_name = error.name or "runtime dependency"
        raise SystemExit(
            f"Missing dependency: {missing_name}. "
            "Install the project runtime dependencies before running tracking."
        ) from error

    cv2 = cv2_module
    np = np_module


def print_startup_intro(
    video_path: Path,
    model_path: Path,
    output_path: Optional[Path],
    frame_width: int,
    frame_height: int,
    tracker_type: str,
) -> None:
    print("\n=== INTERACTIVE TRACKING COUNTER ===")
    print("ROI-based vehicle tracking and counting")
    print(f"Video   : {video_path}")
    print(f"Model   : {model_path}")
    print(f"Tracker : {tracker_type}")
    print(f"Frame   : {frame_width}x{frame_height}")
    print(f"Output  : {output_path if output_path else 'disabled'}")
    print("\nPress --help to see all CLI options.")
    print("The OpenCV window also shows the control panel at startup.\n")


def main() -> None:
    args = parse_args()
    if args.write_default_app_config:
        write_default_app_config(args.app_config)
        return

    app_config = load_app_config(args.app_config)
    config = load_config_from_args(args)
    if config is None:
        return
    video_arg = args.video or app_config["video"]
    if not video_arg:
        raise SystemExit("error: --video is required unless --write-default-config.")

    frame_width = args.width or app_config["width"] or config["dataset"]["image_width"]
    frame_height = args.height or app_config["height"] or config["dataset"]["image_height"]
    if args.no_output:
        output_arg = None
    elif args.save_output is not None:
        output_arg = args.save_output
    elif args.output is not None:
        output_arg = args.output
    else:
        output_arg = app_config["output"]
    output_path = project_path(output_arg) if output_arg else None
    model_path = project_path(args.model or config["paths"]["model_path"])
    tracking_config = get_effective_tracking_config(config, args.tracker)
    args.verbose_detection = args.verbose_detection or app_config["verbose_detection"]
    args.quiet_detection = args.quiet_detection or app_config["quiet_detection"]
    verbose_tracking = args.verbose_tracking or app_config["verbose_tracking"]
    show_control_panel = app_config["show_control_panel"] and not args.hide_controls

    print_startup_intro(
        video_path=project_path(video_arg),
        model_path=model_path,
        output_path=output_path,
        frame_width=frame_width,
        frame_height=frame_height,
        tracker_type=tracking_config["tracker_type"],
    )

    load_runtime_dependencies()

    pipeline = build_pipeline(config, args)
    app = InteractiveTrackingCounter(
        video_path=project_path(video_arg),
        model_path=model_path,
        output_path=output_path,
        frame_width=frame_width,
        frame_height=frame_height,
        pipeline=pipeline,
        show_help_panel=show_control_panel,
        verbose_tracking=verbose_tracking,
    )
    app.run()


if __name__ == "__main__":
    main()
