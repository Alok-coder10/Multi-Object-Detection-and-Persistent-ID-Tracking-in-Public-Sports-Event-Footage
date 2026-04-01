import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import colorsys


# COCO ids used by SportDetector — short labels for overlay
CLASS_DISPLAY = {
    0: "Person",
    32: "Ball",
}

# BGR — stand out from golden-ratio person colors
BALL_COLOR = (0, 200, 255)


def generate_id_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generate a visually distinct BGR color for a given track ID.
    Uses golden-ratio hue spacing for maximum differentiation.
    """
    golden_ratio = 0.6180339887
    hue = (track_id * golden_ratio) % 1.0
    saturation = 0.85
    value = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


class TrajectoryManager:
    """Stores and manages historical center points per track ID."""

    def __init__(self, max_len: int = 60):
        self.trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.max_len = max_len

    def update(self, tracks: List[Dict]):
        """Add current centers to trail history."""
        for t in tracks:
            bbox = t["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            self.trails[t["id"]].append((cx, cy))
            if len(self.trails[t["id"]]) > self.max_len:
                self.trails[t["id"]].pop(0)

    def get_trail(self, track_id: int) -> List[Tuple[int, int]]:
        return self.trails.get(track_id, [])

    def get_all_centers(self) -> List[Tuple[int, int]]:
        """Return all recorded centers (for heatmap)."""
        centers = []
        for pts in self.trails.values():
            centers.extend(pts)
        return centers


class HeatmapAccumulator:
    """Accumulates a movement heatmap across the entire video."""

    def __init__(self, frame_size: Tuple[int, int], blur_radius: int = 25):
        h, w = frame_size
        self.accumulator = np.zeros((h, w), dtype=np.float32)
        self.blur_radius = blur_radius

    def update(self, tracks: List[Dict]):
        """Add centers of current tracks to accumulator."""
        for t in tracks:
            bbox = t["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            if 0 <= cy < self.accumulator.shape[0] and 0 <= cx < self.accumulator.shape[1]:
                self.accumulator[cy, cx] += 1.0

    def render(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """
        Overlay heatmap on a frame with a colormap.
        Returns a new frame with heatmap blended.
        """
        blurred = cv2.GaussianBlur(self.accumulator, (0, 0), self.blur_radius)
        if blurred.max() > 0:
            norm = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            norm = blurred.astype(np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = norm > 5
        output = frame.copy()
        output[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
        return output

    def save(self, path: str, frame_size: Tuple[int, int]):
        """Save standalone heatmap image."""
        blurred = cv2.GaussianBlur(self.accumulator, (0, 0), self.blur_radius)
        if blurred.max() > 0:
            norm = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            norm = blurred.astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        resized = cv2.resize(colored, (frame_size[1], frame_size[0]))
        cv2.imwrite(path, resized)


class Annotator:
    """
    Draws detections, IDs, trails, and overlays on frames.
    """

    def __init__(
        self,
        show_trail: bool = True,
        trail_thickness: int = 2,
        box_thickness: int = 2,
        label_font_scale: float = 0.55,
        show_conf: bool = True,
    ):
        self.show_trail = show_trail
        self.trail_thickness = trail_thickness
        self.box_thickness = box_thickness
        self.label_font_scale = label_font_scale
        self.show_conf = show_conf
        self.traj_manager = TrajectoryManager(max_len=60)

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        frame_num: int = 0,
        total_frames: int = 0,
    ) -> np.ndarray:
        """
        Annotate a frame with all active tracks.

        Args:
            frame: BGR frame
            tracks: List of track dicts from tracker.update()
            frame_num: Current frame index
            total_frames: Total frames in video (for progress bar)

        Returns:
            Annotated BGR frame
        """
        self.traj_manager.update(tracks)
        output = frame.copy()

        for t in tracks:
            track_id = t["id"]
            bbox = t["bbox"]
            conf = t["conf"]
            cls_id = int(t.get("class_id", 0))
            kind = CLASS_DISPLAY.get(cls_id, f"cls{cls_id}")
            # Distinct color for ball; per-ID colors for people (and unknown classes)
            color = BALL_COLOR if cls_id == 32 else generate_id_color(track_id)

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # --- Draw trail ---
            if self.show_trail:
                trail = self.traj_manager.get_trail(track_id)
                if len(trail) > 1:
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        thickness = max(1, int(self.trail_thickness * alpha))
                        cv2.line(output, trail[i - 1], trail[i], color, thickness)

            # --- Bounding box with rounded corners effect ---
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.box_thickness)

            # --- Label background ---
            label = f"{kind} ID:{track_id}"
            if self.show_conf:
                label += f" {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 1
            )
            label_y = max(y1, th + 6)
            cv2.rectangle(
                output,
                (x1, label_y - th - 6),
                (x1 + tw + 4, label_y),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 2, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # --- HUD overlay ---
        self._draw_hud(output, tracks, frame_num, total_frames)

        return output

    def _draw_hud(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        frame_num: int,
        total_frames: int,
    ):
        """Draw head-up display with count and progress info."""
        h, w = frame.shape[:2]
        count = len(tracks)

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Active track count
        cv2.putText(
            frame,
            f"Active: {count}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )

        # Frame counter
        frame_text = f"Frame: {frame_num}"
        if total_frames > 0:
            frame_text += f" / {total_frames}"
            # Progress bar
            bar_w = int(w * (frame_num / max(total_frames, 1)))
            cv2.rectangle(frame, (0, 36), (bar_w, 38), (0, 200, 120), -1)

        cv2.putText(
            frame,
            frame_text,
            (w // 2 - 60, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Watermark
        cv2.putText(
            frame,
            "Sports Tracker v1.0",
            (w - 175, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
