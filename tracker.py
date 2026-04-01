import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman Filter (constant-velocity model for bounding boxes)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Kalman filter for a single bounding box.
    State: [cx, cy, s, r, vx, vy, vs]
    where (cx, cy) = center, s = area, r = aspect ratio, v* = velocities.
    """

    count = 0  # global track ID counter

    def __init__(
        self,
        bbox: List[float],
        track_id: Optional[int] = None,
        conf: float = 1.0,
        cls_id: int = 0,
    ):
        """
        Initialize with [x1, y1, x2, y2].
        """
        self.id = track_id if track_id is not None else KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.conf = float(conf)
        self.cls_id = int(cls_id)

        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Covariances
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001]).astype(float)
        self.R = np.diag([1, 1, 10, 10]).astype(float)

        # Initialize state
        cx, cy, s, r = self._xyxy_to_z(bbox)
        self.x = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=float)

        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history: List[List[float]] = []

    def _xyxy_to_z(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)  # area
        r = (x2 - x1) / (y2 - y1 + 1e-6)  # aspect ratio
        return cx, cy, s, r

    def _z_to_xyxy(self):
        cx, cy, s, r = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        s = max(s, 1)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]

    def predict(self):
        """Advance state via Kalman prediction."""
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        pred_bbox = self._z_to_xyxy()
        self.history.append(pred_bbox)
        return pred_bbox

    def update(
        self,
        bbox: List[float],
        conf: Optional[float] = None,
        cls_id: Optional[int] = None,
    ):
        """Update with a new measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.history = []

        if conf is not None:
            self.conf = float(conf)
        if cls_id is not None:
            self.cls_id = int(cls_id)

        z = np.array(self._xyxy_to_z(bbox), dtype=float).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self) -> List[float]:
        return self._z_to_xyxy()


# ---------------------------------------------------------------------------
# IoU utilities
# ---------------------------------------------------------------------------

def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    bboxes_a: (N, 4), bboxes_b: (M, 4) in xyxy format.
    Returns: (N, M) IoU matrix.
    """
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    inter_x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    inter_y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    inter_x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    inter_y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Main Tracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack-inspired multi-object tracker.

    Two-stage association:
      1. High-confidence detections matched to active tracks via IoU + Kalman.
      2. Low-confidence detections matched to unmatched tracks.

    Lost tracks are kept in a buffer for re-identification.

    Args:
        max_age: Frames to keep a lost track before deletion.
        min_hits: Minimum consecutive hits before a track is confirmed.
        iou_threshold_high: IoU threshold for high-conf association.
        iou_threshold_low: IoU threshold for low-conf association.
        high_conf_threshold: Detection confidence split point.
        low_conf_threshold: Minimum detection confidence to consider.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold_high: float = 0.3,
        iou_threshold_low: float = 0.2,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.track_id_counter = 0

        # Statistics
        self.stats: Dict = defaultdict(int)

    def _assign(
        self,
        trackers_pred: np.ndarray,
        detections: np.ndarray,
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Hungarian assignment between predicted tracker boxes and detections.
        Returns: (matches, unmatched_trackers, unmatched_detections)
        """
        if len(trackers_pred) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(trackers_pred))), []

        iou_matrix = iou_batch(trackers_pred, detections)
        cost = 1 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost)

        matches, unmatched_t, unmatched_d = [], [], []
        matched_set = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_thresh:
                matches.append((r, c))
                matched_set.add((r, c))

        for i in range(len(trackers_pred)):
            if not any(m[0] == i for m in matches):
                unmatched_t.append(i)

        for j in range(len(detections)):
            if not any(m[1] == j for m in matches):
                unmatched_d.append(j)

        return matches, unmatched_t, unmatched_d

    def update(self, detections: List[List[float]]) -> List[Dict]:
        
        # Update tracker with current-frame detections.
        # Args:
        # detections: List of [x1, y1, x2, y2, conf, class_id]
        # Returns:
        # List of active track dicts: {id, bbox, conf, class_id, hits, age}
        
        self.frame_count += 1

        # --- Predict all existing tracks ---
        predicted = []
        to_del = []
        for i, t in enumerate(self.trackers):
            pred = t.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            else:
                predicted.append(pred)

        for i in reversed(to_del):
            self.trackers.pop(i)

        pred_array = np.array(predicted) if predicted else np.empty((0, 4))

        # --- Split detections by confidence ---
        high_dets, low_dets = [], []
        high_idx, low_idx = [], []
        for i, d in enumerate(detections):
            if d[4] >= self.high_conf_threshold:
                high_dets.append(d[:4])
                high_idx.append(i)
            elif d[4] >= self.low_conf_threshold:
                low_dets.append(d[:4])
                low_idx.append(i)

        high_array = np.array(high_dets) if high_dets else np.empty((0, 4))
        low_array = np.array(low_dets) if low_dets else np.empty((0, 4))

        # --- Stage 1: High-conf detections vs active tracks ---
        matches1, unmatched_t1, unmatched_d1 = self._assign(
            pred_array, high_array, self.iou_threshold_high
        )

        for ti, di in matches1:
            gi = high_idx[di]
            d = detections[gi]
            self.trackers[ti].update(high_dets[di], conf=float(d[4]), cls_id=int(d[5]))

        # --- Stage 2: Low-conf detections vs remaining tracks ---
        remaining_t_idx = unmatched_t1
        remaining_t_pred = pred_array[remaining_t_idx] if remaining_t_idx else np.empty((0, 4))

        matches2, unmatched_t2, unmatched_d2 = self._assign(
            remaining_t_pred, low_array, self.iou_threshold_low
        )

        for ri, di in matches2:
            ti = remaining_t_idx[ri]
            gi = low_idx[di]
            d = detections[gi]
            self.trackers[ti].update(low_dets[di], conf=float(d[4]), cls_id=int(d[5]))

        # --- Create new tracks for unmatched high-conf detections ---
        for di in unmatched_d1:
            gi = high_idx[di]
            d = detections[gi]
            new_t = KalmanBoxTracker(
                high_dets[di],
                track_id=self.track_id_counter,
                conf=float(d[4]),
                cls_id=int(d[5]),
            )
            self.track_id_counter += 1
            self.trackers.append(new_t)
            self.stats["tracks_created"] += 1

        # --- Collect active tracks ---
        active_tracks = []
        tracks_to_del = []

        for i, t in enumerate(self.trackers):
            if t.time_since_update > self.max_age:
                tracks_to_del.append(i)
                continue

            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = t.get_state()
                active_tracks.append({
                    "id": t.id,
                    "bbox": bbox,
                    "conf": t.conf,
                    "class_id": t.cls_id,
                    "hits": t.hits,
                    "age": t.age,
                    "time_since_update": t.time_since_update,
                })

        for i in reversed(tracks_to_del):
            self.trackers.pop(i)
            self.stats["tracks_deleted"] += 1

        self.stats["frame_count"] = self.frame_count
        self.stats["active_tracks"] = len(active_tracks)

        return active_tracks

    def get_stats(self) -> Dict:
        return dict(self.stats)
