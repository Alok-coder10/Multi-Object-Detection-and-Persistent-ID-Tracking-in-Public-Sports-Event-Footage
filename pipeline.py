import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

from detector import SportDetector
from tracker import ByteTracker
from visualizer import Annotator, HeatmapAccumulator

logger = logging.getLogger(__name__)


class TrackingPipeline:
    # End-to-end video tracking pipeline.
    # Args:
    # config: dict of pipeline configuration (see configs/default.json)

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.detector = SportDetector(
            model_path=config.get("model", "yolov8m.pt"),
            confidence=config.get("det_conf", 0.35),
            iou_threshold=config.get("det_iou", 0.45),
            device=config.get("device", "cpu"),
            classes=config.get("classes", [0]),
        )

        self.tracker = ByteTracker(
            max_age=config.get("max_age", 30),
            min_hits=config.get("min_hits", 3),
            iou_threshold_high=config.get("iou_high", 0.3),
            iou_threshold_low=config.get("iou_low", 0.2),
            high_conf_threshold=config.get("high_conf_thresh", 0.5),
            low_conf_threshold=config.get("low_conf_thresh", 0.1),
        )

        self.annotator = Annotator(
            show_trail=config.get("show_trail", True),
            trail_thickness=config.get("trail_thickness", 2),
            box_thickness=config.get("box_thickness", 2),
            show_conf=config.get("show_conf", True),
        )

        self.heatmap: Optional[HeatmapAccumulator] = None

        # Stats tracking
        self.count_over_time = []  # [(frame_num, count)]
        self.all_track_ids = set()

    def run(self, video_path: str) -> Dict[str, Any]:
        # Returns:
        # Summary statistics dict.
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_skip = self.config.get("frame_skip", 1)

        logger.info(
            f"Video: {video_path.name} | {width}x{height} @ {fps:.1f} fps | {total_frames} frames"
        )

        # Initialize heatmap
        self.heatmap = HeatmapAccumulator((height, width), blur_radius=30)

        # Output video writer
        out_path = self.output_dir / f"{video_path.stem}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / frame_skip
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

        frame_num = 0
        processed = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames if configured
            if (frame_num - 1) % frame_skip != 0:
                continue

            processed += 1

            # --- Detection ---
            detections = self.detector.detect(frame)

            # --- Tracking ---
            tracks = self.tracker.update(detections)

            # --- Statistics ---
            self.count_over_time.append((frame_num, len(tracks)))
            for t in tracks:
                self.all_track_ids.add(t["id"])

            # --- Heatmap ---
            self.heatmap.update(tracks)

            # --- Annotation ---
            annotated = self.annotator.draw_frame(
                frame, tracks, frame_num, total_frames
            )
            writer.write(annotated)

            # Progress log every 100 processed frames
            if processed % 100 == 0:
                elapsed = time.time() - t_start
                fps_proc = processed / elapsed
                logger.info(
                    f"  Frame {frame_num}/{total_frames} | "
                    f"Tracks: {len(tracks)} | "
                    f"Speed: {fps_proc:.1f} fps"
                )

        cap.release()
        writer.release()

        elapsed = time.time() - t_start
        logger.info(f"Done. Processed {processed} frames in {elapsed:.1f}s")

        # --- Save heatmap ---
        heatmap_path = self.output_dir / f"{video_path.stem}_heatmap.jpg"
        self.heatmap.save(str(heatmap_path), (height, width))

        # --- Save stats ---
        stats = self._build_stats(video_path.name, fps, total_frames, processed, elapsed)
        stats_path = self.output_dir / f"{video_path.stem}_stats.json"
        stats["heatmap_path"] = str(heatmap_path.resolve())
        stats["tracked_video_path"] = str(out_path.resolve())
        stats["stats_json_path"] = str(stats_path.resolve())
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Output video : {out_path}")
        logger.info(f"Heatmap      : {heatmap_path}")
        logger.info(f"Stats        : {stats_path}")

        return stats

    def _build_stats(
        self,
        video_name: str,
        fps: float,
        total_frames: int,
        processed: int,
        elapsed: float,
    ) -> Dict[str, Any]:
        counts = [c for _, c in self.count_over_time]

        return {
            "video": video_name,
            "total_frames": total_frames,
            "processed_frames": processed,
            "processing_time_s": round(elapsed, 2),
            "processing_fps": round(processed / elapsed, 2),
            "unique_track_ids": len(self.all_track_ids),
            "max_simultaneous_tracks": max(counts) if counts else 0,
            "avg_simultaneous_tracks": round(np.mean(counts), 2) if counts else 0,
            "tracker_stats": self.tracker.get_stats(),
            "count_over_time": self.count_over_time[::10],  # sampled
            "model_info": self.detector.get_model_info(),
        }
