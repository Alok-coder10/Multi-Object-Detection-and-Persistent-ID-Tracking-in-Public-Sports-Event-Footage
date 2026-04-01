import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import TrackingPipeline


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking for Sports Video"
    )

    # Input/Output
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to JSON config file")

    # Model
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model variant (n=nano/fastest, x=xlarge/most accurate)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device",
    )

    # Detection
    parser.add_argument("--det-conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.45, help="Detection NMS IoU threshold")

    # Tracking
    parser.add_argument("--max-age", type=int, default=30, help="Max frames to keep lost tracks")
    parser.add_argument("--min-hits", type=int, default=3, help="Min hits to confirm a track")
    parser.add_argument("--iou-high", type=float, default=0.3, help="High-conf matching IoU threshold")
    parser.add_argument("--iou-low", type=float, default=0.2, help="Low-conf matching IoU threshold")

    # Processing
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1=all frames, 2=every other, etc.)",
    )

    # Visualization
    parser.add_argument("--no-trail", action="store_true", help="Disable trajectory trails")
    parser.add_argument("--no-conf", action="store_true", help="Hide confidence scores")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("run_tracker")
    logger.info("=" * 60)
    logger.info("  Sports Multi-Object Tracking Pipeline")
    logger.info("=" * 60)

    # Build config
    config = {
        "output_dir": args.output_dir,
        "model": args.model,
        "device": args.device,
        "det_conf": args.det_conf,
        "det_iou": args.det_iou,
        "classes": [0, 32],
        "max_age": args.max_age,
        "min_hits": args.min_hits,
        "iou_high": args.iou_high,
        "iou_low": args.iou_low,
        "high_conf_thresh": 0.5,
        "low_conf_thresh": 0.1,
        "frame_skip": args.frame_skip,
        "show_trail": not args.no_trail,
        "box_thickness": 2,
        "trail_thickness": 2,
        "show_conf": not args.no_conf,
    }

    # Override with config file if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)
        logger.info(f"Loaded config from {args.config}")

    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Run pipeline
    pipeline = TrackingPipeline(config)
    stats = pipeline.run(args.video)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Unique track IDs assigned : {stats['unique_track_ids']}")
    logger.info(f"  Max simultaneous tracks   : {stats['max_simultaneous_tracks']}")
    logger.info(f"  Avg simultaneous tracks   : {stats['avg_simultaneous_tracks']}")
    logger.info(f"  Processing speed          : {stats['processing_fps']} fps")
    logger.info(f"  Total processing time     : {stats['processing_time_s']}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
