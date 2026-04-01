# 🏏 Sports Multi-Object Detection & Persistent ID Tracking

A production-ready computer vision pipeline for detecting and tracking multiple players/athletes in sports video footage with stable, persistent IDs across frames.

---

## 📌 Source Video

> **Video Used:** *(Replace with your actual video URL)*
> e.g. `https://www.youtube.com/shorts/KYGj2xHDQAg`
>
> **Category:** Table Tennis *(choose one)*
> **License:** Publicly accessible footage

---

## 📁 Project Structure

```
sports_tracker/
├── src/
│   ├── detector.py        # YOLOv8 detection wrapper
│   ├── tracker.py         # ByteTrack-style Kalman + Hungarian tracker
│   ├── visualizer.py      # Annotation, trails, heatmap utilities
│   └── pipeline.py        # Main orchestrator
├── notebooks/
│   └── tracking_pipeline.ipynb   # End-to-end interactive notebook
├── configs/
│   └── default.json       # Default configuration
├── outputs/               # Generated on run
│   ├── <video>_tracked.mp4
│   ├── <video>_heatmap.jpg
│   └── <video>_stats.json
├── run_tracker.py         # CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd sports_tracker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) GPU support

For NVIDIA GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 How to Run

### Option A — Command Line

```bash
# Basic run (CPU, all defaults)
python run_tracker.py --video path/to/video.mp4

# Use GPU, larger model, skip every other frame
python run_tracker.py --video path/to/video.mp4 \
    --model yolov8l.pt \
    --device cuda \
    --frame-skip 2

# Tune tracking parameters
python run_tracker.py --video path/to/video.mp4 \
    --det-conf 0.4 \
    --max-age 40 \
    --min-hits 2 \
    --no-conf

# Use a config file
python run_tracker.py --video path/to/video.mp4 --config configs/default.json
```

### Option B — Jupyter Notebook

```bash
jupyter notebook notebooks/tracking_pipeline.ipynb
```

Run all cells top-to-bottom. The notebook includes:
- Video download via `yt-dlp`
- Sample frame detection visualization
- Full pipeline execution
- Count-over-time plots
- Heatmap display
- YOLOv8n vs YOLOv8m comparison

### Option C — Python API

```python
from src.pipeline import TrackingPipeline

config = {
    "output_dir": "outputs",
    "model": "yolov8m.pt",
    "device": "cpu",
    "frame_skip": 1,
    "show_trail": True,
}

pipeline = TrackingPipeline(config)
stats = pipeline.run("path/to/video.mp4")
print(stats)
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.0 | YOLOv8 detection |
| `opencv-python` | ≥4.8 | Video I/O and drawing |
| `numpy` | ≥1.24 | Numerical operations |
| `scipy` | ≥1.10 | Hungarian assignment |
| `matplotlib` | ≥3.7 | Visualization / plots |
| `yt-dlp` | any | Video download (notebook only) |

Full list in `requirements.txt`.

---

## ⚡ CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | *required* | Input video path |
| `--output-dir` | `outputs` | Where to save results |
| `--model` | `yolov8m.pt` | YOLO model variant |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--det-conf` | `0.35` | Detection confidence threshold |
| `--det-iou` | `0.45` | Detection NMS IoU |
| `--max-age` | `30` | Frames before lost track deleted |
| `--min-hits` | `3` | Detections to confirm track |
| `--iou-high` | `0.30` | High-conf matching threshold |
| `--iou-low` | `0.20` | Low-conf matching threshold |
| `--frame-skip` | `1` | Process every Nth frame |
| `--no-trail` | false | Disable trajectory trails |
| `--no-conf` | false | Hide confidence in labels |

---

## 🧠 Model & Tracker Choices

### Detection: YOLOv8m (medium)
- Chosen for the best balance of **speed vs accuracy** on CPU
- Detects `person` class (COCO class 0) by default
- Swap `yolov8n.pt` for faster processing, `yolov8l.pt` for higher recall

### Tracker: ByteTrack-style Custom Implementation
- **Kalman filter** for motion prediction (constant-velocity model)
- **Two-stage Hungarian assignment**: high-confidence detections first, then low-confidence recovery
- **Re-ID buffer**: tracks survive occlusion up to `max_age` frames

---

## ✅ Assumptions

1. Input video contains human subjects (players, athletes, participants).
2. Camera may pan/zoom but subjects are visible for meaningful durations.
3. `yolov8m.pt` weights are auto-downloaded by `ultralytics` on first run.
4. CPU is used by default; GPU strongly recommended for real-time processing.

---

## ⚠️ Limitations

- **Re-identification:** Pure IoU-based matching struggles when subjects look identical (e.g. same jersey) after long occlusion. Appearance features (ReID) would improve this.
- **Crowd density:** Very dense crowds (e.g. marathon starts) increase ID switching.
- **Fast motion blur:** Very fast ball/limb movement may cause missed detections.
- **Camera motion:** Fast camera pans can confuse Kalman velocity estimates temporarily.
- **Occlusion:** Extended occlusion (> `max_age` frames) causes ID reassignment on re-emergence.

---

## 📊 Outputs

| File | Description |
|------|-------------|
| `<video>_tracked.mp4` | Annotated video with bounding boxes, IDs, trails |
| `<video>_heatmap.jpg` | Movement heatmap (accumulated across all frames) |
| `<video>_stats.json` | JSON with frame counts, track stats, model info |

---

## 🔧 Possible Improvements

- **Appearance ReID:** Add feature-based re-identification (OSNet, FastReID) for long-term occlusion recovery
- **Bird's-eye projection:** Homography transform for top-view positional analysis
- **Team clustering:** Jersey color clustering (k-means in HSV) for team assignment
- **Speed estimation:** Pixel-per-meter calibration for velocity estimation
- **Model ensemble:** Combine YOLO + RT-DETR for higher recall in crowd scenes
