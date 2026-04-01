# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage

**Author:** [Alok Kumar]
**Date:** April 2026
**Video Source:** [Insert public URL here]

---

## 1. Overview

This report describes the design and implementation of a multi-object tracking (MOT) pipeline applied to publicly available sports footage. The system reliably detects all players in every frame and maintains stable, persistent numeric IDs for each subject throughout the video—even under occlusion, rapid motion, and camera movement.

---

## 2. Model / Detector Used

**YOLOv8m (You Only Look Once, version 8 — medium variant)**
from the `ultralytics` library.

YOLOv8 is a single-stage anchor-free detector that predicts bounding boxes and class probabilities in a single forward pass. The medium variant (`yolov8m.pt`) was selected as the default because:

- It achieves **50.2 mAP** on COCO val2017, meaningfully outperforming the nano (37.3) and small (44.9) variants.
- It runs at **~18–25 fps** on a modern CPU at 640×640 input—sufficient for batch processing.
- Its weights (~26M parameters) are compact enough to load without GPU constraints.
- A `--model` argument allows hot-swapping to `yolov8l.pt` or `yolov8x.pt` for higher accuracy in demanding scenes.

Detection was restricted to **COCO class 0 (person)** via the `classes=[0]` parameter. The detection confidence threshold was set to **0.35** with NMS IoU **0.45**—conservative enough to reduce missed detections (false negatives) while filtering obvious noise.

---

## 3. Tracking Algorithm Used

**ByteTrack-style custom tracker** built on:
- A **Kalman filter** (constant-velocity motion model)
- **Hungarian algorithm** (optimal bipartite assignment)
- **Two-stage association** for high and low confidence detections

### 3.1 Kalman Filter
Each track maintains a state vector: `[cx, cy, area, aspect_ratio, vx, vy, v_area]`

On each frame, the Kalman filter:
1. **Predicts** the new position using the constant-velocity motion model.
2. **Updates** the state with any matched detection via the innovation residual.

This allows the tracker to bridge short occlusions (1–5 frames) without losing track IDs.

### 3.2 Two-Stage Hungarian Assignment
**Stage 1:** High-confidence detections (conf ≥ 0.5) are matched to active tracks using IoU-based cost matrix and the Hungarian algorithm (threshold 0.30).

**Stage 2:** Low-confidence detections (0.1 ≤ conf < 0.5) are matched only to *unmatched* tracks from Stage 1 (threshold 0.20). This recovers partially occluded or motion-blurred subjects that a strict confidence filter would discard.

**Unmatched high-conf detections** spawn new tracks.
**Unmatched tracks** accumulate `time_since_update`; they are pruned after `max_age=30` frames.

---

## 4. Why This Combination Was Selected

| Criterion | Decision |
|-----------|----------|
| Speed vs accuracy | YOLOv8m balances both on CPU; swap to YOLOv8l for GPU |
| Robustness in crowds | Two-stage association recovers partially visible players |
| ID stability | Kalman filter predicts position during occlusion |
| Simplicity | No external DeepSORT dependency; pure NumPy/SciPy implementation |
| Extensibility | Modular `detector.py` / `tracker.py` separation allows plug-in of any detector or ReID model |

ByteTrack was preferred over SORT (which uses only high-conf detections) because its two-stage design significantly reduces ID switches in dense scenes. DeepSORT was not used as it requires an external ReID model and GPU; ByteTrack's IoU-only approach proved sufficient for same-frame-resolution matching in well-lit sports footage.

---

## 5. How ID Consistency Is Maintained

ID consistency is maintained through a **predict → match → update** loop:

1. **Predict:** At every frame, all active tracks Kalman-predict their next bounding box location *before* seeing any detections. This means every known track has an estimated position even if the detector misses it.

2. **Match:** The Hungarian algorithm finds the globally optimal assignment between predicted boxes and detected boxes, minimizing total IoU-based cost. This ensures that the detector finding a player slightly off their predicted position doesn't trigger a false ID switch.

3. **Buffer:** Tracks that are unmatched remain alive for up to `max_age=30` frames (approximately 1 second at 30fps). If the player re-appears within that window, the same ID is reused—common during momentary occlusion by another player or the ball.

4. **Confirmation threshold:** New detections only become confirmed tracks after `min_hits=3` consecutive frames. This prevents flickering false detections from polluting the track pool with spurious IDs.

5. **Unique counter:** Track IDs are assigned monotonically from a global integer counter. Once a track is deleted, its ID is never reused.

---

## 6. Challenges Faced

**6.1 Occlusion in Tight Groups**
Players clustering together (e.g., a cricket batting crease or a football corner kick) cause bounding box overlap. The Kalman-predicted positions help disambiguate, but IoU matching can still swap IDs when two players are in contact for multiple frames.

**6.2 Similar Appearance**
Players in the same jersey and team strip are visually indistinguishable. Unlike DeepSORT, the IoU-only tracker has no appearance feature to fall back on when two similar-looking players cross paths. The primary mitigation is the Kalman velocity model, which predicts plausible forward trajectories.

**6.3 Fast Camera Motion**
During a pan or zoom, all Kalman predictions become inaccurate simultaneously. The constant-velocity model assumes a stationary camera; camera motion briefly degrades all predictions and can cause Stage-1 mismatches.

**6.4 Motion Blur**
High-speed cricket shots or sprinting athletes produce motion-blurred frames. YOLOv8 may detect the subject at lower confidence; the two-stage matching helps recover these as low-confidence detections.

**6.5 Partial Visibility at Frame Edges**
Players entering or exiting the frame cause short-lived track creation/deletion. Setting `min_hits=3` suppresses some spurious edge tracks but introduces a 3-frame delay before new players are confirmed.

---

## 7. Failure Cases Observed

- **Extended occlusion (> 1 second):** A player fully hidden behind a group for longer than `max_age` frames is assigned a new ID on re-emergence. This is the most common ID-switch scenario.
- **Identical-looking subjects crossing:** Two players in identical kits crossing paths may swap IDs because both the Kalman prediction and IoU match are ambiguous.
- **Very dense crowd scenes:** In marathon-type footage with 20+ subjects in frame, the IoU matrix becomes large and individual match quality degrades near identical bounding box sizes.

---

## 8. Possible Improvements

**Short-term:**
- **Appearance ReID:** Integrate a lightweight ReID model (e.g., OSNet) to create appearance embeddings. This allows re-identification after extended occlusion independent of position.
- **Camera motion compensation:** Use optical flow (e.g., Lucas-Kanade) to estimate global camera motion and subtract it from Kalman predictions during fast pans.
- **Adaptive `max_age`:** Dynamically increase `max_age` for high-confidence, long-lived tracks that have been visible for many frames.

**Medium-term:**
- **Bird's-eye-view projection:** Apply a homography transform to project all player positions onto a top-down pitch view for tactical analysis.
- **Team clustering:** Perform k-means clustering on player jersey colors (in HSV space) to automatically assign team labels.
- **Speed estimation:** Calibrate pixel-per-meter scale using known pitch dimensions, then compute velocity from trajectory derivatives.

**Long-term:**
- **Transformer-based tracking:** Explore MOTR or TrackFormer which jointly model detection and tracking in a single end-to-end architecture, eliminating the need for manual IoU matching.
- **Evaluation on MOT benchmarks:** Benchmark on MOT17/MOT20 using HOTA, IDF1, and MOTA metrics to formalize tracking quality independent of visual inspection.

---

## 9. Summary

The pipeline successfully implements multi-object detection (YOLOv8m) and persistent ID tracking (ByteTrack-style Kalman + Hungarian) on sports footage. It produces an annotated output video with bounding boxes, unique IDs, and trajectory trails, alongside a movement heatmap and per-frame count statistics. The modular architecture supports straightforward extension with appearance-based ReID, camera motion compensation, or alternative detectors.
