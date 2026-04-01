import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SportDetector:
    # COCO class labels for reference
    COCO_CLASSES = {
        0: "person",
        32: "sports ball",
        36: "sports equipment",
    }

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else [0]  # person only

        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        
        #Run detection on a single BGR frame.
        #Returns:
        #List of detections: [[x1, y1, x2, y2, conf, class_id], ...]
        
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, cls_id])

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[List[float]]]:
        # Run detection on a batch of frames (more efficient for GPU).
        # Returns:
        #     List of detection lists, one per frame.

        results = self.model.predict(
            frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
            device=self.device,
        )

        all_detections = []
        for r in results:
            detections = []
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append([x1, y1, x2, y2, conf, cls_id])
            all_detections.append(detections)

        return all_detections

    def get_model_info(self) -> dict:
        # """Return metadata about the loaded model."""
        return {
            "model": self.model_path,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "classes": [self.COCO_CLASSES.get(c, str(c)) for c in self.classes],
        }
