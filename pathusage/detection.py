"""Object detection for camera-trap images.

Primary detector: MegaDetector V6 (MDV6-yolov9-c) via PytorchWildlife.
Fallback / animal subclass refinement: YOLOv8n (Ultralytics, COCO classes).

Category convention (MegaDetector): 1 = animal, 2 = person, 3 = vehicle.
Each detection is returned as a dict:
    {"category", "label", "conf", "bbox": [x1,y1,x2,y2], "cx", "cy", ["animal"]}
where cx, cy is the bounding-box centre in pixel coordinates.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# COCO class ids used by the YOLOv8n fallback
COCO_DOG     = 16
COCO_HORSE   = 17
COCO_ANIMAL  = {16, 17, 18, 19, 20, 21, 22, 23}   # dog, horse, sheep, cow, ...
COCO_VEHICLE = {1, 2, 3, 5, 7}                     # bicycle, car, motorbike, bus, truck

_LABELS = {1: "animal", 2: "person", 3: "vehicle"}


class Detector:
    """MegaDetector V6 with a YOLOv8n fallback.

    If MegaDetector cannot be loaded (no GPU / package missing), the detector
    transparently falls back to YOLOv8n. YOLOv8n is also used to refine animal
    detections into dog / horse / wildlife.
    """

    def __init__(self, conf=0.20):
        self.conf = conf
        self.use_md = False
        self.md = None
        self.yolo = None
        self._warned = False
        try:
            import torch
            from PytorchWildlife.models import detection as pw
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.md = pw.MegaDetectorV6(device=dev, pretrained=True,
                                        version="MDV6-yolov9-c")
            self.use_md = True
            print(f"MegaDetector V6 loaded ({dev})")
        except Exception as e:
            print(f"WARNING: MegaDetector not available ({e})")
        try:
            from ultralytics import YOLO
            self.yolo = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"WARNING: YOLOv8n not available ({e})")
        if not self.use_md and self.yolo is None:
            raise RuntimeError(
                "No detector available. Install either PytorchWildlife "
                "(MegaDetector) or ultralytics (YOLOv8n).")

    # -- public API ----------------------------------------------------------
    def detect(self, img_path):
        """Return the list of detections for one image."""
        if self.use_md:
            try:
                return self._detect_md(img_path)
            except Exception as e:
                if not self._warned:
                    print(f"  MegaDetector failed ({e}) -- falling back to YOLOv8n")
                    self._warned = True
        return self._detect_yolo(img_path)

    # -- MegaDetector --------------------------------------------------------
    def _detect_md(self, img_path):
        res = self.md.single_image_detection(str(img_path))
        sv = res.get("detections")
        dets = []
        if sv is None or len(sv) == 0:
            return dets
        boxes = np.asarray(sv.xyxy)
        classes = np.asarray(sv.class_id)
        confs = np.asarray(sv.confidence)
        for i in range(len(sv)):
            if float(confs[i]) < self.conf:
                continue
            cat = int(classes[i]) + 1            # 1=animal, 2=person, 3=vehicle
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            dets.append({"category": cat, "label": _LABELS.get(cat, "?"),
                         "conf": float(confs[i]), "bbox": [x1, y1, x2, y2],
                         "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
        return self._refine_animals(img_path, dets)

    def _refine_animals(self, img_path, dets):
        """Use YOLOv8n to refine animal detections into dog / horse / wildlife."""
        if self.yolo is None or not any(d["category"] == 1 for d in dets):
            for d in dets:
                if d["category"] == 1:
                    d["animal"] = "wildlife"
            return dets
        frame = cv2.imread(str(img_path)) if cv2 is not None else None
        sub = []
        if frame is not None:
            res = self.yolo.predict(frame, conf=0.25, verbose=False)[0]
            if res.boxes is not None:
                for b in res.boxes:
                    cid = int(b.cls.item())
                    if cid == COCO_DOG:
                        sub.append("dog")
                    elif cid == COCO_HORSE:
                        sub.append("horse")
        for d in dets:
            if d["category"] == 1:
                d["animal"] = ("dog" if "dog" in sub else
                               "horse" if "horse" in sub else "wildlife")
        return dets

    # -- YOLOv8n fallback ----------------------------------------------------
    def _detect_yolo(self, img_path):
        if self.yolo is None:
            return []
        frame = cv2.imread(str(img_path)) if cv2 is not None else None
        if frame is None:
            return []
        res = self.yolo.predict(frame, conf=0.25, verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        for b in res.boxes:
            cid, cf = int(b.cls.item()), float(b.conf.item())
            if cid == 0:
                cat, ani = 2, None
            elif cid in COCO_ANIMAL:
                cat = 1
                ani = ("dog" if cid == COCO_DOG else
                       "horse" if cid == COCO_HORSE else "wildlife")
            elif cid in COCO_VEHICLE:
                cat, ani = 3, None
            else:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            d = {"category": cat, "label": _LABELS[cat], "conf": cf,
                 "bbox": [x1, y1, x2, y2], "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2}
            if ani:
                d["animal"] = ani
            dets.append(d)
        return dets
