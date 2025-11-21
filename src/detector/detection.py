import json
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

class ObjectDetector:
    def __init__(self, config_path="src/config/detector.yaml", device="cpu"):
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        self.model = YOLO(cfg["model_path"])
        self.conf = cfg["conf_threshold"]
        self.device = device
        with open(cfg["label_path"], "r", encoding="utf-8") as f:
            self.labels_vi = json.load(f)
        self.model.to(self.device)

    def detect(self, image_path):
        results = self.model(image_path, conf=self.conf, device=self.device, verbose=False)
        outputs = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label_en = self.model.names[cls]
            label_vi = self.labels_vi.get(label_en, label_en)
            conf = float(box.conf[0])
            outputs.append(label_vi)
        return set(outputs)