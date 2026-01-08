from .ultralytics import YOLO
import cv2
import numpy as np
import os
import torch

class YOLODetector:
    def __init__(self, model_path):
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在！请检查路径：{model_path}")
        
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.conf_threshold = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU
        
        # 可选：强制将模型加载到GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold
    
    def detect(self, image_path):
        results = self.model(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device
        )
        return self._process_results(results)
    
    def detect_frame(self, frame):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device
        )
        return self._process_results(results)
    
    def detect_and_plot(self, frame):
        """检测并直接生成带标注的图片"""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device
        )
        
        # 使用result.plot()生成标注后的图片
        detected_frame = results[0].plot()
        
        # 处理检测结果
        detections = self._process_results(results)
        
        return detected_frame, detections
    
    def _process_results(self, results):
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': self.class_names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        return detections
    
    def draw_detections(self, frame, detections):
        # 保留原有方法，兼容旧代码
        return self.detect_and_plot(frame)[0]

# Initialize detector instance
detector = None

def get_detector():
    global detector
    if detector is None:
        model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'yolov8-best.pt')
        detector = YOLODetector(model_path)
    return detector