from .ultralytics import YOLO
import cv2
import numpy as np
import os
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.conf_threshold = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU
        
    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold
    
    def detect(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold, device=self.device)
        return self._process_results(results)
    
    def detect_frame(self, frame):
        results = self.model(frame, conf=self.conf_threshold, device=self.device)
        return self._process_results(results)
    
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
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Set color based on class
            if class_name == 'helmet':
                color = (0, 255, 0)  # Green
            elif class_name == 'person':
                color = (255, 0, 0)  # Blue
            elif class_name == 'respirator':
                color = (153, 50, 204)  # Purple
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

# Initialize detector instance
detector = None

def get_detector():
    global detector
    if detector is None:
        model_path = os.path.join(os.path.dirname(__file__), 'static', 'models', 'yolov8-best.pt')
        detector = YOLODetector(model_path)
    return detector