"""
YOLO детектор для детекции людей на тепловизионных изображениях.
"""

import os
from typing import List, Dict, Optional
import torch
from ultralytics import YOLO
import cv2


class ThermalDetector:
    """Детектор людей на тепловизионных изображениях с использованием YOLOv8."""
    
    def __init__(
        self,
        model_path: str = "training/models/best.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Убедитесь, что модель обучена и сохранена в training/models/best.pt"
            )
        
        print(f"Загрузка модели: {model_path}")
        print(f"Устройство: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.class_names = self.model.names
        self.person_class_id = None
        
        for class_id, class_name in self.class_names.items():
            if class_name.lower() == 'person':
                self.person_class_id = class_id
                break
        
        if self.person_class_id is None:
            print("Предупреждение: класс 'person' не найден в модели")
    
    def detect(
        self,
        image_path: str,
        confidence: Optional[float] = None,
        return_image: bool = False
    ) -> Dict:
        """Детектирует людей на изображении."""
        if confidence is None:
            confidence = self.confidence_threshold
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        results = self.model.predict(
            image_path,
            conf=confidence,
            device=self.device,
            verbose=False
        )
        
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            for box, cls_id, conf in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            ):
                cls_id_int = int(cls_id)
                conf_float = float(conf)
                
                if cls_id_int == self.person_class_id:
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf_float,
                        'class': 'person'
                    })
        
        result_dict = {
            'detections': detections,
            'total_detections': len(detections)
        }
        
        if return_image:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 140, 0), 2)
                
                label = f"person {conf:.2f}"
                cv2.putText(
                    img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2
                )
            
            result_dict['image_with_boxes'] = img_rgb
        
        return result_dict
    
    def detect_batch(
        self,
        image_paths: List[str],
        confidence: Optional[float] = None
    ) -> List[Dict]:
        """Детектирует людей на нескольких изображениях."""
        results = []
        for img_path in image_paths:
            try:
                result = self.detect(img_path, confidence=confidence)
                result['filename'] = os.path.basename(img_path)
                result['success'] = True
            except Exception as e:
                result = {
                    'filename': os.path.basename(img_path),
                    'success': False,
                    'error': str(e),
                    'detections': [],
                    'total_detections': 0
                }
            results.append(result)
        
        return results

