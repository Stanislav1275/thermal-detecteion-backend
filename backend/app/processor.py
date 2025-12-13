"""
Обработчик изображений для детекции людей.
"""

import os
from pathlib import Path
from typing import List
import cv2
from .detector import ThermalDetector
from .models import ImageResult, Detection


class ImageProcessor:
    """Обработчик изображений для детекции людей."""
    
    def __init__(self, detector: ThermalDetector):
        self.detector = detector
    
    def process_image(
        self,
        image_path: str,
        output_path: str,
        confidence: float = None
    ) -> ImageResult:
        """Обрабатывает одно изображение."""
        try:
            result = self.detector.detect(
                image_path,
                confidence=confidence,
                return_image=True
            )
            
            detections = []
            for det in result['detections']:
                detections.append(Detection(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    **{'class': det['class']}
                ))
            
            if result['total_detections'] > 0:
                if 'image_with_boxes' in result:
                    img_with_boxes = result['image_with_boxes']
                    img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, img_bgr)
                else:
                    import shutil
                    shutil.copy(image_path, output_path)
            
            return ImageResult(
                filename=os.path.basename(image_path),
                detections=detections,
                success=True
            )
        
        except Exception as e:
            return ImageResult(
                filename=os.path.basename(image_path),
                detections=[],
                success=False,
                error=str(e)
            )
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        confidence: float = None,
        save_only_with_detections: bool = True
    ) -> List[ImageResult]:
        """Обрабатывает пакет изображений."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            
            result = self.process_image(img_path, output_path, confidence)
            
            if save_only_with_detections:
                if result.total_detections == 0 and os.path.exists(output_path):
                    os.remove(output_path)
            else:
                if not os.path.exists(output_path):
                    import shutil
                    shutil.copy(img_path, output_path)
            
            results.append(result)
        
        return results
    
    @staticmethod
    def validate_image_format(file_path: str) -> bool:
        """Проверяет поддерживаемый формат изображения."""
        supported_formats = {'.tiff', '.tif', '.png', '.jpg', '.jpeg', '.webp'}
        ext = Path(file_path).suffix.lower()
        return ext in supported_formats
    
    @staticmethod
    def load_image(file_data: bytes, filename: str, save_path: str) -> str:
        """Сохраняет изображение из байтов на диск."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            f.write(file_data)
        
        return save_path

