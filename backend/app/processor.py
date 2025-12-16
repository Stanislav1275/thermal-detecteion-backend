
import os
import zipfile
import io
import re
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from .detector import ThermalDetector
from .models import ImageResult, Detection


class ImageProcessor:
    
    def __init__(self, detector: ThermalDetector):
        self.detector = detector
    
    def process_image(
        self,
        image_path: str,
        output_path: str,
        confidence: float = None
    ) -> ImageResult:
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            test_img = cv2.imread(image_path)
            if test_img is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}. Файл может быть поврежден или иметь неподдерживаемый формат.")
            
            if test_img.size == 0:
                raise ValueError(f"Изображение пустое: {image_path}")
            
            result = self.detector.detect(
                image_path,
                confidence=confidence,
                return_image=False
            )
            
            detections = []
            for det in result['detections']:
                det_confidence = det['confidence']
                if confidence is not None and det_confidence < confidence:
                    continue
                detections.append(Detection(
                    bbox=det['bbox'],
                    confidence=det_confidence,
                    class_name=det.get('class_name', det.get('class', 'person'))
                ))
            
            import shutil
            shutil.copy(image_path, output_path)
            
            return ImageResult(
                filename=os.path.basename(image_path),
                detections=detections,
                success=True
            )
        
        except Exception as e:
            error_msg = str(e)
            if "need at least one array to stack" in error_msg.lower():
                error_msg = f"Изображение некорректно или повреждено: {os.path.basename(image_path)}"
            return ImageResult(
                filename=os.path.basename(image_path),
                detections=[],
                success=False,
                error=error_msg
            )
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        confidence: float = None,
        save_only_with_detections: bool = True
    ) -> List[ImageResult]:
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            
            result = self.process_image(img_path, output_path, confidence)
            
            if save_only_with_detections:
                if len(result.detections) == 0 and os.path.exists(output_path):
                    os.remove(output_path)
            else:
                if not os.path.exists(output_path):
                    import shutil
                    shutil.copy(img_path, output_path)
            
            results.append(result)
        
        return results
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        if not filename:
            return "unnamed_file"
        
        filename = str(filename)
        
        filename = os.path.basename(filename)
        
        if not filename or filename in ('.', '..'):
            return "unnamed_file"
        
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
        
        filename = filename.strip('. ')
        
        if not filename:
            return "unnamed_file"
        
        return filename
    
    @staticmethod
    def get_mime_type(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    @staticmethod
    def validate_image_format(file_path: str) -> bool:
        supported_formats = {'.tiff', '.tif', '.png', '.jpg', '.jpeg', '.webp'}
        ext = Path(file_path).suffix.lower()
        return ext in supported_formats
    
    @staticmethod
    def load_image(file_data: bytes, filename: str, save_path: str) -> str:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            f.write(file_data)
        
        return save_path
    
    @staticmethod
    def is_zip_file(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in {'.zip'}
    
    @staticmethod
    def extract_zip_archive(zip_data: bytes, extract_to: str) -> List[Tuple[str, str]]:
        os.makedirs(extract_to, exist_ok=True)
        extracted_files = []
        extraction_errors = []
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith('/'):
                        continue
                    
                    sanitized_member = ImageProcessor.sanitize_filename(member)
                    filename = os.path.basename(sanitized_member)
                    if not filename or filename == "unnamed_file":
                        continue
                    
                    if not ImageProcessor.validate_image_format(filename):
                        continue
                    
                    try:
                        file_data = zip_ref.read(member)
                        if not file_data or len(file_data) == 0:
                            extraction_errors.append(f"{member}: файл пустой")
                            continue
                        
                        sanitized_filename = ImageProcessor.sanitize_filename(filename)
                        extract_path = os.path.join(extract_to, sanitized_filename)
                        
                        if os.path.exists(extract_path):
                            base, ext = os.path.splitext(sanitized_filename)
                            counter = 1
                            while os.path.exists(extract_path):
                                extract_path = os.path.join(extract_to, f"{base}_{counter}{ext}")
                                counter += 1
                        
                        with open(extract_path, 'wb') as f:
                            f.write(file_data)
                        
                        test_img = cv2.imread(extract_path)
                        if test_img is None:
                            os.remove(extract_path)
                            extraction_errors.append(f"{member}: не удалось прочитать как изображение")
                            continue
                        
                        extracted_files.append((member, extract_path))
                    except Exception as e:
                        error_msg = str(e)
                        if "need at least one array to stack" in error_msg.lower():
                            error_msg = f"{member}: изображение некорректно или повреждено"
                        extraction_errors.append(f"{member}: {error_msg}")
                        continue
        
        except zipfile.BadZipFile:
            raise ValueError("Некорректный ZIP-архив")
        except Exception as e:
            raise ValueError(f"Ошибка при распаковке архива: {e}")
        
        if not extracted_files:
            error_msg = "ZIP-архив не содержит валидных изображений"
            if extraction_errors:
                error_msg += f". Ошибки при извлечении: {'; '.join(extraction_errors[:5])}"
            raise ValueError(error_msg)
        
        return extracted_files

