
import json
import uuid
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from .models import ImageResult


class JobStorage:
    
    def __init__(self, base_dir: str = "jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_job(
        self,
        job_id: Optional[str] = None,
        confidence_threshold: float = 0.5,
        total_images: int = 0
    ) -> str:
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        job_dir = self.base_dir / job_id
        job_dir.mkdir(exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        
        manifest = {
            "job_id": job_id,
            "status": "queued",
            "total_images": total_images,
            "processed_images": 0,
            "images_with_detections": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "parameters": {
                "confidence_threshold": confidence_threshold
            }
        }
        
        self._save_manifest(job_id, manifest)
        return job_id
    
    def save_image(self, job_id: str, filename: str, image_data: bytes, is_input: bool = True):
        job_dir = self.base_dir / job_id
        subdir = "input" if is_input else "output"
        file_path = job_dir / subdir / filename
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
    
    def save_detections(self, job_id: str, detections: list):
        job_dir = self.base_dir / job_id
        detections_path = job_dir / "detections.json"
        
        detections_dict = []
        for img in detections:
            if isinstance(img, ImageResult):
                detections_dict.append({
                    "filename": img.filename,
                    "detections": [
                        {
                            "bbox": det.bbox,
                            "confidence": det.confidence,
                            "class": getattr(det, 'class_name', 'person')
                        }
                        for det in img.detections
                    ],
                    "success": img.success,
                    "error": img.error
                })
            else:
                detections_dict.append(img)
        
        with open(detections_path, 'w') as f:
            json.dump(detections_dict, f, indent=2)
    
    def update_status(
        self,
        job_id: str,
        status: str,
        processed_images: Optional[int] = None,
        images_with_detections: Optional[int] = None
    ):
        manifest = self._load_manifest(job_id)
        if manifest is None:
            return
        
        manifest["status"] = status
        
        if processed_images is not None:
            manifest["processed_images"] = processed_images
        
        if images_with_detections is not None:
            manifest["images_with_detections"] = images_with_detections
        
        if status in ["completed", "failed"]:
            manifest["completed_at"] = datetime.now().isoformat()
        
        self._save_manifest(job_id, manifest)
    
    def get_status(self, job_id: str) -> Optional[Dict]:
        return self._load_manifest(job_id)
    
    def get_detections(self, job_id: str) -> Optional[list]:
        job_dir = self.base_dir / job_id
        detections_path = job_dir / "detections.json"
        
        if not detections_path.exists():
            return None
        
        with open(detections_path, 'r') as f:
            return json.load(f)
    
    def get_output_image_path(self, job_id: str, filename: str) -> Optional[Path]:
        job_dir = self.base_dir / job_id
        image_path = job_dir / "output" / filename
        
        if image_path.exists():
            return image_path
        return None
    
    def _load_manifest(self, job_id: str) -> Optional[Dict]:
        job_dir = self.base_dir / job_id
        manifest_path = job_dir / "manifest.json"
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def _save_manifest(self, job_id: str, manifest: Dict):
        job_dir = self.base_dir / job_id
        manifest_path = job_dir / "manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

