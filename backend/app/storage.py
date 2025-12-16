import json
import os
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import ImageResult


class JobStorage:

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.getenv("STORAGE_DIR", "jobs")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _generate_unique_name(self, requested_name: Optional[str] = None) -> str:
        if requested_name is None:
            requested_name = f"Task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        existing_names = set()
        if self.base_dir.exists():
            for job_dir in self.base_dir.iterdir():
                if job_dir.is_dir():
                    manifest = self._load_manifest(job_dir.name)
                    if manifest and manifest.get("name"):
                        existing_names.add(manifest["name"])

        if requested_name not in existing_names:
            return requested_name

        index = 1
        while f"{requested_name}_{index}" in existing_names:
            index += 1

        return f"{requested_name}_{index}"

    def create_job(
        self,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        confidence_threshold: float = 0.62,
        total_images: int = 0,
    ) -> str:
        if job_id is None:
            job_id = str(uuid.uuid4())

        unique_name = self._generate_unique_name(name)

        job_dir = self.base_dir / job_id
        job_dir.mkdir(exist_ok=True)
        (job_dir / "input").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)

        manifest = {
            "job_id": job_id,
            "name": unique_name,
            "status": "queued",
            "total_images": total_images,
            "processed_images": 0,
            "images_with_detections": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "parameters": {"confidence_threshold": confidence_threshold},
        }

        self._save_manifest(job_id, manifest)
        return job_id

    def save_image(self, job_id: str, filename: str, image_data: bytes, is_input: bool = True):
        job_dir = self.base_dir / job_id
        subdir = "input" if is_input else "output"
        file_path = job_dir / subdir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

    def save_detections(self, job_id: str, detections: list):
        job_dir = self.base_dir / job_id
        detections_path = job_dir / "detections.json"

        detections_dict = []
        for img in detections:
            if isinstance(img, ImageResult):
                detections_dict.append(
                    {
                        "filename": img.filename,
                        "detections": [
                            {
                                "bbox": det.bbox,
                                "confidence": det.confidence,
                                "class": getattr(det, "class_name", "person"),
                            }
                            for det in img.detections
                        ],
                        "success": img.success,
                        "error": img.error,
                    }
                )
            else:
                detections_dict.append(img)

        with open(detections_path, "w") as f:
            json.dump(detections_dict, f, indent=2)

    def update_status(
        self,
        job_id: str,
        status: str,
        processed_images: Optional[int] = None,
        images_with_detections: Optional[int] = None,
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

    def update_job_name(self, job_id: str, new_name: str) -> bool:
        manifest = self._load_manifest(job_id)
        if manifest is None:
            return False

        old_name = manifest.get("name")

        existing_names = set()
        if self.base_dir.exists():
            for job_dir in self.base_dir.iterdir():
                if job_dir.is_dir():
                    m = self._load_manifest(job_dir.name)
                    if m and m.get("name") and m.get("name") != old_name:
                        existing_names.add(m["name"])

        if new_name in existing_names:
            index = 1
            while f"{new_name}_{index}" in existing_names:
                index += 1
            unique_name = f"{new_name}_{index}"
        else:
            unique_name = new_name

        manifest["name"] = unique_name
        self._save_manifest(job_id, manifest)
        return True

    def list_jobs(self) -> List[Dict]:
        jobs: List[Dict] = []
        if not self.base_dir.exists():
            return jobs

        for job_dir in self.base_dir.iterdir():
            if job_dir.is_dir():
                manifest = self._load_manifest(job_dir.name)
                if manifest:
                    jobs.append(manifest)

        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jobs

    def get_status(self, job_id: str) -> Optional[Dict]:
        return self._load_manifest(job_id)

    def get_detections(self, job_id: str) -> Optional[List[Dict]]:
        job_dir = self.base_dir / job_id
        detections_path = job_dir / "detections.json"

        if not detections_path.exists():
            return None

        with open(detections_path, "r") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def get_input_image_path(self, job_id: str, filename: str) -> Optional[Path]:
        job_dir = self.base_dir / job_id
        image_path = job_dir / "input" / filename

        if image_path.exists():
            return image_path
        return None

    def get_output_image_path(self, job_id: str, filename: str) -> Optional[Path]:
        job_dir = self.base_dir / job_id
        image_path = job_dir / "output" / filename

        if image_path.exists():
            return image_path
        return None

    def create_output_zip(
        self,
        job_id: str,
        use_original: bool = False,
        min_confidence: float = 0.0,
        only_with_detections: bool = False,
    ) -> Optional[Path]:
        job_dir = self.base_dir / job_id
        source_dir = job_dir / ("input" if use_original else "output")

        if not source_dir.exists():
            return None

        detections = self.get_detections(job_id) or []

        detection_map = {}
        for det in detections:
            filename = det.get("filename")
            if filename:
                dets = det.get("detections", [])
                max_confidence = max([d.get("confidence", 0.0) for d in dets], default=0.0)
                detection_map[filename] = {
                    "detections": dets,
                    "max_confidence": max_confidence,
                    "has_detections": len(dets) > 0,
                }

        source_files = list(source_dir.glob("*"))
        image_files = [
            f
            for f in source_files
            if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
        ]

        filtered_files = []
        for image_file in image_files:
            filename = image_file.name
            file_info = detection_map.get(filename, {})
            max_conf = file_info.get("max_confidence", 0.0)
            has_detections = file_info.get("has_detections", False)

            if only_with_detections and not has_detections:
                continue

            if max_conf >= min_confidence:
                filtered_files.append(image_file)

        if not filtered_files:
            return None

        zip_suffix = f"{'original' if use_original else 'processed'}"
        if only_with_detections:
            zip_suffix += "_with_detections"
        if min_confidence > 0.0:
            zip_suffix += f"_conf{min_confidence:.2f}"

        zip_path = job_dir / f"results_{zip_suffix}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for image_file in filtered_files:
                zipf.write(image_file, arcname=image_file.name)

        return zip_path

    def _load_manifest(self, job_id: str) -> Optional[Dict]:
        job_dir = self.base_dir / job_id
        manifest_path = job_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        with open(manifest_path, "r") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_manifest(self, job_id: str, manifest: Dict):
        job_dir = self.base_dir / job_id
        manifest_path = job_dir / "manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def delete_job(self, job_id: str) -> bool:
        """
        Удаляет задачу и все связанные с ней данные.
        Удаляет всю директорию задачи, включая:
        - input/ - оригинальные изображения
        - output/ - обработанные изображения
        - detections.json - файл с детекциями
        - manifest.json - манифест задачи
        - results_*.zip - ZIP архивы результатов

        Args:
            job_id: ID задачи для удаления

        Returns:
            bool: True если задача была удалена, False если не найдена или произошла ошибка
        """
        job_dir = self.base_dir / job_id

        if not job_dir.exists():
            return False

        import shutil

        try:
            shutil.rmtree(job_dir)
            return True
        except Exception as e:
            print(f"Ошибка при удалении задачи {job_id}: {e}")
            return False
