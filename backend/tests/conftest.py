import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from app.processor import ImageProcessor
from app.storage import JobStorage

try:
    from app.detector import ThermalDetector
except ImportError:
    ThermalDetector = None


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Создает временную директорию для тестов."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_storage(temp_dir: Path) -> JobStorage:
    """Создает временное хранилище для тестов."""
    storage_dir = temp_dir / "test_jobs"
    return JobStorage(base_dir=str(storage_dir))


@pytest.fixture
def test_image_path(temp_dir: Path) -> Path:
    """Создает тестовое изображение."""
    img_path = temp_dir / "test_image.jpg"
    img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def mock_yolo_model():
    """Создает мок YOLO модели."""
    mock_model = Mock()
    mock_model.names = {0: "person", 1: "car"}
    mock_model.to = Mock(return_value=mock_model)

    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = np.array([[10, 20, 100, 200], [150, 160, 250, 300]])
    mock_box.cls = np.array([0, 0])
    mock_box.conf = np.array([0.8, 0.9])
    mock_box.__len__ = Mock(return_value=2)
    mock_result.boxes = mock_box
    mock_result.boxes.xyxy = mock_box.xyxy
    mock_result.boxes.cls = mock_box.cls
    mock_result.boxes.conf = mock_box.conf

    mock_model.predict = Mock(return_value=[mock_result])

    return mock_model


@pytest.fixture
def mock_detector(mock_yolo_model, monkeypatch):
    """Создает мок детектора."""
    if ThermalDetector is None:
        from unittest.mock import MagicMock

        detector = MagicMock()
        detector.model = mock_yolo_model
        detector.class_names = {0: "person", 1: "car"}
        detector.person_class_id = 0
        detector.device = "cpu"
        detector.confidence_threshold = 0.5

        def mock_detect(image_path, confidence=None, return_image=False):
            result = {
                "detections": [
                    {"bbox": [10, 20, 100, 200], "confidence": 0.8, "class_name": "person"},
                    {"bbox": [150, 160, 250, 300], "confidence": 0.9, "class_name": "person"},
                ],
                "total_detections": 2,
            }
            if return_image:
                result["image_with_boxes"] = np.random.randint(
                    0, 255, (416, 416, 3), dtype=np.uint8
                )
            return result

        detector.detect = mock_detect
        return detector

    def mock_init(self, model_path=None, confidence_threshold=None, device=None):
        self.confidence_threshold = confidence_threshold or 0.5
        self.device = device or "cpu"
        self.model = mock_yolo_model
        self.class_names = {0: "person", 1: "car"}
        self.person_class_id = 0

    monkeypatch.setattr(ThermalDetector, "__init__", mock_init)

    detector = ThermalDetector()
    detector.model = mock_yolo_model
    detector.class_names = {0: "person", 1: "car"}
    detector.person_class_id = 0
    detector.device = "cpu"
    detector.confidence_threshold = 0.5

    return detector


@pytest.fixture
def mock_processor(mock_detector):
    """Создает мок процессора."""
    return ImageProcessor(detector=mock_detector)


@pytest.fixture
def sample_detections():
    """Возвращает пример детекций."""
    return [
        {
            "filename": "test1.jpg",
            "detections": [
                {"bbox": [10, 20, 100, 200], "confidence": 0.8, "class": "person"},
                {"bbox": [150, 160, 250, 300], "confidence": 0.9, "class": "person"},
            ],
            "success": True,
            "error": None,
        },
        {"filename": "test2.jpg", "detections": [], "success": True, "error": None},
    ]


@pytest.fixture
def test_zip_file(temp_dir: Path) -> Path:
    """Создает тестовый ZIP файл с изображениями."""
    import zipfile

    zip_path = temp_dir / "test_images.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(3):
            img_path = temp_dir / f"img_{i}.jpg"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            zipf.write(img_path, f"img_{i}.jpg")
            img_path.unlink()

    return zip_path
