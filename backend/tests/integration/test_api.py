import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import numpy as np
import cv2
import zipfile
import tempfile
import shutil
from unittest.mock import Mock, patch
from app.main import app
from app.detector import ThermalDetector
from app.processor import ImageProcessor


@pytest.fixture
def client():
    """Создает тестовый клиент FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_detector_and_processor(monkeypatch):
    """Мокирует детектор и процессор для тестов."""
    mock_detector = Mock(spec=ThermalDetector)
    mock_processor = Mock(spec=ImageProcessor)
    
    monkeypatch.setattr("app.main.detector", mock_detector)
    monkeypatch.setattr("app.main.processor", mock_processor)
    
    return mock_detector, mock_processor


@pytest.fixture
def test_image_file(temp_dir: Path) -> Path:
    """Создает тестовый файл изображения."""
    img_path = temp_dir / "test.jpg"
    img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def test_zip_file(temp_dir: Path) -> Path:
    """Создает тестовый ZIP файл."""
    zip_path = temp_dir / "test.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for i in range(2):
            img_path = temp_dir / f"img_{i}.jpg"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            zipf.write(img_path, f"img_{i}.jpg")
            img_path.unlink()
    
    return zip_path


def test_root_endpoint(client):
    """Тест корневого endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "status" in data


def test_health_check_healthy(client, mock_detector_and_processor):
    """Тест health check с загруженной моделью."""
    mock_detector, mock_processor = mock_detector_and_processor
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_check_degraded(client):
    """Тест health check без загруженной модели."""
    from app.main import detector, processor
    original_detector = detector
    original_processor = processor
    
    try:
        import app.main
        app.main.detector = None
        app.main.processor = None
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
    finally:
        app.main.detector = original_detector
        app.main.processor = original_processor


def test_upload_single_image(client, mock_detector_and_processor, test_image_file: Path):
    """Тест загрузки одного изображения."""
    mock_detector, mock_processor = mock_detector_and_processor
    
    mock_processor.sanitize_filename = lambda x: x
    mock_processor.validate_image_format = lambda x: True
    mock_processor.is_zip_file = lambda x: False
    mock_processor.load_image = Mock()
    
    with open(test_image_file, 'rb') as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.5"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "message" in data


def test_upload_zip_archive(client, mock_detector_and_processor, test_zip_file: Path):
    """Тест загрузки ZIP архива."""
    mock_detector, mock_processor = mock_detector_and_processor
    
    mock_processor.sanitize_filename = lambda x: x
    mock_processor.is_zip_file = lambda x: x.endswith('.zip')
    mock_processor.extract_zip_archive = Mock(return_value=[
        ("img_0.jpg", "/tmp/img_0.jpg"),
        ("img_1.jpg", "/tmp/img_1.jpg")
    ])
    
    with open(test_zip_file, 'rb') as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.zip", f, "application/zip")},
            data={"confidence_threshold": "0.5"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


def test_upload_invalid_file(client, mock_detector_and_processor):
    """Тест загрузки некорректного файла."""
    mock_detector, mock_processor = mock_detector_and_processor
    
    mock_processor.sanitize_filename = lambda x: x
    mock_processor.validate_image_format = lambda x: False
    mock_processor.is_zip_file = lambda x: False
    
    response = client.post(
        "/api/upload",
        files={"files": ("test.txt", b"not an image", "text/plain")},
        data={"confidence_threshold": "0.5"}
    )
    
    assert response.status_code == 400


def test_upload_invalid_confidence(client, mock_detector_and_processor, test_image_file: Path):
    """Тест загрузки с некорректным порогом уверенности."""
    mock_detector, mock_processor = mock_detector_and_processor
    
    with open(test_image_file, 'rb') as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "1.5"}
        )
    
    assert response.status_code == 400


def test_list_jobs(client, temp_storage):
    """Тест получения списка задач."""
    import app.main
    original_storage = app.main.storage
    app.main.storage = temp_storage
    
    try:
        job_id = temp_storage.create_job(name="TestTask", total_images=1)
        
        response = client.get("/api/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) >= 1
        assert any(j['job_id'] == job_id for j in jobs)
    finally:
        app.main.storage = original_storage


def test_get_job_status(client, temp_storage):
    """Тест получения статуса задачи."""
    import app.main
    original_storage = app.main.storage
    app.main.storage = temp_storage
    
    try:
        job_id = temp_storage.create_job(name="TestTask", total_images=1)
        
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data['job_id'] == job_id
        assert data['status'] == "queued"
    finally:
        app.main.storage = original_storage


def test_get_job_status_not_found(client):
    """Тест получения статуса несуществующей задачи."""
    response = client.get("/api/jobs/nonexistent-id")
    assert response.status_code == 404


def test_update_job_name(client, temp_storage):
    """Тест переименования задачи."""
    import app.main
    original_storage = app.main.storage
    app.main.storage = temp_storage
    
    try:
        job_id = temp_storage.create_job(name="OldName", total_images=1)
        
        response = client.patch(
            f"/api/jobs/{job_id}/name",
            json={"name": "NewName"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == "NewName"
    finally:
        app.main.storage = original_storage


def test_get_job_results(client, temp_storage):
    """Тест получения результатов задачи."""
    import app.main
    original_storage = app.main.storage
    app.main.storage = temp_storage
    
    try:
        job_id = temp_storage.create_job(total_images=2)
        
        from app.models import ImageResult, Detection
        detections = [
            ImageResult(
                filename="test1.jpg",
                detections=[Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")],
                success=True
            ),
            ImageResult(
                filename="test2.jpg",
                detections=[],
                success=True
            )
        ]
        
        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(job_id, "completed")
        
        response = client.get(f"/api/jobs/{job_id}/results")
        assert response.status_code == 200
        data = response.json()
        assert data['job_id'] == job_id
        assert len(data['images']) == 2
    finally:
        app.main.storage = original_storage


def test_get_job_results_only_with_detections(client, temp_storage):
    """Тест получения результатов с фильтром only_with_detections."""
    import app.main
    original_storage = app.main.storage
    app.main.storage = temp_storage
    
    try:
        job_id = temp_storage.create_job(total_images=2)
        
        from app.models import ImageResult, Detection
        detections = [
            ImageResult(
                filename="test1.jpg",
                detections=[Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")],
                success=True
            ),
            ImageResult(
                filename="test2.jpg",
                detections=[],
                success=True
            )
        ]
        
        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(job_id, "completed")
        
        response = client.get(f"/api/jobs/{job_id}/results?only_with_detections=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data['images']) == 1
        assert data['images'][0]['filename'] == "test1.jpg"
    finally:
        app.main.storage = original_storage

