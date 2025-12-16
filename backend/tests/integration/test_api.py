import zipfile
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from app.detector import ThermalDetector
from app.main import app
from app.processor import ImageProcessor
from fastapi.testclient import TestClient


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

    with zipfile.ZipFile(zip_path, "w") as zipf:
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

    with open(test_image_file, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "0.5"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "message" in data


def test_upload_zip_archive(client, mock_detector_and_processor, test_zip_file: Path):
    """Тест загрузки ZIP архива."""
    mock_detector, mock_processor = mock_detector_and_processor

    mock_processor.sanitize_filename = lambda x: x
    mock_processor.is_zip_file = lambda x: x.endswith(".zip")
    mock_processor.extract_zip_archive = Mock(
        return_value=[("img_0.jpg", "/tmp/img_0.jpg"), ("img_1.jpg", "/tmp/img_1.jpg")]
    )

    with open(test_zip_file, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.zip", f, "application/zip")},
            data={"confidence_threshold": "0.5"},
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
        data={"confidence_threshold": "0.5"},
    )

    assert response.status_code == 400


def test_upload_invalid_confidence(client, mock_detector_and_processor, test_image_file: Path):
    """Тест загрузки с некорректным порогом уверенности."""
    mock_detector, mock_processor = mock_detector_and_processor

    with open(test_image_file, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"files": ("test.jpg", f, "image/jpeg")},
            data={"confidence_threshold": "1.5"},
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
        assert any(j["job_id"] == job_id for j in jobs)
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
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
    finally:
        app.main.storage = original_storage


def test_get_job_status_not_found(client):
    """Тест получения статуса несуществующей задачи."""
    response = client.get("/api/jobs/nonexistent-id")
    assert response.status_code == 404


def test_get_job_status_returns_confidence_threshold(client, temp_storage):
    """Тест, что GET /api/jobs/{job_id} возвращает confidence_threshold в parameters."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(confidence_threshold=0.85, name="TestTask", total_images=3)

        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert "parameters" in data
        assert "confidence_threshold" in data["parameters"]
        assert data["parameters"]["confidence_threshold"] == 0.85
    finally:
        app.main.storage = original_storage


def test_list_jobs_returns_confidence_threshold(client, temp_storage):
    """Тест, что GET /api/jobs возвращает confidence_threshold в parameters для всех задач."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id1 = temp_storage.create_job(confidence_threshold=0.7, name="Task1", total_images=1)
        job_id2 = temp_storage.create_job(confidence_threshold=0.9, name="Task2", total_images=2)

        response = client.get("/api/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) >= 2

        job1 = next((j for j in jobs if j["job_id"] == job_id1), None)
        job2 = next((j for j in jobs if j["job_id"] == job_id2), None)

        assert job1 is not None
        assert job2 is not None
        assert "parameters" in job1
        assert "parameters" in job2
        assert job1["parameters"]["confidence_threshold"] == 0.7
        assert job2["parameters"]["confidence_threshold"] == 0.9
    finally:
        app.main.storage = original_storage


def test_upload_with_confidence_threshold_saves_to_manifest(
    client, temp_storage, test_image_file, mock_detector_and_processor
):
    """Тест, что при загрузке изображения с порогом, порог сохраняется в manifest."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    mock_detector, mock_processor = mock_detector_and_processor

    from app.models import Detection, ImageResult

    mock_result = ImageResult(
        filename=test_image_file.name,
        detections=[Detection(bbox=[10, 20, 100, 200], confidence=0.85, class_name="person")],
        success=True,
    )

    mock_processor.process_image.return_value = mock_result
    mock_processor.sanitize_filename.return_value = test_image_file.name
    mock_processor.validate_image_format.return_value = True
    mock_processor.is_zip_file.return_value = False
    mock_processor.load_image.return_value = None

    try:
        with open(test_image_file, "rb") as f:
            files = {"files": (test_image_file.name, f, "image/jpeg")}
            data = {"confidence_threshold": "0.62", "name": "TestTask"}

            response = client.post("/api/upload", files=files, data=data)
            assert response.status_code == 200
            result = response.json()
            job_id = result["job_id"]

            status = temp_storage.get_status(job_id)
            assert status is not None
            assert "parameters" in status
            assert "confidence_threshold" in status["parameters"]
            assert status["parameters"]["confidence_threshold"] == 0.62
    finally:
        app.main.storage = original_storage


def test_update_job_name(client, temp_storage):
    """Тест переименования задачи."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(name="OldName", total_images=1)

        response = client.patch(f"/api/jobs/{job_id}/name", json={"name": "NewName"})

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "NewName"
    finally:
        app.main.storage = original_storage


def test_get_job_results(client, temp_storage):
    """Тест получения результатов задачи."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(total_images=2)

        from app.models import Detection, ImageResult

        detections = [
            ImageResult(
                filename="test1.jpg",
                detections=[
                    Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")
                ],
                success=True,
            ),
            ImageResult(filename="test2.jpg", detections=[], success=True),
        ]

        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(job_id, "completed")

        response = client.get(f"/api/jobs/{job_id}/results")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert len(data["images"]) == 2
    finally:
        app.main.storage = original_storage


def test_get_job_results_only_with_detections(client, temp_storage):
    """Тест получения результатов с фильтром only_with_detections."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(total_images=2)

        from app.models import Detection, ImageResult

        detections = [
            ImageResult(
                filename="test1.jpg",
                detections=[
                    Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")
                ],
                success=True,
            ),
            ImageResult(filename="test2.jpg", detections=[], success=True),
        ]

        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(job_id, "completed")

        response = client.get(f"/api/jobs/{job_id}/results?only_with_detections=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data["images"]) == 1
        assert data["images"][0]["filename"] == "test1.jpg"
    finally:
        app.main.storage = original_storage


def test_delete_job(client, temp_storage):
    """Тест удаления задачи."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(name="TestTask", total_images=1)

        response = client.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Задача успешно удалена"
        assert data["job_id"] == job_id

        status_response = client.get(f"/api/jobs/{job_id}")
        assert status_response.status_code == 404
    finally:
        app.main.storage = original_storage


def test_delete_job_not_found(client):
    """Тест удаления несуществующей задачи."""
    response = client.delete("/api/jobs/nonexistent-id")
    assert response.status_code == 404
    assert "не найдена" in response.json()["detail"]


def test_job_results_counting_with_errors(client, temp_storage):
    """Тест подсчёта статистики с ошибками обработки."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(total_images=3)

        from app.models import Detection, ImageResult

        detections = [
            ImageResult(
                filename="test1.jpg",
                detections=[
                    Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")
                ],
                success=True,
            ),
            ImageResult(filename="test2.jpg", detections=[], success=True),
            ImageResult(
                filename="test3.jpg", detections=[], success=False, error="Ошибка обработки"
            ),
        ]

        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(
            job_id, "completed", processed_images=3, images_with_detections=1
        )

        response = client.get(f"/api/jobs/{job_id}/results")
        assert response.status_code == 200
        data = response.json()

        metadata = data["metadata"]
        assert metadata["total_images"] == 3
        assert metadata["total_images_with_people"] == 1
        assert metadata["total_errors"] == 1
        assert metadata["total_detections"] == 1

        status_response = client.get(f"/api/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()

        assert status_data["processed_images"] == 3
        assert status_data["images_with_detections"] == 1
        assert status_data["total_images"] == 3

        assert metadata["total_images"] == status_data["processed_images"]
    finally:
        app.main.storage = original_storage


def test_job_results_counting_consistency(client, temp_storage):
    """Тест консистентности подсчёта между JobStatus и JobResults."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(total_images=5)

        from app.models import Detection, ImageResult

        detections = [
            ImageResult(
                filename=f"test{i}.jpg",
                detections=(
                    [Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")]
                    if i < 2
                    else []
                ),
                success=i < 4,
                error="Ошибка" if i >= 4 else None,
            )
            for i in range(5)
        ]

        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(
            job_id, "completed", processed_images=5, images_with_detections=2
        )

        status_response = client.get(f"/api/jobs/{job_id}")
        results_response = client.get(f"/api/jobs/{job_id}/results")

        assert status_response.status_code == 200
        assert results_response.status_code == 200

        status = status_response.json()
        results = results_response.json()

        assert status["processed_images"] == results["metadata"]["total_images"]
        assert status["images_with_detections"] == results["metadata"]["total_images_with_people"]
        assert results["metadata"]["total_errors"] == 1
        assert results["metadata"]["total_images"] == 5
    finally:
        app.main.storage = original_storage


def test_imageresult_total_detections_in_response(client, temp_storage):
    """Тест наличия total_detections в ответе API."""
    import app.main

    original_storage = app.main.storage
    app.main.storage = temp_storage

    try:
        job_id = temp_storage.create_job(total_images=1)

        from app.models import Detection, ImageResult

        detections = [
            ImageResult(
                filename="test.jpg",
                detections=[
                    Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person"),
                    Detection(bbox=[150, 160, 250, 300], confidence=0.9, class_name="person"),
                ],
                success=True,
            )
        ]

        temp_storage.save_detections(job_id, detections)
        temp_storage.update_status(job_id, "completed")

        response = client.get(f"/api/jobs/{job_id}/results")
        assert response.status_code == 200
        data = response.json()

        assert len(data["images"]) == 1
        image_result = data["images"][0]
        assert "total_detections" in image_result or hasattr(image_result, "total_detections")

        from app.models import ImageResult as IR

        parsed = IR(**image_result)
        assert parsed.total_detections == 2
    finally:
        app.main.storage = original_storage
