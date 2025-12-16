import pytest
from pathlib import Path
from app.storage import JobStorage
from app.models import ImageResult, Detection


def test_create_job_with_unique_name(temp_storage: JobStorage):
    """Тест создания задачи с уникальным именем."""
    job_id = temp_storage.create_job(name="TestTask", total_images=5)
    assert job_id is not None
    
    status = temp_storage.get_status(job_id)
    assert status is not None
    assert status['name'] == "TestTask"
    assert status['status'] == "queued"
    assert status['total_images'] == 5


def test_create_job_with_duplicate_name(temp_storage: JobStorage):
    """Тест создания задачи с дублирующимся именем (автоиндексация)."""
    job_id1 = temp_storage.create_job(name="Task", total_images=1)
    job_id2 = temp_storage.create_job(name="Task", total_images=2)
    
    status1 = temp_storage.get_status(job_id1)
    status2 = temp_storage.get_status(job_id2)
    
    assert status1['name'] == "Task"
    assert status2['name'] == "Task_1"


def test_update_job_name(temp_storage: JobStorage):
    """Тест переименования задачи."""
    job_id = temp_storage.create_job(name="OldName", total_images=1)
    
    success = temp_storage.update_job_name(job_id, "NewName")
    assert success is True
    
    status = temp_storage.get_status(job_id)
    assert status['name'] == "NewName"


def test_update_job_name_with_duplicate(temp_storage: JobStorage):
    """Тест переименования с проверкой уникальности."""
    job_id1 = temp_storage.create_job(name="Task1", total_images=1)
    job_id2 = temp_storage.create_job(name="Task2", total_images=1)
    
    temp_storage.update_job_name(job_id2, "Task1")
    
    status1 = temp_storage.get_status(job_id1)
    status2 = temp_storage.get_status(job_id2)
    
    assert status1['name'] == "Task1"
    assert status2['name'] == "Task1_1"


def test_list_jobs_sorted_by_date(temp_storage: JobStorage):
    """Тест списка задач с сортировкой по дате."""
    import time
    
    job_id1 = temp_storage.create_job(name="Task1", total_images=1)
    time.sleep(0.1)
    job_id2 = temp_storage.create_job(name="Task2", total_images=1)
    
    jobs = temp_storage.list_jobs()
    assert len(jobs) == 2
    assert jobs[0]['name'] == "Task2"
    assert jobs[1]['name'] == "Task1"


def test_save_and_get_detections(temp_storage: JobStorage):
    """Тест сохранения и получения детекций."""
    job_id = temp_storage.create_job(total_images=2)
    
    detections = [
        ImageResult(
            filename="test1.jpg",
            detections=[
                Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")
            ],
            success=True
        ),
        ImageResult(
            filename="test2.jpg",
            detections=[],
            success=True
        )
    ]
    
    temp_storage.save_detections(job_id, detections)
    
    loaded = temp_storage.get_detections(job_id)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]['filename'] == "test1.jpg"
    assert len(loaded[0]['detections']) == 1


def test_create_output_zip_with_filtering(temp_storage: JobStorage, temp_dir: Path):
    """Тест создания ZIP архива с фильтрацией."""
    job_id = temp_storage.create_job(total_images=2)
    
    job_dir = Path(temp_storage.base_dir) / job_id
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import cv2
    import numpy as np
    
    img1_path = output_dir / "img1.jpg"
    img2_path = output_dir / "img2.jpg"
    cv2.imwrite(str(img1_path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(img2_path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    detections = [
        {
            "filename": "img1.jpg",
            "detections": [{"bbox": [10, 20, 100, 200], "confidence": 0.9, "class": "person"}],
            "success": True
        },
        {
            "filename": "img2.jpg",
            "detections": [{"bbox": [10, 20, 100, 200], "confidence": 0.3, "class": "person"}],
            "success": True
        }
    ]
    
    temp_storage.save_detections(job_id, detections)
    temp_storage.update_status(job_id, "completed")
    
    zip_path = temp_storage.create_output_zip(job_id, min_confidence=0.5, only_with_detections=True)
    assert zip_path is not None
    assert zip_path.exists()


def test_get_input_and_output_image_paths(temp_storage: JobStorage, temp_dir: Path):
    """Тест получения путей к входным и выходным изображениям."""
    job_id = temp_storage.create_job(total_images=1)
    
    job_dir = Path(temp_storage.base_dir) / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import cv2
    import numpy as np
    
    input_path = input_dir / "test.jpg"
    output_path = output_dir / "test.jpg"
    cv2.imwrite(str(input_path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(output_path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    assert temp_storage.get_input_image_path(job_id, "test.jpg") == input_path
    assert temp_storage.get_output_image_path(job_id, "test.jpg") == output_path
    assert temp_storage.get_input_image_path(job_id, "nonexistent.jpg") is None

