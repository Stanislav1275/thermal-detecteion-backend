import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.processor import ImageProcessor


def test_sanitize_filename():
    """Тест санитизации имени файла (защита от path traversal)."""
    assert ImageProcessor.sanitize_filename("../../../etc/passwd") == "passwd"
    assert ImageProcessor.sanitize_filename("normal_file.jpg") == "normal_file.jpg"
    assert ImageProcessor.sanitize_filename("file<>name|.jpg") == "file__name_.jpg"
    assert ImageProcessor.sanitize_filename("") == "unnamed_file"
    assert ImageProcessor.sanitize_filename("..") == "unnamed_file"
    assert ImageProcessor.sanitize_filename(".") == "unnamed_file"


def test_validate_image_format():
    """Тест валидации формата изображения."""
    assert ImageProcessor.validate_image_format("test.jpg") is True
    assert ImageProcessor.validate_image_format("test.jpeg") is True
    assert ImageProcessor.validate_image_format("test.png") is True
    assert ImageProcessor.validate_image_format("test.tiff") is True
    assert ImageProcessor.validate_image_format("test.tif") is True
    assert ImageProcessor.validate_image_format("test.webp") is True
    assert ImageProcessor.validate_image_format("test.txt") is False
    assert ImageProcessor.validate_image_format("test.pdf") is False


def test_get_mime_type():
    """Тест определения MIME-типа."""
    assert ImageProcessor.get_mime_type("test.jpg") == "image/jpeg"
    assert ImageProcessor.get_mime_type("test.jpeg") == "image/jpeg"
    assert ImageProcessor.get_mime_type("test.png") == "image/png"
    assert ImageProcessor.get_mime_type("test.tiff") == "image/tiff"
    assert ImageProcessor.get_mime_type("test.tif") == "image/tiff"
    assert ImageProcessor.get_mime_type("test.webp") == "image/webp"
    assert ImageProcessor.get_mime_type("test.unknown") == "image/jpeg"


def test_extract_zip_archive_valid(temp_dir: Path):
    """Тест извлечения валидного ZIP архива."""
    zip_path = temp_dir / "test.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(2):
            img_path = temp_dir / f"img_{i}.jpg"
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            zipf.write(img_path, f"img_{i}.jpg")
            img_path.unlink()

    with open(zip_path, "rb") as f:
        zip_data = f.read()

    extract_to = temp_dir / "extracted"
    extracted = ImageProcessor.extract_zip_archive(zip_data, str(extract_to))

    assert len(extracted) == 2
    assert all(Path(path).exists() for _, path in extracted)


def test_extract_zip_archive_empty(temp_dir: Path):
    """Тест извлечения пустого ZIP архива."""
    zip_path = temp_dir / "empty.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr("readme.txt", "No images here")

    with open(zip_path, "rb") as f:
        zip_data = f.read()

    extract_to = temp_dir / "extracted"

    with pytest.raises(ValueError, match="не содержит валидных изображений"):
        ImageProcessor.extract_zip_archive(zip_data, str(extract_to))


def test_extract_zip_archive_invalid():
    """Тест извлечения некорректного ZIP архива."""
    invalid_data = b"This is not a zip file"

    with pytest.raises(ValueError, match="Некорректный ZIP-архив"):
        ImageProcessor.extract_zip_archive(invalid_data, "/tmp/extract")


def test_is_zip_file():
    """Тест проверки ZIP файла."""
    assert ImageProcessor.is_zip_file("test.zip") is True
    assert ImageProcessor.is_zip_file("test.ZIP") is True
    assert ImageProcessor.is_zip_file("test.jpg") is False
    assert ImageProcessor.is_zip_file("test.tar.gz") is False


def test_load_image(temp_dir: Path):
    """Тест загрузки изображения."""
    img_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_bytes = cv2.imencode(".jpg", img_data)[1].tobytes()

    save_path = temp_dir / "loaded.jpg"
    ImageProcessor.load_image(img_bytes, "test.jpg", str(save_path))

    assert save_path.exists()
    loaded = cv2.imread(str(save_path))
    assert loaded is not None


def test_process_image_with_detections(mock_processor, test_image_path: Path, temp_dir: Path):
    """Тест обработки изображения с детекциями."""
    from unittest.mock import Mock

    mock_processor.detector.detect = Mock(
        return_value={
            "detections": [{"bbox": [10, 20, 100, 200], "confidence": 0.8, "class_name": "person"}],
            "total_detections": 1,
            "image_with_boxes": np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8),
        }
    )

    output_path = temp_dir / "output.jpg"

    result = mock_processor.process_image(str(test_image_path), str(output_path), confidence=0.5)

    assert result.success is True
    assert len(result.detections) > 0
    assert output_path.exists()


def test_process_image_without_detections(mock_processor, test_image_path: Path, temp_dir: Path):
    """Тест обработки изображения без детекций."""
    mock_processor.detector.detect = lambda *args, **kwargs: {
        "detections": [],
        "total_detections": 0,
    }

    output_path = temp_dir / "output.jpg"
    result = mock_processor.process_image(str(test_image_path), str(output_path), confidence=0.5)

    assert result.success is True
    assert len(result.detections) == 0
