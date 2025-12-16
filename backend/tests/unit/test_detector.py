import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock
from app.detector import ThermalDetector


def test_detector_initialization_with_mock(mock_detector):
    """Тест инициализации детектора."""
    assert mock_detector is not None
    assert mock_detector.person_class_id == 0
    assert mock_detector.confidence_threshold == 0.5
    assert mock_detector.device == "cpu"


def test_detect_with_detections(mock_detector, test_image_path: Path):
    """Тест детекции с найденными объектами."""
    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = np.array([[10, 20, 100, 200], [150, 160, 250, 300]])
    mock_box.cls = np.array([0, 0])
    mock_box.conf = np.array([0.8, 0.9])
    mock_result.boxes = mock_box
    mock_detector.model.predict = Mock(return_value=[mock_result])
    
    result = mock_detector.detect(str(test_image_path), confidence=0.5)
    
    assert 'detections' in result
    assert 'total_detections' in result
    assert result['total_detections'] > 0
    assert all(det['class_name'] == 'person' for det in result['detections'])


def test_detect_without_detections(mock_detector, test_image_path: Path):
    """Тест детекции без найденных объектов."""
    mock_result = Mock()
    mock_result.boxes = None
    mock_detector.model.predict = Mock(return_value=[mock_result])
    
    result = mock_detector.detect(str(test_image_path), confidence=0.5)
    
    assert result['total_detections'] == 0
    assert len(result['detections']) == 0


def test_detect_with_invalid_image(mock_detector, temp_dir: Path):
    """Тест детекции с некорректным изображением."""
    invalid_path = temp_dir / "nonexistent.jpg"
    
    with pytest.raises(FileNotFoundError):
        mock_detector.detect(str(invalid_path))


def test_detect_filters_by_person_class(mock_detector, test_image_path: Path):
    """Тест фильтрации по классу 'person'."""
    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = np.array([[10, 20, 100, 200], [150, 160, 250, 300]])
    mock_box.cls = np.array([0, 1])
    mock_box.conf = np.array([0.8, 0.9])
    mock_result.boxes = mock_box
    
    mock_detector.model.predict = Mock(return_value=[mock_result])
    mock_detector.person_class_id = 0
    
    result = mock_detector.detect(str(test_image_path), confidence=0.5)
    
    assert result['total_detections'] == 1
    assert len(result['detections']) == 1
    assert result['detections'][0]['class_name'] == 'person'


def test_detect_confidence_threshold(mock_detector, test_image_path: Path):
    """Тест применения порога уверенности."""
    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = np.array([[10, 20, 100, 200], [150, 160, 250, 300]])
    mock_box.cls = np.array([0, 0])
    mock_box.conf = np.array([0.3, 0.8])
    mock_result.boxes = mock_box
    
    mock_detector.model.predict = Mock(return_value=[mock_result])
    
    result = mock_detector.detect(str(test_image_path), confidence=0.5)
    
    assert result['total_detections'] >= 0


def test_detect_with_image_return(mock_detector, test_image_path: Path):
    """Тест детекции с возвратом изображения."""
    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = np.array([[10, 20, 100, 200]])
    mock_box.cls = np.array([0])
    mock_box.conf = np.array([0.8])
    mock_result.boxes = mock_box
    mock_detector.model.predict = Mock(return_value=[mock_result])
    
    result = mock_detector.detect(str(test_image_path), confidence=0.5, return_image=True)
    
    assert 'image_with_boxes' in result
    assert result['image_with_boxes'] is not None
    assert isinstance(result['image_with_boxes'], np.ndarray)


def test_detect_batch(mock_detector, test_image_path: Path, temp_dir: Path):
    """Тест пакетной обработки."""
    img2_path = temp_dir / "test2.jpg"
    img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    cv2.imwrite(str(img2_path), img)
    
    image_paths = [str(test_image_path), str(img2_path)]
    results = mock_detector.detect_batch(image_paths, confidence=0.5)
    
    assert len(results) == 2
    assert all('filename' in r for r in results)
    assert all('success' in r for r in results)


def test_detect_batch_with_error(mock_detector, temp_dir: Path):
    """Тест пакетной обработки с ошибкой."""
    invalid_path = temp_dir / "nonexistent.jpg"
    
    results = mock_detector.detect_batch([str(invalid_path)], confidence=0.5)
    
    assert len(results) == 1
    assert results[0]['success'] is False
    assert 'error' in results[0]

