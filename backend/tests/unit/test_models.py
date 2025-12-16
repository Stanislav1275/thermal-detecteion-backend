from app.models import Detection, ImageResult


def test_imageresult_total_detections_property():
    """Тест computed property total_detections в ImageResult."""
    result_with_detections = ImageResult(
        filename="test1.jpg",
        detections=[
            Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person"),
            Detection(bbox=[150, 160, 250, 300], confidence=0.9, class_name="person"),
        ],
        success=True,
    )

    assert result_with_detections.total_detections == 2
    assert len(result_with_detections.detections) == 2

    result_without_detections = ImageResult(filename="test2.jpg", detections=[], success=True)

    assert result_without_detections.total_detections == 0
    assert len(result_without_detections.detections) == 0


def test_imageresult_total_detections_with_error():
    """Тест total_detections для изображения с ошибкой."""
    result_with_error = ImageResult(
        filename="error.jpg", detections=[], success=False, error="Ошибка обработки"
    )

    assert result_with_error.total_detections == 0
    assert result_with_error.success is False
    assert result_with_error.error is not None


def test_imageresult_serialization():
    """Тест сериализации ImageResult с total_detections."""
    result = ImageResult(
        filename="test.jpg",
        detections=[Detection(bbox=[10, 20, 100, 200], confidence=0.8, class_name="person")],
        success=True,
    )

    json_data = result.model_dump()
    assert "detections" in json_data
    assert len(json_data["detections"]) == 1

    assert result.total_detections == 1
