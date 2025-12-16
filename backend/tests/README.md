# Тесты

## Структура

```
tests/
├── conftest.py          # Общие фикстуры и моки
├── unit/                # Unit тесты
│   ├── test_detector.py
│   ├── test_processor.py
│   └── test_storage.py
└── integration/         # Integration тесты
    └── test_api.py
```

## Запуск тестов

### Все тесты
```bash
cd backend
pytest tests/ -v
```

### Только unit тесты
```bash
pytest tests/unit/ -v
```

### Только integration тесты
```bash
pytest tests/integration/ -v
```

### С покрытием кода
```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Конкретный тест
```bash
pytest tests/unit/test_storage.py::test_create_job_with_unique_name -v
```

## Фикстуры

### `temp_dir`
Создает временную директорию для тестов, автоматически удаляется после теста.

### `temp_storage`
Создает временное хранилище задач (`JobStorage`) в изолированной директории.

### `test_image_path`
Создает тестовое изображение (416x416, случайные пиксели).

### `mock_yolo_model`
Мок YOLO модели для unit тестов.

### `mock_detector`
Мок детектора (`ThermalDetector`) с предустановленными значениями.

### `mock_processor`
Мок процессора (`ImageProcessor`) с моком детектора.

### `test_zip_file`
Создает тестовый ZIP файл с изображениями.

## Моки

Тесты используют моки для:
- YOLO модели (чтобы не требовать реальную обученную модель)
- Детектора и процессора (для изоляции unit тестов)
- Временного хранилища (для изоляции данных)

## Покрытие кода

Минимальное покрытие: 70%

Для просмотра отчета:
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

