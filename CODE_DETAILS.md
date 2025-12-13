# Детали работы кода

## Backend API

### Структура

#### `backend/app/main.py`
FastAPI приложение с эндпоинтами для обработки изображений.

**Инициализация:**
- При старте загружает модель YOLOv8 из `training/models/best.pt`
- Автоматически определяет устройство (MPS/CUDA/CPU)
- Создает экземпляры `ThermalDetector` и `ImageProcessor`

**Эндпоинты:**
- `GET /` - информация об API
- `GET /health` - проверка состояния (healthy/degraded)
- `POST /api/upload` - загрузка изображений, создание задачи обработки
- `GET /api/jobs/{job_id}` - статус задачи
- `GET /api/jobs/{job_id}/results` - результаты обработки
- `GET /api/jobs/{job_id}/output/{filename}` - получение обработанного изображения

**Обработка:**
- Загрузка изображений → создание задачи → фоновая обработка
- Статусы: `queued` → `processing` → `completed`/`failed`
- Сохранение только изображений с детекциями в `output/`

#### `backend/app/detector.py`
Класс `ThermalDetector` - обертка над YOLOv8 для детекции людей.

**Методы:**
- `__init__()` - загрузка модели, определение устройства, поиск класса 'person'
- `detect()` - детекция на одном изображении, опционально возвращает изображение с рамками
- `detect_batch()` - пакетная обработка нескольких изображений

**Детекции:**
- Фильтрует только класс 'person'
- Возвращает bbox [x1, y1, x2, y2], confidence, class_name
- При `return_image=True` рисует рамки оранжевым цветом (255, 140, 0)

#### `backend/app/processor.py`
Класс `ImageProcessor` - обработка изображений и координация с детектором.

**Методы:**
- `process_image()` - обработка одного изображения, сохранение результата
- `process_batch()` - пакетная обработка
- `validate_image_format()` - проверка формата (TIFF, PNG, JPEG, WEBP)
- `load_image()` - сохранение изображения из байтов на диск

**Логика:**
- Изображения без детекций не сохраняются в `output/`
- Обработанные изображения содержат рамки вокруг найденных людей

#### `backend/app/storage.py`
Класс `JobStorage` - файловое хранилище задач.

**Структура:**
```
jobs/{job_id}/
├── manifest.json      # Метаданные задачи
├── detections.json    # Результаты детекций
├── input/            # Входные изображения
└── output/           # Обработанные изображения (только с детекциями)
```

**Методы:**
- `create_job()` - создание задачи, генерация UUID
- `update_status()` - обновление статуса и счетчиков
- `save_detections()` - сохранение результатов в JSON
- `get_status()` - получение статуса задачи
- `get_detections()` - получение результатов детекций
- `get_output_image_path()` - путь к обработанному изображению

#### `backend/app/models.py`
Pydantic модели для валидации данных API.

**Модели:**
- `Detection` - bbox, confidence, class_name
- `ImageResult` - filename, detections[], success, error
- `JobStatus` - статус задачи, счетчики, временные метки
- `JobResults` - результаты обработки с метаданными
- `UploadResponse` - ответ на загрузку (job_id, message)

## Training

### `training/train.py`
Обучение YOLOv8 модели на термальных данных FLIR ADAS.

**Параметры:**
- `model_size` - размер модели (n/s/m/l/x)
- `epochs` - количество эпох
- `imgsz` - размер изображений (416 оптимален для M4)
- `batch` - размер батча
- `device` - устройство (auto/MPS/CUDA/CPU)

**Конфигурация:**
- `workers=0` для MPS (multiprocessing несовместим)
- `amp=True` - mixed precision для ускорения
- Аугментация: fliplr, hsv_v, mosaic, mixup
- Early stopping с patience=10

**Сохранение:**
- Модель сохраняется в `training/runs/{name}/weights/best.pt`
- Автоматически копируется в `training/models/best.pt`
- Если стандартный путь не найден, ищет последний запуск

### `training/validate.py`
Валидация обученной модели на тестовом наборе.

**Функции:**
- Вычисление метрик (mAP@50, mAP@50-95, Precision, Recall)
- Визуализация результатов
- Сохранение метрик в JSON

### `training/check_model.py`
Проверка готовности модели для использования в backend.

**Проверки:**
1. Существование файла модели
2. Корректность загрузки
3. Тестовая инференция
4. Наличие классов (person, car)
5. Совместимость с backend

## Хранение данных

### Модели
- `training/models/best.pt` - лучшая модель для production
- `training/training/runs/*/weights/best.pt` - модели из обучения

### Задачи обработки
- `backend/jobs/{job_id}/` - временное хранилище задач
- Автоматическая очистка не требуется (можно добавить cron)

### Датасет
- `training/datasets/yolo/` - структура YOLO формата
- `train/images/`, `train/labels/` - обучающий набор
- `val/images/`, `val/labels/` - валидационный набор

## Оптимизации для M4

### Обучение
- `imgsz=416` вместо 640 для экономии памяти
- `batch=16` оптимален для 16GB RAM
- `workers=0` обязательно для MPS
- `amp=True` экономит ~40% памяти

### Инференция
- Автоматическое определение MPS
- Batch processing для нескольких изображений
- Сохранение только изображений с детекциями

## Поток данных

1. **Загрузка изображений** → `POST /api/upload`
2. **Создание задачи** → `JobStorage.create_job()`
3. **Фоновая обработка** → `process_images_task()`
   - Сохранение в `input/`
   - Детекция через `ThermalDetector`
   - Обработка через `ImageProcessor`
   - Сохранение результатов в `output/` (только с детекциями)
4. **Получение результатов** → `GET /api/jobs/{job_id}/results`
5. **Получение изображений** → `GET /api/jobs/{job_id}/output/{filename}`

