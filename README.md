# Система детекции людей на тепловизионных изображениях

Система для автоматизированной детекции людей на тепловизионных снимках с использованием YOLOv8 и веб-интерфейса.

## Структура проекта

```
PythonProject/
├── training/              # Обучение модели
│   ├── prepare_dataset.py # Подготовка FLIR ADAS датасета
│   ├── train.py          # Скрипт обучения
│   ├── validate.py       # Валидация модели
│   ├── thermal.yaml      # Конфигурация YOLO
│   └── models/           # Обученные модели
├── backend/              # Backend API
│   ├── app/
│   │   ├── main.py       # FastAPI приложение
│   │   ├── detector.py   # YOLO детектор
│   │   ├── processor.py  # Обработчик изображений
│   │   ├── storage.py    # Управление хранилищем
│   │   └── models.py    # Pydantic модели
│   └── environment.yml   # Conda окружение

└── jobs/                 # Временное хранилище задач
```

## Последовательность шагов перед обучением

1. **Установка зависимостей** (conda или venv)
2. **Скачивание датасета** FLIR ADAS
3. **Подготовка датасета** (конвертация в YOLO формат)
4. **Обучение модели** YOLOv8n
5. **Валидация модели** на тестовом наборе
6. **Запуск API** (после обучения)

Подробные инструкции: [QUICKSTART.md](QUICKSTART.md)

## Быстрый старт

1. **Установка зависимостей:**
   ```bash
   cd backend
   ./setup_venv.sh    # или ./setup.sh для conda
   ```

2. **Скачать датасет и обучить модель:**
   ```bash
   cd training
   ./download_dataset.sh    # Скачать датасет
   ./all_in_one.sh          # Подготовить и обучить (или по шагам: ./prepare.sh && ./train_simple.sh)
   ```

3. **Запустить API:**
   ```bash
   cd ../backend
   ./run.sh
   ```

## Установка и настройка

### 1. Создание окружения

#### Вариант A: Conda

```bash
# Создание нового окружения
conda create -n thermal-detection python=3.11 -y

# Активация окружения
conda activate thermal-detection

# Установка зависимостей
conda env update -f backend/environment.yml --prune
```

#### Вариант B: venv

```bash
cd backend
./setup_venv.sh
```

### 2. Обучение модели

#### Подготовка датасета

```bash
cd training

# Загрузка FLIR ADAS датасета через kagglehub
python -c "import kagglehub; kagglehub.dataset_download('deepnewbie/flir-thermal-images-dataset')"

# Подготовка датасета в YOLO формат
python prepare_dataset.py \
    --dataset-root ~/.cache/kagglehub/datasets/deepnewbie/flir-thermal-images-dataset/versions/1 \
    --output-root ./datasets/yolo \
    --splits train val
```

#### Обучение модели

```bash
# Обучение YOLOv8n на термальных данных
python train.py \
    --data thermal.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

#### Валидация модели

```bash
# Валидация обученной модели
python validate.py \
    --model training/models/best.pt \
    --data thermal.yaml \
    --output training/results \
    --visualize
```

### 3. Запуск Backend API

```bash
cd backend

# Запуск FastAPI сервера
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API будет доступен по адресу: http://localhost:8000

Документация API: http://localhost:8000/docs

## Использование API

### Загрузка изображений

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "confidence_threshold=0.5"
```

### Проверка статуса задачи

```bash
curl "http://localhost:8000/api/jobs/{job_id}"
```

### Получение результатов

```bash
curl "http://localhost:8000/api/jobs/{job_id}/results"
```

### Получение обработанного изображения

```bash
curl "http://localhost:8000/api/jobs/{job_id}/output/{filename}" --output result.jpg
```

## Технологии

- **Backend**: Python 3.11, FastAPI, PyTorch, Ultralytics YOLOv8
- **ML**: YOLOv8n с обучением на FLIR ADAS датасете
- **Frontend**: React + Vite + React Query (Этап 2)

## Особенности

- Локальная работа без БД
- Поддержка MPS (Apple Silicon) для ускорения
- Пакетная обработка изображений
- Временное хранилище результатов в `/jobs/`

## Дополнительная документация

- [QUICKSTART.md](QUICKSTART.md) - Подробная инструкция по быстрому старту
- [training/README.md](training/README.md) - Документация по обучению модели
- [DOCKER.md](DOCKER.md) - Инструкции по использованию Docker
- [USE_CASES.md](USE_CASES.md) - Use-case-ы проекта с точки зрения пользователя

## Docker

Для запуска через Docker см. [DOCKER.md](DOCKER.md)

Быстрый запуск:
```bash
docker-compose up -d
```

## Лицензия

Учебный проект для лабораторной работы.
