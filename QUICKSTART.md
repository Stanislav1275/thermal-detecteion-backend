# Быстрый старт

## Установка

### Вариант 1: Через venv

```bash
cd backend
./setup_venv.sh
```

### Вариант 2: Через conda

```bash
cd backend
./setup.sh
```

Если conda не найдена, найдите путь к conda (обычно `~/miniconda3/bin/conda`) и выполните:

```bash
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### Проверка установки

```bash
cd backend
python check_environment.py
```

## Обучение модели

Обучение модели обязательно перед использованием API.

### Автоматический способ

```bash
cd training
./download_dataset.sh    # Скачать датасет
./all_in_one.sh          # Подготовить и обучить автоматически
```

### Пошаговый способ

#### Шаг 1: Скачивание датасета

```bash
cd training
./download_dataset.sh
```

#### Шаг 2: Подготовка датасета

```bash
./prepare.sh
```

Скрипт конвертирует FLIR ADAS датасет из COCO формата в YOLO формат, создавая структуру:

```
datasets/yolo/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

#### Шаг 3: Обучение модели

```bash
./train_simple.sh         # Обучение на 100 эпох
# или
./train_simple.sh 50      # Быстрый тест на 50 эпох
```

Результаты обучения сохраняются в `training/runs/detect/thermal_detection/`

Лучшая модель автоматически копируется в `training/models/best.pt`

#### Шаг 4: Валидация модели

```bash
python validate.py \
    --model training/models/best.pt \
    --data thermal.yaml \
    --output training/results \
    --visualize
```

Подробнее: [training/README.md](training/README.md)

## Запуск API

### С venv

```bash
cd backend
./run_venv.sh
```

### С conda

```bash
conda activate thermal-detection
cd backend
./run.sh
```

API доступен на: http://localhost:8000  
Документация API: http://localhost:8000/docs

## Тестирование

### Использование тестового скрипта

```bash
# Проверка здоровья API
python test_api.py

# Тест с изображением
python test_api.py path/to/image.jpg
```

### Использование curl

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@image.jpg" \
  -F "confidence_threshold=0.5"
```
