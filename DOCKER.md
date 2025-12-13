# Docker инструкции

## Требования

- Docker Engine 20.10+
- Docker Compose 2.0+

## Быстрый старт

### 1. Обучение модели (обязательно перед запуском)

Модель должна быть обучена и сохранена в `training/models/best.pt` перед запуском Docker контейнера.

```bash
cd training
./download_dataset.sh
./prepare.sh
./train_simple.sh
```

### 2. Запуск через Docker Compose

```bash
docker-compose up -d
```

API будет доступен на http://localhost:8000

### 3. Просмотр логов

```bash
docker-compose logs -f backend
```

### 4. Остановка

```bash
docker-compose down
```

## Сборка образа вручную

### Сборка образа

```bash
cd backend
docker build -t thermal-detection-api .
```

### Запуск контейнера

```bash
docker run -d \
  --name thermal-detection-api \
  -p 8000:8000 \
  -v $(pwd)/../training/models:/app/models:ro \
  -v $(pwd)/../jobs:/app/jobs \
  thermal-detection-api
```

## Структура volumes

- `./training/models:/app/models:ro` - монтирование обученной модели (read-only)
- `./jobs:/app/jobs` - монтирование директории для временных файлов задач

## Переменные окружения

- `MODEL_PATH` - путь к модели (по умолчанию: `/app/models/best.pt`)
- `PYTHONUNBUFFERED=1` - отключение буферизации вывода Python

## Проверка работоспособности

```bash
# Проверка здоровья API
curl http://localhost:8000/health

# Тест загрузки изображения
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@path/to/image.jpg" \
  -F "confidence_threshold=0.5"
```

## Устранение неполадок

### Ошибка: "Модель не найдена"

Убедитесь, что модель обучена и находится в `training/models/best.pt`:

```bash
ls -lh training/models/best.pt
```

### Ошибка: "Port already in use"

Измените порт в `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Внешний порт:внутренний порт
```

### Просмотр логов контейнера

```bash
docker-compose logs backend
```

### Пересборка образа

```bash
docker-compose build --no-cache
docker-compose up -d
```

