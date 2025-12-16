# Docker Development Setup

## Быстрый старт с hot-reload

Для разработки с автоматической перезагрузкой при изменении кода:

### Вариант 1: Использование docker-compose.dev.yml

```bash
# Запуск в режиме разработки
docker-compose -f docker-compose.dev.yml up --build

# Или в фоновом режиме
docker-compose -f docker-compose.dev.yml up -d --build
```

API будет доступен на: http://localhost:8000

### Вариант 2: Использование профиля dev в docker-compose.yml

```bash
# Запуск только dev сервиса
docker-compose --profile dev up backend-dev --build

# Или в фоновом режиме
docker-compose --profile dev up -d backend-dev --build
```

API будет доступен на: http://localhost:8001

### Вариант 3: Прямая сборка и запуск

```bash
# Сборка образа
docker build -f backend/Dockerfile.dev -t thermal-detection-api-dev ./backend

# Запуск контейнера
docker run -d \
  --name thermal-detection-api-dev \
  -p 8000:8000 \
  -v $(pwd)/backend/app:/app/app:ro \
  -v $(pwd)/training/models:/app/models:ro \
  -v $(pwd)/jobs:/app/jobs \
  -e MODEL_PATH=/app/models/best.pt \
  -e STORAGE_DIR=/app/jobs \
  thermal-detection-api-dev
```

## Особенности dev режима

- ✅ Автоматическая перезагрузка при изменении кода (`--reload`)
- ✅ Код монтируется как volume (изменения видны сразу)
- ✅ Все остальные настройки как в production версии

## Остановка

```bash
# Для docker-compose.dev.yml
docker-compose -f docker-compose.dev.yml down

# Для профиля dev
docker-compose --profile dev down

# Для прямого запуска
docker stop thermal-detection-api-dev
docker rm thermal-detection-api-dev
```

## Проверка работы

После запуска проверьте:

```bash
# Health check
curl http://localhost:8000/health

# Документация API
open http://localhost:8000/docs
```

## Отличия от production

| Параметр | Production | Development |
|----------|-----------|-------------|
| Dockerfile | `Dockerfile` | `Dockerfile.dev` |
| Reload | ❌ | ✅ |
| Код в образе | ✅ (копируется) | ❌ (монтируется) |
| Порт | 8000 | 8000 или 8001 |

