# Получение оригинальных изображений из результатов задачи

## API Endpoint

`GET /api/jobs/{job_id}/results`

## Структура ответа

Ответ содержит массив `images`, где каждый элемент имеет следующую структуру:

```typescript
interface ImageResult {
  filename: string;
  detections: Detection[];
  success: boolean;
  error?: string;
  original_image_url: string;      // URL для получения оригинального изображения
  processed_image_url: string;     // URL для получения обработанного изображения
  total_detections: number;
}
```

## Получение оригинального изображения

### Вариант 1: Использовать URL из ответа (рекомендуется)

API автоматически добавляет URL в каждый `ImageResult`:

```typescript
// Получаем результаты задачи
const response = await fetch(`/api/jobs/${jobId}/results`);
const data = await response.json();

// Для каждого изображения
data.images.forEach((image: ImageResult) => {
  // Оригинальное изображение
  const originalUrl = image.original_image_url;
  // Обработанное изображение (без меток, так как мы убрали отрисовку)
  const processedUrl = image.processed_image_url;
  
  // Использовать в <img> теге
  <img src={originalUrl} alt={image.filename} />
});
```

### Вариант 2: Построить URL вручную

Если URL не предоставлен (старая версия API), можно построить его вручную:

```typescript
const baseUrl = 'http://localhost:8000'; // или ваш API URL
const originalUrl = `${baseUrl}/api/jobs/${jobId}/input/${filename}`;
const processedUrl = `${baseUrl}/api/jobs/${jobId}/output/${filename}`;
```

## Прямые эндпоинты для изображений

### Оригинальное изображение
```
GET /api/jobs/{job_id}/input/{filename}
```

### Обработанное изображение
```
GET /api/jobs/{job_id}/output/{filename}
```

### Оригинальное через output эндпоинт
```
GET /api/jobs/{job_id}/output/{filename}?original=true
```

## Пример использования в React

```typescript
import { useState, useEffect } from 'react';

interface ImageResult {
  filename: string;
  detections: Array<{
    bbox: [number, number, number, number];
    confidence: number;
    class_name: string;
  }>;
  success: boolean;
  error?: string;
  original_image_url: string;
  processed_image_url: string;
  total_detections: number;
}

function JobResults({ jobId }: { jobId: string }) {
  const [images, setImages] = useState<ImageResult[]>([]);
  
  useEffect(() => {
    fetch(`/api/jobs/${jobId}/results`)
      .then(res => res.json())
      .then(data => setImages(data.images));
  }, [jobId]);
  
  return (
    <div>
      {images.map((image) => (
        <div key={image.filename}>
          <h3>{image.filename}</h3>
          {/* Оригинальное изображение */}
          <img 
            src={image.original_image_url} 
            alt={image.filename}
            style={{ maxWidth: '100%' }}
          />
          {/* Детекции для отрисовки bounding boxes */}
          {image.detections.map((det, idx) => (
            <div key={idx}>
              Confidence: {det.confidence}, 
              BBox: {det.bbox.join(', ')}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
```

## Пример с отрисовкой bounding boxes на фронтенде

```typescript
function ImageWithBoundingBoxes({ image }: { image: ImageResult }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = image.original_image_url;
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Отрисовка bounding boxes
      image.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = '#FF8C00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Подпись с confidence
        ctx.fillStyle = '#FF8C00';
        ctx.font = '14px Arial';
        ctx.fillText(
          `person ${det.confidence.toFixed(2)}`,
          x1,
          y1 - 5
        );
      });
    };
  }, [image]);
  
  return <canvas ref={canvasRef} />;
}
```

## Важные замечания

1. **Оригинальные изображения** находятся в `/api/jobs/{job_id}/input/{filename}`
2. **Обработанные изображения** (сейчас это копии оригиналов без меток) в `/api/jobs/{job_id}/output/{filename}`
3. **URL автоматически добавляются** в ответ `GET /api/jobs/{job_id}/results` в полях `original_image_url` и `processed_image_url`
4. **Bounding boxes не рисуются на изображениях** - они приходят только в JSON, фронтенд должен отрисовать их сам
5. **CORS** должен быть настроен на бэкенде для доступа к изображениям

## Обработка ошибок

```typescript
try {
  const response = await fetch(`/api/jobs/${jobId}/results`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  
  data.images.forEach((image: ImageResult) => {
    if (!image.success) {
      console.error(`Ошибка обработки ${image.filename}:`, image.error);
      return;
    }
    
    // Использовать image.original_image_url
  });
} catch (error) {
  console.error('Ошибка получения результатов:', error);
}
```

