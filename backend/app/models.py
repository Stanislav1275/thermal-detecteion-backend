"""
Pydantic модели для API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Detection(BaseModel):
    """Модель детекции."""
    bbox: List[int] = Field(..., description="Координаты рамки [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детекции")
    class: str = Field(default="person", description="Класс объекта")


class ImageResult(BaseModel):
    """Результат обработки одного изображения."""
    filename: str = Field(..., description="Имя файла")
    detections: List[Detection] = Field(default_factory=list, description="Список детекций")
    success: bool = Field(default=True, description="Успешность обработки")
    error: Optional[str] = Field(default=None, description="Сообщение об ошибке (если есть)")


class JobStatus(BaseModel):
    """Статус задачи обработки."""
    job_id: str = Field(..., description="ID задачи")
    status: str = Field(..., description="Статус: queued, processing, completed, failed")
    total_images: int = Field(default=0, description="Всего изображений")
    processed_images: int = Field(default=0, description="Обработано изображений")
    images_with_detections: int = Field(default=0, description="Изображений с детекциями")
    created_at: datetime = Field(default_factory=datetime.now, description="Время создания")
    completed_at: Optional[datetime] = Field(default=None, description="Время завершения")
    parameters: dict = Field(default_factory=dict, description="Параметры обработки")


class JobResults(BaseModel):
    """Результаты обработки задачи."""
    job_id: str = Field(..., description="ID задачи")
    images: List[ImageResult] = Field(default_factory=list, description="Результаты по изображениям")
    metadata: dict = Field(default_factory=dict, description="Метаданные обработки")


class UploadResponse(BaseModel):
    """Ответ на загрузку изображений."""
    job_id: str = Field(..., description="ID созданной задачи")
    message: str = Field(default="Задача создана", description="Сообщение")

