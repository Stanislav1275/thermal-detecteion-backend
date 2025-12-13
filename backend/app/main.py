"""
FastAPI приложение для детекции людей на тепловизионных изображениях.
"""

import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import UploadResponse, JobStatus, JobResults, ImageResult
from .detector import ThermalDetector
from .processor import ImageProcessor
from .storage import JobStorage

app = FastAPI(
    title="Thermal Person Detection API",
    description="API для детекции людей на тепловизионных изображениях",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "training/models/best.pt")
STORAGE_BASE_DIR = os.getenv("STORAGE_DIR", "jobs")

detector = None
processor = None
storage = JobStorage(base_dir=STORAGE_BASE_DIR)


@app.on_event("startup")
async def startup_event():
    """Инициализация детектора при запуске."""
    global detector, processor
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Предупреждение: Модель не найдена по пути {MODEL_PATH}")
            print("Убедитесь, что модель обучена и сохранена в training/models/best.pt")
        else:
            detector = ThermalDetector(model_path=MODEL_PATH)
            processor = ImageProcessor(detector=detector)
            print("Детектор инициализирован")
    except Exception as e:
        print(f"Ошибка инициализации детектора: {e}")


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Thermal Person Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Проверка состояния API."""
    model_loaded = detector is not None and processor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5
):
    """Загружает изображения для обработки."""
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Детектор не инициализирован. Убедитесь, что модель обучена."
        )
    
    valid_files = []
    for file in files:
        if processor.validate_image_format(file.filename):
            valid_files.append(file)
        else:
            print(f"Неподдерживаемый формат: {file.filename}")
    
    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="Нет валидных изображений. Поддерживаемые форматы: TIFF, PNG, JPEG, WEBP"
        )
    
    job_id = storage.create_job(
        confidence_threshold=confidence_threshold,
        total_images=len(valid_files)
    )
    
    background_tasks.add_task(
        process_images_task,
        job_id=job_id,
        files=valid_files,
        confidence_threshold=confidence_threshold
    )
    
    return UploadResponse(
        job_id=job_id,
        message=f"Задача создана. Обработка {len(valid_files)} изображений начата."
    )


async def process_images_task(
    job_id: str,
    files: List[UploadFile],
    confidence_threshold: float
):
    """Фоновая обработка изображений."""
    global detector, processor
    
    try:
        storage.update_status(job_id, "processing")
        
        job_dir = Path(STORAGE_BASE_DIR) / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        
        results = []
        processed_count = 0
        detections_count = 0
        
        for file in files:
            try:
                file_data = await file.read()
                input_path = input_dir / file.filename
                processor.load_image(file_data, file.filename, str(input_path))
                
                output_path = output_dir / file.filename
                result = processor.process_image(
                    str(input_path),
                    str(output_path),
                    confidence=confidence_threshold
                )
                
                results.append(result)
                processed_count += 1
                
                if result.total_detections > 0:
                    detections_count += 1
                else:
                    if output_path.exists():
                        output_path.unlink()
                
                storage.update_status(
                    job_id,
                    "processing",
                    processed_images=processed_count,
                    images_with_detections=detections_count
                )
            
            except Exception as e:
                error_result = ImageResult(
                    filename=file.filename,
                    detections=[],
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
                processed_count += 1
        
        storage.save_detections(job_id, results)
        storage.update_status(
            job_id,
            "completed",
            processed_images=processed_count,
            images_with_detections=detections_count
        )
    
    except Exception as e:
        storage.update_status(job_id, "failed")
        print(f"Ошибка обработки задачи {job_id}: {e}")


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Возвращает статус задачи."""
    status = storage.get_status(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return JobStatus(**status)


@app.get("/api/jobs/{job_id}/results", response_model=JobResults)
async def get_job_results(job_id: str):
    """Возвращает результаты обработки (только изображения с детекциями людей)."""
    status = storage.get_status(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    detections = storage.get_detections(job_id)
    
    if detections is None:
        detections = []
    
    image_results = []
    for det in detections:
        from .models import Detection
        detections_list = [
            Detection(**d) for d in det.get('detections', [])
        ]
        
        if len(detections_list) > 0:
            image_results.append(ImageResult(
                filename=det['filename'],
                detections=detections_list,
                success=det.get('success', True),
                error=det.get('error')
            ))
    
    metadata = {
        "total_detections": sum(len(img.detections) for img in image_results),
        "total_images_with_people": len(image_results),
        "status": status['status']
    }
    
    return JobResults(
        job_id=job_id,
        images=image_results,
        metadata=metadata
    )


@app.get("/api/jobs/{job_id}/output/{filename}")
async def get_output_image(job_id: str, filename: str):
    """Возвращает обработанное изображение."""
    image_path = storage.get_output_image_path(job_id, filename)
    
    if image_path is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    
    return FileResponse(
        str(image_path),
        media_type="image/jpeg",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

