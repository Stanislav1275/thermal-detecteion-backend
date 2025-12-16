import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import UploadResponse, JobStatus, JobResults, ImageResult, UpdateJobNameRequest, DeleteJobResponse
from .detector import ThermalDetector
from .processor import ImageProcessor
from .storage import JobStorage

API_VERSION = os.getenv("API_VERSION", "1.0.0")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="Thermal Person Detection API",
    description="API для детекции людей на тепловизионных изображениях",
    version=API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_default_model_path = "training/models/best.pt"
if not os.path.exists(_default_model_path):
    _alt_path = "../training/models/best.pt"
    if os.path.exists(_alt_path):
        _default_model_path = _alt_path

MODEL_PATH = os.getenv("MODEL_PATH", _default_model_path)
STORAGE_BASE_DIR = os.getenv("STORAGE_DIR", "jobs")

detector = None
processor = None
storage = JobStorage(base_dir=STORAGE_BASE_DIR)


@app.on_event("startup")
async def startup_event():
    global detector, processor
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Модель не найдена: {MODEL_PATH}")
            return
        
        print(f"Загрузка модели: {os.path.abspath(MODEL_PATH)}")
        detector = ThermalDetector(model_path=MODEL_PATH)
        processor = ImageProcessor(detector=detector)
        print("✅ Детектор инициализирован")
    except Exception as e:
        import traceback
        print(f"❌ Ошибка инициализации: {e}")
        traceback.print_exc()


@app.get("/")
async def root():
    return {
        "message": "Thermal Person Detection API",
        "version": API_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    model_loaded = detector is not None and processor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.66"))


def validate_confidence(confidence: float) -> None:
    if not isinstance(confidence, (int, float)):
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold должен быть числом"
        )
    
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold должен быть в диапазоне от 0.0 до 1.0"
        )

@app.post("/api/upload", response_model=UploadResponse)
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    confidence_threshold: Optional[str] = Form(None),
    name: Optional[str] = Form(None)
):
    if detector is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Детектор не инициализирован"
        )
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Нет загруженных файлов"
        )
    
    if confidence_threshold is None:
        confidence_value = DEFAULT_CONFIDENCE
    else:
        try:
            confidence_value = float(confidence_threshold)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"confidence_threshold должен быть числом, получено: {confidence_threshold}"
            )
    
    validate_confidence(confidence_value)
    
    import uuid
    temp_dir_name = f"temp_upload_{uuid.uuid4().hex[:8]}"
    job_dir = Path(STORAGE_BASE_DIR) / temp_dir_name
    job_dir.mkdir(parents=True, exist_ok=True)
    
    all_image_files = []
    extraction_errors = []
    
    try:
        for file in files:
            if file.filename is None:
                continue
            
            file_data = await file.read()
            sanitized_filename = processor.sanitize_filename(file.filename)
            
            if processor.is_zip_file(sanitized_filename):
                try:
                    extracted = processor.extract_zip_archive(file_data, str(job_dir))
                    if not extracted:
                        extraction_errors.append(f"Архив {sanitized_filename} не содержит валидных изображений")
                        continue
                    for archive_path, extracted_path in extracted:
                        all_image_files.append(extracted_path)
                except Exception as e:
                    extraction_errors.append(f"Ошибка при распаковке архива {sanitized_filename}: {str(e)}")
                    continue
            elif processor.validate_image_format(sanitized_filename):
                temp_path = job_dir / sanitized_filename
                processor.load_image(file_data, sanitized_filename, str(temp_path))
                all_image_files.append(str(temp_path))
        
        if not all_image_files:
            error_msg = "Нет валидных изображений. Поддерживаемые форматы: TIFF, PNG, JPEG, WEBP, ZIP"
            if extraction_errors:
                error_msg += f". Ошибки: {'; '.join(extraction_errors)}"
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        job_id = storage.create_job(
            name=name,
            confidence_threshold=confidence_value,
            total_images=len(all_image_files)
        )
        
        background_tasks.add_task(
            process_images_from_paths_task,
            job_id=job_id,
            image_paths=all_image_files,
            confidence_threshold=confidence_value,
            temp_dir=temp_dir_name
        )
        
        return UploadResponse(
            job_id=job_id,
            message=f"Задача создана. Обработка {len(all_image_files)} изображений начата."
        )
    except HTTPException:
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir, ignore_errors=True)
        raise


async def process_images_from_paths_task(
    job_id: str,
    image_paths: List[str],
    confidence_threshold: float,
    temp_dir: str
):
    global detector, processor
    
    temp_upload_dir = Path(STORAGE_BASE_DIR) / temp_dir
    
    try:
        storage.update_status(job_id, "processing")
        
        job_dir = Path(STORAGE_BASE_DIR) / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        
        results = []
        processed_count = 0
        detections_count = 0
        used_filenames = {}  # Словарь для отслеживания использованных имен файлов
        
        for image_path in image_paths:
            try:
                base_filename = processor.sanitize_filename(os.path.basename(image_path))
                
                # Обработка дубликатов имен файлов
                if base_filename in used_filenames:
                    used_filenames[base_filename] += 1
                    base, ext = os.path.splitext(base_filename)
                    filename = f"{base}_{used_filenames[base_filename]}{ext}"
                else:
                    used_filenames[base_filename] = 0
                    filename = base_filename
                
                input_path = input_dir / filename
                
                import shutil
                shutil.copy(image_path, input_path)
                
                output_path = output_dir / filename
                result = processor.process_image(
                    str(input_path),
                    str(output_path),
                    confidence=confidence_threshold
                )
                
                # Обновляем filename в результате на уникальное имя
                result.filename = filename
                
                results.append(result)
                processed_count += 1
                
                if len(result.detections) > 0:
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
                base_filename = processor.sanitize_filename(os.path.basename(image_path))
                if base_filename in used_filenames:
                    used_filenames[base_filename] += 1
                    base, ext = os.path.splitext(base_filename)
                    filename = f"{base}_{used_filenames[base_filename]}{ext}"
                else:
                    used_filenames[base_filename] = 0
                    filename = base_filename
                
                error_result = ImageResult(
                    filename=filename,
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
    finally:
        if temp_upload_dir.exists():
            import shutil
            shutil.rmtree(temp_upload_dir, ignore_errors=True)


@app.get("/api/jobs", response_model=List[JobStatus])
async def list_jobs():
    jobs = storage.list_jobs()
    return [JobStatus(**job) for job in jobs]


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    status = storage.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JobStatus(**status)


@app.patch("/api/jobs/{job_id}/name", response_model=JobStatus)
async def update_job_name(job_id: str, request: UpdateJobNameRequest):
    if not storage.update_job_name(job_id, request.name):
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    status = storage.get_status(job_id)
    return JobStatus(**status)


@app.get("/api/jobs/{job_id}/results", response_model=JobResults)
async def get_job_results(request: Request, job_id: str, only_with_detections: bool = False):
    status = storage.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    detections = storage.get_detections(job_id) or []
    
    from .models import Detection
    
    base_url = str(request.base_url).rstrip('/')
    
    image_results = []
    for det in detections:
        detections_list = [Detection(**d) for d in det.get('detections', [])]
        
        if only_with_detections and not detections_list:
            continue
        
        filename = det['filename']
        original_url = f"{base_url}/api/jobs/{job_id}/input/{filename}"
        processed_url = f"{base_url}/api/jobs/{job_id}/output/{filename}"
        
        image_results.append(ImageResult(
            filename=filename,
            detections=detections_list,
            success=det.get('success', True),
            error=det.get('error'),
            original_image_url=original_url,
            processed_image_url=processed_url
        ))
    
    images_with_detections = [img for img in image_results if img.detections]
    images_with_errors = [img for img in image_results if not img.success]
    
    metadata = {
        "total_detections": sum(len(img.detections) for img in image_results),
        "total_images_with_people": len(images_with_detections),
        "total_images": len(image_results),
        "total_errors": len(images_with_errors),
        "status": status['status']
    }
    
    return JobResults(job_id=job_id, images=image_results, metadata=metadata)


@app.get("/api/jobs/{job_id}/input/{filename}")
async def get_input_image(job_id: str, filename: str):
    image_path = storage.get_input_image_path(job_id, filename)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Оригинальное изображение не найдено")
    mime_type = processor.get_mime_type(filename)
    return FileResponse(str(image_path), media_type=mime_type, filename=filename)


@app.get("/api/jobs/{job_id}/output/{filename}")
async def get_output_image(job_id: str, filename: str, original: bool = False):
    if original:
        image_path = storage.get_input_image_path(job_id, filename)
        if image_path is None:
            raise HTTPException(status_code=404, detail="Оригинальное изображение не найдено")
    else:
        image_path = storage.get_output_image_path(job_id, filename)
        if image_path is None:
            raise HTTPException(status_code=404, detail="Обработанное изображение не найдено")
    
    mime_type = processor.get_mime_type(filename)
    return FileResponse(str(image_path), media_type=mime_type, filename=filename)


@app.delete("/api/jobs/{job_id}", response_model=DeleteJobResponse)
async def delete_job(job_id: str):
    """
    Удалить задачу и все связанные с ней данные.
    
    Внимание: Операция необратима. Все данные задачи будут удалены безвозвратно.
    """
    success = storage.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Задача с ID '{job_id}' не найдена")
    
    return DeleteJobResponse(
        message="Задача успешно удалена",
        job_id=job_id
    )


@app.get("/api/jobs/{job_id}/download")
async def download_results_zip(
    job_id: str,
    original: bool = False,
    min_confidence: float = 0.0,
    only_with_detections: bool = False
):
    status = storage.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if status['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {status['status']}"
        )
    
    if not 0.0 <= min_confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_confidence должен быть в диапазоне от 0.0 до 1.0"
        )
    
    zip_path = storage.create_output_zip(
        job_id,
        use_original=original,
        min_confidence=min_confidence,
        only_with_detections=only_with_detections
    )
    if zip_path is None:
        raise HTTPException(
            status_code=404,
            detail="Нет изображений для скачивания (возможно, не найдены изображения с указанными критериями фильтрации)"
        )
    
    job_name = status.get('name', job_id)
    type_label = "original" if original else "processed"
    conf_label = f"_conf{min_confidence:.2f}" if min_confidence > 0.0 else ""
    detections_label = "_with_detections" if only_with_detections else ""
    zip_filename = f"{job_name}_{type_label}{detections_label}{conf_label}.zip"
    
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=zip_filename
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

