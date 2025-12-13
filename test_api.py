"""
Простой скрипт для тестирования API.
"""

import requests
import json
import time
import sys
from pathlib import Path


API_BASE_URL = "http://localhost:8000"


def test_health():
    """Проверка здоровья API."""
    print("🔍 Проверка здоровья API...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API работает: {data}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ API вернул код {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API")
        print("   Убедитесь, что сервер запущен: uvicorn app.main:app --reload")
        return False


def test_upload(image_path):
    """Тестирование загрузки изображения."""
    print(f"\n📤 Загрузка изображения: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Файл не найден: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'files': (Path(image_path).name, f, 'image/jpeg')}
            data = {'confidence_threshold': 0.5}
            
            response = requests.post(
                f"{API_BASE_URL}/api/upload",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✅ Задача создана: {job_id}")
            return job_id
        else:
            print(f"❌ Ошибка загрузки: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def check_job_status(job_id):
    """Проверка статуса задачи."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Ошибка проверки статуса: {e}")
        return None


def wait_for_completion(job_id, max_wait=300):
    """Ожидание завершения обработки."""
    print(f"\n⏳ Ожидание завершения обработки (макс. {max_wait} сек)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = check_job_status(job_id)
        if status:
            current_status = status['status']
            processed = status['processed_images']
            total = status['total_images']
            
            print(f"   Статус: {current_status} | Обработано: {processed}/{total}")
            
            if current_status == 'completed':
                print("✅ Обработка завершена!")
                return True
            elif current_status == 'failed':
                print("❌ Обработка завершилась с ошибкой")
                return False
        
        time.sleep(2)
    
    print("⏱️  Превышено время ожидания")
    return False


def get_results(job_id):
    """Получение результатов."""
    print(f"\n📊 Получение результатов для задачи {job_id}...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/jobs/{job_id}/results")
        if response.status_code == 200:
            results = response.json()
            images = results['images']
            metadata = results['metadata']
            
            print(f"✅ Найдено изображений с людьми: {len(images)}")
            print(f"   Всего детекций: {metadata.get('total_detections', 0)}")
            
            for img in images[:5]:  # Показываем первые 5
                print(f"   - {img['filename']}: {len(img['detections'])} детекций")
            
            if len(images) > 5:
                print(f"   ... и еще {len(images) - 5} изображений")
            
            return results
        else:
            print(f"❌ Ошибка получения результатов: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("Тестирование Thermal Person Detection API")
    print("=" * 60)
    
    # Проверка здоровья
    model_loaded = test_health()
    if not model_loaded:
        print("\n⚠️  Модель не загружена. API может работать некорректно.")
        response = input("Продолжить тестирование? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Тестирование загрузки (если указан файл)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        job_id = test_upload(image_path)
        
        if job_id:
            # Ожидание завершения
            if wait_for_completion(job_id):
                # Получение результатов
                get_results(job_id)
    else:
        print("\n💡 Для тестирования загрузки укажите путь к изображению:")
        print("   python test_api.py path/to/image.jpg")


if __name__ == "__main__":
    main()

