
import os
import sys
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("⚠️  Ошибка: Не найдены необходимые модули (torch, ultralytics)")
    print("Активируйте окружение: conda activate thermal-detection")
    sys.exit(1)


def check_model_readiness(model_path: str = None):
    print("═══════════════════════════════════════════════════════════")
    print("ПРОВЕРКА ГОТОВНОСТИ МОДЕЛИ")
    print("═══════════════════════════════════════════════════════════")
    print("")
    
    # Автоматический поиск модели, если путь не указан
    if model_path is None:
        possible_paths = [
            "models/best.pt",
            "training/models/best.pt",
            "models/best_thermal_m4.pt",
        ]
        
        # Также ищем в последнем запуске
        runs_dirs = ["training/runs", "runs"]
        for runs_dir in runs_dirs:
            if os.path.exists(runs_dir):
                import glob
                pattern = os.path.join(runs_dir, "thermal_detection*", "weights", "best.pt")
                matches = glob.glob(pattern)
                if matches:
                    matches.sort(key=os.path.getmtime, reverse=True)
                    possible_paths.insert(0, matches[0])
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"📦 Автоматически найдена модель: {model_path}")
                break
        
        if model_path is None:
            print("❌ Модель не найдена автоматически")
            print("   Укажите путь вручную: python check_model.py <путь_к_модели>")
            return False
    
    checks_passed = 0
    total_checks = 5
    
    # Проверка 1: Существование файла
    print("1. Проверка существования файла модели...")
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   ✓ Файл найден: {model_path}")
        print(f"   ✓ Размер: {file_size:.2f} MB")
        checks_passed += 1
    else:
        print(f"   ❌ Файл не найден: {model_path}")
        print("")
        print("   Возможные решения:")
        print("   • Дождитесь завершения обучения")
        print("   • Запустите: bash post_training.sh")
        return False
    print("")
    
    # Проверка 2: Загрузка модели
    print("2. Проверка загрузки модели...")
    try:
        model = YOLO(model_path)
        print("   ✓ Модель успешно загружена")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        return False
    print("")
    
    # Проверка 3: Тестовая инференция
    print("3. Проверка тестовой инференции...")
    try:
        import numpy as np
        from PIL import Image
        
        # Создаем тестовое изображение правильного формата (HWC, uint8)
        # YOLOv8 ожидает изображение в формате (height, width, channels) или PIL Image
        test_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Определяем устройство
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # Тестовая инференция
        results = model.predict(
            test_image,
            conf=0.25,
            device=device,
            verbose=False
        )
        print(f"   ✓ Инференция успешна (устройство: {device.upper()})")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Ошибка инференции: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
        import traceback
        print(f"   Детали: {traceback.format_exc()}")
        return False
    print("")
    
    # Проверка 4: Наличие необходимых классов
    print("4. Проверка классов модели...")
    class_names = model.names
    required_classes = ['person', 'car']
    found_classes = []
    
    print(f"   Доступные классы: {list(class_names.values())}")
    
    for class_id, class_name in class_names.items():
        if class_name.lower() in [c.lower() for c in required_classes]:
            found_classes.append(class_name)
    
    if len(found_classes) > 0:
        print(f"   ✓ Найдены необходимые классы: {found_classes}")
        checks_passed += 1
    else:
        print(f"   ⚠️  Предупреждение: Не найдены классы {required_classes}")
        print("   Модель может работать некорректно с backend")
    print("")
    
    # Проверка 5: Совместимость с backend
    print("5. Проверка совместимости с backend...")
    backend_model_paths = ["models/best.pt", "training/models/best.pt"]
    backend_found = False
    
    for backend_path in backend_model_paths:
        if os.path.exists(backend_path):
            try:
                if os.path.samefile(model_path, backend_path) or model_path == backend_path:
                    print(f"   ✓ Модель находится в стандартном месте для backend: {backend_path}")
                    backend_found = True
                    checks_passed += 1
                    break
            except OSError:
                # samefile может не работать на некоторых системах
                if os.path.abspath(model_path) == os.path.abspath(backend_path):
                    print(f"   ✓ Модель находится в стандартном месте для backend: {backend_path}")
                    backend_found = True
                    checks_passed += 1
                    break
    
    if not backend_found:
        print(f"   ⚠️  Модель не в стандартном месте для backend")
        print(f"   Backend ожидает: models/best.pt")
        print(f"   Текущий путь: {model_path}")
        print("   Запустите: bash post_training.sh для копирования модели")
    print("")
    
    # Итоговый результат
    print("═══════════════════════════════════════════════════════════")
    if checks_passed == total_checks:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("═══════════════════════════════════════════════════════════")
        print("")
        print("Модель готова к использованию в backend!")
        print("")
        print("Следующие шаги:")
        print("  1. Запустить backend:")
        print("     cd ../backend && python -m uvicorn app.main:app --reload")
        print("")
        print("  2. Проверить здоровье API:")
        print("     python ../test_api.py")
        return True
    else:
        print(f"⚠️  ПРОЙДЕНО ПРОВЕРОК: {checks_passed}/{total_checks}")
        print("═══════════════════════════════════════════════════════════")
        print("")
        print("Некоторые проверки не пройдены. Исправьте проблемы выше.")
        return False


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = check_model_readiness(model_path)
    sys.exit(0 if success else 1)

